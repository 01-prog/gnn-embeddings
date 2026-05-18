"""Amazon McAuley/UCSD product co-purchase graph dataset.

Loads product metadata (title, description, image URL) and ``also_buy`` /
``also_bought`` edges for one or two top-level Amazon categories, and exposes
them as a single PyTorch Geometric ``Data`` object usable by the GNNs and
flat-classifier baselines defined under ``src/models/``.

Pipeline
--------
1. Read one or more JSONL metadata files from ``data/raw/`` (the McAuley
   group publishes one ``meta_<Category>.json[.gz]`` per top-level category).
2. Normalize records across the 2018 and 2023 schema variants.
3. Keep only products with a usable label and a non-empty title.
4. Build an undirected co-purchase edge_index, keeping only edges where
   both endpoints survive filtering.
5. Stratified train / val / test split, cached to
   ``data/splits/split_{seed}.npz`` so every encoder/model pair can use the
   exact same node assignment.
6. Save the processed ``Data`` object via ``InMemoryDataset``'s standard
   ``processed_paths[0]`` mechanism.

Node features ``x`` are deliberately a placeholder ``torch.empty(N, 0)`` —
the encoders in ``src/encoders/`` fill them in later from cached embeddings.

The reference dataset homepages:

- 2018: https://nijianmo.github.io/amazon/index.html
- 2023: https://amazon-reviews-2023.github.io/

This module never downloads the dataset itself — drop the
``meta_<Category>.json[.gz]`` files into ``data/raw/`` before constructing
the dataset. The smoke test at the bottom of the file uses a fully
synthetic in-memory fixture.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata parsing helpers
# ---------------------------------------------------------------------------


def _open_text(path: Path) -> Iterator[str]:
    """Yield lines from a JSON / JSONL file, transparently handling gzip.

    Args:
        path: File path. May end in ``.gz``.

    Yields:
        Each non-empty line of the file as a ``str``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _parse_record(line: str) -> dict[str, Any] | None:
    """Parse a single metadata line.

    The 2018 dump uses Python-literal dicts (``eval``-style) and the 2023
    dump uses strict JSON. We try strict JSON first, then fall back to
    ``ast.literal_eval`` for older files.

    Args:
        line: One line of a metadata file.

    Returns:
        A ``dict`` with the parsed record, or ``None`` if the line could
        not be parsed.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            import ast

            obj = ast.literal_eval(line)
            return obj if isinstance(obj, dict) else None
        except (ValueError, SyntaxError):
            return None


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Project a raw metadata record onto a stable internal schema.

    Output keys:

    - ``asin``: product identifier (str).
    - ``title``: product title (str, may be empty).
    - ``description``: free-text description (str, may be empty).
    - ``image_url``: first associated image URL or empty string.
    - ``also_buy``: list of co-purchased ASINs.
    - ``categories``: list of category-path lists. Each path goes from
      coarse to fine: ``["Books", "Children's Books", "Animals"]``.

    Args:
        rec: Raw record as parsed from a metadata line.

    Returns:
        Normalized record dict, or ``None`` if the record has no ASIN.
    """
    asin = rec.get("asin") or rec.get("parent_asin")
    if not asin:
        return None

    # Title: 2018 uses "title", 2023 uses "title".
    title = (rec.get("title") or "").strip()

    # Description: 2018 = str; 2023 = list[str].
    desc_raw = rec.get("description") or rec.get("descriptions") or ""
    if isinstance(desc_raw, list):
        description = " ".join(str(x) for x in desc_raw).strip()
    else:
        description = str(desc_raw).strip()

    # Image URL: 2018 = "imUrl" (str). 2023 = "images" (list of dicts).
    image_url = ""
    if isinstance(rec.get("imUrl"), str):
        image_url = rec["imUrl"]
    elif isinstance(rec.get("imageURL"), list) and rec["imageURL"]:
        image_url = str(rec["imageURL"][0])
    elif isinstance(rec.get("images"), list) and rec["images"]:
        first = rec["images"][0]
        if isinstance(first, dict):
            # Prefer "large", fall back to whatever is present.
            image_url = first.get("large") or first.get("hi_res") or first.get("thumb") or ""
        else:
            image_url = str(first)

    # Co-purchase edges. 2018 nests under "related"; 2023 puts them at top.
    also_buy: list[str] = []
    related = rec.get("related") or {}
    if isinstance(related, dict):
        also_buy.extend(related.get("also_bought", []) or [])
        also_buy.extend(related.get("bought_together", []) or [])
    also_buy.extend(rec.get("also_buy", []) or [])
    # De-duplicate, drop self-loops.
    also_buy = list({a for a in also_buy if a and a != asin})

    # Category hierarchy: 2018 = "category" or "categories" -> list of paths.
    # 2023 = "categories" -> single path (list[str]).
    cats_raw = rec.get("categories") if "categories" in rec else rec.get("category")
    categories: list[list[str]] = []
    if isinstance(cats_raw, list):
        if cats_raw and isinstance(cats_raw[0], list):
            categories = [[str(c) for c in path] for path in cats_raw if path]
        elif cats_raw:
            categories = [[str(c) for c in cats_raw]]

    return {
        "asin": str(asin),
        "title": title,
        "description": description,
        "image_url": image_url,
        "also_buy": also_buy,
        "categories": categories,
    }


def _pick_label(
    record: dict[str, Any],
    top_level: set[str],
    label_level: int,
) -> str | None:
    """Pick a label string for a record at the requested category depth.

    Prefers a category path whose first element matches one of the requested
    top-level categories; falls back to the first non-empty path.

    Args:
        record: Normalized record (must have a ``categories`` key).
        top_level: Set of allowed top-level category names. Empty disables
            filtering.
        label_level: 0 for top-level category, 1 for first sub-category,
            etc.

    Returns:
        The label string, or ``None`` if no path is deep enough.
    """
    paths: list[list[str]] = record.get("categories", []) or []
    chosen: list[str] | None = None
    for path in paths:
        if not path:
            continue
        if not top_level or path[0] in top_level:
            chosen = path
            break
    if chosen is None and paths:
        # No path matched the filter — only useful when ``top_level`` is empty.
        chosen = paths[0]
    if chosen is None or len(chosen) <= label_level:
        return None
    return chosen[label_level]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _stratified_split(
    labels: np.ndarray,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of node indices into train / val / test arrays.

    For each class the indices are shuffled with ``seed`` and then sliced
    by ``ratios``. Classes with fewer than 3 samples are forced into train
    so we don't end up with empty val/test partitions for rare classes.

    Args:
        labels: 1-D array of integer labels of length ``N``.
        ratios: Tuple of (train, val, test) fractions summing to ~1.
        seed: Seed for the per-class shuffle.

    Returns:
        ``(train_idx, val_idx, test_idx)`` as 1-D ``int64`` arrays.

    Raises:
        ValueError: If ``ratios`` does not sum to ~1 or has wrong length.
    """
    if len(ratios) != 3:
        raise ValueError(f"ratios must have length 3, got {len(ratios)}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            train.extend(idx.tolist())
            continue
        n_train = max(1, int(round(ratios[0] * n)))
        n_val = max(1, int(round(ratios[1] * n)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train : n_train + n_val].tolist())
        test.extend(idx[n_train + n_val :].tolist())
    return (
        np.array(sorted(train), dtype=np.int64),
        np.array(sorted(val), dtype=np.int64),
        np.array(sorted(test), dtype=np.int64),
    )


def _masks_from_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build boolean masks of shape ``(n_nodes,)`` from index arrays."""
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AmazonCopurchase(InMemoryDataset):
    """Amazon co-purchase graph dataset.

    Loads product metadata and co-purchase edges for one or two top-level
    categories. Saves a processed PyG ``Data`` object to disk.

    Args:
        root: Path to the data directory. Raw files are expected at
            ``root/raw/meta_<Category>.json[.gz]`` and processed artifacts
            are written to ``root/processed/``.
        categories: List of top-level category names to include (1 or 2).
            Each category requires a corresponding raw metadata file.
        split_seed: Random seed for the train / val / test split.
        split_ratios: Tuple ``(train, val, test)`` summing to 1.0.
        label_level: Depth in the category hierarchy used as the label.
            ``None`` (default) means: 0 (top-level) when ``len(categories)
            >= 2`` else 1 (subcategory).
        min_class_count: Drop classes (and their products) with strictly
            fewer than this many samples; prevents tiny classes from
            corrupting stratified splits and statistical reports.
        transform: Optional PyG transform applied at access time.
        pre_transform: Optional PyG transform applied once at processing.

    Attributes:
        data.x: Float placeholder of shape ``(N, 0)``. Filled later by an
            encoder; never use it as features as-is.
        data.edge_index: ``LongTensor`` of shape ``(2, 2E)`` (undirected,
            both directions stored).
        data.y: ``LongTensor`` of shape ``(N,)`` with class indices.
        data.train_mask / val_mask / test_mask: ``BoolTensor (N,)``.
        data.meta: Python ``list`` of length ``N`` with dicts containing
            ``title``, ``description``, ``image_url``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        categories: Sequence[str],
        split_seed: int = 0,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        label_level: int | None = None,
        min_class_count: int = 2,
        transform=None,
        pre_transform=None,
    ) -> None:
        if not categories:
            raise ValueError("categories must contain at least one name")
        if len(categories) > 2:
            raise ValueError(
                f"categories supports at most 2 top-level names, got {len(categories)}"
            )
        self.categories: list[str] = list(categories)
        self.split_seed: int = int(split_seed)
        self.split_ratios: tuple[float, float, float] = tuple(split_ratios)  # type: ignore[assignment]
        self.label_level: int = (
            label_level
            if label_level is not None
            else (0 if len(categories) >= 2 else 1)
        )
        self.min_class_count: int = int(min_class_count)

        super().__init__(str(root), transform=transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # PyG hooks
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        """Filenames expected under ``root/raw/``.

        The loader accepts both ``.json`` and ``.json.gz``; we list the
        plain name and ``process()`` will probe both.
        """
        return [f"meta_{cat}.json" for cat in self.categories]

    @property
    def processed_file_names(self) -> list[str]:
        cats = "_".join(c.replace(" ", "") for c in self.categories)
        return [
            f"amazon_{cats}_seed{self.split_seed}_lvl{self.label_level}.pt"
        ]

    def download(self) -> None:  # noqa: D401  -- PyG hook
        """Not implemented: drop raw metadata files into ``root/raw/`` manually."""
        missing = []
        for name in self.raw_file_names:
            base = Path(self.raw_dir) / name
            if not base.exists() and not base.with_suffix(".json.gz").exists():
                missing.append(str(base))
        if missing:
            raise FileNotFoundError(
                "Missing Amazon metadata files. Download the per-category "
                "JSONL dumps from https://nijianmo.github.io/amazon/ or "
                "https://amazon-reviews-2023.github.io/ and place them at:\n  "
                + "\n  ".join(missing)
            )

    def process(self) -> None:  # noqa: D401  -- PyG hook
        """Build the processed ``Data`` object and persist it to disk."""
        records = self._load_records()
        data = self._build_data(records)
        self.save([data], self.processed_paths[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_raw(self, name: str) -> Path:
        """Return the on-disk path of a raw file, picking ``.gz`` if present."""
        plain = Path(self.raw_dir) / name
        if plain.exists():
            return plain
        gz = plain.with_suffix(plain.suffix + ".gz")
        if gz.exists():
            return gz
        raise FileNotFoundError(f"Could not find {plain} (or its .gz variant)")

    def _load_records(self) -> list[dict[str, Any]]:
        """Read and normalize every metadata file for the requested categories."""
        out: list[dict[str, Any]] = []
        for name in self.raw_file_names:
            path = self._resolve_raw(name)
            n_in = 0
            n_kept = 0
            for line in _open_text(path):
                n_in += 1
                rec = _parse_record(line)
                if rec is None:
                    continue
                norm = _normalize_record(rec)
                if norm is None:
                    continue
                out.append(norm)
                n_kept += 1
            logger.info("Loaded %d/%d records from %s", n_kept, n_in, path.name)
        return out

    def _build_data(self, records: Iterable[dict[str, Any]]) -> Data:
        """Convert normalized records into a PyG ``Data`` object."""
        top_set = set(self.categories)

        # First pass: pick a label for each record, drop those without one.
        kept: list[dict[str, Any]] = []
        labels_str: list[str] = []
        for r in records:
            if not r["title"]:
                continue
            label = _pick_label(r, top_set, self.label_level)
            if label is None:
                continue
            kept.append(r)
            labels_str.append(label)

        # Drop rare classes.
        counts = Counter(labels_str)
        keep_classes = {c for c, n in counts.items() if n >= self.min_class_count}
        kept2: list[dict[str, Any]] = []
        labels2: list[str] = []
        for r, lab in zip(kept, labels_str):
            if lab in keep_classes:
                kept2.append(r)
                labels2.append(lab)
        if not kept2:
            raise RuntimeError(
                "No records left after filtering. Check categories / "
                "label_level / min_class_count."
            )

        # ASIN -> node id (in final order).
        asin_to_idx: dict[str, int] = {r["asin"]: i for i, r in enumerate(kept2)}
        n_nodes = len(kept2)

        # Stable label encoding.
        class_names = sorted(set(labels2))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y = torch.tensor([class_to_idx[lab] for lab in labels2], dtype=torch.long)

        # Edges. Symmetric, deduplicated, self-loops dropped.
        edge_set: set[tuple[int, int]] = set()
        for r in kept2:
            src = asin_to_idx[r["asin"]]
            for tgt_asin in r["also_buy"]:
                tgt = asin_to_idx.get(tgt_asin)
                if tgt is None or tgt == src:
                    continue
                a, b = (src, tgt) if src < tgt else (tgt, src)
                edge_set.add((a, b))
        if edge_set:
            src_arr = np.fromiter((a for a, _ in edge_set), dtype=np.int64)
            tgt_arr = np.fromiter((b for _, b in edge_set), dtype=np.int64)
            ei = np.stack(
                [
                    np.concatenate([src_arr, tgt_arr]),
                    np.concatenate([tgt_arr, src_arr]),
                ]
            )
            edge_index = torch.from_numpy(ei).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Splits. Cached to data/splits/split_{seed}.npz.
        train_idx, val_idx, test_idx = _stratified_split(
            y.numpy(), self.split_ratios, self.split_seed
        )
        train_mask, val_mask, test_mask = _masks_from_indices(
            train_idx, val_idx, test_idx, n_nodes
        )

        splits_dir = Path(self.root) / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            splits_dir / f"split_{self.split_seed}.npz",
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            classes=np.array(class_names),
        )

        meta = [
            {
                "title": r["title"],
                "description": r["description"],
                "image_url": r["image_url"],
            }
            for r in kept2
        ]

        data = Data(
            x=torch.empty((n_nodes, 0), dtype=torch.float32),
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        data.meta = meta
        data.class_names = class_names

        # Stats — useful for the supervisor's "report dataset stats" rule.
        logger.info("n_nodes=%d", n_nodes)
        logger.info("n_edges=%d (undirected)", edge_index.shape[1] // 2)
        logger.info("n_classes=%d", len(class_names))
        for cls, cnt in sorted(Counter(labels2).items(), key=lambda kv: -kv[1]):
            logger.info("  class %-30s : %d", cls, cnt)
        return data


def print_dataset_stats(data: Data, class_names: Sequence[str]) -> None:
    """Print a short summary of a built ``Data`` object to stdout.

    Args:
        data: Processed ``Data`` object from ``AmazonCopurchase``.
        class_names: Class label strings in id order.
    """
    n_nodes = int(data.num_nodes)
    n_edges = int(data.edge_index.shape[1] // 2)
    n_classes = len(class_names)
    print(f"n_nodes:    {n_nodes}")
    print(f"n_edges:    {n_edges}  (undirected)")
    print(f"n_classes:  {n_classes}")
    print(f"train/val/test: "
          f"{int(data.train_mask.sum())} / "
          f"{int(data.val_mask.sum())} / "
          f"{int(data.test_mask.sum())}")
    print("class distribution:")
    counts = Counter(data.y.tolist())
    for cls_id in range(n_classes):
        print(f"  {class_names[cls_id]:<30} {counts.get(cls_id, 0):>6}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a tiny synthetic metadata file to exercise the pipeline end-to-end
    # without needing the real (multi-GB) Amazon dump.
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    fake_records = []
    # 6 "Electronics > Cameras", 6 "Electronics > Phones".
    for i in range(6):
        fake_records.append(
            {
                "asin": f"CAM{i:03d}",
                "title": f"Camera {i}",
                "description": f"A nice camera #{i}",
                "imUrl": f"http://img/cam{i}.jpg",
                "categories": [["Electronics", "Cameras"]],
                "related": {"also_bought": [f"CAM{(i + 1) % 6:03d}", f"PHN{i % 6:03d}"]},
            }
        )
    for i in range(6):
        fake_records.append(
            {
                "asin": f"PHN{i:03d}",
                "title": f"Phone {i}",
                "description": f"A nice phone #{i}",
                "imUrl": f"http://img/phn{i}.jpg",
                "categories": [["Electronics", "Phones"]],
                "related": {"also_bought": [f"PHN{(i + 1) % 6:03d}"]},
            }
        )

    tmp = Path(tempfile.mkdtemp(prefix="amazon_copurchase_smoke_"))
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    with (raw_dir / "meta_Electronics.json").open("w", encoding="utf-8") as fh:
        for r in fake_records:
            fh.write(json.dumps(r) + "\n")

    ds = AmazonCopurchase(
        root=tmp,
        split_seed=0,        
        categories=["Books", "Electronics", "Movies_and_TV", "CDs_and_Vinyl", "Clothing_Shoes_and_Jewelry", "Home_and_Kitchen", "Kindle_Store", "Sports_and_Outdoors", "Cell_Phones_and_Accessories", "Health_and_Personal_Care", "Toys_and_Games", "Video_Games", "Tools_and_Home_Improvement", "Beauty", "Apps_for_Android", "Office_Products", "Pet_Supplies", "Automotive", "Grocery_and_Gourmet_Food", "Patio,_Lawn_and_Garden", "Musical_Instruments", "Baby", "Digital_Music", "Amazon_Instant_Video"],
        split_ratios=(0.6, 0.2, 0.2),
        label_level=1,
        min_class_count=2,
    )
    data = ds[0]
    print_dataset_stats(data, data.class_names)

    # Sanity assertions.
    assert data.num_nodes == 12
    assert data.x.shape == (12, 0)
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] % 2 == 0  # undirected -> even count
    assert data.y.shape == (12,)
    assert int(data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()) == 12
    assert len(data.meta) == 12
    assert (Path(tmp) / "splits" / "split_0.npz").exists()
    print("amazon_dataset.py smoke test OK")
    shutil.rmtree(tmp, ignore_errors=True)
