"""H2 experiment runner: multimodal (text + image) vs single modality.

Hypothesis H2: Multimodal (text + image) embeddings beat single-modality
on a product co-purchase classification task.

Pipeline
--------
1. Determine the best GNN from H1 results (or fallback to config).
2. Determine the best text encoder from H3 results (or fallback to config).
3. Download and cache product images via the concurrent image pipeline.
4. For each modality in ``{text_only, image_only, text+image}`` (configurable):

   text_only   — Compute text embeddings with the H3-best encoder; train with
                 the H1-best GNN using a fresh hyperparameter search.

   image_only  — For each vision encoder in the YAML:
                 Encode image_paths (zero vectors for missing images); train
                 with the H1-best GNN.

   text+image  — For each (vision_encoder × fusion_strategy) pair:
                 Set ``data.x = cat([text_emb, image_emb], dim=-1)``; use a
                 :class:`~src.fusion.fusion.FusedModelFactory` as the
                 ``model_factory`` argument of ``hyperparameter_search`` so
                 that the training loop is completely unchanged.

5. Missing-image confounder analysis:
   For image_only and text+image runs, evaluate the trained model separately
   on test nodes WITH available images and test nodes WITHOUT images.  Flag
   a "⚠  CONFOUNDER" warning when the accuracy gap exceeds 5 percentage
   points.

6. Print the H2 benchmark table and confounder analysis table.

Usage
-----
# Full run (downloads neural encoders and product images on first pass):
    python scripts/run_h2.py --config experiments/h2_multimodal.yaml

# Quick smoke test (synthetic data, sparse text encoder, resnet, concat only):
    python scripts/run_h2.py \\
        --config experiments/h2_multimodal.yaml \\
        --mock --modalities text_only image_only \\
        --vision-encoders resnet --fusion concat

# Skip image_only, use a single fusion strategy:
    python scripts/run_h2.py \\
        --config experiments/h2_multimodal.yaml \\
        --modalities text_only text+image --fusion concat
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import yaml

# Ensure the project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.image_pipeline import fetch_and_cache_images, get_cached_image_paths
from src.encoders.registry import build_encoder
from src.eval.harness import aggregate_metrics, evaluate, run_seed_sweep
from src.fusion.fusion import FusedModelFactory, build_fusion
from src.models.baselines import LogRegClassifier, MLPClassifier
from src.models.gnns import GAT, GCN, GraphSAGE
from src.train import (
    TrainConfig,
    _build_model_for_config,
    _move_data,
    _resolve_device,
    hyperparameter_search,
    train_model,
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ── Model / encoder registries ───────────────────────────────────────────────
MODEL_REGISTRY: dict[str, Any] = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "MLP": MLPClassifier,
    "LogReg": LogRegClassifier,
}

GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT"}

# Keys that appear in YAML encoder configs but are NOT forwarded to build_encoder.
_METADATA_KEYS = frozenset(
    {"modality", "tier", "encoder_size_params", "pretraining_objective"}
)

# Accuracy-gap threshold for confounder flag (percentage points).
_CONFOUNDER_GAP_THRESHOLD_PP = 5.0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_real_dataset(cfg: dict[str, Any]) -> Any:
    """Load the Amazon co-purchase graph from disk.

    Args:
        cfg: The ``dataset`` sub-dict from the experiment YAML.

    Returns:
        A PyG ``Data`` object with ``x = torch.empty(N, 0)`` and a
        ``meta`` attribute containing per-node dicts with ``image_url``.

    Raises:
        FileNotFoundError: If the raw metadata files are not present.
    """
    from src.data.amazon_dataset import AmazonCopurchase

    logger.info("Loading Amazon co-purchase dataset …")
    ds = AmazonCopurchase(
        root=cfg["root"],
        categories=cfg["categories"],
        split_seed=cfg["split_seed"],
        split_ratios=tuple(cfg["split_ratios"]),
        label_level=cfg["label_level"],
    )
    data = ds[0]
    logger.info(
        "Dataset: N=%d  E=%d  C=%d  train=%d val=%d test=%d",
        data.num_nodes,
        data.edge_index.shape[1] // 2,
        len(data.class_names),
        int(data.train_mask.sum()),
        int(data.val_mask.sum()),
        int(data.test_mask.sum()),
    )
    return data


def make_mock_dataset(n_nodes: int = 200, n_classes: int = 4) -> Any:
    """Build a small synthetic dataset for end-to-end smoke testing.

    All ``image_url`` fields are empty strings so that the image pipeline
    returns an all-False availability mask.  Image embeddings will therefore
    be all-zero vectors, which is the expected behaviour for missing images.

    Args:
        n_nodes: Number of synthetic product nodes.
        n_classes: Number of simulated product sub-categories.

    Returns:
        A :class:`types.SimpleNamespace` with the same attributes used by
        the training loop: ``x``, ``edge_index``, ``y``, ``*_mask``,
        ``meta``, ``class_names``, ``num_nodes``.
    """
    torch.manual_seed(0)
    y = torch.randint(0, n_classes, (n_nodes,))

    src = torch.randint(0, n_nodes, (n_nodes * 3,))
    tgt = torch.randint(0, n_nodes, (n_nodes * 3,))
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]
    edge_index = torch.stack([
        torch.cat([src, tgt]),
        torch.cat([tgt, src]),
    ], dim=0)

    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(0))
    n_train = int(0.6 * n_nodes)
    n_val   = int(0.2 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]]               = True
    val_mask[perm[n_train: n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]]        = True

    meta = [
        {
            "title":       f"Product {i}",
            "description": f"Category {'ABCD'[y[i]]} product number {i}",
            "image_url":   "",    # empty → image pipeline returns zero vectors
        }
        for i in range(n_nodes)
    ]
    class_names = [f"Cat_{c}" for c in range(n_classes)]
    data = SimpleNamespace(
        x=torch.empty(n_nodes, 0),
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        meta=meta,
        class_names=class_names,
        num_nodes=n_nodes,
    )
    logger.info(
        "Mock dataset: N=%d  E=%d  C=%d  train=%d val=%d test=%d",
        n_nodes, edge_index.shape[1] // 2, n_classes,
        int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()),
    )
    return data


# ---------------------------------------------------------------------------
# Prior-experiment result loaders
# ---------------------------------------------------------------------------


def load_h1_best_gnn(
    h1_results_path: str | Path,
    fallback_gnn: str,
    fallback_hparams: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Read H1 results and return the best graph model and its hyperparams.

    The "best" GNN is the one with the highest mean test accuracy across its
    seed runs.  Its hyperparameters (from the first run) are returned as
    warm-start candidates for the H2 hyperparameter search.

    Args:
        h1_results_path: Path to ``results/h1_results.json``.
        fallback_gnn: Model name used when the file cannot be read.
        fallback_hparams: Hyperparams used when the file cannot be read.

    Returns:
        ``(model_name, hyperparams_dict)``.
    """
    path = Path(h1_results_path)
    if not path.exists():
        logger.warning(
            "H1 results not found at '%s' — using fallback GNN='%s'.",
            path, fallback_gnn,
        )
        return fallback_gnn, dict(fallback_hparams)

    try:
        with path.open("r", encoding="utf-8") as fh:
            results: list[dict[str, Any]] = json.load(fh)
    except Exception as exc:
        logger.warning("Could not parse '%s' (%s) — using fallback.", path, exc)
        return fallback_gnn, dict(fallback_hparams)

    graph_runs = [r for r in results if r.get("model_name") in GRAPH_MODELS]
    if not graph_runs:
        logger.warning("No graph-model runs found in H1 results — using fallback.")
        return fallback_gnn, dict(fallback_hparams)

    acc_by_model: dict[str, list[float]] = defaultdict(list)
    for run in graph_runs:
        acc_by_model[run["model_name"]].append(run.get("test_accuracy", 0.0))

    best_model = max(
        acc_by_model,
        key=lambda m: sum(acc_by_model[m]) / len(acc_by_model[m]),
    )
    best_mean = sum(acc_by_model[best_model]) / len(acc_by_model[best_model])
    best_runs = [r for r in graph_runs if r["model_name"] == best_model]
    warm_hparams = dict(best_runs[0].get("hyperparams", fallback_hparams))

    logger.info(
        "H1 best GNN: '%s' (mean test_acc=%.4f)  warm-start hparams: %s",
        best_model, best_mean, warm_hparams,
    )
    return best_model, warm_hparams


def load_h3_best_text_encoder(
    h3_results_path: str | Path,
    fallback_name: str,
    text_encoder_cfgs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read H3 results and return the config for the best text encoder.

    The "best" encoder is the one with the highest mean test accuracy in H3,
    restricted to encoders whose ``name`` appears in ``text_encoder_cfgs``.

    Args:
        h3_results_path: Path to ``results/h3_results.json``.
        fallback_name: Encoder name to use when H3 results are unavailable.
            Must match a ``name`` in ``text_encoder_cfgs``.
        text_encoder_cfgs: List of encoder config dicts from the YAML
            ``text_encoders`` key.

    Returns:
        The encoder config dict for the best (or fallback) encoder.
    """
    encoder_by_name = {e["name"]: e for e in text_encoder_cfgs}
    default_cfg = encoder_by_name.get(
        fallback_name, text_encoder_cfgs[0] if text_encoder_cfgs else {}
    )

    path = Path(h3_results_path)
    if not path.exists():
        logger.warning(
            "H3 results not found at '%s' — using fallback text encoder '%s'.",
            path, fallback_name,
        )
        return default_cfg

    try:
        with path.open("r", encoding="utf-8") as fh:
            results: list[dict[str, Any]] = json.load(fh)
    except Exception as exc:
        logger.warning("Could not parse '%s' (%s) — using fallback.", path, exc)
        return default_cfg

    acc_by_enc: dict[str, list[float]] = defaultdict(list)
    for r in results:
        name = r.get("encoder_name", "")
        if name in encoder_by_name:
            acc_by_enc[name].append(r.get("test_accuracy", 0.0))

    if not acc_by_enc:
        logger.warning(
            "No H3 results match YAML text encoders %s — using fallback '%s'.",
            list(encoder_by_name), fallback_name,
        )
        return default_cfg

    best_name = max(
        acc_by_enc,
        key=lambda k: sum(acc_by_enc[k]) / len(acc_by_enc[k]),
    )
    best_mean = sum(acc_by_enc[best_name]) / len(acc_by_enc[best_name])
    logger.info(
        "H3 best text encoder: '%s' (mean test_acc=%.4f)",
        best_name, best_mean,
    )
    return encoder_by_name[best_name]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def compute_text_embeddings(
    encoder_cfg: dict[str, Any],
    data: Any,
    embeddings_dir: Path,
    device: str = "auto",
) -> torch.Tensor:
    """Fit a text encoder on training texts and encode all nodes.

    Statistical encoders (BoW, TF-IDF) are fitted on training-split texts
    only; frozen neural encoders skip the fit step.  Results are cached.

    Args:
        encoder_cfg: Encoder dict from the YAML ``text_encoders`` list.
        data: Dataset with a ``meta`` list (each dict has ``title`` and
            ``description``) and ``train_mask`` tensor.
        embeddings_dir: Root directory for ``.pt`` cache files.
        device: Inference device forwarded to :func:`build_encoder`.

    Returns:
        Dense ``(N, D_text)`` float32 tensor.
    """
    enc_name: str = encoder_cfg["name"]
    texts = [m["title"] + " " + m["description"] for m in data.meta]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    train_texts = [texts[i] for i in train_idx]

    logger.info("Building text encoder '%s' (type=%s) …", enc_name, encoder_cfg.get("type"))
    enc_build_cfg = {k: v for k, v in encoder_cfg.items() if k not in _METADATA_KEYS}
    encoder = build_encoder(enc_build_cfg, cache_dir=embeddings_dir, device=device)

    logger.info("Fitting '%s' on %d training texts …", enc_name, len(train_texts))
    encoder.fit(train_texts)

    cache_key = f"{enc_name}_split0"
    logger.info("Encoding %d texts (cache_key='%s') …", len(texts), cache_key)
    x = encoder.encode_cached(texts, cache_key)
    logger.info("Text embeddings '%s': shape=%s", enc_name, tuple(x.shape))
    return x


def compute_vision_embeddings(
    encoder_cfg: dict[str, Any],
    image_paths: list[Path | None],
    embeddings_dir: Path,
    device: str = "auto",
) -> torch.Tensor:
    """Encode all product images with a vision encoder.

    Nodes whose images are unavailable (``None`` in ``image_paths``) receive
    a **zero vector** from the vision encoder.

    Args:
        encoder_cfg: Encoder dict from the YAML ``vision_encoders`` list.
        image_paths: List of length ``N`` — a ``pathlib.Path`` for each
            cached JPEG, or ``None`` for unavailable images.
        embeddings_dir: Root directory for ``.pt`` cache files.
        device: Inference device forwarded to :func:`build_encoder`.

    Returns:
        Dense ``(N, D_vision)`` float32 tensor (zero rows for missing images).
    """
    enc_name: str = encoder_cfg["name"]
    enc_build_cfg = {k: v for k, v in encoder_cfg.items() if k not in _METADATA_KEYS}

    logger.info("Building vision encoder '%s' (type=%s) …", enc_name, encoder_cfg.get("type"))
    encoder = build_encoder(enc_build_cfg, cache_dir=embeddings_dir, device=device)

    # Vision encoders are always frozen — no fit() needed.
    cache_key = f"{enc_name}_vision_split0"
    logger.info(
        "Encoding %d image paths (cache_key='%s') …",
        len(image_paths), cache_key,
    )
    x = encoder.encode_cached(image_paths, cache_key)
    n_valid = sum(1 for p in image_paths if p is not None)
    logger.info(
        "Vision embeddings '%s': shape=%s  valid=%d / %d",
        enc_name, tuple(x.shape), n_valid, len(image_paths),
    )
    return x


# ---------------------------------------------------------------------------
# Core training function for one modality combo
# ---------------------------------------------------------------------------


def run_modality_combo(
    combo_name: str,
    modality: str,
    combo_meta: dict[str, Any],
    model_cls: Any,
    model_factory: Callable[[TrainConfig], Any] | None,
    data: Any,
    image_available: np.ndarray | None,
    training_cfg: dict[str, Any],
    search_space: dict[str, list[Any]],
    base_config: TrainConfig,
    warm_start_params: dict[str, Any] | None,
    results_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run hyperparameter search and seed sweep for one modality combination.

    The function assumes ``data.x`` has already been set to the appropriate
    embedding tensor for this combination.  For ``text+image`` combos,
    ``data.x = cat([text_emb, image_emb], dim=-1)`` and ``model_factory``
    is a :class:`~src.fusion.fusion.FusedModelFactory`.

    If ``image_available`` is not ``None`` (i.e. modality is ``image_only``
    or ``text+image``), each seed run additionally evaluates the trained
    model on the test-with-images and test-without-images subsets for the
    missing-image confounder analysis.

    Args:
        combo_name: Short identifier used for checkpoint and log file naming
            (e.g. ``"h2_text_only_tfidf"``).
        modality: One of ``"text_only"``, ``"image_only"``, ``"text+image"``.
        combo_meta: Dict of metadata fields (modality, text_enc, vision_enc,
            fusion) that are injected into every result record.
        model_cls: GNN class.  Ignored when ``model_factory`` is not None.
        model_factory: Optional callable ``(TrainConfig) → nn.Module``
            injected into :func:`~src.train.hyperparameter_search` for the
            text+image fused-model case.
        data: Dataset with ``data.x`` set.
        image_available: Boolean ``np.ndarray`` of shape ``(N,)`` used for
            confounder analysis.  Pass ``None`` for text_only combos.
        training_cfg: ``training`` sub-dict from the YAML.
        search_space: GNN hyperparameter search grid.
        base_config: Template :class:`~src.train.TrainConfig`.
        warm_start_params: Optional warm-start hyperparams for trial 0.
        results_dir: Directory for per-seed JSON result files.
        device: Inference device string.

    Returns:
        ``(per_seed_runs, summary)`` from
        :func:`~src.eval.harness.run_seed_sweep`.
    """
    n_hparam = training_cfg["n_hparam_trials"]
    n_seeds  = training_cfg["n_seed_runs"]
    base_seed = training_cfg["base_seed"]
    in_channels = int(data.x.shape[1])

    model_config = replace(
        base_config,
        encoder_name=combo_name,
        model_name=base_config.model_name,
        in_channels=in_channels,
    )

    logger.info("=" * 70)
    logger.info("Combo: %-30s  modality=%s", combo_name, modality)
    logger.info("  in_channels=%d  model_factory=%s", in_channels, model_factory is not None)
    logger.info("=" * 70)

    # ── Hyperparameter search ─────────────────────────────────────────────
    logger.info(
        "Hyperparameter search: %d trials (warm-start=%s) …",
        n_hparam, warm_start_params is not None,
    )
    hparam_result = hyperparameter_search(
        model_cls=model_cls,
        data=data,
        search_space=search_space,
        n_trials=n_hparam,
        base_config=model_config,
        results_dir=results_dir,
        warm_start_params=warm_start_params,
        model_factory=model_factory,
    )
    best_params = hparam_result["best_params"]
    best_val    = hparam_result["best_val_metric"]
    logger.info("Best hparams (val_acc=%.4f): %s", best_val, best_params)

    best_config = replace(
        model_config,
        lr=float(best_params.get("lr",              model_config.lr)),
        weight_decay=float(best_params.get("weight_decay",   model_config.weight_decay)),
        hidden_channels=int(best_params.get("hidden_channels", model_config.hidden_channels)),
        dropout=float(best_params.get("dropout",           model_config.dropout)),
        C=float(best_params.get("C",                 model_config.C)),
    )

    # ── Seed sweep ────────────────────────────────────────────────────────
    logger.info("Seed sweep: %d runs …", n_seeds)

    def train_fn(seed: int) -> dict[str, Any]:
        """Train one seed, with optional confounder subset evaluation.

        Closure captures: ``model_cls``, ``model_factory``, ``data``,
        ``best_config``, ``results_dir``, ``combo_meta``, ``combo_name``,
        ``image_available``, ``modality``.
        """
        run_config = replace(best_config, seed=seed)
        set_seed(seed)

        # Build model.
        if model_factory is not None:
            model = model_factory(run_config)
        else:
            model = _build_model_for_config(model_cls, run_config)

        result = train_model(model, data, run_config)
        result.update(combo_meta)  # inject modality/encoder metadata

        # ── Confounder analysis ────────────────────────────────────────
        # After train_model the model has the best-checkpoint weights loaded.
        # Evaluate on image-available vs image-unavailable test subsets.
        if image_available is not None:
            # Always evaluate on CPU to avoid device-placement complexity.
            model_cpu = model.cpu()
            data_eval = SimpleNamespace(
                x=data.x.detach().cpu(),
                edge_index=data.edge_index.detach().cpu(),
                y=data.y.detach().cpu(),
                train_mask=data.train_mask.detach().cpu(),
                val_mask=data.val_mask.detach().cpu(),
                test_mask=data.test_mask.detach().cpu(),
            )
            img_t = torch.from_numpy(image_available).bool()   # (N,)
            test_with    = data_eval.test_mask & img_t
            test_without = data_eval.test_mask & ~img_t

            if test_with.any():
                m = evaluate(model_cpu, data_eval, test_with)
                result["test_acc_with_images"]   = m["accuracy"]
                result["n_test_with_images"]     = int(test_with.sum())
            else:
                result["test_acc_with_images"]   = float("nan")
                result["n_test_with_images"]     = 0

            if test_without.any():
                m = evaluate(model_cpu, data_eval, test_without)
                result["test_acc_without_images"] = m["accuracy"]
                result["n_test_without_images"]   = int(test_without.sum())
            else:
                result["test_acc_without_images"] = float("nan")
                result["n_test_without_images"]   = 0

        # Write per-seed JSON.
        out_path = results_dir / f"h2_{combo_name}_seed{seed}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            # json.dump cannot serialise nan; replace with null.
            serialisable = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in result.items()
            }
            json.dump(serialisable, fh, indent=2)
        logger.info(
            "Seed %d → test_acc=%.4f | %s",
            seed, result["test_accuracy"], out_path.name,
        )
        return result

    per_seed, summary = run_seed_sweep(train_fn, n_seeds=n_seeds, base_seed=base_seed)
    logger.info(
        "%s: mean_acc=%.4f ± %.4f  CI95=[%.4f, %.4f]",
        combo_name,
        summary.get("test_accuracy_mean", 0.0),
        summary.get("test_accuracy_std", 0.0),
        summary.get("test_accuracy_ci95_lower", 0.0),
        summary.get("test_accuracy_ci95_upper", 0.0),
    )
    return per_seed, summary


# ---------------------------------------------------------------------------
# Benchmark table printing
# ---------------------------------------------------------------------------

_H2_COLS = [
    ("modality",    12),
    ("text_enc",    14),
    ("vision_enc",  12),
    ("fusion",      10),
    ("model",       12),
    ("mean_acc",     9),
    ("std_acc",      9),
    ("ci95_lower",  10),
    ("ci95_upper",  10),
    ("n_runs",       7),
    ("avg_rt_s",    10),
]

_CONF_COLS = [
    ("combo",          30),
    ("modality",       12),
    ("n_w_img",         8),
    ("acc_w_img",      10),
    ("n_wo_img",        8),
    ("acc_wo_img",     11),
    ("gap_pp",          8),
    ("flag",           12),
]


def _fmt_acc(x: float | None) -> str:
    """Format an accuracy value, returning 'N/A' for None / NaN.

    Args:
        x: Float accuracy or ``None``.

    Returns:
        4-decimal string or ``"N/A"``.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A   "
    return f"{x:.4f}"


def print_h2_table(benchmark_rows: list[dict[str, Any]]) -> None:
    """Print the H2 benchmark table sorted by mean accuracy descending.

    Args:
        benchmark_rows: List of benchmark row dicts (one per combo).
    """
    rows = sorted(benchmark_rows, key=lambda r: r.get("mean_acc", 0.0), reverse=True)
    header = "  ".join(c.ljust(w) for c, w in _H2_COLS)
    sep    = "  ".join("-" * w for _, w in _H2_COLS)
    width  = len(sep)

    print(f"\n{'=' * width}")
    print("H2: Multimodal vs Single-Modality Benchmark")
    print("=" * width)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join([
            str(r.get("modality",    "")).ljust(_H2_COLS[0][1]),
            str(r.get("text_enc",    "—")).ljust(_H2_COLS[1][1]),
            str(r.get("vision_enc",  "—")).ljust(_H2_COLS[2][1]),
            str(r.get("fusion",      "—")).ljust(_H2_COLS[3][1]),
            str(r.get("model",       "")).ljust(_H2_COLS[4][1]),
            f"{r.get('mean_acc',   0.0):.4f}".ljust(_H2_COLS[5][1]),
            f"{r.get('std_acc',    0.0):.4f}".ljust(_H2_COLS[6][1]),
            f"{r.get('ci95_lower', 0.0):.4f}".ljust(_H2_COLS[7][1]),
            f"{r.get('ci95_upper', 0.0):.4f}".ljust(_H2_COLS[8][1]),
            str(r.get("n_runs",      "")).ljust(_H2_COLS[9][1]),
            f"{r.get('avg_rt_s',   0.0):.1f}".ljust(_H2_COLS[10][1]),
        ]))
    print("=" * width + "\n")


def print_confounder_table(benchmark_rows: list[dict[str, Any]]) -> None:
    """Print the missing-image confounder analysis table.

    Only rows with image_only or text+image modalities are included.
    A "⚠ CONFOUNDER" flag is shown when |acc_with - acc_without| > 5 pp.

    Args:
        benchmark_rows: Same list as passed to :func:`print_h2_table`.
    """
    vision_rows = [
        r for r in benchmark_rows
        if r.get("modality") in ("image_only", "text+image")
        and r.get("acc_with_images") is not None
    ]
    if not vision_rows:
        logger.info("Confounder analysis: no vision rows with subset metrics — skipping.")
        return

    header = "  ".join(c.ljust(w) for c, w in _CONF_COLS)
    sep    = "  ".join("-" * w for _, w in _CONF_COLS)
    width  = len(sep)

    print(f"\n{'=' * width}")
    print("H2 Missing-Image Confounder Analysis")
    print(f"(flag ⚠ if |acc_with_images − acc_without_images| > {_CONFOUNDER_GAP_THRESHOLD_PP:.0f} pp)")
    print("=" * width)
    print(header)
    print(sep)

    for r in vision_rows:
        combo_name = (
            f"{r.get('modality', '')}/"
            f"{r.get('vision_enc', '')}/"
            f"{r.get('fusion', '')}"
        ).rstrip("/")
        acc_with    = r.get("acc_with_images")
        acc_without = r.get("acc_without_images")

        if (
            acc_with  is not None and not math.isnan(acc_with)
            and acc_without is not None and not math.isnan(acc_without)
        ):
            gap = abs(acc_with - acc_without) * 100.0
            flag = f"⚠ CONFOUNDER" if gap > _CONFOUNDER_GAP_THRESHOLD_PP else "OK"
        else:
            gap = float("nan")
            flag = "N/A"

        print("  ".join([
            combo_name.ljust(_CONF_COLS[0][1]),
            str(r.get("modality", "")).ljust(_CONF_COLS[1][1]),
            str(r.get("n_with_images",    "N/A")).ljust(_CONF_COLS[2][1]),
            _fmt_acc(acc_with).ljust(_CONF_COLS[3][1]),
            str(r.get("n_without_images", "N/A")).ljust(_CONF_COLS[4][1]),
            _fmt_acc(acc_without).ljust(_CONF_COLS[5][1]),
            (f"{gap:.2f}" if not math.isnan(gap) else "N/A").ljust(_CONF_COLS[6][1]),
            flag.ljust(_CONF_COLS[7][1]),
        ]))

    n_flagged = sum(
        1 for r in vision_rows
        if r.get("acc_with_images") is not None
        and r.get("acc_without_images") is not None
        and not math.isnan(r.get("acc_with_images", float("nan")))
        and not math.isnan(r.get("acc_without_images", float("nan")))
        and abs(r["acc_with_images"] - r["acc_without_images"]) * 100 > _CONFOUNDER_GAP_THRESHOLD_PP
    )
    print("=" * width)
    if n_flagged > 0:
        print(
            f"⚠  {n_flagged} combo(s) show a >{_CONFOUNDER_GAP_THRESHOLD_PP:.0f} pp gap — "
            "results may be confounded by image missingness.\n"
        )
    else:
        print("No confounder gaps detected above threshold.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the H2 experiment.

    Args:
        argv: Override ``sys.argv[1:]`` for programmatic invocation.
    """
    parser = argparse.ArgumentParser(
        description="Run the H2 experiment (multimodal text + image comparison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML (e.g. experiments/h2_multimodal.yaml).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help=(
            "Use a small synthetic dataset instead of real Amazon data.  "
            "Images will be all-missing (zero vectors), which is intentional."
        ),
    )
    parser.add_argument(
        "--modalities", nargs="+",
        choices=["text_only", "image_only", "text+image"],
        metavar="MODALITY",
        help="Restrict to specific modalities (default: all in YAML).",
    )
    parser.add_argument(
        "--vision-encoders", nargs="+", metavar="ENC",
        help=(
            "Restrict vision encoders by name "
            "(e.g. --vision-encoders resnet clip-vision).  "
            "In --mock mode defaults to [resnet] if not specified."
        ),
    )
    parser.add_argument(
        "--fusion", nargs="+",
        choices=["concat", "weighted", "gated"],
        metavar="FUSION",
        help=(
            "Restrict fusion strategies for text+image mode "
            "(e.g. --fusion concat).  "
            "In --mock mode defaults to [concat] if not specified."
        ),
    )
    parser.add_argument(
        "--text-encoder", default=None, metavar="ENC",
        help=(
            "Override the text encoder selected from H3 results "
            "(e.g. --text-encoder tfidf)."
        ),
    )
    parser.add_argument(
        "--model", default=None, metavar="MODEL",
        help=(
            "Override the GNN selected from H1 results "
            "(e.g. --model MLP for a quick smoke test without torch_geometric)."
        ),
    )
    parser.add_argument(
        "--skip-image-download", action="store_true",
        help=(
            "Skip downloading images; use only already-cached images.  "
            "Useful when re-running after a previous download pass."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Override inference device (auto / cpu / cuda).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load YAML config ─────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as fh:
        exp_cfg: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config: %s", config_path)

    dataset_cfg   = exp_cfg["dataset"]
    training_cfg  = exp_cfg["training"]
    results_cfg   = exp_cfg["results"]
    search_space  = exp_cfg["gnn_search_space"]
    img_pipe_cfg  = exp_cfg.get("image_pipeline", {})

    results_dir    = Path(results_cfg["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file    = Path(results_cfg["output_file"])
    embeddings_dir = Path(results_cfg.get("embeddings_dir", "data/embeddings"))
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = Path(img_pipe_cfg.get("cache_dir", "data/images"))

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.mock:
        logger.info("Using MOCK dataset (--mock flag).")
        data = make_mock_dataset(n_nodes=200, n_classes=4)
    else:
        try:
            data = load_real_dataset(dataset_cfg)
        except FileNotFoundError as exc:
            logger.error(
                "Raw dataset not found: %s\n"
                "Download from https://nijianmo.github.io/amazon/ and place files "
                "in %s/raw/. Or run with --mock.",
                exc, dataset_cfg["root"],
            )
            sys.exit(1)

    out_channels = len(data.class_names)
    n_nodes      = data.num_nodes if hasattr(data, "num_nodes") else len(data.meta)

    # ── Determine modalities, vision encoders, and fusion strategies ──────────
    all_modalities      = exp_cfg.get("modalities", ["text_only", "image_only", "text+image"])
    all_vision_cfgs     = exp_cfg.get("vision_encoders", [])
    all_fusion_names    = exp_cfg.get("fusion_strategies", ["concat", "weighted", "gated"])
    all_text_enc_cfgs   = exp_cfg.get("text_encoders", [])

    active_modalities = args.modalities if args.modalities else all_modalities

    if args.vision_encoders:
        vision_cfgs = [v for v in all_vision_cfgs if v["name"] in args.vision_encoders]
    elif args.mock:
        vision_cfgs = all_vision_cfgs[:1]   # only first (resnet) in mock mode
        logger.info("--mock: restricting to first vision encoder '%s'.", vision_cfgs[0]["name"])
    else:
        vision_cfgs = all_vision_cfgs

    if args.fusion:
        fusion_names = [f for f in all_fusion_names if f in args.fusion]
    elif args.mock:
        fusion_names = ["concat"]           # only concat in mock mode
        logger.info("--mock: restricting fusion to ['concat'].")
    else:
        fusion_names = all_fusion_names

    # ── H1 best GNN ───────────────────────────────────────────────────────────
    best_gnn_name, warm_start_params = load_h1_best_gnn(
        h1_results_path=exp_cfg.get("h1_results_path", "results/h1_results.json"),
        fallback_gnn=exp_cfg.get("fallback_gnn", "GCN"),
        fallback_hparams=exp_cfg.get("fallback_hparams", {}),
    )
    if args.model is not None:
        logger.info("--model overrides H1 best GNN: '%s' → '%s'.", best_gnn_name, args.model)
        best_gnn_name = args.model
    if best_gnn_name not in MODEL_REGISTRY:
        logger.error("GNN '%s' not in MODEL_REGISTRY.", best_gnn_name)
        sys.exit(1)
    model_cls = MODEL_REGISTRY[best_gnn_name]
    logger.info("Using GNN: %s  warm-start: %s", best_gnn_name, warm_start_params)

    # ── H3 best text encoder ──────────────────────────────────────────────────
    if args.text_encoder is not None:
        text_enc_by_name = {e["name"]: e for e in all_text_enc_cfgs}
        if args.text_encoder not in text_enc_by_name:
            logger.error("--text-encoder '%s' not found in YAML.", args.text_encoder)
            sys.exit(1)
        best_text_cfg = text_enc_by_name[args.text_encoder]
        logger.info("--text-encoder overrides: using '%s'.", args.text_encoder)
    elif args.mock:
        # Default to first sparse encoder in mock mode.
        sparse = [e for e in all_text_enc_cfgs if e["type"] in ("BagOfWords", "TFIDF")]
        best_text_cfg = sparse[0] if sparse else all_text_enc_cfgs[0]
        logger.info("--mock: using text encoder '%s'.", best_text_cfg["name"])
    else:
        best_text_cfg = load_h3_best_text_encoder(
            h3_results_path=exp_cfg.get("h3_results_path", "results/h3_results.json"),
            fallback_name=exp_cfg.get("fallback_text_encoder", "tfidf"),
            text_encoder_cfgs=all_text_enc_cfgs,
        )
    best_text_name = best_text_cfg["name"]
    logger.info("Text encoder for H2: %s", best_text_name)

    # ── Template TrainConfig ──────────────────────────────────────────────────
    base_config = TrainConfig(
        in_channels=0,
        hidden_channels=training_cfg.get("hidden_channels", 128),
        out_channels=out_channels,
        num_layers=training_cfg["num_layers"],
        num_heads=training_cfg.get("num_heads", 2),
        max_epochs=training_cfg["max_epochs"],
        patience=training_cfg["patience"],
        warmup_epochs=training_cfg["warmup_epochs"],
        min_lr=training_cfg["min_lr"],
        seed=training_cfg["base_seed"],
        encoder_name="",
        model_name=best_gnn_name,
        checkpoint_dir=str(results_dir / "checkpoints"),
        device=args.device,
    )

    # ── Image pipeline ────────────────────────────────────────────────────────
    need_images = any(m in active_modalities for m in ("image_only", "text+image"))

    image_available: np.ndarray | None = None
    image_paths: list[Path | None] = [None] * n_nodes

    if need_images and not args.mock:
        if args.skip_image_download:
            logger.info("--skip-image-download: scanning existing cache …")
            image_available, image_paths = get_cached_image_paths(n_nodes, image_cache_dir)
        else:
            logger.info("Running image pipeline (max_workers=%d) …",
                        img_pipe_cfg.get("max_workers", 8))
            image_available, image_paths = fetch_and_cache_images(
                data=data,
                cache_dir=image_cache_dir,
                max_workers=img_pipe_cfg.get("max_workers", 8),
                timeout=img_pipe_cfg.get("timeout", 15),
                max_retries=img_pipe_cfg.get("max_retries", 2),
            )
            n_avail = int(image_available.sum())
            logger.info(
                "Images available: %d / %d (%.1f%%)",
                n_avail, n_nodes, 100.0 * n_avail / max(1, n_nodes),
            )
    elif need_images and args.mock:
        # All images empty in mock mode.
        image_available = np.zeros(n_nodes, dtype=bool)
        image_paths = [None] * n_nodes
        logger.info("Mock mode: all images set to None (zero vectors).")

    # ── Pre-compute text embeddings once (reused across all combos) ───────────
    needs_text = any(m in active_modalities for m in ("text_only", "text+image"))
    text_emb: torch.Tensor | None = None

    if needs_text:
        logger.info("Computing text embeddings with encoder '%s' …", best_text_name)
        text_emb = compute_text_embeddings(
            encoder_cfg=best_text_cfg,
            data=data,
            embeddings_dir=embeddings_dir,
            device=args.device,
        )
        logger.info("Text embeddings ready: shape=%s", tuple(text_emb.shape))

    # ── Per-vision encoder image embeddings (computed on demand) ─────────────
    # Cache the vision embeddings by encoder name so each is computed once even
    # if used in both image_only and text+image.
    vision_emb_cache: dict[str, torch.Tensor] = {}

    def get_vision_emb(vcfg: dict[str, Any]) -> torch.Tensor:
        """Return (or compute) the vision embedding for one encoder config."""
        vname = vcfg["name"]
        if vname not in vision_emb_cache:
            vision_emb_cache[vname] = compute_vision_embeddings(
                encoder_cfg=vcfg,
                image_paths=image_paths,
                embeddings_dir=embeddings_dir,
                device=args.device,
            )
        return vision_emb_cache[vname]

    # ── Main experiment loop ──────────────────────────────────────────────────
    all_results:    list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for modality in active_modalities:

        # ── text_only ─────────────────────────────────────────────────────────
        if modality == "text_only":
            assert text_emb is not None
            combo_name = f"text_only_{best_text_name}"
            combo_meta = {
                "modality":   "text_only",
                "text_enc":   best_text_name,
                "vision_enc": None,
                "fusion":     None,
            }
            data_copy = deepcopy(data)
            data_copy.x = text_emb

            per_seed, summary = run_modality_combo(
                combo_name=combo_name,
                modality="text_only",
                combo_meta=combo_meta,
                model_cls=model_cls,
                model_factory=None,
                data=data_copy,
                image_available=None,   # no confounder analysis for text_only
                training_cfg=training_cfg,
                search_space=search_space,
                base_config=base_config,
                warm_start_params=warm_start_params,
                results_dir=results_dir,
                device=args.device,
            )
            all_results.extend(per_seed)
            avg_rt = sum(r.get("runtime_seconds", 0) for r in per_seed) / max(1, len(per_seed))
            benchmark_rows.append({
                "modality":   "text_only",
                "text_enc":   best_text_name,
                "vision_enc": None,
                "fusion":     None,
                "model":      best_gnn_name,
                "mean_acc":   summary.get("test_accuracy_mean",      0.0),
                "std_acc":    summary.get("test_accuracy_std",        0.0),
                "ci95_lower": summary.get("test_accuracy_ci95_lower", 0.0),
                "ci95_upper": summary.get("test_accuracy_ci95_upper", 0.0),
                "n_runs":     summary.get("n_runs",                   0),
                "avg_rt_s":   avg_rt,
            })

        # ── image_only ────────────────────────────────────────────────────────
        elif modality == "image_only":
            for vcfg in vision_cfgs:
                vname = vcfg["name"]
                image_emb = get_vision_emb(vcfg)

                combo_name = f"image_only_{vname}"
                combo_meta = {
                    "modality":   "image_only",
                    "text_enc":   None,
                    "vision_enc": vname,
                    "fusion":     None,
                }
                data_copy = deepcopy(data)
                data_copy.x = image_emb

                per_seed, summary = run_modality_combo(
                    combo_name=combo_name,
                    modality="image_only",
                    combo_meta=combo_meta,
                    model_cls=model_cls,
                    model_factory=None,
                    data=data_copy,
                    image_available=image_available,
                    training_cfg=training_cfg,
                    search_space=search_space,
                    base_config=base_config,
                    warm_start_params=warm_start_params,
                    results_dir=results_dir,
                    device=args.device,
                )
                all_results.extend(per_seed)
                avg_rt = sum(r.get("runtime_seconds", 0) for r in per_seed) / max(1, len(per_seed))
                row: dict[str, Any] = {
                    "modality":   "image_only",
                    "text_enc":   None,
                    "vision_enc": vname,
                    "fusion":     None,
                    "model":      best_gnn_name,
                    "mean_acc":   summary.get("test_accuracy_mean",      0.0),
                    "std_acc":    summary.get("test_accuracy_std",        0.0),
                    "ci95_lower": summary.get("test_accuracy_ci95_lower", 0.0),
                    "ci95_upper": summary.get("test_accuracy_ci95_upper", 0.0),
                    "n_runs":     summary.get("n_runs",                   0),
                    "avg_rt_s":   avg_rt,
                }
                # Confounder stats (aggregated over seeds).
                if summary.get("test_acc_with_images_mean") is not None:
                    row["acc_with_images"]    = summary["test_acc_with_images_mean"]
                    row["n_with_images"]      = summary.get("n_test_with_images_mean")
                    row["acc_without_images"] = summary.get("test_acc_without_images_mean")
                    row["n_without_images"]   = summary.get("n_test_without_images_mean")
                benchmark_rows.append(row)

        # ── text+image ────────────────────────────────────────────────────────
        elif modality == "text+image":
            assert text_emb is not None
            for vcfg in vision_cfgs:
                vname = vcfg["name"]
                image_emb = get_vision_emb(vcfg)
                text_dim  = int(text_emb.shape[1])
                image_dim = int(image_emb.shape[1])

                for fusion_name in fusion_names:
                    combo_name = f"h2_{best_text_name}_{vname}_{fusion_name}"
                    combo_meta = {
                        "modality":   "text+image",
                        "text_enc":   best_text_name,
                        "vision_enc": vname,
                        "fusion":     fusion_name,
                    }
                    data_copy   = deepcopy(data)
                    # Concatenate: FusedModel will split at [:, text_dim].
                    data_copy.x = torch.cat([text_emb, image_emb], dim=-1)

                    factory = FusedModelFactory(
                        gnn_cls=model_cls,
                        fusion_name=fusion_name,
                        text_dim=text_dim,
                        image_dim=image_dim,
                    )

                    per_seed, summary = run_modality_combo(
                        combo_name=combo_name,
                        modality="text+image",
                        combo_meta=combo_meta,
                        model_cls=model_cls,
                        model_factory=factory,
                        data=data_copy,
                        image_available=image_available,
                        training_cfg=training_cfg,
                        search_space=search_space,
                        base_config=base_config,
                        warm_start_params=warm_start_params,
                        results_dir=results_dir,
                        device=args.device,
                    )
                    all_results.extend(per_seed)
                    avg_rt = (
                        sum(r.get("runtime_seconds", 0) for r in per_seed)
                        / max(1, len(per_seed))
                    )
                    row = {
                        "modality":   "text+image",
                        "text_enc":   best_text_name,
                        "vision_enc": vname,
                        "fusion":     fusion_name,
                        "model":      best_gnn_name,
                        "mean_acc":   summary.get("test_accuracy_mean",      0.0),
                        "std_acc":    summary.get("test_accuracy_std",        0.0),
                        "ci95_lower": summary.get("test_accuracy_ci95_lower", 0.0),
                        "ci95_upper": summary.get("test_accuracy_ci95_upper", 0.0),
                        "n_runs":     summary.get("n_runs",                   0),
                        "avg_rt_s":   avg_rt,
                    }
                    if summary.get("test_acc_with_images_mean") is not None:
                        row["acc_with_images"]    = summary["test_acc_with_images_mean"]
                        row["n_with_images"]      = summary.get("n_test_with_images_mean")
                        row["acc_without_images"] = summary.get("test_acc_without_images_mean")
                        row["n_without_images"]   = summary.get("n_test_without_images_mean")
                    benchmark_rows.append(row)

    # ── Save all results ──────────────────────────────────────────────────────
    serialisable_results = [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        for r in all_results
    ]
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(serialisable_results, fh, indent=2)
    logger.info("All %d per-seed results saved to %s", len(all_results), output_file)

    # ── Print benchmark tables ────────────────────────────────────────────────
    print_h2_table(benchmark_rows)
    print_confounder_table(benchmark_rows)
    logger.info("H2 experiment complete.")


if __name__ == "__main__":
    main()
