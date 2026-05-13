"""Concurrent image downloader and cache manager for the H2 multimodal pipeline.

Role in the pipeline
--------------------
The Amazon product dataset ships *only* metadata (title, description, image URL).
This module is responsible for the "image URL → cached JPEG on disk" step that
must be run **once** before any vision encoder can be called.

``fetch_and_cache_images`` downloads all product images in parallel, validates
each file, and returns a boolean availability mask that propagates through the
rest of the pipeline:

- Vision encoders call ``encode(image_paths)`` where ``image_paths[i]`` is a
  ``Path`` object for available images and ``None`` for missing ones.
- For missing images the vision encoder returns a **zero vector** of the same
  dimensionality as the valid embeddings.

Missing-image policy
--------------------
Filling unavailable images with zero vectors is the simplest choice and
preserves the node count required by transductive GNN training.  However,
it is a **potential confounder** in the H2 results:

    If the missingness pattern correlates with the label (e.g. expensive
    electronics categories tend to have images while accessory categories
    do not), then a model that learns to use the zero-image signal can appear
    better than it is.

The H2 runner therefore separately reports accuracy on:
- All test nodes
- Test nodes *with* available images
- Test nodes *without* available images

and flags a ">5 pp gap" warning when the difference is significant.

Download strategy
-----------------
- ``concurrent.futures.ThreadPoolExecutor`` with up to ``max_workers`` threads.
- Per-request ``timeout`` seconds (connect + read).
- ``max_retries`` attempts per URL with exponential back-off.
- Graceful handling of empty URL, HTTP error, timeout, connection error,
  and PIL decode error (corrupt / non-image response).
- Already-cached images are **skipped** on repeat runs.

Cache layout
------------
``{cache_dir}/{node_id:08d}.jpg``

Using the node index (zero-padded to 8 digits) as the filename makes
cache lookup O(1) and avoids URL sanitisation.
"""

from __future__ import annotations

import io
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel string for error categories logged in statistics.
_ERR_EMPTY_URL  = "empty_url"
_ERR_HTTP       = "http_error"
_ERR_TIMEOUT    = "timeout"
_ERR_CONNECT    = "connection_error"
_ERR_DECODE     = "decode_error"
_ERR_UNKNOWN    = "unknown_error"
_OK             = "ok"


# ---------------------------------------------------------------------------
# Internal download worker
# ---------------------------------------------------------------------------


def _download_one(
    args: tuple[int, str, Path, int, int],
) -> tuple[int, str]:
    """Download one image and save it as a JPEG.

    Retries up to ``max_retries`` times with a 1-second back-off.

    Args:
        args: ``(node_id, url, dest_path, timeout_s, max_retries)``

    Returns:
        ``(node_id, status)`` where ``status`` is one of the ``_ERR_*``
        / ``_OK`` sentinels.
    """
    node_id, url, dest_path, timeout_s, max_retries = args

    if not url or not url.strip():
        return node_id, _ERR_EMPTY_URL

    # Skip if already cached.
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return node_id, _OK

    # Lazy imports so the module doesn't fail if requests/PIL are absent.
    try:
        import requests
    except ImportError:
        return node_id, _ERR_UNKNOWN

    try:
        from PIL import Image  # noqa: F401 — used below for validation
    except ImportError:
        return node_id, _ERR_UNKNOWN

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url.strip(), timeout=timeout_s, stream=True)
            if resp.status_code != 200:
                return node_id, f"{_ERR_HTTP}_{resp.status_code}"

            raw_bytes = resp.content
            # Validate that the bytes form a decodable image.
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")

            # Save as JPEG.
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest_path, format="JPEG", quality=90)
            return node_id, _OK

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1.0)
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1.0)
        except Exception as exc:  # PIL decode, HTTP, etc.
            last_exc = exc
            break  # non-retryable

    if last_exc is None:
        return node_id, _ERR_UNKNOWN
    exc_str = str(last_exc).lower()
    if "timeout" in exc_str:
        return node_id, _ERR_TIMEOUT
    if "connection" in exc_str:
        return node_id, _ERR_CONNECT
    if "decom" in exc_str or "cannot identify" in exc_str or "image" in exc_str:
        return node_id, _ERR_DECODE
    return node_id, _ERR_UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_and_cache_images(
    data: Any,
    cache_dir: str | Path,
    max_workers: int = 8,
    timeout: int = 15,
    max_retries: int = 2,
) -> tuple[np.ndarray, list[Path | None]]:
    """Download product images and return an availability mask and path list.

    Downloads are parallelised with a thread pool.  Images already present
    in ``cache_dir`` are skipped.

    Args:
        data: Dataset object with a ``meta`` attribute — a list of dicts
            each containing an ``"image_url"`` key (may be empty string).
        cache_dir: Directory where downloaded images are saved.
            Files are named ``{node_id:08d}.jpg``.
        max_workers: Maximum number of concurrent download threads.
        timeout: Per-request timeout in seconds (connect + read combined).
        max_retries: Number of retry attempts per URL before giving up.

    Returns:
        A two-tuple ``(image_available, image_paths)`` where:

        * ``image_available``: ``np.ndarray`` of shape ``(N,)`` dtype
          ``bool`` — ``True`` when a valid JPEG is cached for that node.
        * ``image_paths``: ``list[Path | None]`` of length ``N`` — the
          cached JPEG path for available images, ``None`` otherwise.

    Note:
        Nodes whose images cannot be downloaded are filled with zero
        vectors by downstream vision encoders.  This may introduce a
        **missing-data bias** if missingness correlates with the label;
        the H2 runner quantifies this with a subset-accuracy comparison.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = data.meta
    n_nodes = len(meta)
    urls = [m.get("image_url", "") or "" for m in meta]

    # Build list of (node_id, url, dest_path, timeout, max_retries) tuples.
    work_items = [
        (i, urls[i], cache_dir / f"{i:08d}.jpg", timeout, max_retries)
        for i in range(n_nodes)
    ]

    # Count how many are already cached so we can log the real download count.
    already_cached = sum(
        1 for _, _, path, _, _ in work_items
        if path.exists() and path.stat().st_size > 0
    )
    to_download = n_nodes - already_cached
    logger.info(
        "Image pipeline: %d nodes total  %d already cached  %d to download",
        n_nodes, already_cached, to_download,
    )

    status_map: dict[int, str] = {}

    if to_download == 0:
        # All images already cached; still need to record statuses.
        for node_id, _, path, _, _ in work_items:
            status_map[node_id] = _OK if (path.exists() and path.stat().st_size > 0) else _ERR_EMPTY_URL
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, item): item[0] for item in work_items}
            done = 0
            for fut in as_completed(futures):
                node_id, status = fut.result()
                status_map[node_id] = status
                done += 1
                if done % max(1, n_nodes // 10) == 0:
                    logger.info(
                        "  [%d / %d] downloaded …", done, n_nodes
                    )

    # Build output arrays.
    image_available = np.zeros(n_nodes, dtype=bool)
    image_paths: list[Path | None] = [None] * n_nodes

    for node_id, status in status_map.items():
        path = cache_dir / f"{node_id:08d}.jpg"
        if status == _OK and path.exists() and path.stat().st_size > 0:
            image_available[node_id] = True
            image_paths[node_id] = path
        # else: remains False / None

    # Log statistics.
    counts = Counter(status_map.values())
    success = counts[_OK]
    logger.info(
        "Image pipeline complete: %d / %d succeeded (%.1f%%)",
        success, n_nodes, 100.0 * success / max(1, n_nodes),
    )
    failures: dict[str, int] = {k: v for k, v in counts.items() if k != _OK}
    if failures:
        for err_type, cnt in sorted(failures.items(), key=lambda kv: -kv[1]):
            logger.info("  %s: %d nodes", err_type, cnt)

    return image_available, image_paths


def get_cached_image_paths(
    n_nodes: int,
    cache_dir: str | Path,
) -> tuple[np.ndarray, list[Path | None]]:
    """Return availability mask and paths for an already-populated image cache.

    Useful when images were downloaded in a previous run and you only need
    to rebuild the mask without re-downloading.

    Args:
        n_nodes: Total number of nodes in the dataset.
        cache_dir: Directory where images were cached by
            :func:`fetch_and_cache_images`.

    Returns:
        Same ``(image_available, image_paths)`` format as
        :func:`fetch_and_cache_images`.
    """
    cache_dir = Path(cache_dir)
    image_available = np.zeros(n_nodes, dtype=bool)
    image_paths: list[Path | None] = [None] * n_nodes

    for i in range(n_nodes):
        p = cache_dir / f"{i:08d}.jpg"
        if p.exists() and p.stat().st_size > 0:
            image_available[i] = True
            image_paths[i] = p

    n_avail = int(image_available.sum())
    logger.info(
        "Cache scan: %d / %d images available (%.1f%%)",
        n_avail, n_nodes, 100.0 * n_avail / max(1, n_nodes),
    )
    return image_available, image_paths


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile
    from types import SimpleNamespace

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    # Synthetic dataset with two empty URLs (no real download).
    mock_meta = [
        {"image_url": ""},
        {"image_url": ""},
        {"image_url": ""},
    ]
    mock_data = SimpleNamespace(meta=mock_meta)

    tmp = Path(tempfile.mkdtemp(prefix="img_pipeline_smoke_"))
    try:
        avail, paths = fetch_and_cache_images(mock_data, cache_dir=tmp)
        assert avail.shape == (3,), f"Wrong shape: {avail.shape}"
        assert avail.dtype == bool
        assert not any(avail), "Empty URLs should all fail."
        assert all(p is None for p in paths), "All paths should be None."
        logger.info("Empty-URL test OK.")

        # Test get_cached_image_paths with empty cache.
        avail2, paths2 = get_cached_image_paths(3, tmp)
        assert not any(avail2)
        logger.info("Cache-scan test OK.")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("image_pipeline.py smoke test OK")
