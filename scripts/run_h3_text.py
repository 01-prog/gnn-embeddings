"""H3 experiment runner: text encoder comparison.

Hypothesis H3: Modern encoders (BERT, Qwen3, CLIP) beat TF-IDF / BoW baselines
on a product co-purchase classification task.

Pipeline
--------
1. Load the best GNN from H1 JSON results and use its hyperparams as
   warm-start trial 0 for every H3 hyperparameter search.
2. For each text encoder listed in the YAML grid:
   a. Build the encoder from the registry.
   b. ``fit()`` on training-split texts (no-op for frozen neural encoders;
      learns vocabulary for BoW / TF-IDF).
   c. ``encode_cached()`` all N product texts → updates ``data.x``.
   d. Run a fresh random hyperparameter search (new ``in_channels`` per
      encoder invalidates H1 params for architecture dims, but LR /
      weight_decay / dropout warm-start still helps).
   e. Run a seed sweep with the best hyperparams.
   f. Annotate each result with ``encoder_size_params`` and
      ``pretraining_objective`` for sub-table slicing.
3. Print three H3 sub-tables:
   (a) Baseline axis    — all encoders sorted by mean accuracy.
   (b) Size axis        — Qwen3-0.6B vs 4B vs 8B.
   (c) Pretraining-loss — SBERT (MLM) vs CLIP-text (contrastive)
                          vs Qwen3-0.6B (retrieval).

Usage
-----
# Full run (downloads neural encoders on first pass):
    python scripts/run_h3_text.py \\
        --config experiments/h3_text_encoders.yaml

# Quick smoke test with synthetic data and only sparse encoders:
    python scripts/run_h3_text.py \\
        --config experiments/h3_text_encoders.yaml \\
        --mock --encoders bow tfidf

# Restrict to a specific encoder subset without mock:
    python scripts/run_h3_text.py \\
        --config experiments/h3_text_encoders.yaml \\
        --encoders sbert clip-text qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

# Ensure the project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoders.registry import build_encoder
from src.eval.harness import aggregate_metrics, run_seed_sweep
from src.models.baselines import LogRegClassifier, MLPClassifier
from src.models.gnns import GAT, GCN, GraphSAGE, build_gnn
from src.train import (
    TrainConfig,
    _build_model_for_config,
    hyperparameter_search,
    train_model,
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ── model registry (mirrors run_h1.py) ──────────────────────────────────────
MODEL_REGISTRY: dict[str, Any] = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "MLP": MLPClassifier,
    "LogReg": LogRegClassifier,
}

GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT"}

# Metadata keys that live in the YAML encoder config but must NOT be forwarded
# to ``build_encoder`` (they are not constructor arguments).
_METADATA_KEYS = frozenset(
    {"modality", "tier", "encoder_size_params", "pretraining_objective"}
)


# ---------------------------------------------------------------------------
# Dataset helpers (shared with run_h1.py to avoid duplication)
# ---------------------------------------------------------------------------


def load_real_dataset(cfg: dict[str, Any]) -> Any:
    """Load the Amazon co-purchase graph from disk.

    Args:
        cfg: The ``dataset`` sub-dict from the experiment YAML.

    Returns:
        A PyG ``Data`` object with ``x = torch.empty(N, 0)``.

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

    Deterministic: uses fixed random seeds so results are reproducible.

    Args:
        n_nodes: Number of synthetic product nodes.
        n_classes: Number of simulated product sub-categories.

    Returns:
        A :class:`types.SimpleNamespace` with the same attributes used by
        the training loop: ``x``, ``edge_index``, ``y``, ``*_mask``,
        ``meta``, ``class_names``.
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
    n_val = int(0.2 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]]              = True
    val_mask[perm[n_train: n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]]       = True

    meta = [
        {
            "title": f"Product {i}",
            "description": f"Category {'ABCD'[y[i]]} product number {i}",
            "image_url": "",
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
# H1 best-GNN loader
# ---------------------------------------------------------------------------


def load_h1_best_gnn(
    h1_results_path: str | Path,
    fallback_gnn: str,
    fallback_hparams: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Read H1 results and return the best graph model and its hyperparams.

    The "best" GNN is the one with the highest mean test accuracy across its
    seed runs.  Its hyperparameters (from the first run, which all seeds
    share) are returned as warm-start candidates for the H3 hparam search.

    Args:
        h1_results_path: Path to ``results/h1_results.json``.
        fallback_gnn: Model name used when the file cannot be read.
        fallback_hparams: Hyperparams used when the file cannot be read.

    Returns:
        ``(model_name, hyperparams_dict)`` — e.g.
        ``("GCN", {"lr": 0.01, "weight_decay": 0.0001, ...})``.
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

    # Filter to graph models only (GCN, GraphSAGE, GAT).
    graph_runs = [r for r in results if r.get("model_name") in GRAPH_MODELS]
    if not graph_runs:
        logger.warning("No graph-model runs found in H1 results — using fallback.")
        return fallback_gnn, dict(fallback_hparams)

    # Mean test accuracy per model name.
    acc_by_model: dict[str, list[float]] = defaultdict(list)
    for run in graph_runs:
        acc_by_model[run["model_name"]].append(run.get("test_accuracy", 0.0))

    best_model_name = max(
        acc_by_model,
        key=lambda m: sum(acc_by_model[m]) / len(acc_by_model[m]),
    )
    best_mean = sum(acc_by_model[best_model_name]) / len(acc_by_model[best_model_name])

    # Retrieve hyperparams from the first seed run of the best model.
    best_runs = [r for r in graph_runs if r["model_name"] == best_model_name]
    warm_hparams = dict(best_runs[0].get("hyperparams", fallback_hparams))

    logger.info(
        "H1 best GNN: '%s' (mean test_acc=%.4f)  warm-start hparams: %s",
        best_model_name, best_mean, warm_hparams,
    )
    return best_model_name, warm_hparams


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------


def compute_embeddings(
    encoder_cfg: dict[str, Any],
    data: Any,
    embeddings_dir: str | Path,
    device: str = "auto",
) -> torch.Tensor:
    """Build the encoder, fit on training texts, and encode all nodes.

    Statistical encoders (BoW, TF-IDF) are fitted on training-split texts
    only; the resulting vocabulary / IDF table is then applied to the full
    node set.  Frozen neural encoders (SBERT, CLIP, Qwen3) skip the fit
    step.  All outputs are cached to ``{embeddings_dir}/{cache_key}.pt``
    so repeated runs skip the (costly) forward pass.

    Args:
        encoder_cfg: Encoder dict from the YAML ``encoders`` list.  The
            ``name``, ``type``, and constructor kwargs are read here;
            metadata fields are ignored.
        data: Dataset with a ``meta`` list and ``train_mask`` boolean tensor.
        embeddings_dir: Root directory for ``.pt`` cache files.
        device: Inference device forwarded to :func:`build_encoder`.

    Returns:
        Dense ``(N, D)`` float32 tensor for all nodes.
    """
    enc_name: str = encoder_cfg["name"]
    texts = [m["title"] + " " + m["description"] for m in data.meta]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    train_texts = [texts[i] for i in train_idx]

    logger.info(
        "Building encoder '%s' (type=%s) …", enc_name, encoder_cfg.get("type")
    )
    # Strip metadata keys before forwarding to the registry.
    enc_build_cfg = {k: v for k, v in encoder_cfg.items() if k not in _METADATA_KEYS}
    encoder = build_encoder(enc_build_cfg, cache_dir=embeddings_dir, device=device)

    # Fit (no-op for frozen encoders; learns vocab for BoW / TF-IDF).
    logger.info("Fitting encoder '%s' on %d training texts …", enc_name, len(train_texts))
    encoder.fit(train_texts)

    # Encode with caching.  Use split_seed from the dataset config if available.
    cache_key = f"{enc_name}_split0"
    logger.info("Encoding %d texts (cache_key='%s') …", len(texts), cache_key)
    x = encoder.encode_cached(texts, cache_key)
    logger.info("Embeddings '%s': shape=%s  dtype=%s", enc_name, tuple(x.shape), x.dtype)
    return x


# ---------------------------------------------------------------------------
# Per-encoder training workflow
# ---------------------------------------------------------------------------


def run_encoder(
    encoder_cfg: dict[str, Any],
    model_name: str,
    model_cls: Any,
    data: Any,
    training_cfg: dict[str, Any],
    search_space: dict[str, list[Any]],
    base_config: TrainConfig,
    warm_start_params: dict[str, Any] | None,
    results_dir: Path,
    embeddings_dir: Path,
    device: str = "auto",
) -> tuple[list[dict], dict]:
    """Encode → hparam search → seed sweep for one (encoder, model) pair.

    Args:
        encoder_cfg: YAML dict for this encoder (includes metadata).
        model_name: Model class name (e.g. ``"GCN"``).
        model_cls: Model class to instantiate.
        data: Dataset object.  ``data.x`` is replaced in place.
        training_cfg: The ``training`` sub-dict from the YAML.
        search_space: Hyperparameter search grid (overridden per encoder to
            handle the new ``in_channels``).
        base_config: Template :class:`~src.train.TrainConfig` with fields
            shared across all encoders (``out_channels``, schedule, etc.).
        warm_start_params: Optional H1 best-GNN hyperparams.  Passed as
            trial 0 to the hyperparameter search.
        results_dir: Directory for JSONL logs and per-seed JSON files.
        embeddings_dir: Directory for cached embedding tensors.
        device: Inference device.

    Returns:
        ``(per_seed_runs, summary)`` from :func:`~src.eval.harness.run_seed_sweep`.
    """
    enc_name: str = encoder_cfg["name"]
    n_hparam_trials: int = training_cfg["n_hparam_trials"]
    n_seed_runs: int = training_cfg["n_seed_runs"]
    base_seed: int = training_cfg["base_seed"]

    logger.info("=" * 70)
    logger.info("Encoder: %-15s  Model: %s", enc_name, model_name)
    logger.info("=" * 70)

    # --- compute / load embeddings -------------------------------------------
    x = compute_embeddings(encoder_cfg, data, embeddings_dir=embeddings_dir, device=device)
    data.x = x
    in_channels = x.shape[1]

    # Build a config with the correct in_channels for this encoder.
    model_config = replace(
        base_config,
        encoder_name=enc_name,
        model_name=model_name,
        in_channels=in_channels,
    )

    # --- hyperparameter search -----------------------------------------------
    logger.info("Hyperparameter search: %d trials (warm-start=%s) …",
                n_hparam_trials, warm_start_params is not None)
    hparam_result = hyperparameter_search(
        model_cls=model_cls,
        data=data,
        search_space=search_space,
        n_trials=n_hparam_trials,
        base_config=model_config,
        results_dir=results_dir,
        warm_start_params=warm_start_params,
    )
    best_params = hparam_result["best_params"]
    best_val = hparam_result["best_val_metric"]
    logger.info("Best hparams (val_acc=%.4f): %s", best_val, best_params)

    best_config = replace(
        model_config,
        lr=float(best_params.get("lr", model_config.lr)),
        weight_decay=float(best_params.get("weight_decay", model_config.weight_decay)),
        hidden_channels=int(best_params.get("hidden_channels", model_config.hidden_channels)),
        dropout=float(best_params.get("dropout", model_config.dropout)),
        C=float(best_params.get("C", model_config.C)),
    )

    # --- seed sweep ----------------------------------------------------------
    logger.info("Seed sweep: %d runs …", n_seed_runs)

    # Metadata to inject into every result dict.
    meta_fields = {
        "pretraining_objective": encoder_cfg.get("pretraining_objective", "unknown"),
        "encoder_size_params":   encoder_cfg.get("encoder_size_params"),
        "modality":              encoder_cfg.get("modality", "text"),
        "tier":                  encoder_cfg.get("tier"),
    }

    def train_fn(seed: int) -> dict[str, Any]:
        """Train one seed and write a JSON result file.

        Closure captures ``model_cls``, ``data``, ``best_config``,
        ``results_dir``, and ``meta_fields``.
        """
        run_config = replace(best_config, seed=seed)
        set_seed(seed)
        model = _build_model_for_config(model_cls, run_config)
        result = train_model(model, data, run_config)
        # Attach H3-specific metadata.
        result.update(meta_fields)

        out_path = (
            results_dir / f"h3_{enc_name}_{model_name}_seed{seed}.json"
        )
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info(
            "Seed %d → test_acc=%.4f | %s",
            seed, result["test_accuracy"], out_path.name,
        )
        return result

    per_seed, summary = run_seed_sweep(train_fn, n_seeds=n_seed_runs, base_seed=base_seed)
    logger.info(
        "%s/%s: mean_acc=%.4f ± %.4f  CI95=[%.4f, %.4f]",
        enc_name, model_name,
        summary.get("test_accuracy_mean", 0),
        summary.get("test_accuracy_std", 0),
        summary.get("test_accuracy_ci95_lower", 0),
        summary.get("test_accuracy_ci95_upper", 0),
    )
    return per_seed, summary


# ---------------------------------------------------------------------------
# Sub-table printing
# ---------------------------------------------------------------------------

# Column headers and widths for the benchmark tables.
_COLS = [
    ("encoder",             16),
    ("pretraining",         15),
    ("tier",                 5),
    ("model",               12),
    ("mean_acc",             9),
    ("std_acc",              9),
    ("ci95_lower",          10),
    ("ci95_upper",          10),
    ("n_runs",               7),
    ("avg_runtime_s",       14),
    ("encoder_params",      14),
]


def _fmt_params(n: int | None) -> str:
    """Human-readable parameter count (e.g. ``600M``, ``22M``, ``N/A``).

    Args:
        n: Raw integer parameter count, or ``None``.

    Returns:
        Formatted string.
    """
    if n is None:
        return "N/A"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    return str(n)


def print_sub_table(
    rows: list[dict[str, Any]],
    title: str,
    hypothesis_label: str,
) -> None:
    """Print a single H3 sub-table, sorted by ``mean_acc`` descending.

    Args:
        rows: List of benchmark row dicts (one per (encoder, model) pair).
        title: Human-readable sub-table title.
        hypothesis_label: Short label shown in the header (e.g. ``"H3-a"``).
    """
    rows = sorted(rows, key=lambda r: r.get("mean_acc", 0.0), reverse=True)
    header_parts = [c.ljust(w) for c, w in _COLS]
    header = "  ".join(header_parts)
    sep    = "  ".join("-" * w for _, w in _COLS)
    print(f"\n{'=' * len(sep)}")
    print(f"{hypothesis_label}: {title}")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in rows:
        params_str = _fmt_params(r.get("encoder_size_params"))
        line = "  ".join([
            str(r.get("encoder", "")).ljust(_COLS[0][1]),
            str(r.get("pretraining_objective", "")).ljust(_COLS[1][1]),
            str(r.get("tier", "")).ljust(_COLS[2][1]),
            str(r.get("model", "")).ljust(_COLS[3][1]),
            f"{r.get('mean_acc', 0.0):.4f}".ljust(_COLS[4][1]),
            f"{r.get('std_acc', 0.0):.4f}".ljust(_COLS[5][1]),
            f"{r.get('ci95_lower', 0.0):.4f}".ljust(_COLS[6][1]),
            f"{r.get('ci95_upper', 0.0):.4f}".ljust(_COLS[7][1]),
            str(r.get("n_runs", "")).ljust(_COLS[8][1]),
            f"{r.get('avg_runtime_s', 0.0):.1f}".ljust(_COLS[9][1]),
            params_str.ljust(_COLS[10][1]),
        ])
        print(line)
    print("=" * len(sep) + "\n")


def print_all_tables(benchmark_rows: list[dict[str, Any]]) -> None:
    """Print the three H3 sub-tables to stdout.

    Sub-table (a): Baseline axis — all encoders compared together.
    Sub-table (b): Size axis     — Qwen3 scaling (0.6B / 4B / 8B).
    Sub-table (c): Pretraining-loss axis — MLM vs contrastive vs retrieval.

    Args:
        benchmark_rows: Collected benchmark rows for every (encoder, model).
    """
    # (a) All encoders — baseline axis.
    print_sub_table(
        benchmark_rows,
        "Baseline axis: BoW → TF-IDF → SBERT → CLIP-text → Qwen3",
        "H3-a",
    )

    # (b) Size axis: only Qwen3 rows.
    qwen3_rows = [
        r for r in benchmark_rows
        if r.get("pretraining_objective") == "retrieval"
    ]
    if qwen3_rows:
        print_sub_table(
            qwen3_rows,
            "Size axis: Qwen3-0.6B vs 4B vs 8B (retrieval pretraining)",
            "H3-b",
        )
    else:
        logger.info("Sub-table H3-b skipped (no Qwen3 rows found).")

    # (c) Pretraining-loss axis: one representative per objective.
    # Keep only the "modern" rows (tier >= 1) and deduplicate by objective,
    # keeping the highest-accuracy row per objective.
    obj_rows: dict[str, dict] = {}
    modern = [r for r in benchmark_rows if (r.get("tier") or 0) >= 1]
    for r in sorted(modern, key=lambda x: x.get("mean_acc", 0.0), reverse=True):
        obj = r.get("pretraining_objective", "unknown")
        if obj not in obj_rows:
            obj_rows[obj] = r
    if obj_rows:
        print_sub_table(
            list(obj_rows.values()),
            "Pretraining-loss axis: MLM (SBERT) vs contrastive (CLIP) vs retrieval (Qwen3)",
            "H3-c",
        )
    else:
        logger.info("Sub-table H3-c skipped (no modern encoder rows found).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the H3 experiment.

    Args:
        argv: Override ``sys.argv[1:]`` for programmatic invocation or
            testing.
    """
    parser = argparse.ArgumentParser(
        description="Run the H3 experiment (text encoder comparison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML (e.g. experiments/h3_text_encoders.yaml).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a small synthetic dataset instead of the real Amazon data.",
    )
    parser.add_argument(
        "--encoders", nargs="+", metavar="ENC",
        help=(
            "Restrict to a subset of encoders by name "
            "(e.g. --encoders bow tfidf sbert).  "
            "In --mock mode defaults to [bow, tfidf] if not specified."
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

    # --- Load YAML config ---------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as fh:
        exp_cfg: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded config: %s", config_path)

    dataset_cfg  = exp_cfg["dataset"]
    training_cfg = exp_cfg["training"]
    results_cfg  = exp_cfg["results"]
    search_space: dict[str, list[Any]] = exp_cfg["gnn_search_space"]

    results_dir    = Path(results_cfg["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file    = Path(results_cfg["output_file"])
    embeddings_dir = Path(results_cfg.get("embeddings_dir", "data/embeddings"))
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # --- Load / build dataset -----------------------------------------------
    if args.mock:
        logger.info("Using MOCK dataset (--mock flag).")
        data = make_mock_dataset(n_nodes=200, n_classes=4)
    else:
        try:
            data = load_real_dataset(dataset_cfg)
        except FileNotFoundError as exc:
            logger.error(
                "Raw dataset not found: %s\n"
                "Download from https://nijianmo.github.io/amazon/ and "
                "place files in %s/raw/. Or run with --mock.",
                exc, dataset_cfg["root"],
            )
            sys.exit(1)

    out_channels = len(data.class_names)

    # --- Determine which encoders to run ------------------------------------
    all_encoder_cfgs: list[dict[str, Any]] = exp_cfg["encoders"]

    if args.encoders:
        requested = set(args.encoders)
        encoder_cfgs = [e for e in all_encoder_cfgs if e["name"] in requested]
        if not encoder_cfgs:
            logger.error(
                "None of the requested encoders %s found in config.", args.encoders
            )
            sys.exit(1)
        missing = requested - {e["name"] for e in encoder_cfgs}
        if missing:
            logger.warning("Encoders not found in config (skipped): %s", missing)
    elif args.mock:
        # Default to sparse-only in mock mode to avoid model downloads.
        encoder_cfgs = [e for e in all_encoder_cfgs if e["type"] in ("BagOfWords", "TFIDF")]
        logger.info(
            "--mock mode: restricting to sparse encoders %s "
            "(use --encoders to override).",
            [e["name"] for e in encoder_cfgs],
        )
    else:
        encoder_cfgs = all_encoder_cfgs

    # --- Find H1 best GNN + warm-start params --------------------------------
    best_gnn_name, warm_start_params = load_h1_best_gnn(
        h1_results_path=exp_cfg.get("h1_results_path", "results/h1_results.json"),
        fallback_gnn=exp_cfg.get("fallback_gnn", "GCN"),
        fallback_hparams=exp_cfg.get("fallback_hparams", {}),
    )
    # Command-line --model overrides the H1-derived choice.
    if args.model is not None:
        logger.info("--model flag overrides H1 best GNN: '%s' → '%s'.", best_gnn_name, args.model)
        best_gnn_name = args.model
    logger.info("Using GNN model: %s  warm-start: %s", best_gnn_name, warm_start_params)

    if best_gnn_name not in MODEL_REGISTRY:
        logger.error("Best GNN '%s' is not in MODEL_REGISTRY.", best_gnn_name)
        sys.exit(1)
    model_cls = MODEL_REGISTRY[best_gnn_name]

    # --- Template TrainConfig (in_channels filled per encoder) ---------------
    base_config = TrainConfig(
        in_channels=0,             # overridden per encoder
        hidden_channels=training_cfg.get("hidden_channels", 128),
        out_channels=out_channels,
        num_layers=training_cfg["num_layers"],
        num_heads=training_cfg.get("num_heads", 2),
        max_epochs=training_cfg["max_epochs"],
        patience=training_cfg["patience"],
        warmup_epochs=training_cfg["warmup_epochs"],
        min_lr=training_cfg["min_lr"],
        seed=training_cfg["base_seed"],
        encoder_name="",           # overridden per encoder
        model_name=best_gnn_name,
        checkpoint_dir=str(results_dir / "checkpoints"),
        device=args.device,
    )

    # --- Per-encoder loop ---------------------------------------------------
    all_results: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for enc_cfg in encoder_cfgs:
        enc_name = enc_cfg["name"]
        logger.info("─" * 70)
        logger.info("Starting encoder: %s", enc_name)

        per_seed, summary = run_encoder(
            encoder_cfg=enc_cfg,
            model_name=best_gnn_name,
            model_cls=model_cls,
            data=deepcopy(data),      # isolate per-encoder data mutations
            training_cfg=training_cfg,
            search_space=search_space,
            base_config=base_config,
            warm_start_params=warm_start_params,
            results_dir=results_dir,
            embeddings_dir=embeddings_dir,
            device=args.device,
        )
        all_results.extend(per_seed)

        avg_rt = (
            sum(r.get("runtime_seconds", 0) for r in per_seed) / len(per_seed)
            if per_seed else 0.0
        )
        row = {
            "encoder":               enc_name,
            "pretraining_objective": enc_cfg.get("pretraining_objective", "unknown"),
            "tier":                  enc_cfg.get("tier"),
            "encoder_size_params":   enc_cfg.get("encoder_size_params"),
            "modality":              enc_cfg.get("modality", "text"),
            "model":                 best_gnn_name,
            "mean_acc":              summary.get("test_accuracy_mean", 0.0),
            "std_acc":               summary.get("test_accuracy_std", 0.0),
            "ci95_lower":            summary.get("test_accuracy_ci95_lower", 0.0),
            "ci95_upper":            summary.get("test_accuracy_ci95_upper", 0.0),
            "n_runs":                summary.get("n_runs", 0),
            "avg_runtime_s":         avg_rt,
        }
        benchmark_rows.append(row)
        logger.info(
            "Encoder '%s' done: mean_acc=%.4f ± %.4f",
            enc_name,
            row["mean_acc"],
            row["std_acc"],
        )

    # --- Save combined results ----------------------------------------------
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("All %d per-seed results saved to %s", len(all_results), output_file)

    # --- Print benchmark sub-tables -----------------------------------------
    print_all_tables(benchmark_rows)
    logger.info("H3 experiment complete.")


if __name__ == "__main__":
    main()
