"""H1 experiment runner: GNNs vs feature-only classifiers.

Hypothesis H1: GNNs using co-purchase links beat feature-only classifiers
(same features, no graph).

Pipeline
--------
1. Load dataset from ``data/`` (real or synthetic with ``--mock``).
2. Compute TF-IDF embeddings and assign them to ``data.x``.
3. For each model (GCN, GraphSAGE, GAT, MLP, LogReg):
   a. Run random hyperparameter search; log each trial to
      ``results/hparam_search_tfidf_{model}.jsonl``.
   b. Run seed sweep with best hyperparameters.
   c. Append per-seed result dicts to ``results/h1_results.json``.
4. Print the H1 benchmark table sorted by ``mean_acc`` descending.

Usage
-----
# Real data (requires data/raw/meta_Electronics.json[.gz]):
    python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml

# Quick end-to-end test with synthetic data (no download needed):
    python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml --mock

    # Narrow the model list for a fast smoke run:
    python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml \
        --mock --models MLP LogReg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

# Make sure the project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoders.bow_tfidf import TFIDF
from src.eval.harness import aggregate_metrics, run_seed_sweep
from src.models.baselines import LogRegClassifier, MLPClassifier, build_baseline
from src.models.gnns import GAT, GCN, GraphSAGE, build_gnn
from src.train import TrainConfig, _build_model_for_config, hyperparameter_search, train_model
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Map YAML model names → model classes.
MODEL_REGISTRY: dict[str, Any] = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "MLP": MLPClassifier,
    "LogReg": LogRegClassifier,
}

# GNN = graph model; determines the "H1" label in the benchmark table.
GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT"}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_real_dataset(cfg: dict[str, Any]) -> Any:
    """Load the Amazon co-purchase graph from disk.

    The dataset is expected at ``cfg['root']/raw/meta_<Category>.json[.gz]``.

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

    All parameters are fixed so the test is deterministic.

    Args:
        n_nodes: Number of synthetic product nodes.
        n_classes: Number of simulated product sub-categories.

    Returns:
        A SimpleNamespace with the same attributes used by the training
        loop: ``x``, ``edge_index``, ``y``, ``*_mask``, ``meta``,
        ``class_names``.
    """
    torch.manual_seed(0)
    D_placeholder = 0
    y = torch.randint(0, n_classes, (n_nodes,))

    # Random Erdős–Rényi-style edges.
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
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

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
        x=torch.empty(n_nodes, D_placeholder),
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        meta=meta,
        class_names=class_names,
    )
    # Mimic num_nodes property used in logging.
    data.num_nodes = n_nodes
    logger.info(
        "Mock dataset: N=%d  E=%d  C=%d  train=%d val=%d test=%d",
        n_nodes, edge_index.shape[1] // 2, n_classes,
        int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()),
    )
    return data


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def compute_tfidf_embeddings(data: Any, max_features: int = 10_000) -> torch.Tensor:
    """Fit TF-IDF on train texts, transform all nodes.

    We fit only on training nodes to prevent data leakage: the vocabulary
    and IDF weights are learned from training products alone, then applied
    to val and test products.

    Args:
        data: Dataset with ``meta`` list and ``train_mask`` boolean tensor.
        max_features: Maximum vocabulary size for the TF-IDF vectoriser.

    Returns:
        Dense ``(N, D)`` float32 embedding tensor for all nodes.
    """
    texts = [m["title"] + " " + m["description"] for m in data.meta]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    train_texts = [texts[i] for i in train_idx]

    logger.info(
        "Fitting TF-IDF (max_features=%d) on %d training texts …",
        max_features, len(train_texts),
    )
    encoder = TFIDF(max_features=max_features, min_df=1)
    encoder.fit_transform(train_texts)  # fits; return value discarded

    logger.info("Transforming all %d nodes …", len(texts))
    x = encoder.transform(texts)
    logger.info("Embedding shape: %s  dtype=%s", tuple(x.shape), x.dtype)
    return x


# ---------------------------------------------------------------------------
# Per-model training workflow
# ---------------------------------------------------------------------------


def run_model(
    model_name: str,
    model_cls: Any,
    data: Any,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    base_config: TrainConfig,
    results_dir: Path,
) -> tuple[list[dict], dict]:
    """Run hparam search then seed sweep for one model.

    Logs one JSONL trial record per hparam trial and one JSON result file
    per seed sweep run.

    Args:
        model_name: Display name (``"GCN"``, ``"MLP"``, …).
        model_cls: Model class to instantiate.
        data: Dataset with features already in ``data.x``.
        model_cfg: The ``{name, search_space}`` dict from the YAML.
        training_cfg: The ``training`` sub-dict from the YAML.
        base_config: Template ``TrainConfig`` with dataset-level fields
            already set (``in_channels``, ``out_channels``, ``encoder_name``).
        results_dir: Directory for JSONL logs and per-seed JSON files.

    Returns:
        ``(per_seed_runs, summary)`` as returned by ``run_seed_sweep``.
    """
    search_space: dict[str, list[Any]] = model_cfg["search_space"]
    n_hparam_trials: int = training_cfg["n_hparam_trials"]
    n_seed_runs: int = training_cfg["n_seed_runs"]
    base_seed: int = training_cfg["base_seed"]

    logger.info("=" * 60)
    logger.info("Model: %s | hparam trials: %d | seed runs: %d",
                model_name, n_hparam_trials, n_seed_runs)
    logger.info("=" * 60)

    model_config = replace(base_config, model_name=model_name)

    # Step 1: hyperparameter search.
    logger.info("Step 1: Hyperparameter search …")
    hparam_result = hyperparameter_search(
        model_cls=model_cls,
        data=data,
        search_space=search_space,
        n_trials=n_hparam_trials,
        base_config=model_config,
        results_dir=results_dir,
    )
    best_params = hparam_result["best_params"]
    best_val = hparam_result["best_val_metric"]
    logger.info(
        "Best hparams (val_acc=%.4f): %s", best_val, best_params
    )

    # Build a config with best hparams applied.
    best_config = replace(
        model_config,
        lr=float(best_params.get("lr", model_config.lr)),
        weight_decay=float(best_params.get("weight_decay", model_config.weight_decay)),
        hidden_channels=int(best_params.get("hidden_channels", model_config.hidden_channels)),
        dropout=float(best_params.get("dropout", model_config.dropout)),
        C=float(best_params.get("C", model_config.C)),
    )

    # Step 2: seed sweep with best hparams.
    logger.info("Step 2: Seed sweep (%d runs) …", n_seed_runs)

    def train_fn(seed: int) -> dict[str, Any]:
        """Train one seed; write a JSON result file.

        This closure captures ``model_cls``, ``data``, ``best_config``,
        and ``results_dir`` from the enclosing scope so ``run_seed_sweep``
        only needs to pass the seed.
        """
        run_config = replace(best_config, seed=seed)
        set_seed(seed)
        model = _build_model_for_config(model_cls, run_config)
        result = train_model(model, data, run_config)

        # Write one JSON result file per seed per model.
        result_path = (
            results_dir
            / f"h1_{run_config.encoder_name}_{model_name}_seed{seed}.json"
        )
        with result_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info(
            "Seed %d done: test_acc=%.4f | wrote %s",
            seed, result["test_accuracy"], result_path.name,
        )
        return result

    per_seed, summary = run_seed_sweep(train_fn, n_seeds=n_seed_runs, base_seed=base_seed)
    logger.info(
        "%s sweep complete: mean_acc=%.4f ± %.4f  CI95=[%.4f, %.4f]",
        model_name,
        summary.get("test_accuracy_mean", 0),
        summary.get("test_accuracy_std", 0),
        summary.get("test_accuracy_ci95_lower", 0),
        summary.get("test_accuracy_ci95_upper", 0),
    )
    return per_seed, summary


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------


def print_benchmark_table(rows: list[dict[str, Any]]) -> None:
    """Print the H1 benchmark table to stdout, sorted by mean_acc descending.

    Columns match the format required by the supervisor:
    encoder | modality | model | mean_acc | std_acc | ci95_lower |
    ci95_upper | n_runs | avg_runtime_s | hypothesis

    Args:
        rows: List of summary dicts, one per model.
    """
    rows = sorted(rows, key=lambda r: r.get("mean_acc", 0), reverse=True)
    col_w = [12, 8, 12, 9, 9, 10, 10, 7, 14, 12]
    cols = [
        "encoder", "modality", "model", "mean_acc", "std_acc",
        "ci95_lower", "ci95_upper", "n_runs", "avg_runtime_s", "H1_category",
    ]
    header = "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))
    sep = "  ".join("-" * w for w in col_w)
    print("\n" + "=" * len(sep))
    print("H1 Benchmark: GNN + co-purchase graph vs. feature-only classifiers")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in rows:
        line = "  ".join([
            str(r.get("encoder", "")).ljust(col_w[0]),
            str(r.get("modality", "")).ljust(col_w[1]),
            str(r.get("model", "")).ljust(col_w[2]),
            f"{r.get('mean_acc', 0):.4f}".ljust(col_w[3]),
            f"{r.get('std_acc', 0):.4f}".ljust(col_w[4]),
            f"{r.get('ci95_lower', 0):.4f}".ljust(col_w[5]),
            f"{r.get('ci95_upper', 0):.4f}".ljust(col_w[6]),
            str(r.get("n_runs", "")).ljust(col_w[7]),
            f"{r.get('avg_runtime_s', 0):.1f}".ljust(col_w[8]),
            str(r.get("H1_category", "")).ljust(col_w[9]),
        ])
        print(line)
    print("=" * len(sep) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the H1 experiment.

    Args:
        argv: Override ``sys.argv[1:]`` for programmatic invocation /
            testing.
    """
    parser = argparse.ArgumentParser(
        description="Run the H1 experiment (GNN vs. no-graph baselines)."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the experiment YAML."
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic data instead of the real Amazon dataset."
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Restrict to a subset of models (e.g. --models MLP LogReg)."
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"], help="Logging verbosity."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Load YAML config ---
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as fh:
        exp_cfg: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded experiment config: %s", config_path)

    dataset_cfg = exp_cfg["dataset"]
    encoder_cfg = exp_cfg["encoder"]
    training_cfg = exp_cfg["training"]
    results_cfg = exp_cfg["results"]

    results_dir = Path(results_cfg["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(results_cfg["output_file"])

    # --- Load / build dataset ---
    if args.mock:
        logger.info("Using MOCK dataset (--mock flag).")
        data = make_mock_dataset(n_nodes=200, n_classes=4)
    else:
        logger.info("Loading real dataset …")
        try:
            data = load_real_dataset(dataset_cfg)
        except FileNotFoundError as exc:
            logger.error(
                "Raw dataset not found: %s\n"
                "Download from https://nijianmo.github.io/amazon/ and\n"
                "place files in %s/raw/.  Or run with --mock for synthetic data.",
                exc, dataset_cfg["root"],
            )
            sys.exit(1)

    # --- Compute embeddings ---
    logger.info("Computing %s embeddings …", encoder_cfg["name"].upper())
    x = compute_tfidf_embeddings(data, max_features=encoder_cfg["max_features"])
    data.x = x
    in_channels = x.shape[1]
    out_channels = len(data.class_names)
    logger.info("Embeddings assigned: shape=%s", tuple(x.shape))

    # --- Build base TrainConfig ---
    base_config = TrainConfig(
        in_channels=in_channels,
        hidden_channels=training_cfg.get("hidden_channels", 128),
        out_channels=out_channels,
        num_layers=training_cfg["num_layers"],
        num_heads=training_cfg.get("num_heads", 2),
        max_epochs=training_cfg["max_epochs"],
        patience=training_cfg["patience"],
        warmup_epochs=training_cfg["warmup_epochs"],
        min_lr=training_cfg["min_lr"],
        seed=training_cfg["base_seed"],
        encoder_name=encoder_cfg["name"],
        model_name="",        # set per model in run_model()
        checkpoint_dir=str(results_dir / "checkpoints"),
    )

    # --- Select models ---
    model_list = exp_cfg["models"]
    if args.models:
        model_list = [m for m in model_list if m["name"] in args.models]
        if not model_list:
            logger.error("None of the requested models found in config: %s", args.models)
            sys.exit(1)

    # --- Per-model loop ---
    all_results: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for model_cfg in model_list:
        model_name = model_cfg["name"]
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model '%s', skipping.", model_name)
            continue
        model_cls = MODEL_REGISTRY[model_name]

        per_seed, summary = run_model(
            model_name=model_name,
            model_cls=model_cls,
            data=deepcopy(data),   # isolate data mutations per model
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            base_config=base_config,
            results_dir=results_dir,
        )
        all_results.extend(per_seed)

        # Build one benchmark table row.
        avg_rt = (
            sum(r.get("runtime_seconds", 0) for r in per_seed) / len(per_seed)
            if per_seed else 0.0
        )
        is_graph = model_name in GRAPH_MODELS
        row = {
            "encoder": encoder_cfg["name"],
            "modality": "text",
            "model": model_name,
            "mean_acc": summary.get("test_accuracy_mean", 0.0),
            "std_acc": summary.get("test_accuracy_std", 0.0),
            "ci95_lower": summary.get("test_accuracy_ci95_lower", 0.0),
            "ci95_upper": summary.get("test_accuracy_ci95_upper", 0.0),
            "n_runs": summary.get("n_runs", 0),
            "avg_runtime_s": avg_rt,
            "H1_category": "graph" if is_graph else "no-graph",
        }
        benchmark_rows.append(row)

    # --- Save combined results ---
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("All results saved to %s", output_file)

    # --- Print benchmark table ---
    print_benchmark_table(benchmark_rows)
    logger.info("H1 experiment complete.")


if __name__ == "__main__":
    main()
