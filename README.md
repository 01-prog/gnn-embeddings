# Better Embeddings for Better GNNs

A reproducible research project investigating how the choice of node feature
encoder affects Graph Neural Network (GNN) performance on a product
co-purchase graph from Amazon (McAuley/UCSD dataset). The downstream task is
product category classification.

- Duration: May 4 – June 26, 2026 (8 weeks)
- Supervisor: Clément Wang (CIFRE PhD at IP-Paris / Mirakl)

## Hypotheses

| ID | Statement |
|----|-----------|
| H1 | GNNs using co-purchase links beat feature-only classifiers (same features, no graph). |
| H2 | Multimodal (text + image) embeddings beat single-modality embeddings. |
| H3 | Modern encoders (BERT, Qwen3, CLIP, DINOv2) beat TF-IDF / BoW baselines. |

## Repository layout

```
project/
  data/           # raw + processed data, NOT committed
  src/
    data/         # dataset loading, graph construction, splits
    models/       # GCN, GraphSAGE, GAT, MLP baseline
    encoders/     # text and vision encoder wrappers
    eval/         # evaluation harness (shared by all pairs)
    fusion/       # multimodal fusion modules
    utils/        # logging, seeding, config
  experiments/    # one config YAML per experiment run
  notebooks/      # exploratory analysis only
  tests/          # pytest unit tests
  results/        # JSON result files + benchmark CSVs (NOT committed)
  scripts/        # CLI entry points (e.g. make_benchmark_table.py)
```

## Setup

Requires Python 3.11+. PyTorch Geometric wheels for your platform may need
manual installation; see the [PyG install docs](https://pytorch-geometric.readthedocs.io/).

```bash
make install
```

## Running an experiment

Every experiment is driven by a YAML config under `experiments/`. The runner
loads the config, builds the dataset + encoder + model, runs N seeds, and
writes one JSON result file per seed into `results/`.

```bash
make run-experiment CONFIG=experiments/h1_tfidf_gcn.yaml
```

After a sweep, regenerate the benchmark table:

```bash
python scripts/make_benchmark_table.py --results-dir results/ --out results/benchmark.csv
```

## Experiment design rules

1. **Fair**: every embedding type gets its own hyperparameter search.
2. **Reproducible**: fixed seeds, deterministic splits saved as `.npz`.
3. **Statistically sound**: at least 5 seeds per config, with mean ± std and 95% CI reported.
4. **Logged**: each run writes a JSON file with encoder, model, seed,
   hyperparameters, metrics, and runtime.
5. **Comparable**: the benchmark table is auto-generated from result files.

## Benchmark table columns

```
encoder | modality | model | mean_acc | std_acc | ci95_lower | ci95_upper | n_runs | avg_runtime_s
```

## Tests

```bash
make test
```
