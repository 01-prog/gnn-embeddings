# Better Embeddings for Better GNNs — Project Guide

> A reproducible academic study of how node-feature encoders affect GNN performance on an Amazon product co-purchase graph.
>
> Supervised by Clément Wang (CIFRE PhD, IP-Paris / Mirakl).
> Timeline: May 4 – June 26, 2026 (8 weeks).

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The Three Hypotheses](#2-the-three-hypotheses)
3. [Repository Layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Data Preparation](#5-data-preparation)
6. [Running Experiments](#6-running-experiments)
7. [Module Reference](#7-module-reference)
   - [Data layer](#71-data-layer)
   - [Encoders](#72-encoders)
   - [Models](#73-models)
   - [Fusion](#74-fusion)
   - [Training loop](#75-training-loop)
   - [Evaluation harness](#76-evaluation-harness)
   - [Utilities](#77-utilities)
8. [Configuration System (YAML)](#8-configuration-system-yaml)
9. [Results Format](#9-results-format)
10. [Testing](#10-testing)
11. [How to Extend the Project](#11-how-to-extend-the-project)
12. [Reproducibility Contract](#12-reproducibility-contract)
13. [Common Issues](#13-common-issues)

---

## 1. What This Project Does

We train Graph Neural Networks (GNNs) on an Amazon product co-purchase graph where each node is a product and each edge links two products that customers frequently buy together. The task is **node-level multi-class classification** — predicting the product sub-category (e.g. "Headphones", "Cables", "Cameras") from the node's feature vector and its neighborhood.

The key research question is: **does the choice of encoder for the initial node features change how well a GNN performs?**

To answer it fairly we:
- Hold everything else constant (same graph, same splits, same evaluation protocol).
- Give each encoder its own hyperparameter search so no encoder is penalised by sub-optimal settings.
- Run each configuration 5 times with different random seeds and report mean ± std ± 95% CI.

---

## 2. The Three Hypotheses

| ID | Hypothesis | Runner |
|----|-----------|--------|
| **H1** | GNNs (which use the co-purchase edges) beat feature-only classifiers (MLP, LogReg) when given the same node features. | `scripts/run_h1.py` |
| **H2** | Multimodal (text + image) embeddings beat single-modality embeddings on the same GNN. | `scripts/run_h2.py` |
| **H3** | Modern encoders (BERT, Qwen3, CLIP) beat classical TF-IDF / BoW baselines. | `scripts/run_h3_text.py` |

Each hypothesis has its own YAML config file under `experiments/` and a corresponding experiment runner script under `scripts/`.

---

## 3. Repository Layout

```
project/
├── data/                    # Raw + processed data (never committed to git)
│   ├── raw/                 # meta_Electronics.json[.gz] goes here
│   ├── embeddings/          # .pt caches (auto-created)
│   ├── images/              # Downloaded product images (auto-created)
│   └── splits/              # split_0.npz deterministic index arrays (auto-created)
│
├── experiments/             # One YAML config per hypothesis
│   ├── h1_graph_vs_nograph.yaml
│   ├── h2_multimodal.yaml
│   └── h3_text_encoders.yaml
│
├── results/                 # JSON result files + JSONL hparam search logs
│   └── checkpoints/         # Best model .pt checkpoints per run
│
├── scripts/                 # Experiment entry points (one per hypothesis)
│   ├── run_h1.py
│   ├── run_h2.py
│   └── run_h3_text.py
│
├── src/                     # All reusable library code
│   ├── data/
│   │   ├── amazon_dataset.py   # Dataset loading + graph construction
│   │   └── image_pipeline.py   # Concurrent image downloader
│   ├── encoders/
│   │   ├── base.py             # BaseEncoder ABC
│   │   ├── bow_tfidf.py        # Bag-of-Words and TF-IDF (Tier 0)
│   │   ├── sbert.py            # Sentence-BERT (Tier 1)
│   │   ├── clip_text.py        # CLIP text encoder (Tier 1, contrastive)
│   │   ├── qwen3.py            # Qwen3-Embedding 0.6B/4B/8B (Tier 2-3)
│   │   ├── vision.py           # ResNet-50, CLIP vision, DINOv2
│   │   └── registry.py         # build_encoder() factory function
│   ├── models/
│   │   ├── gnns.py             # GCN, GraphSAGE, GAT
│   │   └── baselines.py        # MLP, LogisticRegression
│   ├── fusion/
│   │   └── fusion.py           # ConcatFusion, WeightedFusion, GatedFusion,
│   │                           # FusedModel, FusedModelFactory
│   ├── eval/
│   │   └── harness.py          # evaluate(), run_seed_sweep(), aggregate_metrics()
│   ├── train.py                # TrainConfig, train_model(), hyperparameter_search()
│   └── utils/
│       ├── device.py           # resolve_device("auto" → torch.device)
│       └── seed.py             # set_seed() — seeds random/numpy/torch/cuda
│
├── tests/                   # pytest test suite
│   ├── conftest.py
│   ├── test_encoders.py
│   ├── test_fusion.py
│   └── test_harness.py
│
├── notebooks/               # Exploratory analysis only (not production code)
├── GUIDE.md                 # This file
├── Makefile
└── requirements.txt
```

---

## 4. Setup

### Prerequisites

- Python 3.11 or 3.12 (3.13 also works)
- A CUDA-capable GPU is recommended for the neural encoders (Qwen3, CLIP, DINOv2) but not required — all code falls back to CPU automatically.

### Install

```bash
# Clone the repository, then:
pip install -r requirements.txt
```

The key packages are:

| Package | Role |
|---------|------|
| `torch >= 2.2` | Core tensor framework |
| `torch-geometric >= 2.5` | GCN / GraphSAGE / GAT convolution layers |
| `transformers >= 4.40` | SBERT, CLIP, Qwen3, DINOv2 (via HuggingFace) |
| `sentence-transformers >= 2.7` | Sentence-BERT encoder |
| `scikit-learn >= 1.4` | TF-IDF, BoW, Logistic Regression |
| `torchvision` | ResNet-50 (vision baseline) |
| `Pillow` | Image loading for vision encoders |
| `PyYAML` | YAML config loading |
| `numpy`, `pandas` | Numerical + data utilities |

> **Note on `torch-geometric`:** Installing it via pip sometimes requires matching the CUDA version. If the standard install fails, follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Verify the installation

```bash
# Run the full test suite (should show ~89 passed, 12 skipped):
python -m pytest tests/ -q

# Quick smoke test with synthetic data (no downloads, ~30 seconds):
python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml --mock --models MLP LogReg
```

---

## 5. Data Preparation

### The Amazon dataset

The project uses the **McAuley/UCSD Amazon product review dataset** (2018 or 2023 format). Download the metadata file for your target category (e.g. "Electronics") and place it in `data/raw/`:

```
data/raw/meta_Electronics.json      # or .json.gz
```

Dataset home pages:
- 2018: https://nijianmo.github.io/amazon/index.html
- 2023: https://amazon-reviews-2023.github.io/

The code handles both formats automatically (it normalises 2018 vs. 2023 schema differences internally).

### What `AmazonCopurchase` does on first load

`src/data/amazon_dataset.py` runs this pipeline **once** on first use and caches the result:

1. Parses all JSONL records from `data/raw/meta_<Category>.json[.gz]`.
2. Filters out products without a usable category label or title.
3. Builds an undirected `edge_index` from `also_buy` / `also_bought` links (only edges where both endpoints survived filtering).
4. Creates a stratified train/val/test split (default 60/20/20) keyed by `split_seed` and saves it to `data/splits/split_{seed}.npz` — **the same split is reused by every encoder and model**, which is the foundation of fair comparison.
5. Saves the `torch_geometric.data.Data` object to `data/processed/`.

The `Data` object produced has:
- `data.x = torch.empty(N, 0)` — a placeholder; encoders fill this in later.
- `data.edge_index` — co-purchase edges, shape `(2, E)`.
- `data.y` — integer class labels, shape `(N,)`.
- `data.train_mask`, `data.val_mask`, `data.test_mask` — boolean node masks.
- `data.meta` — list of N dicts, each with `"title"`, `"description"`, `"image_url"`.
- `data.class_names` — list of human-readable category strings.

### Embedding cache

When an encoder runs for the first time, it computes embeddings for all N nodes and saves them to `data/embeddings/{cache_key}.pt`. On repeat runs the `.pt` file is loaded directly, completely skipping the (potentially expensive) model forward pass.

Cache keys follow the convention `"{encoder_name}_split{seed}"`, e.g. `tfidf_split0`, `sbert_split0`, `dinov2-B_vision_split0`.

### Image cache (H2 only)

The `image_pipeline` module downloads product images in parallel to `data/images/{node_id:08d}.jpg`. Already-downloaded images are skipped. If an image cannot be downloaded (404, timeout, corrupt), the node's image embedding becomes a **zero vector** — this is documented and tracked in the H2 confounder analysis.

---

## 6. Running Experiments

### H1 — GNNs vs feature-only baselines

```bash
# Full run (real data, all 5 models):
python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml

# Smoke test with synthetic data (fast, no download needed):
python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml --mock

# Restrict to specific models:
python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml --models GCN MLP

# Force CPU (useful when no GPU is available):
python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml --device cpu
```

**What it does:** Runs TF-IDF encoding once, then for each model in `{GCN, GraphSAGE, GAT, MLP, LogReg}`:
1. Random hyperparameter search over the search space in the YAML (20 trials by default).
2. Seed sweep: 5 runs with seeds 0–4, using the best hyperparams from step 1.
3. Writes per-seed JSON to `results/h1_{model}_seed{N}.json`.
4. Writes JSONL search trace to `results/hparam_search_tfidf_{model}.jsonl`.
5. Prints the H1 benchmark table at the end.

**Output file:** `results/h1_results.json` — a JSON array of all per-seed result dicts.

---

### H3 — Text encoder comparison

**Run H1 first** (H3 uses the H1 best GNN as a warm-start).

```bash
# Full run (downloads neural encoders on first pass):
python scripts/run_h3_text.py --config experiments/h3_text_encoders.yaml

# Smoke test: only sparse encoders, no model downloads:
python scripts/run_h3_text.py --config experiments/h3_text_encoders.yaml \
    --mock --encoders bow tfidf

# Override the GNN (useful to test without torch_geometric):
python scripts/run_h3_text.py --config experiments/h3_text_encoders.yaml \
    --mock --encoders bow tfidf --model MLP
```

**What it does:** For each encoder in the YAML grid (BoW, TF-IDF, SBERT, CLIP-text, Qwen3-0.6B, 4B, 8B):
1. Builds and fits the encoder on training texts only.
2. Encodes all N nodes (with caching).
3. Runs a fresh hyperparameter search (the H1 best-GNN hyperparams are used as warm-start trial 0).
4. Runs a 5-seed sweep.
5. Prints three sub-tables at the end:
   - **H3-a**: All encoders sorted by mean accuracy.
   - **H3-b**: Qwen3 size-scaling axis (0.6B → 4B → 8B).
   - **H3-c**: Pretraining-objective axis (MLM vs contrastive vs retrieval).

**Output file:** `results/h3_results.json`.

---

### H2 — Multimodal fusion

**Run H1 and H3 first** (H2 uses both for warm-starts and to select the best text encoder).

```bash
# Full run:
python scripts/run_h2.py --config experiments/h2_multimodal.yaml

# Smoke test (no downloads, uses all-zero image vectors):
python scripts/run_h2.py --config experiments/h2_multimodal.yaml \
    --mock --modalities text_only image_only --model MLP

# Skip re-downloading images (reuse previous cache):
python scripts/run_h2.py --config experiments/h2_multimodal.yaml --skip-image-download

# Test only one fusion strategy:
python scripts/run_h2.py --config experiments/h2_multimodal.yaml \
    --modalities "text+image" --fusion concat
```

**What it does:**
1. Determines the best text encoder from H3 results (or fallback).
2. Determines the best GNN from H1 results (or fallback).
3. Downloads + caches product images (skips already-cached).
4. For **text_only**: Computes text embeddings → trains the GNN.
5. For **image_only** (each vision encoder): Encodes image paths → trains the GNN. Missing images → zero vectors.
6. For **text+image** (each vision encoder × each fusion strategy): Concatenates `[text_emb, image_emb]` as `data.x`; the `FusedModel` wrapper splits them back and applies the chosen fusion before the GNN backbone.
7. Per-seed evaluates the trained model on "test nodes with images" vs "test nodes without images" and prints a confounder analysis table. Combos with a >5 pp accuracy gap are flagged as potential confounders.

**Output file:** `results/h2_results.json`.

---

## 7. Module Reference

### 7.1 Data Layer

#### `src/data/amazon_dataset.py` — `AmazonCopurchase`

A `torch_geometric.data.InMemoryDataset` subclass. The constructor signature is:

```python
AmazonCopurchase(
    root="data",
    categories=["Electronics"],
    split_seed=0,
    split_ratios=(0.6, 0.2, 0.2),
    label_level=1,       # 1 = second-level subcategories
)
```

Use it as:

```python
ds = AmazonCopurchase(root="data", categories=["Electronics"])
data = ds[0]   # The single Data object for the full graph
```

On first use it downloads nothing (the metadata files must already be in `data/raw/`), processes them, and caches the result. Subsequent `AmazonCopurchase(...)` calls with the same args return the cached result instantly.

#### `src/data/image_pipeline.py` — image downloading

```python
from src.data.image_pipeline import fetch_and_cache_images, get_cached_image_paths

# Download images (skips already-cached):
image_available, image_paths = fetch_and_cache_images(
    data=data,           # needs data.meta[i]["image_url"]
    cache_dir="data/images",
    max_workers=8,
    timeout=15,
    max_retries=2,
)
# image_available: bool np.ndarray (N,)
# image_paths:     list[Path | None]  (None for failed downloads)

# Scan an already-populated cache (no HTTP requests):
image_available, image_paths = get_cached_image_paths(n_nodes=N, cache_dir="data/images")
```

Missing images produce `None` entries in `image_paths`. Vision encoders fill those positions with **zero vectors** of the correct embedding dimension.

---

### 7.2 Encoders

All encoders share a common interface defined in `src/encoders/base.py`:

```python
class BaseEncoder(ABC):
    def fit(self, train_inputs: list[str]) -> None: ...         # no-op for frozen models
    def encode(self, inputs: list[str | Path]) -> Tensor: ...  # (N, D) float32
    def encode_cached(self, inputs, cache_key: str) -> Tensor: ...
    @property
    def embedding_dim(self) -> int: ...
```

The **critical convention**: always call `fit(train_texts)` before `encode(all_texts)` for statistical encoders (BoW, TF-IDF). Frozen neural encoders (SBERT, CLIP, Qwen3, vision encoders) ignore `fit()` automatically.

#### Building encoders

Use the registry instead of constructing encoders directly:

```python
from src.encoders.registry import build_encoder

encoder = build_encoder(
    config={"type": "SentenceBERT", "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    cache_dir="data/embeddings",
    device="auto",
)
```

Supported `type` values:

| Type | Class | Dim | Notes |
|------|-------|-----|-------|
| `BagOfWords` | `BagOfWordsEncoder` | ≤10 000 | Sparse → dense; must call `fit()` |
| `TFIDF` | `TFIDFEncoder` | ≤10 000 | L2-normalised rows; must call `fit()` |
| `SentenceBERT` | `SentenceBERTEncoder` | 384 | all-MiniLM-L6-v2 default |
| `CLIPText` | `CLIPTextEncoder` | 512 | Same CLIP space as `CLIPVision` |
| `Qwen3` | `Qwen3Encoder` | 1024–7168 | 0.6B / 4B / 8B variants |
| `ResNet` | `ResNetEncoder` | 2048 | ResNet-50 avg-pool, supervised |
| `CLIPVision` | `CLIPVisionEncoder` | 512 | ViT-B/32, contrastive |
| `DINOv2` | `DINOv2Encoder` | 768 / 1024 | ViT-B/14 or ViT-L/14, self-supervised |

**Vision encoders** take `list[Path | None]` instead of `list[str]`:

```python
vision_enc = build_encoder({"type": "DINOv2", "model_size": "B"}, cache_dir="data/embeddings")
x = vision_enc.encode_cached(image_paths, cache_key="dinov2-B_vision_split0")
# x.shape == (N, 768), with zeros for None entries
```

---

### 7.3 Models

All models satisfy the same interface:

```python
logits = model(x, edge_index)   # (N, out_channels)
```

This uniformity means the training loop in `src/train.py` works for every model without branching.

#### GNNs — `src/models/gnns.py`

| Class | Paper | Message passing |
|-------|-------|----------------|
| `GCN` | Kipf & Welling 2017 | Symmetric normalised adjacency |
| `GraphSAGE` | Hamilton et al. 2017 | Mean neighbour aggregation |
| `GAT` | Veličković et al. 2018 | Multi-head attention (concat intermediate, average last) |

All three share the same forward pattern:

```
x → [GNNConv → BatchNorm1d → ReLU → Dropout] × num_layers → Linear → logits
```

Constructor (same for all three, GAT adds `num_heads`):

```python
GCN(in_channels=384, hidden_channels=128, out_channels=14, dropout=0.3, num_layers=2)
```

Use `build_gnn("GCN", ...)` / `build_gnn("GAT", ..., num_heads=4)` for programmatic construction.

#### Baselines — `src/models/baselines.py`

`MLPClassifier` — a standard MLP that ignores `edge_index`. Serves as the "no-graph" H1 baseline.

`LogRegClassifier` — a thin wrapper around `sklearn.LogisticRegression`. It exposes `fit(x, y)` and a `forward(x, edge_index)` compatible with the evaluation harness. Detected by `isinstance(model, LogRegClassifier)` in `train_model()`, which routes it through a single `fit()` call instead of the Adam gradient loop.

---

### 7.4 Fusion

`src/fusion/fusion.py` implements the multimodal fusion strategies used in H2.

#### The three strategies

```python
from src.fusion.fusion import build_fusion, FusedModel, FusedModelFactory

# 1. ConcatFusion — parameter-free, output_dim = text_dim + image_dim
fusion = build_fusion("concat", text_dim=384, image_dim=768)

# 2. WeightedFusion — 2 trainable scalars (softmax weights), optional projections
fusion = build_fusion("weighted", text_dim=384, image_dim=768, output_dim=512)

# 3. GatedFusion — instance-level gate MLP (Arevalo et al. 2017)
fusion = build_fusion("gated", text_dim=384, image_dim=768, output_dim=512)
```

#### FusedModel — wrapping fusion + backbone

The H2 runner precomputes:
```python
data.x = torch.cat([text_emb, image_emb], dim=-1)   # (N, text_dim + image_dim)
```

`FusedModel` receives this concatenated tensor in its `forward`, splits it back at `text_dim`, applies the fusion, and calls the GNN backbone on the fused features:

```python
model = FusedModel(
    fusion=fusion,
    backbone=GCN(in_channels=fusion.output_dim, ...),
    text_dim=384,
    image_dim=768,
)
logits = model(data.x, data.edge_index)   # standard interface preserved
```

#### FusedModelFactory — plugging into hyperparameter_search

`hyperparameter_search()` accepts an optional `model_factory: Callable[[TrainConfig], nn.Module]`. The H2 runner passes a `FusedModelFactory` instance so a fresh `FusedModel` is built for each HP trial:

```python
factory = FusedModelFactory(
    gnn_cls=GCN,
    fusion_name="gated",
    text_dim=384,
    image_dim=768,
    fusion_dim=512,   # output_dim for Weighted/Gated
)
# Factory overrides in_channels internally: backbone in_channels = fusion.output_dim
result = hyperparameter_search(..., model_factory=factory)
```

---

### 7.5 Training Loop

#### `TrainConfig` — all hyperparameters in one place

```python
from src.train import TrainConfig
from dataclasses import replace

cfg = TrainConfig(
    in_channels=384,       # set from encoder output
    hidden_channels=128,
    out_channels=14,       # number of classes
    dropout=0.3,
    num_layers=2,
    lr=0.01,
    weight_decay=1e-4,
    max_epochs=200,
    patience=20,
    warmup_epochs=10,
    min_lr=1e-5,
    seed=0,
    encoder_name="sbert",
    model_name="GCN",
    checkpoint_dir="results/checkpoints",
    device="auto",
)
```

`TrainConfig` is a standard Python `dataclass`. Use `replace(cfg, lr=0.005)` to create a modified copy without mutating the original.

#### `train_model(model, data, config)` — single training run

```python
from src.train import train_model

result = train_model(model, data, config)
# Returns dict: encoder_name, model_name, seed, best_epoch, val_metric,
#               test_accuracy, test_f1_macro, test_f1_weighted, runtime_seconds, hyperparams
```

The training loop:
- **Adam** optimizer with **linear warmup + cosine LR decay**.
- **Early stopping**: stops when validation accuracy does not improve for `patience` epochs.
- **Checkpointing**: saves the best-val checkpoint to `results/checkpoints/`.
- **Dispatch**: `isinstance(model, LogRegClassifier)` → calls sklearn `fit` instead.

#### `hyperparameter_search(...)` — random search

```python
from src.train import hyperparameter_search

result = hyperparameter_search(
    model_cls=GCN,
    data=data,
    search_space={"lr": [0.005, 0.01, 0.05], "hidden_channels": [64, 128, 256], ...},
    n_trials=20,
    base_config=cfg,
    results_dir="results",
    warm_start_params={"lr": 0.01, "hidden_channels": 128, ...},  # optional
    model_factory=None,   # or a FusedModelFactory for H2
)
# Returns: {"best_params": {...}, "best_val_metric": 0.812}
```

If `warm_start_params` is provided, it becomes trial 0, which allows re-using the H1 best-GNN hyperparams as an informed starting point. The remaining `n_trials - 1` slots are drawn uniformly at random.

---

### 7.6 Evaluation Harness

`src/eval/harness.py` provides two functions used by every experiment.

#### `evaluate(model, data, mask)` — single-fold evaluation

```python
from src.eval.harness import evaluate

metrics = evaluate(model, data, data.test_mask)
# Returns: {"accuracy": 0.821, "f1_macro": 0.803, "f1_weighted": 0.818}
```

Sets the model to `eval()` mode, runs `model(data.x, data.edge_index)`, and computes accuracy + macro/weighted F1 using sklearn on the nodes selected by `mask`.

#### `run_seed_sweep(train_fn, n_seeds, base_seed)` — statistical repetition

```python
from src.eval.harness import run_seed_sweep

def my_train_fn(seed: int) -> dict:
    model = build_model(...)
    result = train_model(model, data, replace(cfg, seed=seed))
    return result

per_seed, summary = run_seed_sweep(my_train_fn, n_seeds=5, base_seed=0)
# summary contains:
#   "test_accuracy_mean", "test_accuracy_std",
#   "test_accuracy_ci95_lower", "test_accuracy_ci95_upper",
#   "n_runs", and the same keys for f1_macro, f1_weighted, runtime_seconds
```

`aggregate_metrics` (also in harness.py) can be called directly on any list of result dicts to get the same summary.

---

### 7.7 Utilities

#### `src/utils/seed.py` — `set_seed(seed)`

Sets `random`, `numpy`, `torch`, and CUDA seeds simultaneously. Also enables cuDNN deterministic mode. Called at the start of every seed-sweep iteration.

#### `src/utils/device.py` — `resolve_device(device)`

Converts `"auto"` to a `torch.device` (CUDA if available, else CPU). All encoders and the training loop call this internally, so `device="auto"` always works correctly.

---

## 8. Configuration System (YAML)

Every experiment is fully controlled by a YAML file. No hyperparameters are hardcoded in the source files.

### H1 config structure (`experiments/h1_graph_vs_nograph.yaml`)

```yaml
dataset:
  root: data
  categories: [Electronics]
  split_seed: 0
  split_ratios: [0.6, 0.2, 0.2]
  label_level: 1          # classify into 2nd-level subcategories

encoder:                   # Tier-0 baseline used in H1
  name: tfidf
  max_features: 10000

models:
  - name: GCN
    search_space:
      lr:              [0.005, 0.01, 0.05]
      weight_decay:    [0.0, 0.0001, 0.001]
      hidden_channels: [64, 128, 256]
      dropout:         [0.0, 0.3, 0.5]
  # ... more models

training:
  num_layers: 2
  max_epochs: 200
  patience: 20
  n_hparam_trials: 20
  n_seed_runs: 5
  base_seed: 0

results:
  dir: results
  output_file: results/h1_results.json
```

### H3 config additions (`experiments/h3_text_encoders.yaml`)

H3 adds an `encoders` list (replacing the single `encoder` key) and `gnn_search_space`. Each encoder entry has **constructor kwargs** (forwarded to `build_encoder`) plus **metadata fields** (`modality`, `tier`, `encoder_size_params`, `pretraining_objective`) that are stripped before forwarding but injected into result dicts.

```yaml
encoders:
  - name: sbert
    type: SentenceBERT
    model_name: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 64
    normalize_embeddings: true
    # --- metadata (not forwarded to build_encoder) ---
    modality: text
    tier: 1
    encoder_size_params: 22000000
    pretraining_objective: MLM
```

### H2 config additions (`experiments/h2_multimodal.yaml`)

H2 adds:
- `h1_results_path` + `h3_results_path` — paths to prior results for warm-starts.
- `fallback_text_encoder` — encoder name to use if H3 results don't exist yet.
- `vision_encoders` list — same structure as text encoders.
- `fusion_strategies: [concat, weighted, gated]`.
- `modalities: [text_only, image_only, text+image]`.
- `image_pipeline` section with `max_workers`, `timeout`, `max_retries`, `cache_dir`.

---

## 9. Results Format

Every per-seed run writes a JSON file to `results/` with this schema:

```json
{
  "encoder_name": "sbert",
  "model_name": "GCN",
  "seed": 2,
  "best_epoch": 87,
  "val_metric": 0.8314,
  "test_accuracy": 0.8201,
  "test_f1_macro": 0.8093,
  "test_f1_weighted": 0.8197,
  "runtime_seconds": 42.31,
  "hyperparams": {
    "lr": 0.01,
    "weight_decay": 0.0001,
    "hidden_channels": 128,
    "dropout": 0.3,
    "num_layers": 2
  }
}
```

H2 results additionally contain `modality`, `text_enc`, `vision_enc`, `fusion`, and (for image-containing modalities) `test_acc_with_images`, `test_acc_without_images`, `n_test_with_images`, `n_test_without_images`.

### The final output files

| File | Content |
|------|---------|
| `results/h1_results.json` | All H1 per-seed result dicts |
| `results/h3_results.json` | All H3 per-seed result dicts |
| `results/h2_results.json` | All H2 per-seed result dicts |
| `results/hparam_search_{enc}_{model}.jsonl` | One JSON line per HP trial |
| `results/checkpoints/{enc}_{model}_seed{N}.pt` | Best model weights |

The benchmark tables printed at the end of each runner are not currently saved to a file; if you need them, redirect stdout or use `scripts/make_benchmark_table.py` (planned).

---

## 10. Testing

```bash
# Run all tests:
python -m pytest tests/ -v

# Run only fusion tests:
python -m pytest tests/test_fusion.py -v

# Run only encoder tests (12 will be skipped if neural libraries are absent):
python -m pytest tests/test_encoders.py -v
```

### Test structure

| File | What it covers |
|------|----------------|
| `tests/test_harness.py` | `evaluate()`, `run_seed_sweep()`, `aggregate_metrics()` |
| `tests/test_encoders.py` | BoW, TF-IDF, SBERT (mocked), CLIP-text (mocked), Qwen3 (mocked), registry dispatch |
| `tests/test_fusion.py` | ConcatFusion, WeightedFusion, GatedFusion, build_fusion, FusedModel, FusedModelFactory |

Tests that require optional libraries (transformers, sentence-transformers) are skipped rather than failing when those libraries are not installed. This is implemented with `pytest.importorskip("sentence_transformers")` as a statement inside the test body, not in the decorator.

### Module-level smoke tests

Every source module has a smoke test in its `if __name__ == "__main__":` block. Run any of them with:

```bash
python -m src.data.image_pipeline
python -m src.encoders.vision
python -m src.fusion.fusion
python -m src.train
python -m src.eval.harness
```

---

## 11. How to Extend the Project

### Adding a new text encoder

1. Create `src/encoders/my_encoder.py`. Subclass `BaseEncoder` and implement `encode(inputs)` and the `embedding_dim` property. If it requires fitting (e.g. learns a vocabulary), also override `fit(train_inputs)`.

2. Register it in `src/encoders/registry.py`:
   ```python
   if enc_type == "MyEncoder":
       from src.encoders.my_encoder import MyEncoder
       return MyEncoder(cache_dir=cache_dir, device=device, **kwargs)
   ```

3. Add it to the YAML grid in `experiments/h3_text_encoders.yaml`:
   ```yaml
   - name: my-encoder
     type: MyEncoder
     my_param: value
     modality: text
     tier: 2
     encoder_size_params: 50000000
     pretraining_objective: contrastive
   ```

4. No changes to the runner scripts needed — `build_encoder` and `compute_embeddings` are already generic.

### Adding a new GNN architecture

1. Add a class to `src/models/gnns.py` that inherits `BaseGNN` and implements `_build_layers()`.

2. Register it in the `_GNN_REGISTRY` dict at the bottom of the same file.

3. Register it in the `MODEL_REGISTRY` in each runner script that should use it.

4. Add a search space entry to any relevant YAML file.

### Adding a new fusion strategy

1. Add a class to `src/fusion/fusion.py` that inherits `BaseFusion` and implements `forward(text_x, image_x)` and the `output_dim` property.

2. Add a branch in `build_fusion()`:
   ```python
   if key == "myfusion":
       return MyFusion(text_dim, image_dim, output_dim)
   ```

3. Add `"myfusion"` to `fusion_strategies` in `experiments/h2_multimodal.yaml`.

### Changing the hyperparameter search space

Edit the `gnn_search_space` section of the relevant YAML file. The runner performs uniform random sampling over the Cartesian product of the lists:

```yaml
gnn_search_space:
  lr:              [0.001, 0.005, 0.01, 0.05]   # add a new LR candidate
  weight_decay:    [0.0, 0.0001, 0.001]
  hidden_channels: [64, 128, 256, 512]            # add a wider model
  dropout:         [0.0, 0.1, 0.3, 0.5]
```

Increasing `n_hparam_trials` in the `training` section gives the search more budget.

### Running a different dataset category

Change `categories` in the YAML:

```yaml
dataset:
  categories: [Sports_and_Outdoors]
```

The code expects `data/raw/meta_Sports_and_Outdoors.json[.gz]`. The split file will be regenerated automatically.

### Using a different split ratio or seed

```yaml
dataset:
  split_seed: 42
  split_ratios: [0.7, 0.15, 0.15]
```

The split is cached to `data/splits/split_42.npz` and reused across all encoder/model combinations with the same `split_seed`, so the comparison remains valid.

---

## 12. Reproducibility Contract

The project makes the following guarantees:

1. **Deterministic splits**: `AmazonCopurchase` saves train/val/test index arrays to `data/splits/split_{seed}.npz` and reloads them on every subsequent run. Every encoder and model sees the same nodes in each split.

2. **Seed propagation**: `set_seed(seed)` is called at the start of every training run, setting `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and cuDNN deterministic mode.

3. **Embedding caching**: Once computed, embeddings are saved to `data/embeddings/{key}.pt`. Rerunning an experiment never re-encodes the same inputs. The cache key includes the encoder name and split seed.

4. **Five-seed evaluation**: Every reported number is `mean ± std` across 5 seeds, with a 95% CI computed as `mean ± 1.96 * std / sqrt(5)`. This is enforced by `run_seed_sweep(..., n_seeds=5)`.

5. **Per-trial logging**: Every hyperparameter trial is appended to a JSONL file so the full search trace is inspectable after the fact.

6. **Independent HP search per encoder**: Each encoder in H3 and H2 gets its own hyperparameter search from scratch (with only the warm-start from the previous hypothesis as a hint). This prevents the best-performing encoder from unfairly inheriting tuned hyperparameters.

---

## 13. Common Issues

### `ModuleNotFoundError: No module named 'src'`

The `src` package must be on `sys.path`. When running scripts directly (`python scripts/run_h1.py`), each script adds the project root with `sys.path.insert(0, ...)`. When using `python -m` (e.g. `python -m src.train`), the project root is already on the path.

If pytest fails with this error, check that `tests/conftest.py` is present (it adds the root to sys.path automatically).

### `torch_geometric` import fails

PyG must be installed with matching CUDA/CPU wheels. On CPU-only machines:
```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```
Replace `torch-2.2.0` with your actual torch version.

### Image download fails / all images are missing

In mock mode this is expected — all `image_url` fields are empty strings, so all image embeddings are zero vectors. On real data, check your internet connection and firewall settings. Increase `timeout` and `max_retries` in the YAML `image_pipeline` section for slow/unreliable connections.

### `CUDA out of memory`

The vision encoders (`_encode_with_paths`) automatically halve `batch_size` and retry on CUDA OOM. For Qwen3 encoders, reduce `batch_size` in the YAML:
```yaml
- name: qwen3-4B
  type: Qwen3
  model_size: "4B"
  batch_size: 4       # reduce from 8
```

### Results vary between runs despite fixed seeds

Make sure you are not running experiments in parallel (different processes sharing the same `data/embeddings/` cache can cause race conditions). The embedding cache is read-once/write-once per key, so parallel runs of the same encoder will be fine at inference but the training jobs themselves should be run sequentially to avoid interleaved checkpoint writes.

### H2 confounder warning is triggered

A "⚠ CONFOUNDER" flag in the H2 output means the accuracy on nodes with available images is more than 5 percentage points higher than on nodes without images. This does **not** necessarily mean the model is wrong — it means the missing-data pattern correlates with the label. Review the `test_acc_with_images` vs `test_acc_without_images` columns and consider running the experiment with only nodes that have images to get an unbiased comparison.
