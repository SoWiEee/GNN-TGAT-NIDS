# Training Guide

Training details, optimization strategies, and research tooling for GNN-TGAT-NIDS.

---

## Prerequisites (local development)

- Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+
- CUDA 12.4+ (optional; CPU mode works for small datasets)

```bash
git clone https://github.com/SoWiEee/GNN-TGAT-NIDS.git
cd GNN-TGAT-NIDS

uv sync
uv run pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

uv run pytest  # verify installation
```

Or inside Docker:

```bash
docker compose exec backend bash
```

---

## Data Preparation

```bash
# Merge UNSW-NB15 training + testing CSVs → data/raw/NF-UNSW-NB15-v2.csv
# Also creates data/demo/demo_flows.csv (1 000-flow stratified sample)
uv run python scripts/create_demo_dataset.py

# Build PyG time-window graphs from the merged CSV
uv run python src/data/static_builder.py

# Build temporal graphs (required for TGAT/TGN)
uv run python src/data/temporal_builder.py
```

---

## Training Models

```bash
# Static models
uv run python train.py model=graphsage
uv run python train.py model=gat
uv run python train.py model=egraphsage

# Temporal models
uv run python train.py model=tgat data=temporal_default
uv run python train.py model=tgn data=temporal_default

# Early stopping
uv run python train.py model=graphsage train.patience=10

# Inside Docker
docker compose exec backend uv run python train.py model=graphsage
```

---

## Training Optimization

### Proxy Node Identity (TTL + Protocol)

UNSW-NB15 processed CSVs contain no IP address columns, which causes a degenerate 2-node graph per time-window. `static_builder.py` builds proxy node identities:

| Role | Key | Rationale |
|---|---|---|
| Source node | `("src", sttl // 16, proto)` | TTL bin ~ OS type + protocol |
| Destination node | `("dst", dttl // 16, service)` | TTL + service ~ server segment |

This produces **~20-50 distinct nodes per window** instead of 2. Observed effect: val F1 improved from ~0.50 to ~0.84.

### Time-Window Size

Default window: **300s** (`configs/data/static_default.yaml`). Larger windows give denser graphs (~300 edges/window) at the cost of fewer total windows (860 vs 4 295).

### Automatic Mixed Precision (AMP)

Enabled by default on CUDA via `train.use_amp=true`. Uses `torch.amp.autocast` + `GradScaler` — typically **1.5-2x faster**. Disable with `train.use_amp=false`.

### DataLoader Tuning

| Config key | Default | Effect |
|---|---|---|
| `train.batch_size` | `32` | Graph windows per batch |
| `train.num_workers` | `0` | Set to `4` on Linux for async loading |
| `train.val_every` | `1` | Evaluate every N epochs; `5` saves ~20% time |
| `train.save_every` | `10` | Periodic checkpoint cadence |

---

## Macro F1 Improvement

NF-UNSW-NB15-v2 has extreme class imbalance (Benign 93%, rarest 5 attack types < 1%). This causes low macro F1 despite high weighted F1. Three mechanisms address this:

### 1. Class Weight Strategy

```yaml
train:
  class_weight_strategy: sqrt_inverse  # "inverse" | "effective" | "sqrt_inverse"
```

| Strategy | Formula | Effect |
|---|---|---|
| `inverse` (default) | `N / (C × count_c)` | Standard inverse-frequency. Can produce 5000:1 weight ratios |
| `effective` | `(1-β) / (1-β^count_c)` | Cui et al. 2019. Smoother scaling, but collapses Benign detection under extreme imbalance |
| `sqrt_inverse` | `√(N / (C × count_c))` | Damped inverse-frequency. Best balance for this dataset |

### 2. Validation Metric Selection

```yaml
train:
  val_metric: macro_f1  # "f1" (weighted) | "macro_f1"
```

Using `macro_f1` for checkpoint selection forces the model to optimize for rare-class performance rather than Benign-dominated weighted F1.

### 3. Focal Loss Gamma

Higher `focal_gamma` increases focus on hard-to-classify rare samples:

```yaml
train:
  focal_gamma: 3.0  # default 2.0, increase for more rare-class focus
```

### 4. LR Scheduler

```yaml
train:
  scheduler: cosine  # "none" | "cosine" | "plateau"
```

Cosine annealing (`CosineAnnealingLR`) decays LR from `train.lr` to 1% of initial over the full training run. This prevents overshooting on rare-class gradients and was the single most impactful improvement for macro F1.

### Recommended Configuration

```bash
uv run python train.py model=graphsage \
    train.class_weight_strategy=sqrt_inverse \
    train.focal_gamma=3.0 \
    train.val_metric=macro_f1 \
    train.scheduler=cosine \
    train.patience=20
```

### Experiment Results

| Model | Config | Weighted F1 | Macro F1 | Shellcode F1 | Worms F1 |
|-------|--------|:-----------:|:--------:|:------------:|:--------:|
| GraphSAGE | Baseline (`inverse`, γ=2.0, val=f1) | 0.9712 | 0.4657 | 0.466 | 0.116 |
| GraphSAGE | `effective`, γ=2.0, val=macro_f1 | 0.0300 | — | — | — |
| GraphSAGE | `sqrt_inverse`, γ=3.0, val=macro_f1 | 0.9681 | 0.4714 | 0.535 | 0.193 |
| GraphSAGE | `sqrt_inverse`, γ=3.0, macro_f1, cosine | **0.9766** | **0.5389** | **0.746** | **0.227** |
| E-GraphSAGE | Baseline (`inverse`, γ=2.0, val=f1) | 0.9708 | 0.4681 | 0.474 | 0.099 |
| E-GraphSAGE | `sqrt_inverse`, γ=3.0, macro_f1, cosine | **0.9756** | **0.5499** | **0.708** | **0.343** |

### Per-Class Improvement (Best Config vs Baseline)

| Class | Support | Baseline F1 | Improved F1 | Δ |
|-------|--------:|:-----------:|:-----------:|:---:|
| Benign | 369,299 | 0.9935 | 0.9981 | +0.005 |
| Analysis | 239 | 0.1458 | 0.2752 | **+0.129** |
| Backdoor | 234 | 0.1180 | 0.2067 | **+0.089** |
| DoS | 1,587 | 0.2020 | 0.2294 | +0.027 |
| Exploits | 11,986 | 0.7760 | 0.7407 | -0.035 |
| Fuzzers | 7,502 | 0.6978 | 0.7474 | **+0.050** |
| Generic | 1,527 | 0.4076 | 0.4350 | +0.027 |
| Recon | 4,330 | 0.7341 | 0.7839 | **+0.050** |
| Shellcode | 576 | 0.4661 | 0.7460 | **+0.280** |
| Worms | 69 | 0.1163 | 0.2269 | **+0.111** |

**Key findings:**
- `effective` weights are too aggressive for NF-UNSW-NB15-v2 — they reduce Benign weight to ~0.0003, collapsing overall detection
- Cosine annealing LR was the single biggest contributor — it prevents the optimizer from overshooting on rare-class gradients in later epochs
- The best config improves BOTH weighted F1 and macro F1 simultaneously — no trade-off
- E-GraphSAGE benefits even more than GraphSAGE: +0.0818 macro F1 (vs +0.0732), likely because edge-feature-aware message passing provides richer representations for distinguishing attack subtypes
- Edge-level oversampling (`oversample_factor=5`) hurt performance, likely due to duplicate gradient signal causing overfitting
- Post-hoc threshold calibration (prior-adjusted prediction) had no effect — the model's argmax is already optimal given its learned representations
- The confusion is primarily between attack subtypes (DoS↔Exploits, Generic↔DoS/Exploits, Analysis↔Backdoor), not benign vs attack

---

## Hyperparameter Search (Optuna)

```bash
uv sync --group dev

# Static models
uv run python scripts/tune_hyperparams.py --model graphsage --trials 50
uv run python scripts/tune_hyperparams.py --model gat --trials 50
uv run python scripts/tune_hyperparams.py --model egraphsage --trials 50

# Temporal models
uv run python scripts/tune_hyperparams.py --model tgat --trials 30
uv run python scripts/tune_hyperparams.py --model tgn --trials 30

# Live dashboard
uv run optuna-dashboard sqlite:///results/optuna.db
```

Resume-safe (SQLite storage). Best params saved to `results/best_hparams_{model}.json`.

**Search space (static):**

| Hyperparameter | Range / Choices |
|---|---|
| `lr` | 1e-4 ~ 1e-2 (log) |
| `hidden_dim` | 128 / 256 / 512 |
| `num_layers` | 2 / 3 / 4 |
| `dropout` | 0.0 ~ 0.5 |
| `batch_size` | 16 / 32 / 64 |
| `num_heads` (GAT) | 2 / 4 / 8 |
| `aggregation` (SAGE) | mean / max |

**Search space (temporal):**

| Hyperparameter | Range / Choices |
|---|---|
| `lr` | 1e-4 ~ 1e-2 (log) |
| `hidden_dim` | 64 / 128 / 172 / 256 |
| `heads` | 1 / 2 / 4 |
| `n_neighbors` | 10 / 20 / 30 |
| `batch_size` | 100 / 200 / 400 |
| `memory_dim` (TGN) | 64 / 100 / 128 |

**Apply best params:**

```bash
uv run python train.py model=graphsage \
  model.hidden_dim=128 model.num_layers=2 model.dropout=0.1 \
  train.lr=0.0087 train.batch_size=16 train.epochs=200
```

---

## Adversarial Training

```bash
uv run python train.py model=graphsage train.adversarial_training=true \
    train.epochs=30 train.patience=10 \
    train.checkpoint_dir=checkpoints/graphsage_adv

uv run python train.py model=gat train.adversarial_training=true \
    train.adv_epsilon=0.1 train.adv_steps=10 train.adv_ratio=0.3
```

Saved as `{model}_adv_best.pt`. `compute_reliability_metrics.py` automatically picks up `_adv_best.pt` files.

---

## Model Ensemble

```python
from src.models.ensemble import EnsembleModel

ensemble = EnsembleModel(
    models={"graphsage": model_gs, "gat": model_gat, "egraphsage": model_egs},
    strategy="soft_vote",  # or "hard_vote", "weighted"
)
proba = ensemble(data)
```

The API accepts `model=ensemble` in `/api/analyze` for ensemble inference. Weights are learned from validation F1 via `EnsembleModel.from_validation()`.

---

## Streaming Inference (WebSocket)

```
ws://localhost:8000/api/ws/stream?model=graphsage&window_seconds=60
```

- Send: `{"flows": [{"col1": val, ...}, ...]}` — buffered into time windows
- Send: `{"command": "flush"}` — force inference on buffered flows
- Receive: `{"type": "alerts", "window": 0, "alerts": [...], "stats": {...}}`

---

## Explainability

```python
from src.explain.gnn_explainer import explain_flow, explain_top_alerts

result = explain_flow(model, data, edge_idx=42, epochs=200)
results = explain_top_alerts(model, data, top_k=5)
```

- Static models (GraphSAGE, GAT, E-GraphSAGE): GNNExplainer
- Temporal models (TGAT, TGN): integrated gradients

API: `POST /api/explain/{session_id}`, `POST /api/explain-top/{session_id}`

---

## ONNX Export

```bash
uv run python scripts/export_onnx.py --model graphsage
uv run python scripts/export_onnx.py --model gat --quantize
```

Temporal models (TGAT, TGN) cannot be exported to ONNX due to stateful memory.

---

## Reliability Metrics

```bash
uv run python scripts/compute_reliability_metrics.py
```

Runs clean eval + C-PGD attack eval + adversarial-training eval. Saves to `data/metrics/reliability.json`.

---

## Research Scripts

| Script | Purpose |
|---|---|
| `scripts/multi_seed_eval.py` | Multi-seed training for statistical significance |
| `scripts/cross_dataset_validation.py` | Cross-dataset generalization evaluation |
| `scripts/window_size_eval.py` | Time-window size sensitivity analysis |
| `scripts/compute_reliability_metrics.py` | Full model reliability metrics + per-class report |
| `scripts/tune_hyperparams.py` | Optuna Bayesian hyperparameter search |
| `scripts/export_onnx.py` | ONNX export + optional quantization |
| `scripts/pcap_to_netflow.py` | PCAP to NetFlow CSV (nfstream) |
