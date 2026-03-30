# GNN-TGAT-NIDS

**Upload NetFlow traffic → Detect intrusions with GNN → Visualize, alert, and report**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg)](https://pytorch.org/)
[![Vue](https://img.shields.io/badge/Vue-3-42b883.svg)](https://vuejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An interactive web-based Network Intrusion Detection System powered by Graph Neural Networks.
> Upload a NetFlow CSV, explore the traffic graph, review alerts, and export a full security report — including adversarial robustness analysis.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 🔵 | **Interactive Traffic Graph** | IP nodes + flow edges coloured by risk level. Click any node to inspect its connections and threat score. |
| 🔔 | **Alert List** | Per-flow alerts with attack type, confidence score, and the top features that triggered detection (via GAT attention weights). |
| 📊 | **Attack Timeline** | Stacked time-series showing attack-type distribution across 300-second windows. Spot bursts and campaign patterns at a glance. |
| 🛡️ | **Model Reliability Panel** | Pre-computed metrics answering "how trustworthy is this system?": clean F1, detection rate under adversarial attack, and improvement after adversarial training. |
| ⚗️ | **Adversarial Comparison Report** | Side-by-side view of original vs. adversarially-perturbed flows — which features changed, by how much, and whether all network protocol constraints are still satisfied. Exportable as PDF / HTML. |

---

## Architecture

```mermaid
flowchart LR
    subgraph Input
        CSV["📄 NetFlow CSV\n(NF-UNSW-NB15-v2\nor custom upload)"]
    end

    subgraph Backend["Backend — FastAPI"]
        SB["Static Graph Builder\n300 s tumbling windows"]
        GNN["GNN Inference\nGraphSAGE / GAT"]
        ADV["Adversarial Module\nC-PGD + Constraint Check"]
        RPT["Report Generator\nJinja2 → PDF / HTML"]
    end

    subgraph Frontend["Frontend — Vue 3 + Vite"]
        VIZ["① Traffic Graph\nCytoscape.js"]
        ALT["② Alert List"]
        TSC["③ Attack Timeline\nPlotly.js"]
        MRP["④ Model Reliability\nPanel"]
        ACP["⑤ Adversarial\nComparison Report"]
    end

    CSV --> SB --> GNN --> VIZ & ALT & TSC
    GNN --> ADV --> ACP
    ACP --> RPT
    MRP -.->|pre-computed metrics| MRP
```

---

## Quick Start

### Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+
- CUDA 12.4+ (optional; CPU mode is supported for small datasets)

### 1. Backend

```bash
git clone https://github.com/SoWiEee/GNN-TGAT-NIDS.git
cd GNN-TGAT-NIDS

# Install Python dependencies
uv sync
uv run pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# (Optional) Run tests
uv run pytest
```

### 2. Prepare dataset

```bash
# Merge UNSW-NB15 training + testing CSVs → data/raw/NF-UNSW-NB15-v2.csv
# Also creates data/demo/demo_flows.csv (1 000-flow stratified sample)
uv run python scripts/create_demo_dataset.py

# Build PyG time-window graphs from the merged CSV
uv run python src/data/static_builder.py
```

### 3. Train models (or use pre-trained checkpoints)

```bash
uv run python train.py model=graphsage
uv run python train.py model=gat
```

### 4. Pre-compute model reliability metrics

```bash
# Runs clean eval + C-PGD attack eval + adversarial-training eval
# Saves results to data/metrics/reliability.json (loaded by the frontend)
uv run python scripts/compute_reliability_metrics.py
```

### 5. Frontend

```bash
cd frontend
npm install
npm run dev          # development server at http://localhost:5173
# or
npm run build && npm run preview    # production preview
```

The FastAPI backend starts automatically when the frontend makes its first request, or run it manually:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

---

## Training Optimization

### Graph node construction — proxy identity from TTL + protocol

UNSW-NB15 processed CSVs contain no IP address columns, which causes a
degenerate 2-node graph per time-window (all flows share the same
`unknown_src → unknown_dst` pair).  GNN message passing over a 2-node graph
adds global window noise to every edge embedding rather than useful neighbourhood
signal.

**Fix:** `static_builder.py` now builds proxy node identities from columns that
are available in the dataset:

| Role | Key | Rationale |
|---|---|---|
| Source node | `("src", sttl // 16, proto)` | TTL bin ≈ OS type (Linux 64→4, Windows 128→8, Cisco 255→15) + protocol |
| Destination node | `("dst", dttl // 16, service)` | TTL + service ≈ server/service segment |

This produces **~20–50 distinct nodes per window** instead of 2, enabling
meaningful neighbourhood aggregation.  Observed effect: val F1 improved from
~0.50 to ~0.84 with the same architecture and only 5 training epochs.

### Time-window size

Default window changed **60 s → 300 s** (`configs/data/static_default.yaml`).
Larger windows give denser graphs (~300 edges/window) at the cost of fewer total
windows (860 vs 4 295).  Denser graphs give the GNN more neighbours per node.

### Automatic Mixed Precision (AMP)

Enabled by default on CUDA via `train.use_amp=true`.  Uses
`torch.amp.autocast` + `GradScaler` — typically **1.5–2× faster** on modern
GPUs with no accuracy loss.  Disable with `train.use_amp=false` if needed.

### DataLoader tuning

| Config key | Default | Effect |
|---|---|---|
| `train.batch_size` | `32` | Graph windows per batch (was 1) |
| `train.num_workers` | `0` | Set to `4` on Linux for async data loading |
| `train.val_every` | `1` | Evaluate on val set every N epochs; `5` saves ~20% time |
| `train.save_every` | `10` | Periodic checkpoint cadence |

### Hyperparameter search with Optuna

```bash
# Install dev extras (includes optuna + optuna-dashboard)
uv sync --group dev

# Search GraphSAGE — 50 Bayesian trials × 30 epochs each (~45 min on GPU)
uv run python scripts/tune_hyperparams.py --model graphsage --trials 50

# Search GAT in a second terminal
uv run python scripts/tune_hyperparams.py --model gat --trials 50

# Live dashboard while running (open http://localhost:8080)
uv run optuna-dashboard sqlite:///results/optuna.db
```

The search is **resume-safe** — re-running the same command continues from
where it left off (SQLite storage).  Best parameters are saved to
`results/best_hparams_{model}.json`.

**Search space:**

| Hyperparameter | Range / Choices |
|---|---|
| `lr` | 1 × 10⁻⁴ → 1 × 10⁻² (log scale) |
| `hidden_dim` | 128 / 256 / 512 |
| `num_layers` | 2 / 3 / 4 |
| `dropout` | 0.0 → 0.5 |
| `batch_size` | 16 / 32 / 64 |
| `num_heads` (GAT only) | 2 / 4 / 8 |
| `aggregation` (SAGE only) | mean / max |

Pruning (MedianPruner) stops unpromising trials after 10 epochs — effectively
free early stopping during search.

**Apply best params to full training:**
```bash
# Example — substitute values from results/best_hparams_graphsage.json
uv run python train.py model=graphsage \
  model.hidden_dim=128 model.num_layers=2 model.dropout=0.1 \
  train.lr=0.0087 train.batch_size=16 train.epochs=200
```

---

## Project Structure

```
GNN-NIDS-Analyzer/
├── src/                        # ML core (reused from GNN-TGAT-NIDS)
│   ├── data/
│   │   ├── loader.py           ← chronological split, label encoding
│   │   ├── static_builder.py   ← NetFlow CSV → PyG Data (tumbling windows)
│   │   └── static_dataset.py   ← on-demand PyG Dataset loader
│   ├── models/
│   │   ├── base.py             ← BaseNIDSModel ABC
│   │   ├── graphsage.py        ← 3-layer GraphSAGE edge classifier
│   │   └── gat.py              ← 4-head GAT with attention export
│   ├── attack/
│   │   ├── base.py             ← BaseAttack ABC
│   │   ├── constraints.py      ← TCP validity, co-dependency, bounds
│   │   └── cpgd.py             ← Constrained PGD (adversarial comparison)
│   └── utils/
│       ├── seed.py
│       └── checkpoint.py
├── app/                        # FastAPI application
│   ├── main.py                 ← app entry point, CORS, lifespan
│   ├── routers/
│   │   ├── analysis.py         ← POST /analyze
│   │   ├── adversarial.py      ← POST /adversarial
│   │   └── report.py           ← GET  /report/{session_id}
│   ├── services/
│   │   ├── inference.py        ← runs GNN on uploaded data
│   │   ├── graph_builder.py    ← builds Cytoscape.js JSON from PyG output
│   │   └── report_builder.py   ← Jinja2 → HTML → WeasyPrint PDF
│   └── templates/
│       └── report.html.j2
├── frontend/                   # Vue 3 + Vite
│   ├── src/
│   │   ├── views/
│   │   │   ├── TrafficGraph.vue
│   │   │   ├── AlertList.vue
│   │   │   ├── AttackTimeline.vue
│   │   │   ├── ReliabilityPanel.vue
│   │   │   └── AdversarialReport.vue
│   │   ├── components/
│   │   ├── stores/             ← Pinia stores
│   │   └── api/                ← axios API client
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── create_demo_dataset.py      ← merge UNSW-NB15 CSVs, create demo sample
│   ├── compute_reliability_metrics.py ← clean F1 + C-PGD DR → reliability.json
│   └── tune_hyperparams.py         ← Optuna Bayesian hyperparameter search
├── configs/                    # Hydra configs
├── data/
│   ├── raw/                    ← place dataset CSVs here (git-ignored)
│   ├── processed/              ← built by static_builder (git-ignored)
│   ├── demo/                   ← curated 1000-flow demo CSV (tracked)
│   └── metrics/
│       └── reliability.json    ← pre-computed F1 / DR / ΔF1
├── checkpoints/                ← pre-trained model weights (git-ignored)
│   ├── graphsage_best.pt
│   └── gat_best.pt
├── tests/
├── docs/
│   └── spec.md
├── pyproject.toml
└── README.md
```

---

## Demo

> **Try with the included demo dataset** (1 000 flows, subset of NF-UNSW-NB15-v2):
>
> ```bash
> cp data/demo/demo_flows.csv data/raw/demo_flows.csv
> # then follow Quick Start steps 2 → 5
> ```

*Screenshots / demo video — coming in Phase 2*

---

## Model Reliability (Pre-computed on NF-UNSW-NB15-v2 test split)

| Metric | GraphSAGE | GAT |
|--------|:---------:|:---:|
| Weighted F1 (clean) | TBD | TBD |
| DR@attack — C-PGD ε=0.1 | TBD | TBD |
| ΔF1 after adversarial training | TBD | TBD |

*Values will be filled after Phase 1 training is complete.*

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML framework | PyTorch 2.4 + PyTorch Geometric 2.6 |
| GNN models | GraphSAGE, GAT (Phase 2: TGAT, TGN) |
| Backend | FastAPI + uvicorn |
| Frontend | Vue 3 + Vite + TypeScript |
| Graph visualization | Cytoscape.js |
| Charts | Plotly.js |
| Report generation | Jinja2 + WeasyPrint |
| Config management | Hydra |
| Package manager | uv (Python), npm (JS) |

---

## Datasets

- **NF-UNSW-NB15-v2** (~2.5 M flows, 9 attack types) — primary dataset. Place at `data/raw/NF-UNSW-NB15-v2.csv`. Available from [UNSW Sydney](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- `data/demo/demo_flows.csv` — 1 000-flow curated subset included in the repository.

---

## Roadmap

**Phase 1 — Core Tool** (Weeks 1–8)
- [ ] FastAPI backend + CSV upload endpoint
- [ ] GNN inference service
- [ ] Vue 3 frontend scaffold
- [ ] Traffic graph view (Cytoscape.js)
- [ ] Alert list view
- [ ] Attack timeline view
- [ ] Model reliability panel (pre-computed JSON)

**Phase 2 — Adversarial & Depth** (Weeks 9–12)
- [ ] C-PGD adversarial module (`src/attack/cpgd.py`)
- [ ] Adversarial comparison report view
- [ ] PDF / HTML export
- [ ] Curated demo dataset + demo video

**Phase 3 — Temporal Models** (Future)
- [ ] TGAT / TGN models
- [ ] PCAP → NetFlow conversion (nfstream)
- [ ] Memory Poisoning Attack visualization

---

## References

**GNN Models**
- Hamilton et al. "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* — GraphSAGE
- Veličković et al. "Graph Attention Networks." *ICLR 2018.* — GAT
- Xu et al. "Inductive Representation Learning on Temporal Graphs." *ICLR 2020.* — TGAT
- Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." *arXiv 2020.* — TGN

**GNN-based NIDS**
- Lo et al. "E-GraphSAGE: A GNN-based IDS for IoT." *IEEE NOMS 2022.*
- Bilot et al. "Graph Neural Networks for Intrusion Detection: A Survey." *IEEE Access 2023.*

**Adversarial Attacks on NIDS**
- Han et al. "Practical Traffic-Space Adversarial Attacks on Learning-Based NIDSs." *USENIX Security 2021.* — BAAAN
- Pierazzi et al. "Intriguing Properties of Adversarial ML Attacks in the Problem Space." *IEEE S&P 2020.*
- Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018.* — PGD

---

## License

MIT License. See [LICENSE](LICENSE).
