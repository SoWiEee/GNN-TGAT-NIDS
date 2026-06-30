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

## Quick Start

Requires [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/SoWiEee/GNN-TGAT-NIDS.git
cd GNN-TGAT-NIDS

# Place trained checkpoints in checkpoints/ (or train inside the container)
# Place dataset CSV in data/raw/ (or use the included demo)

docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost |
| Backend API | http://localhost:8000 |
| API via nginx | http://localhost/api/* |

**Volume mounts** (host → container):

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `checkpoints/` | `/app/checkpoints` | Model weights |
| `data/raw/` | `/app/data/raw` | Upload datasets |
| `data/processed/` | `/app/data/processed` | Built graphs |
| `data/metrics/` | `/app/data/metrics` | Reliability panel |
| `data/sessions/` | `/app/data/sessions` | Analysis sessions |

**Train inside Docker:**

```bash
docker compose exec backend uv run python train.py model=graphsage
```

> For local development without Docker, training details, hyperparameter search, and ONNX export, see [docs/training.md](docs/training.md).

---

## Features

| Feature | Description |
|---------|-------------|
| **Interactive Traffic Graph** | IP nodes + flow edges coloured by risk level (Cytoscape.js) |
| **Alert List** | Per-flow alerts with attack type, confidence, and top features |
| **Attack Timeline** | Stacked time-series of attack-type distribution (Plotly.js) |
| **Model Reliability Panel** | Pre-computed clean F1, adversarial DR, and ΔF1 after adversarial training |
| **Adversarial Comparison** | Side-by-side original vs. perturbed flows with constraint validation. PDF/HTML export |
| **Explainability** | GNNExplainer (static) + integrated gradients (temporal) with feature importance bars |
| **Streaming Inference** | Real-time NetFlow analysis via WebSocket |

---

## Architecture

```mermaid
flowchart LR
    subgraph Input
        CSV["NetFlow CSV\n(NF-UNSW-NB15-v2\nor custom upload)"]
    end

    subgraph Backend["Backend — FastAPI"]
        SB["Static Graph Builder\n300 s tumbling windows"]
        GNN["GNN Inference\nGraphSAGE / GAT / E-GraphSAGE\nTGAT / TGN"]
        ADV["Adversarial Module\nC-PGD / Edge Injection / GAN"]
        RPT["Report Generator\nJinja2 → PDF / HTML"]
    end

    subgraph Frontend["Frontend — Vue 3 + Vite"]
        VIZ["Traffic Graph\nCytoscape.js"]
        ALT["Alert List +\nExplainability"]
        TSC["Attack Timeline\nPlotly.js"]
        MRP["Model Reliability\nPanel"]
        ACP["Adversarial\nComparison Report"]
    end

    CSV --> SB --> GNN --> VIZ & ALT & TSC
    GNN --> ADV --> ACP
    ACP --> RPT
    MRP -.->|pre-computed metrics| MRP
```

---

## Model Performance

Evaluated on NF-UNSW-NB15-v2 test split (~397K flows, 3312 windows). Hyperparameters tuned via Optuna (TPE sampler + MedianPruner, 50 trials).

| Metric | GraphSAGE | GAT | TGAT | TGN |
|--------|:---------:|:---:|:----:|:---:|
| **Weighted F1** | **0.9773** | — | 0.9475 | 0.9463 |
| **Macro F1** | **0.5735** | — | 0.3643 | 0.3438 |
| Precision / Recall | 0.982 / 0.974 | — | 0.963 / 0.939 | 0.961 / 0.935 |
| ROC-AUC | 0.9974 | — | 0.9963 | 0.9960 |

> GAT retraining with Optuna-tuned hyperparameters is in progress.

### Optuna-Tuned Hyperparameters (GraphSAGE)

Best trial (macro_f1=0.9733 on validation, 50-epoch search):

```bash
uv run python train.py model=graphsage \
    train.lr=0.00124 \
    train.focal_gamma=1.0 \
    train.oversample_factor=20 \
    train.class_weight_strategy=sqrt_inverse \
    train.val_metric=macro_f1 \
    train.scheduler=cosine \
    train.patience=30 \
    model.hidden_dim=256 \
    model.num_layers=2 \
    model.dropout=0.0
```

Key findings from Optuna search:
- `sqrt_inverse` class weights consistently outperform `inverse` and `effective`
- High `oversample_factor` (9-20) is critical for rare class recall
- Lower `focal_gamma` (1.0) works better than the default 2.0
- 2-layer models outperform deeper (3-4 layer) architectures

### Per-Class F1 (GraphSAGE)

| Class | Support | Precision | Recall | F1 |
|-------|--------:|:---------:|:------:|:--:|
| Benign | 369,299 | 0.9998 | 0.9894 | 0.9946 |
| Fuzzers | 7,502 | 0.7378 | 0.8863 | 0.8053 |
| Reconnaissance | 4,330 | 0.8064 | 0.9293 | 0.8635 |
| Exploits | 11,986 | 0.8541 | 0.7319 | 0.7882 |
| Shellcode | 576 | 0.5918 | 0.9236 | 0.7214 |
| Generic | 1,527 | 0.5835 | 0.5102 | 0.5444 |
| Worms | 69 | 0.2353 | 0.6957 | 0.3516 |
| DoS | 1,587 | 0.1993 | 0.4152 | 0.2693 |
| Backdoor | 234 | 0.1266 | 0.5470 | 0.2056 |
| Analysis | 239 | 0.1226 | 0.4310 | 0.1909 |

> **Macro F1 = 0.5735.** The 4 rare attack types (DoS, Backdoor, Analysis, Worms) together have <2000 test samples (<0.5% of the dataset). Despite high recall (0.43-0.70), precision remains low due to false positives from the dominant Benign class. The model achieves strong recall across all classes (worst-case 0.42 for DoS) but struggles with inter-attack discrimination for classes with very few training samples.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | PyTorch 2.4 + PyTorch Geometric 2.6 |
| Models | GraphSAGE, GAT, E-GraphSAGE, TGAT, TGN |
| Backend | FastAPI + uvicorn |
| Frontend | Vue 3 + Vite + TypeScript + Pinia |
| Visualization | Cytoscape.js, Plotly.js |
| Reports | Jinja2 + WeasyPrint |
| Config | Hydra |
| Deploy | Docker + NVIDIA Container Toolkit |

---

## Dataset

- **NF-UNSW-NB15-v2** (~2.5M flows, 9 attack types + Benign). Place at `data/raw/NF-UNSW-NB15-v2.csv`. Source: [UNSW Sydney](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- **Demo**: `data/demo/demo_flows.csv` — 1000-flow curated subset (tracked in git).

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/training.md](docs/training.md) | Training guide, optimization, Optuna search, adversarial training, ONNX export |
| [docs/model_compare.md](docs/model_compare.md) | Detailed model architecture comparison and experiment results |
| [docs/architecture-review.md](docs/architecture-review.md) | Architecture review, open issues, risk table |
| [docs/spec.md](docs/spec.md) | Original project specification |

---

## Project Structure

```
├── src/                        # ML core
│   ├── data/                   # Data pipeline (loader, builder, dataset)
│   ├── models/                 # GraphSAGE, GAT, E-GraphSAGE, TGAT, TGN, Ensemble
│   ├── attack/                 # C-PGD, Edge Injection, GAN, Memory Poisoning
│   ├── defense/                # Adversarial training
│   ├── explain/                # GNNExplainer + temporal gradient attribution
│   └── utils/
├── app/                        # FastAPI backend
│   ├── routers/                # analyze, adversarial, report, streaming, explain
│   ├── services/               # inference, graph_builder, report_builder
│   └── templates/
├── frontend/                   # Vue 3 + Vite
│   └── src/views/              # TrafficGraph, AlertList, Timeline, Reliability, Adversarial, Explain
├── scripts/                    # Training, evaluation, export utilities
├── configs/                    # Hydra configs (model, data, attack, train)
├── data/
│   ├── demo/                   # Demo dataset (tracked)
│   └── metrics/                # Pre-computed reliability.json
├── Dockerfile                  # Backend (PyTorch + CUDA 12.4)
├── frontend/Dockerfile         # Frontend (nginx)
├── docker-compose.yml          # GPU-enabled full stack
└── pyproject.toml
```

---

## Future Work

- **Cross-dataset feature alignment** — Adapters for datasets with different feature schemas (e.g. CIC-IDS-2017, ToN-IoT)
- **E-GraphSAGE / temporal adversarial training** — Edge-feature-aware and temporal adversarial robustness (framework ready, not yet trained)
- **Lightweight temporal models** — GraphMixer / SimpleDyG as faster alternatives to TGAT/TGN

---

## References

**GNN Models**
- Hamilton et al. "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* — GraphSAGE
- Velickovic et al. "Graph Attention Networks." *ICLR 2018.* — GAT
- Xu et al. "Inductive Representation Learning on Temporal Graphs." *ICLR 2020.* — TGAT
- Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." *arXiv 2020.* — TGN

**GNN-based NIDS**
- Lo et al. "E-GraphSAGE: A GNN-based IDS for IoT." *IEEE NOMS 2022.*
- Bilot et al. "Graph Neural Networks for Intrusion Detection: A Survey." *IEEE Access 2023.*

**Adversarial Attacks on NIDS**
- Han et al. "Practical Traffic-Space Adversarial Attacks on Learning-Based NIDSs." *USENIX Security 2021.*
- Pierazzi et al. "Intriguing Properties of Adversarial ML Attacks in the Problem Space." *IEEE S&P 2020.*
- Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018.* — PGD

---

## License

MIT License. See [LICENSE](LICENSE).
