# GNN-TGAT-NIDS

Graph-Aware Adversarial Robustness Framework for Network Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.x-green.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-lightgrey.svg)]()

---

## What Is This?

GARF-NIDS is a research framework that studies the adversarial robustness of **graph neural network-based intrusion detection systems (NIDS)**. It asks a question that existing literature has largely ignored:

> *When an attacker tries to evade a network intrusion detector, does the temporal structure of the traffic graph make the detector harder or easier to fool?*

Most prior adversarial ML work on NIDS either targets conventional classifiers (DNN, RF) or generates perturbations that look effective on paper but are physically unrealizable — they violate network protocol constraints or break the semantic meaning of the attack traffic. This project directly addresses both gaps.

---

## Research Problem

ML-based NIDS achieves high accuracy on clean benchmark data, but faces two underexplored vulnerabilities:

**1. Adversarial evasion via graph-structure manipulation**
Attackers can subtly modify traffic features or inject synthetic flows to shift a malicious flow's neighborhood representation, causing a GNN-based detector to misclassify it as benign.

**2. Unrealistic adversarial examples in prior work**
Existing adversarial attacks on NIDS modify features freely without respecting network constraints — the resulting "adversarial traffic" cannot be reproduced in real networks because it violates protocol state machines or contains internally inconsistent flow statistics.

---

## Research Contributions

**Contribution 1 — Constraint-Aware Adversarial Example Generator (CAAG)**
A perturbation framework that enforces three classes of constraints during adversarial example generation:
- *Protocol validity:* TCP flag sequences must follow legal state transitions
- *Feature co-dependency:* Derived features (e.g., `flow_byts_s`) are recomputed after perturbation to maintain algebraic consistency
- *Semantic preservation:* Attack traffic retains its attack-class characteristics after perturbation

**Contribution 2 — Static vs. Temporal GNN Robustness Comparison**
The first systematic evaluation of whether temporal graph models (TGAT, TGN) offer stronger adversarial robustness than static models (GraphSAGE, GAT), or whether their node memory mechanisms introduce new attack surfaces.

**Contribution 3 — Edge Injection as a Graph-Level Attack**
A topology-level perturbation strategy that injects synthetic flows to corrupt GNN neighborhood aggregation, with timing-aware variants for temporal models.

---

## System Overview

```
NetFlow Data (NF-UNSW-NB15-v2)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
Static Graph Builder                  Dynamic Graph Builder
(time-window snapshots)          (continuous-time, per-flow events)
        │                                      │
        ▼                                      ▼
 GraphSAGE / GAT                        TGAT / TGN
  (Baseline Models)                  (Primary Research Target)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
     Constraint-Aware Adversarial Example Generator
      ├── Constrained PGD (white-box)
      ├── Edge Injection (topology-level)
      └── WGAN-GP Generator (black-box)
                       │
                       ▼
         Adversarial Training & Evaluation
      ├── Attack Success Rate (ASR)
      ├── Detection Rate under attack
      ├── Robustness Gap (clean F1 − adversarial F1)
      └── Cross-model transfer analysis
```

---

## Why This Matters

**For defenders:** Understanding exactly how and why GNN-based detectors fail under adversarial conditions is a prerequisite for building systems that are deployable in real adversarial environments, not just controlled benchmarks.

**For the research community:** The constraint-satisfaction gap in existing work is significant. Published attack success rates are systematically overestimated because unconstrained perturbations are counted as successful even when they could not exist in real traffic. GARF-NIDS provides a reproducible framework for constraint-enforced evaluation.

**For the industry:** SOC teams deploying ML-based NIDS need to know the threat surface of their models before adversaries discover it. This framework enables red-team evaluation of graph-based detectors.

---

## Datasets

| Dataset | Flows | Attack Types | Format | Source |
|---------|-------|-------------|--------|--------|
| NF-UNSW-NB15-v2 | ~2.5M | 9 | NetFlow CSV | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| NF-BoT-IoT-v2 | ~3.6M | 4 | NetFlow CSV | [UNSW](https://research.unsw.edu.au/projects/bot-iot-dataset) |

Both datasets are freely available. Place downloaded files under `data/raw/` as described in the [Data Setup](#data-setup) section.

> NF-BoT-IoT-v2 is used exclusively for cross-dataset generalization evaluation; all model development and hyperparameter tuning is conducted on NF-UNSW-NB15-v2.

---

## Requirements

- Python 3.10+
- CUDA 11.8+ (CPU-only mode available but not recommended for temporal models)
- NVIDIA GPU with ≥ 8GB VRAM (≥ 24GB recommended for TGN full-batch training)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SoWiEee/GNN-TGAT-IDS.git
cd GNN-TGAT-IDS
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

> If you encounter issues with PyG installation, refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and match your CUDA version.

### 4. Verify Installation

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

---

## Data Setup

### Download Datasets

```bash
# NF-UNSW-NB15-v2
# Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Place at: data/raw/NF-UNSW-NB15-v2.csv

# NF-BoT-IoT-v2
# Download from: https://research.unsw.edu.au/projects/bot-iot-dataset
# Place at: data/raw/NF-BoT-IoT-v2.csv
```

### Preprocess

```bash
# Build static graphs (time-window snapshots)
python src/data/static_builder.py --config configs/data/static_default.yaml

# Build temporal graphs (continuous-time dynamic graphs)
python src/data/temporal_builder.py --config configs/data/temporal_default.yaml
```

Processed files will be saved under `data/processed/static/` and `data/processed/temporal/`.

---

## Usage

### Train Baseline Models

```bash
# GraphSAGE
python train.py model=graphsage data=static

# GAT
python train.py model=gat data=static

# TGAT
python train.py model=tgat data=temporal

# TGN
python train.py model=tgn data=temporal
```

### Generate Adversarial Examples

```bash
# Constrained PGD (white-box, against GraphSAGE)
python attack.py method=cpgd model=graphsage epsilon=0.1 steps=40

# Edge Injection (against TGAT)
python attack.py method=edge_injection model=tgat n_inject=50

# GAN-based (black-box)
python attack.py method=gan target_model=graphsage
```

### Evaluate Robustness

```bash
# Full comparison matrix (all models × all attacks)
python eval/comparison.py --config configs/eval/full_matrix.yaml
```

### Adversarial Training

```bash
python defense/adversarial_training.py model=graphsage attack=cpgd mix_ratio=1.0
```

---

## Project Structure

```
garf-nids/
├── configs/                    # Hydra configuration files
│   ├── data/
│   ├── model/
│   ├── attack/
│   └── eval/
├── data/
│   ├── raw/                    # Downloaded datasets (not tracked by git)
│   ├── processed/
│   │   ├── static/
│   │   └── temporal/
│   └── adversarial/
├── docs/
│   └── spec.md                 # Full design specification
├── src/
│   ├── data/                   # Graph construction pipeline
│   ├── models/                 # GNN implementations
│   ├── attack/                 # CAAG — adversarial example generators
│   ├── defense/                # Adversarial training
│   ├── eval/                   # Metrics and comparison
│   └── utils/
├── notebooks/                  # Exploratory analysis
├── tests/
├── requirements.txt
└── README.md
```

---

## Reproducing Key Results

After training and attacking, the main results table is generated by:

```bash
python eval/comparison.py --output results/main_table.csv
```

Expected output format:

| Model | Clean F1 | ASR (C-PGD) | ASR (Edge Inj.) | ASR (GAN) | F1 after AdvTrain |
|-------|----------|-------------|-----------------|-----------|-------------------|
| GraphSAGE | — | — | — | — | — |
| GAT | — | — | — | — | — |
| TGAT | — | — | — | — | — |
| TGN | — | — | — | — | — |

*(Results will be populated as experiments complete.)*

---

## Roadmap

- [x] Repository structure and specification
- [ ] Static graph pipeline (NF-UNSW-NB15-v2)
- [ ] GraphSAGE / GAT baseline training
- [ ] Temporal graph pipeline
- [ ] TGAT / TGN training
- [ ] Constrained PGD implementation
- [ ] Edge Injection implementation
- [ ] Month-6 checkpoint evaluation
- [ ] GAN-based generator
- [ ] Adversarial training
- [ ] Cross-dataset validation (NF-BoT-IoT-v2)
- [ ] Paper draft

---

## Citation

If you use this framework or build on the ideas in your research, please cite:

```bibtex
@misc{gnn-tgat-nids-2026,
  title     = {GNN-TGAT-NIDS: Graph-Aware Adversarial Robustness Framework for Network Intrusion Detection},
  author    = {[SoWiEee]},
  year      = {2026},
  url       = {https://github.com/SoWiEee/gnn-tgat-nids}
}
```

---

## Related Work

- Lo et al., "E-GraphSAGE: A GNN-based IDS for IoT" (IEEE NOMS 2022)
- Xu et al., "Inductive Representation Learning on Temporal Graphs" (ICLR 2020) — TGAT
- Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020) — TGN
- Yuan et al., "A Simple Framework to Enhance Adversarial Robustness of DL-based IDS" (Computers & Security 2024)
- Okada et al., "XAI-driven Adversarial Attacks on Network Intrusion Detectors" (EICC 2024)
- Apruzzese et al., "Modeling Realistic Adversarial Attacks against NIDS" (ACM DTRAP 2021)

---

## License

MIT License. See [LICENSE](https://github.com/SoWiEee/GNN-TGAT-IDS/blob/main/LICENSE) for details.
