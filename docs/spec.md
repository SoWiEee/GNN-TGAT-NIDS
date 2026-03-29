# Design Specification: Graph-Aware Adversarial Robustness Framework for Network Intrusion Detection

**Version:** 0.1.0  
**Status:** Draft  
**Last Updated:** 2026-03

---

## 1. Overview

### 1.1 Problem Statement

ML-based Network Intrusion Detection Systems (NIDS) achieve high accuracy on clean benchmark data but remain vulnerable to adversarial evasion attacks. Existing adversarial example generation methods for NIDS suffer from a fundamental limitation: they operate purely in feature space without enforcing network protocol constraints, producing perturbations that are mathematically effective but physically unrealizable in actual network traffic.

Furthermore, the majority of prior work targets static ML models (DNN, RF, SVM). The adversarial robustness of **graph-based** NIDS — which model traffic as relational structures between communicating endpoints — has not been systematically studied. In particular, no prior work has examined whether **temporal graph models** (e.g., TGAT, TGN) offer inherently stronger resistance to adversarial perturbations compared to static graph models (e.g., GraphSAGE, GAT), or whether they introduce new attack surfaces through their node memory mechanisms.

### 1.2 Research Questions

1. Do temporal graph neural networks provide greater adversarial robustness than static GNNs for intrusion detection, given that attackers must simultaneously deceive both spatial and temporal feature dimensions?
2. Can a constraint-aware adversarial example generation framework produce realistic, protocol-valid evasion traffic that transfers across both static and temporal NIDS architectures?
3. Does adversarial training with constraint-enforced examples improve robustness without significant degradation of clean detection performance?

### 1.3 Proposed Solution

A three-component framework:

1. **Unified Graph Construction Pipeline** — converts NetFlow data into both static and continuous-time dynamic graph formats, enabling fair comparison across model architectures.
2. **Dual-Architecture NIDS** — static GNN (GraphSAGE/GAT) as baseline; temporal GNN (TGAT/TGN) as primary research target.
3. **Constraint-Aware Adversarial Example Generator (CAAG)** — graph-structured attack framework that enforces protocol validity, feature consistency, and semantic preservation constraints during perturbation generation.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                              │
│  NF-UNSW-NB15-v2 / NF-BoT-IoT-v2 (NetFlow CSV)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Graph Construction Pipeline                     │
│                                                                 │
│   ┌─────────────────────┐    ┌──────────────────────────────┐  │
│   │   Static Graph      │    │  Continuous-Time Dynamic     │  │
│   │   Builder           │    │  Graph Builder               │  │
│   │                     │    │                              │  │
│   │  - Time-window      │    │  - Per-flow timestamp edge   │  │
│   │    snapshot graphs  │    │  - Node memory init          │  │
│   │  - Node: IP/port    │    │  - Edge: flow features +     │  │
│   │  - Edge: flow feats │    │    temporal encoding         │  │
│   └──────────┬──────────┘    └───────────────┬──────────────┘  │
└──────────────┼────────────────────────────────┼─────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────┐          ┌─────────────────────────────┐
│   Static NIDS        │          │   Temporal NIDS             │
│                      │          │                             │
│  GraphSAGE / GAT     │          │   TGAT / TGN                │
│  (Baseline)          │          │   (Primary Target)          │
│                      │          │                             │
│  Output: edge-level  │          │  Output: edge-level         │
│  binary/multi-class  │          │  binary/multi-class         │
└──────────┬───────────┘          └──────────────┬──────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          Constraint-Aware Adversarial Example Generator          │
│                          (CAAG)                                 │
│                                                                 │
│   Attack Methods:                                               │
│   ├── PGD on Graphs (gradient-based, white-box)                 │
│   ├── GAN-based Generator (black-box)                           │
│   └── Edge Injection (topology-level perturbation)             │
│                                                                 │
│   Constraint Enforcement:                                       │
│   ├── Protocol Constraint Checker                               │
│   ├── Feature Consistency Validator                             │
│   └── Semantic Preservation Filter                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation & Hardening                        │
│                                                                 │
│   ├── Attack Success Rate (ASR) — static vs temporal           │
│   ├── Detection Rate under attack (DR@attack)                  │
│   ├── Adversarial Training Loop                                 │
│   └── Robustness-Accuracy Trade-off Analysis                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Graph Construction Pipeline

#### 3.1.1 Data Ingestion

| Parameter | Value |
|-----------|-------|
| Primary dataset | NF-UNSW-NB15-v2 |
| Secondary dataset | NF-BoT-IoT-v2 (cross-dataset generalization) |
| Input format | NetFlow CSV (pre-extracted features) |
| Node definition | (IP address, port) tuple |
| Edge definition | Directional flow from src node to dst node |

#### 3.1.2 Static Graph Builder

- **Windowing strategy:** Tumbling windows of configurable duration (default: 60s)
- **Node features:** Aggregated statistics per node per window (degree, total bytes sent/received, unique destination count)
- **Edge features:** Per-flow NetFlow features (34 features from NF-UNSW-NB15-v2)
- **Edge label:** Binary (benign=0, attack=1) and multi-class (attack category)
- **Output format:** PyTorch Geometric `Data` objects, serialized as `.pt` files

#### 3.1.3 Continuous-Time Dynamic Graph Builder

- **Event representation:** Each flow as a timestamped directed edge event
- **Temporal encoding:** Time2Vec encoding for relative time differences
- **Node memory:** Zero-initialized; updated via GRU after each event (TGN-style)
- **Output format:** PyTorch Geometric `TemporalData` objects
- **Chronological split:** Train/val/test split must be strictly chronological (no shuffling)

#### 3.1.4 Feature Normalization

- Z-score normalization fitted on training split only
- Clipping at ±3σ to reduce outlier sensitivity
- Categorical features (protocol type) one-hot encoded

---

### 3.2 NIDS Models

#### 3.2.1 Static GNN (Baseline)

**Model A: GraphSAGE**

| Parameter | Default Value |
|-----------|---------------|
| Aggregation | Mean |
| Layers | 3 |
| Hidden dim | 256 |
| Dropout | 0.3 |
| Task | Edge classification |
| Loss | Weighted cross-entropy (address class imbalance) |

**Model B: GAT**

| Parameter | Default Value |
|-----------|---------------|
| Attention heads | 4 |
| Layers | 3 |
| Hidden dim | 256 |
| Dropout | 0.3 |
| Task | Edge classification |

Both static models serve as baselines. GraphSAGE is used for initial adversarial attack development due to its simpler aggregation mechanism.

#### 3.2.2 Temporal GNN (Primary Research Target)

**Model C: TGAT**

| Parameter | Default Value |
|-----------|---------------|
| Attention heads | 2 |
| Time encoding | Learnable time2vec |
| Neighborhood sampling | Most recent k neighbors (k=20) |
| Hidden dim | 172 |
| Task | Edge classification (flow-level) |

**Model D: TGN**

| Parameter | Default Value |
|-----------|---------------|
| Memory dimension | 172 |
| Message function | MLP |
| Memory updater | GRU |
| Embedding module | Graph attention |
| Task | Edge classification |

> **Checkpoint (Month 6):** Static GNN baseline must reproduce published F1 scores on NF-UNSW-NB15-v2 (≥ 0.90 weighted F1) before temporal model development begins.

---

### 3.3 Constraint-Aware Adversarial Example Generator (CAAG)

This is the primary technical contribution of the project.

#### 3.3.1 Threat Model

| Scenario | Description |
|----------|-------------|
| White-box | Attacker has full access to model weights and architecture |
| Black-box (transfer) | Attacker trains surrogate model; transfers adversarial examples |
| Gray-box | Attacker knows feature space but not model internals |

All three scenarios are evaluated. Primary focus is white-box for robustness analysis, black-box transfer for realistic deployment assessment.

#### 3.3.2 Attack Methods

**Attack 1: Constrained PGD on Graphs (C-PGD)**

Projected Gradient Descent adapted for graph-structured NIDS with constraint projection step after each gradient update.

```
For t = 1 to T:
    x_adv ← x_adv + α · sign(∇_x L(f(G, x_adv), y_target))
    x_adv ← Project(x_adv, C)   # enforce constraint set C
    x_adv ← clip(x_adv, x - ε, x + ε)
```

Constraint projection enforces: protocol validity, feature bounds, feature co-dependency rules.

**Attack 2: Edge Injection**

Injects synthetic benign-looking edges into the graph to pollute GNN neighborhood aggregation for malicious nodes.

- Injected edges must: originate from existing legitimate IP ranges, carry statistically normal flow features, not shift the overall degree distribution beyond 2σ
- Applicable to both static and temporal graphs
- For temporal graphs: injection timing is an additional optimization variable (early injection to corrupt node memory)

**Attack 3: GAN-based Generator (black-box)**

Conditional WGAN-GP trained to generate feature vectors that: (a) fool the NIDS classifier, (b) satisfy the constraint set. Used for black-box scenario evaluation.

#### 3.3.3 Constraint Set Definition

| Constraint Type | Description | Implementation |
|----------------|-------------|----------------|
| **Protocol validity** | TCP flag combinations must be valid state sequences | Rule-based lookup table |
| **Feature co-dependency** | `flow_byts_s = tot_fwd_byts / flow_duration`; derived features must remain consistent | Algebraic recomputation post-perturbation |
| **Feature bounds** | Each feature clipped to empirically observed min/max from training data | Per-feature bound table |
| **Semantic preservation** | Attack traffic must retain attack-class characteristics (e.g., DDoS must maintain high packet rate) | Per-attack-type invariant set |
| **Degree anomaly limit** | Node degree after edge injection must remain within 3σ of training distribution | Statistical check |

#### 3.3.4 Constraint Satisfaction Rate (CSR)

All generated adversarial examples are evaluated on CSR before being used in experiments:

```
CSR = |{x_adv : all constraints satisfied}| / |{x_adv generated}|
```

Only examples with CSR = 1.0 (all constraints satisfied) are used in robustness evaluation. This is a key differentiator from prior work that reports attack success without constraint validation.

---

### 3.4 Evaluation Protocol

#### 3.4.1 Clean Performance Metrics

- Weighted F1, Precision, Recall (per-class and macro)
- ROC-AUC
- Inference latency (ms/batch) and peak GPU memory usage

#### 3.4.2 Adversarial Robustness Metrics

| Metric | Definition |
|--------|------------|
| **Attack Success Rate (ASR)** | Fraction of adversarial examples that cause misclassification |
| **DR@attack** | Detection Rate of the NIDS under adversarial conditions |
| **Robustness Gap** | Clean F1 − Adversarial F1 |
| **Transfer Rate** | ASR when adversarial examples generated on Model A are tested on Model B |

#### 3.4.3 Comparison Matrix

Every attack method is evaluated against every model:

| | GraphSAGE | GAT | TGAT | TGN |
|---|---|---|---|---|
| C-PGD (white-box) | ✓ | ✓ | ✓ | ✓ |
| Edge Injection | ✓ | ✓ | ✓ | ✓ |
| GAN (black-box) | ✓ | ✓ | ✓ | ✓ |
| After Adv. Training | ✓ | ✓ | ✓ | ✓ |

#### 3.4.4 Adversarial Training

- Generate adversarial examples using C-PGD on training set
- Mix with clean examples at ratio 1:1
- Retrain model for same number of epochs
- Evaluate on both clean and adversarial test sets
- Report robustness-accuracy trade-off curve

---

## 4. Data Pipeline

### 4.1 Directory Structure

```
data/
├── raw/
│   ├── NF-UNSW-NB15-v2.csv
│   └── NF-BoT-IoT-v2.csv
├── processed/
│   ├── static/
│   │   ├── train/          # .pt files, one per time window
│   │   ├── val/
│   │   └── test/
│   └── temporal/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
└── adversarial/
    ├── cpgd/
    ├── edge_injection/
    └── gan/
```

### 4.2 Train/Val/Test Split

Strictly chronological — no random shuffling to avoid temporal leakage:

| Split | Proportion | Notes |
|-------|-----------|-------|
| Train | 60% | First 60% of flows by timestamp |
| Val | 20% | Used for hyperparameter tuning |
| Test | 20% | Never used during development; final evaluation only |

---

## 5. Codebase Structure

```
src/
├── data/
│   ├── loader.py           # Raw CSV ingestion and cleaning
│   ├── static_builder.py   # Time-window graph construction
│   ├── temporal_builder.py # Continuous-time dynamic graph construction
│   └── constraints.py      # Constraint set definitions and validators
├── models/
│   ├── graphsage.py
│   ├── gat.py
│   ├── tgat.py
│   └── tgn.py
├── attack/
│   ├── cpgd.py             # Constrained PGD
│   ├── edge_injection.py   # Topology-level perturbation
│   ├── gan_generator.py    # WGAN-GP adversarial generator
│   └── evaluator.py        # ASR, CSR, transfer rate computation
├── defense/
│   └── adversarial_training.py
├── eval/
│   ├── metrics.py
│   └── comparison.py       # Cross-model comparison matrix
└── utils/
    ├── config.py           # Hydra-based config management
    └── logger.py
```

---

## 6. Experimental Configuration

All experiments are managed via Hydra config files under `configs/`.

### 6.1 Key Hyperparameters (Tunable)

| Parameter | Default | Search Range |
|-----------|---------|-------------|
| Time window size | 60s | {30, 60, 120} |
| GNN hidden dim | 256 | {128, 256, 512} |
| PGD steps | 40 | {10, 20, 40} |
| PGD step size α | 0.01 | {0.001, 0.01, 0.1} |
| Perturbation budget ε | 0.1 | {0.05, 0.1, 0.2} |
| Adv. training mix ratio | 1:1 | {1:3, 1:1, 3:1} |

### 6.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1080 (8GB VRAM) | NVIDIA RTX 3090 (24GB VRAM) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |
| CUDA | 11.8+ | 12.x |

---

## 7. Milestones and Deliverables

| Month | Milestone | Deliverable |
|-------|-----------|-------------|
| 1-2 | Static graph pipeline complete | `data/processed/static/` populated; unit tests passing |
| 3 | GraphSAGE/GAT baseline | Reproduced F1 ≥ 0.90 on NF-UNSW-NB15-v2; logged to W&B |
| 4-5 | Temporal graph pipeline + TGAT/TGN | `data/processed/temporal/` populated; models training |
| 6 | **Checkpoint:** Static adversarial experiments | C-PGD and Edge Injection results on GraphSAGE/GAT; ASR and CSR tables |
| 7-8 | Temporal adversarial experiments | Same attacks on TGAT/TGN; cross-model comparison matrix |
| 9 | GAN generator + adversarial training | Black-box results; robustness-accuracy trade-off curves |
| 10 | Cross-dataset validation | Results on NF-BoT-IoT-v2 |
| 11-12 | Paper + system demo | Draft manuscript; interactive demo |

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Temporal graph pipeline takes >2 months | Medium | High | Use PyG built-in `TemporalData`; avoid custom PCAP processing |
| TGAT experiments incomplete by month 9 | Medium | Medium | Month 6 hard checkpoint; TGAT is additive, not core |
| Constraint satisfaction rate too low | Low | High | Iteratively relax constraints with documented justification |
| GAN training instability | Medium | Low | Fall back to C-PGD only; GAN is supplementary |
| NF-UNSW-NB15-v2 label noise | Low | Medium | Manual inspection of ambiguous samples; document exclusions |

---

## 9. Non-Goals

The following are explicitly out of scope for this project:

- Real-time packet capture and online inference (demo uses pre-recorded flows)
- Host-based intrusion detection (audit logs, system calls)
- LLM-based analysis
- Deployment on embedded or edge hardware
- Certified adversarial robustness (formal verification)

---

## 10. References

Key papers to reproduce and compare against:

- Lo et al., "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT" (NOMS 2022)
- Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020)
- Xu et al., "Inductive Representation Learning on Temporal Graphs / TGAT" (ICLR 2020)
- Yuan et al., "A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based IDS" (Computers & Security 2024)
- Okada et al., "XAI-driven Adversarial Attacks on Network Intrusion Detectors" (EICC 2024)
- Apruzzese et al., "Modeling Realistic Adversarial Attacks against Network Intrusion Detection Systems" (ACM DTRAP 2021)
- Liu et al., "TGN-SVDD: One-Class Intrusion Detection with Dynamic Graphs" (2025)
