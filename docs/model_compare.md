# Model Selection: GNN Architecture Comparison for NIDS

**Version:** 1.2  
**Date:** 2026-06-16  
**Scope:** Current experimental comparison for GraphSAGE, GAT, E-GraphSAGE, TGAT, TGN, and adversarial training.

---

## 1. Research Question

The project compares static GNNs and temporal GNNs for NetFlow-based intrusion detection:

> Do temporal graph models (TGAT, TGN) provide better detection robustness or generalization than static graph models (GraphSAGE, GAT) when traffic is represented as graph-structured flow data?

The current implementation supports three comparison groups:

| Family | Models | Role |
|---|---|---|
| Static baselines | GraphSAGE, GAT | Fast training/inference and web-demo primary models |
| Edge-feature-aware | E-GraphSAGE | Edge features in message passing; NIDS-tailored |
| Temporal models | TGAT, TGN | Main research targets for continuous-time graph behavior |
| Static ensemble | GraphSAGE + GAT + E-GraphSAGE | Optional API inference mode through soft/weighted voting |

---

## 2. Current Experiment Summary

Latest results were found in `data/metrics/reliability.json` and Hydra training logs under `outputs/2026-06-14`, `outputs/2026-06-15`, and `outputs/2026-06-16`.

| Model | Test F1 | Precision | Recall | ROC-AUC | Macro F1 | Source |
|---|---:|---:|---:|---:|---:|---|
| GraphSAGE | 0.9712 | 0.9792 | 0.9660 | 0.9992 | 0.4657 | `outputs/2026-06-15/16-38-10/train.log` |
| GAT | 0.9534 | 0.9729 | 0.9433 | 0.9963 | 0.3164 | `data/metrics/reliability.json` |
| E-GraphSAGE | 0.9708 | 0.9784 | 0.9665 | 0.9991 | 0.4681 | `outputs/2026-06-16` |
| TGAT | 0.9475 | 0.9632 | 0.9391 | 0.9963 | 0.3643 | `outputs/2026-06-14/18-15-58/train.log` |
| TGN | 0.9463 | 0.9610 | 0.9351 | 0.9960 | 0.3438 | `data/metrics/reliability.json` |
| Ensemble (3-model) | 0.9700 | 0.9794 | 0.9650 | 0.9992 | 0.4510 | `data/metrics/reliability.json` |
| GraphSAGE + adversarial training | 0.9753 | 0.9803 | 0.9727 | 0.9997 | — | `outputs/2026-06-15/19-50-40/train.log` |
| GAT + adversarial training | 0.9622 | 0.9696 | 0.9581 | 0.9965 | — | `outputs/2026-06-15/19-59-32/train.log` |

Reliability file values:

| Model | Clean F1 | DR under C-PGD eps=0.1 | Scope | Delta F1 after adversarial training |
|---|---:|---:|---|---:|
| GraphSAGE | 0.9712 | 1.0000 | full static test, 3312 windows / 32804 attack edges | 0.0041 |
| GAT | 0.9534 | 1.0000 | full static test, 3312 windows / 35113 attack edges | 0.0087 |
| E-GraphSAGE | 0.9708 | 1.0000 | full static test, 3312 windows / 32993 attack edges | null |
| TGAT | 0.9475 | 1.0000 | sampled temporal, 32 test batches / 256 warm-up batches | null |
| TGN | 0.9463 | 0.9989 | sampled temporal, 32 test batches / 256 warm-up batches | null |
| Ensemble (3-model) | 0.9700 | null | clean-only soft vote (GraphSAGE + GAT + E-GraphSAGE) | null |

Interpretation:

- GraphSAGE is currently the strongest clean static model by recorded F1 (0.9712).
- E-GraphSAGE is nearly identical to GraphSAGE on weighted F1 (0.9708) after only 30 epochs, and achieves the highest Macro F1 (0.4681) of any individual model, suggesting better minority-class handling from edge-feature-aware message passing.
- GAT is behind GraphSAGE on clean F1 but improves more from adversarial training in the metrics file.
- The 3-model ensemble (GraphSAGE + GAT + E-GraphSAGE) reaches 0.9700 F1 with learned validation-based weights, improving over the old 2-model ensemble (0.9670).
- TGAT and TGN have strong temporal results but do not beat the current static baselines.
- Adversarial training improved both static models in the latest recorded experiments.
- Static `dr_under_cpgd_eps01` is now a full-test sweep. Temporal DR is implemented with constrained message-space C-PGD, but the recorded run is bounded because full temporal replay is expensive.
- E-GraphSAGE has full C-PGD DR = 1.0000 (full 3312-window test sweep), showing robustness comparable to GraphSAGE and GAT.

---

## 3. Static Models

### 3.1 GraphSAGE

Configuration:

```text
num_layers: 3
hidden_dim: 256
dropout: 0.3
aggregation: mean
loss: FocalLoss(gamma=2.0) with class weights
```

Current results:

| Run | Test F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Clean | 0.9712 | 0.9792 | 0.9660 | 0.9992 |
| C-PGD adversarial training | 0.9753 | 0.9803 | 0.9727 | 0.9997 |

Adversarial training setup from log:

```text
epsilon = 0.100
steps = 10
ratio = 0.30
epochs = 30
best val_f1 = 0.9764 at epoch 30
test f1 = 0.9753
```

Observations:

- The proxy-node static graph construction is now effective enough for high F1.
- GraphSAGE is the best currently recorded model overall.
- Adversarial training gives a small but consistent improvement: +0.0041 F1 in `reliability.json`.

### 3.2 GAT

Configuration:

```text
num_layers: 3
hidden_dim: 256
num_heads: 4
dropout: 0.3
loss: FocalLoss(gamma=2.0) with class weights
```

Current results:

| Run | Test F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Clean | 0.9534 | 0.9729 | 0.9433 | 0.9963 |
| C-PGD adversarial training | 0.9622 | 0.9696 | 0.9581 | 0.9965 |

Adversarial training setup from log:

```text
epsilon = 0.100
steps = 10
ratio = 0.30
epochs = 30
best val_f1 = 0.9647 at epoch 27
test f1 = 0.9622
```

Observations:

- GAT remains competitive but is below GraphSAGE in the current result set.
- Adversarial training improves GAT by +0.0087 F1 in `reliability.json`, a larger delta than GraphSAGE.
- Attention does not currently translate into better aggregate detection metrics, but it remains useful for explanation-oriented UI features.

### 3.3 E-GraphSAGE

Configuration:

```text
num_layers: 3
hidden_dim: 256
dropout: 0.3
aggregation: mean
loss: FocalLoss(gamma=2.0) with class weights
```

Current 30-epoch result:

| Run | Test F1 | Precision | Recall | ROC-AUC | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.9708 | 0.9784 | 0.9665 | 0.9991 | 0.4681 |

Architecture:

E-GraphSAGE (Lo et al., IEEE NOMS 2022) modifies message passing to include edge features directly: `m_{u→v} = W·concat(h_u, e_{uv})`. This is a natural fit for NIDS where edge attributes (flow features) carry the primary detection signal. The implementation uses PyG's `MessagePassing` base with a custom `message()` that concatenates source node embeddings with edge attributes before projection.

Observations:

- E-GraphSAGE matches GraphSAGE on weighted F1 (0.9708 vs 0.9712) after only 30 epochs.
- Achieves the highest Macro F1 (0.4681) of any individual model, beating GraphSAGE (0.4657), indicating better minority-class detection.
- Recall (0.9665) slightly exceeds GraphSAGE (0.9660), consistent with edge-feature-aware aggregation catching more attack patterns.
- C-PGD robustness is identical to GraphSAGE (DR = 1.0000 on full test).
- Adversarial training has not yet been evaluated for E-GraphSAGE.

---

## 4. Temporal Models

### 4.1 TGAT

Configuration:

```text
hidden_dim: 172
heads: 2
n_neighbors: 20
loss: FocalLoss(gamma=2.0) with class weights
```

Current 30-epoch result:

| Best val epoch | Test F1 | Precision | Recall | ROC-AUC |
|---:|---:|---:|---:|---:|
| 23 | 0.9475 | 0.9632 | 0.9391 | 0.9963 |

Observations:

- TGAT trains successfully on temporal `TemporalData` and reaches strong test ROC-AUC.
- Test F1 is slightly lower than the current static baselines.
- Training time is materially higher than static models because temporal batches and memory warm-up/replay dominate runtime.

### 4.2 TGN

Configuration:

```text
memory_dim: 172
time_dim: 64
hidden_dim: 256
num_neighbors: 20
embedding_module: graph_attention
dropout: 0.1
loss: FocalLoss(gamma=2.0) with class weights
```

Current 30-epoch result:

| Best val epoch | Test F1 | Precision | Recall | ROC-AUC |
|---:|---:|---:|---:|---:|
| 23 | 0.9463 | 0.9610 | 0.9351 | 0.9960 |

Observations:

- TGN is close to TGAT but slightly behind in the current run.
- Memory warm-up is used before final test evaluation by replaying the train split.
- The current result does not yet support the hypothesis that TGN memory beats static proxy graphs on this dataset/configuration.

---

## 5. Key Comparison

| Comparison | Current finding |
|---|---|
| GraphSAGE vs GAT | GraphSAGE is ahead on clean F1: 0.9712 vs 0.9534. |
| GraphSAGE vs E-GraphSAGE | Nearly identical weighted F1 (0.9712 vs 0.9708). E-GraphSAGE has higher Macro F1 (0.4681 vs 0.4657). |
| TGAT vs TGN | TGAT is slightly ahead: 0.9475 vs 0.9463. |
| Static vs temporal | Static models currently outperform TGAT/TGN on F1. |
| Clean vs adversarial training | Static adversarial training improves F1 for both GraphSAGE and GAT. |
| 3-model ensemble | Learned-weight voting (GraphSAGE + GAT + E-GraphSAGE) reaches 0.9700 F1, above the old 2-model ensemble (0.9670). |
| Attention value | GAT/TGAT attention is useful architecturally, but not currently better than GraphSAGE on aggregate metrics. |
| Edge-feature awareness | E-GraphSAGE's direct edge-feature message passing yields the best Macro F1, suggesting edge features help with minority attack classes. |

Current ranking by recorded test F1:

1. GraphSAGE + adversarial training: 0.9753
2. GraphSAGE clean: 0.9712
3. E-GraphSAGE clean: 0.9708
4. Ensemble (3-model): 0.9700
5. GAT + adversarial training: 0.9622
6. GAT clean: 0.9534
7. TGAT: 0.9475
8. TGN: 0.9463

---

## 6. Notes on Historical Results

Older logs from 2026-06-13 show much lower static-model F1 values around 0.29-0.33. Those runs used earlier preprocessing/settings, including 39 edge features and short CPU test runs. They should not be mixed with the current 42-feature static pipeline and latest checkpoints.

This document treats the 2026-06-14 and 2026-06-15 runs plus `data/metrics/reliability.json` as the current authoritative result set.

---

## 7. Recommended Next Experiments

1. Run a full temporal C-PGD sweep if sufficient compute is available; the current temporal DR is a bounded run with 256 warm-up batches and 32 attacked test batches.
2. Add feature-alignment or source-scaler reuse for cross-dataset validation. The tracked demo CSV run is skipped because its 39-feature schema does not match the current 42-feature checkpoints.
3. Compare static 120-second offline training against 60-second web inference windows to quantify the deployment mismatch.
4. Run adversarial training on E-GraphSAGE to see if edge-feature-aware message passing yields stronger robust improvement than standard GraphSAGE.
5. Evaluate GraphMixer / SimpleDyG as lightweight temporal alternatives to TGAT/TGN.

Completed (previously listed):

- ~~Per-class recall and confusion matrices~~ — Implemented in `compute_reliability_metrics.py` (per_class field in reliability.json).
- ~~Multi-seed evaluation~~ — Implemented in `scripts/multi_seed_eval.py` with configurable seed list and statistical aggregation.
- ~~Validation-based ensemble weighting~~ — Implemented in `EnsembleModel.from_validation()` with learned weights (GraphSAGE: 0.335, GAT: 0.330, E-GraphSAGE: 0.335).

---

## 8. Model Roadmap

Near term:

- Keep GraphSAGE as the main static baseline and web-demo default.
- E-GraphSAGE is now available as a near-equivalent alternative with better minority-class metrics; consider as default if Macro F1 matters more than weighted F1.
- Keep GAT for attention-based interpretation and adversarial-training comparison.
- Keep TGAT/TGN as research models, but avoid claiming temporal superiority until stronger results appear.

Future candidates:

- GraphMixer/SimpleDyG for faster temporal aggregation.
- DyGFormer for co-occurrence-aware temporal modeling.
