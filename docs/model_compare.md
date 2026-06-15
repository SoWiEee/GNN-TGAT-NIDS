# Model Selection: GNN Architecture Comparison for NIDS

**Version:** 1.1  
**Date:** 2026-06-15  
**Scope:** Current experimental comparison for GraphSAGE, GAT, TGAT, TGN, and adversarial training.

---

## 1. Research Question

The project compares static GNNs and temporal GNNs for NetFlow-based intrusion detection:

> Do temporal graph models (TGAT, TGN) provide better detection robustness or generalization than static graph models (GraphSAGE, GAT) when traffic is represented as graph-structured flow data?

The current implementation supports three comparison groups:

| Family | Models | Role |
|---|---|---|
| Static baselines | GraphSAGE, GAT | Fast training/inference and web-demo primary models |
| Temporal models | TGAT, TGN | Main research targets for continuous-time graph behavior |
| Static ensemble | GraphSAGE + GAT | Optional API inference mode through soft voting |

---

## 2. Current Experiment Summary

Latest results were found in `data/metrics/reliability.json` and Hydra training logs under `outputs/2026-06-14` and `outputs/2026-06-15`.

| Model | Test F1 | Precision | Recall | ROC-AUC | Source |
|---|---:|---:|---:|---:|---|
| GraphSAGE | 0.9712 | 0.9792 | 0.9660 | 0.9992 | `outputs/2026-06-15/16-38-10/train.log` |
| GAT | 0.9534 | n/a | n/a | n/a | `data/metrics/reliability.json` |
| TGAT | 0.9475 | 0.9632 | 0.9391 | 0.9963 | `outputs/2026-06-14/18-15-58/train.log` |
| TGN | 0.9464 | 0.9619 | 0.9350 | 0.9959 | `outputs/2026-06-15/09-56-41/train.log` |
| GraphSAGE + adversarial training | 0.9753 | 0.9803 | 0.9727 | 0.9997 | `outputs/2026-06-15/19-50-40/train.log` |
| GAT + adversarial training | 0.9622 | 0.9696 | 0.9581 | 0.9965 | `outputs/2026-06-15/19-59-32/train.log` |

Reliability file values:

| Model | Clean F1 | DR under C-PGD eps=0.1 | Delta F1 after adversarial training |
|---|---:|---:|---:|
| GraphSAGE | 0.9712 | null | 0.0041 |
| GAT | 0.9534 | null | 0.0088 |
| TGAT | null | null | null |
| TGN | null | null | null |

Interpretation:

- GraphSAGE is currently the strongest clean static model by recorded F1.
- GAT is behind GraphSAGE on clean F1 but improves more from adversarial training in the metrics file.
- TGAT and TGN have strong temporal results but do not beat the current clean GraphSAGE result.
- Adversarial training improved both static models in the latest recorded experiments.
- `dr_under_cpgd_eps01` has not been populated yet, so attack detection-rate claims should not be made from the current metrics file.

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
| Clean | 0.9534 | n/a | n/a | n/a |
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
- Adversarial training improves GAT by +0.0088 F1 in `reliability.json`, a larger delta than GraphSAGE.
- Attention does not currently translate into better aggregate detection metrics, but it remains useful for explanation-oriented UI features.

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
| 23 | 0.9464 | 0.9619 | 0.9350 | 0.9959 |

Observations:

- TGN is close to TGAT but slightly behind in the current run.
- Memory warm-up is used before final test evaluation by replaying the train split.
- The current result does not yet support the hypothesis that TGN memory beats static proxy graphs on this dataset/configuration.

---

## 5. Key Comparison

| Comparison | Current finding |
|---|---|
| GraphSAGE vs GAT | GraphSAGE is ahead on clean F1: 0.9712 vs 0.9534. |
| TGAT vs TGN | TGAT is slightly ahead: 0.9475 vs 0.9464. |
| Static vs temporal | Static GraphSAGE currently outperforms TGAT/TGN on F1. |
| Clean vs adversarial training | Static adversarial training improves F1 for both GraphSAGE and GAT. |
| Attention value | GAT/TGAT attention is useful architecturally, but not currently better than GraphSAGE on aggregate metrics. |

Current ranking by recorded test F1:

1. GraphSAGE + adversarial training: 0.9753
2. GraphSAGE clean: 0.9712
3. GAT + adversarial training: 0.9622
4. GAT clean: 0.9534
5. TGAT: 0.9475
6. TGN: 0.9464

---

## 6. Notes on Historical Results

Older logs from 2026-06-13 show much lower static-model F1 values around 0.29-0.33. Those runs used earlier preprocessing/settings, including 39 edge features and short CPU test runs. They should not be mixed with the current 42-feature static pipeline and latest checkpoints.

This document treats the 2026-06-14 and 2026-06-15 runs plus `data/metrics/reliability.json` as the current authoritative result set.

---

## 7. Recommended Next Experiments

1. Populate `dr_under_cpgd_eps01` for GraphSAGE and GAT with the current checkpoints.
2. Add TGAT/TGN entries to `data/metrics/reliability.json` so the frontend reliability panel reflects temporal results.
3. Run clean GAT evaluation again and store precision/recall/ROC-AUC alongside the F1 currently in `reliability.json`.
4. Compare static 120-second offline training against 60-second web inference windows to quantify the deployment mismatch.
5. Run ensemble evaluation and record whether soft voting improves over GraphSAGE alone.
6. Add macro F1 and per-class recall, because weighted F1 can hide poor minority attack-class behavior.
7. Run repeated seeds for the top models to separate architecture effects from seed variance.

---

## 8. Model Roadmap

Near term:

- Keep GraphSAGE as the main static baseline and web-demo default.
- Keep GAT for attention-based interpretation and adversarial-training comparison.
- Keep TGAT/TGN as research models, but avoid claiming temporal superiority until stronger results appear.
- Add reliability metrics for temporal models before using them in the UI reliability panel.

Future candidates:

- E-GraphSAGE for edge-feature-aware static message passing.
- GraphMixer/SimpleDyG for faster temporal aggregation.
- DyGFormer for co-occurrence-aware temporal modeling.
