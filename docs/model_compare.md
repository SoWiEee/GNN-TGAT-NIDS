# 模型選型：GNN 架構於 NIDS 之比較

**版本：** 3.0
**日期：** 2026-07-10
**範圍：** GraphSAGE、GAT、TGAT、TGN、DyGFormer 在 NF-UNSW-NB15-v2 上的完整實驗比較，含 Optuna 超參數搜索、靜態與時序模型對抗式訓練。

---

## 1. 研究問題

本專案比較靜態與時序圖神經網路在 NetFlow 入侵偵測上的表現：

> 時序圖模型（TGAT、TGN）在流量以圖結構表示時，是否能提供比靜態圖模型（GraphSAGE、GAT）更好的偵測魯棒性或泛化能力？

目前實作支援以下比較組：

| 類別 | 模型 | 定位 |
|------|------|------|
| 靜態基線 | GraphSAGE, GAT | 快速訓練/推論，Web Demo 主要模型 |
| 時序模型 | TGAT, TGN | 連續時間圖行為的研究目標 |
| 時序進階 | DyGFormer | Transformer-based 共現感知時序模型 |
| 靜態集成 | GraphSAGE + GAT | 加權軟投票的 API 推論模式 |

> E-GraphSAGE 模型程式碼已實作（`src/models/egraphsage.py`），但目前無正式 checkpoint 與評估指標，列入未來工作。

---

## 2. 實驗結果總覽

所有模型超參數均由 Optuna（TPE sampler + MedianPruner）搜索最佳化。評估指標來自 `data/metrics/reliability.json`。

### 2.1 Clean 效能

靜態模型在 ~397K flows / 3312 windows 的測試集上評估；時序模型在 ~508K 事件上評估。

| 模型 | Weighted F1 | Macro F1 | Precision | Recall | ROC-AUC |
|------|:----------:|:--------:|:---------:|:------:|:-------:|
| **GraphSAGE** | **0.9773** | **0.5735** | 0.9818 | 0.9742 | 0.9974 |
| GAT | 0.9585 | 0.4036 | 0.9686 | 0.9526 | 0.9932 |
| TGAT | 0.8963 | 0.3127 | 0.9478 | 0.8618 | 0.9912 |
| TGN | 0.9463 | 0.3438 | 0.9610 | 0.9351 | 0.9960 |
| DyGFormer | 0.9249 | 0.3166 | 0.9692 | 0.8950 | 0.9862 |
| Ensemble | 0.9767 | 0.5620 | 0.9819 | 0.9735 | 0.9970 |

### 2.2 對抗魯棒性（C-PGD）

| 模型 | C-PGD DR (ε=0.1, 40 步) | 測試範圍 | 攻擊邊數 |
|------|:------------------------:|----------|:--------:|
| GraphSAGE | 1.0000 | 全量靜態測試，3312 windows | 31,869 |
| GAT | 1.0000 | 全量靜態測試，3312 windows | 35,910 |
| TGAT | 0.9684 | 全量時序測試 | 110,352 |
| TGN | 0.9965 | 全量時序測試 | 110,491 |

### 2.3 對抗式訓練（Adversarial Training）

使用 C-PGD 對抗式訓練（ε=0.1, 10 步, ratio=0.3）：

| 模型 | Clean F1 | Adv F1 | ΔF1 | Adv ROC-AUC | Adv Macro F1 |
|------|:--------:|:------:|:---:|:-----------:|:------------:|
| GraphSAGE | 0.9773 | 0.9765 | **-0.0008** | — | — |
| GAT | 0.9585 | 0.9699 | **+0.0114** | — | — |
| TGN | 0.9463 | 0.7635 | -0.1828 | 0.8470 | 0.1739 |
| TGAT | 0.8963 | — | — | — | — |

> 靜態模型在對抗式訓練下幾乎無損（ΔF1 在 ±0.012 以內）。TGN 的 GRU-based memory 對擾動梯度敏感，對抗訓練後 F1 下降 0.1828。TGAT 對抗式訓練因 GPU 資源限制暫緩。

---

## 3. 靜態模型

### 3.1 GraphSAGE

**Optuna 最佳組態**（50 trials, val macro_f1=0.9733）：

```text
num_layers: 2
hidden_dim: 256
dropout: 0.0
aggregation: mean
lr: 0.00124
focal_gamma: 1.0
oversample_factor: 20
class_weight_strategy: sqrt_inverse
scheduler: cosine
```

測試結果：

| 指標 | 數值 |
|------|:----:|
| Weighted F1 | 0.9773 |
| Macro F1 | 0.5735 |
| Precision | 0.9818 |
| Recall | 0.9742 |
| ROC-AUC | 0.9974 |
| C-PGD DR | 1.0000 |

觀察：
- GraphSAGE 在所有指標上均為最佳個別模型。
- Optuna 發現淺層（2 層）優於預設的 3 層，且 dropout=0.0 最佳。
- 高 `oversample_factor=20` 對稀有攻擊類別（Worms, Backdoor, Analysis）的召回率至關重要。
- `sqrt_inverse` 類別權重策略優於 `inverse` 和 `effective`。

### 3.2 GAT

**Optuna 最佳組態**（13 trials, val macro_f1=0.9674）：

```text
num_layers: 3
hidden_dim: 128
num_heads: 4
dropout: 0.4
lr: 0.00288
focal_gamma: 1.0
oversample_factor: 3
class_weight_strategy: sqrt_inverse
scheduler: cosine
```

測試結果：

| 指標 | 數值 |
|------|:----:|
| Weighted F1 | 0.9585 |
| Macro F1 | 0.4036 |
| Precision | 0.9686 |
| Recall | 0.9526 |
| ROC-AUC | 0.9932 |
| C-PGD DR | 1.0000 |

觀察：
- GAT 在 weighted F1 上落後 GraphSAGE，但 attention 機制對可解釋性 UI 有價值。
- Optuna 選擇較小的 hidden_dim=128（vs GraphSAGE 的 256）和較高的 dropout=0.4。
- 較低的 oversample_factor=3 暗示 GAT 對過採樣的敏感度不同。

---

## 4. 時序模型

### 4.1 TGAT

**Optuna 最佳組態**（15 trials × 15 epochs, 20% subsample, val macro_f1=0.1917）：

```text
hidden_dim: 256
heads: 4
n_neighbors: 20
lr: 0.0042
focal_gamma: 1.0
oversample_factor: 13
class_weight_strategy: effective
batch_size: 200
```

測試結果（pre-Optuna checkpoint, hidden_dim=172）：

| 指標 | 數值 |
|------|:----:|
| Weighted F1 | 0.8963 |
| Macro F1 | 0.3127 |
| Precision | 0.9478 |
| Recall | 0.8618 |
| ROC-AUC | 0.9912 |
| C-PGD DR | 0.9684 |
| Adv ΔF1 | — |

觀察：
- Optuna 調參後的 hidden_dim=256 版本在全量資料上未顯著優於 pre-Optuna（hidden_dim=172），故保留 pre-Optuna checkpoint。
- 在 Generic 類上表現突出（F1=0.9771），因 Generic 佔時序資料的大量事件。
- Backdoor 類完全無法偵測（F1=0.0），受限於稀有類別的時序稀疏性。
- 對抗式訓練因 GPU 資源限制暫緩。

### 4.2 DyGFormer

**手動調參最佳組態**（Optuna subsample 不可行，手動消融實驗）：

```text
hidden_dim: 172
n_layers: 2
n_heads: 2
patch_size: 4
n_neighbors: 20
time_dim: 64
dropout: 0.15
lr: 0.0005
focal_gamma: 1.0
oversample_factor: 13
class_weight_strategy: effective
scheduler: cosine
batch_size: 200
```

測試結果：

| 指標 | 數值 |
|------|:----:|
| Weighted F1 | 0.9249 |
| Macro F1 | 0.3166 |
| Precision | 0.9692 |
| Recall | 0.8950 |
| ROC-AUC | 0.9862 |

Per-class F1：

| 類別 | Precision | Recall | F1 | Support |
|------|:---------:|:------:|:--:|:-------:|
| Benign | 1.0000 | 0.9379 | 0.9680 | 408,913 |
| Generic | 0.9942 | 0.9212 | 0.9563 | 68,441 |
| Fuzzers | 0.3490 | 0.4049 | 0.3749 | 6,711 |
| Exploits | 0.8993 | 0.2154 | 0.3476 | 12,849 |
| Reconnaissance | 0.2685 | 0.3782 | 0.3141 | 4,064 |
| Shellcode | 0.0370 | 0.8636 | 0.0709 | 440 |
| Analysis | 0.0275 | 0.6106 | 0.0525 | 678 |
| DoS | 0.0695 | 0.0376 | 0.0488 | 5,160 |
| Worms | 0.0119 | 0.8077 | 0.0235 | 52 |
| Backdoor | 0.0048 | 0.0869 | 0.0090 | 702 |

觀察：
- DyGFormer（Yu et al., NeurIPS 2023）使用 Transformer 架構搭配共現編碼和時序 patching，為無狀態設計（無持久記憶體）。
- **Optuna subsample 不可行**：DyGFormer 的 Transformer 架構在 20% 或 50% 子採樣下完全無法收斂（macro_f1<0.004），因為共現編碼和時序 patching 需要完整的鄰居歷史。這與 TGAT/TGN 不同（它們可用 subsample 搜索）。
- **focal_gamma 敏感**：gamma<1.0 導致梯度爆炸（NaN），因 focal loss 梯度中 `(1-pt)^(gamma-1)` 項在 gamma<1 時會發散。gamma=1.5 收斂但效能退化（macro_f1=0.2432）。gamma=1.0 為最佳。
- **hidden_dim=256 導致 NaN**：模型過大（1.48M 參數 vs 172 的 807K），在 lr=0.0005 下 epoch 2 即發散。
- 在 Fuzzers、Reconnaissance 類別上表現相對於其他時序模型較好。
- 整體效能介於 TGAT 和 TGN 之間。

### 4.3 TGN

**Optuna 最佳組態**（15 trials × 15 epochs, 20% subsample, val macro_f1=0.1883）：

```text
hidden_dim: 256
memory_dim: 100
num_neighbors: 30
lr: 0.0027
focal_gamma: 1.0
oversample_factor: 12
class_weight_strategy: effective
batch_size: 200
```

測試結果：

| 指標 | 數值 |
|------|:----:|
| Weighted F1 | 0.9463 |
| Macro F1 | 0.3438 |
| Precision | 0.9610 |
| Recall | 0.9351 |
| ROC-AUC | 0.9960 |
| C-PGD DR | 0.9965 |
| Adv ΔF1 | -0.1828 |

觀察：
- TGN 的 GRU-based memory 在對抗訓練下 F1 降幅顯著（ΔF1=-0.1828），macro_f1 從 0.3438 降至 0.1739。
- 對抗訓練後 ROC-AUC 降至 0.8470，顯示 memory 更新機制對擾動梯度敏感。
- Optuna 選擇 num_neighbors=30（TGAT 為 20），暗示 TGN 受益於更大的鄰居取樣。
- 同樣受 20% subsampling 限制。

---

## 5. 關鍵比較

| 比較 | 發現 |
|------|------|
| GraphSAGE vs GAT | GraphSAGE 在 weighted F1（0.9773 vs 0.9585）和 macro F1（0.5735 vs 0.4036）均領先。 |
| TGAT vs TGN vs DyGFormer | TGN clean F1（0.9463）最佳，DyGFormer（0.9249）居中，TGAT（0.8963）最低。DyGFormer 無狀態設計較 TGN 穩定但效能略低。 |
| 靜態 vs 時序 | 靜態模型在 clean F1 和對抗穩定性上均優於時序模型。時序模型在 Generic 類上有優勢。 |
| 對抗式訓練 | 靜態模型幾乎無損（ΔF1 ±0.012 以內）；TGN 降幅較大（ΔF1=-0.1828）。 |
| Ensemble | 2-model 加權投票（GraphSAGE 0.504 + GAT 0.496）達到 F1=0.9767。 |
| 類別權重策略 | 靜態模型偏好 `sqrt_inverse`；時序模型偏好 `effective`。 |
| Focal gamma | 四個模型一致發現 γ=1.0 優於預設的 γ=2.0。 |

### 效能排名（Weighted F1）

1. GraphSAGE: 0.9773
2. Ensemble: 0.9767
3. GAT: 0.9585
4. TGN: 0.9463
5. DyGFormer: 0.9249
6. TGAT: 0.8963

### 效能排名（Macro F1）

1. GraphSAGE: 0.5735
2. Ensemble: 0.5620
3. GAT: 0.4036
4. TGN: 0.3438
5. DyGFormer: 0.3166
6. TGAT: 0.3127

---

## 6. Optuna 搜索空間與發現

### 共用搜索空間

| 參數 | 範圍 |
|------|------|
| lr | 1e-4 ~ 1e-2（log scale） |
| focal_gamma | 0.5 ~ 3.0（步長 0.5） |
| oversample_factor | 1 ~ 20 |
| weight_strategy | inverse, sqrt_inverse, effective |

### 靜態模型搜索空間

| 參數 | 範圍 |
|------|------|
| hidden_dim | 128, 256, 512 |
| num_layers | 2 ~ 4 |
| dropout | 0.0 ~ 0.5（步長 0.1） |
| batch_size | 16, 32, 64 |
| num_heads（GAT） | 2, 4, 8 |
| aggregation（GraphSAGE） | mean, max |

### 時序模型搜索空間

| 參數 | 範圍 |
|------|------|
| hidden_dim | 64, 128, 172, 256 |
| n_neighbors | 10, 20, 30 |
| batch_size | 100, 200, 400 |
| heads（TGAT） | 1, 2, 4 |
| memory_dim（TGN） | 64, 100, 128 |

### 搜索發現

- 靜態模型偏好 `sqrt_inverse` 類別權重；時序模型偏好 `effective`。
- 高 `oversample_factor`（9-20）對稀有類別的召回率至關重要。
- 較低的 `focal_gamma`（1.0）優於預設的 2.0，四個模型一致。
- GraphSAGE 淺層（2 層）優於深層；GAT 則 3 層表現較好。
- 時序模型最佳 `hidden_dim=256`（TGAT/TGN）或 `172`（DyGFormer），`n_neighbors=20-30`。
- TGAT/TGN 的 Optuna 搜索使用 20% 資料子採樣；DyGFormer 因架構特性無法使用子採樣（見下）。

### DyGFormer Optuna 搜索失敗分析

DyGFormer 無法使用 Optuna 子採樣搜索，原因：

1. **共現編碼依賴完整歷史**：DyGFormer 的 co-occurrence encoding 從鄰居歷史中計算節點共現矩陣，子採樣後鄰居歷史過於稀疏，導致共現信號消失。
2. **時序 patching 需要足夠序列長度**：patch_size=4 要求每個節點至少有 4 個歷史鄰居，子採樣後大量節點不滿足此條件。
3. **實驗驗證**：20% 子採樣 macro_f1=0.0024，50% 子採樣 macro_f1=0.0038，均完全無法收斂。TGAT/TGN 使用相同子採樣比例可正常搜索。

替代方案：在全量資料上進行手動消融實驗（focal_gamma, hidden_dim），確認 gamma=1.0 + hidden_dim=172 為最佳組態。

---

## 7. 歷史紀錄

### Optuna 調參前（~2026-06-15）

| 模型 | 舊 F1 | 新 F1 | 改善 |
|------|:-----:|:-----:|:----:|
| GraphSAGE | 0.9712 | 0.9773 | +0.0061 |
| GAT | 0.9534 | 0.9585 | +0.0051 |
| TGAT | 0.8963 | 0.8963 | 持平（保留 pre-Optuna） |
| TGN | 0.9463 | 0.9463 | 持平 |

> 靜態模型透過 Optuna 調參顯著提升；時序模型 Optuna 調參後（hidden_dim 172→256）在全量資料上未顯著改善，故保留 pre-Optuna checkpoint（hidden_dim=172）。先前 TGAT 報告的 0.9475 為驗證集 F1，實際測試集 F1 為 0.8963。

### 更早期紀錄

2026-06-13 的日誌顯示靜態模型 F1 僅 0.29-0.33，使用了 39 特徵和短期 CPU 測試。這些結果不應與目前 42 特徵管線的結果混合比較。

---

## 8. 建議的後續實驗

### 待執行

1. **TGAT 對抗式訓練** — 完成 TGAT 的 C-PGD 對抗訓練（受 GPU 資源限制暫緩）。
2. **時序模型全量 Optuna** — 在全量資料上執行 TGAT/TGN Optuna 搜索（目前僅用 20%）。
3. **E-GraphSAGE 完整評估** — 以 Optuna 搜索最佳參數訓練並評估，含對抗魯棒性。
4. **輕量時序替代方案** — 評估 GraphMixer / SimpleDyG 作為 TGAT/TGN 的快速替代。

### 已完成

- ~~Per-class recall 與 confusion matrix~~ — 已實作於 `compute_reliability_metrics.py`。
- ~~驗證集加權 ensemble~~ — 已實作 `EnsembleModel.from_validation()`。
- ~~Optuna 超參數搜索（4 模型）~~ — GraphSAGE 50 trials, GAT 13 trials, TGAT 15 trials, TGN 15 trials。
- ~~TGN 對抗式訓練~~ — 完成，ΔF1=-0.1828，memory 對擾動敏感。
- ~~靜態模型對抗式訓練~~ — GraphSAGE ΔF1=-0.0008, GAT ΔF1=+0.0114，幾乎無損。
- ~~DyGFormer 實作與調參~~ — 完成實作、訓練與消融實驗。Optuna subsample 不可行；手動確認 gamma=1.0 最佳。

---

## 9. 模型路線圖

**近期：**
- GraphSAGE 作為主要靜態基線與 Web Demo 預設模型。
- GAT 用於 attention-based 可解釋性和對抗訓練比較。
- TGN 作為時序模型首選（對抗穩定性最佳）。
- Ensemble 提供穩定的雙模型投票推論。

**未來：**
- E-GraphSAGE 邊特徵感知模型的正式評估。
- GraphMixer/SimpleDyG 輕量時序方案。
- DyGFormer 對抗式訓練與魯棒性評估。
