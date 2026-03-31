# Model Selection: GNN Architecture Comparison for NIDS

**版本：** 1.0 | **日期：** 2026-03 | **參照：** `docs/spec.md` v1.0.0

本文件記錄 GARF-NIDS 研究框架中各 GNN 模型的選擇考量、架構差異與實驗觀察，供後續模型擴充與論文撰寫參考。

---

## 1. 研究問題與模型選擇邏輯

核心問題：**時序 GNN（TGAT、TGN）對抗對抗性攻擊（adversarial attacks）的魯棒性是否優於靜態 GNN（GraphSAGE、GAT）？**

模型分為兩層：

| 層次 | 模型 | 角色 |
|------|------|------|
| **靜態基線** | GraphSAGE、GAT | 建立效能下限，也是目前 web app 主力 |
| **時序主角** | TGAT、TGN | 研究假設的主要對象 |
| **現代對比** | E-GraphSAGE、GraphMixer、DyGFormer | 未來擴充，驗證架構選擇是否仍有競爭力 |

---

## 2. 靜態基線模型

### 2.1 GraphSAGE（Hamilton et al., NeurIPS 2017）

**核心機制：** 對每個節點，從鄰居中採樣並以 Mean / LSTM / Pooling 聚合特徵，支援歸納式（inductive）推論。

**在本專案的角色：**
- 最輕量的基線，計算成本低
- 不含 attention，可清楚分離「有無注意力機制」的效果差異
- 對抗攻擊研究中的「最容易攻破」候選

**設定：**
```
Aggregation: Mean | Layers: 3 | Hidden: 256 | Dropout: 0.3
Edge clf: MLP(z_src ∥ z_dst ∥ edge_feat → 256 → n_classes)
```

**目前實驗結果（window=120s, FocalLoss γ=2.0）：**
| 指標 | Val | Test |
|------|-----|------|
| F1 (weighted) | 0.857 | 0.432 |
| Precision | — | 0.802 |
| Recall | — | 0.362 |
| ROC-AUC | — | 0.834 |

---

### 2.2 GAT（Veličković et al., ICLR 2018）

**核心機制：** 以 multi-head attention 對鄰居加權聚合，讓模型自動學習「哪個鄰居更重要」。

**在本專案的角色：**
- 有 attention 的靜態基線，提供「靜態 attention vs 時序 attention」的對比
- GAT 的 attention weights 可用於解釋哪條鄰近流量影響了判斷（可視化用）

**設定：**
```
Heads: 4 | Layers: 3 | Hidden: 256 | Dropout: 0.3
```

**目前實驗結果（window=120s, FocalLoss γ=2.0）：**
| 指標 | Val | Test |
|------|-----|------|
| F1 (weighted) | ~0.860 | 0.472 |
| Precision | — | 0.834 |
| Recall | — | 0.386 |
| ROC-AUC | — | 0.859 |

**觀察：** GAT 在所有指標上均略優於 GraphSAGE，尤其 ROC-AUC 高 0.025。兩者 Val/Test 差距（~0.43 gap）均源於 temporal distribution shift（見 Section 5）。

---

## 3. 時序 GNN 主角

### 3.1 TGAT（Xu et al., ICLR 2020）

**核心機制：**
```
對一條邊 (u→v, t)：
  1. 取 u 在 t 前的 k 個最近鄰互動 (LastNeighborLoader)
  2. time2vec(t - t_neighbor) 編碼時間差
  3. Multi-head attention 對鄰居加權聚合
  → u 在時間 t 的 embedding
```

**無狀態（stateless）：** 每次推論從原始事件序列重算，無節點記憶體。歸納式，支援新節點。

**適合 NIDS 的理由：**
- 能捕捉 burst 型攻擊（短時間大量同類流量）的時序模式
- 時間編碼讓模型感知「多久前發生過類似流量」

**設定：**
```
Heads: 2 | time2vec dim: 64 | k neighbors: 20 | Hidden: 172
```

**目前狀態：** 架構設計完成（`configs/model/tgat.yaml`），實作排定於 TGN 完成後進行。

---

### 3.2 TGN（Rossi et al., arXiv 2020）

**核心機制：** 在 TGAT 之上加入 **per-node GRU memory**：

```
每個節點 n 維護 memory state s_n ∈ R^{172}：
  1. 每次 n 參與互動 → GRU 更新 s_n
  2. Embedding module 在 s_n 基礎上做 1-hop temporal attention
  3. Edge clf: MLP(z_src ∥ z_dst ∥ edge_feat → num_classes)
```

**與 TGAT 的關鍵差異：**

| | TGAT | TGN |
|---|------|-----|
| 記憶機制 | 無狀態 | Per-node GRU memory |
| 長期依賴 | 弱（只看最近 k 鄰居） | 強（memory 累積整個歷史） |
| 推論速度 | 慢（每次重算） | 快（memory 緩存） |
| 對 APT 的建模 | 中 | 佳（長時間低頻攻擊） |
| Training complexity | 低 | 高（memory replay，防 label 洩漏） |

**在本專案的角色：**
- 研究假設的核心：memory 讓模型記住「這個 IP 過去的行為」，即使攻擊流量在測試期減少，memory 中的歷史行為仍可輔助分類
- `memory_reset_policy: before_each_attack` — 對抗評估時每次重置 memory，確保攻擊者無法利用記憶狀態

**設定（當前實作）：**
```
memory_dim: 172 | time_dim: 64 | num_neighbors: 20
message: IdentityMessage | aggregator: LastAggregator | updater: GRU
embedding: graph_attention (可切換為 identity)
```

**節點粒度：** IP-level（非靜態模型的 proxy node），讓 memory 自然對應「主機行為歷史」。

**目前狀態：** 已實作 (`src/models/tgn.py`)，temporal data pipeline 已完成 (`src/data/temporal_builder.py`)，待資料集預處理後訓練。

---

## 4. 現代候選模型（未來擴充）

### 4.1 E-GraphSAGE（Lo et al., IEEE NOMS 2022）

**核心改進：** GraphSAGE 的 edge-featured 變體 — 原始 GraphSAGE 聚合時只使用節點特徵，E-GraphSAGE 在聚合函數中同時納入邊特徵。

**為何值得比較：**
- 本專案的任務是 **edge classification**，邊特徵（NetFlow 39 維）才是主要訊號
- E-GraphSAGE 在 CICIDS2017 等 NIDS benchmark 上報告 F1 > 0.99
- 可驗證「靜態 + 邊特徵強化」能否縮小與時序模型的差距

**注意事項：** 大多數 CICIDS 實驗未做嚴格時序切分，結果偏樂觀；需在 NF-UNSW-NB15-v2 的 chronological split 下重現。

**實作成本：** 低 — 在現有 GraphSAGE 基礎上修改 message passing，約 50 行。

---

### 4.2 GraphMixer / SimpleDynG（Cong et al., NeurIPS 2023）

**核心改進：** 以 **MLP-Mixer** 取代 temporal attention，大幅提速（3–5× vs TGAT）且效果相近。

論文結論：在多數 link prediction benchmark 上，token-mixing MLP 比 self-attention 更有效率，attention 機制對時序圖的邊際貢獻有限。

**在本專案的考量：**
- 如果 TGN 的 graph attention embedding 效果有限，GraphMixer 是速度與效能的平衡點
- 對 NIDS 來說：NetFlow 特徵已非常豐富（39 維），attention 對「哪個鄰居更重要」的判斷可能不如特徵本身重要
- 提供「簡單模型能否匹敵複雜時序 GNN」的消融實驗基礎

**實作成本：** 中 — 需要實作 MLP-Mixer 時序聚合，約 150 行，可在 TGN 框架上替換 embedding module。

---

### 4.3 DyGFormer（Yu et al., NeurIPS 2023）

**核心改進：** 以 **Transformer + neighbor co-occurrence encoding** 取代 attention over neighbors。關鍵創新：兩個節點共同出現過多少次直接編碼進特徵。

**對 NIDS 特別有潛力的原因：**
- IP pair (src_ip, dst_ip) 的重複性是攻擊的強訊號（DoS 的 flooding pair、C2 的固定通信 pair）
- Co-occurrence encoding 自然捕捉「這對 IP 歷史上互動頻率」
- 在 TGB benchmark（Temporal Graph Benchmark）上超越 TGN

**實作成本：** 高 — 需要 Transformer backbone + co-occurrence 矩陣維護，約 300 行。

---

## 5. 實驗共同觀察

### 5.1 Temporal Distribution Shift（關鍵問題）

NF-UNSW-NB15-v2 的攻擊流量在時間軸分布不均，導致所有模型都出現 Val/Test 大幅落差：

| 切割 | 視窗數 | Benign 比例 | Attack 比例 |
|------|--------|------------|------------|
| Train | 1,289 | 26.8% | 73.2% |
| Val | 430 | 19.6% | 80.4% |
| **Test** | **430** | **59.3%** | **40.7%** |

**影響：** 所有靜態模型 Val F1 ~0.86，Test F1 ~0.43–0.47。這不是模型的 bug，而是資料集的時序特性。

**期待 TGN/TGAT 改善的原因：**
- Memory 能記住「這個 IP 在 train 期間的攻擊行為」，即使 test 期間攻擊流量減少，history bias 仍存在
- 時間編碼讓模型感知攻擊事件的時間間隔，而非單純看特徵分布

### 5.2 指標選擇建議

- **主要報告：Val F1（weighted）** — 反映模型學習能力，不受 distribution shift 影響
- **次要報告：Test F1（weighted）** — 反映跨時間泛化能力
- **補充：Macro F1** — Test 集 Benign 佔 59%，weighted 指標受 Benign 主導，Macro 可更公平評估各攻擊類別

---

## 6. 模型演進路線圖

```
Phase 1（已完成）
  GraphSAGE ──────────────────────── Val F1 0.857 / Test F1 0.432
  GAT ─────────────────────────────── Val F1 0.860 / Test F1 0.472

Phase 2（進行中）
  TGN ──── 已實作，等待資料預處理 & 訓練
  TGAT ─── 設計完成，排定 TGN 後實作

Phase 3（規劃中）
  E-GraphSAGE ──── edge-featured 靜態對比
  GraphMixer ────── 速度 vs 效能權衡
  DyGFormer ──────── co-occurrence 時序 Transformer

消融實驗矩陣（規劃）：
  模型 × 攻擊方法（C-PGD / Edge Injection / GAN）× 資料集（UNSW / BoT-IoT）
```

---

## 7. 參考文獻

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.
- Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.
- Xu, D., et al. (2020). Inductive Representation Learning on Temporal Graphs (TGAT). *ICLR*.
- Rossi, E., et al. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *arXiv:2006.10637*.
- Lo, W. W., et al. (2022). E-GraphSAGE: A GNN-Based IDS for IoT. *IEEE NOMS*.
- Cong, W., et al. (2023). Do We Really Need Complicated Model Architectures for Temporal Networks? (GraphMixer). *ICLR 2023*.
- Yu, L., et al. (2023). Towards Better Dynamic Graph Learning: New Architecture and Unified Library (DyGFormer). *NeurIPS*.
