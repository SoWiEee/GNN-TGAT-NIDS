# Architecture Review: GARF-NIDS v0.2

> 從軟體工程（SE）、機器學習（ML）與學術投稿（AC）三個角度，對新研究方向進行系統性審核。
> 目標投稿：NDSS 2027 / USENIX Security 2027。
> 每個問題均標示嚴重程度（🔴 Critical / 🟡 Major / 🟢 Minor）及面向（SE / ML / AC）。

**審核版本：** 0.2 | **參照規格：** `docs/spec.md` v0.3.0 | **日期：** 2026-03

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [研究方向升級後的新問題](#2-研究方向升級後的新問題)
3. [Memory Poisoning Attack 架構](#3-memory-poisoning-attack-架構)
4. [Timing-Aware Edge Injection 架構](#4-timing-aware-edge-injection-架構)
5. [評估協定完整性](#5-評估協定完整性)
6. [投稿可行性審核](#6-投稿可行性審核)
7. [保留問題（v0.1 仍未解決）](#7-保留問題v01-仍未解決)
8. [Risk Summary Table](#8-risk-summary-table)
9. [Recommended Action Items](#9-recommended-action-items)

---

## 1. Executive Summary

研究方向從「靜態 vs 時序 GNN 魯棒性比較」升級為「時序 GNN 記憶機制的可利用性發現」，大幅提升了論文的差異化程度與投稿競爭力。以下是主要問題摘要：

| 優先級 | 問題 | 面向 |
|--------|------|------|
| 🔴 | Memory Poisoning 攻擊的「記憶體歸因」未定義：無法區分哪些節點被污染、污染程度為何 | ML |
| 🔴 | TAEI 的 timing 優化目標函數需要多次模型前向傳播，對 TGN 來說計算成本極高（coarse 搜尋 K×n 次推論） | SE |
| 🔴 | 論文核心主張（「時序 GNN 更脆弱」）需要統計顯著性支撐；目前評估設計缺少 bootstrap CI 或 permutation test | AC |
| 🟡 | MPA 的「注入邊特徵應在良性分佈內」與 TAEI 共享 edge feature generator，但兩者需求不同（MPA 追求記憶體偏移，TAEI 追求分類邊界穿越） | ML |
| 🟡 | Memory Half-Life 定義（DR 恢復至 90%）對部分攻擊可能永遠達不到，缺少 timeout/censored 處理 | ML |
| 🟡 | `memory_poisoning.py` 模組需要直接存取 TGN/TGAT 的內部記憶體狀態，而 `BaseNIDSModel` 介面目前不暴露此功能 | SE |
| 🟡 | 投稿安全頂會（NDSS/USENIX）需要 artifact 評估（可重現性），Docker 容器化尚未規劃 | AC |
| 🟢 | `configs/attack/mpa.yaml` 的 `target_nodes: auto` 未定義選取策略（degree top 10%？或 betweenness centrality？） | ML |
| 🟢 | 缺少 ablation study 規劃：需分離 timing 效果 vs constraint 效果 vs memory 容量效果 | AC |

---

## 2. 研究方向升級後的新問題

### 2.1 論文核心主張需要統計顯著性支撐 🔴 AC

**問題：** 論文主張「時序 GNN 在 TAEI/MPA 下更脆弱」是一個跨模型的定量比較聲明。若只報告單次實驗結果，審稿人會要求：

1. 不同隨機種子（seeds）下的結果穩定性
2. 統計顯著性檢定（t-test 或 bootstrap confidence interval）
3. 多個 epsilon/n_inject 值下的一致性

**建議修正：**

在 `configs/eval/full_matrix.yaml` 中加入 multi-seed 設計：

```yaml
# configs/eval/full_matrix.yaml
seeds: [42, 123, 456, 789, 1024]   # 5 seeds for statistical significance
bootstrap_ci: true                  # 計算 95% CI
```

在 `eval/metrics.py` 中新增：

```python
def bootstrap_confidence_interval(
    scores: list[float], n_boot: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for a list of metric values."""
```

### 2.2 Threat Model 需要形式化，以符合頂會標準 🟡 AC

**問題：** NDSS / USENIX Security 審稿人對 threat model 的要求非常嚴格，現有 spec 的 threat model 過於簡略。

**建議補充的 Threat Model 表格（加入 spec.md Section 3.3.1）：**

| 面向 | 說明 |
|------|------|
| **Attacker goal** | Cause temporal NIDS to misclassify attack traffic as benign（evasion，非 poisoning-to-destroy-model） |
| **Attacker knowledge** | White-box（完整模型權重）for C-PGD；Black-box（query-based）for GAN；Network topology（部分）for TAEI/MPA |
| **Attacker capability** | 可注入或修改少量流量，但不能控制路由器或網路基礎設施 |
| **Attack timing** | 攻擊者在真實攻擊發動前最多 H 秒可執行 MPA；TAEI 注入必須在目標攻擊流量前 |
| **Defender** | NIDS 以固定模型參數執行推論，不做線上更新 |

### 2.3 MPA 與 TAEI 的評估不可交叉污染 🟡 ML

**問題：** 若同一個模型在評估 TAEI 後繼續評估 MPA，記憶體狀態會受前一個攻擊影響。

**建議：** 在 `eval/comparison.py` 中強制執行獨立評估隔離：

```python
# eval/comparison.py
for attack_cfg in cfg.attacks:
    # 每個攻擊前重新載入 clean model checkpoint
    model = instantiate(cfg.model)
    load_checkpoint(model, optimizer=None, path=cfg.checkpoint_dir / "best.pt")
    # 執行 memory_reset_policy
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    results[attack_cfg.name] = run_attack(model, attack_cfg, test_data)
```

---

## 3. Memory Poisoning Attack 架構

### 3.1 BaseNIDSModel 不暴露記憶體介面 🔴 SE

**問題：** MPA 需要讀取和監控 TGN/TGAT 的節點記憶體向量，但當前 `BaseNIDSModel` 只定義了：

```python
def forward(self, data) -> torch.Tensor: ...
def predict_edges(self, data) -> torch.Tensor: ...
def predict_proba(self, data) -> torch.Tensor: ...
```

`memory_poisoning.py` 無法透過統一介面存取記憶體狀態，只能直接存取 TGN/TGAT 的具體實作，破壞 ABC 設計。

**建議修正（擴展 `src/models/base.py`）：**

```python
class BaseNIDSModel(ABC, nn.Module):
    # ... 現有方法 ...

    def get_memory_state(self) -> torch.Tensor | None:
        """Return current node memory matrix (shape: [num_nodes, memory_dim]).
        Returns None for static models (GraphSAGE, GAT)."""
        return None  # default: static model

    def reset_memory(self) -> None:
        """Reset node memory to zero state.
        No-op for static models."""
        pass  # default: no-op

    @property
    def has_memory(self) -> bool:
        """True for temporal models with persistent node state."""
        return False  # default: static
```

TGAT / TGN 覆寫這三個方法。`memory_poisoning.py` 透過 `model.has_memory` 檢查適用性，透過 `model.get_memory_state()` 取得快照。

### 3.2 Memory Half-Life 定義的邊界情況 🟡 ML

**問題：** 若攻擊非常有效（記憶體幾乎永久污染），或 DR@attack 從未恢復至 90%，`T½` 將無法定義（censored observation）。

**建議：** 在 `eval/metrics.py` 中採用存活分析框架：

```python
def compute_memory_half_life(
    dr_recovery_curve: list[float],
    clean_baseline: float,
    recovery_threshold: float = 0.9,
    max_steps: int = 1000,
) -> tuple[float | None, bool]:
    """
    Returns:
        (T_half, censored)
        T_half: steps to recovery; None if never recovered within max_steps
        censored: True if T_half is None (right-censored observation)
    """
```

在論文中以 Kaplan-Meier 曲線呈現跨模型的 DR recovery 過程，這比單一數值更有說服力。

### 3.3 MPA 注入邊特徵生成策略未定義 🟢 ML

**問題：** `target_nodes: auto` 與注入邊特徵策略都用 "auto"，但未定義選取邏輯。

**建議：**

- **目標節點選取：** 選擇 betweenness centrality 最高的 top-K 節點（而非 degree）。Betweenness 高的節點是最多最短路徑的中繼，污染此類節點的記憶體影響最廣泛。
- **注入邊特徵：** 從訓練集 benign 邊的特徵分佈中採樣（multivariate Gaussian fit），確保 CSR = 1.0 的同時最大化記憶體偏移量（loss = cosine_similarity(m_v_after, m_attack_prototype)）。

---

## 4. Timing-Aware Edge Injection 架構

### 4.1 Coarse 搜尋的計算成本 🔴 SE

**問題：** TAEI coarse 搜尋需要對 K 個時間點各執行一次完整的模型推論（含 memory update），對 TGN 這是 `K × O(N × E)` 的操作。預設 K=10，若測試集有 50 個攻擊事件，需要 500 次完整推論。

**建議優化：**

1. **Batched coarse search：** 在 `edge_injection.py` 中，透過複製模型記憶體狀態（`deepcopy(model.get_memory_state())`）避免重複前向計算：

```python
# 粗搜尋時暫存各時間點的記憶體快照，不需重跑整個序列
checkpoints = {}
for i, t in enumerate(window_starts):
    checkpoints[t] = deepcopy(model.get_memory_state())
```

2. **Early termination：** 若 coarse 搜尋前 3 個點的 ASR 差距 < 2%，跳過 fine search。

3. **在設定中提供 `timing_search: none` 選項**，供靜態 GNN baseline 直接跳過搜尋（已存在）。

### 4.2 Timing Sensitivity Curve 的視覺化 🟢 AC

**建議新增至評估輸出：** 以折線圖呈現 ASR vs. injection time offset（Δt），橫軸為注入時間距離攻擊流量的秒數（-300 到 0），縱軸為 ASR。這張圖可以直觀展示時序模型的「脆弱窗口」（vulnerable window），是論文中最有說服力的視覺化之一。

---

## 5. 評估協定完整性

### 5.1 Ablation Study 設計缺失 🟡 AC

**問題：** 審稿人會要求分離各個設計選擇的效果，特別是：

| Ablation | 目的 |
|----------|------|
| TAEI without timing optimization（fixed random t）| 驗證 timing 最佳化本身的貢獻 |
| MPA without constraint enforcement（CSR < 1.0） | 驗證 CSR=1.0 要求對 ASR 的影響 |
| MPA with shorter horizon（H = 60s vs 300s） | 驗證污染持續時間對 Half-Life 的影響 |
| TGAT vs TGN under MPA | 驗證 GRU memory vs attention 的差異 |

**建議：** 在 `configs/eval/ablation.yaml` 中定義完整的 ablation 矩陣。

### 5.2 與 BAAAN (USENIX Security 2021) 的明確對比 🟡 AC

**問題：** Han et al. (USENIX 2021) 的 BAAAN 是目前最相關的 NIDS 對抗攻擊工作，審稿人必然要求比較。

**建議：** 在評估時明確報告：

- BAAAN 在本框架的 CSR 分析（他們的攻擊是否滿足約束？）
- BAAAN vs TAEI 在時序 NIDS 上的 ASR 比較（BAAAN 不考慮 timing）
- 若 BAAAN 的代碼可用，直接在本框架中複現為 baseline

### 5.3 跨資料集遷移的評估範圍 🟢 AC

**問題：** 目前 NF-BoT-IoT-v2 遷移實驗只計劃評估整體 ASR，但頂會審稿人通常要求：
1. TAEI 的最佳 timing offset 是否在兩個資料集間可遷移？
2. MPA 的 Half-Life 是否資料集相關？

---

## 6. 投稿可行性審核

### 6.1 目標投稿的審核標準分析

| 面向 | NDSS 2027 | USENIX Security 2027 | IEEE TIFS |
|------|-----------|----------------------|-----------|
| **新穎性要求** | 高：需要 novel attack or defense | 非常高：需要系統性安全貢獻 | 中：學術嚴謹性優先 |
| **本研究的契合點** | TAEI + MPA 是 novel attack；NIDS 是 NDSS 核心主題 | Threat model 需更完整；需要 real-world 驗證 | 適合完整比較研究，接受 F1 ≥ 0.90 baseline |
| **Artifact 要求** | 鼓勵但非強制 | Artifact Evaluation 是標準流程 | 無強制要求 |
| **頁數限制** | 10 頁（ACM 格式，不含 references） | 14 頁（USENIX 格式） | 14 頁 |
| **建議策略** | **主要投稿目標**：攻擊發現 + CAAG 作為雙貢獻 | 備選：需要更完整的 real-world deployment 討論 | 備選 journal：可擴展為完整比較研究 |

### 6.2 論文貢獻排序建議（與 NDSS 審稿標準對齊）

為最大化接受率，貢獻應以以下順序呈現：

1. **P1（主貢獻）:** Discovery of timing-dependent attack surface in temporal GNNs — TAEI + MPA 作為兩個新攻擊向量的形式化定義與實驗驗證
2. **P2（技術貢獻）:** CAAG with CSR=1.0 enforcement — 解決 NIDS 對抗研究的根本方法論問題
3. **P3（比較貢獻）:** Systematic robustness comparison of static vs temporal GNNs under timing-aware attacks — 反直覺的「時序 GNN 更脆弱」結論

> ⚠️ 若論文將 P3（比較）放在 P1 之前，會被誤讀為 benchmark paper，接受率顯著下降。

### 6.3 Artifact 準備建議（USENIX Security 必要，NDSS 加分）

目前缺少以下 artifact 元件：

```
artifact/
├── Dockerfile              ← 確保環境可重現（CUDA 12.4 + PyTorch 2.4）
├── run_all_experiments.sh  ← 一鍵重現所有表格和圖
├── data/
│   └── README.md           ← 資料集取得說明（UNSW 不可直接打包）
└── results/
    └── expected_outputs/   ← 預期輸出值（供驗證）
```

---

## 7. 保留問題（v0.1 仍未解決）

以下問題在 v0.1 的 architecture review 中已識別，v0.2 確認仍需注意：

| 問題 | 狀態 | 備註 |
|------|------|------|
| `constraints.py` 位於 `src/attack/`（已修正） | ✅ 已解決 | v0.1 R1 |
| PGD 改用 normalized gradient（已修正） | ✅ 已解決 | v0.1 R2 |
| BaseNIDSModel / BaseAttack ABC（已實作） | ✅ 已解決 | v0.1 R3 |
| `utils/seed.py`（已實作） | ✅ 已解決 | v0.1 R4 |
| 靜態圖 on-demand loading（已實作） | ✅ 已解決 | v0.1 R5 |
| TGN memory reset policy（已定義） | ✅ 已解決 | v0.1 R6 |
| 類別不平衡 weighted loss（已統一） | ✅ 已解決 | v0.1 R7 |
| `target_split` 欄位（已加入所有攻擊設定） | ✅ 已解決 | v0.1 R8 |
| `comparison.py` 改用 Hydra instantiate | 🔄 待實作 | Phase 4 |
| Checkpoint 策略（已實作） | ✅ 已解決 | v0.1 R10 |
| Scaler 序列化（已實作） | ✅ 已解決 | v0.1 R12 |

---

## 8. Risk Summary Table

| 編號 | 問題 | 面向 | 嚴重度 | 影響 | 建議行動 |
|------|------|------|--------|------|----------|
| R1 | BaseNIDSModel 不暴露記憶體介面 | SE | 🔴 | MPA 必須 hack 具體實作，破壞 ABC 設計 | 新增 `get_memory_state` / `reset_memory` / `has_memory` |
| R2 | TAEI coarse 搜尋計算成本 | SE | 🔴 | K=10 時 TGN 需 500+ 次推論 | 記憶體快照 + early termination |
| R3 | 論文核心主張缺統計支撐 | AC | 🔴 | NDSS/USENIX 審稿人必要求 | Multi-seed + bootstrap CI |
| R4 | Threat model 不夠完整 | AC | 🟡 | 頂會標準要求明確的能力與知識假設 | 補充 5 維度 threat model 表格 |
| R5 | MPA/TAEI 評估交叉污染 | ML | 🟡 | 不同攻擊的起始記憶體狀態不同 | 每次攻擊前重新載入 clean checkpoint |
| R6 | Memory Half-Life 無 censored 處理 | ML | 🟡 | 永久污染情況無法量化 | 存活分析框架 + Kaplan-Meier |
| R7 | 與 BAAAN (USENIX 2021) 缺乏直接比較 | AC | 🟡 | 審稿人必問；不對比即被視為忽視相關工作 | 複現 BAAAN 作為 baseline |
| R8 | Ablation study 未規劃 | AC | 🟡 | 無法分離各設計選擇的效果 | 補充 `configs/eval/ablation.yaml` |
| R9 | Artifact（Docker + scripts）未規劃 | AC | 🟡 | USENIX Artifact Evaluation 必要；NDSS 加分 | Phase 12 加入 artifact 建置 |
| R10 | MPA 目標節點選取策略模糊 | ML | 🟢 | "auto" 難以重現 | 明確指定 betweenness centrality top-K |
| R11 | Timing Sensitivity Curve 視覺化未規劃 | AC | 🟢 | 論文最有說服力的圖缺失 | 在 `eval/comparison.py` 加入繪圖輸出 |
| R12 | 跨資料集 TAEI/MPA 可遷移性未評估 | AC | 🟢 | 降低 NF-BoT-IoT-v2 實驗的深度 | 在 Section 9 遷移實驗中加入此分析 |

---

## 9. Recommended Action Items

以下依實作優先序排列：

### 即刻（Phase 3 前，不阻塞實作但需要設計決策）

```
[ ] 擴展 BaseNIDSModel：新增 get_memory_state / reset_memory / has_memory
[ ] 在 eval/comparison.py 加入每次攻擊前重新載入 clean checkpoint 的邏輯
[ ] 在 configs/eval/ 加入 multi-seed 設定（seeds: [42, 123, 456, 789, 1024]）
[ ] 補充 spec.md Section 3.3.1 的完整 5 維度 Threat Model 表格
```

### Phase 3（CAAG 實作期間）

```
[ ] edge_injection.py：加入記憶體快照 + early termination 最佳化
[ ] memory_poisoning.py：實作 MPA，透過 get_memory_state() 介面讀取記憶體
[ ] evaluator.py：新增 bootstrap_confidence_interval() 和 compute_memory_half_life()
[ ] configs/attack/mpa.yaml：明確指定 target_nodes 為 betweenness_centrality_top_k
```

### Phase 4（評估重構期間）

```
[ ] eval/comparison.py 改用 Hydra instantiate（清除強耦合，v0.1 遺留問題）
[ ] 加入 Timing Sensitivity Curve 繪圖輸出（ASR vs injection offset）
[ ] 實作 BAAAN baseline 或引用其公開結果進行比較
[ ] configs/eval/ablation.yaml：定義完整 ablation 矩陣
```

### Phase 12（投稿前）

```
[ ] Dockerfile + run_all_experiments.sh（artifact 建置）
[ ] 執行 5-seed multi-run，確認所有表格的 95% CI
[ ] 論文貢獻順序確認：P1 (TAEI+MPA 攻擊發現) → P2 (CAAG) → P3 (比較)
[ ] NDSS 2027 投稿截止日確認（預期 2026 年 5-6 月）
```

---

> 下一次審核應在 Phase 3（CAAG 實作）完成後進行，重點驗證 TAEI 和 MPA 的實驗結果是否支持論文核心主張。
