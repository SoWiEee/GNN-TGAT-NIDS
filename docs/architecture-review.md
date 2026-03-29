# Architecture Review: GARF-NIDS

> 從軟體工程與機器學習角度對 `docs/spec.md` 進行的系統性審核。
> 每個問題均標示嚴重程度（🔴 Critical / 🟡 Major / 🟢 Minor）及所屬面向（SE = 軟體工程、ML = 機器學習）。

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Module Structure](#2-module-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [CAAG Attack Framework](#5-caag-attack-framework)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Cross-Cutting Concerns](#7-cross-cutting-concerns)
8. [Risk Summary Table](#8-risk-summary-table)
9. [Recommended Action Items](#9-recommended-action-items)

---

## 1. Executive Summary

spec.md 整體架構清晰，研究動機與邊界定義完整，資料切分策略正確。以下是主要問題的優先摘要：

| 優先級 | 問題 | 面向 |
|--------|------|------|
| 🔴 | `constraints.py` 放在 `src/data/` 而非 `src/attack/`，耦合位置錯誤 | SE |
| 🔴 | PGD 公式使用 `sign(∇L)`，對連續特徵應使用 full-gradient projection | ML |
| 🔴 | 未定義抽象基底類別（ABC），攻擊與模型無法統一介面 | SE |
| 🟡 | 未指定全域隨機種子策略，實驗無法完整重現 | ML |
| 🟡 | 靜態圖分批載入策略缺失，2.5M flows 全部載入記憶體會 OOM | SE |
| 🟡 | TGN 在攻擊場景中節點記憶體的「重置時機」未定義 | ML |
| 🟡 | `eval/comparison.py` 直接 import 所有模型與攻擊，強耦合 | SE |
| 🟢 | 缺少型別提示（type hints）規範與 docstring 格式標準 | SE |
| 🟢 | GAN 訓練穩定性退路策略缺乏技術細節 | ML |

---

## 2. Module Structure

### 2.1 現狀

```
src/
├── data/
│   ├── loader.py
│   ├── static_builder.py
│   ├── temporal_builder.py
│   └── constraints.py        ← ⚠️ 位置錯誤
├── models/
│   ├── graphsage.py
│   ├── gat.py
│   ├── tgat.py
│   └── tgn.py
├── attack/
│   ├── cpgd.py
│   ├── edge_injection.py
│   ├── gan_generator.py
│   └── evaluator.py
├── defense/
│   └── adversarial_training.py
├── eval/
│   ├── metrics.py
│   └── comparison.py
└── utils/
    ├── config.py
    └── logger.py
```

### 2.2 問題：`constraints.py` 放在 `data/` 模組 🔴 SE

**問題根源：** 約束集合（協定合法性、特徵代數一致性等）是在 *對抗樣本生成期間* 被呼叫的邏輯，與資料前處理無關。將它放在 `src/data/` 會造成：

- `src/attack/` 必須跨模組反向依賴 `src/data/` 的非資料邏輯
- 命名上產生誤導（開發者會以為約束是資料 pipeline 的一部分）

**建議修正：**

```
src/
├── data/
│   ├── loader.py
│   ├── static_builder.py
│   └── temporal_builder.py
└── attack/
    ├── constraints.py        ← 移至此處
    ├── cpgd.py
    ├── edge_injection.py
    ├── gan_generator.py
    └── evaluator.py
```

### 2.3 問題：缺乏抽象基底類別（ABC） 🔴 SE

spec.md 規劃了 4 個模型與 3 種攻擊方法，但未定義統一介面。這會導致 `eval/comparison.py` 必須為每個模型和攻擊手動撰寫各自的呼叫邏輯，無法擴充。

**建議新增：**

```python
# src/models/base.py
from abc import ABC, abstractmethod
import torch

class BaseNIDSModel(ABC):
    @abstractmethod
    def forward(self, data) -> torch.Tensor: ...

    @abstractmethod
    def predict_edges(self, data) -> torch.Tensor: ...

# src/attack/base.py
class BaseAttack(ABC):
    @abstractmethod
    def generate(self, model: BaseNIDSModel, data, **kwargs): ...

    @abstractmethod
    def constraint_check(self, x_adv) -> bool: ...
```

`eval/comparison.py` 只需依賴這兩個抽象介面，新模型或攻擊方法加入時不需修改評估邏輯。

### 2.4 問題：缺少 `utils/seed.py` 🟡 ML

spec.md 未規劃全域隨機種子設定模組。對抗攻擊（特別是 GAN 初始化與 PGD 起始點）具有隨機性，不固定種子會導致實驗結果難以精確重現。

**建議新增 `src/utils/seed.py`：**

```python
import random, numpy as np, torch

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

所有 `train.py`、`attack.py` 入口都應在最開始呼叫 `set_global_seed(cfg.seed)`。

---

## 3. Data Pipeline

### 3.1 靜態圖：記憶體管理 🟡 SE

**現狀：** spec.md 將每個時間窗口儲存為獨立 `.pt` 檔（`data/processed/static/train/`）。訓練時若一次性載入所有窗口，2.5M 流量的資料集將消耗大量 RAM。

**建議：** 使用 PyG 的 `InMemoryDataset`（小資料集）或 `Dataset`（大資料集，按需載入）：

```python
# src/data/static_dataset.py
from torch_geometric.data import Dataset

class StaticNIDSDataset(Dataset):
    def get(self, idx):
        return torch.load(self.processed_paths[idx])  # 按需載入單一窗口
```

`DataLoader` 配合 `num_workers` 可做預取，避免 GPU 等待 I/O。

### 3.2 時序圖：嚴格時序切分驗證 🟡 ML

spec.md 正確指定了「嚴格時序順序」切分，但未說明如何驗證切分後無洩漏。

**建議新增驗證邏輯（`src/data/temporal_builder.py`）：**

```python
assert train_data.t.max() < val_data.t.min(), "時序洩漏：訓練集時間戳晚於驗證集"
assert val_data.t.max() < test_data.t.min(),  "時序洩漏：驗證集時間戳晚於測試集"
```

### 3.3 特徵正規化：inverse transform 缺失 🟢 ML

CAAG 在約束投影後需要將特徵還原（反正規化）才能進行代數一致性重算，之後再重新正規化。spec.md 未提及保存 scaler 物件供攻擊模組使用。

**建議：** `static_builder.py` / `temporal_builder.py` 執行完後，應將 `StandardScaler` 序列化存放：

```
data/processed/static/scaler.pkl
data/processed/temporal/scaler.pkl
```

攻擊模組載入 scaler 做 `inverse_transform → 代數重算 → transform`。

---

## 4. Model Architecture

### 4.1 邊分類的鄰域洩漏（Edge Label Leakage） 🔴 ML

**問題：** 對邊進行分類時，若訓練集中某條邊 `(u, v)` 的標籤可以透過其他連接 `u` 或 `v` 的邊特徵被推斷，則存在洩漏。在 GraphSAGE / GAT 的鄰域聚合中，目標邊的兩端點會聚合來自其他邊的特徵，這本身是正常的。但若在 mini-batch 採樣時不小心把同一時間窗口的正標籤邊暴露給自身的計算圖，就會發生洩漏。

**建議：** 使用 PyG 的 `LinkNeighborLoader` 搭配 `neg_sampling` 時明確指定 `time_attr`（時序模型），確保計算圖中不包含未來的邊。

### 4.2 TGN 在攻擊場景中的記憶體重置 🟡 ML

**問題：** TGN 的節點記憶體在訓練結束後會保留狀態。評估攻擊效果時，需要明確定義以下問題：

| 問題 | 影響 |
|------|------|
| 攻擊前是否重置記憶體？ | 影響 ASR 的可比較性（不同攻擊的起始狀態可能不同） |
| Edge Injection 注入後記憶體如何演化？ | 記憶體污染是時序模型特有攻擊面，需追蹤記憶體狀態 |
| 對抗訓練時記憶體是否在每個 epoch 重置？ | 影響訓練收斂行為 |

**建議：** 在 spec.md 或 `configs/attack/` 中明確定義 `memory_reset_policy: {before_each_attack, never, per_epoch}`。

### 4.3 類別不平衡處理不一致 🟡 ML

spec.md 在 GraphSAGE 設定中提到「Weighted cross-entropy (address class imbalance)」，但其他三個模型（GAT、TGAT、TGN）未提及。NF-UNSW-NB15-v2 攻擊流量比例遠低於正常流量，不處理不平衡會使所有模型偏向預測 benign。

**建議：** 在 `configs/model/` 的所有模型設定中統一加入：

```yaml
loss:
  type: weighted_cross_entropy
  class_weights: auto   # 由訓練集標籤頻率計算
```

---

## 5. CAAG Attack Framework

### 5.1 PGD 公式錯誤：`sign(∇L)` vs. full gradient 🔴 ML

**現狀（spec.md Section 3.3.2）：**

```
x_adv ← x_adv + α · sign(∇_x L(f(G, x_adv), y_target))
```

**問題：** `sign(∇L)` 是 FGSM 的更新方式，適用於 `L∞` 範數約束下的離散擾動（如圖像像素）。對於 NetFlow 的連續特徵（流量 bytes、持續時間等），使用 sign 會：

1. 丟失梯度大小資訊，無法細緻控制每個特徵的擾動量
2. 在 `L2` 或非均一特徵空間下，sign 更新不具最佳性

**建議修正（`L2` 版本）：**

```
g = ∇_x L(f(G, x_adv), y_target)
x_adv ← x_adv + α · g / (‖g‖₂ + ε_small)   # normalized gradient
x_adv ← Project(x_adv, C)                     # 約束投影
x_adv ← clip(x_adv, x - ε, x + ε)            # L∞ ball（若需要）
```

若確實要用 `L∞` 版 PGD，則 `sign` 是正確的，但需要在 spec 中明確說明使用的範數類型。

### 5.2 Edge Injection 時機最佳化缺乏形式化定義 🟡 ML

spec.md 提到「注入時機是時序模型的額外最佳化變數」，但未定義如何搜尋最佳注入時間點。

**建議補充定義：**

```
injection_time* = argmin_{t ∈ [t_start, t_attack]} ASR(model, inject_at=t)
```

搜尋策略建議：
- 粗粒度：均勻切分時間軸，取 ASR 最高的區間
- 細粒度：在粗粒度最優區間內做 binary search

### 5.3 WGAN-GP 訓練穩定性退路缺技術細節 🟢 ML

spec.md 在風險表中提到「GAN training instability → 退回 C-PGD only」，但未說明判斷 GAN 訓練不穩定的標準。

**建議新增判斷標準：**

| 指標 | 不穩定判斷門檻 |
|------|----------------|
| Critic loss 振盪 | 連續 500 iter 內 loss 範圍 > 10 |
| CSR | 生成樣本 CSR < 0.3（無法滿足約束） |
| Mode collapse | 生成樣本特徵多樣性 < 訓練集的 20% |

---

## 6. Evaluation Protocol

### 6.1 對抗樣本生成集合未定義 🟡 ML

spec.md 描述了訓練/驗證/測試切分，但未指定對抗樣本應在哪個集合上生成：

| 用途 | 應使用的集合 | 原因 |
|------|-------------|------|
| 攻擊 hyperparameter 調整（ε、steps）| Validation set | 避免在測試集上過擬合攻擊參數 |
| 最終 ASR 報告 | Test set | 與乾淨 F1 使用相同集合才具可比性 |
| 對抗訓練的對抗樣本 | Training set | 不能用測試集資料訓練模型 |

**建議：** 在 `configs/attack/` 的每個攻擊設定中明確加入 `target_split: [val|test|train]`。

### 6.2 `eval/comparison.py` 強耦合 🟡 SE

**問題：** 若 `comparison.py` 直接 import 每個模型類別與每個攻擊類別，新增一個模型就需要修改評估腳本本身。

**建議：** 使用 Hydra 的 `instantiate` 搭配設定檔動態載入：

```python
# eval/comparison.py
from hydra.utils import instantiate

model = instantiate(cfg.model)    # 由 configs/model/*.yaml 決定
attack = instantiate(cfg.attack)  # 由 configs/attack/*.yaml 決定
```

這樣新增模型只需新增一個 YAML 設定，不需修改評估邏輯。

### 6.3 跨模型轉移率計算的細節缺失 🟢 ML

spec.md 定義了「Transfer Rate = ASR when adv. examples generated on Model A tested on Model B」，但未說明：

- 對抗樣本在 Model A 的 train/test 集上生成？
- 轉移後在 Model B 的哪個集合評估？
- 是否需要 Model A 與 Model B 使用相同的資料切分？（對靜態 vs. 時序模型而言，圖格式不同，直接轉移需要特別處理）

---

## 7. Cross-Cutting Concerns

### 7.1 Checkpointing 與訓練恢復 🟡 SE

spec.md 提到使用 W&B 記錄實驗，但未規劃 checkpoint 儲存策略。TGN 在 RTX 3090 上訓練 2.5M 流量預計需數小時，若中途中斷無法恢復。

**建議在 `utils/` 加入：**

```python
# utils/checkpoint.py
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
```

### 7.2 型別提示與 Docstring 標準 🟢 SE

spec.md 未定義程式碼風格規範。建議在 `pyproject.toml` 中已配置的 `ruff` 基礎上，統一採用：

- **型別提示：** Python 3.12 原生 `list[int]`、`dict[str, Any]` 語法（不需 `from typing import List`）
- **Docstring 格式：** Google style（已與 Sphinx 和大多數 IDE 相容）

```python
def generate(self, model: BaseNIDSModel, data: TemporalData, epsilon: float) -> TemporalData:
    """Generate adversarial examples using constrained PGD.

    Args:
        model: Target NIDS model.
        data: Input graph data.
        epsilon: Maximum L-inf perturbation budget.

    Returns:
        Perturbed graph data with CSR = 1.0 guarantee.
    """
```

### 7.3 `data/adversarial/` 版本化問題 🟢 SE

不同超參數（ε、steps、n_inject）生成的對抗樣本若都儲存在同一個目錄下，容易產生覆蓋或混淆。建議目錄結構帶入超參數：

```
data/adversarial/
├── cpgd/eps0.1_steps40/
│   ├── graphsage_test.pt
│   └── tgat_test.pt
├── edge_injection/n50/
└── gan/
```

---

## 8. Risk Summary Table

| 編號 | 問題 | 面向 | 嚴重度 | 影響 | 建議行動 |
|------|------|------|--------|------|----------|
| R1 | `constraints.py` 位置錯誤 | SE | 🔴 | 模組依賴混亂，難以維護 | 移至 `src/attack/` |
| R2 | PGD 使用 `sign(∇L)` | ML | 🔴 | 連續特徵擾動次優，影響 ASR 可信度 | 改為 normalized gradient 或明確指定 `L∞` 範數 |
| R3 | 無 ABC 介面 | SE | 🔴 | `comparison.py` 無法統一呼叫，無法擴充 | 新增 `BaseNIDSModel`、`BaseAttack` |
| R4 | 無全域種子策略 | ML | 🟡 | 實驗不可完整重現 | 新增 `utils/seed.py`，Hydra cfg 注入 seed |
| R5 | 靜態圖無分批載入 | SE | 🟡 | 訓練時 OOM | 改用 PyG `Dataset`（按需載入） |
| R6 | TGN 記憶體重置策略未定 | ML | 🟡 | 不同攻擊的 ASR 不具可比性 | 在攻擊設定中定義 `memory_reset_policy` |
| R7 | 類別不平衡處理不一致 | ML | 🟡 | GAT / TGAT / TGN 偏向預測 benign | 所有模型設定統一加入 weighted loss |
| R8 | 對抗樣本生成集合未定義 | ML | 🟡 | 測試集洩漏風險 | 在攻擊設定中明確 `target_split` |
| R9 | `comparison.py` 強耦合 | SE | 🟡 | 新增模型需修改評估腳本 | 改用 Hydra `instantiate` |
| R10 | Checkpoint 策略缺失 | SE | 🟡 | 長時間訓練中斷無法恢復 | 新增 `utils/checkpoint.py` |
| R11 | 邊注入時機無形式化定義 | ML | 🟡 | 時序攻擊效果無法最佳化 | 補充搜尋策略定義 |
| R12 | Scaler 未序列化 | ML | 🟢 | 攻擊模組無法做 inverse transform | 存為 `data/processed/*/scaler.pkl` |
| R13 | 型別提示與 Docstring 未規範 | SE | 🟢 | 程式碼可讀性不一致 | 採用 Google-style docstring + PEP 526 |
| R14 | 對抗樣本目錄無版本化 | SE | 🟢 | 不同參數結果互相覆蓋 | 在路徑中帶入超參數 |

---

## 9. Recommended Action Items

以下行動項目依建議實作順序排列（後期項目依賴前期完成）。

### Phase 0：基礎設施（開始實作前完成）

```
[ ] 新增 src/models/base.py (BaseNIDSModel ABC)
[ ] 新增 src/attack/base.py (BaseAttack ABC)
[ ] 將 src/data/constraints.py → src/attack/constraints.py
[ ] 新增 src/utils/seed.py
[ ] 新增 src/utils/checkpoint.py
[ ] 所有模型設定統一加入 weighted cross-entropy
[ ] 所有攻擊設定加入 target_split 欄位
```

### Phase 1：資料 Pipeline 修正（Static Graph Builder）

```
[ ] 改用 PyG Dataset（按需載入）取代一次性載入
[ ] 新增時序切分驗證 assert（timestamp 不重疊）
[ ] static_builder / temporal_builder 執行後序列化 scaler.pkl
```

### Phase 2：模型實作

```
[ ] GraphSAGE / GAT 繼承 BaseNIDSModel
[ ] TGAT / TGN 繼承 BaseNIDSModel，並在 forward() 中暴露 memory_state
[ ] 在 configs/model/ 中為每個模型定義 memory_reset_policy（TGAT/TGN）
```

### Phase 3：CAAG 實作

```
[ ] C-PGD 改用 normalized gradient（或明確記錄使用 L∞ + sign）
[ ] 對 edge_injection 定義並實作時機搜尋策略
[ ] GAN 訓練加入不穩定早停判斷邏輯
[ ] 攻擊樣本輸出路徑帶入超參數（避免覆蓋）
```

### Phase 4：評估重構

```
[ ] comparison.py 改用 Hydra instantiate 動態載入模型與攻擊
[ ] 明確分離：val set 用於攻擊超參數調整，test set 用於最終報告
[ ] 定義 transfer 實驗的集合歸屬（特別是靜態 vs 時序跨格式轉移）
```

---

> **審核版本：** 0.1 | **參照規格：** `docs/spec.md` v0.1.0 | **日期：** 2026-03
