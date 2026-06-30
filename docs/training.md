# 訓練指南

GNN-TGAT-NIDS 的訓練細節、優化策略與研究工具。

---

## 前置需求（本機開發）

- Python 3.12+、[uv](https://docs.astral.sh/uv/)、Node.js 20+
- CUDA 12.4+（選配；小型資料集可用 CPU 模式）

```bash
git clone https://github.com/SoWiEee/GNN-TGAT-NIDS.git
cd GNN-TGAT-NIDS

uv sync
uv run pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

uv run pytest  # 驗證安裝
```

或在 Docker 容器內操作：

```bash
docker compose exec backend bash
```

---

## 資料準備

```bash
# 合併 UNSW-NB15 訓練 + 測試 CSV → data/raw/NF-UNSW-NB15-v2.csv
# 同時建立 data/demo/demo_flows.csv（1,000 筆分層抽樣子集）
uv run python scripts/create_demo_dataset.py

# 從合併後的 CSV 建構 PyG 時間視窗圖
uv run python src/data/static_builder.py

# 建構時序圖（TGAT/TGN 需要）
uv run python src/data/temporal_builder.py
```

---

## 模型訓練

```bash
# 靜態模型
uv run python train.py model=graphsage
uv run python train.py model=gat

# 時序模型
uv run python train.py model=tgat data=temporal_default
uv run python train.py model=tgn data=temporal_default

# Early stopping
uv run python train.py model=graphsage train.patience=30

# 在 Docker 內訓練
docker compose exec backend uv run python train.py model=graphsage
```

---

## 訓練優化

### 代理節點身份（Port/Protocol）

NF-UNSW-NB15-v2 的處理後 CSV 不含 IP 位址欄位，直接使用會導致每個時間視窗只產生退化的 2 節點圖。`static_builder.py` 使用兩種代理節點策略：

**策略 1：TTL 分桶**（適用於含 sttl/dttl 欄位的資料集）

| 角色 | 鍵值 | 原理 |
|---|---|---|
| 來源節點 | `("src", sttl // 16, proto)` | TTL 分桶 ≈ 作業系統類型 + 協定 |
| 目的節點 | `("dst", dttl // 16, service)` | TTL + 服務 ≈ 伺服器區段 |

**策略 2：Port/Protocol**（適用於 NF 格式資料集，無 TTL 欄位）

| 角色 | 鍵值 | 原理 |
|---|---|---|
| 來源節點 | `("src", L7_PROTO, PROTOCOL)` | 應用層協定 + 傳輸協定 |
| 目的節點 | `("dst", L4_DST_PORT, PROTOCOL)` | 目的埠號 + 傳輸協定 |

兩種策略皆可產生每個視窗 **~40-70 個不同節點**，使 GNN 的訊息傳遞機制有意義。

### 時間視窗大小

預設視窗：**120 秒**（`configs/data/static_default.yaml`）。較大的視窗會產生更密集的圖（~120 條邊/視窗），但總視窗數較少。

### 自動混合精度（AMP）

在 CUDA 上預設啟用，透過 `train.use_amp=true`。使用 `torch.amp.autocast` + `GradScaler`——通常快 **1.5-2 倍**。以 `train.use_amp=false` 停用。

### DataLoader 調校

| 設定鍵 | 預設值 | 效果 |
|---|---|---|
| `train.batch_size` | `32` | 每批次的圖視窗數 |
| `train.num_workers` | `0` | Linux 上設為 `4` 可非同步載入 |
| `train.val_every` | `1` | 每 N 個 epoch 評估一次；`5` 可節省 ~20% 時間 |
| `train.save_every` | `10` | 定期存檔頻率 |

---

## Macro F1 改善

NF-UNSW-NB15-v2 存在極端類別不平衡（Benign 93%，最稀少的 5 種攻擊類型 < 1%），導致 weighted F1 高但 macro F1 低。以下機制可改善此問題：

### 1. 類別權重策略

```yaml
train:
  class_weight_strategy: sqrt_inverse  # "inverse" | "effective" | "sqrt_inverse"
```

| 策略 | 公式 | 效果 |
|---|---|---|
| `inverse` | `N / (C × count_c)` | 標準逆頻率。可能產生 5000:1 的權重比 |
| `effective` | `(1-β) / (1-β^count_c)` | Cui et al. 2019。更平滑的縮放，但在極端不平衡下會崩潰 |
| `sqrt_inverse` | `√(N / (C × count_c))` | 阻尼逆頻率。在此資料集上表現最佳 |

### 2. 驗證指標選擇

```yaml
train:
  val_metric: macro_f1  # "f1"（weighted）| "macro_f1"
```

使用 `macro_f1` 進行 checkpoint 選擇，迫使模型優化稀有類別的表現，而非被 Benign 主導的 weighted F1。

### 3. Focal Loss Gamma

```yaml
train:
  focal_gamma: 1.0  # 預設 2.0；Optuna 搜索發現 1.0 效果更佳
```

### 4. 邊級過取樣

```yaml
train:
  oversample_factor: 20  # 稀有類別的邊級過取樣倍數（1 = 停用）
```

高過取樣因子（9-20）對稀有類別的 recall 至關重要。

### 5. LR 排程器

```yaml
train:
  scheduler: cosine  # "none" | "cosine" | "plateau"
```

Cosine annealing（`CosineAnnealingLR`）將學習率從 `train.lr` 衰減至初始值的 1%。防止在稀有類別梯度上過度更新。

### 最佳設定（Optuna 調校後）

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

### 實驗結果

| 模型 | 設定 | Weighted F1 | Macro F1 | Recall |
|-------|--------|:-----------:|:--------:|:------:|
| GraphSAGE | 基線（`inverse`, γ=2.0, val=f1） | 0.1063 | 0.1645 | 0.0639 |
| GraphSAGE | Optuna 最佳（`sqrt_inverse`, γ=1.0, oversample=20） | **0.9773** | **0.5735** | **0.9742** |

### 每類別 F1（GraphSAGE，Optuna 最佳設定）

| 類別 | 樣本數 | Precision | Recall | F1 |
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

**關鍵發現：**
- `sqrt_inverse` 類別權重一致優於 `inverse` 和 `effective`
- 高 `oversample_factor`（9-20）對稀有類別的 recall 至關重要
- 較低的 `focal_gamma`（1.0）比預設的 2.0 效果更好
- 2 層模型優於較深的 3-4 層架構
- 混淆主要在攻擊子類型之間（DoS↔Exploits、Generic↔DoS/Exploits），非 benign vs attack

---

## 超參數搜索（Optuna）

```bash
uv sync --group dev

# 靜態模型
uv run python scripts/tune_hyperparams.py --model graphsage --trials 50
uv run python scripts/tune_hyperparams.py --model gat --trials 50

# 時序模型
uv run python scripts/tune_hyperparams.py --model tgat --trials 30
uv run python scripts/tune_hyperparams.py --model tgn --trials 30

# 即時儀表板
uv run optuna-dashboard sqlite:///results/optuna.db
```

支援中斷恢復（SQLite 儲存）。最佳參數儲存於 `results/best_hparams_{model}.json`。

> **注意：** 靜態和時序模型的 Optuna 搜索均以 **macro F1** 為最佳化目標，確保稀有攻擊類型也得到適當的調校。

**搜索空間（靜態模型）：**

| 超參數 | 範圍 / 選項 |
|---|---|
| `lr` | 1e-4 ~ 1e-2（對數尺度）|
| `focal_gamma` | 0.5 ~ 3.0（步長 0.5）|
| `hidden_dim` | 128 / 256 / 512 |
| `num_layers` | 2 / 3 / 4 |
| `dropout` | 0.0 ~ 0.5 |
| `batch_size` | 16 / 32 / 64 |
| `oversample_factor` | 1 ~ 20 |
| `weight_strategy` | inverse / sqrt_inverse / effective |
| `num_heads`（GAT）| 2 / 4 / 8 |
| `aggregation`（SAGE）| mean / max |

**搜索空間（時序模型）：**

| 超參數 | 範圍 / 選項 |
|---|---|
| `lr` | 1e-4 ~ 1e-2（對數尺度）|
| `hidden_dim` | 64 / 128 / 172 / 256 |
| `heads` | 1 / 2 / 4 |
| `n_neighbors` | 10 / 20 / 30 |
| `batch_size` | 100 / 200 / 400 |
| `memory_dim`（TGN）| 64 / 100 / 128 |

**套用最佳參數：**

```bash
# 範例：套用 Optuna 搜索結果
uv run python train.py model=graphsage \
  model.hidden_dim=256 model.num_layers=2 model.dropout=0.0 \
  train.lr=0.00124 train.focal_gamma=1.0 \
  train.oversample_factor=20 \
  train.class_weight_strategy=sqrt_inverse \
  train.val_metric=macro_f1 \
  train.scheduler=cosine \
  train.patience=30 \
  train.epochs=200
```

---

## 對抗訓練

```bash
uv run python train.py model=graphsage train.adversarial_training=true \
    train.epochs=30 train.patience=10 \
    train.checkpoint_dir=checkpoints/graphsage_adv

uv run python train.py model=gat train.adversarial_training=true \
    train.adv_epsilon=0.1 train.adv_steps=10 train.adv_ratio=0.3
```

儲存為 `{model}_adv_best.pt`。`compute_reliability_metrics.py` 會自動載入 `_adv_best.pt` 檔案。

### 訓練結束時的 Per-class F1 輸出

`train.py` 在最終測試評估後會自動輸出每類別的 Precision/Recall/F1，方便在不執行完整 reliability 腳本的情況下快速檢視各攻擊類型的表現。

---

## 模型集成

```python
from src.models.ensemble import EnsembleModel

ensemble = EnsembleModel(
    models={"graphsage": model_gs, "gat": model_gat},
    strategy="soft_vote",  # 或 "hard_vote"、"weighted"
)
proba = ensemble(data)
```

API 的 `/api/analyze` 接受 `model=ensemble` 進行集成推論。權重透過 `EnsembleModel.from_validation()` 從驗證 F1 學習。

---

## 串流推論（WebSocket）

```
ws://localhost:8000/api/ws/stream?model=graphsage&window_seconds=60
```

- 傳送：`{"flows": [{"col1": val, ...}, ...]}` — 緩衝至時間視窗
- 傳送：`{"command": "flush"}` — 強制對緩衝的流進行推論
- 接收：`{"type": "alerts", "window": 0, "alerts": [...], "stats": {...}}`

---

## 可解釋性

```python
from src.explain.gnn_explainer import explain_flow, explain_top_alerts

result = explain_flow(model, data, edge_idx=42, epochs=200,
                      feature_names=feature_cols, class_names=class_names)
results = explain_top_alerts(model, data, top_k=5,
                             feature_names=feature_cols, class_names=class_names)
```

- 靜態模型（GraphSAGE、GAT）：GNNExplainer，顯示語義特徵名稱與類別名稱
- 時序模型（TGAT、TGN）：整合梯度近似法

API：`POST /api/explain/{session_id}`、`POST /api/explain-top/{session_id}`

---

## ONNX 匯出與量化

將訓練好的靜態模型匯出為 ONNX 格式，可選 INT8 動態量化以縮小模型體積：

```bash
# 基本匯出
uv run python scripts/export_onnx.py --model graphsage

# 匯出 + INT8 量化
uv run python scripts/export_onnx.py --model graphsage --quantize
uv run python scripts/export_onnx.py --model gat --quantize
```

時序模型（TGAT、TGN）因具有狀態記憶體，無法匯出為 ONNX。

### 模型大小

| 格式 | 大小 | 相對基線 |
|------|-----:|:--------:|
| PyTorch checkpoint（FP32） | 1,612 KB | 100% |
| ONNX FP32 | 1,613 KB | 100% |
| ONNX INT8 量化 | **433 KB** | **27%（減少 73%）** |

### 推論精度比較

| 比較 | 預測一致率 | 最大 logit 差異 |
|------|:----------:|:--------------:|
| PyTorch vs ONNX-FP32 | 100.00% | 0.000023 |
| PyTorch vs ONNX-INT8 | 99.17% | 0.914536 |

INT8 量化僅犧牲極微小的精度（< 1% 預測差異），換取 73% 的模型體積縮減。

### CPU 推論效能

| 執行環境 | 平均延遲 | P95 延遲 |
|---------|:--------:|:--------:|
| PyTorch CPU | 0.96 ms | 3.70 ms |
| ONNX FP32 | **0.77 ms** | 0.89 ms |
| ONNX INT8 | 0.99 ms | 1.22 ms |

ONNX FP32 比 PyTorch CPU 快約 20%，且 P95 延遲更穩定。INT8 在此模型規模下速度提升不明顯（模型本身就很小），但體積優勢在邊緣部署場景下非常重要。

### 部署場景

| 場景 | 推薦格式 | 原因 |
|------|---------|------|
| 有 GPU 的伺服器 | PyTorch（CUDA） | 最高吞吐量 |
| 無 GPU 的桌機/筆電 | ONNX FP32 | 去除 PyTorch 依賴，僅需 onnxruntime |
| 邊緣設備 / IoT 閘道 | ONNX INT8 | 433 KB 體積，適合記憶體受限環境 |
| 學校實驗室 | ONNX FP32/INT8 | 任何 4GB RAM 的機器皆可運行 |

### ONNX Runtime 推論範例

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("exports/graphsage.quant.onnx",
                            providers=["CPUExecutionProvider"])

logits = sess.run(None, {
    "x": node_features,        # (num_nodes, 5)
    "edge_index": edge_index,   # (2, num_edges), int64
    "edge_attr": edge_features, # (num_edges, 41)
})[0]

predictions = logits.argmax(axis=1)  # (num_edges,)
```

---

## 可靠性指標

```bash
uv run python scripts/compute_reliability_metrics.py
```

執行乾淨評估 + C-PGD 攻擊評估 + 對抗訓練評估。結果儲存至 `data/metrics/reliability.json`。

---

## 研究腳本

| 腳本 | 用途 |
|---|---|
| `scripts/multi_seed_eval.py` | 多種子訓練以驗證統計顯著性 |
| `scripts/cross_dataset_validation.py` | 跨資料集泛化能力評估 |
| `scripts/window_size_eval.py` | 時間視窗大小敏感度分析 |
| `scripts/compute_reliability_metrics.py` | 完整模型可靠性指標 + 每類別報告 |
| `scripts/tune_hyperparams.py` | Optuna 貝氏超參數搜索 |
| `scripts/export_onnx.py` | ONNX 匯出 + 選配量化 |
| `scripts/pcap_to_netflow.py` | PCAP 轉 NetFlow CSV（nfstream）|

---

## 部署設定

### Docker Compose

```bash
docker compose up --build    # 後端 :8000 + 前端 :80
```

後端 Dockerfile 已包含 `HEALTHCHECK`，前端等後端 healthy 後才啟動。

### 環境變數

後端（`.env` 或 `docker-compose.yml` environment）：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ALLOWED_ORIGINS` | `http://localhost:5173` | CORS 允許來源（逗號分隔） |
| `ENABLE_ATTACK_ENDPOINTS` | `true` | 攻擊端點開關（生產環境設 `false`） |
| `MAX_CONCURRENT_INFER` | `3` | 最大並發推論數（防止 GPU OOM） |
| `MAX_CONCURRENT_WS` | `5` | 最大 WebSocket 並發連線數 |
| `MAX_UPLOAD_BYTES` | `52428800` | 上傳檔案大小上限（bytes） |
| `ADV_TIMEOUT_SECONDS` | `30.0` | 對抗樣本生成逾時（秒） |
| `SESSION_TTL_SECONDS` | `3600` | Session 存活時間（秒） |
| `CLEANUP_INTERVAL_SECONDS` | `300` | Session 清理檢查間隔（秒） |

前端（`.env.local`）：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | 後端 API 位址 |
| `VITE_MAX_UPLOAD_BYTES` | `52428800` | 前端上傳檔案大小上限 |
| `VITE_POLL_INTERVAL_MS` | `2000` | 狀態輪詢間隔（毫秒） |

### CI/CD

GitHub Actions（`.github/workflows/ci.yml`）：
- **Lint**：ruff check
- **Test**：CPU PyTorch + pytest（含覆蓋率報告）
- **Frontend**：vue-tsc 型別檢查 + vite build
