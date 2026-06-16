# Architecture Review: GNN-NIDS Analyzer v2.0

> 以 AI/Web 工程師的角度，針對實用工具方向的架構與效能進行系統性審核。
> 涵蓋 Backend（FastAPI）、Frontend（Vue 3 + Vite）、API 設計、安全性與可部署性。

**審核版本：** 2.0 | **參照規格：** `docs/spec.md` v1.0.0 | **日期：** 2026-06-16  
**前版：** v1.0（2026-03）— v1.0 提出的 15 個問題中 13 個已修復（87%），本版聚焦未解決問題與新增功能審核。

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [待解決問題](#2-待解決問題)
3. [新增功能審核](#3-新增功能審核)
4. [Risk Summary Table](#4-risk-summary-table)
5. [Recommended Action Items](#5-recommended-action-items)
6. [GNN 訓練效能紀錄](#6-gnn-訓練效能紀錄)

---

## 1. Executive Summary

架構從 v1.0 的「雙模型 demo」進化為支持 5 個 GNN 模型 + ensemble 的完整系統。核心 v1.0 問題（event loop 阻塞、session 管理、安全驗證、前端效能）均已修復。當前關注點轉向部署便利性和新功能的存取控制。

### 當前架構概覽

| 面向 | 現狀 |
|------|------|
| 模型 | 6 個（GraphSAGE, GAT, E-GraphSAGE, TGAT, TGN, Ensemble） |
| 推論 | `run_in_threadpool` 非同步，不阻塞 event loop |
| 即時推論 | WebSocket `/ws/stream` |
| 可解釋性 | GNNExplainer (static) + attention-based (temporal) |
| 對抗 | C-PGD / Edge Injection / GAN 攻擊 + adversarial training |
| Session | lifespan cleanup task, 1 小時 TTL |
| 安全 | UUID 驗證 + JSON scaler + 50MB 上傳限制 + CORS env var |
| 前端 | plotly-basic + manualChunks + 後端 spring_layout + Pinia 輪詢管理 |

### 當前主要問題

| 優先級 | 問題 | 面向 |
|--------|------|------|
| ✅ | Docker GPU 部署已就緒（Dockerfile + docker-compose + nginx proxy） | DEP |
| 🟡 | WebSocket 串流推論缺少認證和 rate limiting | SEC |
| 🟡 | Memory poisoning 端點允許外部修改 TGN 記憶體，需要存取控制 | SEC |
| 🟢 | 模型 checkpoint 用 `weights_only=False` 載入（受信環境可接受） | SEC |
| ✅ | 前端已整合 explainability UI（靜態 + temporal） | FE |

---

## 2. 待解決問題

### ~~2.1 Docker 設定~~ ✅ 已解決

GPU-enabled Docker 部署已就緒：`Dockerfile`（PyTorch 2.6.0 + CUDA 12.4）、`frontend/Dockerfile`（nginx SPA + API proxy）、`docker-compose.yml`（NVIDIA runtime + healthcheck）。README 已加入 Docker Quick Start。

### 2.2 WebSocket 缺少存取控制 🟡 SEC

**問題：** `/api/ws/stream` 接受任何 WebSocket 連線，無認證或 rate limiting。惡意用戶可大量連線消耗 GPU/CPU 資源。

**建議：** 加入並發連線數限制和 token 驗證：

```python
MAX_CONCURRENT_WS = int(os.getenv("MAX_CONCURRENT_WS", "5"))
_ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)

@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    if not _ws_semaphore.locked() or _ws_semaphore._value > 0:
        async with _ws_semaphore:
            # ... existing logic
    else:
        await ws.close(code=1013, reason="Server busy")
```

### 2.3 Memory Poisoning 端點存取控制 🟡 SEC

**問題：** `POST /api/memory-poisoning` 允許修改 TGN 模型的內部記憶體。這是研究用端點，不應在生產環境暴露。

**建議：** 加入環境變數控制：

```python
ENABLE_ATTACK_ENDPOINTS = os.getenv("ENABLE_ATTACK_ENDPOINTS", "false").lower() == "true"
```

### 2.4 模型 checkpoint pickle 載入 🟢 SEC

**現狀：** `torch.load(path, weights_only=False)` 使用 pickle 反序列化。受信環境可接受，但若 checkpoint 來自外部源則有 RCE 風險。

**建議（公開部署時）：** 改為 `state_dict` 載入，需同時儲存 Hydra config 以恢復模型類別。

### ~~2.5 前端 temporal explainability UI~~ ✅ 已解決

ExplainView 元件已整合至前端：支援 5 種模型的 Top-K 解釋、特徵重要性長條圖、AlertList 快捷按鈕。靜態模型使用 GNNExplainer，temporal 模型使用 integrated gradients。

---

## 3. 新增功能審核

### 3.1 WebSocket 串流推論

`app/routers/streaming.py`：接收 JSON NetFlow records → 時間視窗緩衝 → 建圖 → GNN 推論 → 即時推送告警。支援 `flush` 命令。

**架構正確性：** 良好。WebSocket handler async，GNN 推論透過 thread executor 不阻塞其他連線。

### 3.2 GNN 可解釋性

- `src/explain/gnn_explainer.py` — static models（GraphSAGE, GAT, E-GraphSAGE）edge-level 解釋
- `src/explain/temporal_explainer.py` — temporal models attention-based 解釋
- API：`POST /api/explain/{id}` 和 `POST /api/explain-top/{id}`

### 3.3 對抗訓練

`src/defense/adversarial_training.py` — 混合 clean + C-PGD adversarial batches，可配置 epsilon/steps/ratio。已驗證 GraphSAGE (+0.0041 F1) 和 GAT (+0.0087 F1) 效果。

### 3.4 E-GraphSAGE

`src/models/egraphsage.py` — 自訂 `EGraphSAGEConv(MessagePassing)` 實作 edge-feature-aware message passing。Test F1 = 0.9708，Macro F1 = 0.4681（所有個別模型中最高）。遵循 `BaseNIDSModel` ABC，完全整合現有管線。

### 3.5 Ensemble

`src/models/ensemble.py` — Soft/Hard/Weighted voting + `from_validation()` 自動學習權重。3-model ensemble F1 = 0.9700。

### 3.6 Temporal Models（TGAT / TGN）

- TGAT：stateless temporal attention
- TGN：GRU-based persistent memory
- `LastNeighborLoader` CPU ring buffers for temporal neighbourhood sampling

### 3.7 研究工具

| 腳本 | 用途 |
|------|------|
| `scripts/multi_seed_eval.py` | 多 seed 訓練，統計顯著性 |
| `scripts/cross_dataset_validation.py` | 跨資料集泛化評估 |
| `scripts/window_size_eval.py` | 時間視窗大小敏感性分析 |
| `scripts/compute_reliability_metrics.py` | 全模型可靠性指標 + per-class 報告 |

### 3.8 API 端點總覽

| 方法 | 路由 | 功能 | 非同步 |
|------|------|------|--------|
| POST | `/api/analyze` | CSV 上傳 + GNN 推論 | ✅ threadpool |
| POST | `/api/adversarial` | C-PGD 攻擊 | ✅ threadpool + 30s timeout |
| GET | `/api/report/{id}` | PDF 報告 | ✅ threadpool |
| WS | `/api/ws/stream` | 即時串流推論 | ✅ async |
| POST | `/api/explain/{id}` | GNNExplainer | ✅ threadpool |
| POST | `/api/explain-top/{id}` | 前 K 告警解釋 | ✅ threadpool |
| POST | `/api/memory-poisoning` | TGN memory poisoning | ✅ threadpool |

---

## 4. Risk Summary Table

| 編號 | 問題 | 面向 | 嚴重度 | 建議行動 |
|------|------|------|--------|----------|
| ~~R11~~ | ~~Docker 部署~~ | DEP | ✅ | 已完成：GPU Dockerfile + compose + nginx |
| R16 | WebSocket 缺認證和 rate limiting | SEC | 🟡 | token 驗證 + 並發限制 |
| R17 | Memory poisoning 端點無存取控制 | SEC | 🟡 | 環境變數開關 |
| R18 | torch.load weights_only=False | SEC | 🟢 | 公開部署改 state_dict |
| ~~R19~~ | ~~前端 explainability UI~~ | FE | ✅ | 已完成：ExplainView + AlertList 按鈕 |

---

## 5. Recommended Action Items

### 高優先（公開部署前）

```
[x] R11: Docker GPU 部署（Dockerfile + docker-compose + nginx proxy + healthcheck）
[ ] R16: WebSocket 加入連線數限制和 token 驗證
[ ] R17: Memory poisoning 端點加入環境變數開關
```

### 中優先（功能完善）

```
[x] R19: 前端整合 explainability（ExplainView + Top-K + AlertList 按鈕）
```

### 低優先（長期改善）

```
[ ] R18: 模型載入改為 state_dict 模式
[ ] 加入 API rate limiting middleware
[ ] 前端新增模型比較 dashboard
```

---

## 6. GNN 訓練效能紀錄

### 6.1 核心問題：Temporal Distribution Shift

NF-UNSW-NB15-v2 的攻擊流量在時間軸上分布不均：

| 資料集切割 | 視窗數 | Benign 比例 | Attack 比例 |
|-----------|--------|------------|------------|
| Train (60%) | 1,289 | 26.8% | 73.2% |
| Val (20%) | 430 | 19.6% | 80.4% |
| Test (20%) | 430 | 59.3% | 40.7% |

### 6.2 關鍵優化

| 優化 | 效果 |
|------|------|
| Focal Loss (γ=2.0) + class weights | 聚焦困難樣本，處理類別不平衡 |
| Proxy Node Identity (TTL+Protocol) | 2 節點 → ~20-50 節點/視窗，val F1 0.50 → 0.84 |
| Window Size 120s | graph density 和樣本量平衡 |
| AMP (bfloat16) | 訓練速度 ~1.5-2× |

### 6.3 當前最佳結果（2026-06-16）

| 模型 | Test F1 | Precision | Recall | ROC-AUC | Macro F1 |
|------|---------|-----------|--------|---------|----------|
| GraphSAGE + adv | 0.9753 | 0.9803 | 0.9727 | 0.9997 | — |
| GraphSAGE | 0.9712 | 0.9792 | 0.9660 | 0.9992 | 0.4657 |
| E-GraphSAGE | 0.9708 | 0.9784 | 0.9665 | 0.9991 | 0.4681 |
| Ensemble (3-model) | 0.9700 | 0.9794 | 0.9650 | 0.9992 | 0.4510 |
| GAT + adv | 0.9622 | 0.9696 | 0.9581 | 0.9965 | — |
| GAT | 0.9534 | 0.9729 | 0.9433 | 0.9963 | 0.3164 |
| TGAT | 0.9475 | 0.9632 | 0.9391 | 0.9963 | 0.3643 |
| TGN | 0.9463 | 0.9610 | 0.9351 | 0.9960 | 0.3438 |

### 6.4 後續優化方向

1. **E-GraphSAGE 對抗訓練**：edge-feature-aware message passing 是否更能受益於 adversarial training。
2. **Macro F1 優化**：當前 0.31–0.47，少數攻擊類別辨識能力不足。可嘗試 oversampling 或 threshold calibration。
3. **輕量 temporal 替代**：GraphMixer / SimpleDyG 可能降低 temporal 訓練時間。

---

> 下一次審核建議在公開部署前進行，重點驗證：Docker 端到端功能、WebSocket 存取控制、攻擊端點隔離策略。
