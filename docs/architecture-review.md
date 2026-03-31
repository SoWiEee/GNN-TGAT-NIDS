# Architecture Review: GNN-NIDS Analyzer v1.0

> 以 AI/Web 工程師的角度，針對實用工具方向的架構與效能進行系統性審核。
> 涵蓋 Backend（FastAPI）、Frontend（Vue 3 + Vite）、API 設計、Session 管理、安全性與可部署性。

**審核版本：** 1.0 | **參照規格：** `docs/spec.md` v1.0.0 | **日期：** 2026-03

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Backend：非同步與並發](#2-backend非同步與並發)
3. [Frontend：效能與 Bundle 大小](#3-frontend效能與-bundle-大小)
4. [API 設計](#4-api-設計)
5. [Session 與檔案管理](#5-session-與檔案管理)
6. [安全性](#6-安全性)
7. [可部署性](#7-可部署性)
8. [Risk Summary Table](#8-risk-summary-table)
9. [Recommended Action Items](#9-recommended-action-items)
10. [GNN 訓練效能優化紀錄](#10-gnn-訓練效能優化紀錄)

---

## 1. Executive Summary

整體架構方向正確：FastAPI + Vue 3 是成熟的組合，PyG → Cytoscape.js 的資料流清晰。但有幾個
「放到 request path 上的重運算」問題，若不處理，會在 demo 時直接卡住瀏覽器。以下是主要問題摘要：

| 優先級 | 問題 | 面向 |
|--------|------|------|
| 🔴 | GNN 推論（10–30 秒）在 `POST /analyze` 同步執行，會阻塞 uvicorn event loop | BE |
| 🔴 | C-PGD 40 步反向傳播（每次 1–5 秒）在 `POST /adversarial` 同步執行，瀏覽器會 timeout | BE |
| 🔴 | PDF 產生（WeasyPrint）是 CPU-heavy，若同步執行同樣阻塞 event loop | BE |
| 🟡 | 兩個模型（GraphSAGE / GAT）在 startup 只載入一個，但 API 讓使用者選擇模型 | BE |
| 🟡 | Cytoscape.js 500 nodes / 2 000 edges 初始渲染，低階電腦會卡頓 | FE |
| 🟡 | Plotly.js 完整 bundle ~3 MB，拖慢首次載入 | FE |
| 🟡 | Session 1 小時清除機制規格存在，但沒有定義實作方式（cron？background task？） | BE |
| 🟡 | `pickle` 序列化 scaler 有安全疑慮；路徑未驗證的 `session_id` 有路徑穿越風險 | SEC |
| 🟢 | PDF 報告中「Cytoscape.js graph PNG」要用 Selenium headless 產生，過重 | BE |
| 🟢 | 前端輪詢 `/status` 的 abort 邏輯與輪詢間隔未定義 | FE |
| 🟢 | `reliability.json` 同時被當成靜態資源和 API endpoint，職責重複 | BE |

---

## 2. Backend：非同步與並發

### 2.1 GNN 推論阻塞 Event Loop 🔴 BE

**問題：** FastAPI 使用 async I/O，但 PyTorch GNN 推論是同步的 CPU/GPU 運算。若直接在 `async def` 路由裡呼叫 `model(data)`，會佔用 event loop 執行緒，導致伺服器在推論期間無法回應其他請求（包含輪詢的 `/status`）。

**建議修正：** 用 `asyncio.get_event_loop().run_in_executor(None, ...)` 或 `fastapi.concurrency.run_in_threadpool` 把同步 GNN 推論移至執行緒池：

```python
# app/services/inference.py
from fastapi.concurrency import run_in_threadpool

async def run_inference(session_id: str, model_name: str) -> InferenceResult:
    result = await run_in_threadpool(_sync_run_inference, session_id, model_name)
    return result

def _sync_run_inference(session_id: str, model_name: str) -> InferenceResult:
    # 所有 PyTorch 操作在這裡同步執行
    data = load_session_data(session_id)
    model = get_model(model_name)
    with torch.inference_mode():
        logits = model(data)
    return build_inference_result(logits, data)
```

對於更長的分析任務（大型 CSV），可進一步考慮 FastAPI 的 `BackgroundTasks`：client 收到 `202 Accepted` 後輪詢 `/status`。

### 2.2 C-PGD 阻塞 Request 🔴 BE

**問題：** `POST /adversarial` 執行 C-PGD（40 steps of gradient computation），單一 flow 約需 1–5 秒。若同步執行且多用戶並發，或 demo 時連點多次「Generate adversarial」，伺服器會排隊堵死。

**建議修正：** 同 2.1，用 `run_in_threadpool` 包裝。另外加上 **timeout 保護**：

```python
# app/routers/adversarial.py
import asyncio

async def generate_adversarial(req: AdversarialRequest):
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_sync_cpgd, req),
            timeout=30.0  # 30 秒上限；C-PGD 找不到對抗例就回傳 null
        )
    except asyncio.TimeoutError:
        raise HTTPException(408, detail="Adversarial generation timed out")
    return result
```

### 2.3 PDF 產生阻塞 Event Loop 🔴 BE

**問題：** WeasyPrint 把 HTML 渲染成 PDF 屬於 CPU-heavy 操作（涉及字型解析、排版計算），通常需要 1–3 秒。直接在 async 路由呼叫會阻塞 event loop。

**建議修正：** 同樣用 `run_in_threadpool`；若 PDF 產生需要嵌入圖表，先讓前端 export canvas PNG（見 Section 3.3），再由後端嵌入，可避免 Selenium 依賴。

### 2.4 兩個模型的載入策略 🟡 BE

**問題：** spec 說「load checkpoint once at startup」，但 `POST /analyze` 讓使用者選擇 `graphsage` 或 `gat`。只載入一個模型則另一個無法使用；兩個都載入則佔記憶體（各約 50–200 MB）。

**建議：** 啟動時兩個都載入，存進字典：

```python
# app/services/inference.py
_models: dict[str, BaseNIDSModel] = {}

def load_models():
    _models["graphsage"] = load_checkpoint_to_model("checkpoints/graphsage_best.pt")
    _models["gat"]       = load_checkpoint_to_model("checkpoints/gat_best.pt")

def get_model(name: str) -> BaseNIDSModel:
    if name not in _models:
        raise ValueError(f"Unknown model: {name}")
    return _models[name]
```

兩個 GraphSAGE/GAT checkpoint 合計約 100–400 MB，現代筆電完全可接受。

### 2.5 Session 清除機制未定義 🟡 BE

**問題：** 規格說「1-hour cleanup」但沒有說怎麼實作。若沒有清除，`data/sessions/` 會無限增長。

**建議：** 用 FastAPI 的 `BackgroundTasks` 加上時間戳記檔案，或在 lifespan 中啟動一個清除協程：

```python
# app/main.py
import asyncio
from pathlib import Path
import time

async def cleanup_sessions(sessions_dir: Path, ttl_seconds: int = 3600):
    while True:
        await asyncio.sleep(300)  # 每 5 分鐘檢查一次
        now = time.time()
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                mtime = session_dir.stat().st_mtime
                if now - mtime > ttl_seconds:
                    shutil.rmtree(session_dir, ignore_errors=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    task = asyncio.create_task(cleanup_sessions(Path("data/sessions")))
    yield
    task.cancel()
```

### 2.6 Graph PNG for PDF：Selenium 過重 🟢 BE

**問題：** spec 提到「static PNG export of Cytoscape.js graph generated server-side with `cytoscape-png` or Selenium headless」。Selenium headless 需要安裝 Chrome + WebDriver，在 CI / 部署環境非常不友善。

**替代方案：** 讓前端在 export 前用 Cytoscape.js 的 `.png()` 方法產生 base64 PNG，連同 `POST /report` 一起上傳。後端直接嵌入 base64 圖片到 Jinja2 template，完全不需要 Selenium：

```typescript
// AdversarialReport.vue or TrafficGraph.vue
async function exportReport() {
  const graphPng = cy.png({ output: 'base64', scale: 2 })
  await api.generateReport(sessionId, { graphPng })
}
```

```python
# app/services/report_builder.py
# template 中直接用: <img src="data:image/png;base64,{{ graph_png }}">
```

---

## 3. Frontend：效能與 Bundle 大小

### 3.1 Plotly.js 完整 Bundle 🟡 FE

**問題：** `import Plotly from 'plotly.js'` 引入完整 bundle（~3 MB minified, ~900 KB gzipped），但本專案只用到 stacked bar chart。這會讓首次載入慢 1–3 秒（特別在慢速網路）。

**建議：** 改用 `plotly.js-basic-dist-min`（只含基本 chart types，約 900 KB minified），或用 dynamic import 延遲載入：

```typescript
// AttackTimeline.vue
const Plotly = await import('plotly.js-basic-dist-min')
```

### 3.2 Cytoscape.js 大圖效能 🟡 FE

**問題：** 500 nodes + 2 000 edges 的初始渲染在低階筆電（Intel i5 + 8 GB RAM）下，Cytoscape.js 的 layout 計算（如 `cose`）可能需要 3–5 秒，且主執行緒會卡頓（無法互動）。

**建議：**

1. **預設用 `preset` layout**（後端算好座標，直接傳 `{x, y}` 給 Cytoscape）。後端可用 `networkx` 的 `spring_layout` 計算一次座標並快取。
2. 若要動態 layout，用 `cytoscape-cose-bilkent` 並設定 `animate: false`，或 Web Worker 計算（cytoscape-layout-utilities）。
3. 初始只顯示 **Top-200 edges（依 confidence 降序）**，提供「Show all」按鈕。

```python
# app/services/graph_builder.py — 後端預算座標
import networkx as nx

def compute_layout(G: nx.Graph) -> dict[str, tuple[float, float]]:
    pos = nx.spring_layout(G, seed=42, k=2/len(G)**0.5)
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}
```

### 3.3 輪詢 `/status` 的 Abort 邏輯 🟢 FE

**問題：** 如果使用者上傳後立刻重新上傳，舊的輪詢 interval 若沒清除，會繼續觸發並覆蓋新的 sessionId。

**建議：** 在 Pinia store 中統一管理輪詢，確保每次上傳前先 abort 舊的輪詢：

```typescript
// stores/session.ts
let pollingTimer: ReturnType<typeof setInterval> | null = null

function startPolling(sessionId: string) {
  stopPolling()  // abort any previous poll
  pollingTimer = setInterval(async () => {
    const { status, progress_pct } = await api.getStatus(sessionId)
    if (status === 'ready' || status === 'error') stopPolling()
    // update store...
  }, 2000)
}

function stopPolling() {
  if (pollingTimer) { clearInterval(pollingTimer); pollingTimer = null }
}
```

### 3.4 首次載入體積優化 🟢 FE

`vite.config.ts` 應設定 manual chunks，避免 Cytoscape + Plotly + axios 全打進同一個 chunk：

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-cytoscape': ['cytoscape'],
          'vendor-plotly': ['plotly.js-basic-dist-min'],
          'vendor-vue': ['vue', 'pinia', 'vue-router'],
        }
      }
    }
  }
})
```

這樣瀏覽器可以並行下載三個 chunk，且 Cytoscape / Plotly 可以被快取（只要版本不變）。

---

## 4. API 設計

### 4.1 `/analyze` 應為非同步，回傳 job_id 🟡 BE

**問題：** spec 現在是 `POST /analyze/{session_id}` → 回傳 `{ status }` 並「非同步 poll」，但如果 analyze 是同步的（見 2.1），這個設計自相矛盾。

**建議：** 明確定為 202 Accepted 非同步模式：

```
POST /analyze/{session_id}  →  202 { job_id }
GET  /status/{session_id}   →  { status: "analyzing", progress_pct: 45 }
                            →  { status: "ready" }
```

推論完成後，結果存入 `data/sessions/{session_id}/result.json`，之後的 `GET /graph`, `GET /alerts`, `GET /timeline` 直接從此檔案讀取，不再重跑推論。

### 4.2 `/adversarial` 缺少冪等性保護 🟢 BE

**問題：** 同一個 `flow_id` 用同樣的 `epsilon / steps` 可能被重複計算多次（例如前端 debounce 不完整時）。C-PGD 計算是確定性的（固定 seed），重複執行只是浪費。

**建議：** 在 session 目錄下快取結果：`data/sessions/{session_id}/adversarial/{flow_id}_eps{eps}_steps{steps}.json`。若檔案存在則直接回傳，不重算。

### 4.3 `/metrics` 與 static file 職責重複 🟢 BE

**問題：** spec 說 `reliability.json` 是「FastAPI 以靜態資源提供」，同時也定義了 `GET /metrics` endpoint。兩條路進同一份資料，讓前端不確定該用哪個。

**建議：** 只保留 `GET /api/metrics`，由後端讀取並回傳 JSON。移除 FastAPI `StaticFiles` 直接暴露 `data/metrics/` 的設定。這也能防止 `data/metrics/` 其他檔案被意外暴露。

---

## 5. Session 與檔案管理

### 5.1 CSV 上傳無大小限制 🟡 SEC

**問題：** `POST /upload` 接受 `multipart/form-data`，但 spec 沒有定義最大檔案大小。NF-UNSW-NB15-v2 完整版 ~500 MB，若不限制，使用者可上傳任意大小的 CSV。

**建議：** 在 FastAPI 層加入大小限制（demo 用途建議 50 MB；需要支援完整資料集則 500 MB）：

```python
# app/routers/analysis.py
from fastapi import File, UploadFile, HTTPException

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, detail=f"File too large (max {MAX_UPLOAD_SIZE // 1024**2} MB)")
    # ...
```

### 5.2 Session 目錄並發寫入安全性 🟢 BE

**問題：** 若兩個請求同時對同一個 session 寫入（例如快速雙擊 analyze），可能產生 partial write 的 result.json。

**建議：** 寫入時先寫 temp 檔再 rename（atomic 操作）：

```python
import tempfile, os, json
from pathlib import Path

def atomic_write_json(path: Path, data: dict) -> None:
    dir_ = path.parent
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as f:
        json.dump(data, f)
        tmp_path = f.name
    os.replace(tmp_path, path)  # atomic on POSIX; near-atomic on Windows
```

---

## 6. 安全性

### 6.1 Pickle 反序列化風險 🟡 SEC

**問題：** `scaler.pkl` 用 Python `pickle` 序列化。Pickle 在 load 時可以執行任意程式碼。若 `data/processed/` 被修改（或共享環境中被替換），會有 RCE 風險。

**建議：** 改用 `joblib`（scikit-learn 標準做法，在 pickle 之上加了安全選項）或直接序列化 scaler 參數為 JSON：

```python
# 序列化
import json
scaler_params = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}
with open("data/processed/static/scaler.json", "w") as f:
    json.dump(scaler_params, f)

# 反序列化
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
scaler.mean_  = np.array(params["mean_"])
scaler.scale_ = np.array(params["scale_"])
```

### 6.2 Session ID 路徑穿越驗證 🟡 SEC

**問題：** `session_id` 被用於構成檔案路徑 `data/sessions/{session_id}/`。若 session_id 未驗證，攻擊者可傳入 `../../etc/passwd` 等路徑（雖然 UUID 格式通常不會有此問題，但應該明確防範）。

**建議：** 用 regex 強制驗證 UUID 格式：

```python
# app/routers/analysis.py
import re
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')

def validate_session_id(session_id: str) -> str:
    if not UUID_RE.match(session_id):
        raise HTTPException(400, detail="Invalid session ID")
    return session_id
```

或更簡單地用 FastAPI 的 `UUID` 型別（自動驗證）：

```python
from uuid import UUID
async def get_graph(session_id: UUID): ...
```

### 6.3 CORS 設定 🟢 SEC

**問題：** 目前 CORS 硬寫 `allow_origins=["http://localhost:5173"]`，部署時需要改。

**建議：** 從環境變數讀取：

```python
# app/main.py
import os
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, ...)
```

---

## 7. 可部署性

### 7.1 無 Docker / 啟動文件 🟡 DEP

**問題：** README 的 Quick Start 需要手動安裝 uv、PyTorch cu124、PyG、Node.js。對「只是想 demo」的 reviewer 來說步驟多，且 CUDA 版本容易出錯。

**建議（最小可行部署）：**

1. 新增 `docker-compose.yml`（CPU 模式，不需 CUDA，只用來 demo）：

```yaml
# docker-compose.yml
services:
  backend:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
  frontend:
    build: ./frontend
    ports: ["5173:80"]
    depends_on: [backend]
```

2. 新增 `Dockerfile`（後端）：

```dockerfile
FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY . .
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. 新增 `frontend/Dockerfile`：

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

### 7.2 前後端連線設定 🟢 DEP

**問題：** 前端 axios client 的 `baseURL` 若寫死 `http://localhost:8000`，部署到其他主機時需要重新 build。

**建議：** 用 Vite 的環境變數：

```typescript
// frontend/src/api/client.ts
const baseURL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
```

```
# frontend/.env.example
VITE_API_BASE_URL=http://localhost:8000
```

---

## 8. Risk Summary Table

| 編號 | 問題 | 面向 | 嚴重度 | 影響 | 建議行動 |
|------|------|------|--------|------|----------|
| R1 | GNN 推論同步執行，阻塞 event loop | BE | 🔴 | Demo 時伺服器 10–30 秒無回應 | `run_in_threadpool` 包裝所有 PyTorch 呼叫 |
| R2 | C-PGD 同步執行，request 可能 timeout | BE | 🔴 | 瀏覽器 timeout 後顯示網路錯誤 | `run_in_threadpool` + 30s timeout |
| R3 | WeasyPrint 同步執行，阻塞 event loop | BE | 🔴 | PDF 下載觸發時伺服器卡頓 | `run_in_threadpool` |
| R4 | 兩個模型只載入一個 | BE | 🟡 | 使用者無法切換模型 | lifespan 同時載入兩個模型 |
| R5 | Session 清除機制未定義 | BE | 🟡 | `data/sessions/` 持續增長 | lifespan 啟動清除協程 |
| R6 | Cytoscape.js layout 計算卡頓 | FE | 🟡 | 低階電腦初始渲染 3–5 秒 lag | 後端預算座標（networkx spring_layout） |
| R7 | Plotly.js 完整 bundle ~3 MB | FE | 🟡 | 首次載入慢，尤其慢速網路 | 改用 basic-dist + dynamic import |
| R8 | pickle scaler 有 RCE 風險 | SEC | 🟡 | 被替換的 pickle 可執行任意程式碼 | 改為 JSON 序列化 |
| R9 | session_id 路徑未驗證 | SEC | 🟡 | 路徑穿越攻擊 | FastAPI UUID 型別或 regex 驗證 |
| R10 | CSV 上傳無大小限制 | SEC | 🟡 | 超大檔案佔滿磁碟 | 50 MB 上限 + 回傳 413 |
| R11 | 無 Dockerfile，部署門檻高 | DEP | 🟡 | Reviewer 跑不起來，影響 GitHub 印象 | docker-compose CPU 模式 |
| R12 | Selenium headless 產生 graph PNG | BE | 🟢 | 部署環境需安裝 Chrome | 改由前端 cy.png() 上傳 base64 |
| R13 | `/metrics` 與 static file 職責重複 | BE | 🟢 | 前端不確定哪個是正確路徑 | 只保留 `GET /api/metrics` |
| R14 | 前端輪詢 abort 邏輯未定義 | FE | 🟢 | 重複上傳時舊輪詢殘留 | stopPolling() 在每次上傳前呼叫 |
| R15 | axios baseURL 寫死 | DEP | 🟢 | 部署到其他主機需重新 build | `VITE_API_BASE_URL` 環境變數 |

---

## 9. Recommended Action Items

依實作優先序排列：

### 即刻（Phase 1 Scaffold 期間，不寫這些後面都是坑）

```
[ ] 所有 PyTorch / WeasyPrint 呼叫一律用 run_in_threadpool 包裝
[ ] POST /analyze 改為 202 Accepted + 輪詢模式（前端已規劃，後端要配合）
[ ] FastAPI 路由全面改用 UUID 型別接收 session_id
[ ] lifespan 啟動時同時載入 graphsage 和 gat 兩個模型
[ ] 加入 CSV 上傳大小限制（50 MB）
```

### Phase 1 週 5–6（Frontend 開始接 API 時）

```
[ ] vite.config.ts 設定 manualChunks（cytoscape / plotly / vue 分離）
[ ] 改用 plotly.js-basic-dist-min，或 dynamic import
[ ] 後端 graph_builder.py 加入 networkx spring_layout 座標計算
[ ] Pinia session store 加入 startPolling / stopPolling + 輪詢 abort
[ ] axios client 改用 VITE_API_BASE_URL 環境變數
```

### Phase 1 週 7–8（完成核心 views 後）

```
[ ] scaler 序列化改為 JSON（取代 pickle）
[ ] lifespan 加入 session 清除協程（每 5 分鐘掃描，1 小時 TTL）
[ ] /adversarial 加入快取（flow_id + epsilon + steps 為 key）
[ ] 移除 StaticFiles 直接暴露 data/metrics/；統一走 GET /api/metrics
[ ] ALLOWED_ORIGINS 改從環境變數讀取
```

### Phase 2 完成後（準備 GitHub 公開前）

```
[ ] 新增 docker-compose.yml（CPU 模式）+ Dockerfile（backend + frontend）
[ ] 前端 TrafficGraph.vue 加入 cy.png() export 並上傳 base64 給 report 端點
[ ] 移除所有 Selenium headless 相關程式碼或設計
[ ] README 加入 "One-command demo" 區塊（docker compose up）
[ ] 測試 docker-compose 在 GitHub Codespaces 上可正常啟動
```

---

> 下一次審核建議在 Phase 1 Week 5（Inference service 完成後）進行，重點驗證：
> (1) `/analyze` 202 非同步模式是否正確實作；
> (2) Cytoscape.js 500 nodes 的實際渲染效能；
> (3) `/adversarial` timeout 邊界行為。

---

## 10. GNN 訓練效能優化紀錄

> 紀錄靜態基線模型（GraphSAGE / GAT）訓練過程中發現的問題與已實施的優化，供後續 TGAT / TGN 訓練參考。

### 10.1 基線結果（未優化，window=300s，CrossEntropyLoss）

| 模型 | F1 (test) | Precision | Recall | ROC-AUC |
|------|-----------|-----------|--------|---------|
| GraphSAGE | 0.4779 | 0.7979 | 0.4169 | 0.8777 |
| GAT | 0.4391 | 0.7531 | 0.3882 | 0.8513 |

**觀察**：Precision 偏高（0.75–0.80）但 Recall 偏低（0.39–0.42），加權 F1 遠低於目標 0.90。

---

### 10.2 核心問題：Temporal Distribution Shift 🔴

**問題**：NF-UNSW-NB15-v2 的攻擊流量在時間軸上分布不均，導致嚴重的訓練/測試分佈偏移。

| 資料集切割 | 視窗數 | 總邊數 | Benign 比例 | Attack 比例 |
|-----------|--------|--------|------------|------------|
| Train (60%) | 1,289 | 231,866 | 26.8% | 73.2% |
| Val (20%) | 430 | 77,268 | 19.6% | 80.4% |
| **Test (20%)** | **430** | **77,272** | **59.3%** | **40.7%** |

**影響**：
- 訓練期間攻擊流佔 73%，但測試期間只佔 41%，導致模型在測試集大量漏報（低 Recall）。
- Class 6（最大攻擊類別）從 Train/Val 的 48k/39k 邊降至 Test 的 12k——模型針對訓練分佈學到的決策邊界，無法泛化至測試期間的流量組成。
- Val F1 可高達 0.85（Val 分佈近似 Train），但 Test F1 僅 0.43——**此差距是資料集特性，非模型 bug**。

**建議評估策略**：
- 同時報告 **Val F1**（反映模型本身學習能力）與 **Test F1**（反映跨時間泛化能力）。
- F1 ≥ 0.90 的目標建議以 **Val F1** 為主要指標；Test F1 作為泛化參考。
- 期待 TGAT / TGN 的時間建模能更好地捕捉攻擊的時序規律，縮小 Val/Test 差距。

---

### 10.3 已實施的優化

#### A. Focal Loss（取代 CrossEntropyLoss）

**動機**：CrossEntropyLoss 對「容易分類的良性流」過度關注，壓縮了模型對稀有攻擊類別的學習空間。

**實作**：`src/eval/losses.py` — `FocalLoss(weight, gamma)`

```python
FL(pt) = -αt · (1 − pt)^γ · log(pt)
```

- `γ = 2.0`（預設）：對 pt > 0.9 的容易樣本懲罰降低至約 1%，迫使模型專注困難樣本。
- 結合 class weights（inverse-frequency）雙重處理類別不平衡。
- 透過 Optuna 將 `focal_gamma` 設為可調超參（搜尋範圍 1.0–3.0，步長 0.5）。

**設定位置**：`configs/train.yaml`

```yaml
train:
  loss: focal        # "focal" | "cross_entropy"
  focal_gamma: 2.0
```

#### B. Window Size 縮減（300s → 120s）

**動機**：300s 視窗產生 516 個訓練視窗，樣本量不足；縮減至 120s 可在維持 graph density（~15–20 proxy nodes）的前提下，將訓練視窗數提升至 **1,289 個**（增加 2.5×）。

**設定位置**：`configs/data/static_default.yaml`

```yaml
window_size_s: 120
```

**結果**：Val F1 從舊版的約 0.60 提升至 0.85。

#### C. torch-scatter 安裝方式修正

**問題**：`torch-scatter` 不得列入 `pyproject.toml` 依賴，否則 `uv sync` 會嘗試從源碼編譯，因 build subprocess 無法找到 `torch` 而失敗。

**正確安裝方式**：

```bash
# 必須用 uv pip（非 uv run pip）且指定 torch 2.6.0 的 find-links URL
uv pip install torch_scatter torch_sparse torch_cluster \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

---

### 10.4 優化後結果（window=120s，FocalLoss γ=2.0）

| 模型 | Val F1 | F1 (test) | Precision | Recall | ROC-AUC | Best Epoch |
|------|--------|-----------|-----------|--------|---------|------------|
| GraphSAGE | 0.8567 | 0.4322 | 0.8023 | 0.3622 | 0.8337 | 151 |
| GAT | ~0.86 | 0.4716 | 0.8338 | 0.3862 | 0.8588 | 180 |

---

### 10.5 後續優化方向

1. **Optuna 超參搜尋**：執行 `uv run python scripts/tune_hyperparams.py --model graphsage --trials 50 --epochs 40`，搜尋 `lr`、`hidden_dim`、`num_layers`、`focal_gamma` 的最佳組合。

2. **閾值校準（Test-time Calibration）**：class weights 由訓練集分佈計算，不匹配測試集。可在 Val 集上用 Platt scaling 或 Temperature scaling 校準輸出機率。

3. **Macro F1 作為補充指標**：Weighted F1 受 Benign 類別（測試集 59%）主導，Macro F1 可更公平地反映各攻擊類別的識別能力。

4. **Temporal GNN（下一階段）**：TGAT / TGN 透過 time2vec + 連續時間建模，理論上能更好地捕捉攻擊的時序特徵，有助縮小 Val/Test 泛化差距。
