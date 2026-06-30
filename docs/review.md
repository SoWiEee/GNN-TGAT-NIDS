# 專案審核：GNN-TGAT-NIDS

> 針對安全性、API 設計、程式碼品質、測試覆蓋與部署進行系統性審核。
> 僅列出需要改進的項目。

**審核版本：** 3.0 | **參照規格：** `docs/spec.md` v1.2.0 | **日期：** 2026-07-01

---

## 1. 安全性

### 1.1 CORS 設定過於寬鬆 `HIGH`

**檔案：** `app/main.py:63-65`

```python
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
```

`allow_credentials=True` 搭配萬用字元方法和標頭違反 CORS 安全最佳實踐。

**建議：** 明確列出允許的方法和標頭：

```python
allow_methods=["GET", "POST", "OPTIONS"],
allow_headers=["Content-Type", "Authorization"],
```

### 1.2 WebSocket 缺少存取控制 `HIGH`

**檔案：** `app/routers/streaming.py:182`

`/api/ws/stream` 接受任何連線，無認證、無並發限制。惡意用戶可大量連線消耗 GPU/CPU 資源。

**建議：** 加入並發連線數限制：

```python
MAX_CONCURRENT_WS = int(os.getenv("MAX_CONCURRENT_WS", "5"))
_ws_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WS)
```

### 1.3 Memory Poisoning 端點無存取控制 `HIGH`

**檔案：** `app/routers/memory_poisoning.py`

`POST /api/memory-poisoning` 允許外部修改 TGN 模型記憶體。研究用端點不應在生產環境暴露。

**建議：** 加入環境變數開關：

```python
ENABLE_ATTACK_ENDPOINTS = os.getenv("ENABLE_ATTACK_ENDPOINTS", "false").lower() == "true"
```

### 1.4 WebSocket `window_seconds` 未驗證 `MEDIUM`

**檔案：** `app/routers/streaming.py:182`

```python
window_seconds: float = DEFAULT_WINDOW_SECONDS,
```

無邊界檢查，可傳入負數、零或極大值。

**建議：** 連線建立後立即驗證：

```python
if not (1.0 <= window_seconds <= 3600.0):
    await websocket.close(code=1008, reason="window_seconds must be 1-3600")
    return
```

### 1.5 例外訊息洩漏實作細節 `MEDIUM`

**檔案：**
- `app/routers/adversarial.py:65` — `f"C-PGD failed: {exc}"`
- `app/routers/memory_poisoning.py:135` — `f"Memory poisoning experiment failed: {exc}"`
- `app/routers/analysis.py:70` — `str(exc)` 寫入 status 檔案

500 錯誤不應將完整例外訊息回傳給客戶端。

**建議：** 回傳通用訊息，將完整堆疊記錄到 server log：

```python
except Exception as exc:
    logger.exception("C-PGD failed")
    raise HTTPException(500, detail="Adversarial generation failed")
```

### 1.6 `torch.load(weights_only=False)` `LOW`

**檔案：**
- `app/services/inference.py:48`
- `app/routers/explain.py:71`
- `app/routers/memory_poisoning.py:24,37,38`

Pickle 反序列化在受信環境可接受，但若 checkpoint 來自外部源有 RCE 風險。

**建議（公開部署時）：** 改為 `state_dict` 載入模式，需同時儲存模型類別與 Hydra config。

---

## 2. API 輸入驗證

### 2.1 分頁參數無邊界檢查 `MEDIUM`

**檔案：** `app/routers/analysis.py:115-117`

```python
sort: str = "confidence",
page: int = 1,
limit: int = 50,
```

`page` 可為負數、`limit` 可為 0 或極大值、`sort` 無白名單驗證。

**建議：**

```python
from fastapi import Query

sort: str = Query(default="confidence", pattern="^(confidence|timestamp|attack_type)$"),
page: int = Query(default=1, ge=1),
limit: int = Query(default=50, ge=1, le=200),
```

### 2.2 `max_edges` 無上限 `LOW`

**檔案：** `app/routers/analysis.py:101`

```python
async def get_graph(session_id: UUID, max_edges: int = 2000):
```

**建議：** `max_edges: int = Query(default=2000, ge=10, le=10000)`

---

## 3. 資源管理

### 3.1 高成本端點無 Rate Limiting `MEDIUM`

以下端點可被濫用進行 DoS：

| 端點 | 風險 |
|------|------|
| `POST /upload` | 頻繁上傳大型 CSV 消耗磁碟 |
| `POST /analyze/{id}` | 並發推論消耗 GPU 記憶體 |
| `POST /adversarial` | C-PGD 運算密集 |
| `WS /ws/stream` | 持續連線消耗資源 |

**建議：** 加入 `slowapi` 或自訂 middleware：

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/upload")
@limiter.limit("10/minute")
async def upload_csv(...):
```

### 3.2 推論任務無並發限制 `MEDIUM`

多個 `/analyze` 請求可同時佔用所有 GPU 記憶體。

**建議：** 使用 `asyncio.Semaphore` 限制最大並發推論數：

```python
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "3"))
_infer_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFER)
```

---

## 4. 可配置性

### 4.1 硬編碼常數應改為環境變數 `LOW`

| 檔案 | 常數 | 目前值 |
|------|------|--------|
| `app/main.py:22` | `CLEANUP_INTERVAL_SECONDS` | 300 |
| `app/routers/analysis.py:17` | `MAX_UPLOAD_BYTES` | 50 MB |
| `app/routers/adversarial.py:17` | `ADV_TIMEOUT_SECONDS` | 30.0 |
| `app/routers/streaming.py:35` | `MAX_BUFFER_SIZE` | 10,000 |

**建議：** 統一改為 `os.getenv()` 讀取，搭配合理預設值。

---

## 5. 測試覆蓋

### 5.1 缺少 API 端點測試 `HIGH`

現有 26 個測試檔案全部針對 ML 核心（模型、資料管線、攻擊模組）。`app/routers/` 和 `app/services/` 幾乎無測試覆蓋。

**需要新增的測試：**

| 測試範圍 | 優先級 |
|---------|--------|
| CSV 上傳（檔案大小限制、格式驗證、session 建立） | HIGH |
| `/analyze` 非同步推論（狀態輪詢、模型名稱驗證） | HIGH |
| `/graph`、`/alerts`、`/timeline` 資料檢索 | MEDIUM |
| `/adversarial` C-PGD（timeout、快取） | MEDIUM |
| WebSocket 串流（緩衝管理、視窗完成） | MEDIUM |
| `/explain` 可解釋性端點 | LOW |

### 5.2 無測試覆蓋率設定 `LOW`

**檔案：** `pyproject.toml:81-83`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**建議：** 加入覆蓋率設定：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov=app --cov-report=term-missing"
```

---

## 6. 總結

| 嚴重度 | 數量 | 主要範圍 |
|--------|:----:|---------|
| HIGH | 4 | CORS、WebSocket/攻擊端點存取控制、API 測試缺失 |
| MEDIUM | 6 | 輸入驗證、錯誤訊息洩漏、Rate Limiting、並發控制 |
| LOW | 4 | 硬編碼常數、`torch.load` 安全、覆蓋率設定 |

**建議優先順序：**

1. **立即修復：** CORS 設定收緊、WebSocket 並發限制、攻擊端點環境變數開關
2. **公開部署前：** API 輸入驗證、錯誤訊息脫敏、Rate Limiting
3. **長期改善：** API 端點測試、`state_dict` 載入模式、硬編碼常數配置化
