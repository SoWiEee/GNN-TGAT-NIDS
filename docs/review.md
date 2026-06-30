# 專案審核：GNN-TGAT-NIDS

> 針對安全性、API 設計、程式碼品質、測試覆蓋與部署進行系統性審核。
> 僅列出需要改進的項目。

**審核版本：** 3.1 | **參照規格：** `docs/spec.md` v1.2.0 | **日期：** 2026-07-01

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

**檔案：** `app/routers/streaming.py`

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

### 1.4 `torch.load(weights_only=False)` `LOW`

**檔案：**
- `app/services/inference.py:48`
- `app/routers/explain.py:71`
- `app/routers/memory_poisoning.py:24,37,38`

Pickle 反序列化在受信環境可接受，但若 checkpoint 來自外部源有 RCE 風險。

**建議（公開部署時）：** 改為 `state_dict` 載入模式，需同時儲存模型類別與 Hydra config。

---

## 2. 資源管理

### 2.1 高成本端點無 Rate Limiting `MEDIUM`

以下端點可被濫用進行 DoS：

| 端點 | 風險 |
|------|------|
| `POST /upload` | 頻繁上傳大型 CSV 消耗磁碟 |
| `POST /analyze/{id}` | 並發推論消耗 GPU 記憶體 |
| `POST /adversarial` | C-PGD 運算密集 |
| `WS /ws/stream` | 持續連線消耗資源 |

**建議：** 加入 `slowapi` 或自訂 middleware。

### 2.2 推論任務無並發限制 `MEDIUM`

多個 `/analyze` 請求可同時佔用所有 GPU 記憶體。

**建議：** 使用 `asyncio.Semaphore` 限制最大並發推論數。

---

## 3. 已修復項目

| 項目 | 原嚴重度 | 修復方式 |
|------|:--------:|---------|
| 分頁/排序參數無邊界檢查 | MEDIUM | `Query(ge=, le=, pattern=)` 驗證 `page`、`limit`、`sort`、`max_edges` |
| `window_seconds` 未驗證 | MEDIUM | WebSocket 連線時驗證 1.0–3600.0 範圍 |
| 例外訊息洩漏實作細節 | MEDIUM | 改為 `logger.exception()` + 通用錯誤訊息 |
| 硬編碼常數 | LOW | 後端 `os.getenv()` + 前端 `import.meta.env` 配置化 |
| 缺少 API 端點測試 | HIGH | 新增 23 個 API 測試（upload、status、graph、alerts、adversarial、WebSocket） |
| 無測試覆蓋率設定 | LOW | `pyproject.toml` 加入 `--cov=src --cov=app --cov-report=term-missing` |

---

## 4. 總結

| 嚴重度 | 數量 | 主要範圍 |
|--------|:----:|---------|
| HIGH | 3 | CORS、WebSocket/攻擊端點存取控制 |
| MEDIUM | 2 | Rate Limiting、並發控制 |
| LOW | 1 | `torch.load` 安全 |

**建議優先順序：**

1. **立即修復：** CORS 設定收緊、WebSocket 並發限制、攻擊端點環境變數開關
2. **公開部署前：** Rate Limiting、推論並發限制
3. **長期改善：** `state_dict` 載入模式
