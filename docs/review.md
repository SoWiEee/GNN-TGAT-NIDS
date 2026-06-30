# 專案審核：GNN-TGAT-NIDS

> 針對安全性、API 設計、程式碼品質、測試覆蓋與部署進行系統性審核。
> 僅列出需要改進的項目。

**審核版本：** 3.2 | **參照規格：** `docs/spec.md` v1.2.0 | **日期：** 2026-07-01

---

## 1. 資源管理

### 1.1 高成本端點無 Rate Limiting `MEDIUM`

以下端點可被濫用進行 DoS：

| 端點 | 風險 |
|------|------|
| `POST /upload` | 頻繁上傳大型 CSV 消耗磁碟 |
| `POST /analyze/{id}` | 並發推論消耗 GPU 記憶體 |
| `POST /adversarial` | C-PGD 運算密集 |

**建議：** 加入 `slowapi` 或自訂 middleware。

---

## 2. 已修復項目

| 項目 | 原嚴重度 | 修復方式 |
|------|:--------:|---------|
| CORS 萬用字元 methods/headers | HIGH | 明確列出 `GET, POST, OPTIONS` 和 `Content-Type, Authorization` |
| WebSocket 無並發限制 | HIGH | `asyncio.Semaphore` 限制最大 `MAX_CONCURRENT_WS` 連線數 |
| 攻擊端點無存取控制 | HIGH | `ENABLE_ATTACK_ENDPOINTS` 環境變數開關（adversarial + memory_poisoning） |
| 推論任務無並發限制 | MEDIUM | `asyncio.Semaphore` 限制最大 `MAX_CONCURRENT_INFER` 推論數 |
| 分頁/排序參數無邊界檢查 | MEDIUM | `Query(ge=, le=, pattern=)` 驗證 |
| `window_seconds` 未驗證 | MEDIUM | WebSocket 連線時驗證 1.0–3600.0 範圍 |
| 例外訊息洩漏實作細節 | MEDIUM | `logger.exception()` + 通用錯誤訊息 |
| `torch.load(weights_only=False)` | LOW | 集中至 `app/services/torch_load.py`，單一審計點 |
| 硬編碼常數 | LOW | 後端 `os.getenv()` + 前端 `import.meta.env` 配置化 |
| 缺少 API 端點測試 | HIGH | 新增 23 個 API 測試 |
| 無測試覆蓋率設定 | LOW | `pyproject.toml` 加入 `--cov` |

---

## 3. 總結

| 嚴重度 | 剩餘數量 | 主要範圍 |
|--------|:--------:|---------|
| MEDIUM | 1 | Rate Limiting |

**建議：** 當專案需要對外公開時加入 Rate Limiting middleware。
