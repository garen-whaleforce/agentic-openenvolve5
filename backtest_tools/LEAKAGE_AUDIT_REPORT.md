# Lookahead Bias / Data Leakage Audit Report

**審計日期**: 2026-01-01
**審計對象**: Long-only v1.1-live-safe 策略
**審計結論**: **發現 3 個高嚴重度 Lookahead 問題**

---

## Executive Summary

經過程式碼審計，確認 **86% 勝率存在嚴重的數據洩漏問題**，主要來自以下三個函數：

| 問題 | 嚴重度 | 影響範圍 | 狀態 |
|------|--------|----------|------|
| `get_historical_earnings_facts()` | **🔴 高** | 所有 Helper Agent | 待修 |
| `get_historical_financials_facts()` | **🔴 高** | 財務分析 | 待修 |
| `get_quarterly_financials()` | **🔴 高** | 主要分析流程 | 待修 |
| Memory Prompt | ✅ 低 | 已停用 | 無需處理 |

---

## 問題 1: `get_historical_earnings_facts()` - 直接洩漏預測目標

### 位置
`pg_client.py` 第 955-1022 行

### 問題描述

```sql
SELECT
    et.year, et.quarter, et.transcript_date_str,
    et.market_timing,
    pa.pct_change_t as earnings_day_return,
    pa.pct_change_t_plus_20 as return_20d,
    pa.pct_change_t_plus_30 as return_30d,  -- ⚠️ 這是預測目標！
    pa.trend_category
FROM earnings_transcripts et
LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
WHERE UPPER(et.symbol) = %s
    AND NOT (et.year = %s AND et.quarter = %s)  -- ⚠️ 只排除當季，未來季度會被抓進來
ORDER BY et.year DESC, et.quarter DESC
LIMIT %s
```

### 洩漏機制

1. **SQL 條件問題**: `NOT (year = X AND quarter = Y)` 只排除當前季度
   - 回測 2017Q1 時，2018-2025 的資料都會被抓進來

2. **直接洩漏標籤**: `pct_change_t_plus_30` 是 30 天報酬 = 預測目標
   - LLM 看到「這家公司過去的 30 天報酬都是正的」→ 自然會預測正

3. **影響估計**: 足以將勝率從 ~60% 推高到 80%+

### 修復建議

```sql
-- 修改為嚴格的時間限制
WHERE UPPER(et.symbol) = %s
    AND (et.year < %s OR (et.year = %s AND et.quarter < %s))
ORDER BY et.year DESC, et.quarter DESC
LIMIT %s
```

同時移除或遮蔽 `return_20d`, `return_30d`, `trend_category` 這些事後才知道的欄位。

---

## 問題 2: `get_historical_financials_facts()` - 抓取未來財報

### 位置
`pg_client.py` 第 876-952 行

### 問題描述

```sql
SELECT
    inc.date, inc.period,
    inc.revenue, inc.net_income, inc.eps, inc.ebitda,
    inc.revenue_growth, inc.gross_profit, inc.operating_income
FROM income_statements inc
WHERE UPPER(inc.symbol) = %s
ORDER BY inc.date DESC  -- ⚠️ 取最新的，沒有時間限制
LIMIT %s
```

```python
# Python 層只排除當季，未來季度照樣進來
if year == current_year and q == current_q:
    continue
```

### 洩漏機制

1. **SQL 無時間限制**: `ORDER BY date DESC` 抓最新財報
2. **回測 2017 時會抓到 2024/2025 的財報**
3. **YoY/QoQ 計算使用未來數據**

### 修復建議

```sql
-- 加入 as-of-date 限制
WHERE UPPER(inc.symbol) = %s
    AND inc.date < %s  -- 傳入 reaction_date
ORDER BY inc.date DESC
LIMIT %s
```

---

## 問題 3: `get_quarterly_financials()` - FMP API 無時間限制

### 位置
`fmp_client.py` 第 882-920 行

### 問題描述

```python
def get_quarterly_financials(symbol: str, limit: int = 4) -> Dict:
    """
    Fetch recent quarterly financial statements.
    ⚠️ 沒有 as-of-date 參數，永遠抓最新的
    """
```

此函數在 `get_earnings_context()` 中被調用：

```python
def get_earnings_context(symbol, year, quarter):
    ...
    financials = get_quarterly_financials(symbol, limit=4)  # ⚠️ 抓最新 4 季
```

### 洩漏機制

1. **回測 2017Q1 時**：`get_quarterly_financials()` 會抓到 2024Q4-2025Q1 的財報
2. **這些財報被放進 LLM 的 context**
3. **LLM 看到的是「未來」的財務數據**

### 修復建議

```python
def get_quarterly_financials(symbol: str, before_date: str, limit: int = 4) -> Dict:
    """
    Fetch quarterly financial statements as of a specific date.

    Args:
        symbol: Stock ticker
        before_date: Only return statements filed before this date (YYYY-MM-DD)
        limit: Number of quarters
    """
```

---

## 問題 4: Memory Prompt - 已停用 ✅

### 位置
`EarningsCallAgenticRag/agents/mainAgent.py` 第 468-471 行

### 狀態

```python
# TODO: remove
final_prompt = core_prompt  # ← memory_txt 已被註解掉
```

**結論**: Memory prompt 目前未被使用，無 lookahead 風險。

---

## 驗證步驟

### Step A - 輸出日期審計欄位

在生成 signals 時，新增以下欄位到 CSV：

```python
{
    "max_financial_date_used": "...",
    "max_historical_quarter_used": "...",
    "max_any_date_in_prompt": "...",
    "reaction_date": "..."
}
```

加入 hard assert：
```python
assert max_any_date_in_prompt <= reaction_date, "LOOKAHEAD DETECTED!"
```

### Step B - Ablation 測試

跑兩個對照組（各 200-500 筆）：

1. **禁用 PG historical facts**：強制走 Neo4j 或純 transcript
2. **禁用 financial_statements_facts**：只用 transcript + market anchors

**預期結果**：如果修改後勝率大幅下降（如 60-70%），證明原本的 80%+ 來自 lookahead。

### Step C - 清除 Cache 重跑

```bash
# 清 Redis cache
redis-cli KEYS "call:*" | xargs redis-cli DEL

# 重跑回測
python run_long_only_test.py --skip-cache --years 2017-2025
```

---

## 修補 PR 清單

### PR #1: 修復 `get_historical_earnings_facts()`

**檔案**: `pg_client.py`

**變更**:
1. SQL WHERE 條件改為 `(year < X) OR (year = X AND quarter < Y)`
2. 移除 `return_20d`, `return_30d`, `trend_category` 欄位

### PR #2: 修復 `get_historical_financials_facts()`

**檔案**: `pg_client.py`

**變更**:
1. 新增 `before_date` 參數
2. SQL 加入 `AND inc.date < %s`

### PR #3: 修復 `get_quarterly_financials()`

**檔案**: `fmp_client.py`, `pg_client.py`

**變更**:
1. 新增 `before_date` 參數
2. 修改所有呼叫點傳入 `reaction_date`

### PR #4: 新增 Lookahead Assertion

**檔案**: `analysis_engine.py` 或 orchestrator

**變更**:
1. 新增 `validate_no_lookahead()` 函數
2. 在每次分析前執行 assertion

---

## 修補後預期結果

| 指標 | 修補前（有 Lookahead） | 修補後（預估） |
|------|------------------------|----------------|
| Win Rate | 86% | 60-70% |
| Sharpe | 2.0 | 0.8-1.2 |
| CAGR | 13.5% | 8-10% |

**如果修補後仍維持 80%+ 勝率，則策略確實強勁。**
**如果大幅下降，則原本的績效來自 lookahead。**

---

## 結論

**86% 勝率的可信度：低**

在修補上述三個高嚴重度問題之前，無法確認策略的真實績效。建議：

1. **立即暫停 Live Trading 計畫**
2. **執行修補 PR #1-#4**
3. **清除 cache 後重跑完整回測**
4. **根據修補後的績效重新評估**

---

*報告產生者: Claude Code Audit*
*審計版本: v1.0*
