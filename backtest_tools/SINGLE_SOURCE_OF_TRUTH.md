# Long-only v1.1 Single Source of Truth

**確立時間**: 2026-01-01
**版本**: v1.1-live-safe (修正日期版)

---

## 正式基準資料

| 檔案 | 路徑 | 說明 |
|------|------|------|
| **Signals** | `signals_fixed_dates.csv` | 修正 reaction_date 後的信號資料 |
| **Trades** | `out_backtest_fixed/trades.csv` | 正式交易紀錄 |
| **NAV** | `out_backtest_fixed/nav.csv` | 每日淨值序列 |
| **Metrics** | `out_backtest_fixed/metrics.json` | 回測指標 |

---

## 正式指標 (2017-01-25 至 2024-12-16)

### Trade-Level
| 指標 | 數值 | 95% CI |
|------|------|--------|
| Total Trades | 180 | - |
| Win Rate | 86.11% | [81.1%, 91.1%] |
| Avg Return | +7.33% | [+5.5%, +9.1%] |
| Max Gain | +43.92% | - |
| Max Loss | -42.33% | [-42.3%, -25.4%] |
| Profit Factor | 4.06 | [2.65, 7.08] |

### NAV-Level
| 指標 | 數值 | 95% CI |
|------|------|--------|
| Total Return | 184.27% | - |
| CAGR | 14.16% | [7.15%, 21.30%] |
| Sharpe | 1.91 | [0.91, 2.92] |
| Max Drawdown | -16.95% | [-25.16%, -4.46%] |
| Trading Days | 1,987 | - |

---

## 策略規格 (v1.1-live-safe)

| 參數 | 設定 |
|------|------|
| Entry | T+1 Open |
| Exit | T+31 Close (30 trading days) |
| CAP | 12/季 (First-N) |
| Max Positions | 12 (等權 1/12) |
| Slippage | 10 bps |
| Commission | 2 bps |

---

## Two-Tier 選股規則

### D7 CORE
- Direction Score ≥ 7
- Earnings Day Return ≥ 1.5%
- EPS Positive = True

### D6 STRICT
- Direction Score = 6
- Min Positives ≥ 1
- Earnings Day Return ≥ 1.0%

---

## 已棄用資料 (僅供歷史參考)

| 檔案 | 問題 |
|------|------|
| `out_backtest_complete/` | reaction_date 錯誤 (fiscal/calendar 錯位) |
| `signals_backtest_ready.csv` | 舊版日期 |

---

## 口徑定義

### Win Rate
- 定義: ret_pct > 0 為 win
- 計算: winners / total_trades

### Max Drawdown
- 定義: NAV 從高點的最大回落
- 計算: min((nav - cummax) / cummax)

### Sharpe Ratio
- 定義: 年化風險調整報酬
- 計算: (mean_daily_return / std_daily_return) * sqrt(252)

### Profit Factor
- 定義: 總獲利 / 總虧損 (絕對值)
- 計算: sum(gains) / abs(sum(losses))

---

*本文件為 Long-only v1.1 策略的唯一正式資料來源*
