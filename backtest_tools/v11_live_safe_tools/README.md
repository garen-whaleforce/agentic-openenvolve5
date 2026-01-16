
# v1.1-live-safe Local Backtest + Paper Trading Monitor (Starter Kit)

This folder provides a **local-only** implementation to:
1) Monitor paper trading (v1.1-live-safe) using a signals CSV + OHLC prices.
2) Run a local event-driven backtest consistent with your spec:
   - Decision: reaction-day close
   - Entry: T+1 open
   - Exit: T+31 close (30 trading sessions after entry)

It also includes a grid-search driver to tune portfolio/backtest parameters and
(optionally) simple threshold-based signal selection.

## Why this exists
- Whaleforce `weighted_rebalance` is often **close-to-close** and "hold-until-next-rebalance".
- Your spec is **open entry + fixed 30 trading days**.
- This starter kit lets you compute **CAGR/ARR, Sharpe, drawdowns, total return** locally
  in a live-safe way, then later decide what to offload to Whaleforce.

---

## Inputs

### 1) Signals CSV (required)
Minimal columns:
- `symbol` (ticker)
- `reaction_date` (YYYY-MM-DD, must be a trading session date)
- `trade_long` (True/False)

Optional (used only for monitoring/diagnostics):
- `eps_surprise`, `earnings_day_return`, `sector`, ...

### 2) OHLC prices folder (required)
Folder with per-symbol CSV files:
- `prices/AAPL.csv`
- `prices/MSFT.csv`
- ...

Each price CSV must include:
- `date` (or datetime/timestamp)
- `open`
- `close`
Optionally:
- `high`, `low`, `volume`

Dates are interpreted as trading sessions and normalized to midnight.

---

## Quick Start

### Local backtest
```bash
python run_backtest.py \
  --signals signals_2017_2025.csv \
  --prices-folder ./prices \
  --outdir ./out_backtest \
  --cap-per-quarter 12 \
  --max-positions 12 \
  --holding-sessions 30 \
  --entry-lag-sessions 1
```

Outputs in `./out_backtest/`:
- `nav.csv` (daily NAV at close)
- `exposure.csv` (daily gross exposure)
- `trades.csv` (trade ledger with entry/exit and net_ret)
- `metrics.json`, `summary.md`

### Paper trading monitor (as-of date)
```bash
python paper_trading_monitor.py \
  --signals signals_live.csv \
  --prices-folder ./prices \
  --as-of 2026-01-01 \
  --cap-per-quarter 12 \
  --max-positions 12 \
  --outdir ./paper_outputs
```

### Parameter tuning (portfolio params only; keep existing trade_long)
```bash
python tune_backtest_params.py \
  --signals signals_2017_2025.csv \
  --prices-folder ./prices \
  --out grid_results.csv \
  --use-existing-trade-long 1 \
  --cap-list 8,10,12 \
  --maxpos-list 8,12,16 \
  --commission-bps-list 0,1 \
  --slippage-bps-list 0,1
```

---

## Notes / Caveats

1) **reaction_date is required**
Your current research CSVs often contain `year,quarter` but not the exact date.
For NAV/Sharpe/CAGR you need exact trading dates.

2) **Calendar**
Default is `exchange_calendars` `XNYS` sessions. If you trade non-US listings,
parameterize calendar or provide a custom calendar.

3) **Position sizing**
Default is **equal notional** with `max_concurrent_positions` (cash-limited).
If you want Whaleforce-like equal-weight rebalancing, extend `backtester.py`
with an explicit rebalance rule.

4) **Costs**
Commission + slippage are applied per side as bps on notional.

---

## Files
- `price_providers.py` – CSV price loader and provider interface
- `calendar_utils.py` – trading session helpers
- `backtester.py` – event-driven backtest engine
- `metrics.py` – CAGR/Sharpe/DD/etc
- `run_backtest.py` – CLI runner
- `paper_trading_monitor.py` – monitor + guardrails outputs
- `tune_backtest_params.py` – grid search driver
