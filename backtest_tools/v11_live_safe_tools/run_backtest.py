
"""
run_backtest.py

CLI wrapper to run the local backtest engine and output:
- daily NAV
- trade ledger
- metrics JSON + Markdown

Requires:
- signals CSV with: symbol,reaction_date,trade_long
- prices folder with per-symbol OHLC CSV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtester import BacktestConfig, run_backtest
from metrics import compute_perf_metrics, to_dict
from price_providers import CSVPriceProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices-folder", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calendar", default="XNYS")
    ap.add_argument("--cap-per-quarter", type=int, default=12)
    ap.add_argument("--max-positions", type=int, default=12)
    ap.add_argument("--holding-sessions", type=int, default=30)
    ap.add_argument("--entry-lag-sessions", type=int, default=1, help="Entry lag in trading sessions (1=T+1)")
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--annual-rf", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sig = pd.read_csv(args.signals)
    pp = CSVPriceProvider(folder=Path(args.prices_folder))

    cfg = BacktestConfig(
        calendar_name=args.calendar,
        holding_sessions=args.holding_sessions,
        entry_lag_sessions=args.entry_lag_sessions,
        cap_entries_per_quarter=args.cap_per_quarter,
        max_concurrent_positions=args.max_positions,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        annual_rf_rate=args.annual_rf,
    )

    nav, trades, exposure = run_backtest(sig, pp, cfg)
    nav.rename("nav").to_csv(outdir/"nav.csv", index=True)
    exposure.rename("gross_exposure").to_csv(outdir/"exposure.csv", index=True)
    trades.to_csv(outdir/"trades.csv", index=False)

    m = compute_perf_metrics(nav, daily_rf_rate=cfg.daily_rf_rate(), trades=trades, exposure=exposure)
    md = []
    md.append("# Local Backtest Summary\n\n")
    md.append(f"- Total Return: {m.total_return*100:.2f}%\n")
    md.append(f"- CAGR (ARR): {m.cagr*100:.2f}%\n")
    md.append(f"- Annual Vol: {m.ann_vol*100:.2f}%\n")
    md.append(f"- Sharpe: {m.sharpe:.2f}\n")
    md.append(f"- Sortino: {m.sortino:.2f}\n")
    md.append(f"- Max Drawdown: {m.max_drawdown*100:.2f}%\n")
    md.append(f"- Calmar: {m.calmar:.2f}\n")
    md.append(f"- Trades: {m.n_trades}\n")
    md.append(f"- Win Rate: {m.win_rate*100:.2f}%\n")
    md.append(f"- Profit Factor: {m.profit_factor:.2f}\n")
    md.append(f"- Avg Exposure: {m.exposure_avg:.2f}\n")

    (outdir/"summary.md").write_text("".join(md), encoding="utf-8")
    (outdir/"metrics.json").write_text(json.dumps(to_dict(m), indent=2))
    print(f"Wrote backtest outputs to {outdir}")


if __name__ == "__main__":
    main()
