
"""
paper_trading_monitor.py

Local paper-trading monitor for v1.1-live-safe (LLM trade_long + CAP First-N).

This script does NOT place orders. It produces:
- accepted trades under CAP First-N
- portfolio NAV time series (paper fills)
- closed-trade ledger
- open positions snapshot as-of a date
- guardrail checks (rolling Wilson LB, eps_surprise missing rate, etc.)

Typical usage:

1) Prepare signals CSV with columns:
   symbol,reaction_date,trade_long
   Optional: eps_surprise, earnings_day_return, sector, direction_score, ...

2) Prepare OHLC CSV folder:
   prices/<SYMBOL>.csv with date/open/high/low/close

3) Run:
   python paper_trading_monitor.py \
       --signals signals.csv \
       --prices-folder ./prices \
       --as-of 2026-01-01 \
       --cap-per-quarter 12 \
       --max-positions 12 \
       --outdir ./paper_outputs

Outputs:
- nav.csv
- trades_closed.csv
- positions_open.csv
- guardrails.json
- summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from backtester import BacktestConfig, run_backtest, prepare_signals
from calendar_utils import TradingCalendar
from metrics import compute_perf_metrics, to_dict
from price_providers import CSVPriceProvider


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    phat = wins / n
    denom = 1 + (z**2)/n
    centre = phat + (z**2)/(2*n)
    adj = z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)
    return float((centre - adj) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="Signals CSV with symbol,reaction_date,trade_long (+ optional columns)")
    ap.add_argument("--prices-folder", required=True, help="Folder with per-symbol OHLC CSV files")
    ap.add_argument("--as-of", default=None, help="As-of date YYYY-MM-DD. Default: last available date in price files (best effort)")
    ap.add_argument("--calendar", default="XNYS", help="Exchange calendar, default XNYS")
    ap.add_argument("--cap-per-quarter", type=int, default=12, help="CAP (First-N) entries per quarter")
    ap.add_argument("--max-positions", type=int, default=12, help="Max concurrent positions for sizing (equal notional)")
    ap.add_argument("--holding-sessions", type=int, default=30, help="Holding sessions after entry (exit at close)")
    ap.add_argument("--entry-lag-sessions", type=int, default=1, help="Entry lag in sessions (1=T+1)")
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--annual-rf", type=float, default=0.0)
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--rolling-n", type=int, default=50, help="Rolling trade window for Wilson guardrail")
    ap.add_argument("--wilson-lb-halt", type=float, default=0.70, help="Halt threshold for rolling Wilson LB")
    ap.add_argument("--eps-missing-halt", type=float, default=0.15, help="Halt threshold for eps_surprise missing rate (recent signals)")
    ap.add_argument("--eps-missing-window", type=int, default=200, help="Window size for eps missing rate check (recent signals)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.signals)
    # If as-of not set, we estimate from signals max date (safe lower bound).
    signals_norm = prepare_signals(signals)

    if args.as_of is None:
        as_of = signals_norm["reaction_date"].max()
    else:
        as_of = pd.Timestamp(args.as_of).normalize()

    # Filter signals up to as_of (reaction date must be <= as_of)
    signals_upto = signals_norm.loc[signals_norm["reaction_date"] <= as_of].copy()
    if signals_upto.empty:
        raise SystemExit("No signals on or before as-of date.")

    cfg = BacktestConfig(
        calendar_name=args.calendar,
        holding_sessions=args.holding_sessions,
        entry_lag_sessions=args.entry_lag_sessions,
        cap_entries_per_quarter=args.cap_per_quarter,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        max_concurrent_positions=args.max_positions,
        annual_rf_rate=args.annual_rf,
    )

    pp = CSVPriceProvider(folder=Path(args.prices_folder))

    # Run backtest across full horizon, then truncate to as-of.
    nav, trades_df, exposure = run_backtest(signals_upto, pp, cfg)

    cal = TradingCalendar(args.calendar)
    # Ensure nav includes as_of session if it's a session
    # If as_of isn't a session, find previous session
    sessions = nav.index
    if as_of not in sessions:
        # Use last session <= as_of
        as_of_sessions = sessions[sessions <= as_of]
        if len(as_of_sessions) == 0:
            raise SystemExit("as-of is before the first available session in NAV.")
        as_of_session = as_of_sessions.max()
    else:
        as_of_session = as_of

    nav_upto = nav.loc[nav.index <= as_of_session].copy()
    exposure_upto = exposure.loc[exposure.index <= as_of_session].copy()

    # Closed vs open trades
    if trades_df.empty:
        closed = trades_df.copy()
    else:
        closed = trades_df.loc[trades_df["exit_date"] <= as_of_session].copy()

    # Open positions: accepted entries whose exit_date > as_of and entry_date <= as_of
    # We rebuild from trades + accepted list by re-running acceptance logic quickly:
    accepted = signals_upto.loc[signals_upto["trade_long"]].copy()
    accepted["yq"] = accepted["reaction_date"].apply(lambda d: f"{d.year}Q{(d.month-1)//3+1}")
    accepted = accepted.sort_values(["reaction_date","symbol"]).reset_index(drop=True)

    # apply CAP First-N again for reporting
    cap_count: Dict[str,int] = {}
    acc_rows = []
    for _, r in accepted.iterrows():
        yq = r["yq"]
        cap_count.setdefault(yq, 0)
        if cap_count[yq] >= args.cap_per_quarter:
            continue
        cap_count[yq] += 1
        acc_rows.append(r)
    accepted_cap = pd.DataFrame(acc_rows) if acc_rows else pd.DataFrame(columns=accepted.columns)

    accepted_cap["entry_date"] = accepted_cap["reaction_date"].apply(lambda d: cal.add_sessions(d, args.entry_lag_sessions))
    accepted_cap["exit_date"] = accepted_cap["entry_date"].apply(lambda d: cal.add_sessions(d, args.holding_sessions))

    open_pos = accepted_cap.loc[
        (accepted_cap["entry_date"] <= as_of_session) & (accepted_cap["exit_date"] > as_of_session)
    ][["symbol","reaction_date","entry_date","exit_date"]].copy()

    # Metrics
    summary = {}
    if len(nav_upto) >= 2:
        m = compute_perf_metrics(nav_upto, daily_rf_rate=cfg.daily_rf_rate(), trades=closed, exposure=exposure_upto)
        summary.update(to_dict(m))

    # Guardrails
    guard = {"as_of": str(as_of_session.date())}

    # Rolling 50 Wilson LB
    if closed is not None and not closed.empty:
        recent = closed.tail(args.rolling_n)
        wins = int((recent["net_ret"] > 0).sum())
        n = int(len(recent))
        lb = wilson_lower_bound(wins, n)
        guard["rolling_n"] = n
        guard["rolling_wins"] = wins
        guard["rolling_wilson_lb"] = lb
        guard["halt_wilson_lb_threshold"] = args.wilson_lb_halt
        guard["halt_wilson_lb"] = bool(n >= min(10, args.rolling_n) and lb < args.wilson_lb_halt)
    else:
        guard["rolling_n"] = 0
        guard["rolling_wins"] = 0
        guard["rolling_wilson_lb"] = float("nan")
        guard["halt_wilson_lb"] = False

    # EPS missing rate check
    eps_missing_rate = None
    if "eps_surprise" in signals_upto.columns:
        recent_sigs = signals_upto.tail(args.eps_missing_window)
        eps_missing_rate = float(recent_sigs["eps_surprise"].isna().mean())
        guard["eps_missing_window"] = int(len(recent_sigs))
        guard["eps_missing_rate"] = eps_missing_rate
        guard["halt_eps_missing_threshold"] = args.eps_missing_halt
        guard["halt_eps_missing"] = bool(eps_missing_rate > args.eps_missing_halt)
    else:
        guard["eps_missing_rate"] = None
        guard["halt_eps_missing"] = False

    guard["halt_new_trades"] = bool(guard["halt_wilson_lb"] or guard["halt_eps_missing"])

    # Write outputs
    nav_upto.rename("nav").to_csv(outdir/"nav.csv", index=True)
    exposure_upto.rename("gross_exposure").to_csv(outdir/"exposure.csv", index=True)
    closed.to_csv(outdir/"trades_closed.csv", index=False)
    open_pos.to_csv(outdir/"positions_open.csv", index=False)
    (outdir/"guardrails.json").write_text(json.dumps(guard, indent=2))

    # Markdown summary
    md = []
    md.append(f"# Paper Trading Monitor (v1.1-live-safe)\n")
    md.append(f"- As-of: **{as_of_session.date()}**\n")
    md.append(f"- Signals processed: {len(signals_upto)}\n")
    md.append(f"- Accepted (trade_long & CAP): {len(accepted_cap)}\n")
    md.append(f"- Open positions: {len(open_pos)}\n")
    md.append(f"- Closed trades: {len(closed)}\n")

    if summary:
        md.append("\n## Performance (to-date)\n")
        md.append(f"- Total Return: {summary['total_return']*100:.2f}%\n")
        md.append(f"- CAGR: {summary['cagr']*100:.2f}%\n")
        md.append(f"- Sharpe: {summary['sharpe']:.2f}\n")
        md.append(f"- Max Drawdown: {summary['max_drawdown']*100:.2f}%\n")
        md.append(f"- Profit Factor: {summary['profit_factor']:.2f}\n")
        md.append(f"- Win Rate (closed trades): {summary['win_rate']*100:.2f}%\n")
        md.append(f"- Avg Exposure: {summary['exposure_avg']:.2f}\n")

    md.append("\n## Guardrails\n")
    md.append(f"- Rolling {guard['rolling_n']} Wilson LB: {guard['rolling_wilson_lb'] if guard['rolling_n']>0 else 'N/A'}\n")
    md.append(f"- Halt (Wilson LB < {args.wilson_lb_halt}): {guard['halt_wilson_lb']}\n")
    if eps_missing_rate is not None:
        md.append(f"- EPS missing rate (recent): {eps_missing_rate*100:.2f}%\n")
        md.append(f"- Halt (EPS missing > {args.eps_missing_halt*100:.1f}%): {guard['halt_eps_missing']}\n")
    md.append(f"- HALT_NEW_TRADES: **{guard['halt_new_trades']}**\n")

    (outdir/"summary.md").write_text("".join(md), encoding="utf-8")

    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
