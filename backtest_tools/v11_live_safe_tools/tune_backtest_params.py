
"""
tune_backtest_params.py

Grid-search runner for portfolio/backtest parameters (CAP/max_positions/costs, etc.)
and optionally simple threshold-based trade selection.

This is designed to be run *locally* where you have:
- a full signals/features CSV with reaction_date and either trade_long or raw columns
- OHLC CSV folder for prices

Example (tune portfolio params only, keeping existing trade_long):
python tune_backtest_params.py \
  --signals signals_2017_2025.csv \
  --prices-folder ./prices \
  --out grid_results.csv \
  --use-existing-trade-long 1 \
  --cap-list 8,10,12 \
  --maxpos-list 8,12,16 \
  --commission-bps-list 0,1 \
  --slippage-bps-list 0,1

Example (also tune selection thresholds):
python tune_backtest_params.py \
  --signals features_2017_2025.csv \
  --prices-folder ./prices \
  --out grid_results.csv \
  --use-existing-trade-long 0 \
  --dir-min-list 6,7 \
  --day-min-list 0.5,1.0,1.5 \
  --eps-min-list 0.0,0.02 \
  --cap-list 8,10,12 \
  --maxpos-list 12
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from backtester import BacktestConfig, run_backtest
from metrics import compute_perf_metrics
from price_providers import CSVPriceProvider


def parse_list(s: str, cast=float) -> List:
    if s is None or s.strip() == "":
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(cast(part))
    return out


def make_trade_long_from_thresholds(df: pd.DataFrame, dir_min: int, day_min: float, eps_min: float) -> pd.Series:
    # Minimal, deterministic approximation. You can replace with your project-specific logic.
    # Required columns check:
    required = ["direction_score", "earnings_day_return", "eps_surprise", "computed_vetoes", "risk_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for threshold-based selection: {missing}")

    ok = (
        (df["direction_score"].astype(float) >= dir_min) &
        (df["earnings_day_return"].astype(float) >= day_min) &
        (df["eps_surprise"].astype(float) >= eps_min) &
        (df["computed_vetoes"].astype(int) == 0) &
        (~df["risk_code"].astype(str).str.upper().isin(["HIGH"]))
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="CSV with symbol,reaction_date plus trade_long or feature cols")
    ap.add_argument("--prices-folder", required=True)
    ap.add_argument("--out", required=True, help="Output CSV of grid results")

    ap.add_argument("--use-existing-trade-long", type=int, default=1, help="1=use trade_long column; 0=generate from thresholds")

    ap.add_argument("--cap-list", default="12", help="Comma list, e.g. 8,10,12")
    ap.add_argument("--maxpos-list", default="12", help="Comma list, e.g. 8,12,16")
    ap.add_argument("--commission-bps-list", default="0", help="Comma list")
    ap.add_argument("--slippage-bps-list", default="0", help="Comma list")

    ap.add_argument("--holding-sessions", type=int, default=30)
    ap.add_argument("--entry-lag-sessions", type=int, default=1)
    ap.add_argument("--calendar", default="XNYS")
    ap.add_argument("--annual-rf", type=float, default=0.0)

    # Optional selection threshold tuning
    ap.add_argument("--dir-min-list", default="7", help="Comma list of int thresholds")
    ap.add_argument("--day-min-list", default="1.5", help="Comma list of floats (percent)")
    ap.add_argument("--eps-min-list", default="0.0", help="Comma list of floats (fraction, e.g., 0.02=2%)")

    args = ap.parse_args()

    df = pd.read_csv(args.signals)
    if "reaction_date" not in df.columns:
        raise SystemExit("signals CSV must contain reaction_date (YYYY-MM-DD). Add it to your export first.")

    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.normalize()
    if df["reaction_date"].isna().any():
        raise SystemExit("Found invalid reaction_date values; please fix before tuning.")

    df["symbol"] = df["symbol"].astype(str).str.upper()

    pp = CSVPriceProvider(folder=Path(args.prices_folder))

    caps = parse_list(args.cap_list, cast=int)
    maxps = parse_list(args.maxpos_list, cast=int)
    comms = parse_list(args.commission_bps_list, cast=float)
    slips = parse_list(args.slippage_bps_list, cast=float)

    dirs = parse_list(args.dir_min_list, cast=int)
    days = parse_list(args.day_min_list, cast=float)
    epss = parse_list(args.eps_min_list, cast=float)

    rows = []
    total = 0

    for cap, maxp, comm, slip in itertools.product(caps, maxps, comms, slips):
        if args.use_existing_trade_long:
            # single selection
            df_sel = df.copy()
            if "trade_long" not in df_sel.columns:
                raise SystemExit("use-existing-trade-long=1 but CSV missing trade_long column.")
            # Normalize trade_long
            if df_sel["trade_long"].dtype == object:
                df_sel["trade_long"] = df_sel["trade_long"].astype(str).str.lower().isin(["true","1","yes","y"])
            else:
                df_sel["trade_long"] = df_sel["trade_long"].astype(bool)

            # Keep only required columns for engine
            sig = df_sel[["symbol","reaction_date","trade_long"]].copy()

            cfg = BacktestConfig(
                calendar_name=args.calendar,
                holding_sessions=args.holding_sessions,
                    entry_lag_sessions=args.entry_lag_sessions,
                cap_entries_per_quarter=cap,
                max_concurrent_positions=maxp,
                commission_bps=comm,
                slippage_bps=slip,
                annual_rf_rate=args.annual_rf,
            )

            try:
                nav, trades, exposure = run_backtest(sig, pp, cfg)
                m = compute_perf_metrics(nav, daily_rf_rate=cfg.daily_rf_rate(), trades=trades, exposure=exposure)
            except Exception as e:
                rows.append({
                    "cap": cap, "maxpos": maxp, "commission_bps": comm, "slippage_bps": slip,
                    "error": str(e)[:200],
                })
                continue

            rows.append({
                "cap": cap, "maxpos": maxp, "commission_bps": comm, "slippage_bps": slip,
                "n_trades": m.n_trades,
                "total_return": m.total_return,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "profit_factor": m.profit_factor,
                "win_rate": m.win_rate,
                "exposure_avg": m.exposure_avg,
                "error": "",
            })
        else:
            # also tune selection thresholds
            for dir_min, day_min, eps_min in itertools.product(dirs, days, epss):
                df_sel = df.copy()
                df_sel["trade_long"] = make_trade_long_from_thresholds(df_sel, dir_min=dir_min, day_min=day_min, eps_min=eps_min)
                sig = df_sel[["symbol","reaction_date","trade_long"]].copy()

                cfg = BacktestConfig(
                    calendar_name=args.calendar,
                    holding_sessions=args.holding_sessions,
                    entry_lag_sessions=args.entry_lag_sessions,
                    cap_entries_per_quarter=cap,
                    max_concurrent_positions=maxp,
                    commission_bps=comm,
                    slippage_bps=slip,
                    annual_rf_rate=args.annual_rf,
                )

                try:
                    nav, trades, exposure = run_backtest(sig, pp, cfg)
                    m = compute_perf_metrics(nav, daily_rf_rate=cfg.daily_rf_rate(), trades=trades, exposure=exposure)
                except Exception as e:
                    rows.append({
                        "dir_min": dir_min, "day_min": day_min, "eps_min": eps_min,
                        "cap": cap, "maxpos": maxp, "commission_bps": comm, "slippage_bps": slip,
                        "error": str(e)[:200],
                    })
                    continue

                rows.append({
                    "dir_min": dir_min, "day_min": day_min, "eps_min": eps_min,
                    "cap": cap, "maxpos": maxp, "commission_bps": comm, "slippage_bps": slip,
                    "n_trades": m.n_trades,
                    "total_return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                    "profit_factor": m.profit_factor,
                    "win_rate": m.win_rate,
                    "exposure_avg": m.exposure_avg,
                    "error": "",
                })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote grid results: {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
