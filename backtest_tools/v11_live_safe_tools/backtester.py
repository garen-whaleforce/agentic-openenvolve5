
"""
backtester.py

Local event-driven backtester aligned to the v1.1-live-safe spec:

- Decision time: earnings reaction day close
- Entry: T+1 open (next trading session open)
- Exit: T+31 close (30 trading sessions after entry date; exit at close)

This engine simulates a simple cash+positions portfolio to produce:
- daily NAV series (close-to-close)
- trade ledger
- performance metrics (ARR/CAGR, Sharpe, max drawdown, etc.)

Notes:
- For exact matching with your internal DB labels, ensure reaction_date corresponds
  to the trading session of the earnings reaction day.
- This implementation assumes US trading calendar (XNYS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from calendar_utils import TradingCalendar
from price_providers import PriceProvider, PriceProviderError


@dataclass
class BacktestConfig:
    calendar_name: str = "XNYS"
    entry_lag_sessions: int = 1  # 1 => T+1 (next session). 0 => same-session entry.
    holding_sessions: int = 30  # exit at entry_idx + 30 close
    cap_entries_per_quarter: int = 12  # CAP方案A：每季最多12筆（先到先得）
    allow_multiple_positions_same_symbol: bool = False

    # Costs (per side)
    commission_bps: float = 0.0  # e.g., 1.0 means 1bp = 0.01%
    slippage_bps: float = 0.0

    # Position sizing
    max_concurrent_positions: int = 12
    min_allocation_cash: float = 0.0  # if remaining cash < this, skip new entries

    # Risk-free (annual)
    annual_rf_rate: float = 0.0

    # Price columns
    entry_price_col: str = "open"
    exit_price_col: str = "close"

    def daily_rf_rate(self) -> float:
        return float(self.annual_rf_rate) / 252.0


@dataclass
class Trade:
    symbol: str
    reaction_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    gross_ret: float  # fraction
    net_ret: float    # fraction after costs


def _bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0


def _year_quarter(d: pd.Timestamp) -> Tuple[int, int]:
    y = int(d.year)
    q = int((d.month - 1) // 3 + 1)
    return y, q


def _required_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Signals DF missing required columns: {missing}. Found: {df.columns.tolist()}")


def prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize signals:
    - requires: symbol, reaction_date, trade_long
    - reaction_date parsed to Timestamp normalized
    - trade_long cast to bool
    Sort by reaction_date then symbol.
    """
    _required_cols(signals, ["symbol", "reaction_date", "trade_long"])
    df = signals.copy()

    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.normalize()
    if df["reaction_date"].isna().any():
        bad = df[df["reaction_date"].isna()].head(10)
        raise ValueError(f"Found NaN reaction_date after parsing. Sample rows:\n{bad}")

    # trade_long can be bool or strings
    if df["trade_long"].dtype == object:
        df["trade_long"] = df["trade_long"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["trade_long"] = df["trade_long"].astype(bool)

    df["symbol"] = df["symbol"].astype(str).str.upper()

    df = df.sort_values(["reaction_date", "symbol"]).reset_index(drop=True)
    return df


def run_backtest(
    signals: pd.DataFrame,
    price_provider: PriceProvider,
    config: BacktestConfig,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns:
      nav_close: pd.Series daily NAV at close, indexed by session date
      trades_df: pd.DataFrame trade ledger
      exposure: pd.Series daily gross exposure (sum of position market values / NAV)
    """
    df = prepare_signals(signals)

    cal = TradingCalendar(config.calendar_name)

    # Determine all candidate trades (after CAP, but before price lookup we need dates).
    accepted_rows = []
    cap_counter: Dict[Tuple[int, int], int] = {}
    # FIX: Changed from global set to per-quarter set to allow same symbol trading across quarters
    # Previously: held_symbols: set = set()  # This blocked ALL repeat trades for a symbol forever!
    # Now: Each quarter has its own set, so AAPL can trade in Q1 and again in Q3
    held_symbols_by_quarter: Dict[Tuple[int, int], set] = {}

    for _, r in df.iterrows():
        if not bool(r["trade_long"]):
            continue

        rd = pd.Timestamp(r["reaction_date"]).normalize()
        yq = _year_quarter(rd)
        cap_counter.setdefault(yq, 0)
        held_symbols_by_quarter.setdefault(yq, set())

        if cap_counter[yq] >= config.cap_entries_per_quarter:
            continue

        sym = str(r["symbol"]).upper()
        # FIX: Only prevent duplicate entries within the SAME quarter, not across all time
        if (not config.allow_multiple_positions_same_symbol) and (sym in held_symbols_by_quarter[yq]):
            # If multiple signals for same symbol appear in same quarter (rare), keep the first one
            continue

        cap_counter[yq] += 1
        held_symbols_by_quarter[yq].add(sym)
        accepted_rows.append({"symbol": sym, "reaction_date": rd})

    if not accepted_rows:
        raise ValueError("No accepted trades after applying trade_long and CAP. Nothing to backtest.")

    accepted = pd.DataFrame(accepted_rows)
    # Compute entry and exit dates via trading calendar
    accepted["entry_date"] = accepted["reaction_date"].apply(lambda d: cal.add_sessions(d, config.entry_lag_sessions))
    accepted["exit_date"] = accepted["entry_date"].apply(lambda d: cal.add_sessions(d, config.holding_sessions))

    # Price range
    start = accepted["entry_date"].min()
    end = accepted["exit_date"].max()

    symbols = sorted(accepted["symbol"].unique().tolist())
    prices = price_provider.get_ohlc(symbols, start=start, end=end)

    # Build session index (close series)
    sessions = cal.sessions_in_range(start, end)
    nav = pd.Series(index=sessions, dtype=float)
    exposure = pd.Series(index=sessions, dtype=float)

    cash = 1.0  # start NAV=1
    positions: Dict[str, Dict] = {}  # symbol -> dict(shares, entry_price, exit_date, entry_date)
    trade_records: List[Dict] = []

    # Pre-index accepted entries by entry_date and exits by exit_date
    entries_by_date: Dict[pd.Timestamp, List[Dict]] = {}
    exits_by_date: Dict[pd.Timestamp, List[str]] = {}

    for _, r in accepted.iterrows():
        entries_by_date.setdefault(r["entry_date"], []).append(r.to_dict())
        exits_by_date.setdefault(r["exit_date"], []).append(r["symbol"])

    commission = _bps_to_frac(config.commission_bps)
    slip = _bps_to_frac(config.slippage_bps)
    per_side_cost = commission + slip

    def get_px(sym: str, d: pd.Timestamp, col: str) -> float:
        dfp = prices[sym]
        if d not in dfp.index:
            raise PriceProviderError(f"Missing price for {sym} on {d.date()}")
        px = float(dfp.loc[d, col])
        if not np.isfinite(px) or px <= 0:
            raise PriceProviderError(f"Bad price for {sym} on {d.date()}: {col}={px}")
        return px

    # Simulate day by day
    for i, session in enumerate(sessions):
        session = pd.Timestamp(session).normalize()

        # 1) Execute exits at CLOSE of session (we will value NAV at close; so sell before nav calc)
        if session in exits_by_date:
            for sym in list(exits_by_date[session]):
                if sym not in positions:
                    continue  # already closed or never opened
                px_exit = get_px(sym, session, config.exit_price_col)
                pos = positions.pop(sym)
                shares = float(pos["shares"])
                entry_px = float(pos["entry_price"])
                proceeds = shares * px_exit
                # costs on exit
                proceeds_net = proceeds * (1.0 - per_side_cost)
                cash += proceeds_net

                gross_ret = px_exit / entry_px - 1.0
                # Entry cost already applied at buy; approximate net ret on position value
                # We can compute net ret based on actual cash flows:
                invested = float(pos["invested_cash"])
                net_ret = (proceeds_net / invested) - 1.0 if invested > 0 else np.nan

                trade_records.append({
                    "symbol": sym,
                    "reaction_date": pos["reaction_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": session,
                    "entry_price": entry_px,
                    "exit_price": px_exit,
                    "shares": shares,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "invested_cash": invested,
                })

        # 2) Execute entries at OPEN of session (after we executed exits at close; order doesn't overlap same day)
        if session in entries_by_date:
            # Allocation per trade: equity / max_concurrent_positions
            # Equity here is previous close NAV; we can approximate using last nav (or cash+mark-to-close from previous day).
            # We'll compute a "start-of-day equity" using previous day's close values (i-1), if i>0, else cash.
            if i == 0:
                equity_prev = cash
            else:
                # value positions at previous close
                prev = sessions[i-1]
                prev = pd.Timestamp(prev).normalize()
                pos_val_prev = 0.0
                for sym, pos in positions.items():
                    px_prev = get_px(sym, prev, "close")
                    pos_val_prev += float(pos["shares"]) * px_prev
                equity_prev = cash + pos_val_prev

            alloc_target = equity_prev / float(config.max_concurrent_positions)
            for r in entries_by_date[session]:
                sym = r["symbol"]
                if (not config.allow_multiple_positions_same_symbol) and (sym in positions):
                    continue
                if cash < max(config.min_allocation_cash, 1e-12):
                    continue
                invest = min(cash, alloc_target)
                if invest <= 0:
                    continue
                px_entry = get_px(sym, session, config.entry_price_col)
                # costs on entry
                invest_net = invest * (1.0 - per_side_cost)
                shares = invest_net / px_entry
                cash -= invest
                positions[sym] = {
                    "shares": shares,
                    "entry_price": px_entry,
                    "entry_date": session,
                    "exit_date": r["exit_date"],
                    "reaction_date": r["reaction_date"],
                    "invested_cash": invest,
                }

        # 3) Value NAV at CLOSE
        pos_val = 0.0
        gross_exposure = 0.0
        for sym, pos in positions.items():
            px_close = get_px(sym, session, "close")
            mv = float(pos["shares"]) * px_close
            pos_val += mv

        nav_close = cash + pos_val
        nav.loc[session] = nav_close

        gross_exposure = pos_val / nav_close if nav_close > 0 else np.nan
        exposure.loc[session] = gross_exposure

    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
        # convenience
        trades_df["ret_pct"] = trades_df["net_ret"] * 100.0

    return nav, trades_df, exposure
