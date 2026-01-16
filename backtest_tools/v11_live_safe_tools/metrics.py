
"""
metrics.py

Portfolio performance metrics for backtest and paper trading monitoring.

We assume you will produce a daily NAV series indexed by trading session (date).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) else np.nan


@dataclass
class PerfMetrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_trades: int
    exposure_avg: float


def compute_drawdowns(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    if nav.empty:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return dd


def compute_perf_metrics(
    nav: pd.Series,
    daily_rf_rate: float = 0.0,
    trades: Optional[pd.DataFrame] = None,
    exposure: Optional[pd.Series] = None,
) -> PerfMetrics:
    """
    nav: daily net asset value series (>=0), indexed by session date
    daily_rf_rate: daily risk-free rate (e.g., 0.05/252)
    trades: optional trade ledger with column 'ret_pct' (percent, e.g., 12.3) or 'ret' (fraction)
    exposure: optional daily gross exposure series (0..1+)
    """
    nav = nav.dropna()
    if len(nav) < 2:
        raise ValueError("NAV series too short to compute metrics")

    # Daily returns
    rets = nav.pct_change().dropna()
    if rets.empty:
        raise ValueError("No daily returns computed from NAV")

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0

    # CAGR using trading days
    n_days = len(rets)
    years = n_days / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    ann_vol = rets.std(ddof=1) * np.sqrt(252.0)

    excess = rets - daily_rf_rate
    sharpe = _safe_div(excess.mean(), rets.std(ddof=1)) * np.sqrt(252.0)

    downside = rets.copy()
    downside[downside > 0] = 0
    downside_std = downside.std(ddof=1)
    sortino = _safe_div(excess.mean(), downside_std) * np.sqrt(252.0)

    dd = compute_drawdowns(nav)
    max_drawdown = float(dd.min()) if not dd.empty else np.nan
    calmar = _safe_div(cagr, abs(max_drawdown)) if max_drawdown < 0 else np.nan

    # Trades
    n_trades = 0
    win_rate = np.nan
    profit_factor = np.nan
    avg_win = np.nan
    avg_loss = np.nan

    if trades is not None and not trades.empty:
        n_trades = len(trades)
        if "ret" in trades.columns:
            t_rets = trades["ret"].astype(float)
        elif "ret_pct" in trades.columns:
            t_rets = trades["ret_pct"].astype(float) / 100.0
        else:
            raise ValueError("Trades DF must have 'ret' or 'ret_pct'")

        wins = t_rets[t_rets > 0]
        losses = t_rets[t_rets <= 0]
        win_rate = float(len(wins) / len(t_rets)) if len(t_rets) else np.nan
        gross_win = wins.sum()
        gross_loss = losses.sum()  # negative or 0
        profit_factor = _safe_div(gross_win, abs(gross_loss)) if gross_loss < 0 else np.inf
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0

    exposure_avg = float(exposure.mean()) if exposure is not None and not exposure.empty else np.nan

    return PerfMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_drawdown),
        calmar=float(calmar),
        win_rate=float(win_rate) if not np.isnan(win_rate) else np.nan,
        profit_factor=float(profit_factor) if profit_factor is not None else np.nan,
        avg_win=float(avg_win) if avg_win is not None else np.nan,
        avg_loss=float(avg_loss) if avg_loss is not None else np.nan,
        n_trades=int(n_trades),
        exposure_avg=float(exposure_avg) if exposure_avg is not None else np.nan,
    )


def to_dict(m: PerfMetrics) -> Dict[str, float]:
    return {
        "total_return": m.total_return,
        "cagr": m.cagr,
        "ann_vol": m.ann_vol,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
        "n_trades": m.n_trades,
        "exposure_avg": m.exposure_avg,
    }
