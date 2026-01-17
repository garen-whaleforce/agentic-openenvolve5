"""
Paper Trading Module

Provides infrastructure for forward testing the G250 strategy:
- paper_trading_db: SQLite database for signals, trades, and audit trail
- daily_run: Daily script to generate signals from earnings calls
- fill_prices: Fill entry/exit prices for paper trades
- weekly_report: Generate performance reports

Quick Start:
    1. Initialize database:
       python -m paper_trading.paper_trading_db

    2. Run daily signal generation:
       python -m paper_trading.daily_run

    3. Fill prices for signals:
       python -m paper_trading.fill_prices

    4. Generate weekly report:
       python -m paper_trading.weekly_report
"""

from .paper_trading_db import (
    init_db,
    insert_signal,
    update_signal_status,
    fill_entry_price,
    fill_exit_price,
    get_open_positions,
    get_signals_for_date,
    get_closed_trades,
    get_performance_summary,
    log_daily_run,
)

__all__ = [
    "init_db",
    "insert_signal",
    "update_signal_status",
    "fill_entry_price",
    "fill_exit_price",
    "get_open_positions",
    "get_signals_for_date",
    "get_closed_trades",
    "get_performance_summary",
    "log_daily_run",
]
