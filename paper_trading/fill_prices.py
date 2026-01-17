#!/usr/bin/env python3
"""
Paper Trading Price Fill Script

Fills entry and exit prices for paper trading signals.

Rules:
- Entry: T+1 open price (next trading day after signal_date)
- Exit: T+30 close price (30 trading days after entry)
- Slippage: 10 bps default

Usage:
    # Fill all pending prices
    python fill_prices.py

    # Fill for specific date range
    python fill_prices.py --from-date 2026-01-01 --to-date 2026-01-15

    # Dry run
    python fill_prices.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# Add project paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "EarningsCallAgenticRag"))

from dotenv import load_dotenv
load_dotenv(project_dir / ".env")

import paper_trading_db as db

# Configuration
ENTRY_LAG_DAYS = 1  # T+1 entry
HOLDING_DAYS = 30   # 30 trading days
SLIPPAGE_BPS = 10   # 10 basis points slippage


def get_trading_calendar() -> List[date]:
    """
    Get list of trading days.
    For simplicity, use weekdays. In production, use exchange_calendars.
    """
    # Generate trading days for the next year
    start = date(2024, 1, 1)
    end = date(2027, 12, 31)

    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days.append(current)
        current += timedelta(days=1)

    return days


def get_next_trading_day(d: date, offset: int = 1) -> date:
    """Get the next trading day after d with offset."""
    calendar = get_trading_calendar()
    try:
        idx = calendar.index(d)
        return calendar[idx + offset]
    except (ValueError, IndexError):
        # If date not in calendar, find next available
        for day in calendar:
            if day > d:
                if offset == 1:
                    return day
                offset -= 1
        return d + timedelta(days=offset)


def get_price(symbol: str, target_date: date, price_type: str = "close") -> Optional[float]:
    """
    Get price for symbol on date.

    Args:
        symbol: Stock symbol
        target_date: Date to get price for
        price_type: 'open', 'close', 'high', 'low'

    Returns:
        Price or None if not available
    """
    # Try from paper_prices table first
    with db.get_connection() as conn:
        row = conn.execute("""
            SELECT open_price, high_price, low_price, close_price
            FROM paper_prices
            WHERE symbol = ? AND date = ?
        """, (symbol, target_date.isoformat())).fetchone()

        if row:
            price_map = {
                "open": row["open_price"],
                "close": row["close_price"],
                "high": row["high_price"],
                "low": row["low_price"],
            }
            return price_map.get(price_type)

    # Fallback to pg_client
    try:
        from pg_client import get_cursor
        with get_cursor() as cur:
            if cur is None:
                return None

            cur.execute("""
                SELECT open, high, low, close
                FROM daily_prices
                WHERE UPPER(symbol) = %s AND date = %s
            """, (symbol.upper(), target_date.isoformat()))

            row = cur.fetchone()
            if row:
                price_map = {
                    "open": row["open"],
                    "close": row["close"],
                    "high": row["high"],
                    "low": row["low"],
                }
                return float(price_map.get(price_type)) if price_map.get(price_type) else None
    except Exception as e:
        print(f"  Price lookup error for {symbol} {target_date}: {e}")

    return None


def fill_entries(dry_run: bool = False) -> int:
    """Fill entry prices for SIGNAL status with trade_long=True."""
    filled = 0

    with db.get_connection() as conn:
        # Get signals that need entry fill
        rows = conn.execute("""
            SELECT id, symbol, signal_date
            FROM paper_signals
            WHERE status = 'SIGNAL' AND trade_long = 1
            ORDER BY signal_date
        """).fetchall()

        print(f"Found {len(rows)} signals needing entry fill")

        for row in rows:
            signal_id = row["id"]
            symbol = row["symbol"]
            signal_date = datetime.strptime(row["signal_date"], "%Y-%m-%d").date()

            # Calculate entry date (T+1)
            entry_date = get_next_trading_day(signal_date, ENTRY_LAG_DAYS)

            # Get entry price (open)
            entry_price = get_price(symbol, entry_date, "open")

            if entry_price is None:
                print(f"  {symbol}: No price for {entry_date}")
                continue

            print(f"  {symbol}: Entry {entry_date} @ ${entry_price:.2f}")

            if not dry_run:
                db.fill_entry_price(signal_id, entry_date, entry_price, SLIPPAGE_BPS)
                filled += 1

    return filled


def fill_exits(dry_run: bool = False) -> int:
    """Fill exit prices for OPEN positions past holding period."""
    filled = 0
    today = date.today()

    with db.get_connection() as conn:
        # Get open positions
        rows = conn.execute("""
            SELECT id, symbol, entry_date
            FROM paper_signals
            WHERE status = 'OPEN'
            ORDER BY entry_date
        """).fetchall()

        print(f"Found {len(rows)} open positions")

        for row in rows:
            signal_id = row["id"]
            symbol = row["symbol"]
            entry_date = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()

            # Calculate exit date (T+30 from entry)
            exit_date = get_next_trading_day(entry_date, HOLDING_DAYS)

            # Only fill if exit date has passed
            if exit_date > today:
                days_until = (exit_date - today).days
                print(f"  {symbol}: Exit in {days_until} days ({exit_date})")
                continue

            # Get exit price (close)
            exit_price = get_price(symbol, exit_date, "close")

            if exit_price is None:
                print(f"  {symbol}: No price for {exit_date}")
                continue

            print(f"  {symbol}: Exit {exit_date} @ ${exit_price:.2f}")

            if not dry_run:
                db.fill_exit_price(signal_id, exit_date, exit_price, SLIPPAGE_BPS)
                filled += 1

    return filled


def mark_to_market() -> List[Dict]:
    """Calculate mark-to-market for open positions."""
    today = date.today()
    mtm = []

    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT id, symbol, entry_date, entry_price, trade_long_tier, direction_score
            FROM paper_signals
            WHERE status = 'OPEN'
            ORDER BY entry_date
        """).fetchall()

        for row in rows:
            symbol = row["symbol"]
            entry_price = row["entry_price"]
            entry_date = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()

            # Get current price
            current_price = get_price(symbol, today, "close")
            if current_price is None:
                # Try yesterday
                yesterday = today - timedelta(days=1)
                current_price = get_price(symbol, yesterday, "close")

            if current_price and entry_price:
                unrealized_pct = (current_price / entry_price - 1) * 100
                days_held = (today - entry_date).days

                mtm.append({
                    "symbol": symbol,
                    "tier": row["trade_long_tier"],
                    "direction_score": row["direction_score"],
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pct": unrealized_pct,
                    "days_held": days_held,
                })

    return mtm


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Price Fill")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--entries-only", action="store_true", help="Only fill entries")
    parser.add_argument("--exits-only", action="store_true", help="Only fill exits")
    parser.add_argument("--mtm", action="store_true", help="Show mark-to-market only")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PAPER TRADING PRICE FILL")
    print(f"{'='*60}")
    print(f"Date: {date.today()}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Initialize database
    db.init_db()

    if args.mtm:
        # Mark-to-market only
        mtm = mark_to_market()
        print(f"\nMARK-TO-MARKET ({len(mtm)} positions):\n")
        print(f"{'Symbol':<8} {'Tier':<10} {'Entry':<12} {'Current':<10} {'P&L':>8} {'Days':>6}")
        print("-" * 60)
        total_pnl = 0
        for pos in mtm:
            print(f"{pos['symbol']:<8} {pos['tier'] or 'N/A':<10} "
                  f"${pos['entry_price']:<10.2f} ${pos['current_price']:<8.2f} "
                  f"{pos['unrealized_pct']:>7.2f}% {pos['days_held']:>6}")
            total_pnl += pos['unrealized_pct']

        if mtm:
            print("-" * 60)
            print(f"{'TOTAL':<30} {'':<18} {total_pnl:>7.2f}%")
        return

    # Fill entries
    if not args.exits_only:
        print("\n--- FILLING ENTRIES ---\n")
        entries_filled = fill_entries(dry_run=args.dry_run)
        print(f"\nEntries filled: {entries_filled}")

    # Fill exits
    if not args.entries_only:
        print("\n--- FILLING EXITS ---\n")
        exits_filled = fill_exits(dry_run=args.dry_run)
        print(f"\nExits filled: {exits_filled}")

    # Show current summary
    print("\n--- CURRENT STATUS ---\n")
    summary = db.get_performance_summary()
    print(f"Closed trades: {summary['closed_trades']}")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"Avg return: {summary['avg_return_pct']:.2f}%")
    print(f"Open positions: {summary['open_positions']}")
    print(f"Pending signals: {summary['pending_signals']}")


if __name__ == "__main__":
    main()
