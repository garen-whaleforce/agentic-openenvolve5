#!/usr/bin/env python3
"""
Paper Trading Daily Run Script

Runs daily to:
1. Fetch today's earnings transcripts
2. Run agentic RAG analysis on each
3. Store signals in paper_trading.db
4. Notify via Slack/LINE if trade_long signals generated

Usage:
    # Run for today
    python daily_run.py

    # Run for specific date (backfill)
    python daily_run.py --date 2026-01-15

    # Dry run (no DB writes)
    python daily_run.py --dry-run

    # With notification
    python daily_run.py --notify
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Add project paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "EarningsCallAgenticRag"))

from dotenv import load_dotenv
load_dotenv(project_dir / ".env")

import paper_trading_db as db

# Strategy version - MUST bump when changing prompts/logic
STRATEGY_VERSION = "g250_v1.0"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=project_dir
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_env_snapshot() -> Dict:
    """Get environment snapshot for auditing."""
    return {
        "MAIN_MODEL": os.environ.get("MAIN_MODEL"),
        "HELPER_MODEL": os.environ.get("HELPER_MODEL"),
        "LOOKAHEAD_ASSERTIONS": os.environ.get("LOOKAHEAD_ASSERTIONS"),
        "STRATEGY_VERSION": STRATEGY_VERSION,
        "code_commit": get_git_commit(),
    }


def get_earnings_for_date(target_date: date) -> List[Dict]:
    """
    Get earnings calls that happened on target_date.

    Returns list of dicts with: symbol, year, quarter, transcript_date
    """
    from pg_client import get_cursor

    with get_cursor() as cur:
        if cur is None:
            print("WARNING: Database connection failed")
            return []

        # Find transcripts with transcript_date matching target_date
        cur.execute("""
            SELECT
                et.symbol,
                et.year,
                et.quarter,
                et.transcript_date_str,
                c.name as company_name,
                c.sector
            FROM earnings_transcripts et
            LEFT JOIN companies c ON et.symbol = c.symbol
            WHERE DATE(et.transcript_date_str) = %s
                AND et.transcript_date_str IS NOT NULL
            ORDER BY et.symbol
        """, (target_date.isoformat(),))

        results = []
        for row in cur.fetchall():
            results.append({
                "symbol": row["symbol"],
                "year": row["year"],
                "quarter": row["quarter"],
                "transcript_date": row["transcript_date_str"],
                "company_name": row["company_name"],
                "sector": row["sector"],
            })

        return results


async def analyze_single(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """Run analysis on a single earnings call."""
    try:
        from analysis_engine import analyze_earnings_async

        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,  # Always fresh for paper trading
        )

        return result.get("agentic_result", {})
    except Exception as e:
        print(f"  ERROR analyzing {symbol}: {e}")
        return None


async def run_daily_analysis(
    target_date: date,
    dry_run: bool = False,
    notify: bool = False,
) -> Dict:
    """
    Run daily paper trading analysis.

    Returns summary dict.
    """
    print(f"\n{'='*60}")
    print(f"PAPER TRADING DAILY RUN")
    print(f"{'='*60}")
    print(f"Date: {target_date}")
    print(f"Strategy: {STRATEGY_VERSION}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    start_time = time.time()
    env_snapshot = get_env_snapshot()
    code_commit = env_snapshot["code_commit"]

    # Initialize database if needed
    if not dry_run:
        db.init_db()

    # Get earnings for this date
    earnings = get_earnings_for_date(target_date)
    print(f"Found {len(earnings)} earnings calls for {target_date}")

    if not earnings:
        print("No earnings to process.")
        return {
            "date": target_date,
            "symbols_processed": 0,
            "signals_generated": 0,
            "trade_long_count": 0,
            "errors": 0,
        }

    # Process each earnings call
    signals_generated = 0
    trade_long_count = 0
    errors = 0
    trade_long_signals = []

    for i, earning in enumerate(earnings, 1):
        symbol = earning["symbol"]
        year = earning["year"]
        quarter = earning["quarter"]

        print(f"[{i}/{len(earnings)}] Analyzing {symbol} {year}Q{quarter}...")

        result = await analyze_single(symbol, year, quarter)

        if result is None:
            errors += 1
            continue

        signals_generated += 1

        # Extract key fields
        trade_long = result.get("trade_long", False)
        trade_long_tier = result.get("trade_long_tier")
        direction_score = result.get("le_DirectionScore") or result.get("direction_score")

        if trade_long:
            trade_long_count += 1
            trade_long_signals.append({
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "tier": trade_long_tier,
                "direction_score": direction_score,
                "company_name": earning.get("company_name"),
                "sector": earning.get("sector"),
            })
            print(f"  -> TRADE_LONG! tier={trade_long_tier}, score={direction_score}")

        # Store in database
        if not dry_run:
            try:
                db.insert_signal(
                    signal_date=target_date,
                    symbol=symbol,
                    year=year,
                    quarter=quarter,
                    strategy_version=STRATEGY_VERSION,
                    result=result,
                    model_main=os.environ.get("MAIN_MODEL"),
                    model_helper=os.environ.get("HELPER_MODEL"),
                    code_commit=code_commit,
                )
            except Exception as e:
                if "UNIQUE constraint" in str(e):
                    print(f"  Signal already exists for {symbol} {year}Q{quarter}")
                else:
                    print(f"  DB error: {e}")

    # Calculate duration
    duration = time.time() - start_time

    # Log the run
    if not dry_run:
        db.log_daily_run(
            run_date=target_date,
            strategy_version=STRATEGY_VERSION,
            symbols_processed=len(earnings),
            signals_generated=signals_generated,
            trade_long_count=trade_long_count,
            errors_count=errors,
            duration_seconds=duration,
            env_snapshot=env_snapshot,
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"DAILY RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Date: {target_date}")
    print(f"Symbols processed: {len(earnings)}")
    print(f"Signals generated: {signals_generated}")
    print(f"trade_long signals: {trade_long_count}")
    print(f"Errors: {errors}")
    print(f"Duration: {duration:.1f}s")

    if trade_long_signals:
        print(f"\nTRADE_LONG SIGNALS:")
        for sig in trade_long_signals:
            print(f"  {sig['symbol']} ({sig['tier']}): {sig['company_name']} - {sig['sector']}")

    print(f"{'='*60}")

    # Send notification if requested
    if notify and trade_long_signals:
        await send_notification(target_date, trade_long_signals)

    return {
        "date": target_date,
        "symbols_processed": len(earnings),
        "signals_generated": signals_generated,
        "trade_long_count": trade_long_count,
        "errors": errors,
        "trade_long_signals": trade_long_signals,
    }


async def send_notification(target_date: date, signals: List[Dict]):
    """Send notification for trade_long signals."""
    # TODO: Implement Slack/LINE notification
    # For now, just print
    print(f"\n[NOTIFICATION] {len(signals)} trade_long signals for {target_date}:")
    for sig in signals:
        print(f"  - {sig['symbol']} ({sig['tier']})")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Daily Run")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default today")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--notify", action="store_true", help="Send notifications")
    parser.add_argument("--backfill-days", type=int, help="Backfill N days (starting from --date)")
    args = parser.parse_args()

    # Determine target date(s)
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    if args.backfill_days:
        # Run for multiple days
        dates = [target_date - timedelta(days=i) for i in range(args.backfill_days)]
        dates.reverse()  # Oldest first

        print(f"Backfilling {len(dates)} days: {dates[0]} to {dates[-1]}")

        for d in dates:
            asyncio.run(run_daily_analysis(d, dry_run=args.dry_run, notify=args.notify))
            print()
    else:
        # Single day
        asyncio.run(run_daily_analysis(target_date, dry_run=args.dry_run, notify=args.notify))


if __name__ == "__main__":
    main()
