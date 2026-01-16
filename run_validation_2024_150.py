#!/usr/bin/env python3
"""
2024 Validation Script - 150 samples with lookahead fix v2.1

Runs 150 samples from 2024 (balanced across Q1-Q4) using the fixed code.
Compares with previous results to verify fix effectiveness.

Usage:
    python run_validation_2024_150.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Force unbuffered output for real-time monitoring
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Force lookahead protection
os.environ["LOOKAHEAD_ASSERTIONS"] = "true"
os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"
os.environ["CALL_CACHE_VERSION"] = "v2.2"  # Bump version to invalidate old cache

from pg_client import get_cursor

# =============================================================================
# Configuration
# =============================================================================

TOTAL_SAMPLES = 150
CONCURRENCY = 5
TEST_SAMPLES = 3  # Run 3 first to verify output
YEAR = 2024

# =============================================================================
# Sample Selection
# =============================================================================

def get_2024_balanced_samples(limit: int = 150) -> List[Dict]:
    """Get balanced samples from 2024 across all quarters."""
    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        # Get samples balanced across quarters
        per_quarter = limit // 4 + 1
        samples = []

        for quarter in [1, 2, 3, 4]:
            cur.execute("""
                SELECT
                    et.symbol,
                    et.year,
                    et.quarter,
                    et.transcript_date_str,
                    pa.pct_change_t_plus_30 as actual_return,
                    c.name as company_name,
                    c.sector
                FROM earnings_transcripts et
                LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
                LEFT JOIN companies c ON et.symbol = c.symbol
                WHERE et.year = %s
                    AND et.quarter = %s
                    AND et.transcript_date_str IS NOT NULL
                    AND pa.pct_change_t_plus_30 IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
            """, (YEAR, quarter, per_quarter))

            for row in cur.fetchall():
                samples.append({
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "actual_return": float(row["actual_return"]) if row["actual_return"] else None,
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                })

        # Shuffle and limit
        import random
        random.shuffle(samples)
        return samples[:limit]


# =============================================================================
# Analysis Runner
# =============================================================================

async def analyze_sample(sample: Dict) -> Dict:
    """Run analysis on a single sample."""
    from analysis_engine import analyze_earnings_async

    start_time = time.time()
    symbol = sample["symbol"]
    year = sample["year"]
    quarter = sample["quarter"]

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,  # Force fresh analysis
        )

        elapsed = time.time() - start_time
        agentic = result.get("agentic_result", {})

        # Extract prediction and confidence
        prediction = agentic.get("prediction", "UNKNOWN")
        confidence = agentic.get("confidence")
        direction_score = None

        # Parse direction score from summary
        summary = agentic.get("summary", "")
        import re
        match = re.search(r"Direction\s*:\s*(\d+)", summary, re.IGNORECASE)
        if match:
            direction_score = int(match.group(1))

        # Calculate correctness
        actual_return = sample.get("actual_return")
        correct = None
        if actual_return is not None and prediction in ["UP", "DOWN"]:
            if prediction == "UP":
                correct = actual_return > 0
            else:
                correct = actual_return < 0

        # Extract trade info
        trade_long = agentic.get("trade_long", False)
        trade_long_tier = agentic.get("trade_long_tier", "")
        risk_code = agentic.get("risk_code", "")
        long_block_reason = agentic.get("long_block_reason", "")

        # Market anchors
        anchors = agentic.get("market_anchors", {}) or {}

        # Token usage
        raw = agentic.get("raw", {}) or {}
        token_usage = raw.get("token_usage", {}) or {}

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name"),
            "sector": sample.get("sector"),
            "success": True,
            "error": None,
            "time_seconds": elapsed,
            "prediction": prediction,
            "confidence": confidence,
            "direction_score": direction_score,
            "actual_return": actual_return,
            "correct": correct,
            "trade_long": trade_long,
            "trade_long_tier": trade_long_tier,
            "risk_code": risk_code,
            "long_block_reason": long_block_reason,
            "eps_surprise": anchors.get("eps_surprise"),
            "earnings_day_return": anchors.get("earnings_day_return"),
            "pre_earnings_5d_return": anchors.get("pre_earnings_5d_return"),
            "computed_positives": agentic.get("computed_positives"),
            "computed_vetoes": agentic.get("computed_vetoes"),
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "cost_usd": token_usage.get("total_cost_usd", 0),
            "summary": summary[:500] if summary else "",
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name"),
            "sector": sample.get("sector"),
            "success": False,
            "error": str(e),
            "time_seconds": time.time() - start_time,
        }


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.results: List[Dict] = []
        self.start_time = time.time()
        self.last_report_time = 0

    def add_result(self, result: Dict):
        self.results.append(result)
        self.completed += 1

        # Report progress every 30 seconds or every 10 samples
        now = time.time()
        if now - self.last_report_time >= 30 or self.completed % 10 == 0:
            self.report_progress()
            self.last_report_time = now

    def report_progress(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.completed
        eta = remaining / rate if rate > 0 else 0

        successful = [r for r in self.results if r.get("success")]
        correct_count = sum(1 for r in successful if r.get("correct") == True)
        total_with_result = sum(1 for r in successful if r.get("correct") is not None)
        accuracy = correct_count / total_with_result * 100 if total_with_result > 0 else 0

        long_trades = [r for r in successful if r.get("trade_long")]
        long_wins = sum(1 for r in long_trades if r.get("correct") == True)
        long_win_rate = long_wins / len(long_trades) * 100 if long_trades else 0
        long_avg_ret = sum(r.get("actual_return", 0) or 0 for r in long_trades) / len(long_trades) if long_trades else 0

        total_cost = sum(r.get("cost_usd", 0) or 0 for r in self.results)

        print(f"\n{'='*70}", flush=True)
        print(f"PROGRESS [{self.completed}/{self.total}] ({self.completed/self.total*100:.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min | Rate: {rate*60:.1f}/min", flush=True)
        print(f"Success: {len(successful)} | Failed: {len(self.results) - len(successful)}", flush=True)
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_with_result})", flush=True)
        print(f"Long-only: {len(long_trades)} trades | Win Rate: {long_win_rate:.1f}% ({long_wins}/{len(long_trades)})", flush=True)
        print(f"Long Avg Return: {long_avg_ret:.2f}%", flush=True)
        print(f"Total Cost: ${total_cost:.4f}", flush=True)
        print(f"{'='*70}", flush=True)


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    idx: int,
    sample: Dict,
    tracker: ProgressTracker,
):
    """Run single sample with semaphore for concurrency control."""
    async with semaphore:
        symbol = sample["symbol"]
        quarter = f"Q{sample['quarter']}"
        print(f"[{idx+1}/{tracker.total}] Starting {symbol} {YEAR}{quarter}...", flush=True)

        result = await analyze_sample(sample)
        tracker.add_result(result)

        status = "OK" if result.get("success") else "FAIL"
        pred = result.get("prediction", "?")
        correct = result.get("correct")
        correct_str = "Y" if correct == True else ("N" if correct == False else "?")
        trade = "LONG" if result.get("trade_long") else ""
        ret = result.get("actual_return")
        ret_str = f"{ret:+.1f}%" if ret is not None else "?"

        print(f"[{idx+1}/{tracker.total}] {symbol} {quarter}: {status} | {pred} | Correct={correct_str} | {trade} | Ret={ret_str}", flush=True)


async def main():
    print("=" * 70)
    print("2024 VALIDATION - 150 SAMPLES (Lookahead Fix v2.1)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"CALL_CACHE_VERSION: {os.environ.get('CALL_CACHE_VERSION')}")
    print()

    # Get samples
    print("Fetching 2024 samples (balanced across quarters)...")
    samples = get_2024_balanced_samples(TOTAL_SAMPLES)
    print(f"Found {len(samples)} samples")

    # Show quarter distribution
    quarter_counts = {}
    for s in samples:
        q = s["quarter"]
        quarter_counts[q] = quarter_counts.get(q, 0) + 1
    print(f"Quarter distribution: {quarter_counts}")
    print()

    # Phase 1: Test with 3 samples
    print(f"Phase 1: Testing with {TEST_SAMPLES} samples...")
    print("-" * 70)

    test_tracker = ProgressTracker(TEST_SAMPLES)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    test_tasks = [
        run_with_semaphore(semaphore, i, samples[i], test_tracker)
        for i in range(TEST_SAMPLES)
    ]
    await asyncio.gather(*test_tasks)

    # Show test results
    test_tracker.report_progress()

    # Check if test passed
    test_success_count = sum(1 for r in test_tracker.results if r.get("success"))
    if test_success_count < TEST_SAMPLES * 0.5:
        print(f"\nWARNING: Only {test_success_count}/{TEST_SAMPLES} test samples succeeded!")
        print("Aborting full run. Check errors above.")
        return

    print(f"\nTest phase passed ({test_success_count}/{TEST_SAMPLES} succeeded)")
    print("Proceeding with full run...")
    print()

    # Phase 2: Full run
    print(f"Phase 2: Running all {TOTAL_SAMPLES} samples...")
    print("-" * 70)

    full_tracker = ProgressTracker(TOTAL_SAMPLES)
    # Copy test results
    for r in test_tracker.results:
        full_tracker.results.append(r)
        full_tracker.completed += 1

    # Run remaining samples
    remaining_tasks = [
        run_with_semaphore(semaphore, i, samples[i], full_tracker)
        for i in range(TEST_SAMPLES, TOTAL_SAMPLES)
    ]
    await asyncio.gather(*remaining_tasks)

    # Final report
    full_tracker.report_progress()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"validation_2024_150_{timestamp}.csv"

    df = pd.DataFrame(full_tracker.results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Summary statistics
    successful = [r for r in full_tracker.results if r.get("success")]
    correct_count = sum(1 for r in successful if r.get("correct") == True)
    total_with_result = sum(1 for r in successful if r.get("correct") is not None)

    long_trades = [r for r in successful if r.get("trade_long")]
    long_wins = sum(1 for r in long_trades if r.get("correct") == True)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total Samples: {TOTAL_SAMPLES}")
    print(f"Successful: {len(successful)}")
    print(f"Overall Accuracy: {correct_count}/{total_with_result} ({correct_count/total_with_result*100:.1f}%)" if total_with_result > 0 else "N/A")
    print(f"Long Trades: {len(long_trades)}")
    print(f"Long Win Rate: {long_wins}/{len(long_trades)} ({long_wins/len(long_trades)*100:.1f}%)" if long_trades else "N/A")

    if long_trades:
        long_avg = sum(r.get("actual_return", 0) or 0 for r in long_trades) / len(long_trades)
        print(f"Long Avg Return: {long_avg:.2f}%")

        # Breakdown by tier
        d7_trades = [r for r in long_trades if r.get("trade_long_tier") == "D7_CORE"]
        d6_trades = [r for r in long_trades if r.get("trade_long_tier") == "D6_STRICT"]
        d7_wins = sum(1 for r in d7_trades if r.get("correct") == True)
        d6_wins = sum(1 for r in d6_trades if r.get("correct") == True)

        print(f"  D7_CORE: {len(d7_trades)} trades, {d7_wins} wins ({d7_wins/len(d7_trades)*100:.1f}%)" if d7_trades else "  D7_CORE: 0 trades")
        print(f"  D6_STRICT: {len(d6_trades)} trades, {d6_wins} wins ({d6_wins/len(d6_trades)*100:.1f}%)" if d6_trades else "  D6_STRICT: 0 trades")

        # Long trade details
        print("\nLong Trade Details:")
        for t in long_trades:
            win_str = "WIN" if t.get("correct") else "LOSS"
            print(f"  {t['symbol']} Q{t['quarter']}: {t.get('actual_return', 0):+.2f}% ({win_str}) - {t.get('trade_long_tier', 'N/A')}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
