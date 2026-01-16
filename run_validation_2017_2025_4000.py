#!/usr/bin/env python3
"""
2017-2025 Q3 Validation Script - 4000 samples with lookahead fix v2.2

Runs up to 4000 samples from 2017-2025 Q3 (balanced across quarters) using the fixed code.
CAP=20 per quarter for long trades.

Features:
- Balanced sampling across all quarters (2017Q1 to 2025Q3)
- Progress reporting every 5 samples and 30 seconds
- Test phase (5 samples) before full run
- Results saved to CSV

Usage:
    python run_validation_2017_2025_4000.py [--test-only] [--samples N]
"""

import argparse
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
os.environ["CALL_CACHE_VERSION"] = "v2.2"  # Bump version for v2.2 fixes

from pg_client import get_cursor

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLES = 4000
CONCURRENCY = 10
TEST_SAMPLES = 5
START_YEAR = 2017
END_YEAR = 2025
END_QUARTER = 3  # Up to 2025 Q3

# =============================================================================
# Sample Selection
# =============================================================================

def get_all_quarters() -> List[tuple]:
    """Get all quarter combinations from 2017Q1 to 2025Q3."""
    quarters = []
    for year in range(START_YEAR, END_YEAR + 1):
        max_q = END_QUARTER if year == END_YEAR else 4
        for q in range(1, max_q + 1):
            quarters.append((year, q))
    return quarters


def get_balanced_samples(limit: int = 4000) -> List[Dict]:
    """Get balanced samples from 2017-2025 across all quarters."""
    quarters = get_all_quarters()
    num_quarters = len(quarters)
    per_quarter = (limit // num_quarters) + 2  # Extra buffer for uneven distribution

    print(f"Fetching samples from {num_quarters} quarters ({START_YEAR}Q1 to {END_YEAR}Q{END_QUARTER})...")
    print(f"Target ~{per_quarter} samples per quarter...")

    all_samples = []
    quarter_counts = {}

    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        for year, quarter in quarters:
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
            """, (year, quarter, per_quarter))

            quarter_samples = []
            for row in cur.fetchall():
                quarter_samples.append({
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "actual_return": float(row["actual_return"]) if row["actual_return"] else None,
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                })

            all_samples.extend(quarter_samples)
            quarter_key = f"{year}Q{quarter}"
            quarter_counts[quarter_key] = len(quarter_samples)

    # Shuffle and limit
    import random
    random.shuffle(all_samples)
    samples = all_samples[:limit]

    # Recalculate quarter distribution after limiting
    final_counts = {}
    for s in samples:
        key = f"{s['year']}Q{s['quarter']}"
        final_counts[key] = final_counts.get(key, 0) + 1

    print(f"\nFetched {len(all_samples)} total samples, using {len(samples)}")
    print(f"Quarter distribution (sample): {dict(list(final_counts.items())[:5])}...")

    return samples


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

        # LongEligible data
        le_data = agentic.get("long_eligible", {}) or {}

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
            # LongEligible fields
            "le_DirectionScore": le_data.get("DirectionScore"),
            "le_HardPositivesCount": le_data.get("HardPositivesCount"),
            "le_HardVetoCount": le_data.get("HardVetoCount"),
            "le_PricedInRisk": le_data.get("PricedInRisk"),
            "le_GuidanceRaised": le_data.get("GuidanceRaised"),
            "le_DemandAcceleration": le_data.get("DemandAcceleration"),
            "le_MarginExpansion": le_data.get("MarginExpansion"),
            "le_FCFImprovement": le_data.get("FCFImprovement"),
            "le_VisibilityImproving": le_data.get("VisibilityImproving"),
            "le_GuidanceCut": le_data.get("GuidanceCut"),
            "le_DemandSoftness": le_data.get("DemandSoftness"),
            "le_MarginWeakness": le_data.get("MarginWeakness"),
            "le_CashBurn": le_data.get("CashBurn"),
            "le_VisibilityWorsening": le_data.get("VisibilityWorsening"),
            # Token usage
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "cost_usd": token_usage.get("total_cost_usd", 0),
            "summary": summary[:500] if summary else "",
        }

    except Exception as e:
        import traceback
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name"),
            "sector": sample.get("sector"),
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()[:500],
            "time_seconds": time.time() - start_time,
        }


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.results: List[Dict] = []
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = 30  # seconds

    def add_result(self, result: Dict):
        self.results.append(result)
        self.completed += 1

        # Report progress every 30 seconds or every 5 samples
        now = time.time()
        if now - self.last_report_time >= self.report_interval or self.completed % 5 == 0:
            self.report_progress()
            self.last_report_time = now

    def report_progress(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.completed
        eta = remaining / rate if rate > 0 else 0

        successful = [r for r in self.results if r.get("success")]
        failed = len(self.results) - len(successful)
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
        print(f"Success: {len(successful)} | Failed: {failed}", flush=True)
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
        year = sample["year"]
        quarter = f"Q{sample['quarter']}"
        print(f"[{idx+1}/{tracker.total}] Starting {symbol} {year}{quarter}...", flush=True)

        result = await analyze_sample(sample)
        tracker.add_result(result)

        status = "OK" if result.get("success") else "FAIL"
        pred = result.get("prediction", "?")
        correct = result.get("correct")
        correct_str = "Y" if correct == True else ("N" if correct == False else "?")
        trade = "LONG" if result.get("trade_long") else ""

        print(f"[{idx+1}/{tracker.total}] {symbol} {year}{quarter}: {status} | {pred} | Correct={correct_str} | {trade}", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="2017-2025 Validation (4000 samples)")
    parser.add_argument("--test-only", action="store_true", help="Only run test phase")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of samples to run")
    args = parser.parse_args()

    total_samples = args.samples

    print("=" * 70)
    print(f"2017-2025 VALIDATION - {total_samples} SAMPLES (Lookahead Fix v2.2)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"CALL_CACHE_VERSION: {os.environ.get('CALL_CACHE_VERSION')}")
    print(f"Test samples: {TEST_SAMPLES}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # Get samples
    samples = get_balanced_samples(total_samples)
    print(f"Loaded {len(samples)} samples")
    print()

    # Phase 1: Test with 5 samples
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
        print(f"\nERROR: Only {test_success_count}/{TEST_SAMPLES} test samples succeeded!")
        print("Aborting full run. Check errors above.")

        # Show errors
        for r in test_tracker.results:
            if not r.get("success"):
                print(f"  {r.get('symbol')}: {r.get('error')}")
        return

    print(f"\n✓ Test phase passed ({test_success_count}/{TEST_SAMPLES} succeeded)")

    if args.test_only:
        print("--test-only flag set, stopping here.")
        return

    print("Proceeding with full run...")
    print()

    # Phase 2: Full run
    print(f"Phase 2: Running all {len(samples)} samples...")
    print("-" * 70)

    full_tracker = ProgressTracker(len(samples))
    # Copy test results
    for r in test_tracker.results:
        full_tracker.results.append(r)
        full_tracker.completed += 1

    # Run remaining samples
    remaining_tasks = [
        run_with_semaphore(semaphore, i, samples[i], full_tracker)
        for i in range(TEST_SAMPLES, len(samples))
    ]
    await asyncio.gather(*remaining_tasks)

    # Final report
    full_tracker.report_progress()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"validation_2017_2025_{len(samples)}_{timestamp}.csv"

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
    print(f"Total Samples: {len(samples)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(full_tracker.results) - len(successful)}")
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

        # Yearly breakdown
        print("\nLong trades by year:")
        yearly = {}
        for r in long_trades:
            y = r.get("year")
            if y not in yearly:
                yearly[y] = {"trades": 0, "wins": 0}
            yearly[y]["trades"] += 1
            if r.get("correct") == True:
                yearly[y]["wins"] += 1

        for y in sorted(yearly.keys()):
            data = yearly[y]
            wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
            print(f"  {y}: {data['trades']} trades, {data['wins']} wins ({wr:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
