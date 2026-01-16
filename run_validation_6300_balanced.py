#!/usr/bin/env python3
"""
2017-2025 Q3 Validation Script - 6300 samples (180/quarter) with lookahead fix v2.2

Runs 6300 samples from 2017-2025 Q3 with FIXED 180 samples per quarter (stratified sampling).
This ensures reproducible, balanced sampling for statistical validity.

Target: ~500 long trades (based on ~8% trade_long rate from 4000 sample run)

Features:
- FIXED 180 samples per quarter (deterministic stratified sampling)
- Reproducible with seed (default: 42)
- Progress reporting every 10 samples and 60 seconds
- Test phase (5 samples) before full run
- Results saved to CSV

Usage:
    python run_validation_6300_balanced.py [--test-only] [--samples N] [--seed SEED]
"""

import argparse
import asyncio
import hashlib
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

DEFAULT_SAMPLES = 6300  # 35 quarters × 180 samples
SAMPLES_PER_QUARTER = 180
CONCURRENCY = 10
TEST_SAMPLES = 5
START_YEAR = 2017
END_YEAR = 2025
END_QUARTER = 3  # Up to 2025 Q3
DEFAULT_SEED = 42

# =============================================================================
# Sample Selection (Stratified, Reproducible)
# =============================================================================

def get_all_quarters() -> List[tuple]:
    """Get all quarter combinations from 2017Q1 to 2025Q3."""
    quarters = []
    for year in range(START_YEAR, END_YEAR + 1):
        max_q = END_QUARTER if year == END_YEAR else 4
        for q in range(1, max_q + 1):
            quarters.append((year, q))
    return quarters


def deterministic_hash(symbol: str, year: int, quarter: int, seed: int) -> str:
    """Generate deterministic hash for ordering (reproducible across runs)."""
    key = f"{symbol}:{year}:Q{quarter}:seed{seed}"
    return hashlib.md5(key.encode()).hexdigest()


def get_stratified_samples(per_quarter: int = 180, seed: int = 42) -> List[Dict]:
    """
    Get FIXED number of samples per quarter using deterministic ordering.

    This ensures:
    1. Reproducibility: Same seed = same samples
    2. Balance: Exactly per_quarter samples from each quarter
    3. No selection bias: Hash-based ordering is independent of actual_return
    """
    quarters = get_all_quarters()
    num_quarters = len(quarters)

    print(f"Stratified sampling: {per_quarter} samples/quarter × {num_quarters} quarters = {per_quarter * num_quarters} total")
    print(f"Seed: {seed}")

    all_samples = []
    quarter_counts = {}

    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        for year, quarter in quarters:
            # Fetch more than needed to allow hash-based selection
            fetch_limit = per_quarter * 3  # Buffer for quarters with less data

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
                LIMIT %s
            """, (year, quarter, fetch_limit))

            quarter_samples = []
            for row in cur.fetchall():
                sample = {
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "actual_return": float(row["actual_return"]) if row["actual_return"] else None,
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                }
                # Add deterministic hash for ordering
                sample["_hash"] = deterministic_hash(
                    row["symbol"], year, quarter, seed
                )
                quarter_samples.append(sample)

            # Sort by hash (deterministic) and take first per_quarter
            quarter_samples.sort(key=lambda x: x["_hash"])
            selected = quarter_samples[:per_quarter]

            # Remove hash from final output
            for s in selected:
                del s["_hash"]

            all_samples.extend(selected)
            quarter_key = f"{year}Q{quarter}"
            quarter_counts[quarter_key] = len(selected)

            if len(selected) < per_quarter:
                print(f"  Warning: {quarter_key} only has {len(selected)} samples (target: {per_quarter})")

    # Final shuffle (but deterministic with seed)
    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Quarters with full coverage: {sum(1 for v in quarter_counts.values() if v >= per_quarter)}/{len(quarters)}")

    # Verify distribution
    final_counts = {}
    for s in all_samples:
        key = f"{s['year']}Q{s['quarter']}"
        final_counts[key] = final_counts.get(key, 0) + 1

    min_count = min(final_counts.values())
    max_count = max(final_counts.values())
    print(f"Samples per quarter: min={min_count}, max={max_count}")

    return all_samples


def save_universe_manifest(samples: List[Dict], seed: int, output_dir: str = "."):
    """Save universe manifest for reproducibility and audit."""
    manifest_path = Path(output_dir) / f"universe_2017Q1_2025Q3_{SAMPLES_PER_QUARTER}perQ_seed{seed}.csv"

    manifest_data = []
    for s in samples:
        manifest_data.append({
            "symbol": s["symbol"],
            "year": s["year"],
            "quarter": s["quarter"],
            "transcript_date": s["transcript_date"],
            "sector": s["sector"],
        })

    df = pd.DataFrame(manifest_data)
    df.to_csv(manifest_path, index=False)
    print(f"Universe manifest saved to: {manifest_path}")
    return manifest_path


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
    transcript_date = sample.get("transcript_date")
    actual_return = sample.get("actual_return")

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            transcript_date=transcript_date,
        )

        elapsed = time.time() - start_time

        # Extract prediction info
        prediction = result.get("prediction", "")
        confidence = result.get("confidence", 0)
        direction_score = result.get("le_DirectionScore")
        trade_long = result.get("trade_long", False)
        trade_long_tier = result.get("trade_long_tier", "")
        risk_code = result.get("risk_code", "")
        long_block_reason = result.get("long_block_reason", "")

        # Determine correctness
        if actual_return is not None and prediction:
            pred_up = prediction.upper() == "UP"
            actual_up = actual_return > 0
            correct = pred_up == actual_up
        else:
            correct = None

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name", ""),
            "sector": sample.get("sector", ""),
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
            "eps_surprise": result.get("eps_surprise"),
            "earnings_day_return": result.get("earnings_day_return"),
            "pre_earnings_5d_return": result.get("pre_earnings_5d_return"),
            "computed_positives": result.get("computed_positives"),
            "computed_vetoes": result.get("computed_vetoes"),
            # LongEligible structured output fields
            "le_DirectionScore": result.get("le_DirectionScore"),
            "le_HardPositivesCount": result.get("le_HardPositivesCount"),
            "le_HardVetoCount": result.get("le_HardVetoCount"),
            "le_PricedInRisk": result.get("le_PricedInRisk"),
            "le_GuidanceRaised": result.get("le_GuidanceRaised"),
            "le_DemandAcceleration": result.get("le_DemandAcceleration"),
            "le_MarginExpansion": result.get("le_MarginExpansion"),
            "le_FCFImprovement": result.get("le_FCFImprovement"),
            "le_VisibilityImproving": result.get("le_VisibilityImproving"),
            "le_GuidanceCut": result.get("le_GuidanceCut"),
            "le_DemandSoftness": result.get("le_DemandSoftness"),
            "le_MarginWeakness": result.get("le_MarginWeakness"),
            "le_CashBurn": result.get("le_CashBurn"),
            "le_VisibilityWorsening": result.get("le_VisibilityWorsening"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
            "summary": result.get("summary", "")[:500] if result.get("summary") else "",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name", ""),
            "sector": sample.get("sector", ""),
            "success": False,
            "error": str(e)[:200],
            "time_seconds": elapsed,
            "prediction": None,
            "confidence": None,
            "direction_score": None,
            "actual_return": actual_return,
            "correct": None,
            "trade_long": False,
            "trade_long_tier": None,
            "risk_code": None,
            "long_block_reason": None,
        }


async def run_batch(samples: List[Dict], concurrency: int = CONCURRENCY) -> List[Dict]:
    """Run analysis on all samples with concurrency limit."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    completed = 0
    start_time = time.time()
    last_report_time = start_time
    report_interval = 60  # Report every 60 seconds
    report_count = 10  # Also report every N completions

    async def run_with_semaphore(idx: int, sample: Dict) -> Dict:
        nonlocal completed, last_report_time
        async with semaphore:
            print(f"[{idx+1}/{len(samples)}] Starting {sample['symbol']} {sample['year']}Q{sample['quarter']}...")
            result = await analyze_sample(sample)
            completed += 1

            # Progress indicator
            status = "OK" if result["success"] else "FAIL"
            pred = result.get("prediction", "N/A")
            correct = "Y" if result.get("correct") else ("N" if result.get("correct") is False else "?")
            trade_long = "LONG" if result.get("trade_long") else ""
            print(f"[{idx+1}/{len(samples)}] {sample['symbol']} Q{sample['quarter']}: {status} | {pred} | Correct={correct} | {trade_long}")

            # Periodic progress report
            current_time = time.time()
            if completed % report_count == 0 or (current_time - last_report_time) >= report_interval:
                last_report_time = current_time
                await print_progress(results + [result], completed, len(samples), start_time)

            return result

    tasks = [run_with_semaphore(i, s) for i, s in enumerate(samples)]
    results = await asyncio.gather(*tasks)

    return results


async def print_progress(results: List[Dict], completed: int, total: int, start_time: float):
    """Print progress summary."""
    elapsed = time.time() - start_time
    rate = completed / elapsed * 60 if elapsed > 0 else 0
    eta = (total - completed) / (completed / elapsed) if completed > 0 else 0

    successful = [r for r in results if r.get("success")]
    correct = [r for r in successful if r.get("correct")]
    long_trades = [r for r in successful if r.get("trade_long")]
    long_wins = [r for r in long_trades if r.get("actual_return", 0) > 0]

    print(f"\n{'='*70}")
    print(f"PROGRESS [{completed}/{total}] ({100*completed/total:.1f}%)")
    print(f"{'='*70}")
    print(f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min | Rate: {rate:.1f}/min")
    print(f"Success: {len(successful)} | Failed: {completed - len(successful)}")

    if successful:
        acc = len(correct) / len(successful) * 100 if successful else 0
        print(f"Overall Accuracy: {acc:.1f}% ({len(correct)}/{len(successful)})")

    if long_trades:
        win_rate = len(long_wins) / len(long_trades) * 100
        avg_return = sum(r.get("actual_return", 0) for r in long_trades) / len(long_trades)
        print(f"Long-only: {len(long_trades)} trades | Win Rate: {win_rate:.1f}% ({len(long_wins)}/{len(long_trades)})")
        print(f"Long Avg Return: {avg_return:.2f}%")

        # Tier breakdown
        d7 = [r for r in long_trades if r.get("trade_long_tier") == "D7_CORE"]
        d6 = [r for r in long_trades if r.get("trade_long_tier") == "D6_STRICT"]
        if d7:
            d7_wins = len([r for r in d7 if r.get("actual_return", 0) > 0])
            print(f"  D7_CORE: {len(d7)} trades, {d7_wins} wins ({100*d7_wins/len(d7):.1f}%)")
        if d6:
            d6_wins = len([r for r in d6 if r.get("actual_return", 0) > 0])
            print(f"  D6_STRICT: {len(d6)} trades, {d6_wins} wins ({100*d6_wins/len(d6):.1f}%)")

    print(f"{'='*70}\n")


def save_results(results: List[Dict], prefix: str = "validation_6300") -> str:
    """Save results to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    return filename


def print_final_summary(results: List[Dict], seed: int):
    """Print comprehensive final summary."""
    successful = [r for r in results if r.get("success")]
    correct = [r for r in successful if r.get("correct")]
    long_trades = [r for r in successful if r.get("trade_long")]
    long_wins = [r for r in long_trades if r.get("actual_return", 0) > 0]

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY (Seed: {seed})")
    print(f"{'='*70}")
    print(f"Total Samples: {len(results)}")
    print(f"Successful: {len(successful)}")

    if successful:
        print(f"Overall Accuracy: {len(correct)}/{len(successful)} ({100*len(correct)/len(successful):.1f}%)")

    print(f"\nLong Trades: {len(long_trades)}")
    if long_trades:
        win_rate = len(long_wins) / len(long_trades) * 100
        avg_return = sum(r.get("actual_return", 0) for r in long_trades) / len(long_trades)
        print(f"Long Win Rate: {len(long_wins)}/{len(long_trades)} ({win_rate:.1f}%)")
        print(f"Long Avg Return: {avg_return:.2f}%")

        # Tier breakdown
        d7 = [r for r in long_trades if r.get("trade_long_tier") == "D7_CORE"]
        d6 = [r for r in long_trades if r.get("trade_long_tier") == "D6_STRICT"]
        if d7:
            d7_wins = len([r for r in d7 if r.get("actual_return", 0) > 0])
            d7_avg = sum(r.get("actual_return", 0) for r in d7) / len(d7)
            print(f"  D7_CORE: {len(d7)} trades, {d7_wins} wins ({100*d7_wins/len(d7):.1f}%), avg {d7_avg:.2f}%")
        if d6:
            d6_wins = len([r for r in d6 if r.get("actual_return", 0) > 0])
            d6_avg = sum(r.get("actual_return", 0) for r in d6) / len(d6)
            print(f"  D6_STRICT: {len(d6)} trades, {d6_wins} wins ({100*d6_wins/len(d6):.1f}%), avg {d6_avg:.2f}%")

        # Yearly breakdown
        print(f"\nYearly Breakdown:")
        by_year = {}
        for r in long_trades:
            y = r.get("year")
            if y not in by_year:
                by_year[y] = {"trades": 0, "wins": 0, "returns": []}
            by_year[y]["trades"] += 1
            if r.get("actual_return", 0) > 0:
                by_year[y]["wins"] += 1
            by_year[y]["returns"].append(r.get("actual_return", 0))

        for year in sorted(by_year.keys()):
            data = by_year[year]
            wr = 100 * data["wins"] / data["trades"] if data["trades"] > 0 else 0
            avg = sum(data["returns"]) / len(data["returns"]) if data["returns"] else 0
            print(f"  {year}: {data['trades']} trades, {wr:.1f}% win rate, {avg:.2f}% avg")

        # Wilson CI
        try:
            from scipy import stats
            n = len(long_trades)
            k = len(long_wins)
            p = k / n
            z = stats.norm.ppf(0.975)
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * ((p * (1-p) + z**2 / (4*n)) / n) ** 0.5 / denominator
            lower = max(0, center - margin)
            upper = min(1, center + margin)
            print(f"\nWilson 95% CI for Long Win Rate: [{100*lower:.1f}%, {100*upper:.1f}%]")
        except ImportError:
            pass

    print(f"{'='*70}")


async def main():
    parser = argparse.ArgumentParser(description="Run 6300 sample validation (stratified)")
    parser.add_argument("--test-only", action="store_true", help="Only run test samples")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--per-quarter", type=int, default=SAMPLES_PER_QUARTER, help="Samples per quarter")
    parser.add_argument("--save-manifest", action="store_true", help="Save universe manifest CSV")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"6300 SAMPLE STRATIFIED VALIDATION (v2.2)")
    print(f"{'='*70}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"CALL_CACHE_VERSION: {os.environ.get('CALL_CACHE_VERSION')}")
    print(f"Seed: {args.seed}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"{'='*70}\n")

    # Get stratified samples
    samples = get_stratified_samples(per_quarter=args.per_quarter, seed=args.seed)

    # Optionally save manifest
    if args.save_manifest:
        save_universe_manifest(samples, args.seed)

    # Limit if specified
    if args.samples < len(samples):
        samples = samples[:args.samples]
        print(f"Limited to {len(samples)} samples")

    # Test phase
    print(f"\n--- TEST PHASE ({TEST_SAMPLES} samples) ---\n")
    test_results = await run_batch(samples[:TEST_SAMPLES], concurrency=TEST_SAMPLES)

    test_success = sum(1 for r in test_results if r.get("success"))
    print(f"\nTest Results: {test_success}/{TEST_SAMPLES} successful")

    if test_success < TEST_SAMPLES:
        print("WARNING: Some test samples failed. Check configuration.")

    if args.test_only:
        print("\n--test-only flag set. Stopping after test phase.")
        save_results(test_results, prefix="validation_6300_test")
        return

    # Full run
    print(f"\n--- FULL RUN ({len(samples)} samples) ---\n")
    all_results = await run_batch(samples, concurrency=CONCURRENCY)

    # Save and summarize
    output_file = save_results(all_results, prefix="validation_6300_balanced")
    print_final_summary(all_results, args.seed)

    print(f"\nResults file: {output_file}")


if __name__ == "__main__":
    # Change to EarningsCallAgenticRag directory
    script_dir = Path(__file__).parent
    rag_dir = script_dir / "EarningsCallAgenticRag"
    if rag_dir.exists():
        os.chdir(rag_dir)

    asyncio.run(main())
