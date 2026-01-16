#!/usr/bin/env python3
"""
Lookahead-Fix Validation Script v2.0

Runs backtests using the SAME samples as previous runs for fair comparison.
Outputs real-time progress with win rate, cost, and time estimates.

Usage:
    python run_validation_v2_clean.py --samples 2000 --parallel 10 --test
    python run_validation_v2_clean.py --samples 2000 --parallel 10
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Force lookahead protection
os.environ["LOOKAHEAD_ASSERTIONS"] = "true"
os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_engine import analyze_earnings_async

# Import LLM provider utilities
try:
    from EarningsCallAgenticRag.utils.llm import (
        switch_to_azure_fallback,
        get_current_provider,
        build_litellm_client,
    )
    HAS_PROVIDER_UTILS = True
except ImportError:
    HAS_PROVIDER_UTILS = False


def check_and_switch_provider():
    """Check if LiteLLM is available, switch to Azure if not."""
    if not HAS_PROVIDER_UTILS:
        print("⚠️  Provider utils not available, using default provider")
        return

    print("🔍 Checking LiteLLM availability...")
    try:
        client, model = build_litellm_client("gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"✅ LiteLLM is available (provider: {get_current_provider()})")
    except Exception as e:
        error_str = str(e).lower()
        if "503" in error_str or "502" in error_str or "401" in error_str or "connection" in error_str:
            print(f"⚠️  LiteLLM unavailable: {str(e)[:100]}")
            print("🔄 Switching to Azure OpenAI fallback...")
            switch_to_azure_fallback()
            print(f"✅ Now using: {get_current_provider()}")
        else:
            print(f"⚠️  LiteLLM error (not switching): {str(e)[:100]}")


def load_previous_samples(target_count: int = 2000) -> List[Dict]:
    """Load samples from previous runs, evenly distributed across years."""

    # Load previous results
    df1 = pd.read_csv('long_only_test_2017_2018_20260101_131829.csv')
    df2 = pd.read_csv('long_only_test_2019_2024_combined.csv')
    df = pd.concat([df1, df2], ignore_index=True)

    # Calculate samples per year (evenly distributed)
    years = sorted(df['year'].unique())
    samples_per_year = target_count // len(years)
    remainder = target_count % len(years)

    selected = []
    for i, year in enumerate(years):
        year_df = df[df['year'] == year]
        n = samples_per_year + (1 if i < remainder else 0)
        n = min(n, len(year_df))

        # Sample evenly from this year
        sampled = year_df.sample(n=n, random_state=42)
        selected.append(sampled)

    result_df = pd.concat(selected, ignore_index=True)

    # Convert to list of dicts
    samples = []
    for _, row in result_df.iterrows():
        samples.append({
            'symbol': row['symbol'],
            'year': int(row['year']),
            'quarter': int(row['quarter']),
            'sector': row.get('sector', 'Unknown'),
            'category': row.get('category', 'Unknown'),
            'actual_return_30d': float(row['actual_return_30d_pct']) if pd.notna(row.get('actual_return_30d_pct')) else None,
            # Store previous result for comparison
            'prev_prediction': row.get('prediction'),
            'prev_correct': row.get('correct'),
            'prev_trade_long': row.get('trade_long'),
        })

    return samples


async def run_single_analysis(
    symbol: str,
    year: int,
    quarter: int,
    actual_return_30d: float = None,
) -> Dict:
    """Run single analysis and return result."""
    start = time.time()
    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,
        )
        elapsed = time.time() - start

        agentic_result = result.get("agentic_result", {})
        prediction = agentic_result.get("prediction", "UNKNOWN")
        confidence = agentic_result.get("confidence")
        direction_score = agentic_result.get("direction_score")
        trade_long = agentic_result.get("trade_long", False)
        trade_long_tier = agentic_result.get("trade_long_tier", "")
        long_eligible_json = agentic_result.get("long_eligible_json", {})
        summary = agentic_result.get("summary", "")

        # Determine correctness
        correct = None
        if actual_return_30d is not None and prediction in ["UP", "DOWN"]:
            if prediction == "UP":
                correct = actual_return_30d > 0
            else:
                correct = actual_return_30d < 0

        # Extract token usage
        raw = agentic_result.get("raw", {})
        token_usage = raw.get("token_usage", {}) if isinstance(raw, dict) else {}
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)

        # Estimate cost (GPT-4o-mini pricing)
        cost_usd = (input_tokens * 0.15 + output_tokens * 0.6) / 1_000_000

        return {
            "success": True,
            "error": None,
            "time_seconds": elapsed,
            "prediction": prediction,
            "confidence": confidence,
            "direction_score": direction_score,
            "correct": correct,
            "trade_long": trade_long,
            "trade_long_tier": trade_long_tier,
            "summary": summary[:500] if summary else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "risk_code": long_eligible_json.get("RiskCode", ""),
            "long_block_reason": long_eligible_json.get("LongBlockReason", ""),
            # Long eligible fields
            "le_DirectionScore": long_eligible_json.get("DirectionScore"),
            "le_LongEligible": long_eligible_json.get("LongEligible"),
            "le_HardPositivesCount": long_eligible_json.get("HardPositivesCount"),
            "le_HardVetoCount": long_eligible_json.get("HardVetoCount"),
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "error": str(e)[:200],
            "time_seconds": elapsed,
            "prediction": None,
            "confidence": None,
            "direction_score": None,
            "correct": None,
            "trade_long": False,
            "trade_long_tier": "",
            "summary": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0,
            "risk_code": "",
            "long_block_reason": "",
            "le_DirectionScore": None,
            "le_LongEligible": None,
            "le_HardPositivesCount": None,
            "le_HardVetoCount": None,
        }


class ProgressTracker:
    def __init__(self, total: int, report_interval: int = 25):
        self.total = total
        self.report_interval = report_interval
        self.completed = 0
        self.results = []
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def add_result(self, result: Dict):
        async with self.lock:
            self.results.append(result)
            self.completed += 1

            if self.completed % self.report_interval == 0 or self.completed == self.total:
                self.print_progress()

    def print_progress(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0

        successful = [r for r in self.results if r.get("success")]
        correct_count = sum(1 for r in successful if r.get("correct") == True)
        total_correct = sum(1 for r in successful if r.get("correct") is not None)
        accuracy = correct_count / total_correct * 100 if total_correct > 0 else 0

        long_trades = [r for r in successful if r.get("trade_long")]
        long_wins = sum(1 for r in long_trades if r.get("correct") == True)
        long_win_rate = long_wins / len(long_trades) * 100 if long_trades else 0
        long_avg_ret = sum(r.get("actual_return", 0) or 0 for r in long_trades) / len(long_trades) if long_trades else 0

        total_cost = sum(r.get("cost_usd", 0) for r in self.results)

        print(f"\n{'='*70}", flush=True)
        print(f"📊 PROGRESS [{self.completed}/{self.total}] ({self.completed/self.total*100:.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min | Rate: {rate*60:.1f}/min", flush=True)
        print(f"✅ Success: {len(successful)} | ❌ Failed: {len(self.results) - len(successful)}", flush=True)
        print(f"🎯 Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_correct})", flush=True)
        print(f"📈 Long-only: {len(long_trades)} trades | Win Rate: {long_win_rate:.1f}% ({long_wins}/{len(long_trades)})", flush=True)
        print(f"💰 Total Cost: ${total_cost:.4f} | Avg: ${total_cost/self.completed:.4f}/sample", flush=True)
        print(f"{'='*70}", flush=True)


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    idx: int,
    sample: Dict,
    tracker: ProgressTracker,
):
    """Run single analysis with semaphore control."""
    async with semaphore:
        symbol = sample["symbol"]
        year = sample["year"]
        quarter = sample["quarter"]

        print(f"[START {idx+1}/{tracker.total}] {symbol} {year}Q{quarter}")

        result = await run_single_analysis(
            symbol=symbol,
            year=year,
            quarter=quarter,
            actual_return_30d=sample.get("actual_return_30d"),
        )

        # Add sample info to result
        result["symbol"] = symbol
        result["year"] = year
        result["quarter"] = quarter
        result["sector"] = sample.get("sector", "")
        result["category"] = sample.get("category", "")
        result["actual_return"] = sample.get("actual_return_30d")
        result["prev_prediction"] = sample.get("prev_prediction")
        result["prev_correct"] = sample.get("prev_correct")
        result["prev_trade_long"] = sample.get("prev_trade_long")

        status = "✅" if result["success"] else "❌"
        pred = result.get("prediction", "?")
        ds = result.get("direction_score", "?")
        tl = "TL:True" if result.get("trade_long") else "TL:False"
        ret = f"Ret:{result.get('actual_return', 0):.1f}%" if result.get('actual_return') else ""
        correct = "Correct:True" if result.get("correct") == True else ("Correct:False" if result.get("correct") == False else "")

        print(f"[DONE {idx+1}/{tracker.total}] {symbol}: {pred}(D:{ds}) {tl} {ret} {correct}")

        await tracker.add_result(result)
        return result


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel workers")
    parser.add_argument("--test", action="store_true", help="Run only 10 samples for quick test")
    parser.add_argument("--report-interval", type=int, default=50, help="Progress report interval")
    args = parser.parse_args()

    num_samples = 10 if args.test else args.samples
    parallel = args.parallel
    report_interval = 5 if args.test else args.report_interval

    print("=" * 70)
    print("LOOKAHEAD-FIX VALIDATION v2.0 (Clean Data)")
    print("=" * 70)
    print(f"Samples: {num_samples} | Parallel: {parallel}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS: {os.environ.get('HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS')}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check provider
    check_and_switch_provider()

    # Load samples from previous runs
    print("\n📂 Loading samples from previous runs...")
    all_samples = load_previous_samples(target_count=num_samples)
    samples = all_samples[:num_samples]

    print(f"Loaded {len(samples)} samples")
    df_samples = pd.DataFrame(samples)
    print(f"Year distribution: {df_samples['year'].value_counts().sort_index().to_dict()}")

    # Run tests
    semaphore = asyncio.Semaphore(parallel)
    tracker = ProgressTracker(total=len(samples), report_interval=report_interval)

    print(f"\n🚀 Running {len(samples)} tests with {parallel} parallel workers...")

    tasks = [
        run_with_semaphore(semaphore, i, sample, tracker)
        for i, sample in enumerate(samples)
    ]

    results = await asyncio.gather(*tasks)

    # Final summary
    print("\n" + "=" * 70)
    print("🏁 FINAL SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n📊 Execution Stats:")
    print(f"  Total Samples: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    # Overall accuracy
    correct_results = [r for r in successful if r.get("correct") is not None]
    correct_count = sum(1 for r in correct_results if r.get("correct") == True)
    if correct_results:
        print(f"\n🎯 Overall Accuracy: {correct_count/len(correct_results)*100:.1f}% ({correct_count}/{len(correct_results)})")

    # Long-only stats
    long_trades = [r for r in successful if r.get("trade_long")]
    if long_trades:
        long_wins = sum(1 for r in long_trades if r.get("correct") == True)
        long_losses = sum(1 for r in long_trades if r.get("correct") == False)
        long_returns = [r.get("actual_return", 0) or 0 for r in long_trades]

        print(f"\n📈 Long-only Strategy:")
        print(f"  Coverage: {len(long_trades)}/{len(successful)} ({len(long_trades)/len(successful)*100:.1f}%)")
        print(f"  Win Rate: {long_wins/len(long_trades)*100:.1f}% ({long_wins}W / {long_losses}L)")
        print(f"  Avg Return: {sum(long_returns)/len(long_returns):+.2f}%")

        # By tier
        tier_stats = {}
        for r in long_trades:
            tier = r.get("trade_long_tier", "Unknown")
            if tier not in tier_stats:
                tier_stats[tier] = {"count": 0, "wins": 0, "returns": []}
            tier_stats[tier]["count"] += 1
            if r.get("correct") == True:
                tier_stats[tier]["wins"] += 1
            tier_stats[tier]["returns"].append(r.get("actual_return", 0) or 0)

        print(f"\n  By Tier:")
        for tier, stats in sorted(tier_stats.items()):
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg_ret = sum(stats["returns"]) / len(stats["returns"]) if stats["returns"] else 0
            print(f"    {tier}: {stats['count']} trades | {wr:.1f}% win | {avg_ret:+.1f}% avg")
    else:
        print("\n📈 No samples qualified for Long-only (trade_long=True)")

    # Comparison with previous run
    prev_correct = sum(1 for r in successful if r.get("prev_correct") == True)
    prev_total = sum(1 for r in successful if r.get("prev_correct") is not None)
    if prev_total > 0:
        print(f"\n🔄 Comparison with Previous Run:")
        print(f"  Previous Accuracy: {prev_correct/prev_total*100:.1f}% ({prev_correct}/{prev_total})")
        print(f"  Current Accuracy:  {correct_count/len(correct_results)*100:.1f}% ({correct_count}/{len(correct_results)})")

        prev_long = sum(1 for r in successful if r.get("prev_trade_long") == True)
        curr_long = len(long_trades)
        print(f"  Previous Long Trades: {prev_long}")
        print(f"  Current Long Trades:  {curr_long}")

    # Cost stats
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_time = time.time() - tracker.start_time
    print(f"\n💰 Cost & Time:")
    print(f"  Total Time: {total_time/60:.1f} min")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"  Avg Cost/Sample: ${total_cost/len(results):.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"validation_v2_clean_{timestamp}.csv"

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n📁 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
