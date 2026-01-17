#!/usr/bin/env python3
"""
2025 Full Year Holdout Validation Script
- Q1-Q4 complete coverage
- Balanced GAINER/LOSER samples
- Out-of-sample validation for entire 2025
"""

import asyncio
import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_engine import analyze_earnings_async
from pg_client import get_cursor

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
            switch_to_azure_fallback()
            print(f"✅ Now using: {get_current_provider()}")


def check_data_availability():
    """Check how many samples are available in 2025 by quarter."""
    with get_cursor() as cur:
        if cur is None:
            print("❌ Database connection failed")
            return

        print("\n" + "=" * 70)
        print("📊 2025 FULL YEAR DATA AVAILABILITY")
        print("=" * 70)

        # Total transcripts by quarter
        cur.execute("""
            SELECT et.quarter, COUNT(*) as cnt
            FROM earnings_transcripts et
            JOIN transcript_content tc ON et.id = tc.transcript_id
            WHERE et.year = 2025
              AND tc.content IS NOT NULL
              AND LENGTH(tc.content) > 1000
            GROUP BY et.quarter
            ORDER BY et.quarter
        """)
        print("\n📄 Transcripts with content (>1000 chars):")
        total_transcripts = 0
        for row in cur.fetchall():
            print(f"  Q{row[0]}: {row[1]:,} 筆")
            total_transcripts += row[1]
        print(f"  Total: {total_transcripts:,} 筆")

        # With T+30 return by quarter
        cur.execute("""
            SELECT et.quarter, COUNT(*) as cnt
            FROM earnings_transcripts et
            JOIN transcript_content tc ON et.id = tc.transcript_id
            JOIN price_analysis pa ON et.id = pa.transcript_id
            WHERE et.year = 2025
              AND tc.content IS NOT NULL
              AND LENGTH(tc.content) > 1000
              AND pa.pct_change_t_plus_30 IS NOT NULL
            GROUP BY et.quarter
            ORDER BY et.quarter
        """)
        print("\n📈 With T+30 return data:")
        total_with_return = 0
        for row in cur.fetchall():
            print(f"  Q{row[0]}: {row[1]:,} 筆")
            total_with_return += row[1]
        print(f"  Total: {total_with_return:,} 筆")

        # GAINER/LOSER distribution at different thresholds
        print("\n📊 GAINER/LOSER Distribution by Threshold:")
        for threshold in [10, 7, 5, 3]:
            cur.execute("""
                SELECT
                    et.quarter,
                    SUM(CASE WHEN pa.pct_change_t_plus_30 > %s THEN 1 ELSE 0 END) as gainers,
                    SUM(CASE WHEN pa.pct_change_t_plus_30 < %s THEN 1 ELSE 0 END) as losers
                FROM earnings_transcripts et
                JOIN transcript_content tc ON et.id = tc.transcript_id
                JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE et.year = 2025
                  AND tc.content IS NOT NULL
                  AND LENGTH(tc.content) > 1000
                  AND pa.pct_change_t_plus_30 IS NOT NULL
                GROUP BY et.quarter
                ORDER BY et.quarter
            """, (threshold, -threshold))

            print(f"\n  ±{threshold}% threshold:")
            total_g, total_l = 0, 0
            for row in cur.fetchall():
                print(f"    Q{row[0]}: GAINER={row[1]}, LOSER={row[2]}")
                total_g += row[1]
                total_l += row[2]
            print(f"    Total: GAINER={total_g}, LOSER={total_l}")

        print("\n" + "=" * 70)


def get_2025_samples(
    limit_per_category: int = 500,
    gainer_threshold: float = 5.0,
    loser_threshold: float = -5.0,
) -> List[Dict]:
    """
    Get balanced samples from 2025 Q1-Q4.
    Distributes evenly across quarters.
    """
    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        samples = []
        quarters = [1, 2, 3, 4]
        samples_per_quarter_per_cat = limit_per_category // len(quarters) + 1

        # GAINERS by quarter
        for quarter in quarters:
            cur.execute("""
                SELECT
                    et.symbol,
                    et.year,
                    et.quarter,
                    et.transcript_date_str,
                    c.name as company_name,
                    c.sector,
                    pa.pct_change_t_plus_30 as actual_return_30d
                FROM earnings_transcripts et
                JOIN companies c ON et.symbol = c.symbol
                JOIN transcript_content tc ON et.id = tc.transcript_id
                JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE et.year = 2025
                    AND et.quarter = %s
                    AND tc.content IS NOT NULL
                    AND LENGTH(tc.content) > 1000
                    AND pa.pct_change_t_plus_30 IS NOT NULL
                    AND pa.pct_change_t_plus_30 > %s
                ORDER BY RANDOM()
                LIMIT %s
            """, (quarter, gainer_threshold, samples_per_quarter_per_cat))

            for row in cur.fetchall():
                samples.append({
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "actual_return_30d": float(row["actual_return_30d"]),
                    "category": "GAINER"
                })

        # LOSERS by quarter
        for quarter in quarters:
            cur.execute("""
                SELECT
                    et.symbol,
                    et.year,
                    et.quarter,
                    et.transcript_date_str,
                    c.name as company_name,
                    c.sector,
                    pa.pct_change_t_plus_30 as actual_return_30d
                FROM earnings_transcripts et
                JOIN companies c ON et.symbol = c.symbol
                JOIN transcript_content tc ON et.id = tc.transcript_id
                JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE et.year = 2025
                    AND et.quarter = %s
                    AND tc.content IS NOT NULL
                    AND LENGTH(tc.content) > 1000
                    AND pa.pct_change_t_plus_30 IS NOT NULL
                    AND pa.pct_change_t_plus_30 < %s
                ORDER BY RANDOM()
                LIMIT %s
            """, (quarter, loser_threshold, samples_per_quarter_per_cat))

            for row in cur.fetchall():
                samples.append({
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "actual_return_30d": float(row["actual_return_30d"]),
                    "category": "LOSER"
                })

        return samples


async def run_single_test(symbol: str, year: int, quarter: int, category: str,
                          company_name: str, sector: str, actual_return_30d: float = None):
    """Run a single test and return results."""
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
        summary = agentic_result.get("summary", "")
        trade_long = agentic_result.get("trade_long", False)
        long_eligible_json = agentic_result.get("long_eligible_json", {})

        actual_return = actual_return_30d

        correct = None
        if actual_return is not None and prediction in ["UP", "DOWN"]:
            correct = (actual_return > 0) if prediction == "UP" else (actual_return < 0)

        direction_score = long_eligible_json.get("DirectionScore") if long_eligible_json else None

        risk_code = agentic_result.get("risk_code", "unknown")
        long_block_reason = agentic_result.get("long_block_reason", "")
        trade_long_tier = agentic_result.get("trade_long_tier", "")
        market_anchors = agentic_result.get("market_anchors", {}) or {}
        computed_positives = agentic_result.get("computed_positives", 0)
        computed_vetoes = agentic_result.get("computed_vetoes", 0)

        token_usage = result.get("agentic_result", {}).get("raw", {}).get("token_usage", {})
        input_tokens = token_usage.get("input_tokens", 0) if token_usage else 0
        output_tokens = token_usage.get("output_tokens", 0) if token_usage else 0
        cost_usd = token_usage.get("cost_usd", 0.0) if token_usage else 0.0

        return {
            "symbol": symbol, "year": year, "quarter": quarter,
            "category": category, "company_name": company_name, "sector": sector,
            "success": True, "error": None, "time_seconds": elapsed,
            "prediction": prediction, "confidence": confidence,
            "direction_score": direction_score,
            "actual_return_30d_pct": actual_return, "correct": correct,
            "trade_long": trade_long, "trade_long_tier": trade_long_tier,
            "long_eligible_json": long_eligible_json,
            "summary": summary[:500] if summary else "",
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd,
            "risk_code": risk_code, "long_block_reason": long_block_reason,
            "eps_surprise": market_anchors.get("eps_surprise"),
            "earnings_day_return": market_anchors.get("earnings_day_return"),
            "pre_earnings_5d_return": market_anchors.get("pre_earnings_5d_return"),
            "computed_positives": computed_positives, "computed_vetoes": computed_vetoes,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "symbol": symbol, "year": year, "quarter": quarter,
            "category": category, "company_name": company_name, "sector": sector,
            "success": False, "error": str(e), "time_seconds": elapsed,
            "prediction": None, "confidence": None, "direction_score": None,
            "actual_return_30d_pct": actual_return_30d, "correct": None,
            "trade_long": False, "trade_long_tier": "",
            "long_eligible_json": None, "summary": None,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "risk_code": "unknown", "long_block_reason": "EXCEPTION",
            "eps_surprise": None, "earnings_day_return": None,
            "pre_earnings_5d_return": None,
            "computed_positives": 0, "computed_vetoes": 0,
        }


class ProgressTracker:
    def __init__(self, total: int, report_interval: int = 50):
        self.total = total
        self.report_interval = report_interval
        self.completed = 0
        self.results: List[Dict] = []
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def add_result(self, result: Dict) -> int:
        async with self.lock:
            self.results.append(result)
            self.completed += 1
            current = self.completed

            if current % self.report_interval == 0 or current == self.total:
                self._print_progress_report()

            return current

    def _print_progress_report(self):
        elapsed = time.time() - self.start_time
        successful = [r for r in self.results if r.get("success")]
        failed = len(self.results) - len(successful)

        with_prediction = [r for r in successful if r.get("correct") is not None]
        overall_correct = sum(1 for r in with_prediction if r.get("correct"))
        overall_acc = (overall_correct / len(with_prediction) * 100) if with_prediction else 0

        long_trades = [r for r in successful if r.get("trade_long")]
        long_correct = [r for r in long_trades if r.get("correct") is not None]
        long_wins = sum(1 for r in long_correct if r.get("correct"))
        long_win_rate = (long_wins / len(long_correct) * 100) if long_correct else 0
        long_avg_ret = (sum(r.get("actual_return_30d_pct", 0) or 0 for r in long_trades) / len(long_trades)) if long_trades else 0

        total_cost = sum(r.get("cost_usd", 0) or 0 for r in self.results)
        avg_time = elapsed / self.completed if self.completed else 0
        eta_min = (avg_time * (self.total - self.completed)) / 60

        print("\n" + "=" * 70)
        print(f"📊 PROGRESS [{self.completed}/{self.total}] ({self.completed/self.total*100:.1f}%)")
        print("=" * 70)
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {eta_min:.1f} min")
        print(f"✅ Success: {len(successful)} | ❌ Failed: {failed}")
        print(f"🎯 Overall Accuracy: {overall_acc:.1f}% ({overall_correct}/{len(with_prediction)})")
        print(f"📈 Long-only: {len(long_trades)} trades | Win Rate: {long_win_rate:.1f}% | Avg Ret: {long_avg_ret:+.1f}%")
        print(f"💰 Cost: ${total_cost:.4f}")

        # Quarter breakdown
        for q in [1, 2, 3, 4]:
            q_results = [r for r in successful if r.get("quarter") == q]
            q_long = [r for r in q_results if r.get("trade_long")]
            if q_results:
                q_acc = sum(1 for r in q_results if r.get("correct")) / len([r for r in q_results if r.get("correct") is not None]) * 100 if [r for r in q_results if r.get("correct") is not None] else 0
                print(f"  Q{q}: {len(q_results)} samples | {q_acc:.0f}% acc | {len(q_long)} long")
        print("=" * 70 + "\n")


async def run_single_test_with_semaphore(semaphore, idx, total, sample, tracker):
    async with semaphore:
        symbol = sample["symbol"]
        year = sample["year"]
        quarter = sample["quarter"]
        category = sample["category"]

        print(f"[START {idx}/{total}] {symbol} {year}Q{quarter} ({category})")

        result = await run_single_test(
            symbol=symbol, year=year, quarter=quarter, category=category,
            company_name=sample.get("company_name", ""),
            sector=sample.get("sector", ""),
            actual_return_30d=sample.get("actual_return_30d"),
        )

        done = await tracker.add_result(result)

        if result["success"]:
            pred = result['prediction']
            direction = result['direction_score']
            trade_long = result['trade_long']
            ret_pct = result['actual_return_30d_pct']
            ret_str = f"{ret_pct:.1f}%" if ret_pct else "N/A"
            correct = result['correct']
            print(f"[DONE {done}/{total}] {symbol}: {pred}(D:{direction}) TL:{trade_long} Ret:{ret_str} Correct:{correct}")
        else:
            err_msg = result['error'][:50] if result.get('error') else 'Unknown'
            print(f"[DONE {done}/{total}] {symbol}: ERROR - {err_msg}")

        return result


def print_final_summary(results: List[Dict], start_time: float):
    """Print comprehensive final summary."""
    print("\n" + "=" * 70)
    print("🏁 2025 FULL YEAR HOLDOUT - FINAL SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    successful = df[df["success"] == True]
    failed = df[df["success"] == False]

    print(f"\n📊 Execution Stats:")
    print(f"  Total Samples: {len(df)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")

    if len(successful) == 0:
        return

    # Overall accuracy
    with_prediction = successful[successful["correct"].notna()]
    overall_accuracy = 0
    if len(with_prediction) > 0:
        overall_accuracy = with_prediction["correct"].mean() * 100
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.1f}% ({int(with_prediction['correct'].sum())}/{len(with_prediction)})")

    # Long-only stats
    trade_long_df = successful[successful["trade_long"] == True]
    long_accuracy = 0
    avg_return = 0
    if len(trade_long_df) > 0:
        trade_long_correct = trade_long_df[trade_long_df["correct"].notna()]
        if len(trade_long_correct) > 0:
            long_accuracy = trade_long_correct["correct"].mean() * 100
            long_wins = int(trade_long_correct['correct'].sum())
            long_losses = len(trade_long_correct) - long_wins
            avg_return = trade_long_df["actual_return_30d_pct"].mean()

            print(f"\n📈 Long-only Strategy:")
            print(f"  Coverage: {len(trade_long_df)}/{len(successful)} ({len(trade_long_df)/len(successful)*100:.1f}%)")
            print(f"  Win Rate: {long_accuracy:.1f}% ({long_wins}W / {long_losses}L)")
            print(f"  Avg Return: {avg_return:+.2f}%")

            winners = trade_long_df[trade_long_df["correct"] == True]
            losers = trade_long_df[trade_long_df["correct"] == False]
            if len(winners) > 0:
                print(f"  Winners Avg: +{winners['actual_return_30d_pct'].mean():.2f}%")
            if len(losers) > 0:
                print(f"  Losers Avg: {losers['actual_return_30d_pct'].mean():.2f}%")

            # Tier breakdown
            print(f"\n📊 Tier Breakdown:")
            for tier in ["D7_CORE", "D6_STRICT"]:
                tier_df = trade_long_df[trade_long_df["trade_long_tier"] == tier]
                if len(tier_df) > 0:
                    tier_correct = tier_df[tier_df["correct"].notna()]
                    tier_win = tier_correct["correct"].mean() * 100 if len(tier_correct) > 0 else 0
                    tier_ret = tier_df["actual_return_30d_pct"].mean()
                    print(f"  {tier}: {len(tier_df)} trades | {tier_win:.0f}% win | {tier_ret:+.1f}% avg")
    else:
        print(f"\n📈 No samples qualified for Long-only")

    # Quarter breakdown
    print(f"\n📅 Performance by Quarter:")
    print(f"  {'Quarter':<8} {'Samples':<10} {'Accuracy':<12} {'Long Trades':<12} {'Long Win%':<12} {'Long Ret':<10}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for q in sorted(successful["quarter"].unique()):
        q_df = successful[successful["quarter"] == q]
        q_correct = q_df[q_df["correct"].notna()]
        q_acc = q_correct["correct"].mean() * 100 if len(q_correct) > 0 else 0

        q_long = q_df[q_df["trade_long"] == True]
        q_long_correct = q_long[q_long["correct"].notna()]
        q_long_win = q_long_correct["correct"].mean() * 100 if len(q_long_correct) > 0 else 0
        q_long_ret = q_long["actual_return_30d_pct"].mean() if len(q_long) > 0 else 0

        print(f"  Q{int(q):<7} {len(q_df):<10} {q_acc:<12.1f}% {len(q_long):<12} {q_long_win:<12.0f}% {q_long_ret:<+10.1f}%")

    # H1 vs H2 comparison
    print(f"\n📊 H1 vs H2 Comparison:")
    for half, qs in [("H1 (Q1-Q2)", [1, 2]), ("H2 (Q3-Q4)", [3, 4])]:
        h_df = successful[successful["quarter"].isin(qs)]
        if len(h_df) > 0:
            h_correct = h_df[h_df["correct"].notna()]
            h_acc = h_correct["correct"].mean() * 100 if len(h_correct) > 0 else 0
            h_long = h_df[h_df["trade_long"] == True]
            h_long_correct = h_long[h_long["correct"].notna()]
            h_long_win = h_long_correct["correct"].mean() * 100 if len(h_long_correct) > 0 else 0
            h_long_ret = h_long["actual_return_30d_pct"].mean() if len(h_long) > 0 else 0
            print(f"  {half}: {len(h_df)} samples | {h_acc:.1f}% acc | {len(h_long)} long | {h_long_win:.0f}% win | {h_long_ret:+.1f}% ret")

    # Direction score distribution
    print(f"\n📊 Direction Score Distribution:")
    direction_scores = successful["direction_score"].dropna()
    if len(direction_scores) > 0:
        for score in sorted(direction_scores.unique()):
            count = (direction_scores == score).sum()
            pct = count / len(direction_scores) * 100
            print(f"  Score {int(score)}: {count} ({pct:.1f}%)")

    # Sector breakdown for long trades
    if len(trade_long_df) > 0:
        print(f"\n🏭 Long-only by Sector (top 10):")
        for sector in trade_long_df["sector"].value_counts().head(10).index:
            s_df = trade_long_df[trade_long_df["sector"] == sector]
            s_correct = s_df[s_df["correct"].notna()]
            s_win = s_correct["correct"].mean() * 100 if len(s_correct) > 0 else 0
            s_ret = s_df["actual_return_30d_pct"].mean()
            print(f"  {sector}: {len(s_df)} trades | {s_win:.0f}% win | {s_ret:+.1f}%")

    # Cost summary
    total_time = time.time() - start_time
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)

    print("\n" + "-" * 40)
    print("💰 EXECUTION STATS")
    print("-" * 40)
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Avg Cost/Sample: ${total_cost/len(results):.4f}")

    # Comparison with training set
    print("\n" + "=" * 70)
    print("📊 COMPARISON: TRAINING (2017-2024) vs HOLDOUT (2025)")
    print("=" * 70)
    print("  Training (12,003 samples):")
    print("    Overall Acc: 54.6% | Long: 995 trades | Win Rate: 81.0% | Avg Ret: +8.39%")
    print(f"  Holdout 2025 ({len(successful)} samples):")
    print(f"    Overall Acc: {overall_accuracy:.1f}% | Long: {len(trade_long_df)} trades | Win Rate: {long_accuracy:.0f}% | Avg Ret: {avg_return:+.1f}%")
    print("=" * 70)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="2025 Full Year Holdout Validation")
    parser.add_argument("--samples", type=int, default=1000, help="Total samples (default: 1000)")
    parser.add_argument("--test", action="store_true", help="Quick test with 40 samples")
    parser.add_argument("--check", action="store_true", help="Only check data availability")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel workers (default: 10)")
    parser.add_argument("--gainer-threshold", type=float, default=5.0, help="GAINER threshold %")
    parser.add_argument("--loser-threshold", type=float, default=-5.0, help="LOSER threshold %")
    parser.add_argument("--report-interval", type=int, default=50, help="Progress report interval")
    args = parser.parse_args()

    if args.check:
        check_data_availability()
        return

    NUM_SAMPLES = 40 if args.test else args.samples
    PARALLEL = args.parallel

    total_start_time = time.time()

    print("=" * 70)
    print("🔬 2025 FULL YEAR HOLDOUT VALIDATION")
    print("=" * 70)
    print(f"📅 Period: 2025 Q1-Q4 (Full Year)")
    print(f"📊 Target Samples: {NUM_SAMPLES}")
    print(f"⚡ Parallel Workers: {PARALLEL}")
    print(f"📈 GAINER threshold: >{args.gainer_threshold}%")
    print(f"📉 LOSER threshold: <{args.loser_threshold}%")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    check_and_switch_provider()

    # Get samples
    all_samples = get_2025_samples(
        limit_per_category=NUM_SAMPLES // 2,
        gainer_threshold=args.gainer_threshold,
        loser_threshold=args.loser_threshold,
    )
    random.shuffle(all_samples)
    samples = all_samples[:NUM_SAMPLES]

    print(f"\n📦 Loaded {len(samples)} samples from 2025")
    if samples:
        df_samples = pd.DataFrame(samples)
        print(f"  Quarter distribution: {df_samples['quarter'].value_counts().sort_index().to_dict()}")
        print(f"  Category distribution: {df_samples['category'].value_counts().to_dict()}")
        print(f"  Sector distribution (top 5): {df_samples['sector'].value_counts().head().to_dict()}")

    if not samples:
        print("❌ No samples available. Run with --check to see data availability.")
        return

    # Run tests
    semaphore = asyncio.Semaphore(PARALLEL)
    tracker = ProgressTracker(total=len(samples), report_interval=args.report_interval)

    print(f"\n🚀 Running {len(samples)} tests with {PARALLEL} parallel workers...\n")

    tasks = [
        run_single_test_with_semaphore(semaphore, i, len(samples), sample, tracker)
        for i, sample in enumerate(samples, 1)
    ]

    results = await asyncio.gather(*tasks)

    # Print final summary
    print_final_summary(results, total_start_time)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"holdout_2025_full_{timestamp}.csv"

    for r in results:
        if r.get("long_eligible_json"):
            for k, v in r["long_eligible_json"].items():
                r[f"le_{k}"] = v
        if "long_eligible_json" in r:
            del r["long_eligible_json"]

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
