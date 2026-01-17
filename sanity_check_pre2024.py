#!/usr/bin/env python3
"""
Sanity Check - 10 samples from 2017-2023 (pre-2024)
Verify results are consistent with historical performance.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_engine import analyze_earnings_async
from pg_client import get_cursor


def get_pre2024_samples(n_samples: int = 10):
    """Get samples from 2017-2023 for sanity check."""
    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        # Get 5 GAINERS and 5 LOSERS from 2017-2023
        samples = []

        # GAINERS
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
            WHERE et.year BETWEEN 2017 AND 2023
                AND tc.content IS NOT NULL
                AND LENGTH(tc.content) > 1000
                AND pa.pct_change_t_plus_30 IS NOT NULL
                AND pa.pct_change_t_plus_30 > 10
            ORDER BY RANDOM()
            LIMIT %s
        """, (n_samples // 2,))

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

        # LOSERS
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
            WHERE et.year BETWEEN 2017 AND 2023
                AND tc.content IS NOT NULL
                AND LENGTH(tc.content) > 1000
                AND pa.pct_change_t_plus_30 IS NOT NULL
                AND pa.pct_change_t_plus_30 < -10
            ORDER BY RANDOM()
            LIMIT %s
        """, (n_samples - n_samples // 2,))

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
    """Run a single test."""
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
        trade_long = agentic_result.get("trade_long", False)
        trade_long_tier = agentic_result.get("trade_long_tier", "")
        long_eligible_json = agentic_result.get("long_eligible_json", {})
        direction_score = long_eligible_json.get("DirectionScore") if long_eligible_json else None

        actual_return = actual_return_30d
        correct = None
        if actual_return is not None and prediction in ["UP", "DOWN"]:
            correct = (actual_return > 0) if prediction == "UP" else (actual_return < 0)

        token_usage = result.get("agentic_result", {}).get("raw", {}).get("token_usage", {})
        input_tokens = token_usage.get("input_tokens", 0) if token_usage else 0
        output_tokens = token_usage.get("output_tokens", 0) if token_usage else 0
        cost_usd = token_usage.get("cost_usd", 0.0) if token_usage else 0.0

        return {
            "symbol": symbol, "year": year, "quarter": quarter, "category": category,
            "success": True, "error": None, "time_seconds": elapsed,
            "prediction": prediction, "confidence": confidence,
            "direction_score": direction_score,
            "actual_return_30d_pct": actual_return, "correct": correct,
            "trade_long": trade_long, "trade_long_tier": trade_long_tier,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "symbol": symbol, "year": year, "quarter": quarter, "category": category,
            "success": False, "error": str(e), "time_seconds": elapsed,
            "prediction": None, "confidence": None, "direction_score": None,
            "actual_return_30d_pct": actual_return_30d, "correct": None,
            "trade_long": False, "trade_long_tier": "",
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        }


async def main():
    print("=" * 70)
    print("🔬 SANITY CHECK - PRE-2024 DATA (2017-2023)")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check database connection
    print("\n📡 Testing database connection...")
    try:
        samples = get_pre2024_samples(10)
        print(f"✅ Database connected. Got {len(samples)} samples.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    print("\n📦 Sample Distribution:")
    df_samples = pd.DataFrame(samples)
    print(f"  Category: {df_samples['category'].value_counts().to_dict()}")
    print(f"  Years: {sorted(df_samples['year'].unique())}")

    print("\n🚀 Running tests sequentially...")
    print("-" * 70)

    results = []
    total_start = time.time()

    for i, sample in enumerate(samples, 1):
        symbol = sample["symbol"]
        year = sample["year"]
        quarter = sample["quarter"]
        category = sample["category"]
        actual_ret = sample["actual_return_30d"]

        print(f"\n[{i}/10] {symbol} {year}Q{quarter} ({category}) Actual: {actual_ret:+.1f}%")

        result = await run_single_test(
            symbol=symbol, year=year, quarter=quarter, category=category,
            company_name=sample.get("company_name", ""),
            sector=sample.get("sector", ""),
            actual_return_30d=actual_ret,
        )
        results.append(result)

        if result["success"]:
            pred = result["prediction"]
            direction = result["direction_score"]
            trade_long = result["trade_long"]
            tier = result["trade_long_tier"]
            correct = result["correct"]
            cost = result["cost_usd"]
            time_s = result["time_seconds"]

            status = "✅" if correct else "❌" if correct is False else "❓"
            print(f"  {status} Pred: {pred} | Dir: {direction} | TradeLong: {trade_long} ({tier})")
            print(f"     Correct: {correct} | Time: {time_s:.1f}s | Cost: ${cost:.4f}")
        else:
            print(f"  ❌ ERROR: {result['error'][:80]}")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("📊 SANITY CHECK SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    successful = df[df["success"] == True]
    failed = df[df["success"] == False]

    print(f"\n📈 Execution Stats:")
    print(f"  Total: {len(df)} | Success: {len(successful)} | Failed: {len(failed)}")
    print(f"  Total Time: {total_time:.1f}s ({total_time/len(df):.1f}s per sample)")
    print(f"  Total Cost: ${df['cost_usd'].sum():.4f}")

    if len(successful) > 0:
        with_prediction = successful[successful["correct"].notna()]
        if len(with_prediction) > 0:
            accuracy = with_prediction["correct"].mean() * 100
            print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({int(with_prediction['correct'].sum())}/{len(with_prediction)})")

        trade_long_df = successful[successful["trade_long"] == True]
        if len(trade_long_df) > 0:
            long_correct = trade_long_df[trade_long_df["correct"].notna()]
            if len(long_correct) > 0:
                long_win_rate = long_correct["correct"].mean() * 100
                avg_return = trade_long_df["actual_return_30d_pct"].mean()
                print(f"\n📈 Long-only:")
                print(f"  Trades: {len(trade_long_df)}")
                print(f"  Win Rate: {long_win_rate:.1f}%")
                print(f"  Avg Return: {avg_return:+.1f}%")

                # Tier breakdown
                for tier in ["D7_CORE", "D6_STRICT"]:
                    tier_df = trade_long_df[trade_long_df["trade_long_tier"] == tier]
                    if len(tier_df) > 0:
                        tier_correct = tier_df[tier_df["correct"].notna()]
                        tier_win = tier_correct["correct"].mean() * 100 if len(tier_correct) > 0 else 0
                        print(f"    {tier}: {len(tier_df)} trades | {tier_win:.0f}% win")
        else:
            print(f"\n📈 Long-only: 0 trades qualified")

    # Comparison with expected
    print("\n" + "=" * 70)
    print("📊 COMPARISON WITH TRAINING SET (2017-2024, 12,003 samples)")
    print("=" * 70)
    print("  Training Set Benchmarks:")
    print("    Overall Accuracy: 54.6%")
    print("    Long Win Rate: 81.0%")
    print("    Long Avg Return: +8.39%")
    print("\n  Note: 10 samples is too small for statistical comparison.")
    print("  This sanity check verifies the pipeline works correctly.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
