#!/usr/bin/env python3
"""
Rerun failed samples from a previous test run.
Reads existing CSV, reruns only failed samples, and merges results.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_engine import analyze_earnings_async


async def run_single_test(sample: Dict) -> Dict:
    """Run a single test and return results."""
    symbol = sample["symbol"]
    year = int(sample["year"])
    quarter = int(sample["quarter"])
    category = sample.get("category", "UNKNOWN")
    company_name = sample.get("company_name", "")
    sector = sample.get("sector", "")
    actual_return_30d = sample.get("actual_return_30d_pct")

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

        # Determine correctness
        correct = None
        if actual_return is not None and prediction in ["UP", "DOWN"]:
            if prediction == "UP":
                correct = actual_return > 0
            else:
                correct = actual_return < 0

        # Extract direction score from long_eligible_json
        direction_score = None
        if long_eligible_json:
            direction_score = long_eligible_json.get("DirectionScore")

        # Extract token usage from result
        token_usage = result.get("agentic_result", {}).get("raw", {}).get("token_usage", {})
        input_tokens = token_usage.get("input_tokens", 0) if token_usage else 0
        output_tokens = token_usage.get("output_tokens", 0) if token_usage else 0
        cost_usd = token_usage.get("cost_usd", 0.0) if token_usage else 0.0

        # Extract new fields for offline grid search
        risk_code = agentic_result.get("risk_code", "unknown")
        long_block_reason = agentic_result.get("long_block_reason", "")
        market_anchors = agentic_result.get("market_anchors", {}) or {}
        computed_positives = agentic_result.get("computed_positives", 0)
        computed_vetoes = agentic_result.get("computed_vetoes", 0)

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "category": category,
            "company_name": company_name,
            "sector": sector,
            "success": True,
            "error": None,
            "time_seconds": elapsed,
            "prediction": prediction,
            "confidence": confidence,
            "direction_score": direction_score,
            "actual_return_30d_pct": actual_return,
            "correct": correct,
            "trade_long": trade_long,
            "long_eligible_json": long_eligible_json,
            "summary": summary[:500] if summary else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "risk_code": risk_code,
            "long_block_reason": long_block_reason,
            "eps_surprise": market_anchors.get("eps_surprise"),
            "earnings_day_return": market_anchors.get("earnings_day_return"),
            "pre_earnings_5d_return": market_anchors.get("pre_earnings_5d_return"),
            "computed_positives": computed_positives,
            "computed_vetoes": computed_vetoes,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "category": category,
            "company_name": company_name,
            "sector": sector,
            "success": False,
            "error": str(e),
            "time_seconds": elapsed,
            "prediction": None,
            "confidence": None,
            "direction_score": None,
            "actual_return_30d_pct": actual_return_30d,
            "correct": None,
            "trade_long": False,
            "long_eligible_json": None,
            "summary": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "risk_code": "unknown",
            "long_block_reason": "EXCEPTION",
            "eps_surprise": None,
            "earnings_day_return": None,
            "pre_earnings_5d_return": None,
            "computed_positives": 0,
            "computed_vetoes": 0,
        }


async def run_single_test_with_semaphore(
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
    sample: Dict,
) -> Dict:
    """Run a single test with semaphore for concurrency control."""
    async with semaphore:
        symbol = sample["symbol"]
        year = int(sample["year"])
        quarter = int(sample["quarter"])

        print(f"[START {idx}/{total}] {symbol} {year}Q{quarter}")

        result = await run_single_test(sample)

        # Print result
        if result["success"]:
            pred = result['prediction']
            direction = result['direction_score']
            trade_long = result['trade_long']
            ret_pct = result['actual_return_30d_pct']
            ret_str = f"{ret_pct:.1f}%" if ret_pct else "N/A"
            correct = result['correct']
            print(f"[DONE {idx}/{total}] {symbol}: {pred}(D:{direction}) TL:{trade_long} Ret:{ret_str} Correct:{correct}")
        else:
            print(f"[DONE {idx}/{total}] {symbol}: ERROR - {result['error'][:50]}")

        return result


async def main():
    """Main function to rerun failed samples."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to original CSV file")
    parser.add_argument("--parallel", type=int, default=3, help="Number of parallel workers (default: 3)")
    args = parser.parse_args()

    csv_path = args.csv
    parallel = args.parallel

    print("=" * 70)
    print(f"Rerunning Failed Samples")
    print(f"Source CSV: {csv_path}")
    print(f"Parallel Workers: {parallel}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load original CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} total samples")

    # Filter failed samples
    failed_df = df[df["success"] == False].copy()
    success_df = df[df["success"] == True].copy()

    print(f"Success: {len(success_df)}, Failed: {len(failed_df)}")

    if len(failed_df) == 0:
        print("No failed samples to rerun!")
        return

    # Convert to list of dicts for processing
    failed_samples = failed_df.to_dict("records")

    print(f"\nRerunning {len(failed_samples)} failed samples with {parallel} parallel workers...")

    # Run tests in parallel with semaphore
    semaphore = asyncio.Semaphore(parallel)

    tasks = [
        run_single_test_with_semaphore(
            semaphore=semaphore,
            idx=i,
            total=len(failed_samples),
            sample=sample,
        )
        for i, sample in enumerate(failed_samples, 1)
    ]

    rerun_results = await asyncio.gather(*tasks)

    # Process rerun results
    rerun_df = pd.DataFrame(rerun_results)

    # Flatten long_eligible_json for CSV
    for r in rerun_results:
        if r.get("long_eligible_json"):
            for k, v in r["long_eligible_json"].items():
                r[f"le_{k}"] = v
        if "long_eligible_json" in r:
            del r["long_eligible_json"]

    rerun_df = pd.DataFrame(rerun_results)

    # Summary of rerun
    rerun_success = len(rerun_df[rerun_df["success"] == True])
    rerun_failed = len(rerun_df[rerun_df["success"] == False])

    print("\n" + "=" * 70)
    print("RERUN SUMMARY")
    print("=" * 70)
    print(f"Rerun Success: {rerun_success}/{len(failed_samples)}")
    print(f"Still Failed: {rerun_failed}/{len(failed_samples)}")

    # Merge with success_df
    # First flatten success_df long_eligible_json if present
    if "long_eligible_json" in success_df.columns:
        success_df = success_df.drop(columns=["long_eligible_json"])

    # Combine
    merged_df = pd.concat([success_df, rerun_df], ignore_index=True)

    # Save merged CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = csv_path.replace(".csv", f"_merged_{timestamp}.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged results saved to: {output_path}")

    # Final stats
    final_success = len(merged_df[merged_df["success"] == True])
    print(f"Final success rate: {final_success}/{len(merged_df)} ({final_success/len(merged_df)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
