#!/usr/bin/env python3
"""
Run LLM analysis on expanded manifest samples.
Supports resumable execution and progress monitoring.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

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
    """Check if LiteLLM is available, switch to Azure if not."""
    if not HAS_PROVIDER_UTILS:
        print("Provider utils not available, using default")
        return

    print("Checking LiteLLM availability...")
    try:
        client, model = build_litellm_client("gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"LiteLLM available (provider: {get_current_provider()})")
    except Exception as e:
        error_str = str(e).lower()
        if "503" in error_str or "502" in error_str or "401" in error_str or "connection" in error_str:
            print(f"LiteLLM unavailable, switching to Azure...")
            switch_to_azure_fallback()
            print(f"Now using: {get_current_provider()}")


def get_reaction_date_from_db(transcript_id: int) -> str:
    """Get reaction date (t_day) from database."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT t_day FROM earnings_transcripts WHERE id = %s
        """, (transcript_id,))
        row = cur.fetchone()
        if row and row['t_day']:
            return str(row['t_day'])
    return None


async def run_single_analysis(sample: Dict) -> Dict:
    """Run LLM analysis on a single sample."""
    start = time.time()
    symbol = sample['symbol']
    year = int(sample['year'])
    quarter = int(sample['quarter'])

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,
        )
        elapsed = time.time() - start

        agentic_result = result.get("agentic_result", {})
        long_eligible_json = agentic_result.get("long_eligible_json", {})
        market_anchors = agentic_result.get("market_anchors", {}) or {}
        token_usage = agentic_result.get("raw", {}).get("token_usage", {}) or {}

        # Get reaction_date from DB
        reaction_date = get_reaction_date_from_db(sample.get('transcript_id'))

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "category": "GAINER" if sample.get('return_30d', 0) > 10 else ("LOSER" if sample.get('return_30d', 0) < -10 else "NEUTRAL"),
            "company_name": sample.get('company_name', ''),
            "sector": sample.get('sector', ''),
            "success": True,
            "error": None,
            "time_seconds": elapsed,
            "prediction": agentic_result.get("prediction", "UNKNOWN"),
            "confidence": agentic_result.get("confidence"),
            "direction_score": long_eligible_json.get("DirectionScore") if long_eligible_json else None,
            "actual_return_30d_pct": sample.get('return_30d'),
            "correct": None,  # Will be computed later
            "trade_long": agentic_result.get("trade_long", False),
            "trade_long_tier": agentic_result.get("trade_long_tier", ""),
            "summary": (agentic_result.get("summary", "") or "")[:500],
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "cost_usd": token_usage.get("cost_usd", 0.0),
            "risk_code": agentic_result.get("risk_code", "unknown"),
            "long_block_reason": agentic_result.get("long_block_reason", ""),
            "eps_surprise": market_anchors.get("eps_surprise"),
            "earnings_day_return": market_anchors.get("earnings_day_return"),
            "pre_earnings_5d_return": market_anchors.get("pre_earnings_5d_return"),
            "computed_positives": agentic_result.get("computed_positives", 0),
            "computed_vetoes": agentic_result.get("computed_vetoes", 0),
            "reaction_date": reaction_date,
            # LongEligible JSON fields
            "le_DirectionScore": long_eligible_json.get("DirectionScore") if long_eligible_json else None,
            "le_LongEligible": long_eligible_json.get("LongEligible") if long_eligible_json else None,
            "le_HardPositivesCount": long_eligible_json.get("HardPositivesCount") if long_eligible_json else None,
            "le_HardVetoCount": long_eligible_json.get("HardVetoCount") if long_eligible_json else None,
            "le_GuidanceRaised": long_eligible_json.get("GuidanceRaised") if long_eligible_json else None,
            "le_DemandAcceleration": long_eligible_json.get("DemandAcceleration") if long_eligible_json else None,
            "le_MarginExpansion": long_eligible_json.get("MarginExpansion") if long_eligible_json else None,
            "le_FCFImprovement": long_eligible_json.get("FCFImprovement") if long_eligible_json else None,
            "le_VisibilityImproving": long_eligible_json.get("VisibilityImproving") if long_eligible_json else None,
            "le_GuidanceCut": long_eligible_json.get("GuidanceCut") if long_eligible_json else None,
            "le_DemandSoftness": long_eligible_json.get("DemandSoftness") if long_eligible_json else None,
            "le_MarginWeakness": long_eligible_json.get("MarginWeakness") if long_eligible_json else None,
            "le_CashBurn": long_eligible_json.get("CashBurn") if long_eligible_json else None,
            "le_VisibilityWorsening": long_eligible_json.get("VisibilityWorsening") if long_eligible_json else None,
            "le_PricedInRisk": long_eligible_json.get("PricedInRisk") if long_eligible_json else None,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "category": "UNKNOWN",
            "company_name": sample.get('company_name', ''),
            "sector": sample.get('sector', ''),
            "success": False,
            "error": str(e)[:200],
            "time_seconds": elapsed,
            "prediction": None,
            "confidence": None,
            "direction_score": None,
            "actual_return_30d_pct": sample.get('return_30d'),
            "correct": None,
            "trade_long": False,
            "trade_long_tier": "",
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
            "reaction_date": None,
        }


async def run_batch_with_semaphore(
    samples: List[Dict],
    semaphore: asyncio.Semaphore,
    results: List[Dict],
    progress_callback=None,
):
    """Run batch of samples with concurrency control."""

    async def process_one(idx: int, sample: Dict):
        async with semaphore:
            result = await run_single_analysis(sample)
            results.append(result)
            if progress_callback:
                progress_callback(idx, len(samples), result)
            return result

    tasks = [process_one(i, s) for i, s in enumerate(samples, 1)]
    await asyncio.gather(*tasks)


def print_progress(idx: int, total: int, result: Dict):
    """Print progress update."""
    symbol = result['symbol']
    success = result['success']
    trade_long = result.get('trade_long', False)
    direction = result.get('direction_score', '?')
    cost = result.get('cost_usd', 0)

    status = "OK" if success else "ERR"
    tl = "TL" if trade_long else "--"
    print(f"[{idx:4d}/{total}] {symbol:6s} D:{direction} {tl} ${cost:.4f} {status}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--output", type=str, default="expanded_signals.csv", help="Output CSV path")
    parser.add_argument("--existing", type=str, default=None, help="Existing signals CSV to skip")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel workers")
    parser.add_argument("--report-interval", type=int, default=50, help="Progress report interval")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Expanded Manifest LLM Analysis")
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check provider
    check_and_switch_provider()

    # Load manifest
    manifest = pd.read_csv(args.manifest)
    manifest['key'] = manifest['symbol'] + '_' + manifest['year'].astype(str) + '_' + manifest['quarter'].astype(str)
    print(f"\nManifest loaded: {len(manifest)} samples")

    # Load existing if provided
    existing_keys: Set[str] = set()
    existing_results: List[Dict] = []

    if args.existing and Path(args.existing).exists():
        existing = pd.read_csv(args.existing)
        existing_keys = set(existing['symbol'] + '_' + existing['year'].astype(str) + '_' + existing['quarter'].astype(str))
        existing_results = existing.to_dict('records')
        print(f"Existing signals loaded: {len(existing_keys)} already analyzed")

    # Filter to new samples
    new_samples = manifest[~manifest['key'].isin(existing_keys)].to_dict('records')
    print(f"New samples to analyze: {len(new_samples)}")

    if args.limit:
        new_samples = new_samples[:args.limit]
        print(f"Limited to: {len(new_samples)} samples")

    if len(new_samples) == 0:
        print("No new samples to analyze!")
        return

    # Run analysis
    semaphore = asyncio.Semaphore(args.parallel)
    results: List[Dict] = []
    start_time = time.time()

    last_report = [0]
    total_cost = [0.0]

    def progress_callback(idx: int, total: int, result: Dict):
        total_cost[0] += result.get('cost_usd', 0)

        if idx % args.report_interval == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate if rate > 0 else 0

            success_count = sum(1 for r in results if r.get('success'))
            trade_long_count = sum(1 for r in results if r.get('trade_long'))

            print("\n" + "=" * 50)
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
            print(f"Success: {success_count} | trade_long: {trade_long_count}")
            print(f"Total Cost: ${total_cost[0]:.4f}")
            print("=" * 50 + "\n")

    print(f"\nStarting analysis with {args.parallel} parallel workers...")
    await run_batch_with_semaphore(new_samples, semaphore, results, progress_callback)

    # Combine with existing
    all_results = existing_results + results

    # Compute correctness
    for r in all_results:
        actual = r.get('actual_return_30d_pct')
        pred = r.get('prediction')
        if actual is not None and pred in ['UP', 'DOWN']:
            if pred == 'UP':
                r['correct'] = actual > 0
            else:
                r['correct'] = actual < 0

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {len(results)}")
    print(f"Total time: {elapsed/60:.1f} min")
    print(f"Total cost: ${total_cost[0]:.4f}")
    print(f"Avg cost/sample: ${total_cost[0]/len(results):.4f}" if results else "N/A")

    successful = [r for r in results if r.get('success')]
    trade_long_new = [r for r in results if r.get('trade_long')]
    print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)" if results else "N/A")
    print(f"New trade_long signals: {len(trade_long_new)}")

    print(f"\nOutput saved to: {args.output}")
    print(f"Total signals in output: {len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())
