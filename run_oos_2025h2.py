#!/usr/bin/env python3
"""
2025H2 Out-of-Sample Validation Script

Purpose: Test strategy accuracy on 2025Q3 + 2025Q4 data (holdout period)
Target: ~330 trade_long signals from ~4,000 universe samples

Key differences from in-sample validation:
- Only uses 2025Q3 and 2025Q4 data
- Stratified sampling: equal samples per quarter
- NO future data used in sampling (only uses transcript_date and T+30 return availability)
- Comprehensive drift detection reporting

Usage:
    # Test run (10 samples)
    python run_oos_2025h2.py --test-only

    # Full run (~4000 samples, 2000/quarter)
    python run_oos_2025h2.py --per-quarter 2000 --parallel 10

    # Smaller run for quick validation (~1000 samples)
    python run_oos_2025h2.py --per-quarter 500 --parallel 10
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CRITICAL: Lookahead Protection Enforcement
# =============================================================================

def enforce_lookahead_protection():
    """Enforce lookahead protection settings."""
    if os.environ.get("DISABLE_LEAKAGE_CHECK", "").lower() == "true":
        raise RuntimeError("FATAL: DISABLE_LEAKAGE_CHECK=true is not allowed for OOS validation")

    os.environ["LOOKAHEAD_ASSERTIONS"] = "true"
    os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"

    if not os.environ.get("CALL_CACHE_VERSION"):
        os.environ["CALL_CACHE_VERSION"] = "v2.2"

enforce_lookahead_protection()

# Add EarningsCallAgenticRag to path
script_dir = Path(__file__).parent
rag_dir = script_dir / "EarningsCallAgenticRag"
if rag_dir.exists():
    sys.path.insert(0, str(rag_dir))

from pg_client import get_cursor

# =============================================================================
# Configuration - 2025H2 Specific
# =============================================================================

OOS_QUARTERS = [(2025, 3), (2025, 4)]  # 2025Q3 and 2025Q4 only
DEFAULT_PER_QUARTER = 2000  # Target ~4000 total samples
DEFAULT_CONCURRENCY = 10
DEFAULT_SEED = 42
TEST_SAMPLES = 10

# =============================================================================
# Utility Functions
# =============================================================================

def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    from scipy import stats
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * ((p * (1-p) + z**2 / (4*n)) / n) ** 0.5 / denominator
    return max(0, center - margin), min(1, center + margin), (min(1, center + margin) - max(0, center - margin)) / 2


def deterministic_hash(symbol: str, year: int, quarter: int, seed: int) -> str:
    key = f"{symbol}:{year}:Q{quarter}:seed{seed}"
    return hashlib.md5(key.encode()).hexdigest()


# =============================================================================
# Sample Selection - 2025H2 Only
# =============================================================================

def get_2025h2_samples(per_quarter: int = 2000, seed: int = 42) -> List[Dict]:
    """Get stratified samples from 2025Q3 and 2025Q4 only."""
    print(f"\n{'='*60}")
    print(f"2025H2 OOS SAMPLING")
    print(f"{'='*60}")
    print(f"Quarters: {OOS_QUARTERS}")
    print(f"Per quarter: {per_quarter}")
    print(f"Seed: {seed}")
    print(f"Expected total: {per_quarter * len(OOS_QUARTERS)}")
    print(f"{'='*60}\n")

    all_samples = []
    quarter_stats = {}

    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        for year, quarter in OOS_QUARTERS:
            quarter_key = f"{year}Q{quarter}"

            # Fetch more than needed, then select deterministically
            fetch_limit = per_quarter * 3

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
                    "_hash": deterministic_hash(row["symbol"], year, quarter, seed)
                }
                quarter_samples.append(sample)

            # Deterministic selection
            quarter_samples.sort(key=lambda x: x["_hash"])
            selected = quarter_samples[:per_quarter]

            for s in selected:
                del s["_hash"]

            all_samples.extend(selected)

            # Stats for reporting
            quarter_stats[quarter_key] = {
                "available": len(quarter_samples),
                "selected": len(selected),
                "sectors": len(set(s.get("sector") for s in selected if s.get("sector"))),
            }

            print(f"{quarter_key}: {len(selected)} samples selected (from {len(quarter_samples)} available)")

    # Deterministic shuffle
    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    print(f"\nTotal samples: {len(all_samples)}")
    return all_samples, quarter_stats


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
    actual_return = sample.get("actual_return")

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,
        )

        elapsed = time.time() - start_time
        ar = result.get("agentic_result", {}) or {}

        prediction = ar.get("prediction", "")
        confidence = ar.get("confidence", 0)
        trade_long = ar.get("trade_long", False)
        trade_long_tier = ar.get("trade_long_tier", "")
        risk_code = ar.get("risk_code", "")
        long_block_reason = ar.get("long_block_reason", "")

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
            "direction_score": ar.get("le_DirectionScore"),
            "actual_return": actual_return,
            "correct": correct,
            "trade_long": trade_long,
            "trade_long_tier": trade_long_tier,
            "risk_code": risk_code,
            "long_block_reason": long_block_reason,
            "eps_surprise": ar.get("eps_surprise"),
            "earnings_day_return": ar.get("earnings_day_return"),
            "pre_earnings_5d_return": ar.get("pre_earnings_5d_return"),
            "computed_positives": ar.get("computed_positives"),
            "computed_vetoes": ar.get("computed_vetoes"),
            "le_DirectionScore": ar.get("le_DirectionScore"),
            "le_HardPositivesCount": ar.get("le_HardPositivesCount"),
            "le_HardVetoCount": ar.get("le_HardVetoCount"),
            "le_PricedInRisk": ar.get("le_PricedInRisk"),
            "le_GuidanceRaised": ar.get("le_GuidanceRaised"),
            "le_DemandAcceleration": ar.get("le_DemandAcceleration"),
            "le_MarginExpansion": ar.get("le_MarginExpansion"),
            "le_FCFImprovement": ar.get("le_FCFImprovement"),
            "le_VisibilityImproving": ar.get("le_VisibilityImproving"),
            "le_GuidanceCut": ar.get("le_GuidanceCut"),
            "le_DemandSoftness": ar.get("le_DemandSoftness"),
            "le_MarginWeakness": ar.get("le_MarginWeakness"),
            "le_CashBurn": ar.get("le_CashBurn"),
            "le_VisibilityWorsening": ar.get("le_VisibilityWorsening"),
            "input_tokens": ar.get("input_tokens"),
            "output_tokens": ar.get("output_tokens"),
            "cost_usd": ar.get("cost_usd"),
            "summary": ar.get("summary", "")[:500] if ar.get("summary") else "",
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


async def run_batch(samples: List[Dict], concurrency: int, report_interval: int = 100) -> List[Dict]:
    """Run analysis with concurrency limit."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    completed = 0
    start_time = time.time()

    async def run_with_semaphore(sample: Dict) -> Dict:
        nonlocal completed
        async with semaphore:
            result = await analyze_sample(sample)
            completed += 1

            status = "OK" if result["success"] else "FAIL"
            trade_long = "LONG" if result.get("trade_long") else ""
            print(f"[{completed}/{len(samples)}] {sample['symbol']} {sample['year']}Q{sample['quarter']}: {status} {trade_long}")

            if completed % report_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed * 60
                print(f"  Progress: {completed}/{len(samples)} | Rate: {rate:.1f}/min")

            return result

    tasks = [run_with_semaphore(s) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)


# =============================================================================
# OOS-Specific Reporting
# =============================================================================

def generate_oos_report(results: List[Dict], output_dir: Path, quarter_stats: Dict, seed: int) -> Dict[str, str]:
    """Generate OOS-specific reports with drift detection."""
    files = {}

    df = pd.DataFrame(results)
    valid = df[df['success'] & df['actual_return'].notna()].copy()
    long_df = valid[valid['trade_long'] == True].copy()

    # Main OOS report
    md_lines = [
        f"# 2025H2 Out-of-Sample Validation Report",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Seed**: {seed}",
        f"**Period**: 2025Q3 + 2025Q4",
        f"",
        f"---",
        f"",
        f"## 1. Universe Statistics",
        f"",
        f"| Quarter | Available | Selected | Sectors |",
        f"|---------|-----------|----------|---------|",
    ]

    for qtr, stats in quarter_stats.items():
        md_lines.append(f"| {qtr} | {stats['available']} | {stats['selected']} | {stats['sectors']} |")

    md_lines.extend([
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Samples | {len(df)} |",
        f"| Success Rate | {100*df['success'].mean():.1f}% |",
        f"| Valid Samples | {len(valid)} |",
    ])

    # Long trades analysis
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Model-Eligible Long Trades (Core KPI)",
        f"",
    ])

    if len(long_df) > 0:
        long_wins = (long_df['actual_return'] > 0).sum()
        coverage = len(long_df) / len(valid) * 100

        md_lines.extend([
            f"| Metric | Value | Benchmark (Historical) |",
            f"|--------|-------|------------------------|",
            f"| **trade_long count** | **{len(long_df)}** | - |",
            f"| **Coverage** | **{coverage:.1f}%** | ~8.3% |",
            f"| **Win Rate** | **{100*long_wins/len(long_df):.1f}%** | ~81% |",
            f"| **Avg Return** | **{long_df['actual_return'].mean():+.2f}%** | - |",
            f"| Median Return | {long_df['actual_return'].median():+.2f}% | - |",
        ])

        # Wilson CI
        lower, upper, hw = wilson_ci(long_wins, len(long_df))
        md_lines.extend([
            f"",
            f"### Wilson 95% Confidence Interval",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {100*long_wins/len(long_df):.1f}% |",
            f"| CI Lower | {100*lower:.1f}% |",
            f"| CI Upper | {100*upper:.1f}% |",
            f"| Half-width | {100*hw:.1f}% |",
        ])

        # Tier breakdown
        md_lines.extend([
            f"",
            f"### Tier Breakdown",
            f"",
            f"| Tier | Trades | Win Rate | Avg Return | % of Total |",
            f"|------|--------|----------|------------|------------|",
        ])

        for tier in ['D7_CORE', 'D6_STRICT']:
            tier_df = long_df[long_df['trade_long_tier'] == tier]
            if len(tier_df) > 0:
                tier_wins = (tier_df['actual_return'] > 0).sum()
                tier_pct = 100 * len(tier_df) / len(long_df)
                md_lines.append(
                    f"| {tier} | {len(tier_df)} | {100*tier_wins/len(tier_df):.1f}% | "
                    f"{tier_df['actual_return'].mean():+.2f}% | {tier_pct:.1f}% |"
                )

        # Quarter comparison
        md_lines.extend([
            f"",
            f"### Quarter Comparison (Drift Detection)",
            f"",
            f"| Quarter | Trades | Win Rate | Avg Return |",
            f"|---------|--------|----------|------------|",
        ])

        for year, qtr in OOS_QUARTERS:
            q_df = long_df[(long_df['year'] == year) & (long_df['quarter'] == qtr)]
            if len(q_df) > 0:
                q_wins = (q_df['actual_return'] > 0).sum()
                md_lines.append(
                    f"| {year}Q{qtr} | {len(q_df)} | {100*q_wins/len(q_df):.1f}% | "
                    f"{q_df['actual_return'].mean():+.2f}% |"
                )

        # Loss tail
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Risk Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| P05 | {long_df['actual_return'].quantile(0.05):.2f}% |",
            f"| P10 | {long_df['actual_return'].quantile(0.10):.2f}% |",
            f"| Max Loss | {long_df['actual_return'].min():.2f}% |",
            f"| Max Gain | {long_df['actual_return'].max():.2f}% |",
        ])

        wins_sum = long_df[long_df['actual_return'] > 0]['actual_return'].sum()
        losses_sum = abs(long_df[long_df['actual_return'] <= 0]['actual_return'].sum())
        pf = wins_sum / losses_sum if losses_sum > 0 else float('inf')
        md_lines.append(f"| Profit Factor | {pf:.2f} |")

        # Block reason distribution
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Drift Indicators",
            f"",
            f"### Risk Code Distribution",
            f"",
        ])

        risk_dist = valid['risk_code'].value_counts()
        md_lines.append(f"| Risk Code | Count | % |")
        md_lines.append(f"|-----------|-------|---|")
        for code, count in risk_dist.items():
            md_lines.append(f"| {code or 'N/A'} | {count} | {100*count/len(valid):.1f}% |")

        # Block reason top 5
        blocked = valid[valid['trade_long'] == False]
        if len(blocked) > 0:
            block_dist = blocked['long_block_reason'].value_counts().head(5)
            md_lines.extend([
                f"",
                f"### Top Block Reasons (non-trade_long)",
                f"",
                f"| Reason | Count | % |",
                f"|--------|-------|---|",
            ])
            for reason, count in block_dist.items():
                md_lines.append(f"| {reason or 'N/A'} | {count} | {100*count/len(blocked):.1f}% |")

    else:
        md_lines.append(f"**No trade_long signals generated!** This indicates a potential issue.")

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    # Save report
    report_path = output_dir / "oos_2025h2_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(md_lines))
    files['report'] = str(report_path)

    # Save worst losses
    if len(long_df) > 0:
        worst = long_df.nsmallest(15, 'actual_return')[
            ['symbol', 'year', 'quarter', 'sector', 'trade_long_tier', 'actual_return', 'direction_score']
        ]
        worst_path = output_dir / "oos_worst_losses.csv"
        worst.to_csv(worst_path, index=False)
        files['worst_losses'] = str(worst_path)

    return files


def save_metadata(output_dir: Path, seed: int, samples_count: int, per_quarter: int, results_path: str, stats_files: Dict) -> str:
    """Save run metadata."""
    metadata = {
        "run_type": "OOS_2025H2",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "samples_count": samples_count,
        "per_quarter": per_quarter,
        "quarters": [f"{y}Q{q}" for y, q in OOS_QUARTERS],
        "results_path": results_path,
        "stats_files": stats_files,
        "env_snapshot": {
            "MAIN_MODEL": os.environ.get("MAIN_MODEL"),
            "HELPER_MODEL": os.environ.get("HELPER_MODEL"),
            "LOOKAHEAD_ASSERTIONS": os.environ.get("LOOKAHEAD_ASSERTIONS"),
            "CALL_CACHE_VERSION": os.environ.get("CALL_CACHE_VERSION"),
        }
    }

    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return str(meta_path)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="2025H2 Out-of-Sample Validation")
    parser.add_argument("--test-only", action="store_true", help="Only run test samples")
    parser.add_argument("--per-quarter", type=int, default=DEFAULT_PER_QUARTER, help="Samples per quarter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--parallel", type=int, default=DEFAULT_CONCURRENCY, help="Concurrency")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"2025H2 OUT-OF-SAMPLE VALIDATION")
    print(f"{'='*70}")
    print(f"MAIN_MODEL: {os.environ.get('MAIN_MODEL', 'default')}")
    print(f"HELPER_MODEL: {os.environ.get('HELPER_MODEL', 'default')}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Setup output directory (use absolute path to avoid issues during long runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (Path(__file__).parent / f"EarningsCallAgenticRag/oos_runs/oos_2025h2_{args.per_quarter * 2}_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get samples
    samples, quarter_stats = get_2025h2_samples(per_quarter=args.per_quarter, seed=args.seed)

    # Save manifest
    manifest_path = output_dir / f"oos_manifest_seed{args.seed}.csv"
    pd.DataFrame(samples).to_csv(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path}")

    # Test mode
    if args.test_only:
        print(f"\n--- TEST MODE ({TEST_SAMPLES} samples) ---\n")
        test_results = await run_batch(samples[:TEST_SAMPLES], concurrency=TEST_SAMPLES, report_interval=5)

        test_path = output_dir / "test_results.csv"
        pd.DataFrame(test_results).to_csv(test_path, index=False)
        print(f"Test results saved: {test_path}")

        long_count = sum(1 for r in test_results if r.get("trade_long"))
        print(f"\nTest Summary: {len(test_results)} samples, {long_count} trade_long")
        return

    # Full run
    print(f"\n--- FULL RUN ({len(samples)} samples) ---\n")
    all_results = await run_batch(samples, concurrency=args.parallel, report_interval=100)

    # Save results
    results_path = output_dir / "oos_results.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Generate reports
    stats_files = generate_oos_report(all_results, output_dir, quarter_stats, args.seed)

    # Save metadata
    save_metadata(output_dir, args.seed, len(samples), args.per_quarter, str(results_path), stats_files)

    # Print summary
    df = pd.DataFrame(all_results)
    valid = df[df['success'] & df['actual_return'].notna()]
    long_trades = valid[valid['trade_long'] == True]

    print(f"\n{'='*70}")
    print(f"2025H2 OOS SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(df)} | Valid: {len(valid)} | Long: {len(long_trades)}")

    if len(long_trades) > 0:
        wins = (long_trades['actual_return'] > 0).sum()
        print(f"Win Rate: {wins}/{len(long_trades)} ({100*wins/len(long_trades):.1f}%)")
        print(f"Avg Return: {long_trades['actual_return'].mean():+.2f}%")

        lower, upper, _ = wilson_ci(wins, len(long_trades))
        print(f"Wilson 95% CI: [{100*lower:.1f}%, {100*upper:.1f}%]")

    print(f"{'='*70}")
    print(f"\nReport: {stats_files.get('report')}")


if __name__ == "__main__":
    # Change to script directory
    os.chdir(Path(__file__).parent)
    asyncio.run(main())
