#!/usr/bin/env python3
"""
Lookahead Fix Validation Script

Step-by-step validation:
1. Run 2017Q1/Q2 earliest samples (3-5 stocks) with assertions enabled
2. Force skip_cache to ensure fresh data
3. Check that no lookahead guard triggers
4. Verify financials dates <= transcript_date
5. Verify historical facts quarters < current quarter

Usage:
    python backtest_tools/validate_lookahead_fix.py

Environment:
    LOOKAHEAD_ASSERTIONS=true (default)
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Force lookahead assertions on
os.environ["LOOKAHEAD_ASSERTIONS"] = "true"
os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"

from pg_client import (
    get_cursor,
    get_historical_earnings_facts,
    get_historical_financials_facts,
    get_quarterly_financials,
)


def get_2017_early_samples(limit: int = 5) -> List[Dict]:
    """Get the earliest 2017 samples (Q1/Q2) for validation."""
    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        cur.execute("""
            SELECT
                et.symbol,
                et.year,
                et.quarter,
                et.transcript_date_str,
                pa.pct_change_t_plus_30 as actual_return
            FROM earnings_transcripts et
            LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
            WHERE et.year = 2017
                AND et.quarter IN (1, 2)
                AND et.transcript_date_str IS NOT NULL
                AND pa.pct_change_t_plus_30 IS NOT NULL
            ORDER BY et.transcript_date_str ASC
            LIMIT %s
        """, (limit,))

        samples = []
        for row in cur.fetchall():
            samples.append({
                "symbol": row["symbol"],
                "year": row["year"],
                "quarter": row["quarter"],
                "transcript_date": row["transcript_date_str"],
                "actual_return": float(row["actual_return"]) if row["actual_return"] else None
            })

        return samples


def validate_historical_earnings(symbol: str, year: int, quarter: int, transcript_date: str) -> Tuple[bool, str]:
    """
    Validate get_historical_earnings_facts() returns no future data.

    Returns: (passed, message)
    """
    current_quarter_str = f"{year}Q{quarter}"
    facts = get_historical_earnings_facts(
        symbol=symbol,
        current_quarter=current_quarter_str,
        num_quarters=10
    )

    if not facts:
        return True, "No historical earnings data (OK)"

    issues = []
    for fact in facts:
        fact_year = fact.get("year", 0)
        fact_quarter = fact.get("quarter", 0)

        # Check 1: No future quarters
        if fact_year > year or (fact_year == year and fact_quarter >= quarter):
            issues.append(f"Future quarter: {fact_year}Q{fact_quarter}")

        # Check 2: No post-earnings returns (should be None or not present)
        if fact.get("return_20d") is not None:
            issues.append(f"Leaked return_20d: {fact.get('return_20d')}")
        if fact.get("return_30d") is not None:
            issues.append(f"Leaked return_30d: {fact.get('return_30d')}")

    if issues:
        return False, f"LOOKAHEAD: {issues}"

    return True, f"OK: {len(facts)} facts, all before {year}Q{quarter}"


def validate_historical_financials(symbol: str, quarter_str: str, transcript_date: str) -> Tuple[bool, str]:
    """
    Validate get_historical_financials_facts() respects as_of_date.

    Returns: (passed, message)
    """
    facts = get_historical_financials_facts(
        symbol=symbol,
        current_quarter=quarter_str,
        num_quarters=8,
        as_of_date=transcript_date
    )

    if not facts or not facts.get("income"):
        return True, "No historical financials (OK)"

    issues = []
    for stmt in facts.get("income", []):
        stmt_date = stmt.get("date", "")
        if stmt_date and stmt_date >= transcript_date:
            issues.append(f"Future financial: {stmt_date} >= {transcript_date}")

    if issues:
        return False, f"LOOKAHEAD: {issues}"

    dates = [s.get("date", "")[:10] for s in facts.get("income", []) if s.get("date")]
    return True, f"OK: {len(facts.get('income', []))} statements, max date: {max(dates) if dates else 'N/A'}"


def validate_quarterly_financials(symbol: str, transcript_date: str) -> Tuple[bool, str]:
    """
    Validate get_quarterly_financials() with before_date.

    Returns: (passed, message)
    """
    result = get_quarterly_financials(
        symbol=symbol,
        limit=4,
        before_date=transcript_date
    )

    if not result or not result.get("income"):
        return True, "No quarterly financials (OK)"

    issues = []
    for stmt in result.get("income", []):
        stmt_date = stmt.get("date", "")
        if stmt_date and stmt_date >= transcript_date:
            issues.append(f"Future statement: {stmt_date}")

    if issues:
        return False, f"LOOKAHEAD: {issues}"

    dates = [s.get("date", "")[:10] for s in result.get("income", []) if s.get("date")]
    return True, f"OK: {len(result.get('income', []))} statements, max date: {max(dates) if dates else 'N/A'}"


def run_validation():
    """Run the full validation sequence."""
    print("=" * 70)
    print("LOOKAHEAD FIX VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS', 'not set')}")
    print(f"HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS: {os.environ.get('HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS', 'not set')}")
    print()

    # Step 1: Get 2017 early samples
    print("Step 1: Fetching 2017Q1/Q2 earliest samples...")
    try:
        samples = get_2017_early_samples(limit=5)
    except Exception as e:
        print(f"  ERROR: Failed to get samples: {e}")
        return False

    if not samples:
        print("  WARNING: No 2017Q1/Q2 samples found in database")
        return False

    print(f"  Found {len(samples)} samples:")
    for s in samples:
        print(f"    - {s['symbol']} {s['year']}Q{s['quarter']} ({s['transcript_date']})")
    print()

    # Step 2: Validate each sample
    print("Step 2: Validating lookahead protection...")
    all_passed = True
    results = []

    for sample in samples:
        symbol = sample["symbol"]
        year = sample["year"]
        quarter = sample["quarter"]
        transcript_date = sample["transcript_date"]
        quarter_str = f"{year}Q{quarter}"

        print(f"\n  --- {symbol} {quarter_str} (transcript: {transcript_date}) ---")

        # Test 1: Historical earnings
        passed1, msg1 = validate_historical_earnings(symbol, year, quarter, transcript_date)
        status1 = "✅" if passed1 else "❌"
        print(f"    {status1} Historical Earnings: {msg1}")

        # Test 2: Historical financials
        passed2, msg2 = validate_historical_financials(symbol, quarter_str, transcript_date)
        status2 = "✅" if passed2 else "❌"
        print(f"    {status2} Historical Financials: {msg2}")

        # Test 3: Quarterly financials
        passed3, msg3 = validate_quarterly_financials(symbol, transcript_date)
        status3 = "✅" if passed3 else "❌"
        print(f"    {status3} Quarterly Financials: {msg3}")

        sample_passed = passed1 and passed2 and passed3
        all_passed = all_passed and sample_passed

        results.append({
            "symbol": symbol,
            "quarter": quarter_str,
            "passed": sample_passed,
            "tests": {
                "historical_earnings": (passed1, msg1),
                "historical_financials": (passed2, msg2),
                "quarterly_financials": (passed3, msg3),
            }
        })

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    print(f"Total Samples: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\n❌ FAILURES DETECTED:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['symbol']} {r['quarter']}")
                for test_name, (passed, msg) in r["tests"].items():
                    if not passed:
                        print(f"      {test_name}: {msg}")

    print()
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED")
        print("   No lookahead bias detected in 2017 early samples.")
        print("   Ready for small-scale backtest verification.")
    else:
        print("❌ VALIDATION FAILED")
        print("   Lookahead bias still present. Review the fixes.")

    return all_passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
