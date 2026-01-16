#!/usr/bin/env python3
"""
Lookahead Bias / Data Leakage Smoke Test

This script validates that the historical data functions properly enforce
time boundaries and do not leak future data into backtesting contexts.

Usage:
    python backtest_tools/leakage_smoke_test.py

Environment:
    DATABASE_URL: PostgreSQL connection string

Exit Codes:
    0: All tests passed
    1: One or more leakage tests failed
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pg_client import (
    get_historical_earnings_facts,
    get_historical_financials_facts,
    get_quarterly_financials,
    get_peer_facts_summary,
)
from fmp_client import get_quarterly_financials as fmp_get_quarterly_financials


class LeakageTestResult:
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details

    def __str__(self):
        status = "" if self.passed else ""
        return f"{status} {self.name}: {self.details}"


def test_historical_earnings_no_future_quarters(
    symbol: str = "AAPL",
    test_year: int = 2020,
    test_quarter: int = 2
) -> LeakageTestResult:
    """
    Test that get_historical_earnings_facts() doesn't return future quarters.
    """
    result = get_historical_earnings_facts(
        symbol=symbol,
        current_year=test_year,
        current_quarter=test_quarter,
        limit=20
    )

    if not result:
        return LeakageTestResult(
            "historical_earnings_no_future",
            True,
            f"No data for {symbol} {test_year}Q{test_quarter} (expected for some symbols)"
        )

    # Check each returned record
    leaks = []
    for record in result:
        rec_year = record.get("year", 0)
        rec_quarter = record.get("quarter", 0)

        # Check for future data
        if rec_year > test_year or (rec_year == test_year and rec_quarter >= test_quarter):
            leaks.append(f"{rec_year}Q{rec_quarter}")

    if leaks:
        return LeakageTestResult(
            "historical_earnings_no_future",
            False,
            f"LEAKAGE: Found future quarters: {leaks}"
        )

    return LeakageTestResult(
        "historical_earnings_no_future",
        True,
        f"OK: {len(result)} records, all before {test_year}Q{test_quarter}"
    )


def test_historical_earnings_no_post_returns(
    symbol: str = "AAPL",
    test_year: int = 2020,
    test_quarter: int = 2
) -> LeakageTestResult:
    """
    Test that get_historical_earnings_facts() doesn't return T+20/T+30 returns by default.
    """
    # Ensure env var is not set
    original = os.environ.get("HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS")
    os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"

    try:
        result = get_historical_earnings_facts(
            symbol=symbol,
            current_year=test_year,
            current_quarter=test_quarter,
            limit=5
        )

        if not result:
            return LeakageTestResult(
                "historical_earnings_no_post_returns",
                True,
                f"No data for {symbol}"
            )

        # Check for leaked return columns
        for record in result:
            if "return_20d" in record or "return_30d" in record:
                if record.get("return_20d") is not None or record.get("return_30d") is not None:
                    return LeakageTestResult(
                        "historical_earnings_no_post_returns",
                        False,
                        f"LEAKAGE: Post-earnings returns found in output"
                    )

        return LeakageTestResult(
            "historical_earnings_no_post_returns",
            True,
            "OK: No post-earnings returns in output"
        )
    finally:
        if original is None:
            os.environ.pop("HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS", None)
        else:
            os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = original


def test_historical_financials_date_filter(
    symbol: str = "AAPL",
    as_of_date: str = "2020-06-01"
) -> LeakageTestResult:
    """
    Test that get_historical_financials_facts() respects the as_of_date.
    """
    result = get_historical_financials_facts(
        symbol=symbol,
        quarter="2020Q2",
        as_of_date=as_of_date,
        limit=10
    )

    if not result or not result.get("income"):
        return LeakageTestResult(
            "historical_financials_date_filter",
            True,
            f"No financial data for {symbol} before {as_of_date}"
        )

    # Check each income statement date
    leaks = []
    for stmt in result.get("income", []):
        stmt_date = stmt.get("date", "")
        if stmt_date and stmt_date >= as_of_date:
            leaks.append(stmt_date)

    if leaks:
        return LeakageTestResult(
            "historical_financials_date_filter",
            False,
            f"LEAKAGE: Found statements dated >= {as_of_date}: {leaks}"
        )

    return LeakageTestResult(
        "historical_financials_date_filter",
        True,
        f"OK: All {len(result.get('income', []))} statements before {as_of_date}"
    )


def test_quarterly_financials_before_date(
    symbol: str = "AAPL",
    before_date: str = "2020-06-01"
) -> LeakageTestResult:
    """
    Test that get_quarterly_financials() with before_date doesn't return future data.
    """
    result = get_quarterly_financials(
        symbol=symbol,
        limit=4,
        before_date=before_date
    )

    if not result or not result.get("income"):
        return LeakageTestResult(
            "quarterly_financials_before_date",
            True,
            f"No financial data for {symbol} before {before_date}"
        )

    # Check each statement date
    leaks = []
    for stmt in result.get("income", []):
        stmt_date = stmt.get("date", "")
        if stmt_date and stmt_date >= before_date:
            leaks.append(stmt_date)

    if leaks:
        return LeakageTestResult(
            "quarterly_financials_before_date",
            False,
            f"LEAKAGE: Found statements dated >= {before_date}: {leaks}"
        )

    return LeakageTestResult(
        "quarterly_financials_before_date",
        True,
        f"OK: All statements before {before_date}"
    )


def test_peer_facts_no_future_returns(
    symbol: str = "AAPL",
    quarter: str = "2020Q2",
    as_of_date: str = "2020-06-01"
) -> LeakageTestResult:
    """
    Test that get_peer_facts_summary() with as_of_date doesn't leak post-earnings returns.
    """
    result = get_peer_facts_summary(
        symbol=symbol,
        quarter=quarter,
        limit=5,
        as_of_date=as_of_date
    )

    if not result:
        return LeakageTestResult(
            "peer_facts_no_future_returns",
            True,
            f"No peer data for {symbol} sector"
        )

    # Check for leaked return columns
    for peer in result:
        if "return_20d" in peer or "return_30d" in peer:
            if peer.get("return_20d") is not None or peer.get("return_30d") is not None:
                return LeakageTestResult(
                    "peer_facts_no_future_returns",
                    False,
                    f"LEAKAGE: Post-earnings returns found for peer {peer.get('symbol')}"
                )

    return LeakageTestResult(
        "peer_facts_no_future_returns",
        True,
        f"OK: No post-earnings returns in peer data"
    )


def run_all_tests(symbols: List[str] = None) -> Tuple[int, int]:
    """
    Run all leakage tests and return (passed, failed) counts.
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]

    print("=" * 60)
    print("LOOKAHEAD BIAS / DATA LEAKAGE SMOKE TEST")
    print("=" * 60)
    print(f"Test Date: {datetime.now().isoformat()}")
    print(f"Test Symbols: {symbols}")
    print()

    results = []
    test_cases = [
        ("2020Q2", "2020-06-01", 2020, 2),
        ("2018Q4", "2019-01-15", 2018, 4),
        ("2017Q1", "2017-04-01", 2017, 1),
    ]

    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")

        for quarter_str, as_of_date, year, q in test_cases:
            print(f"\n  Scenario: {quarter_str} (as_of: {as_of_date})")

            # Test 1: No future quarters in historical earnings
            r1 = test_historical_earnings_no_future_quarters(symbol, year, q)
            results.append(r1)
            print(f"    {r1}")

            # Test 2: No post-earnings returns
            r2 = test_historical_earnings_no_post_returns(symbol, year, q)
            results.append(r2)
            print(f"    {r2}")

            # Test 3: Historical financials date filter
            r3 = test_historical_financials_date_filter(symbol, as_of_date)
            results.append(r3)
            print(f"    {r3}")

            # Test 4: Quarterly financials before_date
            r4 = test_quarterly_financials_before_date(symbol, as_of_date)
            results.append(r4)
            print(f"    {r4}")

            # Test 5: Peer facts no future returns
            r5 = test_peer_facts_no_future_returns(symbol, quarter_str, as_of_date)
            results.append(r5)
            print(f"    {r5}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\n FAILURES:")
        for r in results:
            if not r.passed:
                print(f"  - {r}")

    print()
    return passed, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run lookahead bias smoke tests")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Symbols to test"
    )
    args = parser.parse_args()

    passed, failed = run_all_tests(args.symbols)

    if failed > 0:
        print(" LEAKAGE DETECTED - DO NOT USE FOR LIVE TRADING")
        sys.exit(1)
    else:
        print(" ALL TESTS PASSED - No obvious lookahead bias detected")
        sys.exit(0)
