#!/usr/bin/env python3
"""
Sanity check: Test 3 samples from 2024 to verify we can reproduce original validation results.

Expected results from agentic-openenvolve2 validation:
1. OMC 2024-Q1: trade_long=True, tier=D7_CORE, direction_score>=7
2. SPGI 2024-Q2: trade_long=False, reason=D6_DAY_RET_LOW
3. NXPI 2024-Q3: trade_long=False, reason=HIGH_RISK
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Add the EarningsCallAgenticRag to path
sys.path.insert(0, str(Path(__file__).parent / "EarningsCallAgenticRag"))

import pg_client
from agentic_rag_bridge import run_single_call_from_context


# Test cases with expected results
TEST_CASES = [
    {
        "symbol": "OMC",
        "year": 2024,
        "quarter": 1,
        "expected_trade_long": True,
        "expected_tier": "D7_CORE",
        "expected_direction": ">=7",
    },
    {
        "symbol": "SPGI",
        "year": 2024,
        "quarter": 2,
        "expected_trade_long": False,
        "expected_reason": "D6_DAY_RET_LOW",
        "expected_direction": "6",
    },
    {
        "symbol": "NXPI",
        "year": 2024,
        "quarter": 3,
        "expected_trade_long": False,
        "expected_reason": "HIGH_RISK",
        "expected_direction": "<=5",
    },
]


def get_transcript(symbol: str, year: int, quarter: int) -> dict:
    """Fetch transcript from database."""
    result = pg_client.get_transcript(symbol, year, quarter)
    if not result:
        raise ValueError(f"No transcript found for {symbol} {year}Q{quarter}")
    return result


def run_test(test_case: dict) -> dict:
    """Run a single test case."""
    symbol = test_case["symbol"]
    year = test_case["year"]
    quarter = test_case["quarter"]

    print(f"\n{'='*60}")
    print(f"Testing: {symbol} {year}Q{quarter}")
    print(f"{'='*60}")

    # Fetch transcript
    transcript_data = get_transcript(symbol, year, quarter)

    # Get company info for sector
    company_info = pg_client.get_company_info(symbol)
    sector = company_info.get("sector") if company_info else None

    context = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_text": transcript_data.get("content", ""),
        "transcript_date": transcript_data.get("date", ""),
        "sector": sector,
    }

    # Run the agentic RAG pipeline
    result = run_single_call_from_context(context)

    # Extract key metrics
    trade_long = result.get("trade_long", False)
    tier = result.get("trade_long_tier", "")
    block_reason = result.get("long_block_reason", "")
    long_json = result.get("long_eligible_json", {})
    direction_score = long_json.get("DirectionScore", 0) if long_json else 0

    return {
        "symbol": symbol,
        "quarter": f"{year}Q{quarter}",
        "trade_long": trade_long,
        "tier": tier,
        "block_reason": block_reason,
        "direction_score": direction_score,
        "expected": test_case,
    }


def evaluate_result(result: dict) -> bool:
    """Evaluate if result matches expected."""
    expected = result["expected"]

    # Check trade_long
    if result["trade_long"] != expected["expected_trade_long"]:
        print(f"  ❌ trade_long: got {result['trade_long']}, expected {expected['expected_trade_long']}")
        return False

    # Check tier/reason
    if expected["expected_trade_long"]:
        if result["tier"] != expected.get("expected_tier", ""):
            print(f"  ❌ tier: got {result['tier']}, expected {expected.get('expected_tier')}")
            return False
    else:
        if result["block_reason"] != expected.get("expected_reason", ""):
            print(f"  ⚠️ block_reason: got {result['block_reason']}, expected {expected.get('expected_reason')}")
            # Don't fail on block_reason mismatch if trade_long is correct

    # Check direction score range
    expected_dir = expected.get("expected_direction", "")
    if expected_dir.startswith(">="):
        threshold = int(expected_dir[2:])
        if result["direction_score"] < threshold:
            print(f"  ❌ direction_score: got {result['direction_score']}, expected {expected_dir}")
            return False
    elif expected_dir.startswith("<="):
        threshold = int(expected_dir[2:])
        if result["direction_score"] > threshold:
            print(f"  ❌ direction_score: got {result['direction_score']}, expected {expected_dir}")
            return False
    elif expected_dir.isdigit():
        if result["direction_score"] != int(expected_dir):
            print(f"  ⚠️ direction_score: got {result['direction_score']}, expected {expected_dir}")

    return True


def main():
    print("=" * 60)
    print("Sanity Check: Reproducing 2024 Validation Results")
    print("=" * 60)
    print(f"MAIN_MODEL: {os.getenv('MAIN_MODEL', 'NOT SET')}")
    print(f"HELPER_MODEL: {os.getenv('HELPER_MODEL', 'NOT SET')}")
    print(f"LITELLM_ENDPOINT: {os.getenv('LITELLM_ENDPOINT', 'NOT SET')}")

    results = []
    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        try:
            result = run_test(test_case)
            results.append(result)

            print(f"\nResult for {result['symbol']} {result['quarter']}:")
            print(f"  trade_long: {result['trade_long']}")
            print(f"  tier: {result['tier']}")
            print(f"  block_reason: {result['block_reason']}")
            print(f"  direction_score: {result['direction_score']}")

            if evaluate_result(result):
                print(f"  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL")
                failed += 1

        except Exception as e:
            print(f"\n❌ Error testing {test_case['symbol']}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
