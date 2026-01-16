#!/usr/bin/env python3
"""
Prompt Leakage Validation Script

This script scans for forbidden keywords in prompts and context passed to LLM agents.
These keywords could indicate lookahead bias if present:

Forbidden keywords (prediction targets / future data):
- pct_change_t_plus_30, pct_change_t_plus_20, pct_change_t_plus
- return_30d, return_20d
- post_earnings_return
- trend_category (derived from future returns)

Usage:
    python backtest_tools/validate_prompt_no_leakage.py

Environment:
    LOOKAHEAD_ASSERTIONS=true (triggers assertion failures)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Forbidden Keywords Configuration
# =============================================================================

FORBIDDEN_KEYWORDS = [
    # Direct prediction targets
    "pct_change_t_plus_30",
    "pct_change_t_plus_20",
    "pct_change_t_plus_10",
    "pct_change_t_plus",  # Catch-all for any horizon
    "return_30d",
    "return_20d",
    "return_10d",
    "post_earnings_return",
    # Derived from future data
    "trend_category",
    "correct_prediction",  # Whether prediction was correct
]

# Patterns to match forbidden content
FORBIDDEN_PATTERNS = [
    # Numeric patterns that might be leaked returns
    re.compile(r'return[_\s]*[+]?\s*\d+\s*d', re.IGNORECASE),  # return+30d, return 20 d, etc.
    re.compile(r't\s*\+\s*\d+\s*day', re.IGNORECASE),  # T+30 day return
]


def env_bool(key: str, default: bool = False) -> bool:
    """Parse environment variable as boolean."""
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def scan_text_for_leakage(text: str, context_name: str = "unknown") -> List[Tuple[str, str]]:
    """
    Scan text for forbidden keywords indicating lookahead bias.

    Args:
        text: The text to scan (prompt, context, etc.)
        context_name: Name for logging purposes

    Returns:
        List of (keyword, context_snippet) tuples for any matches found
    """
    if not text:
        return []

    violations = []
    text_lower = text.lower()

    # Check exact keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword.lower() in text_lower:
            # Find the context around the match
            idx = text_lower.find(keyword.lower())
            start = max(0, idx - 50)
            end = min(len(text), idx + len(keyword) + 50)
            snippet = text[start:end].replace("\n", " ").strip()
            violations.append((keyword, f"...{snippet}..."))

    # Check patterns
    for pattern in FORBIDDEN_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            snippet = text[start:end].replace("\n", " ").strip()
            violations.append((match.group(), f"...{snippet}..."))

    return violations


def validate_dict_for_leakage(
    data: Dict[str, Any],
    path: str = "",
    max_depth: int = 10
) -> List[Tuple[str, str, str]]:
    """
    Recursively scan a dictionary for forbidden keywords.

    Args:
        data: Dictionary to scan
        path: Current path in the dict (for reporting)
        max_depth: Maximum recursion depth

    Returns:
        List of (path, keyword, snippet) tuples for violations
    """
    if max_depth <= 0:
        return []

    violations = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            # Check key name
            key_violations = scan_text_for_leakage(str(key), current_path)
            for keyword, snippet in key_violations:
                violations.append((current_path, keyword, f"key: {snippet}"))
            # Recurse into value
            violations.extend(validate_dict_for_leakage(value, current_path, max_depth - 1))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            violations.extend(validate_dict_for_leakage(item, current_path, max_depth - 1))

    elif isinstance(data, str):
        text_violations = scan_text_for_leakage(data, path)
        for keyword, snippet in text_violations:
            violations.append((path, keyword, snippet))

    return violations


def validate_no_lookahead_in_prompt(
    prompt: str,
    context: Dict[str, Any] = None,
    raise_on_violation: bool = None
) -> bool:
    """
    Validate that a prompt and context contain no lookahead data.

    Args:
        prompt: The prompt text being sent to LLM
        context: Optional context dictionary
        raise_on_violation: If True, raise AssertionError on violation.
                           Defaults to LOOKAHEAD_ASSERTIONS env var.

    Returns:
        True if validation passes (no violations), False otherwise

    Raises:
        AssertionError: If raise_on_violation is True and violations found
    """
    if raise_on_violation is None:
        raise_on_violation = env_bool("LOOKAHEAD_ASSERTIONS", default=True)

    violations = []

    # Scan prompt text
    prompt_violations = scan_text_for_leakage(prompt, "prompt")
    for keyword, snippet in prompt_violations:
        violations.append(("prompt", keyword, snippet))

    # Scan context dict
    if context:
        context_violations = validate_dict_for_leakage(context, "context")
        violations.extend(context_violations)

    if violations:
        msg_lines = ["LOOKAHEAD DETECTED IN PROMPT/CONTEXT:"]
        for path, keyword, snippet in violations:
            msg_lines.append(f"  - {path}: '{keyword}' in {snippet}")

        message = "\n".join(msg_lines)
        print(f"WARNING: {message}", file=sys.stderr)

        if raise_on_violation:
            raise AssertionError(message)
        return False

    return True


# =============================================================================
# Test the validator with sample data
# =============================================================================

def run_self_tests():
    """Run self-tests to verify the validator works correctly."""
    print("=" * 60)
    print("PROMPT LEAKAGE VALIDATOR SELF-TEST")
    print("=" * 60)

    test_cases = [
        # (name, text, should_fail)
        ("clean_prompt", "Analyze this earnings call for AAPL Q1 2024", False),
        ("has_return_30d", "The stock had return_30d of 15%", True),
        ("has_pct_change", "pct_change_t_plus_30 was positive", True),
        ("has_post_return", "post_earnings_return: 0.12", True),
        ("has_trend", "trend_category: positive", True),
        ("clean_financials", "Revenue: $100B, EPS: $2.50", False),
        ("clean_guidance", "Management raised guidance for Q2", False),
    ]

    passed = 0
    failed = 0

    for name, text, should_fail in test_cases:
        violations = scan_text_for_leakage(text, name)
        has_violations = len(violations) > 0

        if has_violations == should_fail:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"  {status}: {name}")
        if violations:
            for kw, snippet in violations:
                print(f"       Found: {kw}")

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()

    # Test dict validation
    print("Testing dict validation...")
    test_dict = {
        "symbol": "AAPL",
        "market_anchors": {
            "eps_surprise": 0.05,
            "earnings_day_return": 2.5,
        },
        "forbidden_field": {
            "return_30d": 0.15,  # Should trigger
        }
    }

    violations = validate_dict_for_leakage(test_dict)
    if violations:
        print(f"  Found {len(violations)} violation(s) in test dict (expected)")
        for path, kw, snippet in violations:
            print(f"    - {path}: {kw}")
    else:
        print("  ERROR: Should have found violations in test dict!")
        failed += 1

    print()
    return failed == 0


def main():
    """Main entry point."""
    success = run_self_tests()

    if success:
        print("ALL SELF-TESTS PASSED")
        print()
        print("Usage in code:")
        print("  from backtest_tools.validate_prompt_no_leakage import validate_no_lookahead_in_prompt")
        print("  validate_no_lookahead_in_prompt(prompt, context)  # raises on violation")
        print()
        sys.exit(0)
    else:
        print("SOME SELF-TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
