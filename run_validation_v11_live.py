#!/usr/bin/env python3
"""
v1.1-live Validation Runner (Lookahead-Free)

Changes from v1.1 (offline):
1. CAP: "First N" instead of "Top N" - no replacement after cap reached
2. TOPUP: Fixed score_floor_value instead of per-quarter percentile
3. Decision timestamp: Assumes reaction day close (not intraday)

This version can be safely used for paper trading without lookahead bias.

Author: Claude Code
Date: 2026-01-01
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# FROZEN v1.1-live PARAMETERS
# =============================================================================
FROZEN_PARAMS = {
    # Coverage control
    'cap_pct': 8.0,              # Max 8% coverage per quarter
    'min_quota': 3,              # Min 3 trades per quarter

    # TOPUP: Fixed score floor (estimated from 2019-2023 tune set)
    # This is P80 of relaxed_pool scores in tune set = 87.71
    'score_floor_value': 87.71,  # v1.1-live: FIXED value, no lookahead

    # Pre-run penalty
    'prerun_penalty_weight': 2.0,
    'prerun_threshold': 10.0,

    # Scoring weights
    'w_direction': 10.0,
    'w_positives': 3.0,
    'w_low_risk': 2.0,
    'w_eps': 5.0,
    'w_day_ret': 3.0,

    # Penalties
    'veto_penalty': 20.0,
    'high_risk_penalty': 15.0,
    'medium_risk_penalty': 5.0,
}

# Thresholds
D7_CORE = {
    'min_direction': 7,
    'max_vetoes': 0,
    'allow_high_risk': False,
    'min_eps_surprise': 0.0,  # > 0
    'min_day_return': 1.5,
}

D6_STRICT = {
    'direction': 6,
    'max_vetoes': 0,
    'min_day_return': 1.0,
    'min_positives': 1,
}


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0, center - spread), min(1, center + spread))


def load_and_process_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and preprocess."""
    df = pd.read_csv(csv_path)

    bool_cols = ['le_GuidanceRaised', 'le_DemandAcceleration', 'le_MarginExpansion',
                 'le_FCFImprovement', 'le_VisibilityImproving', 'le_GuidanceCut',
                 'le_DemandSoftness', 'le_MarginWeakness', 'le_CashBurn',
                 'le_VisibilityWorsening']

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'YES': True, 'NO': False, True: True, False: False})

    numeric_cols = ['le_DirectionScore', 'le_HardPositivesCount', 'le_HardVetoCount',
                    'eps_surprise', 'earnings_day_return', 'actual_return_30d_pct',
                    'pre_earnings_5d_return']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    pos_flags = ['le_GuidanceRaised', 'le_DemandAcceleration', 'le_MarginExpansion',
                 'le_FCFImprovement', 'le_VisibilityImproving']
    df['calc_positives'] = df[pos_flags].sum(axis=1)

    veto_flags = ['le_GuidanceCut', 'le_DemandSoftness', 'le_MarginWeakness',
                  'le_CashBurn', 'le_VisibilityWorsening']
    df['calc_vetoes'] = df[veto_flags].sum(axis=1)

    if 'le_PricedInRisk' in df.columns:
        df['is_high_risk'] = df['le_PricedInRisk'].str.lower() == 'high'
        df['is_low_risk'] = df['le_PricedInRisk'].str.lower() == 'low'
        df['is_medium_risk'] = df['le_PricedInRisk'].str.lower() == 'medium'
    else:
        df['is_high_risk'] = False
        df['is_low_risk'] = True
        df['is_medium_risk'] = False

    if 'year' in df.columns and 'quarter' in df.columns:
        df['quarter_str'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

    return df


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LongQualityScore with frozen parameters."""
    df = df.copy()
    p = FROZEN_PARAMS

    direction_score = df['le_DirectionScore'].fillna(0).astype(float) * p['w_direction']
    positives_score = df['calc_positives'].fillna(0).astype(float) * p['w_positives']
    low_risk_score = df['is_low_risk'].fillna(False).astype(int) * p['w_low_risk']

    eps_normalized = (df['eps_surprise'].fillna(0) / 0.20).clip(lower=-1, upper=1)
    eps_score = eps_normalized * p['w_eps']

    day_ret_normalized = (df['earnings_day_return'].fillna(0) / 5.0).clip(lower=-1, upper=1)
    day_ret_score = day_ret_normalized * p['w_day_ret']

    veto_penalty_score = df['calc_vetoes'].fillna(0).astype(float) * p['veto_penalty']
    high_risk_penalty_score = df['is_high_risk'].fillna(False).astype(int) * p['high_risk_penalty']
    medium_risk_penalty_score = df['is_medium_risk'].fillna(False).astype(int) * p['medium_risk_penalty']

    pre_run = df['pre_earnings_5d_return'].fillna(0) if 'pre_earnings_5d_return' in df.columns else 0
    prerun_excess = np.maximum(pre_run - p['prerun_threshold'], 0)
    prerun_penalty_score = prerun_excess * p['prerun_penalty_weight']

    df['long_quality_score'] = (
        direction_score + positives_score + low_risk_score +
        eps_score + day_ret_score -
        veto_penalty_score - high_risk_penalty_score -
        medium_risk_penalty_score - prerun_penalty_score
    )

    return df


def check_strict_eligibility(row: pd.Series) -> bool:
    """Check if a sample meets STRICT threshold (D7 CORE or D6 STRICT)."""
    direction = row.get('le_DirectionScore', 0) or 0
    vetoes = row.get('calc_vetoes', 0) or 0
    positives = row.get('calc_positives', 0) or 0
    eps_surprise = row.get('eps_surprise', 0) or 0
    day_return = row.get('earnings_day_return', 0) or 0
    is_high_risk = row.get('is_high_risk', False)

    # D7+ CORE
    if direction >= D7_CORE['min_direction'] and vetoes <= D7_CORE['max_vetoes']:
        if not is_high_risk and eps_surprise > D7_CORE['min_eps_surprise']:
            if day_return >= D7_CORE['min_day_return']:
                return True

    # D6 STRICT
    if direction == D6_STRICT['direction'] and vetoes <= D6_STRICT['max_vetoes']:
        if day_return >= D6_STRICT['min_day_return'] and positives >= D6_STRICT['min_positives']:
            return True

    return False


def check_relaxed_eligibility(row: pd.Series) -> bool:
    """Check if a sample is eligible for TOPUP (relaxed pool)."""
    vetoes = row.get('calc_vetoes', 0) or 0
    is_high_risk = row.get('is_high_risk', False)
    day_return = row.get('earnings_day_return', 0) or 0
    eps_surprise = row.get('eps_surprise', 0) or 0

    if vetoes > 0:
        return False
    if is_high_risk:
        return False
    if day_return < 0.5:
        return False
    if eps_surprise <= 0:
        return False

    return True


def apply_live_overlay_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply v1.1-live overlay strategy (lookahead-free).

    Key differences from offline v1.1:
    1. CAP: First N (chronological), not Top N
    2. TOPUP: Uses fixed score_floor_value, not per-quarter percentile
    """
    df = df.copy()
    p = FROZEN_PARAMS

    df['live_selected'] = False
    df['selection_type'] = ''
    df['selection_order'] = 0

    # Sort by quarter, then by some deterministic order within quarter
    # In live, this would be chronological order of earnings release
    # For backtest, we use index as proxy (assuming data is roughly chronological)
    df = df.sort_values(['quarter_str', df.index.name or 'symbol']).reset_index(drop=True)

    quarters = sorted(df['quarter_str'].unique())

    for q in quarters:
        q_mask = df['quarter_str'] == q
        q_indices = df[q_mask].index.tolist()
        n_samples = len(q_indices)

        cap_n = max(p['min_quota'], int(n_samples * p['cap_pct'] / 100))

        selected_count = 0
        strict_count = 0

        # Phase 1: Process samples in order, select STRICT until cap
        for idx in q_indices:
            row = df.loc[idx]

            if check_strict_eligibility(row):
                if selected_count < cap_n:
                    df.loc[idx, 'live_selected'] = True
                    df.loc[idx, 'selection_type'] = 'STRICT' if strict_count < cap_n else 'CAP'
                    selected_count += 1
                    strict_count += 1
                    df.loc[idx, 'selection_order'] = selected_count
                else:
                    # CAP reached - in v1.1-live, we DO NOT replace
                    df.loc[idx, 'selection_type'] = 'CAPPED_OUT'

        # Phase 2: TOPUP if below min_quota
        if selected_count < p['min_quota']:
            remaining = p['min_quota'] - selected_count

            # Find relaxed candidates with score >= fixed floor
            for idx in q_indices:
                if remaining <= 0:
                    break

                row = df.loc[idx]

                # Skip already selected
                if df.loc[idx, 'live_selected']:
                    continue

                # Check relaxed eligibility
                if not check_relaxed_eligibility(row):
                    continue

                # Check fixed score floor (v1.1-live: no lookahead)
                if row['long_quality_score'] < p['score_floor_value']:
                    continue

                # Select for TOPUP
                df.loc[idx, 'live_selected'] = True
                df.loc[idx, 'selection_type'] = 'TOPUP'
                selected_count += 1
                df.loc[idx, 'selection_order'] = selected_count
                remaining -= 1

    return df


def analyze_period(df: pd.DataFrame, period_name: str) -> Dict[str, Any]:
    """Analyze a single time period."""
    selected = df[df['live_selected'] == True]

    n_total = len(df)
    n_trades = len(selected)
    coverage = n_trades / n_total * 100 if n_total > 0 else 0

    if n_trades == 0:
        return {
            'period': period_name,
            'n_samples': n_total,
            'n_trades': 0,
            'coverage_pct': 0,
            'win_rate_pct': 0,
            'wins': 0,
            'losses': 0,
            'wilson_lower': 0,
            'wilson_upper': 0,
            'avg_return': 0,
            'strict_count': 0,
            'cap_count': 0,
            'topup_count': 0,
            'capped_out': 0,
        }

    returns = selected['actual_return_30d_pct']
    wins = (returns > 10).sum()
    win_rate = wins / n_trades * 100

    ci_lower, ci_upper = wilson_ci(wins, n_trades)

    strict_count = len(selected[selected['selection_type'] == 'STRICT'])
    cap_count = len(selected[selected['selection_type'] == 'CAP'])
    topup_count = len(selected[selected['selection_type'] == 'TOPUP'])
    capped_out = len(df[df['selection_type'] == 'CAPPED_OUT'])

    # Tail risk
    p05 = returns.quantile(0.05) if len(returns) >= 20 else returns.min()
    p10 = returns.quantile(0.10) if len(returns) >= 10 else returns.min()
    max_loss = returns.min()

    return {
        'period': period_name,
        'n_samples': n_total,
        'n_trades': n_trades,
        'coverage_pct': coverage,
        'win_rate_pct': win_rate,
        'wins': int(wins),
        'losses': n_trades - int(wins),
        'wilson_lower': ci_lower * 100,
        'wilson_upper': ci_upper * 100,
        'avg_return': returns.mean(),
        'strict_count': strict_count,
        'cap_count': cap_count,
        'topup_count': topup_count,
        'capped_out': capped_out,
        'p05': p05,
        'p10': p10,
        'max_loss': max_loss,
    }


def print_report(all_metrics: List[Dict]):
    """Print validation report."""
    print("\n" + "=" * 80)
    print("v1.1-LIVE VALIDATION REPORT (Lookahead-Free)")
    print("=" * 80)

    print("\n[Frozen Parameters - v1.1-live]")
    print(f"  score_floor_value: {FROZEN_PARAMS['score_floor_value']:.2f} (FIXED, from 2019-2023 tune)")
    print(f"  cap_pct: {FROZEN_PARAMS['cap_pct']}%")
    print(f"  min_quota: {FROZEN_PARAMS['min_quota']}")
    print("  CAP mode: FIRST-N (no replacement after cap reached)")

    print("\n" + "-" * 80)
    print("SUMMARY BY PERIOD")
    print("-" * 80)
    print(f"{'Period':<20} {'Samples':<10} {'Trades':<8} {'Coverage':<10} {'Win Rate':<12} {'Wilson LB':<12}")
    print("-" * 80)

    for m in all_metrics:
        print(f"{m['period']:<20} {m['n_samples']:<10} {m['n_trades']:<8} "
              f"{m['coverage_pct']:>6.1f}%    {m['win_rate_pct']:>6.1f}%      "
              f"{m['wilson_lower']:>6.1f}%")

    # Overall
    total_samples = sum(m['n_samples'] for m in all_metrics)
    total_trades = sum(m['n_trades'] for m in all_metrics)
    total_wins = sum(m['wins'] for m in all_metrics)

    if total_trades > 0:
        overall_win_rate = total_wins / total_trades * 100
        overall_coverage = total_trades / total_samples * 100
        overall_wilson_lower, _ = wilson_ci(total_wins, total_trades)

        print("-" * 80)
        print(f"{'OVERALL':<20} {total_samples:<10} {total_trades:<8} "
              f"{overall_coverage:>6.1f}%    {overall_win_rate:>6.1f}%      "
              f"{overall_wilson_lower*100:>6.1f}%")

    # Selection breakdown
    print("\n" + "-" * 80)
    print("SELECTION BREAKDOWN")
    print("-" * 80)
    print(f"{'Period':<20} {'STRICT':<10} {'CAP':<10} {'TOPUP':<10} {'Capped Out':<12}")
    print("-" * 80)

    for m in all_metrics:
        print(f"{m['period']:<20} {m['strict_count']:<10} {m['cap_count']:<10} "
              f"{m['topup_count']:<10} {m['capped_out']:<12}")

    # Tail risk
    print("\n" + "-" * 80)
    print("TAIL RISK")
    print("-" * 80)
    print(f"{'Period':<20} {'P05':<12} {'P10':<12} {'Max Loss':<12}")
    print("-" * 80)

    for m in all_metrics:
        print(f"{m['period']:<20} {m['p05']:>+8.1f}%   {m['p10']:>+8.1f}%   {m['max_loss']:>+8.1f}%")

    # Verdict
    print("\n" + "=" * 80)
    print("VALIDATION VERDICT (v1.1-live)")
    print("=" * 80)

    if total_trades > 0:
        if overall_wilson_lower >= 0.80:
            print("✅ PASS: Wilson 95% lower bound >= 80%")
        elif overall_wilson_lower >= 0.75:
            print("⚠️ MARGINAL: Wilson lower bound 75-80%")
        else:
            print("❌ FAIL: Wilson lower bound < 75%")

        if 5 <= overall_coverage <= 10:
            print("✅ PASS: Coverage within 5-10% target")
        else:
            print(f"⚠️ Coverage {overall_coverage:.1f}% outside 5-10% target")


def main():
    parser = argparse.ArgumentParser(description="v1.1-live Validation (Lookahead-Free)")
    parser.add_argument('--backward', type=str, help='2017-2018 CSV')
    parser.add_argument('--tune', type=str, help='2019-2023 CSV')
    parser.add_argument('--validate', type=str, help='2024 CSV')
    parser.add_argument('--forward', type=str, help='2025 CSV')

    args = parser.parse_args()

    periods = [
        ('backward_holdout', 'Backward Holdout', args.backward),
        ('tune', 'Tune Set', args.tune),
        ('validate', 'Validate Set', args.validate),
        ('forward_holdout', 'Forward Holdout', args.forward),
    ]

    all_metrics = []

    for key, name, csv_path in periods:
        if csv_path and os.path.exists(csv_path):
            print(f"\nProcessing: {name}")
            df = load_and_process_csv(csv_path)
            df = compute_quality_score(df)
            df = apply_live_overlay_strategy(df)

            metrics = analyze_period(df, name)
            all_metrics.append(metrics)

            print(f"  Samples: {metrics['n_samples']}, Trades: {metrics['n_trades']}")

    if all_metrics:
        print_report(all_metrics)


if __name__ == "__main__":
    main()
