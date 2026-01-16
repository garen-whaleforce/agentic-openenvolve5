#!/usr/bin/env python3
"""
Prepare signals CSV with reaction_date for backtesting.

Merges LLM output CSVs with FMP earnings dates to create
a signals file compatible with the v1.1-live-safe backtest tools.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_fmp_earnings_dates(fmp_files: list) -> pd.DataFrame:
    """Load and combine FMP earnings surprise files to get dates."""
    all_dates = []

    for f in fmp_files:
        if not Path(f).exists():
            print(f"Warning: {f} not found, skipping")
            continue

        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = df['symbol'].str.upper()

        # Extract year and quarter from date
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter

        all_dates.append(df[['symbol', 'date', 'year', 'quarter']])

    if not all_dates:
        raise ValueError("No FMP files loaded successfully")

    combined = pd.concat(all_dates, ignore_index=True)

    # Remove duplicates (keep first occurrence per symbol/year/quarter)
    combined = combined.drop_duplicates(subset=['symbol', 'year', 'quarter'], keep='first')

    return combined.rename(columns={'date': 'reaction_date'})


def load_signals(csv_path: str) -> pd.DataFrame:
    """Load signals CSV from LLM output."""
    df = pd.read_csv(csv_path)
    df['symbol'] = df['symbol'].str.upper()
    return df


def merge_signals_with_dates(signals: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Merge signals with earnings dates."""
    merged = signals.merge(
        dates[['symbol', 'year', 'quarter', 'reaction_date']],
        on=['symbol', 'year', 'quarter'],
        how='left'
    )

    # Report missing dates
    missing = merged['reaction_date'].isna().sum()
    total = len(merged)
    print(f"Merged: {total - missing}/{total} samples have reaction_date ({missing} missing)")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Prepare signals CSV with reaction_date")
    parser.add_argument('--signals', type=str, nargs='+', required=True,
                        help='Input signals CSV files')
    parser.add_argument('--fmp', type=str, nargs='+', required=True,
                        help='FMP earnings surprise CSV files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    parser.add_argument('--require-date', action='store_true',
                        help='Only include samples with valid reaction_date')

    args = parser.parse_args()

    # Load earnings dates
    print("Loading FMP earnings dates...")
    dates = load_fmp_earnings_dates(args.fmp)
    print(f"Loaded {len(dates)} earnings date records")

    # Load and combine signals
    print("\nLoading signals...")
    all_signals = []
    for f in args.signals:
        print(f"  Loading {f}...")
        df = load_signals(f)
        print(f"    {len(df)} samples")
        all_signals.append(df)

    signals = pd.concat(all_signals, ignore_index=True)

    # Remove duplicates
    before = len(signals)
    signals = signals.drop_duplicates(subset=['symbol', 'year', 'quarter'], keep='first')
    print(f"\nAfter dedup: {len(signals)} samples (removed {before - len(signals)})")

    # Merge with dates
    print("\nMerging with earnings dates...")
    merged = merge_signals_with_dates(signals, dates)

    # Filter if required
    if args.require_date:
        before = len(merged)
        merged = merged[merged['reaction_date'].notna()]
        print(f"After filtering for valid dates: {len(merged)} samples (removed {before - len(merged)})")

    # Save
    merged.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total samples: {len(merged)}")
    print(f"trade_long=True: {merged['trade_long'].sum()}")
    print(f"Year range: {merged['year'].min()} - {merged['year'].max()}")
    print(f"Samples with reaction_date: {merged['reaction_date'].notna().sum()}")


if __name__ == "__main__":
    main()
