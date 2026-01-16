#!/usr/bin/env python3
"""
Download OHLC prices from Whaleforce Backtest API.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
import argparse


BACKTEST_API_URL = "https://backtest.api.whaleforce.dev"


def get_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Get OHLCV data from Whaleforce API."""
    url = f"{BACKTEST_API_URL}/data-management/ohlcv"
    params = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "interval": interval,
    }

    all_items = []
    offset = 0
    limit = 200

    try:
        while True:
            params["offset"] = offset
            params["limit"] = limit
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data or "items" not in data:
                break

            items = data["items"]
            if not items:
                break

            all_items.extend(items)

            # Check if we got all data
            if len(items) < limit or len(all_items) >= data.get("total", len(all_items)):
                break

            offset += limit
            time.sleep(0.05)  # Small delay between pagination calls

        if not all_items:
            return pd.DataFrame()

        df = pd.DataFrame(all_items)
        if 'ts' in df.columns:
            df['date'] = pd.to_datetime(df['ts']).dt.date
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download OHLC prices")
    parser.add_argument('--signals', type=str, required=True,
                        help='Signals CSV with symbol column')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for price CSVs')
    parser.add_argument('--start', type=str, default='2017-01-01',
                        help='Start date')
    parser.add_argument('--end', type=str, default='2026-02-01',
                        help='End date')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay between API calls')

    args = parser.parse_args()

    # Get symbols
    signals = pd.read_csv(args.signals)
    symbols = sorted(signals['symbol'].unique().tolist())
    print(f"Downloading prices for {len(symbols)} symbols")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download each symbol
    success = 0
    failed = []

    for i, sym in enumerate(symbols):
        out_file = out_dir / f"{sym}.csv"

        # Skip if already exists
        if out_file.exists():
            df = pd.read_csv(out_file)
            if len(df) > 100:
                success += 1
                continue

        print(f"[{i+1}/{len(symbols)}] Downloading {sym}...")
        df = get_ohlcv(sym, args.start, args.end)

        if len(df) > 0:
            df.to_csv(out_file, index=False)
            success += 1
            print(f"  Saved {len(df)} rows")
        else:
            failed.append(sym)
            print(f"  No data")

        time.sleep(args.delay)

    print(f"\nDone. Success: {success}, Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {failed[:20]}...")


if __name__ == "__main__":
    main()
