
"""
price_providers.py

Lightweight price provider interfaces for local backtests / paper trading.

This module intentionally avoids hard-coding any single data source.
You can start with CSVPriceProvider and later add Postgres/FMP/Whaleforce
providers without changing the backtest engine.

Expected output schema for OHLC:
- index: pandas.DatetimeIndex (timezone-naive, date at midnight)
- columns: open, high, low, close (floats)
Optionally: adj_close, volume

All prices are assumed to be split-adjusted (or consistently unadjusted).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


class PriceProviderError(RuntimeError):
    pass


class PriceProvider:
    """Abstract interface."""

    def get_ohlc(
        self,
        symbols: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a dict: symbol -> OHLC dataframe between [start, end], inclusive.

        The dataframe MUST include columns: open, close
        and be indexed by session date (DatetimeIndex).
        """
        raise NotImplementedError


@dataclass
class CSVPriceProvider(PriceProvider):
    """
    Loads per-symbol OHLCV files from a folder.

    File naming:
      - <SYMBOL>.csv  (case-insensitive match)

    Accepted column names (case-insensitive):
      - date (or datetime)
      - open, high, low, close
      - adj close / adj_close (optional)

    Dates are parsed as UTC-naive dates and normalized to midnight.
    """
    folder: Path
    date_col: str = "date"

    def _find_file(self, symbol: str) -> Optional[Path]:
        sym = symbol.upper()
        # Try exact and case-insensitive
        candidates = [
            self.folder / f"{sym}.csv",
            self.folder / f"{sym.lower()}.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        # fallback: scan once (could be slow for huge folders)
        for p in self.folder.glob("*.csv"):
            if p.stem.upper() == sym:
                return p
        return None

    def _load_one(self, symbol: str) -> pd.DataFrame:
        fp = self._find_file(symbol)
        if fp is None:
            raise PriceProviderError(f"CSV price file not found for symbol={symbol} in folder={self.folder}")

        df = pd.read_csv(fp)
        if df.empty:
            raise PriceProviderError(f"CSV price file is empty: {fp}")

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        date_candidates = ["date", "datetime", "timestamp"]
        date_col = None
        for c in date_candidates:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            raise PriceProviderError(f"CSV missing date column (tried {date_candidates}): {fp}")

        required = ["open", "close"]
        for r in required:
            if r not in df.columns:
                raise PriceProviderError(f"CSV missing required column '{r}': {fp}. Found cols={df.columns.tolist()}")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col]).copy()
        df[date_col] = df[date_col].dt.normalize()
        df = df.set_index(date_col).sort_index()

        # Keep standard columns if present
        keep = []
        for c in ["open", "high", "low", "close", "adj_close", "adjclose", "volume"]:
            if c in df.columns:
                keep.append(c)
        df = df[keep].copy()

        # normalize adj close naming
        if "adjclose" in df.columns and "adj_close" not in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})

        # Ensure numeric
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "close"])

        return df

    def get_ohlc(
        self,
        symbols: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()

        for sym in symbols:
            df = self._load_one(sym)
            df = df.loc[(df.index >= start) & (df.index <= end)].copy()
            if df.empty:
                raise PriceProviderError(f"No OHLC rows for symbol={sym} in range {start.date()}..{end.date()}")
            out[sym] = df
        return out
