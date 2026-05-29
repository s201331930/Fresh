"""Market-data acquisition with caching and a reproducible synthetic fallback.

The loader tries, in order:
    1. A local parquet/CSV cache (fast, offline-friendly).
    2. A live download from Yahoo Finance via ``yfinance``.
    3. A reproducible synthetic geometric-Brownian-motion series (only if
       enabled in config) so the whole pipeline still runs when offline.

Every series is normalised to OHLCV columns with a tz-naive ``DatetimeIndex``
and adjusted prices (splits/dividends folded in) so back-tests are honest.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class PriceData:
    """Container holding aligned OHLCV frames for a universe of symbols.

    ``frames[symbol]`` is a DataFrame indexed by date with columns ``OHLCV``.
    ``synthetic`` lists any symbols that were synthesised rather than downloaded.
    """

    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    synthetic: List[str] = field(default_factory=list)

    @property
    def symbols(self) -> List[str]:
        return list(self.frames.keys())

    def close_matrix(self) -> pd.DataFrame:
        """Wide matrix of adjusted closes (date x symbol)."""
        cols = {s: df["close"] for s, df in self.frames.items()}
        return pd.DataFrame(cols).sort_index()

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        return self.frames[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.frames


class DataLoader:
    def __init__(
        self,
        cache_dir: str = "data/cache",
        allow_synthetic_fallback: bool = True,
        synthetic_seed: int = 7,
    ) -> None:
        self.cache_dir = cache_dir
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.synthetic_seed = synthetic_seed
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(
        self,
        symbols: List[str],
        start: str,
        end: Optional[str] = None,
    ) -> PriceData:
        symbols = list(dict.fromkeys(symbols))  # dedupe, preserve order
        end = end or datetime.utcnow().strftime("%Y-%m-%d")
        out = PriceData()
        for sym in symbols:
            df = self._load_one(sym, start, end)
            if df is not None and len(df) > 50:
                out.frames[sym] = df
            else:
                print(f"  [data] WARNING: insufficient data for {sym}, skipping")
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _cache_path(self, symbol: str) -> str:
        safe = symbol.replace("/", "_").replace("=", "_").replace("^", "_")
        return os.path.join(self.cache_dir, f"{safe}.csv")

    def _load_one(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        cached = self._read_cache(symbol)
        if cached is not None and self._covers(cached, start, end):
            return self._slice(cached, start, end)

        downloaded = self._download(symbol, start, end)
        if downloaded is not None and len(downloaded) > 50:
            self._write_cache(symbol, downloaded)
            return self._slice(downloaded, start, end)

        if cached is not None and len(cached) > 50:
            print(f"  [data] using stale cache for {symbol} (download failed)")
            return self._slice(cached, start, end)

        if self.allow_synthetic_fallback:
            print(f"  [data] synthesising {symbol} (no download/cache available)")
            return self._synthesize(symbol, start, end)

        return None

    @staticmethod
    def _covers(df: pd.DataFrame, start: str, end: str) -> bool:
        if df.empty:
            return False
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        # Allow a 5-day slack at the recent edge for weekends/holidays.
        return df.index.min() <= s and df.index.max() >= (e - pd.Timedelta(days=5))

    @staticmethod
    def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()

    def _read_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            return df[OHLCV].sort_index()
        except Exception:
            return None

    def _write_cache(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            out = df.copy()
            out.index.name = "date"
            out.to_csv(self._cache_path(symbol))
        except Exception as exc:  # pragma: no cover - best effort
            print(f"  [data] cache write failed for {symbol}: {exc}")

    def _download(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception:
            return None
        for attempt in range(3):
            try:
                raw = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.rename(columns=str.lower)
                raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
                cols = [c for c in OHLCV if c in raw.columns]
                df = raw[cols].dropna(how="all")
                if "volume" not in df.columns:
                    df["volume"] = 0.0
                return df[OHLCV].astype(float)
            except Exception:
                continue
        return None

    def _synthesize(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Reproducible GBM with mild regime shifts; deterministic per symbol."""
        idx = pd.bdate_range(start=start, end=end)
        n = len(idx)
        seed = self.synthetic_seed + (abs(hash(symbol)) % 100_000)
        rng = np.random.default_rng(seed)

        mu = rng.uniform(0.03, 0.12) / 252.0          # annual drift -> daily
        sigma = rng.uniform(0.12, 0.45) / np.sqrt(252.0)
        # Inject a couple of regime shifts to create drawdowns/recoveries.
        regimes = rng.normal(0, sigma, n)
        for _ in range(3):
            start_i = rng.integers(0, max(1, n - 60))
            length = rng.integers(20, 60)
            regimes[start_i : start_i + length] -= sigma * rng.uniform(1.0, 3.0)

        rets = mu + regimes
        close = 100.0 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, sigma, n))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]])
        vol = rng.integers(1_000_000, 10_000_000, n).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )
        df.index.name = "date"
        return df
