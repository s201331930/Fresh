"""Strategy base class and shared technical indicators.

A Strategy is a pure signal generator: given price data it returns, per symbol,
a DataFrame of boolean ``entry``/``exit`` events (and an ``atr`` column when the
engine needs it for ATR-based stops/sizing). All risk management and execution
lives in the engine, keeping strategies small, testable and comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data import PriceData
from ..engine.backtester import RiskConfig


# --------------------------------------------------------------------------- #
# Indicators (vectorised, no look-ahead)
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder-style smoothing.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def momentum(series: pd.Series, lookback: int) -> pd.Series:
    return series / series.shift(lookback) - 1.0


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Strategy(ABC):
    name: str = "strategy"

    def __init__(self, universe: List[str], **params) -> None:
        self.universe = list(dict.fromkeys(universe))
        self.params = params

    @abstractmethod
    def generate(self, data: PriceData) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: DataFrame[entry, exit, (atr)]}``."""

    @abstractmethod
    def risk_config(self) -> RiskConfig:
        ...

    def tradable(self, data: PriceData) -> List[str]:
        return [s for s in self.universe if s in data]

    @staticmethod
    def _empty_signals(index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            {"entry": False, "exit": False}, index=index
        )
