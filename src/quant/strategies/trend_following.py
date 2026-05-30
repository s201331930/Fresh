"""Trend Following -- time-series momentum on a diversified ETF basket.

Return driver: persistent multi-month trends across equities, bonds,
commodities and real estate. The book is long the assets that are trending up
and sits in cash otherwise, so it tends to make money in sustained bull phases
and step aside during sustained declines.

Entry  : price > slow MA  AND  fast MA > slow MA  AND  N-day breakout high.
Exit   : fast MA crosses back below slow MA (trend break). The engine also
         applies a wide ATR trailing stop so winners are ridden but protected.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..data import PriceData
from ..engine.backtester import RiskConfig
from .base import Strategy, atr, rolling_high, sma


class TrendFollowing(Strategy):
    name = "trend_following"

    def generate(self, data: PriceData) -> Dict[str, pd.DataFrame]:
        p = self.params
        fast_w = p.get("fast_ma", 50)
        slow_w = p.get("slow_ma", 200)
        bo = p.get("breakout_lookback", 100)
        atr_p = p.get("atr_period", 20)

        out: Dict[str, pd.DataFrame] = {}
        for s in self.tradable(data):
            df = data[s]
            close = df["close"]
            fast = sma(close, fast_w)
            slow = sma(close, slow_w)
            hh = rolling_high(close, bo).shift(1)  # shift -> prior high, no peeking

            uptrend = (close > slow) & (fast > slow)
            entry = uptrend & (close >= hh)
            exit_ = fast < slow

            sig = pd.DataFrame(index=df.index)
            sig["entry"] = entry.fillna(False)
            sig["exit"] = exit_.fillna(False)
            sig["atr"] = atr(df, atr_p)
            out[s] = sig
        return out

    def risk_config(self) -> RiskConfig:
        p = self.params
        return RiskConfig(
            stop_atr_mult=p.get("stop_atr_mult", 3.0),
            atr_trailing_mult=p.get("atr_trailing_mult", 4.0),
            sizing="risk",
            risk_per_trade=p.get("risk_per_trade", 0.01),
            max_positions=p.get("max_positions", 6),
        )
