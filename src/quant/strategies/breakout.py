"""Breakout -- Donchian channel breakout (turtle style).

Return driver: volatility expansion / range breakouts on trendy, higher-vol
instruments (Nasdaq, metals, energy, commodities, EM, crypto proxy). Buys new
N-day highs and exits on the opposing M-day low channel, with an ATR stop and
ATR trailing stop layered on top.

Breakout timing keys off range structure rather than moving-average slope or
oversold oscillators, so its entries cluster at different moments than the
other sleeves -- another source of decorrelation.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..data import PriceData
from ..engine.backtester import RiskConfig
from .base import Strategy, atr, rolling_high, rolling_low


class Breakout(Strategy):
    name = "breakout"

    def generate(self, data: PriceData) -> Dict[str, pd.DataFrame]:
        p = self.params
        entry_lb = p.get("entry_lookback", 55)
        exit_lb = p.get("exit_lookback", 20)
        atr_p = p.get("atr_period", 20)

        out: Dict[str, pd.DataFrame] = {}
        for s in self.tradable(data):
            df = data[s]
            close = df["close"]
            upper = rolling_high(df["high"], entry_lb).shift(1)
            lower = rolling_low(df["low"], exit_lb).shift(1)

            entry = close >= upper
            exit_ = close <= lower

            sig = pd.DataFrame(index=df.index)
            sig["entry"] = entry.fillna(False)
            sig["exit"] = exit_.fillna(False)
            sig["atr"] = atr(df, atr_p)
            out[s] = sig
        return out

    def risk_config(self) -> RiskConfig:
        p = self.params
        return RiskConfig(
            stop_atr_mult=p.get("stop_atr_mult", 2.0),
            atr_trailing_mult=p.get("atr_trailing_mult", 3.0),
            sizing="risk",
            risk_per_trade=p.get("risk_per_trade", 0.01),
            max_positions=p.get("max_positions", 5),
        )
