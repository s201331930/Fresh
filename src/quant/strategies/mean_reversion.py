"""Mean Reversion -- Connors-style RSI(2) dip buying inside an uptrend.

Return driver: short-term oversold bounces. We only buy weakness ABOVE the
long-term trend (regime filter) so we are "buying dips in a bull", not catching
falling knives. Holding periods are a handful of days, which makes the P&L
stream structurally decorrelated from the slow trend-following sleeve.

Entry : close > 200d MA  AND  RSI(2) < entry threshold (deeply oversold).
Exit  : RSI(2) > exit threshold (snap-back). Engine adds a hard % stop, a %
        take-profit and a time stop.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..data import PriceData
from ..engine.backtester import RiskConfig
from .base import Strategy, rsi, sma


class MeanReversion(Strategy):
    name = "mean_reversion"

    def generate(self, data: PriceData) -> Dict[str, pd.DataFrame]:
        p = self.params
        trend_w = p.get("trend_filter_ma", 200)
        rsi_p = p.get("rsi_period", 2)
        rsi_entry = p.get("rsi_entry", 10.0)
        rsi_exit = p.get("rsi_exit", 60.0)

        out: Dict[str, pd.DataFrame] = {}
        for s in self.tradable(data):
            df = data[s]
            close = df["close"]
            trend = sma(close, trend_w)
            r = rsi(close, rsi_p)

            entry = (close > trend) & (r < rsi_entry)
            exit_ = r > rsi_exit

            sig = pd.DataFrame(index=df.index)
            sig["entry"] = entry.fillna(False)
            sig["exit"] = exit_.fillna(False)
            out[s] = sig
        return out

    def risk_config(self) -> RiskConfig:
        p = self.params
        return RiskConfig(
            stop_loss_pct=p.get("stop_loss_pct", 0.08),
            take_profit_pct=p.get("take_profit_pct", 0.10),
            max_holding_days=p.get("max_holding_days", 10),
            sizing="equal",
            max_positions=p.get("max_positions", 4),
        )
