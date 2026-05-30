"""Dual Momentum -- cross-sectional + absolute momentum asset rotation.

Return driver: relative strength across asset classes (equities, intl, EM,
gold, bonds, REITs, commodities) combined with an absolute-momentum (trend)
switch into short T-bills when nothing is going up. This is Gary Antonacci's
dual-momentum idea generalised to the top-N assets.

Mechanics: at each month end, rank the universe by trailing momentum. Hold the
top-N names *whose own momentum is positive*; any slot whose candidate has
negative momentum is parked in the safe asset (cash proxy). Between rebalances
a protective trailing stop defends open positions.

This sleeve trades monthly and holds only a couple of names, giving it a very
different P&L cadence from the daily trend / mean-reversion / breakout sleeves.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..data import PriceData
from ..engine.backtester import RiskConfig
from .base import Strategy


class DualMomentum(Strategy):
    name = "dual_momentum"

    def _all_symbols(self) -> List[str]:
        safe = self.params.get("safe_asset")
        syms = list(self.universe)
        if safe and safe not in syms:
            syms.append(safe)
        return syms

    def generate(self, data: PriceData) -> Dict[str, pd.DataFrame]:
        p = self.params
        lookback_m = p.get("lookback_months", 6)
        top_n = p.get("top_n", 2)
        safe = p.get("safe_asset")

        risk_syms = [s for s in self.universe if s in data]
        safe_ok = safe in data if safe else False

        # Build aligned monthly closes for the risk universe.
        closes = pd.DataFrame({s: data[s]["close"] for s in risk_syms}).sort_index()
        if closes.empty:
            return {}
        lookback_days = max(21 * lookback_m, 21)
        month_ends = closes.resample("ME").last().index
        # Map each desired month-end to the last actual trading day on/before it.
        rebal_dates = []
        for me in month_ends:
            prior = closes.index[closes.index <= me]
            if len(prior):
                rebal_dates.append(prior[-1])
        rebal_dates = pd.DatetimeIndex(sorted(set(rebal_dates)))

        # Momentum = trailing return over the lookback window.
        mom = closes / closes.shift(lookback_days) - 1.0

        # Determine the target basket at each rebalance.
        targets_by_date: Dict[pd.Timestamp, List[str]] = {}
        for d in rebal_dates:
            if d not in mom.index:
                continue
            row = mom.loc[d].dropna()
            if row.empty:
                targets_by_date[d] = [safe] if safe_ok else []
                continue
            ranked = row.sort_values(ascending=False)
            chosen: List[str] = []
            for sym in ranked.index[:top_n]:
                if ranked[sym] > 0:
                    chosen.append(sym)
                elif safe_ok:
                    chosen.append(safe)
            # Pad remaining slots with the safe asset.
            while len(chosen) < top_n and safe_ok:
                chosen.append(safe)
            targets_by_date[d] = chosen

        all_syms = risk_syms + ([safe] if safe_ok else [])
        all_syms = list(dict.fromkeys(all_syms))
        sigs: Dict[str, pd.DataFrame] = {
            s: pd.DataFrame({"entry": False, "exit": False}, index=data[s].index)
            for s in all_syms
        }

        prev: set = set()
        for d in rebal_dates:
            if d not in targets_by_date:
                continue
            tgt = set(targets_by_date[d])
            for s in tgt - prev:           # newly added -> enter
                if s in sigs and d in sigs[s].index:
                    sigs[s].at[d, "entry"] = True
            for s in prev - tgt:           # dropped -> exit
                if s in sigs and d in sigs[s].index:
                    sigs[s].at[d, "exit"] = True
            prev = tgt
        return sigs

    def risk_config(self) -> RiskConfig:
        p = self.params
        top_n = p.get("top_n", 2)
        return RiskConfig(
            stop_loss_pct=None,  # safety-net hard stop applied by engine
            trailing_pct=p.get("trailing_pct", 0.15),
            sizing="equal",
            max_positions=top_n + 1,  # room for the safe-asset slot(s)
        )
