"""Combine multiple strategy sleeves into one leverage-free book.

Each strategy is back-tested standalone on the full notional to obtain a clean
return stream. The portfolio return on any day is a convex combination
(weights >= 0, sum to 1) of the sleeve returns, which keeps the aggregate book
leverage-free by construction. An optional inverse-volatility ("vol parity")
overlay rebalances the weights toward whichever sleeves are currently calmest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .backtester import BacktestResult


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    exposure: pd.Series
    sleeves: Dict[str, BacktestResult] = field(default_factory=dict)
    sleeve_returns: pd.DataFrame = field(default_factory=pd.DataFrame)


class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        base_weights: Dict[str, float],
        use_vol_parity: bool = True,
        vol_parity_lookback: int = 90,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.base_weights = base_weights
        self.use_vol_parity = use_vol_parity
        self.vol_lookback = vol_parity_lookback

    def combine(self, results: Dict[str, BacktestResult]) -> PortfolioResult:
        names = [n for n in results]
        # Align all sleeve return streams on a common calendar.
        ret_df = pd.DataFrame({n: results[n].returns for n in names}).fillna(0.0)
        expo_df = pd.DataFrame(
            {n: results[n].exposure for n in names}
        ).reindex(ret_df.index).fillna(0.0)

        base = np.array([max(self.base_weights.get(n, 0.0), 0.0) for n in names], dtype=float)
        if base.sum() <= 0:
            base = np.ones(len(names))
        base = base / base.sum()

        if self.use_vol_parity:
            vol = ret_df.rolling(self.vol_lookback, min_periods=20).std().shift(1)
            inv = 1.0 / vol.replace(0.0, np.nan)
            raw = inv.mul(base, axis=1)
            weights = raw.div(raw.sum(axis=1), axis=0)
            # Before enough history exists, fall back to static base weights.
            weights = weights.fillna(pd.Series(base, index=names))
        else:
            weights = pd.DataFrame(
                np.tile(base, (len(ret_df), 1)), index=ret_df.index, columns=names
            )

        port_ret = (weights * ret_df).sum(axis=1)
        equity = self.initial_capital * (1 + port_ret).cumprod()
        exposure = (weights * expo_df).sum(axis=1)

        return PortfolioResult(
            equity=equity.rename("portfolio"),
            returns=port_ret.rename("portfolio"),
            weights=weights,
            exposure=exposure.rename("exposure"),
            sleeves=results,
            sleeve_returns=ret_df,
        )
