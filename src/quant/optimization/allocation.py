"""Allocation backtester for an optimised, rules-driven multi-asset book.

This is the execution layer that turns optimiser weights into a tradable,
risk-managed strategy. On top of the periodic mean-variance optimisation it
layers the tools the mandate calls for:

    * **Rebalancing** to optimiser targets on a fixed schedule (monthly default).
    * **Trend / regime filter** -- an asset is only eligible while it trades
      above its long-term moving average; otherwise that capital goes to cash.
      This is the "stay in cash" switch that slashes drawdowns.
    * **Volatility targeting** -- total invested weight is scaled so the ex-ante
      portfolio volatility hits a target (scaled DOWN only -> leverage free).
    * **Per-asset trailing stop** -- a held name is liquidated to cash if it
      falls a set % from its high-water mark since entry.
    * **Portfolio circuit breaker** -- the whole book moves to cash if equity
      draws down beyond a threshold from its peak, re-risking at the next
      rebalance once the trend has repaired.
    * **Dynamic universe** -- assets enter the optimisation only once they have
      enough listed history, so newly-listed holdings (e.g. recent REITs) are
      added automatically without breaking a 15-year backtest.

Leverage-free and long-only are guaranteed: weights are non-negative and total
invested weight never exceeds 1 (the remainder earns the cash rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .optimizer import PortfolioOptimizer, OptimizerConfig, shrink_covariance

TRADING_DAYS = 252


@dataclass
class AllocationConfig:
    rebalance: str = "ME"            # pandas resample alias (month-end)
    lookback_days: int = 504         # trailing window for return/cov estimation
    min_history: int = 252           # min observations to include an asset
    # Trend / regime filter
    use_trend_filter: bool = True
    trend_window: int = 200
    # Volatility targeting
    use_vol_target: bool = True
    target_vol: float = 0.10
    vol_lookback: int = 90
    max_total_exposure: float = 1.0  # leverage-free cap
    # Risk management
    per_asset_trail: Optional[float] = 0.20
    circuit_breaker: Optional[float] = 0.20
    # Frictions
    commission_bps: float = 2.0
    slippage_bps: float = 5.0
    rf: float = 0.02


@dataclass
class AllocationResult:
    equity: pd.Series
    weights: pd.DataFrame            # realised weights (date x asset)
    exposure: pd.Series              # invested fraction (0..1)
    turnover: pd.Series              # traded notional / equity per rebalance
    events: List[dict] = field(default_factory=list)  # stops / circuit breaks
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


class AllocationBacktester:
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        config: AllocationConfig,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.opt = optimizer
        self.cfg = config
        self.initial_capital = float(initial_capital)
        self.commission = config.commission_bps / 1e4
        self.slippage = config.slippage_bps / 1e4

    # ------------------------------------------------------------------ #
    def run(self, prices: pd.DataFrame) -> AllocationResult:
        cfg = self.cfg
        prices = prices.sort_index()
        assets = list(prices.columns)
        rets = prices.pct_change()
        trend = prices.rolling(cfg.trend_window, min_periods=cfg.trend_window).mean()

        calendar = prices.index
        rebal_dates = self._rebalance_dates(calendar, cfg.rebalance)

        cash = self.initial_capital
        units = {a: 0.0 for a in assets}
        hw = {a: 0.0 for a in assets}            # per-asset high-water price
        stopped: set = set()                      # assets stopped until next rebalance
        risk_off = False
        peak_equity = self.initial_capital

        daily_rf = (1 + cfg.rf) ** (1 / TRADING_DAYS) - 1 if cfg.rf else 0.0

        eq_curve = np.empty(len(calendar))
        expo_curve = np.empty(len(calendar))
        w_hist = np.zeros((len(calendar), len(assets)))
        tgt_hist = np.full((len(calendar), len(assets)), np.nan)
        turnover_rows: Dict[pd.Timestamp, float] = {}
        events: List[dict] = []

        price_arr = prices.values
        col_idx = {a: j for j, a in enumerate(assets)}

        def cur_prices(i):
            return {a: price_arr[i, col_idx[a]] for a in assets}

        def equity_at(i, cash_val):
            p = cur_prices(i)
            val = cash_val
            for a in assets:
                if units[a] > 0 and not np.isnan(p[a]):
                    val += units[a] * p[a]
            return val

        for i, date in enumerate(calendar):
            if daily_rf:
                cash *= 1 + daily_rf
            p = cur_prices(i)

            # ---- daily per-asset trailing stops ----
            if cfg.per_asset_trail:
                for a in assets:
                    if units[a] > 0 and not np.isnan(p[a]):
                        hw[a] = max(hw[a], p[a])
                        if p[a] <= hw[a] * (1 - cfg.per_asset_trail):
                            cash += self._sell(units[a], p[a])
                            units[a] = 0.0
                            stopped.add(a)
                            events.append({"date": date, "asset": a, "type": "trailing_stop",
                                           "price": round(p[a], 4)})

            # ---- portfolio circuit breaker ----
            equity = equity_at(i, cash)
            peak_equity = max(peak_equity, equity)
            if (cfg.circuit_breaker and not risk_off
                    and equity <= peak_equity * (1 - cfg.circuit_breaker)):
                for a in assets:
                    if units[a] > 0 and not np.isnan(p[a]):
                        cash += self._sell(units[a], p[a])
                        units[a] = 0.0
                risk_off = True
                events.append({"date": date, "asset": "PORTFOLIO", "type": "circuit_breaker",
                               "drawdown": round(equity / peak_equity - 1, 4)})

            # ---- rebalance ----
            if date in rebal_dates:
                risk_off = False
                stopped = set()
                target = self._target_weights(date, i, prices, rets, trend, p)
                equity = equity_at(i, cash)
                traded = 0.0
                for a in assets:
                    pa = p[a]
                    if np.isnan(pa) or pa <= 0:
                        continue
                    tgt_val = equity * target.get(a, 0.0)
                    cur_val = units[a] * pa
                    delta_val = tgt_val - cur_val
                    if abs(delta_val) < 1e-6:
                        continue
                    traded += abs(delta_val)
                    if delta_val > 0:        # buy
                        cost = delta_val * (1 + self.commission + self.slippage)
                        if cost > cash:
                            delta_val = cash / (1 + self.commission + self.slippage)
                            cost = cash
                        units[a] += delta_val / pa
                        cash -= cost
                        if hw[a] <= 0:
                            hw[a] = pa
                        else:
                            hw[a] = max(hw[a], pa)
                    else:                    # sell
                        units[a] += delta_val / pa  # delta_val negative
                        cash += (-delta_val) * (1 - self.commission - self.slippage)
                        if units[a] <= 1e-9:
                            units[a] = 0.0
                            hw[a] = 0.0
                for a in assets:
                    tgt_hist[i, col_idx[a]] = target.get(a, 0.0)
                turnover_rows[date] = traded / equity if equity > 0 else 0.0

            # ---- mark to market ----
            equity = equity_at(i, cash)
            eq_curve[i] = equity
            invested = 0.0
            for a in assets:
                if units[a] > 0 and not np.isnan(p[a]):
                    val = units[a] * p[a]
                    w_hist[i, col_idx[a]] = val / equity if equity > 0 else 0.0
                    invested += val
            expo_curve[i] = invested / equity if equity > 0 else 0.0

        equity_s = pd.Series(eq_curve, index=calendar, name="equity")
        weights_df = pd.DataFrame(w_hist, index=calendar, columns=assets)
        target_df = pd.DataFrame(tgt_hist, index=calendar, columns=assets).dropna(how="all")
        expo_s = pd.Series(expo_curve, index=calendar, name="exposure")
        turn_s = pd.Series(turnover_rows, name="turnover").sort_index()
        return AllocationResult(
            equity=equity_s, weights=weights_df, exposure=expo_s,
            turnover=turn_s, events=events, target_weights=target_df,
        )

    # ------------------------------------------------------------------ #
    def _target_weights(self, date, i, prices, rets, trend, p) -> Dict[str, float]:
        cfg = self.cfg
        assets = list(prices.columns)

        # Eligibility: enough history and a valid price today.
        eligible = []
        for a in assets:
            if np.isnan(p[a]) or p[a] <= 0:
                continue
            hist = rets[a].iloc[max(0, i - cfg.lookback_days):i + 1].dropna()
            if len(hist) < cfg.min_history:
                continue
            if cfg.use_trend_filter:
                ma = trend[a].iat[i]
                if np.isnan(ma) or p[a] < ma:   # below long-term trend -> stand aside
                    continue
            eligible.append(a)

        if not eligible:
            return {a: 0.0 for a in assets}

        window = rets[eligible].iloc[max(0, i - cfg.lookback_days):i + 1].dropna(how="any")
        if len(window) < max(40, cfg.min_history // 4):
            # Not enough overlapping history -> fall back to equal weight.
            w = pd.Series(1.0 / len(eligible), index=eligible)
        else:
            w = self.opt.optimize(window)

        # Volatility targeting (scale down only -> leverage free).
        if cfg.use_vol_target and len(window) >= cfg.vol_lookback:
            recent = rets[eligible].iloc[max(0, i - cfg.vol_lookback):i + 1].dropna(how="any")
            if len(recent) >= 20:
                cov = shrink_covariance(recent, None)
                wv = w.reindex(eligible).fillna(0.0).values
                port_vol = float(np.sqrt(max(wv @ cov @ wv, 1e-12)))
                if port_vol > 0:
                    scale = min(cfg.max_total_exposure, cfg.target_vol / port_vol)
                    w = w * scale

        # Enforce leverage-free aggregate exposure.
        total = float(w.sum())
        if total > cfg.max_total_exposure:
            w = w * (cfg.max_total_exposure / total)
        return {a: float(w.get(a, 0.0)) for a in assets}

    @staticmethod
    def _rebalance_dates(calendar: pd.DatetimeIndex, alias: str) -> set:
        s = pd.Series(1, index=calendar)
        marks = s.resample(alias).last()
        out = set()
        for d in marks.index:
            prior = calendar[calendar <= d]
            if len(prior):
                out.add(prior[-1])
        return out

    def _sell(self, units: float, price: float) -> float:
        return units * price * (1 - self.commission - self.slippage)
