"""Signal-driven alpha research & tilting -- evidence-based "smart" allocation.

This module forms predictive views from classic technical/quant signals
(trend/MA, 12-1 momentum, short-term reversal, RSI, Bollinger %b, VWAP, volume)
plus a VIX macro throttle, and provides tools to (a) *measure* whether those
signals actually predict forward returns on a given universe (information
coefficient), and (b) *tilt* a diversified, risk-budgeted, long-only book toward
the signals -- overweighting conviction names and underweighting the rest.

Two allocators are provided:

    * :class:`TiltAllocator` -- diversified exp-tilt around an inverse-vol base
      (stays diversified; good when signals are weak/noisy -- the realistic case).
    * :class:`SignalAllocator` -- aggressive conviction gating that cuts weak
      names to cash and doubles down on the strong (use only if signals are
      strong and stable, which the IC test should confirm first).

Everything is causal (no look-ahead) and inherits the rebalancing / trailing-
stop / circuit-breaker / vol-target / dynamic-universe execution from
:class:`AllocationBacktester`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data import PriceData
from .allocation import AllocationBacktester, AllocationConfig, shrink_covariance


# --------------------------------------------------------------------------- #
# Indicators (vectorised, causal)
# --------------------------------------------------------------------------- #
def _sma(s, w):
    return s.rolling(w, min_periods=max(2, w // 2)).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _bollinger_pctb(close, window=20, k=2.0):
    ma = _sma(close, window)
    sd = close.rolling(window, min_periods=max(2, window // 2)).std()
    upper, lower = ma + k * sd, ma - k * sd
    width = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / width).clip(0.0, 1.0)


def _rolling_vwap(df, window=20):
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)
    pv = (typical * vol).rolling(window, min_periods=max(2, window // 2)).sum()
    vv = vol.rolling(window, min_periods=max(2, window // 2)).sum().replace(0.0, np.nan)
    return (pv / vv).fillna(_sma(typical, window))


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date, ignoring NaN (un-listed assets)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-3, 3)


def signal_components(data: PriceData, calendar: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    """Return cross-sectionally z-scored signal components (date x asset).

    Each is oriented so that *higher = more bullish for that signal's thesis*:
        trend   : price above 200d MA (trend-following)
        mom     : 12-1 month momentum (medium-term momentum)
        rev1    : 1-month reversal (overweight 1m losers)
        rev3    : 3-month reversal (overweight 3m losers)
        rsi     : RSI below 50 (oversold)
        dip     : low Bollinger %b (near lower band)
        vwap    : price above rolling VWAP (strength)
        volume  : volume expansion while above 50d MA (conviction)
    """
    assets = data.symbols

    def pa(fn):
        return pd.DataFrame({s: fn(data[s].reindex(calendar).ffill()) for s in assets}).reindex(calendar)

    raw = {
        "trend": pa(lambda df: df["close"] / _sma(df["close"], 200) - 1),
        "mom": pa(lambda df: df["close"].shift(21) / df["close"].shift(252) - 1),
        "rev1": pa(lambda df: -(df["close"] / df["close"].shift(21) - 1)),
        "rev3": pa(lambda df: -(df["close"] / df["close"].shift(63) - 1)),
        "rsi": pa(lambda df: -(_rsi(df["close"], 14) - 50.0)),
        "dip": pa(lambda df: 0.5 - _bollinger_pctb(df["close"], 20)),
        "vwap": pa(lambda df: df["close"] / _rolling_vwap(df, 20) - 1),
        "volume": pa(lambda df: (df["volume"].fillna(0.0) / _sma(df["volume"].fillna(0.0), 50)
                                 .replace(0.0, np.nan) - 1).where(df["close"] > _sma(df["close"], 50), 0.0)),
    }
    return {k: _zscore_rows(v) for k, v in raw.items()}


def realized_vol(data: PriceData, calendar: pd.DatetimeIndex, window: int = 63) -> pd.DataFrame:
    cols = {}
    for s in data.symbols:
        r = data[s].reindex(calendar).ffill()["close"].pct_change()
        cols[s] = r.rolling(window, min_periods=20).std() * np.sqrt(252)
    return pd.DataFrame(cols).reindex(calendar)


def information_coefficient(signal: pd.DataFrame, prices: pd.DataFrame,
                            horizon: int = 21, step: int = 21) -> dict:
    """Cross-sectional Spearman rank-IC of ``signal`` vs forward returns.

    Returns mean IC, its standard error, the implied t-stat and sample size.
    Sampling every ``step`` days reduces overlap-induced autocorrelation.
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    idx = prices.index[::step]
    s = signal.reindex(idx)
    f = fwd.reindex(idx)
    ics = []
    for d in idx:
        a = s.loc[d].dropna()
        b = f.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= 3 and a[common].nunique() > 1:
            ics.append(a[common].corr(b[common], method="spearman"))
    ics = pd.Series(ics).dropna()
    if len(ics) == 0:
        return {"ic": np.nan, "se": np.nan, "t": np.nan, "n": 0}
    se = ics.std() / np.sqrt(len(ics))
    return {"ic": float(ics.mean()), "se": float(se),
            "t": float(ics.mean() / se) if se > 0 else np.nan, "n": int(len(ics))}


def vix_regime(vix_close: Optional[pd.Series], calendar: pd.DatetimeIndex,
               floor: float = 0.6) -> pd.Series:
    """Exposure multiplier in [floor, 1]: throttle down when VIX is elevated.

    Full exposure below the 60th trailing-1y percentile of VIX, linearly down to
    ``floor`` at the 100th percentile (a market-stress / risk-off overlay).
    """
    if vix_close is None or vix_close.empty:
        return pd.Series(1.0, index=calendar)
    v = vix_close.reindex(calendar).ffill()
    rank = v.rolling(252, min_periods=60).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    mult = 1.0 - (1.0 - floor) * ((rank - 0.6) / 0.4).clip(0.0, 1.0)
    return mult.fillna(1.0)


# --------------------------------------------------------------------------- #
# Allocators
# --------------------------------------------------------------------------- #
class TiltAllocator(AllocationBacktester):
    """Diversified exp-tilt around an inverse-vol base (long-only, leverage-free).

    target_w_i  ∝  base_i * exp(k * score_i),  then capped & renormalised, then
    optionally vol-targeted and VIX-throttled. Staying diversified (rather than
    concentrating) is the right choice when signals are weak/noisy.
    """

    def __init__(self, config: AllocationConfig, score: pd.DataFrame,
                 tilt_strength: float = 0.7, max_weight: float = 0.40,
                 base: str = "invvol", regime: Optional[pd.Series] = None,
                 initial_capital: float = 1_000_000.0):
        super().__init__(optimizer=None, config=config, initial_capital=initial_capital)
        self.score = score
        self.k = tilt_strength
        self.max_weight = max_weight
        self.base = base
        self.regime = regime

    def _target_weights(self, date, i, prices, rets, trend, p) -> Dict[str, float]:
        cfg = self.cfg
        assets = list(prices.columns)
        elig = []
        for a in assets:
            if np.isnan(p[a]) or p[a] <= 0:
                continue
            if len(rets[a].iloc[max(0, i - cfg.lookback_days):i + 1].dropna()) < cfg.min_history:
                continue
            if cfg.use_trend_filter:
                ma = trend[a].iat[i]
                if np.isnan(ma) or p[a] < ma:
                    continue
            elig.append(a)
        if not elig:
            return {a: 0.0 for a in assets}

        win = rets[elig].iloc[max(0, i - cfg.lookback_days):i + 1].dropna(how="any")
        if len(win) >= 60:
            cov = shrink_covariance(win, None)
            vol = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
        else:
            cov = None
            vol = np.array([max(rets[a].iloc[max(0, i - 90):i + 1].std() * np.sqrt(252), 1e-3)
                            for a in elig])
        base = (1.0 / vol) if self.base == "invvol" else np.ones(len(elig))
        base = base / base.sum()

        srow = self.score.iloc[i] if i < len(self.score) else pd.Series(dtype=float)
        z = np.nan_to_num(np.array([float(srow.get(a, 0.0)) for a in elig]))
        w = base * np.exp(self.k * z)
        w = w / w.sum()
        w = np.minimum(w, self.max_weight)
        if w.sum() > 0:
            w = w / w.sum()
        wser = pd.Series(w, index=elig)

        if cfg.use_vol_target and cov is not None:
            pv = float(np.sqrt(max(wser.values @ cov @ wser.values, 1e-12)))
            if pv > 0:
                wser = wser * min(cfg.max_total_exposure, cfg.target_vol / pv)
        if self.regime is not None and i < len(self.regime):
            wser = wser * float(self.regime.iloc[i])
        total = float(wser.sum())
        if total > cfg.max_total_exposure:
            wser = wser * (cfg.max_total_exposure / total)
        return {a: float(wser.get(a, 0.0)) for a in assets}


@dataclass
class SignalConfig:
    w_trend: float = 0.30
    w_mom: float = 0.30
    w_vwap: float = 0.15
    w_volume: float = 0.10
    w_dip: float = 0.10
    w_rsi: float = 0.05


def composite_score(components: Dict[str, pd.DataFrame], cfg: SignalConfig = SignalConfig()) -> pd.DataFrame:
    """A trend/momentum-leaning composite (the 'obvious' double-down view)."""
    return (cfg.w_trend * components["trend"] + cfg.w_mom * components["mom"]
            + cfg.w_vwap * components["vwap"] + cfg.w_volume * components["volume"]
            + cfg.w_dip * components["dip"] + cfg.w_rsi * components["rsi"])


class SignalAllocator(AllocationBacktester):
    """Aggressive conviction gating: cut weak names to cash, double down on strong."""

    def __init__(self, config: AllocationConfig, score: pd.DataFrame, vol: pd.DataFrame,
                 regime: Optional[pd.Series] = None, score_threshold: float = 0.0,
                 conviction_power: float = 1.0, max_weight: float = 0.50,
                 initial_capital: float = 1_000_000.0):
        super().__init__(optimizer=None, config=config, initial_capital=initial_capital)
        self.score = score
        self.vol = vol
        self.regime = regime
        self.score_threshold = score_threshold
        self.conviction_power = conviction_power
        self.max_weight = max_weight

    def _target_weights(self, date, i, prices, rets, trend, p) -> Dict[str, float]:
        cfg = self.cfg
        assets = list(prices.columns)
        srow = self.score.iloc[i] if i < len(self.score) else pd.Series(dtype=float)
        candidates = []
        for a in assets:
            if np.isnan(p[a]) or p[a] <= 0:
                continue
            if len(rets[a].iloc[max(0, i - cfg.lookback_days):i + 1].dropna()) < cfg.min_history:
                continue
            if cfg.use_trend_filter:
                ma = trend[a].iat[i]
                if np.isnan(ma) or p[a] < ma:
                    continue
            sc = srow.get(a, np.nan)
            if np.isnan(sc) or sc <= self.score_threshold:
                continue
            candidates.append(a)
        if not candidates:
            return {a: 0.0 for a in assets}

        raw = {}
        for a in candidates:
            conv = max(float(srow.get(a, 0.0)) - self.score_threshold, 0.0) ** self.conviction_power
            v = self.vol[a].iat[i] if a in self.vol.columns else np.nan
            v = v if (not np.isnan(v) and v > 0) else 0.20
            raw[a] = conv / v
        tot = sum(raw.values())
        w = (pd.Series({a: raw[a] / tot for a in candidates}) if tot > 0
             else pd.Series({a: 1.0 / len(candidates) for a in candidates}))
        w = w.clip(upper=self.max_weight)
        if w.sum() > 0:
            w = w / w.sum()
        if cfg.use_vol_target:
            recent = rets[candidates].iloc[max(0, i - cfg.vol_lookback):i + 1].dropna(how="any")
            if len(recent) >= 20:
                cov = shrink_covariance(recent, None)
                wv = w.reindex(candidates).fillna(0.0).values
                pv = float(np.sqrt(max(wv @ cov @ wv, 1e-12)))
                if pv > 0:
                    w = w * min(cfg.max_total_exposure, cfg.target_vol / pv)
        if self.regime is not None and i < len(self.regime):
            w = w * float(self.regime.iloc[i])
        total = float(w.sum())
        if total > cfg.max_total_exposure:
            w = w * (cfg.max_total_exposure / total)
        return {a: float(w.get(a, 0.0)) for a in assets}
