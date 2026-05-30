import numpy as np
import pandas as pd

from quant.data import PriceData
from quant.optimization import (
    AllocationConfig, TiltAllocator, signal_components, information_coefficient, vix_regime,
)


def _ohlcv(seed, n=900, drift=0.0003, vol=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)
    return df


def _universe(k=4):
    return PriceData(frames={f"A{i}": _ohlcv(i) for i in range(k)})


def test_signal_components_shapes():
    data = _universe(4)
    cal = data["A0"].index
    comps = signal_components(data, cal)
    for key in ["trend", "mom", "rev1", "rev3", "rsi", "dip", "vwap", "volume"]:
        assert key in comps
        assert comps[key].shape == (len(cal), 4)


def test_information_coefficient_detects_perfect_signal():
    data = _universe(5)
    prices = pd.DataFrame({s: data[s]["close"] for s in data.symbols})
    fwd = prices.shift(-21) / prices - 1.0
    ic_pos = information_coefficient(fwd, prices, horizon=21, step=21)
    ic_neg = information_coefficient(-fwd, prices, horizon=21, step=21)
    assert ic_pos["ic"] > 0.9       # perfect foresight -> IC ~ +1
    assert ic_neg["ic"] < -0.9      # inverted -> IC ~ -1


def test_tilt_allocator_leverage_free_long_only():
    data = _universe(4)
    cal = data["A0"].index
    prices = pd.DataFrame({s: data[s]["close"] for s in data.symbols})
    comps = signal_components(data, cal)
    score = comps["rev1"] + 0.5 * comps["rev3"]
    cfg = AllocationConfig(use_trend_filter=False, use_vol_target=True, target_vol=0.10,
                           per_asset_trail=None, circuit_breaker=None, min_history=120)
    res = TiltAllocator(cfg, score, tilt_strength=1.0, max_weight=0.40,
                        initial_capital=1_000_000).run(prices)
    assert (res.exposure <= 1.0 + 1e-6).all()
    assert (res.weights.values >= -1e-9).all()
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-6).all()
    assert len(res.equity) == len(cal)


def test_vix_regime_throttles_when_elevated():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=500)
    # Calm regime (noisy ~14) then a sustained spike ramping to ~50.
    calm = 14 + rng.normal(0, 1.5, 300)
    spike = np.linspace(20, 50, 200)
    vix = pd.Series(np.concatenate([calm, spike]), index=idx)
    mult = vix_regime(vix, idx, floor=0.5)
    assert mult.iloc[200] > 0.9     # mid-calm -> near full exposure
    assert mult.iloc[-1] <= 0.6     # elevated -> throttled toward the floor
    assert (mult >= 0.5 - 1e-9).all() and (mult <= 1.0 + 1e-9).all()
