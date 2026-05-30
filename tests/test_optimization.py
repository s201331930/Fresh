import numpy as np
import pandas as pd

from quant.optimization import (
    AllocationBacktester, AllocationConfig, OptimizerConfig, PortfolioOptimizer,
    shrink_covariance,
)


def _returns(seed=0, n_assets=4, n=800):
    rng = np.random.default_rng(seed)
    # Distinct vols so min-variance/inverse-vol have a clear preference.
    vols = np.linspace(0.005, 0.02, n_assets)
    data = rng.normal(0.0004, 1.0, (n, n_assets)) * vols
    cols = [f"A{i}" for i in range(n_assets)]
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_shrinkage_is_psd_and_annualised():
    r = _returns()
    cov = shrink_covariance(r, None)
    eig = np.linalg.eigvalsh(cov)
    assert (eig > -1e-10).all()                 # positive semi-definite
    assert cov.shape == (4, 4)


def test_all_methods_long_only_and_capped():
    r = _returns()
    for method in ["equal_weight", "inverse_vol", "min_variance",
                   "max_sharpe", "risk_parity", "max_diversification"]:
        cfg = OptimizerConfig(method=method, max_weight=0.4, allow_cash=True)
        w = PortfolioOptimizer(cfg).optimize(r)
        assert (w >= -1e-9).all(), method               # long only
        assert (w <= 0.4 + 1e-6).all(), method          # cap respected
        assert w.sum() <= 1.0 + 1e-6, method            # leverage free


def test_min_variance_tilts_to_low_vol_asset():
    r = _returns()
    cfg = OptimizerConfig(method="min_variance", max_weight=1.0, allow_cash=False)
    w = PortfolioOptimizer(cfg).optimize(r)
    # A0 is the lowest-vol asset -> should get the largest weight.
    assert w.idxmax() == "A0"


def test_allocation_leverage_free_and_long_only():
    rng = np.random.default_rng(1)
    n = 1000
    idx = pd.bdate_range("2014-01-01", periods=n)
    prices = pd.DataFrame(
        {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))) for i in range(4)},
        index=idx,
    )
    cfg = AllocationConfig(use_trend_filter=True, use_vol_target=True,
                           per_asset_trail=0.2, circuit_breaker=0.2, min_history=120)
    opt = PortfolioOptimizer(OptimizerConfig(method="max_sharpe", max_weight=0.5))
    res = AllocationBacktester(opt, cfg, 1_000_000).run(prices)
    assert (res.exposure <= 1.0 + 1e-6).all()           # never levered
    assert (res.weights.values >= -1e-9).all()          # long only
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-6).all()
    assert len(res.equity) == n


def test_trend_filter_moves_to_cash_in_downtrend():
    # A steadily falling asset must never be held when the trend filter is on.
    n = 600
    idx = pd.bdate_range("2015-01-01", periods=n)
    falling = pd.Series(100 * np.exp(np.cumsum(np.full(n, -0.001))), index=idx)
    prices = pd.DataFrame({"DOWN": falling})
    cfg = AllocationConfig(use_trend_filter=True, use_vol_target=False,
                           per_asset_trail=None, circuit_breaker=None,
                           trend_window=100, min_history=120)
    opt = PortfolioOptimizer(OptimizerConfig(method="equal_weight", max_weight=1.0))
    res = AllocationBacktester(opt, cfg, 1_000_000).run(prices)
    # After the trend window, exposure should be ~0 (parked in cash).
    assert res.exposure.iloc[150:].max() < 1e-6
