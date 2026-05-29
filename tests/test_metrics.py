import numpy as np
import pandas as pd

from quant.metrics import compute_kpis, drawdown_series
from quant.metrics.kpis import cagr, sharpe_ratio


def _equity(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_cagr_doubling_over_one_year():
    eq = _equity(np.linspace(100, 200, 252))
    # ~1 year, doubled -> CAGR ~100%
    assert abs(cagr(eq) - 1.0) < 0.05


def test_drawdown_is_nonpositive_and_recovers():
    eq = _equity([100, 120, 90, 130])
    dd = drawdown_series(eq)
    assert (dd <= 1e-12).all()
    assert dd.iloc[-1] == 0.0  # new high -> no drawdown
    assert abs(dd.iloc[2] - (90 / 120 - 1)) < 1e-9


def test_sharpe_positive_for_steady_growth():
    eq = _equity(100 * (1.0005 ** np.arange(300)))
    assert sharpe_ratio(eq.pct_change().fillna(0)) > 0


def test_trade_stats_in_report():
    eq = _equity(np.linspace(100, 150, 200))
    trades = [
        {"pnl": 100, "return_pct": 0.10, "holding_days": 5},
        {"pnl": -50, "return_pct": -0.05, "holding_days": 3},
        {"pnl": 200, "return_pct": 0.20, "holding_days": 7},
    ]
    rep = compute_kpis(eq, trades=trades, name="t")
    assert rep.kpis["num_trades"] == 3
    assert abs(rep.kpis["win_rate"] - 2 / 3) < 1e-9
    # profit factor = gross win / gross loss = 300/50 = 6
    assert abs(rep.kpis["profit_factor"] - 6.0) < 1e-9
