import numpy as np
import pandas as pd

from quant.data import PriceData
from quant.engine import Backtester, RiskConfig


def _frame(closes, highs=None, lows=None, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    closes = np.array(closes, dtype=float)
    opens = np.array(opens, dtype=float) if opens is not None else closes.copy()
    highs = np.array(highs, dtype=float) if highs is not None else np.maximum(opens, closes)
    lows = np.array(lows, dtype=float) if lows is not None else np.minimum(opens, closes)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.ones(len(closes)) * 1e6},
        index=idx,
    )


def _signals(index, entry_idx=(), exit_idx=()):
    sig = pd.DataFrame({"entry": False, "exit": False}, index=index)
    for i in entry_idx:
        sig.iloc[i, sig.columns.get_loc("entry")] = True
    for i in exit_idx:
        sig.iloc[i, sig.columns.get_loc("exit")] = True
    return sig


def test_take_profit_exit():
    # Price rallies; 10% take-profit should trigger.
    closes = [100, 101, 102, 103, 115, 116]
    highs = [100, 101, 102, 103, 116, 117]
    df = _frame(closes, highs=highs)
    data = PriceData(frames={"AAA": df})
    sig = _signals(df.index, entry_idx=[0])
    risk = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.10, max_positions=1, sizing="equal")
    res = Backtester(initial_capital=10_000, commission_bps=0, slippage_bps=0).run(
        data, {"AAA": sig}, risk, name="tp")
    assert len(res.trades) == 1
    assert res.trades[0]["exit_reason"] == "target"
    assert res.trades[0]["pnl"] > 0


def test_stop_loss_exit():
    closes = [100, 99, 98, 90, 88]
    lows = [100, 98, 95, 90, 88]
    df = _frame(closes, lows=lows)
    data = PriceData(frames={"BBB": df})
    sig = _signals(df.index, entry_idx=[0])
    risk = RiskConfig(stop_loss_pct=0.05, max_positions=1, sizing="equal")
    res = Backtester(initial_capital=10_000, commission_bps=0, slippage_bps=0).run(
        data, {"BBB": sig}, risk, name="sl")
    assert len(res.trades) == 1
    assert res.trades[0]["exit_reason"] == "stop"
    assert res.trades[0]["pnl"] < 0


def test_trailing_stop_locks_in_gain():
    # Rally to 130 then fall back; a 10% trailing stop exits well above entry.
    closes = [100, 110, 120, 130, 118, 110]
    highs = [100, 111, 121, 131, 119, 111]
    lows = [100, 109, 119, 129, 116, 109]
    df = _frame(closes, highs=highs, lows=lows)
    data = PriceData(frames={"CCC": df})
    sig = _signals(df.index, entry_idx=[0])
    risk = RiskConfig(stop_loss_pct=0.20, trailing_pct=0.10, max_positions=1, sizing="equal")
    res = Backtester(initial_capital=10_000, commission_bps=0, slippage_bps=0).run(
        data, {"CCC": sig}, risk, name="trail")
    assert len(res.trades) == 1
    assert res.trades[0]["pnl"] > 0  # exited in profit thanks to trailing stop


def test_leverage_free_invariant():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 400)))
    df = _frame(closes)
    data = PriceData(frames={"DDD": df})
    # Enter often; engine must never lever up or go cash-negative.
    sig = _signals(df.index, entry_idx=list(range(0, 400, 5)))
    risk = RiskConfig(stop_loss_pct=0.05, max_positions=1, sizing="equal")
    res = Backtester(initial_capital=10_000).run(data, {"DDD": sig}, risk, name="lev")
    assert (res.exposure <= 1.0 + 1e-6).all()
    assert (res.cash >= -1e-6).all()


def test_no_lookahead_entry_fills_next_open():
    # Entry signal on day 0 close -> fill at day 1 open.
    closes = [100, 100, 100]
    opens = [100, 105, 105]
    df = _frame(closes, opens=opens, highs=[100, 105, 105], lows=[100, 100, 100])
    data = PriceData(frames={"EEE": df})
    sig = _signals(df.index, entry_idx=[0], exit_idx=[1])
    risk = RiskConfig(stop_loss_pct=0.50, max_positions=1, sizing="equal")
    res = Backtester(initial_capital=10_000, commission_bps=0, slippage_bps=0).run(
        data, {"EEE": sig}, risk, name="la")
    # Entry price should be day-1 open (105), not day-0 close (100).
    assert len(res.trades) == 1
    assert abs(res.trades[0]["entry_price"] - 105) < 1e-6
