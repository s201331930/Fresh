"""Backtesting KPIs.

Everything a desk would want before signing off on going live:
return/risk ratios, drawdown analytics, tail risk, and trade-level statistics.
All return-based metrics work off a daily equity curve (a pandas Series indexed
by date). Trade-level metrics work off a trade blotter (list of dicts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return-series helpers
# --------------------------------------------------------------------------- #
def equity_to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in observations) spent below a prior peak."""
    dd = drawdown_series(equity)
    in_dd = dd < 0
    longest = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(drawdown_series(equity).min())
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - level) * 100))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def time_in_market(returns: pd.Series, exposure: Optional[pd.Series] = None) -> float:
    if exposure is not None and len(exposure):
        return float((exposure > 0).mean())
    return float((returns != 0).mean())


# --------------------------------------------------------------------------- #
# Trade-level statistics
# --------------------------------------------------------------------------- #
def trade_stats(trades: List[dict]) -> Dict[str, float]:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "avg_holding_days": 0.0,
        }
    df = pd.DataFrame(trades)
    pnl = df["pnl"].astype(float)
    ret = df["return_pct"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    win_rate = len(wins) / len(df)
    expectancy = pnl.mean()
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    pf = gross_win / gross_loss if gross_loss != 0 else float("inf")
    return {
        "num_trades": int(len(df)),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "payoff_ratio": float(payoff),
        "expectancy": float(expectancy),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_return_pct": float(ret.mean()),
        "best_trade_pct": float(ret.max()),
        "worst_trade_pct": float(ret.min()),
        "avg_holding_days": float(df["holding_days"].mean()) if "holding_days" in df else 0.0,
    }


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class PerformanceReport:
    name: str
    kpis: Dict[str, float] = field(default_factory=dict)
    monthly_returns: Optional[pd.DataFrame] = None
    annual_returns: Optional[pd.Series] = None

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.kpis.get(key, default)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    m = (1 + returns).resample("ME").prod() - 1
    table = m.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot_table(index="year", columns="month", values="ret")
    pivot.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
        for c in pivot.columns
    ]
    pivot["Year"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    return pivot


def annual_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return (1 + returns).resample("YE").prod() - 1


def compute_kpis(
    equity: pd.Series,
    trades: Optional[List[dict]] = None,
    rf: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
    exposure: Optional[pd.Series] = None,
    name: str = "strategy",
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceReport:
    equity = equity.dropna()
    returns = equity_to_returns(equity)

    kpis: Dict[str, float] = {}
    kpis["start"] = equity.index[0] if len(equity) else None
    kpis["end"] = equity.index[-1] if len(equity) else None
    kpis["initial_equity"] = float(equity.iloc[0]) if len(equity) else 0.0
    kpis["final_equity"] = float(equity.iloc[-1]) if len(equity) else 0.0
    kpis["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    kpis["cagr"] = cagr(equity, periods_per_year)
    kpis["volatility"] = annualized_volatility(returns, periods_per_year)
    kpis["sharpe"] = sharpe_ratio(returns, rf, periods_per_year)
    kpis["sortino"] = sortino_ratio(returns, rf, periods_per_year)
    kpis["calmar"] = calmar_ratio(equity, periods_per_year)

    dd = drawdown_series(equity)
    kpis["max_drawdown"] = float(dd.min())
    kpis["avg_drawdown"] = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    kpis["max_dd_duration_days"] = _max_drawdown_duration(equity)

    kpis["var_95"] = value_at_risk(returns, 0.95)
    kpis["cvar_95"] = conditional_var(returns, 0.95)
    kpis["skew"] = float(returns.skew())
    kpis["kurtosis"] = float(returns.kurtosis())
    kpis["best_day"] = float(returns.max()) if len(returns) else 0.0
    kpis["worst_day"] = float(returns.min()) if len(returns) else 0.0
    kpis["time_in_market"] = time_in_market(returns, exposure)

    # Benchmark-relative analytics.
    if benchmark_returns is not None and len(benchmark_returns) > 2:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) > 2:
            r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
            var_b = b.var(ddof=0)
            beta = float(np.cov(r, b, ddof=0)[0, 1] / var_b) if var_b > 0 else 0.0
            ann_r = r.mean() * periods_per_year
            ann_b = b.mean() * periods_per_year
            kpis["beta"] = beta
            kpis["alpha"] = float(ann_r - (rf + beta * (ann_b - rf)))
            kpis["correlation_to_benchmark"] = float(r.corr(b))

    if trades is not None:
        kpis.update(trade_stats(trades))

    report = PerformanceReport(name=name, kpis=kpis)
    report.monthly_returns = monthly_return_table(returns)
    report.annual_returns = annual_returns(returns)
    return report
