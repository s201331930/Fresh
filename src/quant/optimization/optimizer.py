"""Mathematical portfolio optimisation (long-only, leverage-free).

Implements the classic mean-variance toolkit plus more robust alternatives that
behave better out-of-sample, where naive Markowitz notoriously over-fits noisy
inputs:

    * ``equal_weight``       -- 1/N (estimation-free baseline)
    * ``inverse_vol``        -- weights proportional to 1/volatility
    * ``min_variance``       -- minimise portfolio variance
    * ``max_sharpe``         -- maximise the ex-ante Sharpe ratio (tangency)
    * ``risk_parity``        -- equal risk contribution (ERC)
    * ``max_diversification``-- maximise the diversification ratio

Two estimation-error defences are built in:
    1. Ledoit-Wolf style *covariance shrinkage* toward a constant-correlation
       target, which stabilises the inverse covariance used by the optimisers.
    2. Hard *position caps* and an optional *cash floor*, so no single noisy
       estimate can dominate the book.

All optimisers respect: ``w >= 0`` (long only), ``sum(w) <= 1`` (leverage free,
remainder is cash), and ``w <= max_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def shrink_covariance(returns: pd.DataFrame, intensity: Optional[float] = None) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    If ``intensity`` is None a simple data-driven intensity is used; otherwise
    the supplied value in [0, 1] is applied. Returns an annualised covariance
    matrix that is symmetric positive-definite.
    """
    X = returns.values
    t, n = X.shape
    if t < 2 or n == 0:
        return np.eye(max(n, 1)) * 1e-6

    sample = np.cov(X, rowvar=False, ddof=1)
    if n == 1:
        cov = np.atleast_2d(sample)
        return cov * TRADING_DAYS

    var = np.diag(sample)
    std = np.sqrt(np.clip(var, 1e-12, None))
    corr = sample / np.outer(std, std)
    # Constant-correlation target: average off-diagonal correlation.
    off = corr[~np.eye(n, dtype=bool)]
    r_bar = float(np.mean(off)) if off.size else 0.0
    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)

    if intensity is None:
        # Heuristic intensity: more shrinkage with fewer observations.
        intensity = float(np.clip(n / (t + n), 0.1, 0.9))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    shrunk = intensity * target + (1 - intensity) * sample
    shrunk += np.eye(n) * 1e-10  # numerical PD guard
    return shrunk * TRADING_DAYS


@dataclass
class OptimizerConfig:
    method: str = "max_sharpe"
    max_weight: float = 0.35          # per-asset cap
    min_weight: float = 0.0           # long only
    allow_cash: bool = True           # sum(w) <= 1 instead of == 1
    rf: float = 0.02
    shrinkage: Optional[float] = None
    # Expected-return estimator: "mean" | "ewm" | "capm_blend"
    mu_estimator: str = "ewm"
    ewm_halflife: int = 126
    # Blend toward the grand mean to tame outliers (0 = raw, 1 = all grand mean).
    mu_shrink: float = 0.5


class PortfolioOptimizer:
    def __init__(self, config: OptimizerConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    def expected_returns(self, returns: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        if cfg.mu_estimator == "mean":
            mu = returns.mean().values
        elif cfg.mu_estimator == "ewm":
            mu = returns.ewm(halflife=cfg.ewm_halflife, min_periods=20).mean().iloc[-1].values
        else:  # capm_blend / fallback
            mu = returns.mean().values
        mu = np.nan_to_num(mu, nan=0.0) * TRADING_DAYS
        # Shrink cross-sectionally toward the grand mean (James-Stein flavour).
        grand = float(np.mean(mu)) if len(mu) else 0.0
        mu = (1 - cfg.mu_shrink) * mu + cfg.mu_shrink * grand
        return mu

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Return target weights (index = columns of ``returns``)."""
        cols = list(returns.columns)
        n = len(cols)
        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series([min(1.0, self.cfg.max_weight if self.cfg.max_weight < 1 else 1.0)],
                             index=cols)

        cov = shrink_covariance(returns, self.cfg.shrinkage)
        mu = self.expected_returns(returns)
        method = self.cfg.method

        if method == "equal_weight":
            w = np.ones(n) / n
        elif method == "inverse_vol":
            w = self._inverse_vol(cov)
        elif method == "min_variance":
            w = self._solve(cov, mu, objective="min_var")
        elif method == "max_sharpe":
            w = self._solve(cov, mu, objective="max_sharpe")
        elif method == "risk_parity":
            w = self._risk_parity(cov)
        elif method == "max_diversification":
            w = self._solve(cov, mu, objective="max_div")
        else:
            raise ValueError(f"unknown method '{method}'")

        w = self._apply_caps(w)
        return pd.Series(w, index=cols)

    # ------------------------------------------------------------------ #
    # Solvers
    # ------------------------------------------------------------------ #
    def _bounds(self, n: int):
        return [(self.cfg.min_weight, self.cfg.max_weight) for _ in range(n)]

    def _constraints(self):
        # Fully invested unless cash is allowed (then sum <= 1).
        if self.cfg.allow_cash:
            return [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}]
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def _solve(self, cov: np.ndarray, mu: np.ndarray, objective: str) -> np.ndarray:
        n = len(mu)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))

        def port_var(w):
            return float(w @ cov @ w)

        if objective == "min_var":
            fun = port_var
        elif objective == "max_sharpe":
            def fun(w):
                v = np.sqrt(max(port_var(w), 1e-18))
                ret = float(w @ mu)
                return -(ret - self.cfg.rf) / v
        elif objective == "max_div":
            def fun(w):
                v = np.sqrt(max(port_var(w), 1e-18))
                return -float(w @ std) / v
        else:
            raise ValueError(objective)

        x0 = np.ones(n) / n
        res = minimize(
            fun, x0, method="SLSQP",
            bounds=self._bounds(n), constraints=self._constraints(),
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w = res.x if res.success else x0
        w = np.clip(w, 0, None)
        return w

    def _inverse_vol(self, cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        inv = 1.0 / std
        return inv / inv.sum()

    def _risk_parity(self, cov: np.ndarray) -> np.ndarray:
        n = cov.shape[0]

        def fun(w):
            w = np.clip(w, 1e-9, None)
            port_var = w @ cov @ w
            mrc = cov @ w                     # marginal risk contribution
            rc = w * mrc                       # risk contribution
            target = port_var / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        res = minimize(
            fun, x0, method="SLSQP",
            bounds=[(0.0, self.cfg.max_weight) for _ in range(n)],
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = res.x if res.success else x0
        return np.clip(w, 0, None)

    def _apply_caps(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, self.cfg.max_weight)
        s = w.sum()
        if s <= 0:
            return np.zeros_like(w)
        # If not allowing cash, renormalise to fully invested (respecting caps).
        if not self.cfg.allow_cash:
            w = w / s
            # A single renormalise can breach caps; clip + renormalise once more.
            w = np.clip(w, 0.0, self.cfg.max_weight)
            s2 = w.sum()
            if s2 > 0:
                w = w / s2
        else:
            # Allow cash: only scale down if the book is over-invested.
            if s > 1.0:
                w = w / s
        return w
