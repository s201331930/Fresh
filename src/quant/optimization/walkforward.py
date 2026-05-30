"""Walk-forward evaluation with train / test / validate splits.

The allocation backtester is already causal -- at every rebalance it estimates
inputs from *trailing* data only, so a single run over the full history is a
genuine out-of-sample walk-forward. This module:

    1. carves the timeline into three contiguous eras (train, test, validate),
    2. runs each candidate optimisation method once over the full history,
    3. scores every method on each era, and
    4. selects the **champion** on the *test* era, then reports its performance
       on the untouched *validate* era as the unbiased estimate of live edge.

Selecting the model on test and judging it on validate is the standard guard
against curve-fitting: the validate era never influences any choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..metrics import compute_kpis, PerformanceReport
from .allocation import AllocationBacktester, AllocationConfig, AllocationResult
from .optimizer import PortfolioOptimizer, OptimizerConfig


@dataclass
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class SplitResult:
    method: str
    split: str
    report: PerformanceReport


@dataclass
class MethodRun:
    method: str
    result: AllocationResult
    split_reports: Dict[str, PerformanceReport] = field(default_factory=dict)
    full_report: Optional[PerformanceReport] = None


class WalkForward:
    def __init__(
        self,
        alloc_config: AllocationConfig,
        initial_capital: float = 1_000_000.0,
        rf: float = 0.02,
        train_frac: float = 0.45,
        test_frac: float = 0.30,
    ) -> None:
        self.alloc_config = alloc_config
        self.initial_capital = initial_capital
        self.rf = rf
        self.train_frac = train_frac
        self.test_frac = test_frac

    # ------------------------------------------------------------------ #
    def make_splits(self, index: pd.DatetimeIndex) -> List[Split]:
        n = len(index)
        i_train = int(n * self.train_frac)
        i_test = int(n * (self.train_frac + self.test_frac))
        return [
            Split("train", index[0], index[max(0, i_train - 1)]),
            Split("test", index[i_train], index[max(i_train, i_test - 1)]),
            Split("validate", index[i_test], index[-1]),
        ]

    def _split_kpis(self, equity: pd.Series, split: Split, name: str) -> PerformanceReport:
        sub = equity.loc[split.start:split.end]
        if len(sub) > 2:
            sub = sub / sub.iloc[0] * self.initial_capital  # rebase each era
        return compute_kpis(sub, rf=self.rf, name=f"{name}/{split.name}")

    def run(
        self,
        prices: pd.DataFrame,
        methods: List[str],
        base_optimizer_cfg: OptimizerConfig,
    ) -> Dict[str, MethodRun]:
        splits = self.make_splits(prices.index)
        self.splits = splits
        runs: Dict[str, MethodRun] = {}
        for method in methods:
            cfg = OptimizerConfig(**{**base_optimizer_cfg.__dict__, "method": method})
            opt = PortfolioOptimizer(cfg)
            bt = AllocationBacktester(opt, self.alloc_config, self.initial_capital)
            res = bt.run(prices)
            mr = MethodRun(method=method, result=res)
            for sp in splits:
                mr.split_reports[sp.name] = self._split_kpis(res.equity, sp, method)
            mr.full_report = compute_kpis(
                res.equity, rf=self.rf, exposure=res.exposure, name=method
            )
            runs[method] = mr
        return runs

    # ------------------------------------------------------------------ #
    @staticmethod
    def score(report: PerformanceReport) -> float:
        """Composite selection score: reward Sharpe, respect drawdown control."""
        s = report.kpis.get("sharpe", 0.0)
        c = report.kpis.get("calmar", 0.0)
        if not np.isfinite(s):
            s = 0.0
        if not np.isfinite(c):
            c = 0.0
        return 0.7 * s + 0.3 * min(c, 5.0)

    def select_champion(self, runs: Dict[str, MethodRun], on: str = "test") -> str:
        best, best_score = None, -np.inf
        for method, mr in runs.items():
            sc = self.score(mr.split_reports[on])
            if sc > best_score:
                best, best_score = method, sc
        return best
