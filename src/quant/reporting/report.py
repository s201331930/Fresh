"""Reporting: a markdown tear-sheet plus PNG charts.

Produces a single, desk-ready report covering the combined book, each strategy
sleeve, the cross-strategy correlation matrix (the key diversification check),
benchmark-relative stats, and full trade statistics.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..metrics import PerformanceReport, drawdown_series  # noqa: E402

PCT = {
    "total_return", "cagr", "volatility", "max_drawdown", "avg_drawdown",
    "var_95", "cvar_95", "best_day", "worst_day", "time_in_market",
    "win_rate", "avg_return_pct", "best_trade_pct", "worst_trade_pct",
    "alpha", "correlation_to_benchmark",
}
RATIO = {"sharpe", "sortino", "calmar", "beta", "profit_factor", "payoff_ratio", "skew", "kurtosis"}
MONEY = {"initial_equity", "final_equity", "expectancy", "avg_win", "avg_loss"}

LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "volatility": "Annualised Volatility",
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "max_drawdown": "Max Drawdown",
    "avg_drawdown": "Avg Drawdown",
    "max_dd_duration_days": "Max DD Duration (days)",
    "var_95": "Daily VaR (95%)",
    "cvar_95": "Daily CVaR (95%)",
    "skew": "Return Skew",
    "kurtosis": "Return Kurtosis",
    "best_day": "Best Day",
    "worst_day": "Worst Day",
    "time_in_market": "Time in Market",
    "beta": "Beta vs Benchmark",
    "alpha": "Annualised Alpha",
    "correlation_to_benchmark": "Correlation vs Benchmark",
    "num_trades": "Number of Trades",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "payoff_ratio": "Payoff Ratio (avg win/loss)",
    "expectancy": "Expectancy / Trade",
    "avg_win": "Avg Winning Trade",
    "avg_loss": "Avg Losing Trade",
    "avg_return_pct": "Avg Trade Return",
    "best_trade_pct": "Best Trade",
    "worst_trade_pct": "Worst Trade",
    "avg_holding_days": "Avg Holding (days)",
    "initial_equity": "Initial Equity",
    "final_equity": "Final Equity",
}

KPI_ORDER = [
    "initial_equity", "final_equity", "total_return", "cagr", "volatility",
    "sharpe", "sortino", "calmar", "max_drawdown", "avg_drawdown",
    "max_dd_duration_days", "var_95", "cvar_95", "skew", "kurtosis",
    "best_day", "worst_day", "time_in_market", "beta", "alpha",
    "correlation_to_benchmark", "num_trades", "win_rate", "profit_factor",
    "payoff_ratio", "expectancy", "avg_win", "avg_loss", "avg_return_pct",
    "best_trade_pct", "worst_trade_pct", "avg_holding_days",
]


def _fmt(key: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if key in PCT:
        return f"{value * 100:.2f}%"
    if key in MONEY:
        return f"${value:,.0f}"
    if key in RATIO:
        return f"{value:.2f}"
    if key == "num_trades" or key == "max_dd_duration_days":
        return f"{int(value):,}"
    if key == "avg_holding_days":
        return f"{value:.1f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


class ReportBuilder:
    def __init__(self, out_dir: str = "reports") -> None:
        self.out_dir = out_dir
        self.fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #
    def plot_equity(self, curves: Dict[str, pd.Series], benchmark: Optional[pd.Series], title: str) -> str:
        plt.figure(figsize=(11, 6))
        for name, eq in curves.items():
            norm = eq / eq.iloc[0] * 100
            lw = 2.4 if name.lower().startswith("portfolio") else 1.2
            plt.plot(norm.index, norm.values, label=name, linewidth=lw)
        if benchmark is not None and len(benchmark) > 1:
            bnorm = benchmark / benchmark.iloc[0] * 100
            plt.plot(bnorm.index, bnorm.values, label="Benchmark (B&H)", linewidth=1.4,
                     linestyle="--", color="black", alpha=0.7)
        plt.yscale("log")
        plt.title(title)
        plt.ylabel("Growth of $100 (log scale)")
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.fig_dir, "equity_curves.png")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path

    def plot_drawdown(self, equity: pd.Series, title: str) -> str:
        dd = drawdown_series(equity) * 100
        plt.figure(figsize=(11, 3.5))
        plt.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
        plt.plot(dd.index, dd.values, color="crimson", linewidth=0.8)
        plt.title(title)
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.fig_dir, "drawdown.png")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path

    def plot_correlation(self, corr: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title("Strategy Return Correlation Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = os.path.join(self.fig_dir, "correlation.png")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path

    def plot_weights(self, weights: pd.DataFrame) -> str:
        plt.figure(figsize=(11, 4))
        w = weights.resample("ME").last()
        plt.stackplot(w.index, [w[c].values for c in w.columns], labels=list(w.columns), alpha=0.85)
        plt.title("Strategy Capital Allocation Over Time (vol-parity)")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.legend(loc="upper left", fontsize=8, ncol=len(w.columns))
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.fig_dir, "weights.png")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path

    # ------------------------------------------------------------------ #
    # Markdown helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kpi_block(report: PerformanceReport) -> str:
        rows = ["| Metric | Value |", "|---|---|"]
        for key in KPI_ORDER:
            if key in report.kpis:
                rows.append(f"| {LABELS.get(key, key)} | {_fmt(key, report.kpis[key])} |")
        return "\n".join(rows)

    @staticmethod
    def _comparison_table(reports: List[PerformanceReport]) -> str:
        keys = ["cagr", "volatility", "sharpe", "sortino", "calmar", "max_drawdown",
                "win_rate", "profit_factor", "num_trades", "time_in_market"]
        header = "| Metric | " + " | ".join(r.name for r in reports) + " |"
        sep = "|" + "---|" * (len(reports) + 1)
        lines = [header, sep]
        for k in keys:
            cells = [LABELS.get(k, k)] + [_fmt(k, r.kpis.get(k)) for r in reports]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def _monthly_table(report: PerformanceReport) -> str:
        mt = report.monthly_returns
        if mt is None or mt.empty:
            return "_No data._"
        df = (mt * 100).round(2)
        cols = list(df.columns)
        header = "| Year | " + " | ".join(cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        lines = [header, sep]
        for year, row in df.iterrows():
            cells = [str(int(year))] + ["" if pd.isna(v) else f"{v:.1f}" for v in row.values]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def _exit_reason_breakdown(trades: List[dict]) -> str:
        if not trades:
            return "_No trades._"
        df = pd.DataFrame(trades)
        g = df.groupby("exit_reason").agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            total_pnl=("pnl", "sum"),
        )
        lines = ["| Exit Reason | Trades | Win Rate | Total P&L |", "|---|---|---|---|"]
        for reason, r in g.iterrows():
            lines.append(f"| {reason} | {int(r['trades'])} | {r['win_rate']*100:.1f}% | ${r['total_pnl']:,.0f} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Master report
    # ------------------------------------------------------------------ #
    def build(
        self,
        portfolio_report: PerformanceReport,
        sleeve_reports: List[PerformanceReport],
        benchmark_report: Optional[PerformanceReport],
        correlation: pd.DataFrame,
        sleeve_trades: Dict[str, List[dict]],
        figures: Dict[str, str],
        meta: Dict[str, str],
    ) -> str:
        md: List[str] = []
        md.append("# Multi-Strategy Long-Only Trading System -- Backtest Report\n")
        md.append(f"*Generated: {meta.get('generated')}*\n")
        md.append(
            "> **Mandate:** long-only, leverage-free, every position risk-managed "
            "with a stop-loss and (where applicable) a take-profit and/or trailing "
            "stop. Capital is split across deliberately uncorrelated strategy sleeves.\n"
        )

        # Run parameters
        md.append("## 1. Run Parameters\n")
        md.append("| Parameter | Value |\n|---|---|")
        for k, v in meta.items():
            if k == "generated":
                continue
            md.append(f"| {k} | {v} |")
        md.append("")

        # Headline portfolio
        md.append("## 2. Combined Portfolio Performance\n")
        md.append(self._kpi_block(portfolio_report))
        md.append("")
        if "equity" in figures:
            md.append(f"![Equity Curves]({os.path.relpath(figures['equity'], self.out_dir)})\n")
        if "drawdown" in figures:
            md.append(f"![Drawdown]({os.path.relpath(figures['drawdown'], self.out_dir)})\n")

        # Benchmark comparison
        if benchmark_report is not None:
            md.append("### 2.1 Portfolio vs Buy & Hold Benchmark\n")
            md.append(self._comparison_table([portfolio_report, benchmark_report]))
            md.append("")

        # Diversification / correlation
        md.append("## 3. Diversification -- Strategy Correlation Matrix\n")
        md.append(
            "The core thesis is that these sleeves make money at different times. "
            "Low pairwise correlation is what lets the blended Sharpe exceed the "
            "individual sleeves'.\n"
        )
        corr_md = ["| | " + " | ".join(correlation.columns) + " |",
                   "|" + "---|" * (len(correlation.columns) + 1)]
        for idx, row in correlation.iterrows():
            corr_md.append("| **" + str(idx) + "** | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
        md.append("\n".join(corr_md))
        md.append("")
        avg_corr = correlation.where(~np.eye(len(correlation), dtype=bool)).stack().mean()
        md.append(f"**Average pairwise correlation: {avg_corr:.2f}** "
                  f"(lower is better for diversification).\n")
        if "correlation" in figures:
            md.append(f"![Correlation]({os.path.relpath(figures['correlation'], self.out_dir)})\n")
        if "weights" in figures:
            md.append(f"![Weights]({os.path.relpath(figures['weights'], self.out_dir)})\n")

        # Sleeve comparison
        md.append("## 4. Strategy Sleeves -- Side by Side\n")
        md.append(self._comparison_table(sleeve_reports))
        md.append("")

        # Per-sleeve detail
        md.append("## 5. Strategy Sleeve Detail\n")
        for r in sleeve_reports:
            md.append(f"### 5.{sleeve_reports.index(r)+1} {r.name}\n")
            md.append(self._kpi_block(r))
            md.append("")
            md.append("**Exit reason breakdown:**\n")
            md.append(self._exit_reason_breakdown(sleeve_trades.get(r.name, [])))
            md.append("")

        # Monthly returns for the portfolio
        md.append("## 6. Combined Portfolio -- Monthly Returns (%)\n")
        md.append(self._monthly_table(portfolio_report))
        md.append("")

        md.append("## 7. Methodology & Caveats\n")
        md.append(
            "- **No look-ahead:** signals are computed on the close of day *t* and "
            "filled at the open of day *t+1*. Rolling highs used for breakouts are "
            "shifted one bar.\n"
            "- **Conservative intrabar fills:** if a bar could hit both stop and "
            "target, the stop is assumed to fill first; gaps through the stop fill "
            "at the open.\n"
            "- **Costs:** per-side commission and adverse slippage are applied to "
            "every fill; idle cash earns the risk-free rate.\n"
            "- **Leverage-free:** every purchase is gated by available cash, so "
            "aggregate exposure can never exceed equity.\n"
            "- **Survivorship / data:** ETF proxies are used per asset class; "
            "adjusted prices fold in dividends and splits. Past performance does "
            "not guarantee future results -- size live risk accordingly.\n"
        )

        report_md = "\n".join(md)
        path = os.path.join(self.out_dir, "backtest_report.md")
        with open(path, "w") as f:
            f.write(report_md)
        return path
