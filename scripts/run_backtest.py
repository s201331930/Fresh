#!/usr/bin/env python3
"""End-to-end backtest runner for the multi-strategy long-only system.

Usage:
    python scripts/run_backtest.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

# Make ``src`` importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quant.data import DataLoader  # noqa: E402
from quant.engine import Backtester, Portfolio  # noqa: E402
from quant.metrics import compute_kpis  # noqa: E402
from quant.reporting import ReportBuilder  # noqa: E402
from quant.strategies import STRATEGY_REGISTRY  # noqa: E402


def _collect_symbols(cfg: dict) -> list:
    syms = set()
    for name, scfg in cfg["strategies"].items():
        if not scfg.get("enabled", True):
            continue
        syms.update(scfg.get("universe", []))
        if scfg.get("safe_asset"):
            syms.add(scfg["safe_asset"])
    bench = cfg["backtest"].get("benchmark")
    if bench:
        syms.add(bench)
    return sorted(syms)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-strategy long-only backtest")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bt_cfg = cfg["backtest"]
    start = bt_cfg["start"]
    end = bt_cfg.get("end")
    capital = float(bt_cfg["initial_capital"])
    rf = float(bt_cfg.get("risk_free_rate", 0.0))
    benchmark_sym = bt_cfg.get("benchmark")

    print("=" * 70)
    print("MULTI-STRATEGY LONG-ONLY BACKTEST")
    print("=" * 70)

    # ---- Data ---------------------------------------------------------- #
    dcfg = cfg.get("data", {})
    loader = DataLoader(
        cache_dir=dcfg.get("cache_dir", "data/cache"),
        allow_synthetic_fallback=dcfg.get("allow_synthetic_fallback", True),
        synthetic_seed=dcfg.get("synthetic_seed", 7),
    )
    symbols = _collect_symbols(cfg)
    print(f"\nLoading {len(symbols)} symbols from {start} to {end or 'today'} ...")
    data = loader.load(symbols, start=start, end=end)
    print(f"Loaded {len(data.symbols)} symbols.")
    if data.synthetic:
        print(f"  NOTE: synthetic fallback used for: {', '.join(data.synthetic)}")

    # ---- Run each strategy sleeve ------------------------------------- #
    engine = Backtester(
        initial_capital=capital,
        risk_free_rate=rf,
        commission_bps=bt_cfg.get("commission_bps", 1.0),
        slippage_bps=bt_cfg.get("slippage_bps", 5.0),
    )

    results = {}
    sleeve_trades = {}
    for name, scfg in cfg["strategies"].items():
        if not scfg.get("enabled", True):
            continue
        if name not in STRATEGY_REGISTRY:
            print(f"  skipping unknown strategy '{name}'")
            continue
        params = {k: v for k, v in scfg.items() if k not in ("enabled", "universe")}
        strat = STRATEGY_REGISTRY[name](universe=scfg.get("universe", []), **params)
        print(f"\n[{name}] generating signals ...")
        signals = strat.generate(data)
        if not signals:
            print(f"  [{name}] produced no signals; skipping.")
            continue
        res = engine.run(data, signals, strat.risk_config(), name=name)
        results[name] = res
        sleeve_trades[name] = res.trades
        print(f"  [{name}] {len(res.trades)} trades, "
              f"final equity ${res.equity.iloc[-1]:,.0f}")

    if not results:
        print("No strategies produced results. Aborting.")
        return 1

    # ---- Benchmark ----------------------------------------------------- #
    benchmark_returns = None
    benchmark_equity = None
    if benchmark_sym and benchmark_sym in data:
        bclose = data[benchmark_sym]["close"].reindex(
            next(iter(results.values())).equity.index
        ).ffill()
        benchmark_equity = capital * (bclose / bclose.iloc[0])
        benchmark_returns = benchmark_equity.pct_change().fillna(0.0)

    # ---- Combine into portfolio --------------------------------------- #
    pcfg = cfg.get("portfolio", {})
    portfolio = Portfolio(
        initial_capital=capital,
        base_weights=pcfg.get("weights", {}),
        use_vol_parity=pcfg.get("use_vol_parity", True),
        vol_parity_lookback=pcfg.get("vol_parity_lookback", 90),
    )
    pres = portfolio.combine(results)

    # ---- KPIs ---------------------------------------------------------- #
    sleeve_reports = []
    for name, res in results.items():
        rep = compute_kpis(
            res.equity, trades=res.trades, rf=rf,
            benchmark_returns=benchmark_returns, exposure=res.exposure, name=name,
        )
        sleeve_reports.append(rep)

    portfolio_report = compute_kpis(
        pres.equity, trades=[t for ts in sleeve_trades.values() for t in ts],
        rf=rf, benchmark_returns=benchmark_returns, exposure=pres.exposure,
        name="portfolio",
    )

    benchmark_report = None
    if benchmark_equity is not None:
        benchmark_report = compute_kpis(benchmark_equity, rf=rf, name=f"{benchmark_sym} (B&H)")

    correlation = pres.sleeve_returns.corr()

    # ---- Reporting ----------------------------------------------------- #
    os.makedirs(args.out, exist_ok=True)
    builder = ReportBuilder(out_dir=args.out)

    curves = {"Portfolio": pres.equity}
    for name, res in results.items():
        curves[name] = res.equity
    figures = {
        "equity": builder.plot_equity(curves, benchmark_equity, "Equity Curves -- Growth of $100"),
        "drawdown": builder.plot_drawdown(pres.equity, "Combined Portfolio Drawdown"),
        "correlation": builder.plot_correlation(correlation),
        "weights": builder.plot_weights(pres.weights),
    }

    meta = {
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "Period": f"{pres.equity.index[0].date()} to {pres.equity.index[-1].date()}",
        "Initial Capital": f"${capital:,.0f}",
        "Risk-Free Rate": f"{rf*100:.2f}%",
        "Commission": f"{bt_cfg.get('commission_bps',1.0)} bps/side",
        "Slippage": f"{bt_cfg.get('slippage_bps',5.0)} bps/side",
        "Vol-Parity Overlay": str(pcfg.get("use_vol_parity", True)),
        "Strategies": ", ".join(results.keys()),
        "Benchmark": benchmark_sym or "-",
        "Synthetic Data Used": ", ".join(data.synthetic) if data.synthetic else "none",
    }

    report_path = builder.build(
        portfolio_report=portfolio_report,
        sleeve_reports=sleeve_reports,
        benchmark_report=benchmark_report,
        correlation=correlation,
        sleeve_trades=sleeve_trades,
        figures=figures,
        meta=meta,
    )

    # ---- Persist artefacts -------------------------------------------- #
    pres.equity.to_csv(os.path.join(args.out, "portfolio_equity.csv"))
    pres.weights.to_csv(os.path.join(args.out, "strategy_weights.csv"))
    correlation.to_csv(os.path.join(args.out, "correlation_matrix.csv"))
    all_trades = []
    for name, ts in sleeve_trades.items():
        for t in ts:
            t = dict(t); t["strategy"] = name
            all_trades.append(t)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(os.path.join(args.out, "trades.csv"), index=False)

    # ---- Console summary ---------------------------------------------- #
    _print_summary(portfolio_report, benchmark_report, sleeve_reports, correlation)
    print(f"\nFull report : {report_path}")
    print(f"Artefacts   : {args.out}/  (figures, CSVs)")
    return 0


def _line(label, value):
    print(f"  {label:<26}{value}")


def _print_summary(port, bench, sleeves, corr):
    print("\n" + "=" * 70)
    print("PORTFOLIO KPIs")
    print("=" * 70)
    k = port.kpis
    _line("Total Return", f"{k['total_return']*100:.1f}%")
    _line("CAGR", f"{k['cagr']*100:.2f}%")
    _line("Volatility", f"{k['volatility']*100:.2f}%")
    _line("Sharpe", f"{k['sharpe']:.2f}")
    _line("Sortino", f"{k['sortino']:.2f}")
    _line("Calmar", f"{k['calmar']:.2f}")
    _line("Max Drawdown", f"{k['max_drawdown']*100:.2f}%")
    _line("Win Rate", f"{k.get('win_rate',0)*100:.1f}%")
    _line("Profit Factor", f"{k.get('profit_factor',0):.2f}")
    _line("Num Trades", f"{k.get('num_trades',0)}")
    if bench is not None:
        bk = bench.kpis
        print("\nBENCHMARK (Buy & Hold)")
        _line("CAGR", f"{bk['cagr']*100:.2f}%")
        _line("Sharpe", f"{bk['sharpe']:.2f}")
        _line("Max Drawdown", f"{bk['max_drawdown']*100:.2f}%")
    print("\nAVERAGE PAIRWISE CORRELATION")
    import numpy as np
    avg = corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()
    _line("Avg corr (sleeves)", f"{avg:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
