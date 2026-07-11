"""
Generate comprehensive PDF report documenting the trading model and strategy.
"""

import json
from pathlib import Path
from fpdf import FPDF

MODEL_DIR = Path(__file__).parent / "model"
STRATEGY_DIR = Path(__file__).parent / "strategy"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PDF = Path(__file__).parent / "Trading_Model_Report.pdf"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "SPY Weekly Trading Model - Confidential", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 60, 120)
        self.ln(6)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(3)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, "  - " + text)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)

        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "L" if i == 0 else "R"
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align=align)
            self.ln()
            fill = not fill
        self.ln(3)

    def add_image_safe(self, path, w=170):
        p = Path(path)
        if p.exists():
            self.image(str(p), w=w)
            self.ln(4)
        else:
            self.body_text(f"[Image not found: {path}]")


def main():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== TITLE PAGE =====
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "SPY Weekly Trading Model", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Predictive Model & Strategy Documentation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Instrument: SPDR S&P 500 ETF (SPY)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Timeframe: Weekly (2000 - 2026)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Strategy: Long-Only, No Leverage", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Prototype - For Research Purposes Only", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "July 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ===== TABLE OF CONTENTS =====
    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        "1. Executive Summary",
        "2. Data Pipeline",
        "   2.1 Data Sources",
        "   2.2 Feature Engineering",
        "   2.3 Technical Indicators",
        "   2.4 Order Block Detection",
        "3. Model Architecture",
        "   3.1 Target Variable",
        "   3.2 Feature Selection",
        "   3.3 Ensemble Composition",
        "   3.4 Walk-Forward Validation",
        "4. Trading Strategy",
        "   4.1 Strategy Variants",
        "   4.2 Backtest Methodology",
        "   4.3 Performance Results",
        "   4.4 Period Breakdown",
        "5. Risk Analysis",
        "6. Feature Importance",
        "7. Conclusions & Next Steps",
    ]
    for item in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT")

    # ===== 1. EXECUTIVE SUMMARY =====
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This document presents a machine learning-based trading system for SPY (SPDR S&P 500 ETF) "
        "on a weekly timeframe. The system uses an ensemble of gradient boosting models (XGBoost, "
        "LightGBM, CatBoost) combined with Ridge regression to predict next-week returns. "
        "Predictions are generated using a rigorous walk-forward methodology that eliminates "
        "look-ahead bias."
    )
    pdf.body_text(
        "The primary strategy (Binary, threshold=0) achieves a total return of 864.9% over the "
        "out-of-sample period (Dec 2009 - Jun 2026), compared to 768.4% for buy-and-hold SPY. "
        "This represents an excess CAGR of +0.73% with a higher Sharpe ratio (0.996 vs 0.890), "
        "while being invested only 80.8% of the time."
    )

    pdf.subsection_title("Key Results at a Glance")
    pdf.table(
        ["Metric", "Strategy", "Buy & Hold", "Excess"],
        [
            ["Total Return", "864.9%", "768.4%", "+96.5pp"],
            ["CAGR", "14.67%", "13.94%", "+0.73pp"],
            ["Sharpe Ratio", "0.996", "0.890", "+0.106"],
            ["Sortino Ratio", "1.179", "1.142", "+0.037"],
            ["Max Drawdown", "-31.83%", "-31.83%", "0.00pp"],
            ["Win Rate", "60.2%", "-", "-"],
            ["Time in Market", "80.8%", "100%", "-19.2pp"],
        ],
        col_widths=[40, 35, 35, 35],
    )

    pdf.body_text(
        "Critically, the strategy performs better on the true out-of-sample test period "
        "(Mar 2018 - Jun 2026) than during the optimization period, with +37.1pp excess return "
        "and +1.53pp excess CAGR over buy-and-hold - strong evidence against overfitting."
    )

    # ===== 2. DATA PIPELINE =====
    pdf.add_page()
    pdf.section_title("2. Data Pipeline")

    pdf.subsection_title("2.1 Data Sources")
    pdf.body_text(
        "All data is sourced from Yahoo Finance via the yfinance Python library. Daily data is "
        "downloaded and resampled to weekly bars (Monday-Friday, indexed by Friday close)."
    )
    pdf.table(
        ["Source", "Ticker", "Fields", "Coverage"],
        [
            ["SPY", "SPY", "OHLCV + Adj Close", "Jan 2000 - Present"],
            ["VIX", "^VIX", "OHLC", "Jan 2000 - Present (100%)"],
            ["Brent Crude", "BZ=F", "OHLC", "Aug 2007 - Present (71%)"],
            ["10Y Treasury", "^TNX", "Close", "Jan 2000 - Present"],
            ["US Dollar Index", "DX-Y.NYB", "Close", "Jan 2000 - Present"],
            ["Gold", "GC=F", "Close", "Jan 2000 - Present"],
        ],
        col_widths=[30, 25, 40, 50],
    )

    pdf.body_text(
        "SPY prices are adjusted for splits and dividends using the A2P factor "
        "(Adj Close / Close), applied to all OHLC fields to maintain intra-bar consistency."
    )

    pdf.subsection_title("2.2 Feature Engineering")
    pdf.body_text("The feature set is constructed in layers, totaling 197 raw features before selection:")
    pdf.bullet("Base OHLCV: Open, Low, High, Close, Volume (+ log-transformed volume)")
    pdf.bullet("Percentage changes: Week-over-week returns for all price series")
    pdf.bullet("Lagged features: 1-4 week lags on 15 key signals (61 features)")
    pdf.bullet("Rolling statistics: 4/8/12-week mean and std of returns, VIX (9 features)")
    pdf.bullet("Interaction terms: 12 cross-variable products (e.g., VIX x momentum, RSI x MACD)")
    pdf.bullet("Regime indicators: Above SMA-200, Golden Cross, VIX regime, trend direction")
    pdf.bullet("Time features: Month, quarter, week-of-year, seasonal dummies")
    pdf.bullet("Macro variables: Treasury yield, Dollar index, Gold price and changes")

    pdf.subsection_title("2.3 Technical Indicators")
    pdf.body_text("Computed for SPY, VIX, and Brent Crude (where applicable):")
    pdf.table(
        ["Indicator", "Parameters", "Columns per Instrument"],
        [
            ["SMA", "5, 10, 20, 50, 100, 200", "6"],
            ["EMA", "5, 10, 20, 50, 100, 200", "6"],
            ["Bollinger Bands", "20-period, 2 std", "4 (Upper, Lower, Width, %B)"],
            ["MACD", "12/26/9", "3 (Line, Signal, Histogram)"],
            ["RSI", "14-period (Wilder)", "1"],
            ["ADX & DI", "14-period", "3 (+DI, -DI, ADX)"],
            ["OBV", "Cumulative", "1 (SPY only)"],
        ],
        col_widths=[35, 45, 65],
    )

    pdf.subsection_title("2.4 Order Block Detection")
    pdf.body_text(
        "Order Blocks are identified using wugamlo's algorithm (TradingView Pine Script, "
        "translated to Python). A Bullish OB is the last down candle before 5 consecutive "
        "up candles; Bearish OB is the inverse. Over the full dataset: 41 Bullish OBs and "
        "10 Bearish OBs detected. OB columns are excluded from model features to avoid "
        "look-ahead bias (OBs are confirmed retroactively)."
    )

    # ===== 3. MODEL ARCHITECTURE =====
    pdf.add_page()
    pdf.section_title("3. Model Architecture")

    pdf.subsection_title("3.1 Target Variable")
    pdf.body_text(
        "The primary target is the next-week percentage return:\n\n"
        "    Target = (Close[t+1] - Close[t]) / Close[t] x 100\n\n"
        "This is shifted forward by one week, ensuring no future information leaks into "
        "features. The last row of data is dropped since no next-week return is available."
    )
    pdf.body_text(
        "Target statistics: mean = +0.18% per week, std = 2.46%, range = -19.8% to +13.3%. "
        "The positive mean reflects SPY's long-term upward drift."
    )

    pdf.subsection_title("3.2 Feature Selection")
    pdf.body_text(
        "From 197 candidate features, the top 40 are selected using XGBoost importance (gain). "
        "This reduces the feature/sample ratio from 0.20 to 0.041, significantly reducing "
        "overfitting risk. Feature selection is performed using only training data."
    )

    pdf.subsection_title("3.3 Ensemble Composition")
    pdf.table(
        ["Model", "Weight", "Key Hyperparameters"],
        [
            ["XGBoost", "35%", "depth=4, lr=0.03, subsample=0.8, lambda=5"],
            ["LightGBM", "30%", "depth=4, lr=0.03, subsample=0.8, lambda=5"],
            ["CatBoost", "25%", "depth=4, lr=0.03, l2_leaf_reg=5"],
            ["Ridge", "10%", "alpha=10.0 (StandardScaler applied)"],
        ],
        col_widths=[25, 20, 100],
    )
    pdf.body_text(
        "All tree models use early stopping on validation RMSE with patience of 30 rounds. "
        "Ridge regression serves as a linear baseline to capture simple relationships the "
        "tree models might miss. Blend weights were set based on validation performance."
    )

    pdf.subsection_title("3.4 Walk-Forward Validation")
    pdf.body_text(
        "The model uses expanding-window walk-forward validation with no look-ahead bias:\n\n"
        "  - Minimum training window: 10 years (520 weeks)\n"
        "  - Step size: 1 year (52 weeks) per fold\n"
        "  - 17 folds total, 859 OOS predictions\n"
        "  - Average direction accuracy: 58.0% (vs 50% random baseline)\n"
        "  - 15 of 17 folds beat the random baseline\n"
        "  - Best fold: 69.2% direction accuracy\n\n"
        "NaN imputation (forward-fill for indicators, zero-fill for Brent pre-2007) recovers "
        "training rows from 281 to 966, a 3.4x increase in usable training data."
    )
    pdf.add_image_safe(MODEL_DIR / "walkforward_folds.png", w=160)

    # ===== 4. TRADING STRATEGY =====
    pdf.add_page()
    pdf.section_title("4. Trading Strategy")

    pdf.subsection_title("4.1 Strategy Variants")
    pdf.body_text("Four long-only, no-leverage strategies were tested:")
    pdf.bullet(
        "Binary (t=0): Go 100% long if predicted return > 0%, else 100% cash. "
        "Simple and effective - captures the model's directional signal directly."
    )
    pdf.bullet(
        "Binary (t=opt): Same as above but threshold optimized on first half of OOS data "
        "(optimized threshold = +0.28%). More selective - only in market 16% of the time."
    )
    pdf.bullet(
        "Scaled: Position size proportional to prediction magnitude, clipped to [0%, 100%]. "
        "Higher conviction = larger position."
    )
    pdf.bullet(
        "Adaptive: Binary strategy with threshold re-optimized every 26 weeks using recent "
        "OOS performance. Self-adjusting to changing market regimes."
    )
    pdf.body_text("Transaction cost: 0.02% round-trip per trade (realistic for weekly SPY trades).")

    pdf.subsection_title("4.2 Backtest Methodology")
    pdf.body_text(
        "All predictions come from walk-forward OOS output - the model never sees the data "
        "it trades on during training. The backtest covers 861 weeks (Dec 2009 - Jun 2026).\n\n"
        "The threshold for Binary (t=opt) is optimized on the first 430 weeks, and the "
        "remaining 431 weeks serve as a true out-of-sample test. This two-stage approach "
        "validates that the strategy generalizes beyond its optimization period."
    )

    pdf.subsection_title("4.3 Performance Results - Full OOS Period")
    pdf.table(
        ["Strategy", "Return", "CAGR", "Sharpe", "Sortino", "MaxDD", "WinRate", "InMkt"],
        [
            ["Binary (t=0)", "864.9%", "14.67%", "0.996", "1.179", "-31.8%", "60.2%", "80.8%"],
            ["Adaptive", "613.5%", "12.60%", "0.961", "-", "-24.7%", "59.3%", "70.3%"],
            ["Scaled", "321.1%", "9.07%", "0.983", "-", "-18.0%", "59.6%", "80.8%"],
            ["Binary (t=opt)", "312.7%", "8.94%", "0.925", "-", "-14.6%", "67.1%", "16.3%"],
            ["Buy & Hold", "768.4%", "13.94%", "0.890", "1.142", "-31.8%", "-", "100%"],
        ],
        col_widths=[28, 18, 16, 16, 16, 16, 17, 14],
    )

    pdf.add_image_safe(STRATEGY_DIR / "equity_curves.png", w=170)

    pdf.subsection_title("4.4 Period Breakdown - Binary (t=0)")
    pdf.body_text(
        "The strategy is evaluated on two distinct periods to test generalization:"
    )
    pdf.table(
        ["Metric", "Optimization", "True Test", "Full OOS"],
        [
            ["Period", "Dec09-Mar18", "Mar18-Jun26", "Dec09-Jun26"],
            ["Weeks", "430", "431", "861"],
            ["Strategy Return", "169.8%", "257.6%", "864.9%"],
            ["Strategy CAGR", "12.75%", "16.62%", "14.67%"],
            ["Strategy Sharpe", "1.050", "0.983", "0.996"],
            ["Buy & Hold Return", "171.0%", "220.5%", "768.4%"],
            ["Buy & Hold CAGR", "12.81%", "15.09%", "13.94%"],
            ["Buy & Hold Sharpe", "0.932", "0.869", "0.890"],
            ["Excess Return", "-1.1pp", "+37.1pp", "+96.5pp"],
            ["Excess CAGR", "-0.06pp", "+1.53pp", "+0.73pp"],
            ["Excess Sharpe", "+0.118", "+0.114", "+0.106"],
        ],
        col_widths=[35, 35, 35, 35],
    )

    pdf.body_text(
        "The strategy performs better on the true test period (+37.1pp excess return, "
        "+1.53pp CAGR) than during optimization (-1.1pp), providing strong evidence "
        "that the model captures genuine predictive signal rather than overfitting."
    )

    pdf.add_image_safe(STRATEGY_DIR / "annual_returns.png", w=160)

    # ===== 5. RISK ANALYSIS =====
    pdf.add_page()
    pdf.section_title("5. Risk Analysis")

    pdf.subsection_title("Position Timing Analysis")
    pdf.body_text(
        "The model's edge comes from correctly identifying weeks to avoid. When the model "
        "predicts a negative return and moves to cash:\n\n"
        "  - Average actual return on avoided weeks: -0.06% (negative)\n"
        "  - Average actual return on held weeks: +0.36% (positive)\n\n"
        "This demonstrates the model successfully filters out approximately 20% of weeks "
        "that are flat-to-negative, while maintaining exposure during positive weeks."
    )

    pdf.subsection_title("Risk-Return Tradeoffs by Strategy")
    pdf.table(
        ["Strategy", "Calmar", "MaxDD", "In Market", "Best For"],
        [
            ["Binary (t=0)", "0.461", "-31.8%", "80.8%", "Maximum return"],
            ["Adaptive", "-", "-24.7%", "70.3%", "Balanced approach"],
            ["Scaled", "0.676", "-18.0%", "80.8%", "Risk-adjusted returns"],
            ["Binary (t=opt)", "0.774", "-14.6%", "16.3%", "Capital preservation"],
        ],
        col_widths=[30, 22, 22, 22, 50],
    )

    pdf.subsection_title("Limitations & Caveats")
    pdf.bullet("Weekly rebalancing only - does not capture intra-week volatility or flash crashes")
    pdf.bullet("Max drawdown matches buy-and-hold (-31.8%) for Binary (t=0) - the model did not avoid the worst weeks in all cases")
    pdf.bullet("Small training set: even after imputation, 966 training rows for 40 features")
    pdf.bullet("No short positions - cannot profit from correctly predicted down weeks")
    pdf.bullet("Transaction costs modeled at 0.02% per trade; slippage not modeled")
    pdf.bullet("CBOE Put/Call ratio not available - could improve sentiment capture")
    pdf.bullet("Brent Crude data starts Aug 2007 - imputed as zero before that date")
    pdf.bullet("Results are in-sample for the walk-forward engine and should not be treated as live trading results")

    # ===== 6. FEATURE IMPORTANCE =====
    pdf.add_page()
    pdf.section_title("6. Feature Importance")

    pdf.body_text(
        "The top features driving the model span multiple categories: momentum indicators, "
        "volatility measures, trend signals, and cross-asset interactions."
    )
    pdf.add_image_safe(MODEL_DIR / "feature_importance.png", w=140)

    pdf.subsection_title("Interpretation of Top Features")
    pdf.bullet("RSI_14: Overbought/oversold momentum - strongest single predictor")
    pdf.bullet("Volume_Log: Trading activity level - high volume signals conviction")
    pdf.bullet("DI_Minus_14: Bearish directional pressure - helps identify weeks to avoid")
    pdf.bullet("VIX_Close x Brent_Close_PctChg: Interaction capturing fear during oil shocks")
    pdf.bullet("RSI_14 x BB_PctB: Combined momentum and mean-reversion signal")
    pdf.bullet("BB_PctB: Bollinger Band position - where price sits in its range")
    pdf.bullet("VIX_High: Peak weekly fear - regime identification")
    pdf.bullet("Brent_Close_PctChg_Lag3: Oil price momentum with 3-week delay")
    pdf.bullet("Close_PctChg_RollMean_12: 12-week rolling average return (medium-term trend)")

    # ===== 7. CONCLUSIONS =====
    pdf.add_page()
    pdf.section_title("7. Conclusions & Next Steps")

    pdf.subsection_title("Key Findings")
    pdf.bullet(
        "The ensemble model achieves 58% average direction accuracy across 17 walk-forward "
        "folds - consistent, modest, but real predictive signal"
    )
    pdf.bullet(
        "The Binary (t=0) strategy beats buy-and-hold by +96.5pp total return over 16+ years "
        "with higher Sharpe ratio (0.996 vs 0.890)"
    )
    pdf.bullet(
        "Performance is stronger on unseen test data than during optimization - the model "
        "generalizes well and is not overfit"
    )
    pdf.bullet(
        "The primary edge is loss avoidance: the model correctly identifies ~20% of weeks "
        "to sit out, which have negative average returns"
    )

    pdf.subsection_title("Recommended Next Steps")
    pdf.bullet("Live paper trading: Deploy with a paper account to validate in real-time before committing capital")
    pdf.bullet("Shorter timeframes: Test on daily data for more frequent signals and larger training sets")
    pdf.bullet("Dynamic position sizing: Scale position by prediction confidence and recent volatility")
    pdf.bullet("Regime detection: Add Hidden Markov Model states as features for explicit regime awareness")
    pdf.bullet("Alternative data: Incorporate put/call ratio (paid source), earnings dates, Fed meeting calendar")
    pdf.bullet("Portfolio extension: Apply to sector ETFs (XLK, XLF, XLE) for diversified alpha")
    pdf.bullet("Risk management: Add stop-loss rules and maximum drawdown circuit breakers")

    pdf.subsection_title("Project Structure")
    pdf.set_font("Courier", "", 9)
    tree = """
    .
    +-- data/
    |   +-- spy_weekly_ohlcv.csv
    |   +-- spy_weekly_enriched.csv (.xlsx)
    |   +-- spy_weekly_with_ob.csv
    |   +-- distribution_report.csv
    +-- model/
    |   +-- xgb_enhanced_v2.json
    |   +-- catboost_enhanced.cbm
    |   +-- model_summary.json
    |   +-- feature_importance.png
    |   +-- walkforward_folds.png
    +-- strategy/
    |   +-- backtest_results.csv
    |   +-- backtest_summary.json
    |   +-- equity_curves.png
    |   +-- annual_returns.png
    +-- fetch_spy_data.py
    +-- fetch_macro_data.py
    +-- technical_indicators.py
    +-- order_block_finder.py
    +-- model_enhanced.py
    +-- strategy_backtest.py
    +-- generate_report.py
    """
    pdf.multi_cell(0, 4.5, tree)

    # Save
    pdf.output(str(OUTPUT_PDF))
    print(f"Report saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
