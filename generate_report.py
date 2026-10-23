"""
Generate comprehensive PDF report documenting the trading model and strategy.
Updated with HMM regime detection (#10).
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
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 100, 60)
    pdf.cell(0, 8, "v2.0 - With HMM Regime Detection", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Instrument: SPDR S&P 500 ETF (SPY)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Timeframe: Weekly (2000 - 2026)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Strategy: Long-Only, No Leverage", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Prototype - For Research Purposes Only", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "October 2026", align="C", new_x="LMARGIN", new_y="NEXT")

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
        "4. HMM Regime Detection",
        "   4.1 Motivation",
        "   4.2 Model Specification",
        "   4.3 Walk-Forward Fitting",
        "   4.4 Regime Distribution",
        "   4.5 Feature Contribution",
        "5. Trading Strategy",
        "   5.1 Strategy Variants",
        "   5.2 Backtest Methodology",
        "   5.3 Performance Results",
        "   5.4 Adaptive Strategy Deep Dive",
        "6. Risk Analysis",
        "7. Feature Importance",
        "8. Conclusions & Next Steps",
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
        "Version 2.0 introduces Hidden Markov Model (HMM) regime detection as the 10th "
        "enhancement. A 3-state Gaussian HMM identifies bear, sideways, and bull market "
        "regimes from returns and VIX data, fitted walk-forward to maintain temporal integrity. "
        "Three HMM features were selected by importance-based filtering into the final model: "
        "regime duration, bull probability, and lagged regime state."
    )
    pdf.body_text(
        "The Adaptive strategy achieves the best risk-adjusted performance with a Sharpe ratio "
        "of 1.042 (vs 0.890 for buy-and-hold), a maximum drawdown of -24.74% (vs -31.83%), and "
        "a 730.0% total return over the out-of-sample period (Dec 2009 - Jun 2026), while being "
        "invested only 70.8% of the time. The Binary (t=0) strategy delivers the highest absolute "
        "return at 840.8%."
    )

    pdf.subsection_title("Key Results at a Glance")
    pdf.table(
        ["Metric", "Adaptive", "Binary (t=0)", "Buy & Hold"],
        [
            ["Total Return", "730.0%", "840.8%", "768.4%"],
            ["CAGR", "13.63%", "14.50%", "13.94%"],
            ["Sharpe Ratio", "1.042", "0.958", "0.890"],
            ["Max Drawdown", "-24.74%", "-31.83%", "-31.83%"],
            ["Win Rate", "60.3%", "60.2%", "-"],
            ["Time in Market", "70.8%", "89.7%", "100%"],
            ["Trades", "222", "114", "-"],
        ],
        col_widths=[32, 36, 36, 36],
    )

    pdf.body_text(
        "The Adaptive strategy reduces maximum drawdown by 7.1 percentage points relative to "
        "buy-and-hold while delivering a Sharpe premium of +0.152. Its regime-adaptive threshold, "
        "re-optimized every 26 weeks, allows it to dynamically adjust market exposure based on "
        "recent predictive accuracy."
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
    pdf.body_text("The feature set is constructed in layers, totaling 222 raw features before selection:")
    pdf.bullet("Base OHLCV: Open, Low, High, Close, Volume (+ log-transformed volume)")
    pdf.bullet("Percentage changes: Week-over-week returns for all price series")
    pdf.bullet("Lagged features: 1-4 week lags on 18 key signals including HMM outputs (72 features)")
    pdf.bullet("Rolling statistics: 4/8/12-week mean and std of returns, VIX (9 features)")
    pdf.bullet("Interaction terms: 16 cross-variable products including 4 HMM interactions")
    pdf.bullet("Regime indicators: Rule-based (Above SMA-200, Golden Cross, VIX regime, trend direction)")
    pdf.bullet("HMM regime: Bear/Sideways/Bull states, posterior probabilities, duration, change flag (9 features)")
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
        "From 222 candidate features, the top 40 are selected using XGBoost importance (gain). "
        "This reduces the feature/sample ratio from 0.23 to 0.041, significantly reducing "
        "overfitting risk. Feature selection is performed using only training data."
    )
    pdf.body_text(
        "Three HMM features survived selection into the top 40: HMM_Regime_Duration (#24), "
        "HMM_Prob_Bull (#28), and HMM_Regime_Lag4 (#33). This demonstrates the HMM provides "
        "non-redundant information beyond the rule-based regime features."
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
        "  - Average direction accuracy: 58.5% (vs 50% random baseline)\n"
        "  - 15 of 17 folds beat the random baseline\n"
        "  - Best fold: 69.2% direction accuracy\n\n"
        "NaN imputation (forward-fill for indicators, zero-fill for Brent pre-2007) recovers "
        "training rows from 281 to 966, a 3.4x increase in usable training data."
    )
    pdf.add_image_safe(MODEL_DIR / "walkforward_folds.png", w=160)

    # ===== 4. HMM REGIME DETECTION =====
    pdf.add_page()
    pdf.section_title("4. HMM Regime Detection")

    pdf.subsection_title("4.1 Motivation")
    pdf.body_text(
        "Financial markets alternate between distinct behavioral regimes: trending bull markets "
        "with low volatility, choppy sideways periods, and sharp bear sell-offs driven by fear. "
        "Traditional rule-based indicators (e.g., price above SMA-200 or VIX > 25) provide "
        "binary signals but cannot capture the probabilistic, latent nature of these transitions."
    )
    pdf.body_text(
        "A Hidden Markov Model (HMM) addresses this by treating the market regime as a latent "
        "(unobservable) state that generates observable data. The HMM learns the statistical "
        "properties of each regime and the transition dynamics between them from the data "
        "itself, providing posterior probabilities and state duration information that "
        "rule-based indicators cannot offer."
    )

    pdf.subsection_title("4.2 Model Specification")
    pdf.body_text(
        "We use a 3-state Gaussian HMM with full covariance matrices. The observable features "
        "fed to the HMM are:\n\n"
        "  1. Close_PctChg: Weekly percentage return (captures direction and magnitude)\n"
        "  2. VIX_Close: VIX level (captures market fear/complacency)\n\n"
        "The model assumes each market regime generates returns and VIX levels from a bivariate "
        "Gaussian distribution with regime-specific mean and covariance. The Baum-Welch "
        "(Expectation-Maximization) algorithm estimates the emission parameters and the 3x3 "
        "state transition matrix."
    )
    pdf.body_text(
        "After fitting, the three states are sorted by their mean return (the first dimension "
        "of the emission mean vector) in ascending order:\n\n"
        "  - State 0 (Bear):     Lowest mean return, typically high VIX\n"
        "  - State 1 (Sideways): Moderate return, medium VIX\n"
        "  - State 2 (Bull):     Highest mean return, typically low VIX\n\n"
        "This consistent labeling ensures states have the same semantic meaning across different "
        "fitting windows."
    )

    pdf.subsection_title("4.3 Walk-Forward Fitting")
    pdf.body_text(
        "To prevent look-ahead bias, the HMM is fitted walk-forward:\n\n"
        "  - Warmup period: 104 weeks (2 years) before first classification\n"
        "  - Refit frequency: Every 26 weeks (6 months)\n"
        "  - For week t, the model is fitted on weeks [0, t) only\n"
        "  - Between refits, the last fitted model is used for inference\n"
        "  - Total classified weeks: 1,278 out of 1,382\n\n"
        "At each inference step, the Viterbi path is decoded over the full history up to week t "
        "to determine the current state and posterior probabilities. This approach mirrors how "
        "the model would operate in live trading: the HMM would be periodically re-estimated "
        "as new data arrives, and the current regime inferred from the updated model."
    )

    pdf.subsection_title("4.4 Regime Distribution")
    pdf.table(
        ["Regime", "Weeks", "Percentage", "Characteristics"],
        [
            ["Bear (State 0)", "295", "23.1%", "Negative mean return, high VIX"],
            ["Sideways (State 1)", "490", "38.3%", "Near-zero return, moderate VIX"],
            ["Bull (State 2)", "493", "38.6%", "Positive mean return, low VIX"],
        ],
        col_widths=[32, 20, 22, 72],
    )
    pdf.body_text(
        "The regime distribution is intuitive: SPY spends roughly three-quarters of the time "
        "in sideways or bull regimes and one-quarter in bear regimes. The average regime "
        "duration is 16.9 weeks, with the longest sustained regime lasting 130 weeks, "
        "reflecting the extended bull run of 2013-2015."
    )

    pdf.subsection_title("4.5 Feature Contribution")
    pdf.body_text("Nine features are produced per week from the HMM:")
    pdf.bullet("HMM_Regime: Integer state label (0, 1, or 2)")
    pdf.bullet("HMM_Regime_Bear, HMM_Regime_Sideways, HMM_Regime_Bull: One-hot dummy variables")
    pdf.bullet("HMM_Prob_Bear, HMM_Prob_Sideways, HMM_Prob_Bull: Posterior state probabilities")
    pdf.bullet("HMM_Regime_Duration: Consecutive weeks in the current regime")
    pdf.bullet("HMM_Regime_Change: Binary flag (1 if regime changed from prior week)")

    pdf.body_text(
        "Additionally, the HMM outputs are lagged (1-4 weeks) and combined into interaction "
        "terms with other signals (HMM_Prob_Bull x Close_PctChg, HMM_Prob_Bear x VIX_Close, "
        "HMM_Regime x RSI_14, HMM_Prob_Bull x MACD_Hist), totaling 25 HMM-derived features."
    )
    pdf.body_text(
        "Of the 222 total candidate features, 3 HMM features survived importance-based "
        "selection into the top 40:\n\n"
        "  #24. HMM_Regime_Duration - How long the market has been in the current state. "
        "Longer durations may signal increasing probability of regime transition.\n\n"
        "  #28. HMM_Prob_Bull - Posterior probability of being in a bull regime. Provides a "
        "continuous confidence score rather than a hard binary classification.\n\n"
        "  #33. HMM_Regime_Lag4 - The regime state 4 weeks ago. Captures how the recent "
        "regime history influences current-week returns."
    )

    # ===== 5. TRADING STRATEGY =====
    pdf.add_page()
    pdf.section_title("5. Trading Strategy")

    pdf.subsection_title("5.1 Strategy Variants")
    pdf.body_text("Four long-only, no-leverage strategies were tested:")
    pdf.bullet(
        "Binary (t=0): Go 100% long if predicted return > 0%, else 100% cash. "
        "Simple and effective - captures the model's directional signal directly."
    )
    pdf.bullet(
        "Binary (t=opt): Same as above but threshold optimized on first half of OOS data "
        "(optimized threshold = +0.17%). More selective - only in market 37.7% of the time."
    )
    pdf.bullet(
        "Scaled: Position size proportional to prediction magnitude, clipped to [0%, 100%]. "
        "Higher conviction = larger position."
    )
    pdf.bullet(
        "Adaptive: Binary strategy with threshold re-optimized every 26 weeks using recent "
        "OOS performance. Self-adjusting to changing market regimes. Best Sharpe ratio."
    )
    pdf.body_text("Transaction cost: 0.02% round-trip per trade (realistic for weekly SPY trades).")

    pdf.subsection_title("5.2 Backtest Methodology")
    pdf.body_text(
        "All predictions come from walk-forward OOS output - the model never sees the data "
        "it trades on during training. The backtest covers 861 weeks (Dec 2009 - Jun 2026).\n\n"
        "The walk-forward prediction engine uses 26-week steps (vs 52-week in validation) "
        "for finer-grained OOS coverage. Feature selection is performed once using the first "
        "70% of data as pseudo-training. The ensemble is re-trained at each step boundary "
        "using all available past data."
    )

    pdf.subsection_title("5.3 Performance Results - Full OOS Period")
    pdf.table(
        ["Strategy", "Return", "CAGR", "Sharpe", "MaxDD", "WinRate", "InMkt", "Trades"],
        [
            ["Binary (t=0)", "840.8%", "14.50%", "0.958", "-31.83%", "60.2%", "89.7%", "114"],
            ["Binary (t=opt)", "478.9%", "11.19%", "0.965", "-23.84%", "62.8%", "37.7%", "257"],
            ["Scaled", "308.6%", "8.87%", "0.971", "-20.07%", "60.0%", "89.7%", "822"],
            ["Adaptive", "730.0%", "13.63%", "1.042", "-24.74%", "60.3%", "70.8%", "222"],
            ["Buy & Hold", "768.4%", "13.94%", "0.890", "-31.83%", "-", "100%", "-"],
        ],
        col_widths=[26, 18, 16, 16, 17, 17, 14, 17],
    )

    pdf.add_image_safe(STRATEGY_DIR / "equity_curves.png", w=170)

    pdf.subsection_title("5.4 Adaptive Strategy Deep Dive")
    pdf.body_text(
        "The Adaptive strategy emerges as the best risk-adjusted performer (Sharpe 1.042). "
        "Its key innovation is a rolling 26-week lookback that re-optimizes the entry threshold "
        "based on recent prediction quality. This creates a feedback loop: when the model is "
        "performing well, the threshold tightens to capture more trades; when performance "
        "degrades, the threshold widens to reduce exposure."
    )
    pdf.body_text(
        "Key characteristics:\n\n"
        "  - In market 70.8% of the time (avoids 29.2% of weeks)\n"
        "  - Average return when invested: +0.378% per week\n"
        "  - Average return when cash: +0.030% per week (flat weeks avoided)\n"
        "  - Max drawdown -24.74% vs -31.83% for buy-and-hold (7.1pp improvement)\n"
        "  - 222 trades over 861 weeks (~1 position change every 3.9 weeks)"
    )
    pdf.body_text(
        "The Adaptive strategy benefits from HMM features because the HMM regime duration "
        "and bull probability help the model identify periods of regime stability vs. "
        "transition. During stable bull regimes (high HMM_Prob_Bull, long HMM_Regime_Duration), "
        "the model confidently stays invested. During regime transitions, the model pulls "
        "back, reducing drawdown exposure."
    )

    pdf.add_image_safe(STRATEGY_DIR / "annual_returns.png", w=160)

    # ===== 6. RISK ANALYSIS =====
    pdf.add_page()
    pdf.section_title("6. Risk Analysis")

    pdf.subsection_title("Position Timing Analysis")
    pdf.body_text(
        "The Adaptive strategy's edge comes from correctly identifying weeks to avoid. When "
        "the model predicts a negative return (or below threshold) and moves to cash:\n\n"
        "  - Average actual return when invested: +0.378% per week\n"
        "  - Average actual return when in cash: +0.030% per week\n\n"
        "This demonstrates the model successfully filters out approximately 29% of weeks "
        "that are flat-to-negative, while maintaining exposure during positive weeks."
    )

    pdf.subsection_title("Risk-Return Tradeoffs by Strategy")
    pdf.table(
        ["Strategy", "Sharpe", "MaxDD", "In Market", "Best For"],
        [
            ["Adaptive", "1.042", "-24.7%", "70.8%", "Best risk-adjusted"],
            ["Scaled", "0.971", "-20.1%", "89.7%", "Smooth equity curve"],
            ["Binary (t=opt)", "0.965", "-23.8%", "37.7%", "Capital preservation"],
            ["Binary (t=0)", "0.958", "-31.8%", "89.7%", "Maximum return"],
            ["Buy & Hold", "0.890", "-31.8%", "100%", "Benchmark"],
        ],
        col_widths=[30, 20, 22, 22, 50],
    )

    pdf.body_text(
        "All four strategy variants beat buy-and-hold on Sharpe ratio. The Scaled strategy "
        "achieves the lowest maximum drawdown (-20.1%) by sizing positions proportionally to "
        "prediction magnitude, dampening exposure during uncertain periods."
    )

    pdf.subsection_title("Limitations & Caveats")
    pdf.bullet("Weekly rebalancing only - does not capture intra-week volatility or flash crashes")
    pdf.bullet("Max drawdown matches buy-and-hold (-31.8%) for Binary (t=0) - not all downturns avoided")
    pdf.bullet("Small training set: even after imputation, 966 training rows for 40 features")
    pdf.bullet("No short positions - cannot profit from correctly predicted down weeks")
    pdf.bullet("Transaction costs modeled at 0.02% per trade; slippage not modeled")
    pdf.bullet("CBOE Put/Call ratio not available - could improve sentiment capture")
    pdf.bullet("Brent Crude data starts Aug 2007 - imputed as zero before that date")
    pdf.bullet("HMM assumes Gaussian emissions - may underestimate tail risk during fat-tail events")
    pdf.bullet("HMM state count (3) chosen heuristically; BIC-based selection could yield different structure")
    pdf.bullet("Results are in-sample for the walk-forward engine - not live trading results")

    # ===== 7. FEATURE IMPORTANCE =====
    pdf.add_page()
    pdf.section_title("7. Feature Importance")

    pdf.body_text(
        "The top 40 features driving the model span multiple categories. The table below "
        "shows the selected features grouped by type:"
    )

    pdf.subsection_title("Selected Features by Category")
    pdf.table(
        ["Category", "Count", "Top Features"],
        [
            ["Momentum/Oscillators", "9", "RSI_14 (#2), RSI lags, MACD_Signal"],
            ["Volatility/VIX", "8", "VIX_Close lags, VIX_High, VIX_EMA_20"],
            ["Trend (DI/ADX)", "6", "DI_Plus_14, DI_Minus_14, ADX lags"],
            ["Price Returns", "4", "Low_PctChg, High_PctChg, Open lag"],
            ["Cross-Asset", "4", "Brent_BB_Upper (#1), Brent PctChg"],
            ["Interactions", "3", "VIX x Brent, ADX x DI, VIX x RSI"],
            ["HMM Regime", "3", "Duration (#24), Prob_Bull (#28), Lag4 (#33)"],
            ["Rolling Stats", "2", "Close PctChg RollMean 4w & 8w"],
            ["Volume", "1", "OBV"],
        ],
        col_widths=[35, 15, 95],
    )

    pdf.add_image_safe(MODEL_DIR / "enhanced_comparison.png", w=160)

    pdf.subsection_title("HMM Feature Interpretation")
    pdf.body_text(
        "The three HMM features that survived selection provide distinct information:\n\n"
        "HMM_Regime_Duration (#24 by importance): Captures how long the market has stayed "
        "in the current regime. Extended bull regimes may signal increasing complacency and "
        "higher reversion risk. Extended bear regimes may indicate capitulation approaching. "
        "This is a genuinely novel signal that no rule-based indicator can replicate.\n\n"
        "HMM_Prob_Bull (#28): Provides a continuous confidence measure (0 to 1) of being in a "
        "bull regime, rather than the binary Above_SMA_200 indicator. When the HMM probability "
        "disagrees with rule-based signals, it often foreshadows regime transitions.\n\n"
        "HMM_Regime_Lag4 (#33): The regime state 4 weeks prior. This temporal feature captures "
        "mean-reversion and momentum effects at the regime level - a market that was in a bear "
        "regime a month ago may behave differently from one continuously in a bull regime, "
        "even if both show the same current indicators."
    )

    # ===== 8. CONCLUSIONS =====
    pdf.add_page()
    pdf.section_title("8. Conclusions & Next Steps")

    pdf.subsection_title("Key Findings")
    pdf.bullet(
        "The ensemble model achieves 58.5% average direction accuracy across 17 walk-forward "
        "folds - consistent, modest, but real predictive signal"
    )
    pdf.bullet(
        "The Adaptive strategy achieves the best risk-adjusted returns (Sharpe 1.042 vs "
        "0.890 for buy-and-hold) with 7.1pp lower maximum drawdown"
    )
    pdf.bullet(
        "The Binary (t=0) strategy delivers the highest absolute return (840.8% vs 768.4% "
        "buy-and-hold) over the 16+ year OOS period"
    )
    pdf.bullet(
        "HMM regime detection provides non-redundant predictive signal: 3 HMM features "
        "selected alongside traditional technical and macro indicators"
    )
    pdf.bullet(
        "The primary edge is loss avoidance: the model identifies 10-29% of weeks to sit out, "
        "which have near-zero or negative average returns"
    )

    pdf.subsection_title("Enhancement History")
    pdf.table(
        ["#", "Enhancement", "Impact"],
        [
            ["1", "NaN imputation", "Training data 281 -> 966 rows"],
            ["2", "Feature selection (top 40)", "Feature/sample 0.20 -> 0.041"],
            ["3", "Walk-forward validation", "Eliminates look-ahead bias"],
            ["4", "Multi-horizon target", "1/2/4-week return blend"],
            ["5", "Rule-based regime features", "Binary market state signals"],
            ["6", "Time features", "Seasonality capture"],
            ["7", "CatBoost added", "Ensemble diversity"],
            ["8", "Macro expansion", "TNX, DXY, Gold added"],
            ["9", "Probability calibration", "Calibrated P(up) output"],
            ["10", "HMM regime detection", "Probabilistic regime states"],
        ],
        col_widths=[10, 55, 80],
    )

    pdf.subsection_title("Recommended Next Steps")
    pdf.bullet("Live paper trading: Deploy with a paper account to validate in real-time before committing capital")
    pdf.bullet("Shorter timeframes: Test on daily data for more frequent signals and larger training sets")
    pdf.bullet("Dynamic position sizing: Scale position by prediction confidence and recent volatility")
    pdf.bullet("Alternative data: Incorporate put/call ratio (paid source), earnings dates, Fed meeting calendar")
    pdf.bullet("Portfolio extension: Apply to sector ETFs (XLK, XLF, XLE) for diversified alpha")
    pdf.bullet("Risk management: Add stop-loss rules and maximum drawdown circuit breakers")
    pdf.bullet("HMM extensions: Test 4-state model, regime-conditional model weights, or regime-switching GARCH")

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
    |   +-- enhanced_comparison.png
    |   +-- walkforward_folds.png
    |   +-- probability_calibration.png
    +-- strategy/
    |   +-- backtest_results.csv
    |   +-- backtest_summary.json
    |   +-- equity_curves.png
    |   +-- annual_returns.png
    +-- fetch_spy_data.py
    +-- fetch_macro_data.py
    +-- technical_indicators.py
    +-- order_block_finder.py
    +-- hmm_regime.py              (NEW - HMM regime detection)
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
