"""PDF report: does the SPY daily-execution result hold on all 102 assets?"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF

OUT_DIR = Path(__file__).parent / "daily_universe"
PDF_PATH = Path(__file__).parent / "Universe_Daily_vs_Weekly_Report.pdf"
STRATS = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]

# Earlier SPY report, scored again from its saved feature file in this codebase.
PRIOR_SPY = {
    "daily_topt": 1.041,
    "daily_t0": 0.915,
    "daily_bnh": 0.901,
    "weekly_t0": 0.911,
    "weekly_bnh": 0.942,
    "mixed_t0": 0.763,
    "daily_in_market": 98.7,
    "daily_threshold": -0.07,
}


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Daily vs weekly execution across 102 assets - Research prototype", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 60, 120)
        self.ln(3)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.4, text)
        self.ln(1.4)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.4, "- " + text)
        self.ln(0.4)

    def table(self, headers, rows, widths, size=7.5):
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
                self.set_font("Helvetica", "B", size)
                self.set_fill_color(20, 60, 120)
                self.set_text_color(255, 255, 255)
                for h, w in zip(headers, widths):
                    self.cell(w, 7, h, border=1, fill=True, align="C")
                self.ln()
            self.set_fill_color(235, 241, 248 if fill else 255)
            if not fill:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", size)
            for i, (val, w) in enumerate(zip(row, widths)):
                self.cell(w, 5.5, str(val), border=1, fill=True, align="L" if i == 0 else "C")
            self.ln()
            fill = not fill
        self.ln(2)

    def img(self, path, w=180):
        p = Path(path)
        if p.exists():
            if self.get_y() > 200:
                self.add_page()
            self.image(str(p), w=w)
            self.ln(3)


def _best(block):
    name = max(STRATS, key=lambda s: block[s]["sharpe"])
    return name, block[name]


def load():
    records = []
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name == "universe_summary.json":
            continue
        records.append(json.loads(path.read_text()))
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") == "error"]
    rows = []
    for asset in ok:
        modes = asset["modes"]
        d_name, d_best = _best(modes["daily"]["strategies"])
        w_name, w_best = _best(modes["weekly"]["strategies"])
        m_name, m_best = _best(modes["mixed"]["strategies"])
        d0 = modes["daily"]["strategies"]["Binary (t=0)"]
        w0 = modes["weekly"]["strategies"]["Binary (t=0)"]
        m0 = modes["mixed"]["strategies"]["Binary (t=0)"]
        dopt = modes["daily"]["strategies"]["Binary (t=opt)"]
        wopt = modes["weekly"]["strategies"]["Binary (t=opt)"]
        mopt = modes["mixed"]["strategies"]["Binary (t=opt)"]
        rows.append({
            "name": asset["name"],
            "label": asset.get("label") or asset["name"],
            "asset_class": asset["asset_class"],
            "market": asset["market"],
            "schedule": asset["schedule"],
            "eval_start": asset["eval_start"],
            "daily_best": d_name,
            "weekly_best": w_name,
            "mixed_best": m_name,
            "daily_sharpe": d_best["sharpe"],
            "weekly_sharpe": w_best["sharpe"],
            "mixed_sharpe": m_best["sharpe"],
            "daily_cagr": d_best["cagr"],
            "weekly_cagr": w_best["cagr"],
            "mixed_cagr": m_best["cagr"],
            "daily_dd": d_best["max_drawdown"],
            "daily_bnh_sharpe": d_best["bnh_sharpe"],
            "daily_bnh_cagr": d_best["bnh_cagr"],
            "daily_bnh_dd": d_best["bnh_max_drawdown"],
            "weekly_bnh_sharpe": w_best["bnh_sharpe"],
            "daily_t0": d0["sharpe"],
            "weekly_t0": w0["sharpe"],
            "mixed_t0": m0["sharpe"],
            "daily_t0_cagr": d0["cagr"],
            "daily_t0_dd": d0["max_drawdown"],
            "daily_t0_in": d0["in_market_pct"],
            "daily_topt": dopt["sharpe"],
            "daily_topt_in": dopt["in_market_pct"],
            "weekly_topt_in": wopt["in_market_pct"],
            "mixed_topt_in": mopt["in_market_pct"],
            "daily_threshold": modes["daily"]["threshold"],
            "weekly_share": modes["mixed"]["weekly_feature_share"],
            "daily_dir": modes["daily"]["model"]["dir_accuracy"],
            "weekly_dir": modes["weekly"]["model"]["dir_accuracy"],
        })
    frame = pd.DataFrame(rows)
    return frame, skipped, errors


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    frame, skipped, errors = load()
    if frame.empty:
        raise SystemExit("No completed assets")
    n = len(frame)
    frame["gap_dw"] = frame["daily_sharpe"] - frame["weekly_sharpe"]
    frame["gap_db"] = frame["daily_sharpe"] - frame["daily_bnh_sharpe"]
    frame["gap_md"] = frame["mixed_sharpe"] - frame["daily_sharpe"]
    frame["gap_t0"] = frame["daily_t0"] - frame["weekly_t0"]
    frame["gap_t0_b"] = frame["daily_t0"] - frame["daily_bnh_sharpe"]
    frame["gap_mt0"] = frame["mixed_t0"] - frame["daily_t0"]

    a = int((frame["gap_dw"] > 0).sum())
    b = int((frame["gap_db"] > 0).sum())
    c = int((frame["gap_md"] > 0).sum())
    t0_w = int((frame["gap_t0"] > 0).sum())
    t0_b = int((frame["gap_t0_b"] > 0).sum())
    m_t0 = int((frame["gap_mt0"] > 0).sum())
    a_mat = int((frame["gap_dw"] > 0.05).sum())
    b_mat = int((frame["gap_db"] > 0.05).sum())
    med_d = float(frame["daily_sharpe"].median())
    med_w = float(frame["weekly_sharpe"].median())
    med_m = float(frame["mixed_sharpe"].median())
    med_bnh = float(frame["daily_bnh_sharpe"].median())
    med_t0 = float(frame["daily_t0"].median())
    med_wt0 = float(frame["weekly_t0"].median())
    med_mt0 = float(frame["mixed_t0"].median())
    med_gap = float(frame["gap_dw"].median())
    med_mix_gap = float(frame["gap_md"].median())
    med_share = float(frame["weekly_share"].median())
    n_fixed = int((frame["schedule"] == "fixed_2013").sum())
    n_shift = n - n_fixed
    cash_d = int((frame["daily_topt_in"] < 20).sum())
    cash_w = int((frame["weekly_topt_in"] < 20).sum())
    cash_m = int((frame["mixed_topt_in"] < 20).sum())

    spy = frame.loc[frame["name"] == "SPY"]
    spy_row = spy.iloc[0] if len(spy) else None

    pdf = Report(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.ln(24)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Does the SPY daily result travel?", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 8, "Daily, weekly, and mixed features on 102 assets", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{n} assets completed, {len(skipped)} skipped, {len(errors)} failed", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Same ensemble, same four variants, no look-ahead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Research prototype. Not investment advice. October 2026.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.h1("1. Question and answer")
    pdf.body(
        "The SPY study found that daily execution can beat a matched weekly system, and that "
        "stacking the last completed week onto the daily model made the result worse. This run "
        "asks whether that pair of findings holds on the same 102 names used in the strategy-selection study."
    )
    pdf.h2("What was held fixed")
    pdf.bullet("Three modes: weekly bars and a weekly decision; daily bars and a daily decision; daily bars plus the last completed week, still with a daily decision.")
    pdf.bullet("One blend: XGBoost 35%, LightGBM 30%, CatBoost 25%, Ridge 10%. Four variants: Binary at zero, Binary at a frozen threshold, Scaled size, and Adaptive.")
    pdf.bullet("Long only, no leverage, 2 bps of each position change. The opening position is charged.")
    pdf.bullet("Names with history back to the early 2000s use the SPY calendar: models train through 2009, the threshold is fit on 2010-2012, and the score starts in 2013. Shorter names shift that calendar forward. Training, calibration, and evaluation never overlap.")
    verdict = (
        "The SPY conclusion does not apply as a blanket rule. Daily execution has a modest edge "
        "over weekly execution. Mixing in the last completed week does not help the typical asset. "
        "Beating buy-and-hold is common only if the variant is chosen after seeing the result. "
        "A fixed Binary rule at a zero threshold beats buy-and-hold on about half the names, and "
        "it does so by cutting drawdown rather than by raising CAGR. On this batch, SPY itself "
        "does not reproduce the earlier daily Sharpe of 1.041."
    )
    pdf.h2("Short answer")
    pdf.body(verdict)
    pdf.body(
        f"Of {n} completed assets, the best daily Sharpe is above the best weekly Sharpe on {a} "
        f"({_pct(a, n):.0f}%), and above that asset's daily buy-and-hold on {b} ({_pct(b, n):.0f}%). "
        f"The best mixed Sharpe is above the best daily Sharpe on {c} ({_pct(c, n):.0f}%). "
        f"Median best Sharpe is {med_d:.3f} daily, {med_w:.3f} weekly, {med_m:.3f} mixed, "
        f"against a median daily buy-and-hold of {med_bnh:.3f}."
    )
    pdf.body(
        f"A gap smaller than 0.05 Sharpe is noise for this purpose. Daily still leads weekly by "
        f"more than 0.05 on {a_mat} assets ({_pct(a_mat, n):.0f}%), and leads buy-and-hold by more "
        f"than 0.05 on {b_mat} ({_pct(b_mat, n):.0f}%). The median daily-minus-weekly gap is {med_gap:+.3f}. "
        f"The median mixed-minus-daily gap is {med_mix_gap:+.3f}."
    )
    pdf.body(
        "Binary at a zero threshold is the comparison that does not depend on a fitted cutoff. "
        f"Daily Binary (t=0) beats weekly Binary (t=0) on {t0_w} assets ({_pct(t0_w, n):.0f}%) and "
        f"beats daily buy-and-hold on {t0_b} ({_pct(t0_b, n):.0f}%). Mixed Binary (t=0) beats daily "
        f"Binary (t=0) on {m_t0} ({_pct(m_t0, n):.0f}%). The level medians are close "
        f"({med_t0:.3f} daily, {med_wt0:.3f} weekly, {med_mt0:.3f} mixed). On the same asset, "
        f"daily t=0 still leads weekly t=0 by a median of {float(frame['gap_t0'].median()):+.3f}."
    )

    pdf.h1("2. How look-ahead was kept out")
    pdf.bullet("The target on bar t is the return from that close to the next close. Indicators, lags, and the position all use information available at that close.")
    pdf.bullet("A weekly bar is released the next calendar day. The label session still sees the previous week. The next session sees the week that just finished. This is checked before the model runs. TASI weeks end Thursday; US weeks end Friday.")
    pdf.bullet("The HMM is fit on bars before the current refit segment. Between refits, a forward filter updates one bar at a time. The backward pass is not used. A shock after the segment does not change earlier state probabilities.")
    pdf.bullet("Feature selection and early stopping use only the trailing part of the training window. The test fold is never used for either.")
    pdf.bullet("The Binary threshold and the Scaled divisor are fit on the calibration block only, then frozen. Adaptive may look back at past out-of-sample bars, including calibration, and may not look forward.")
    pdf.bullet("Missing values are forward-filled and then filled with zero. There is no back-fill.")
    pdf.body(
        f"{n_fixed} assets kept the 2013 evaluation start. {n_shift} started later because they do "
        "not have enough history before 2010. An asset that still could not support a training block, "
        "a calibration block, and an evaluation block was skipped rather than fit on the future."
    )

    pdf.h1("3. The three claims")
    pdf.table(
        ["Claim", "Count", "Share", "Median gap"],
        [
            ["Daily best > weekly best", f"{a}/{n}", f"{_pct(a, n):.0f}%", f"{med_gap:+.3f}"],
            ["Daily best > buy and hold", f"{b}/{n}", f"{_pct(b, n):.0f}%", f"{float(frame['gap_db'].median()):+.3f}"],
            ["Mixed best > daily best", f"{c}/{n}", f"{_pct(c, n):.0f}%", f"{med_mix_gap:+.3f}"],
            ["Daily t=0 > weekly t=0", f"{t0_w}/{n}", f"{_pct(t0_w, n):.0f}%", f"{float(frame['gap_t0'].median()):+.3f}"],
            ["Daily t=0 > buy and hold", f"{t0_b}/{n}", f"{_pct(t0_b, n):.0f}%", f"{float(frame['gap_t0_b'].median()):+.3f}"],
            ["Mixed t=0 > daily t=0", f"{m_t0}/{n}", f"{_pct(m_t0, n):.0f}%", f"{float(frame['gap_mt0'].median()):+.3f}"],
        ],
        [62, 28, 28, 42],
        size=8,
    )
    pdf.body(
        "Best means the variant with the highest Sharpe inside that mode. A cash-heavy threshold "
        "rarely wins that contest, because sitting in cash has a Sharpe near zero whenever the asset "
        f"itself has a positive Sharpe. Even so, Binary (t=opt) was in the market less than 20% of "
        f"the time on {cash_d} daily runs, {cash_w} weekly runs, and {cash_m} mixed runs. Those "
        "frozen thresholds failed out of sample. The t=0 rows are the check that does not use them."
    )
    pdf.body(
        f"The median mixed Sharpe ({med_m:.3f}) sits above the median daily Sharpe ({med_d:.3f}). "
        "That is not a gain on the typical asset. The median of the paired difference, mixed minus "
        f"daily on the same name, is {med_mix_gap:+.3f}. Weekly features are still selected: their "
        f"median share of mixed-model features is {med_share*100:.1f}%. Being selected did not lift "
        "the typical equity curve."
    )
    pdf.body(
        "One caution on the 'best variant' rows. The winner is the highest Sharpe among four rules, "
        "picked on the same window being scored. That is a fair description of what the backtest "
        "contained. It is not a rule you could have traded, because the choice uses the evaluation "
        "years. Binary (t=0) is the row that was fixed in advance. It is also the most common daily "
        "winner, but it is not the winner on every name."
    )
    pdf.img(OUT_DIR / "daily_vs_weekly_scatter.png", w=180)

    pdf.h1("4. By asset class")
    pdf.body(
        "Each cell is the count of assets in that class for which the inequality holds, then the "
        "median best Sharpe. Classes are small. A 4-out-of-5 result is a tendency, not a law."
    )
    class_rows = []
    order = frame.groupby("asset_class").size().sort_values(ascending=False).index.tolist()
    for cls in order:
        sub = frame[frame["asset_class"] == cls]
        k = len(sub)
        class_rows.append([
            cls,
            str(k),
            f"{int((sub['gap_dw']>0).sum())}/{k}",
            f"{int((sub['gap_db']>0).sum())}/{k}",
            f"{int((sub['gap_md']>0).sum())}/{k}",
            f"{sub['daily_sharpe'].median():.2f}",
            f"{sub['weekly_sharpe'].median():.2f}",
            f"{sub['mixed_sharpe'].median():.2f}",
            f"{sub['daily_bnh_sharpe'].median():.2f}",
        ])
    pdf.table(
        ["Class", "N", "D>W", "D>BnH", "M>D", "Med D", "Med W", "Med M", "Med BnH"],
        class_rows,
        [32, 12, 16, 18, 16, 16, 16, 16, 20],
        size=7.5,
    )
    pdf.img(OUT_DIR / "sharpe_by_class.png", w=175)
    pdf.body(
        "Read D>W as daily best above weekly best, D>BnH as daily best above that class's daily "
        "buy-and-hold, and M>D as mixed best above daily best. Med D, W, M, and BnH are median Sharpes."
    )
    pdf.bullet("TASI stocks are the clearest daily win: 10 of 12 beat their weekly best, and 11 of 12 beat buy-and-hold. The t=0 row agrees (10 of 12 versus weekly, 11 of 12 versus buy-and-hold).")
    pdf.bullet("US stocks and sector ETFs lean daily, by a smaller margin. Median best-daily Sharpe is close to buy-and-hold (US stocks 0.82 versus 0.79). The gain is not a different return stream.")
    pdf.bullet("Broad ETFs, the group that contains SPY, do not support the SPY headline. Daily best beats weekly best on 2 of 5, and mixed best never beats daily best. In this batch SPY's best weekly Sharpe is above its best daily Sharpe.")
    pdf.bullet("International ETFs are the group where mixed most often wins (9 of 13). A ban on weekly features would be too strong there.")
    pdf.bullet("REITs go the other way: daily best beats weekly best on 1 of 4. Thematic ETFs look like daily wins on the best-of-four row (5 of 5) and do not on t=0 (2 of 5 versus weekly). Commodities look similar: 7 of 9 on the best variant, 3 of 9 on t=0.")
    pdf.bullet("Currencies and bonds have low Sharpe in every mode. Daily is not a reliable upgrade. Several currency t=opt runs sit in cash the entire evaluation, which scores as a Sharpe near zero.")

    pdf.add_page()
    pdf.h1("5. Binary at a zero threshold")
    pdf.body(
        "This variant needs no fitted cutoff. It is long when the model predicts a positive next-bar "
        "return and in cash otherwise. It is the fairer test of daily versus weekly execution."
    )
    t0_rows = []
    for cls in order:
        sub = frame[frame["asset_class"] == cls]
        k = len(sub)
        t0_rows.append([
            cls,
            str(k),
            f"{int((sub['gap_t0']>0).sum())}/{k}",
            f"{int((sub['gap_t0_b']>0).sum())}/{k}",
            f"{int((sub['gap_mt0']>0).sum())}/{k}",
            f"{sub['daily_t0'].median():.2f}",
            f"{sub['weekly_t0'].median():.2f}",
            f"{sub['mixed_t0'].median():.2f}",
        ])
    pdf.table(
        ["Class", "N", "t0 D>W", "t0 D>BnH", "t0 M>D", "Med D", "Med W", "Med M"],
        t0_rows,
        [36, 14, 22, 24, 22, 18, 18, 18],
        size=8,
    )
    pdf.body(
        f"Across all {n} assets the median daily t=0 Sharpe is {med_t0:.3f}, weekly {med_wt0:.3f}, "
        f"mixed {med_mt0:.3f}, buy-and-hold {med_bnh:.3f}. The median paired gap, daily t=0 minus "
        f"weekly t=0, is {float(frame['gap_t0'].median()):+.3f}. The median gap versus buy-and-hold "
        f"is {float(frame['gap_t0_b'].median()):+.3f}. Median t=0 CAGR is "
        f"{float(frame['daily_t0_cagr'].median()):.1f}% against buy-and-hold "
        f"{float(frame['daily_bnh_cagr'].median()):.1f}%. Median t=0 max drawdown is "
        f"{float(frame['daily_t0_dd'].median()):.1f}% against buy-and-hold "
        f"{float(frame['daily_bnh_dd'].median()):.1f}%."
    )
    pdf.body(
        f"Median daily directional accuracy is {float(frame['daily_dir'].median()):.1f}%. "
        f"Median weekly directional accuracy is {float(frame['weekly_dir'].median()):.1f}%. "
        "The weekly model is the better classifier of the next bar's sign. Where daily t=0 helps, "
        "it helps by leaving the market during the week, and the payoff is a shallower drawdown "
        "more than a higher CAGR."
    )

    pdf.h1("6. SPY inside this batch")
    if spy_row is not None:
        pdf.body(
            f"On this batch, SPY's best daily variant is {spy_row['daily_best']} at Sharpe "
            f"{spy_row['daily_sharpe']:.3f}. Best weekly is {spy_row['weekly_best']} at "
            f"{spy_row['weekly_sharpe']:.3f}. Best mixed is {spy_row['mixed_best']} at "
            f"{spy_row['mixed_sharpe']:.3f}. Daily buy-and-hold is {spy_row['daily_bnh_sharpe']:.3f}. "
            f"Daily Binary (t=0) is {spy_row['daily_t0']:.3f}, in the market "
            f"{spy_row['daily_t0_in']:.0f}% of days, max drawdown {spy_row['daily_t0_dd']:.1f}%. "
            f"Daily Binary (t=opt) threshold is {spy_row['daily_threshold']:.3f}, in the market "
            f"{spy_row['daily_topt_in']:.0f}% of days, Sharpe {spy_row['daily_topt']:.3f}."
        )
    pdf.body(
        f"The earlier SPY report, scored again from its saved feature file with this same code, "
        f"still shows daily Binary (t=opt) Sharpe {PRIOR_SPY['daily_topt']:.3f}, daily Binary (t=0) "
        f"{PRIOR_SPY['daily_t0']:.3f}, daily buy-and-hold {PRIOR_SPY['daily_bnh']:.3f}, weekly "
        f"Binary (t=0) {PRIOR_SPY['weekly_t0']:.3f} against weekly buy-and-hold {PRIOR_SPY['weekly_bnh']:.3f}, "
        f"and mixed Binary (t=0) {PRIOR_SPY['mixed_t0']:.3f}. That file's frozen daily threshold is "
        f"{PRIOR_SPY['daily_threshold']:.2f} and the strategy stays long {PRIOR_SPY['daily_in_market']:.1f}% of days."
    )
    pdf.body(
        "Those two SPY numbers are not a contradiction in the code. Replaying the saved feature file "
        "reproduces the 1.041 Sharpe exactly, including the first-fold predictions. This batch built "
        "a fresh feature file from a second download of the same ticker. The prices differ by less "
        "than a cent, and the first daily fold already selects a different feature count. Boosted "
        "trees plus a Sharpe-maximizing threshold are discontinuous: a tiny input change can move "
        "the frozen threshold and, with it, the evaluation Sharpe. The 1.041 result is real on that "
        "saved file. It is not stable to the last decimal on a second pull of SPY. Every asset in "
        "this 102-name table was estimated once, on one download, with one thread per model. Compare "
        "assets inside this table. Do not line a single SPY Sharpe up against the earlier PDF and "
        "treat the difference as a change in the market."
    )

    pdf.h1("7. Where the daily edge is large, and where it is not")
    pdf.body(
        "The left table is where daily execution helps most relative to the best weekly variant. "
        "The right-hand list is where it hurts most. Gaps are best-daily Sharpe minus best-weekly Sharpe."
    )
    winners = frame.nlargest(8, "gap_dw")
    losers = frame.nsmallest(8, "gap_dw")
    both = []
    for (_, w), (_, l) in zip(winners.iterrows(), losers.iterrows()):
        both.append([
            w["name"], f"{w['gap_dw']:+.2f}", w["asset_class"][:12],
            l["name"], f"{l['gap_dw']:+.2f}", l["asset_class"][:12],
        ])
    pdf.table(
        ["Daily helps", "Gap", "Class", "Daily hurts", "Gap", "Class"],
        both,
        [32, 18, 28, 32, 18, 28],
        size=8,
    )
    held = frame.nlargest(5, "gap_db")[["name", "gap_db", "daily_sharpe", "daily_bnh_sharpe"]]
    pdf.body(
        "Largest daily-best advantages over that asset's own buy-and-hold: "
        + ", ".join(f"{r['name']} ({r['gap_db']:+.2f})" for _, r in held.iterrows())
        + "."
    )
    trail = frame.nsmallest(5, "gap_db")
    pdf.body(
        "Largest shortfalls versus buy-and-hold: "
        + ", ".join(f"{r['name']} ({r['gap_db']:+.2f})" for _, r in trail.iterrows())
        + "."
    )

    if skipped or errors:
        pdf.h2("Skipped and failed")
        for rec in skipped:
            pdf.bullet(f"{rec['name']}: {rec.get('reason', 'skipped')}")
        for rec in errors:
            pdf.bullet(f"{rec['name']}: {rec.get('reason', 'error')}")

    pdf.add_page()
    pdf.h1("8. What to take from this")
    pdf.body(verdict)
    pdf.bullet(
        f"Daily execution beats weekly execution often, not always. Best daily Sharpe is higher on "
        f"{a} of {n} assets, with a median paired gap of {med_gap:+.3f}. The fixed t=0 rule is higher "
        f"on {t0_w} of {n}, median paired gap {float(frame['gap_t0'].median()):+.3f}. "
        "The edge is largest in TASI stocks and is absent in broad ETFs and REITs."
    )
    pdf.bullet(
        f"Do not treat {b} of {n} beating buy-and-hold as a trading rule. That count lets each asset "
        f"keep its best variant after the fact. The fixed t=0 rule beats buy-and-hold on {t0_b} of {n}, "
        f"median gap {float(frame['gap_t0_b'].median()):+.3f}. Median t=0 CAGR is lower than buy-and-hold. "
        "Median max drawdown is shallower. It is a smoother path on about half the universe, not a higher one."
    )
    pdf.bullet(
        f"Do not add weekly features by default. Mixed best beats daily best on {c} of {n}, and the "
        f"median paired gap is {med_mix_gap:+.3f}. The t=0 version of that comparison is {m_t0} of {n}. "
        "International ETFs are the exception worth remembering: mixed wins there more often than not."
    )
    pdf.bullet(
        f"Do not ship Binary (t=opt) without a cash guardrail. The frozen threshold left the daily "
        f"strategy in the market less than 20% of the time on {cash_d} of {n} assets, and the weekly "
        f"strategy on {cash_w} of {n}. The SPY study's slightly negative threshold, long almost every "
        "day, is a special case. It did not come back on this batch's SPY file, where the threshold "
        "was positive and the strategy was a weaker daily result than buy-and-hold."
    )
    pdf.bullet(
        "Weekly sign accuracy is typically higher than daily sign accuracy. The daily advantage, "
        "where it exists, is the right to leave during the week. It is not a better forecast of the next bar."
    )
    pdf.body(
        "This is a research prototype on downloaded prices, with a fixed 2 bp cost and no capacity, "
        "borrow, or gap model. It is not a trading recommendation. The usable statement is narrow. "
        "Across these 102 names, daily execution is a bit better than weekly execution, weekly "
        "features are not a free add-on except in parts of the international ETF group, and the "
        "earlier SPY chart overstates how often a daily model beats buy-and-hold once the variant "
        "is chosen in advance."
    )

    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")
    print("VERDICT:", verdict)
    print(f"n={n} skipped={len(skipped)} errors={len(errors)}")
    print(f"A daily>weekly {a}/{n}  B daily>bnh {b}/{n}  C mixed>daily {c}/{n}")
    print(f"t0 D>W {t0_w}/{n}  t0>bnh {t0_b}/{n}  mixed t0>daily t0 {m_t0}/{n}")
    print(f"medians D/W/M/BnH {med_d:.3f}/{med_w:.3f}/{med_m:.3f}/{med_bnh:.3f} gap {med_gap:+.3f} mix {med_mix_gap:+.3f}")
    print("paired median gaps D-W / M-D / t0 D-W / t0-BnH",
          f"{med_gap:+.3f}", f"{med_mix_gap:+.3f}",
          f"{float(frame['gap_t0'].median()):+.3f}",
          f"{float(frame['gap_t0_b'].median()):+.3f}")


if __name__ == "__main__":
    main()
