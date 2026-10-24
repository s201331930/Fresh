"""PDF report for the SPY daily vs weekly vs mixed execution study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF

STUDY = Path(__file__).parent / "daily_study"
OUT = Path(__file__).parent / "SPY_Daily_vs_Weekly_Report.pdf"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "SPY Daily Execution Study - Research prototype", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 60, 120)
        self.ln(4)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.4, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.4, "  * " + text)

    def table(self, headers, rows, widths, size=7.5):
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        fill = False
        for row in rows:
            self.set_fill_color(240, 245, 250 if fill else 255)
            if not fill:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", size)
            for val, w in zip(row, widths):
                align = "L" if val == row[0] or (len(row) > 1 and val == row[1] and not str(val)[:1].isdigit() and str(val)[:1] not in "-+") else "R"
                if row.index(val) == 0:
                    align = "L"
                self.cell(w, 5.6, str(val), border=1, fill=True, align="C" if row.index(val) else "L")
            self.ln()
            fill = not fill
        self.ln(2)

    def img(self, path, w=170):
        p = Path(path)
        if p.exists():
            self.image(str(p), w=w)
            self.ln(3)


def _window_metrics(pos, rets, label_cost=0.0002):
    pos = pos.astype(float)
    turn = pos.diff().abs()
    turn.iloc[0] = pos.iloc[0]
    strat = pos * rets - turn * label_cost
    eq = (1 + strat).cumprod()
    bnh = (1 + rets).cumprod()
    years = max((strat.index[-1] - strat.index[0]).days / 365.25, 0.25)

    def cagr(series):
        return float(series.iloc[-1] ** (1 / years) - 1)

    def sharpe(series):
        vol = float(series.std())
        return float(series.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    def dd(series):
        return float((series / series.cummax() - 1).min())

    return sharpe(strat), cagr(eq) * 100, dd(eq) * 100, sharpe(rets), cagr(bnh) * 100, dd(bnh) * 100


def main():
    summary = json.loads((STUDY / "comparison_summary.json").read_text())
    modes = summary["modes"]
    order = ["weekly", "daily", "mixed"]
    strats = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]

    daily = pd.read_csv(STUDY / "daily_predictions.csv", index_col=0, parse_dates=True)
    ev = daily[daily.index >= "2013-01-01"].copy()
    rets = ev["actual"] / 100.0
    thr = modes["daily"]["threshold"]
    pos_opt = (ev["prediction"] > thr).astype(float)
    pos_t0 = (ev["prediction"] > 0).astype(float)
    forced = pos_opt.copy()
    forced.loc["2020-02-15":"2020-04-30"] = 1.0
    ex2020 = ev.index.year != 2020

    sens = {
        "t=opt, full window": _window_metrics(pos_opt, rets),
        "t=opt, long through COVID": _window_metrics(forced, rets),
        "t=opt, 2020 removed": _window_metrics(pos_opt.loc[ex2020], rets.loc[ex2020]),
        "t=0, full window": _window_metrics(pos_t0, rets),
        "t=0, 2020 removed": _window_metrics(pos_t0.loc[ex2020], rets.loc[ex2020]),
    }

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)

    pdf.add_page()
    pdf.ln(28)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "SPY Daily Execution Study", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 8, "Weekly features, daily features, and the mix of both", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Same ensemble. Same four strategy variants. Daily trading allowed.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Evaluation window: 2013-01-01 through October 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Research prototype. Not investment advice. October 2026.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.h1("1. Question and answer")
    pdf.body(
        "The weekly SPY system decides once a week. This study asks whether building the same "
        "features on daily bars, and allowing a trade every day, improves the result. It also "
        "asks whether the daily model should see the latest completed week as extra context."
    )
    pdf.h2("What was compared")
    pdf.bullet("Weekly: weekly bars, next-week target, one decision per week.")
    pdf.bullet("Daily: daily bars, next-day target, a decision every session.")
    pdf.bullet("Mixed: daily bars plus the last completed week's features, next-day target, daily decisions.")
    pdf.body(
        "All three use the same blend: XGBoost 35%, LightGBM 30%, CatBoost 25%, Ridge 10%. "
        "The four strategy variants are unchanged: Binary at a zero threshold, Binary at a "
        "frozen optimized threshold, Scaled position size, and Adaptive with a rolling threshold."
    )
    pdf.h2("Short answer")
    d_opt = modes["daily"]["strategies"]["Binary (t=opt)"]
    d_bnh = d_opt["bnh_sharpe"]
    w_best_name = max(strats, key=lambda s: modes["weekly"]["strategies"][s]["sharpe"])
    w_best = modes["weekly"]["strategies"][w_best_name]
    pdf.body(
        f"Daily execution improves SPY. Binary (t=opt) on daily bars is the best result in the "
        f"study: Sharpe {d_opt['sharpe']:.3f}, CAGR {d_opt['cagr']:.2f}%, max drawdown "
        f"{d_opt['max_drawdown']:.1f}%. Buy-and-hold over the same daily window is Sharpe "
        f"{d_bnh:.3f}, CAGR {d_opt['bnh_cagr']:.2f}%, max drawdown {d_opt['bnh_max_drawdown']:.1f}%."
    )
    pdf.body(
        f"The matched weekly system does not beat its own buy-and-hold. Its best variant, "
        f"{w_best_name}, has Sharpe {w_best['sharpe']:.3f} against a weekly buy-and-hold Sharpe "
        f"of {w_best['bnh_sharpe']:.3f}."
    )
    pdf.body(
        "Adding weekly features on top of the daily model does not help. The mixed model is "
        "worse than the daily-only model on every variant. Weekly context was used (about "
        f"{modes['mixed']['weekly_feature_share']*100:.0f}% of selected features), and it diluted the signal."
    )
    pdf.body(
        "The daily edge is a crash filter, not a high-accuracy day-trading signal. Directional "
        f"accuracy on daily bars is {modes['daily']['model']['dir_accuracy']:.1f}%. The strategy "
        "stays long almost every day and steps aside on a small number of sharp down days. "
        "March 2020 is the largest single episode. The edge does not disappear if that window "
        "is removed, but it gets smaller. Details are in Section 5."
    )

    pdf.h1("2. How look-ahead was kept out")
    pdf.bullet("The target on bar t is the return from that close to the next close. Features on bar t use information available at that close.")
    pdf.bullet("A weekly bar is released the next calendar day. Friday's session still sees the previous week. Monday sees the week that just finished. This was checked on 29 Fridays before the model ran.")
    pdf.bullet("The HMM is fit only on bars before the current refit segment, then a forward filter updates the state one bar at a time. A shock in the future does not change earlier states.")
    pdf.bullet("Each walk-forward fold selects features and stops training on the tail of the training window. The test bars are never used for early stopping or feature selection.")
    pdf.bullet("Predictions begin in 2010. The Binary threshold and the Scaled divisor are fit on 2010-2012 only, then frozen. Every number in the tables is from 2013 onward.")
    pdf.bullet("Transaction cost is 2 bps of the position change. Costs of 1 bp and 5 bps are reported as a sensitivity. The initial position is charged, so entry is not free.")
    pdf.body(
        "This weekly run is the control for the daily test. It is stricter than the earlier SPY "
        "report: no test-set early stopping, and the threshold is not fit on the years being scored. "
        "Do not line these weekly Sharpes up against that earlier 1.04 figure. The three modes here "
        "share one codebase, one cost, and one evaluation clock."
    )

    pdf.add_page()
    pdf.h1("3. Results on the common window")
    pdf.body(
        "CAGR uses calendar time, so a weekly path and a daily path can be compared directly. "
        "Sharpe is annualized with sqrt(52) or sqrt(252). Each mode has its own buy-and-hold, "
        "because Friday-to-Friday sampling is not identical to every session."
    )
    rows = []
    for mode in order:
        block = modes[mode]["strategies"]
        for name in strats:
            m = block[name]
            rows.append([
                mode, name.replace("Binary ", "Bin "),
                f"{m['sharpe']:.3f}", f"{m['cagr']:.2f}", f"{m['max_drawdown']:.1f}",
                f"{m['in_market_pct']:.0f}", f"{m['turnover_per_year']:.1f}",
                f"{m['n_trades']}",
            ])
        b = block["Binary (t=0)"]
        rows.append([
            mode, "Buy & Hold",
            f"{b['bnh_sharpe']:.3f}", f"{b['bnh_cagr']:.2f}", f"{b['bnh_max_drawdown']:.1f}",
            "100", "-", "-",
        ])
    pdf.table(
        ["Mode", "Strategy", "Sharpe", "CAGR%", "MaxDD%", "In mkt", "Turn/yr", "Trades"],
        rows,
        [22, 32, 18, 18, 20, 16, 18, 18],
        size=7,
    )
    pdf.body(
        "Turnover is the sum of absolute position changes per year. A round trip from cash to "
        "fully long and back counts as 2. Binary strategies do not trade every day. Scaled does."
    )
    pdf.img(STUDY / "comparison.png", w=175)

    pdf.h2("Reading the table")
    pdf.bullet(
        f"Daily Binary (t=opt) is the only variant that beats buy-and-hold on Sharpe, CAGR, and drawdown at once "
        f"({d_opt['sharpe']:.3f} / {d_opt['cagr']:.2f}% / {d_opt['max_drawdown']:.1f}% versus "
        f"{d_opt['bnh_sharpe']:.3f} / {d_opt['bnh_cagr']:.2f}% / {d_opt['bnh_max_drawdown']:.1f}%). "
        f"It is long {d_opt['in_market_pct']:.1f}% of days. The frozen threshold is {modes['daily']['threshold']:.3f} percent, "
        "slightly below zero, so the model has to be unusually negative before the strategy goes to cash."
    )
    d0 = modes["daily"]["strategies"]["Binary (t=0)"]
    pdf.bullet(
        f"Daily Binary (t=0) needs no fitted threshold. Sharpe {d0['sharpe']:.3f} is about even with "
        f"buy-and-hold ({d0['bnh_sharpe']:.3f}). CAGR is lower ({d0['cagr']:.2f}% vs {d0['bnh_cagr']:.2f}%), "
        f"and max drawdown is much shallower ({d0['max_drawdown']:.1f}% vs {d0['bnh_max_drawdown']:.1f}%). "
        "This is the more conservative daily result."
    )
    pdf.bullet(
        "Weekly Binary (t=0) and weekly Adaptive stay close to weekly buy-and-hold and do not beat it. "
        f"Weekly directional accuracy is the best of the three models at {modes['weekly']['model']['dir_accuracy']:.1f}%, "
        "but a once-a-week decision cannot leave during a crash that happens between Fridays."
    )
    pdf.bullet(
        "Scaled is the weak variant at every frequency. On daily bars it changes size almost continuously "
        f"(turnover {modes['daily']['strategies']['Scaled']['turnover_per_year']:.0f} per year) and gives up most of the market's CAGR."
    )
    pdf.bullet(
        "A threshold fit on 2010-2012 can fail when it is frozen. Weekly Binary (t=opt) chose "
        f"{modes['weekly']['threshold']:.3f} and was long only {modes['weekly']['strategies']['Binary (t=opt)']['in_market_pct']:.1f}% "
        "of the evaluation weeks. Mixed Binary (t=opt) chose "
        f"{modes['mixed']['threshold']:.3f} and was long "
        f"{modes['mixed']['strategies']['Binary (t=opt)']['in_market_pct']:.1f}% of days. Both sit in cash and earn almost nothing. "
        "That is an out-of-sample result, not a bug: the calibration window liked a high bar that later predictions rarely cleared."
    )

    pdf.add_page()
    pdf.h1("4. Equity paths")
    pdf.img(STUDY / "equity_by_mode.png", w=180)
    pdf.body(
        "Each panel starts at $1 on the first evaluation date. Buy-and-hold is the dashed line. "
        "The weekly panel never pulls away from the benchmark. The daily panel is where Binary (t=opt) "
        "compounds ahead and where Binary (t=0) gives up some upside to stay smoother. The mixed panel "
        "falls back toward, and at times through, the benchmark."
    )

    pdf.h1("5. Where the daily edge comes from")
    pdf.body(
        "Daily directional accuracy is only "
        f"{modes['daily']['model']['dir_accuracy']:.1f}% and the correlation between predicted and "
        f"realized next-day return is {modes['daily']['model']['corr']}. The model is not calling most days correctly. "
        "Binary (t=opt) is flat on 45 of 3,473 evaluation days. Those days include the August 2015 break, "
        "a few sessions in 2016 and 2018, the March 2020 crash, and a stretch in April 2025."
    )
    pdf.body(
        "The weekly model stayed long through February and March 2020, including weeks of about -11%, "
        "-9%, and -15%. It could not react until the next Friday, and by then its prediction was still positive. "
        "The daily model went flat on 5, 10, 11 and 13 March 2020 and skipped those down closes. It then stayed "
        "flat into the rebound and missed a large part of the bounce. Net of that giveback, the crash exit was still helpful."
    )
    pdf.h2("Sensitivity")
    pdf.body(
        "The rows below replay the daily positions. 'Long through COVID' forces the strategy to stay "
        "invested from 15 Feb 2020 through 30 Apr 2020. '2020 removed' drops that calendar year from the score entirely."
    )
    sens_rows = []
    for label, vals in sens.items():
        sens_rows.append([
            label, f"{vals[0]:.3f}", f"{vals[1]:.2f}", f"{vals[2]:.1f}",
            f"{vals[3]:.3f}", f"{vals[4]:.2f}", f"{vals[5]:.1f}",
        ])
    pdf.table(
        ["Daily replay", "Sharpe", "CAGR%", "MaxDD%", "BnH Sh", "BnH CAGR", "BnH DD"],
        sens_rows,
        [48, 18, 18, 18, 18, 22, 18],
        size=7.5,
    )
    pdf.bullet("Forcing the strategy to hold through the COVID window removes most of the drawdown improvement. Sharpe stays a little above buy-and-hold (0.934 vs 0.901) and CAGR stays a little above (15.06% vs 14.61%).")
    pdf.bullet("With 2020 removed, Binary (t=opt) still leads buy-and-hold on Sharpe (1.047 vs 1.003) and CAGR (13.86% vs 13.41%). The max-drawdown gap closes. Outside that year, this variant is a small edge, not a different risk profile.")
    pdf.bullet("Binary (t=0) keeps a shallower drawdown even with 2020 removed (-19.5% vs -24.5%), but it gives up CAGR and no longer matches buy-and-hold Sharpe. It is a smoother path, not a higher one.")

    pdf.h1("6. Did weekly features help the daily model?")
    pdf.body(
        f"The mixed model had {modes['mixed']['n_candidates']} candidate columns against "
        f"{modes['daily']['n_candidates']} for daily-only. Walk-forward selection used weekly columns "
        f"for {modes['mixed']['weekly_feature_share']*100:.1f}% of its picks. "
        "W_HMM_Regime_Duration is among the features chosen most often, so the weekly state was not ignored."
    )
    pdf.body(
        "Performance still fell. Mixed Binary (t=0) Sharpe is "
        f"{modes['mixed']['strategies']['Binary (t=0)']['sharpe']:.3f} versus "
        f"{modes['daily']['strategies']['Binary (t=0)']['sharpe']:.3f} for daily-only, and max drawdown "
        f"returns to {modes['mixed']['strategies']['Binary (t=0)']['max_drawdown']:.1f}%, in line with buy-and-hold. "
        "The extra columns made the next-day forecast noisier. Correlation fell from "
        f"{modes['daily']['model']['corr']} to {modes['mixed']['model']['corr']}."
    )
    pdf.body(
        "A completed week is a slow variable next to a one-day target. On this test, the daily indicators "
        "already carry the useful state (VIX level, recent return, distance from the 200-day average). "
        "Repeating that state at weekly resolution did not add a decision the daily model could trade."
    )

    pdf.add_page()
    pdf.h1("7. Transaction costs")
    pdf.body(
        "Headline results use 2 bps round-trip on the change in position. SPY is liquid enough that "
        "1 bp is realistic for a small account and 5 bps is a harsh assumption. Scaled feels the cost. "
        "The binary rules barely do, because they are usually fully in or fully out."
    )
    cost_rows = []
    for mode in order:
        for name in ["Binary (t=0)", "Binary (t=opt)", "Scaled"]:
            sens_c = modes[mode]["cost_sensitivity"][name]
            cost_rows.append([
                mode, name.replace("Binary ", "Bin "),
                f"{sens_c['1.0']['sharpe']:.3f}",
                f"{sens_c['2.0']['sharpe']:.3f}",
                f"{sens_c['5.0']['sharpe']:.3f}",
                f"{sens_c['5.0']['cagr']:.2f}",
            ])
    pdf.table(
        ["Mode", "Strategy", "Sh @ 1bp", "Sh @ 2bp", "Sh @ 5bp", "CAGR @ 5bp"],
        cost_rows,
        [24, 32, 24, 24, 24, 28],
    )
    pdf.body(
        "Daily Binary (t=opt) is 1.043 at 1 bp and 1.037 at 5 bps. The conclusion does not depend on "
        "the cost assumption inside this range. Daily Scaled stays well behind at every cost."
    )

    pdf.h1("8. What to use for SPY")
    pdf.bullet("Trade the daily bar, not the weekly bar, if the goal is a better SPY path than buy-and-hold.")
    pdf.bullet("Use Binary (t=opt) with the threshold estimated on a past window and then left alone. In this study that threshold was -0.07 percent, fit on 2010-2012 and applied from 2013. Re-estimate it on a rolling past window only if you also keep a rule that rejects a threshold which would have left the strategy in cash most of the time.")
    pdf.bullet("If you do not want a fitted threshold, use daily Binary (t=0). Expect a smoother drawdown and a lower CAGR than buy-and-hold, not a higher one.")
    pdf.bullet("Do not stack the weekly feature set onto the daily model. It was tested and it reduced performance.")
    pdf.bullet("Do not run the Scaled variant on daily bars. The position changes too often for the signal quality.")
    pdf.bullet("Do not expect the weekly Adaptive rule to be the one that benefits from daily data. Once decisions are daily, the simple binary rules were stronger.")
    pdf.h2("Data produced")
    pdf.bullet("daily_study/spy_daily_features.csv: daily feature matrix and next-day target.")
    pdf.bullet("daily_study/spy_weekly_features.csv: weekly feature matrix and next-week target.")
    pdf.bullet("daily_study/spy_mixed_features.csv: daily matrix plus released weekly columns.")
    pdf.bullet("daily_study/*_predictions.csv: walk-forward out-of-sample predictions.")
    pdf.bullet("daily_study/comparison_summary.json: the metrics behind this report.")

    pdf.h1("9. Limits")
    pdf.bullet("One asset. SPY is the most liquid equity index fund. The same daily edge is not assumed for a single stock, a currency, or a TASI name.")
    pdf.bullet("The daily advantage is concentrated in a small number of exits. March 2020 is the largest. A study that starts after 2020 would show a thinner gap.")
    pdf.bullet("The frozen-threshold rule can select a bar that later predictions never reach. Weekly and mixed Binary (t=opt) both did that. Any live use needs a guardrail on how often the rule is allowed to sit in cash.")
    pdf.bullet("Costs are a flat bps charge. They do not include borrow, taxes, or a missed open in a halted session.")
    pdf.bullet("The model refresh is every 26 weeks or 126 trading days. A faster refresh could change the mixed-model result. It was not tested here.")

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
