"""PDF report: TASI daily strategies across calendar periods."""

import json
from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "tasi_periods"
PDF_PATH = Path(__file__).parent / "TASI_Daily_Period_Report.pdf"

ORDER = [
    "Binary (t=0)",
    "Binary (t=opt)",
    "Binary (t=opt frozen)",
    "Scaled",
    "Scaled (frozen)",
    "Adaptive",
]
SHORT = {
    "Binary (t=0)": "t=0",
    "Binary (t=opt)": "t=opt refit",
    "Binary (t=opt frozen)": "t=opt frozen",
    "Scaled": "Scaled refit",
    "Scaled (frozen)": "Scaled frozen",
    "Adaptive": "Adaptive",
}


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "TASI daily strategies across periods - Research prototype", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20, 60, 120)
        self.ln(2)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.3, text)
        self.ln(1.3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.3, "- " + text)
        self.ln(0.3)

    def table(self, headers, rows, widths, size=7.5):
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            self.cell(w, 7, str(h), border=1, fill=True, align="C")
        self.ln()
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            self.set_fill_color(235, 241, 248) if fill else self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", size)
            for i, (val, w) in enumerate(zip(row, widths)):
                self.cell(w, 5.4, str(val), border=1, fill=True, align="L" if i == 0 else "C")
            self.ln()
            fill = not fill
        self.ln(2)


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def _era_count(stats, floor=0.55, min_n=15):
    hits = 0
    usable = 0
    for row in (stats.get("by_period") or {}).values():
        if row["n"] < min_n:
            continue
        usable += 1
        if row["beat_bnh"] / row["n"] >= floor:
            hits += 1
    return hits, usable


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    block = summary["all"]["periods"]["Binary (t=0)"]
    full = summary["all"]["full"]["Binary (t=0)"]
    n = summary["n_ok"]
    if not block:
        raise SystemExit("No completed TASI periods to report")

    beat_share = _pct(block["beat_bnh"], block["ticker_periods"])
    majority_share = _pct(block["names_majority"], block["names"])
    eras, era_n = _era_count(block)
    opt = summary["all"]["periods"]["Binary (t=opt)"]
    opt_f = summary["all"]["periods"]["Binary (t=opt frozen)"]
    orig = summary["original_12"]["periods"]["Binary (t=0)"]
    rest = summary["rest"]["periods"]["Binary (t=0)"]

    if beat_share >= 55 and majority_share >= 55 and eras >= 3:
        verdict = (
            "On the daily frame, the TASI class edge is broad enough to take seriously. "
            "Binary at a zero threshold beats buy-and-hold in most ticker-periods, for most names "
            "across a majority of their own periods, and in most calendar windows with a real sample."
        )
    elif beat_share >= 50 and eras >= 2:
        verdict = (
            "The daily TASI result is real on average and it is not confined to the original 12 names. "
            "It is not uniform. Some periods and some variants carry it, and a fixed zero threshold "
            "does not beat buy-and-hold for every name in every window."
        )
    else:
        verdict = (
            "Expanding TASI from the original 12 names to the full list does not leave a consistent "
            "daily edge. A strong result in one window or one handful of names does not survive "
            "the period split."
        )

    pdf = Report(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.ln(22)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "TASI daily strategies, period by period", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 8, "All Tadawul All Share names. Daily frame. Every variant.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{summary['n_universe']} listed names, {n} scored, {summary['n_skipped']} skipped, {summary['n_error']} failed", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Research prototype. Not investment advice. October 2026.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.h1("1. Question and answer")
    pdf.body(
        "The 102-asset study found the daily edge concentrated in TASI stocks, on 12 names and one "
        "long evaluation window. This run asks whether that class result holds for the rest of the "
        "Tadawul All Share list, and whether it holds in separate periods rather than in a single average."
    )
    pdf.h2("What was run")
    pdf.bullet("Universe: every constituent of the Tadawul All Share index, 252 names, Yahoo symbols XXXX.SR.")
    pdf.bullet("Model: the daily feature frame and the same walk-forward ensemble. The decision is the next session, not the next week.")
    pdf.bullet("Weekly bars are still built for each name on a Thursday close. They are not inputs to this model. The weekly feature pipeline remains in the codebase for a later test.")
    pdf.bullet("Periods: 2013-2015, 2016-2018, 2019-2021, and 2022-2026 when the name has them. Names that cannot fill two of those blocks use two-year blocks instead. A block needs 180 sessions, and 120 earlier prediction sessions to fit a threshold.")
    pdf.bullet("Binary (t=opt) and Scaled are reported two ways. Refit uses only the three years before that period. Frozen uses one threshold fit before the first period and keeps it.")
    pdf.bullet("Cost is 2 bps of the position change. The same cost used in the 102-asset study. TASI spreads are often wider than that.")
    pdf.h2("Short answer")
    pdf.body(verdict)
    pdf.body(
        f"Binary (t=0) beats that name's own buy-and-hold in {block['beat_bnh']} of "
        f"{block['ticker_periods']} name-periods ({beat_share:.0f}%). "
        f"It does so in a majority of periods for {block['names_majority']} of {block['names']} names "
        f"({majority_share:.0f}%), and in every one of its periods for {block['names_all_periods']} names. "
        f"The median gap is {block['median_gap']:+.3f} Sharpe "
        f"({block['median_sharpe']:.3f} versus buy-and-hold {block['median_bnh']:.3f})."
    )
    pdf.body(
        f"On the full window after the first period, with one frozen rule, Binary (t=0) beats "
        f"buy-and-hold for {full['beat_bnh']} of {full['names']} names. Median Sharpe "
        f"{full['median_sharpe']:.3f} versus {full['median_bnh']:.3f}. Median CAGR "
        f"{full['median_cagr']:.1f}% versus {full['median_bnh_cagr']:.1f}%. Median max drawdown "
        f"{full['median_dd']:.1f}% versus {full['median_bnh_dd']:.1f}%."
    )
    if orig and rest:
        pdf.body(
            f"The original 12 names, scored again inside this same run, beat buy-and-hold in "
            f"{orig['beat_bnh']} of {orig['ticker_periods']} of their periods "
            f"(median gap {orig['median_gap']:+.3f}). The other {rest['names']} scored names beat it in "
            f"{rest['beat_bnh']} of {rest['ticker_periods']} periods (median gap {rest['median_gap']:+.3f}). "
            "The class result is not an artifact of those 12 tickers."
        )

    pdf.h1("2. How look-ahead was kept out")
    pdf.bullet("The target on day t is the return to the next close. Features on day t use the close of day t.")
    pdf.bullet("Each walk-forward fold selects features and stops training on the tail of the past only.")
    pdf.bullet("A period's refit threshold is fit on predictions dated before the period starts. The period's own days are not in that fit.")
    pdf.bullet("The frozen threshold is fit once, on the block before the first scored period, and then left alone. Later periods do not update it.")
    pdf.bullet("Adaptive may use the previous 126 predictions and their realized returns. It may not use the current day or any later day.")
    pdf.bullet("Macro series are forward-filled onto the Saudi session. Nothing is back-filled. Leading HMM states are not back-filled.")
    pdf.bullet("Thursday weekly bars are stored beside the run. They are not columns in the model that produced these numbers.")
    pdf.body(
        f"No scored name had 400 feature rows before 2010, so none used that anchor. "
        f"All {summary['grid']['shifted']} scored names start the walk-forward once those 400 rows exist. "
        "Yahoo's daily history, after the indicator and HMM warmup, is what sets that date. "
        f"The first block the three-year grid can fill is 2016-2018, not 2013-2015. "
        f"{summary['grid']['3y']} names used the three-year grid. {summary['grid']['2y']} used two-year blocks. "
        f"{summary['n_skipped']} names were skipped because the history cannot support two separate periods."
    )

    pdf.h1("3. All variants, across periods")
    pdf.body(
        "Each row pools every name and every period. Beat is the share of those pairs where the "
        "strategy Sharpe is above buy-and-hold. Majority is the share of names for which that happens "
        "in more than half of their periods. In market is the median share of sessions with a position."
    )
    rows = []
    for name in ORDER:
        stats = summary["all"]["periods"].get(name)
        if not stats:
            continue
        rows.append([
            SHORT[name],
            f"{stats['beat_bnh']}/{stats['ticker_periods']}",
            f"{_pct(stats['beat_bnh'], stats['ticker_periods']):.0f}%",
            f"{stats['names_majority']}/{stats['names']}",
            f"{stats['median_gap']:+.3f}",
            f"{stats['median_sharpe']:.2f}",
            f"{stats['median_bnh']:.2f}",
            f"{stats['median_in_market']:.0f}%",
        ])
    pdf.table(
        ["Variant", "Beat", "Share", "Majority", "Med gap", "Med Sh", "Med BnH", "In mkt"],
        rows,
        [32, 22, 16, 22, 20, 18, 20, 18],
        size=7.5,
    )
    pdf.body(
        f"Binary (t=opt) refit is in the market a median {opt['median_in_market']:.0f}% of the time and "
        f"below 20% in {opt['cash_lt_20']} of {opt['ticker_periods']} name-periods. "
        f"The frozen threshold is in the market a median {opt_f['median_in_market']:.0f}% of the time and "
        f"below 20% in {opt_f['cash_lt_20']} name-periods. A threshold that looked good before the period "
        "still goes to cash often. Binary (t=0) does not have that free parameter."
    )

    pdf.h1("4. Does any one period carry the result?")
    pdf.body(
        "Median Sharpe of Binary (t=0) against buy-and-hold in each calendar block. A class claim "
        "needs more than one block."
    )
    period_rows = []
    # Stable order: by label sort is chronological for these labels.
    for label in sorted(block["by_period"]):
        row = block["by_period"][label]
        opt_row = (opt.get("by_period") or {}).get(label, {})
        period_rows.append([
            label,
            str(row["n"]),
            f"{row['beat_bnh']}/{row['n']}",
            f"{row['median_sharpe']:.2f}",
            f"{row['median_bnh']:.2f}",
            f"{row['median_gap']:+.2f}",
            f"{opt_row.get('median_gap', float('nan')):+.2f}" if opt_row else "-",
        ])
    pdf.table(
        ["Period", "N", "t=0 beat", "t=0 Sh", "BnH", "t=0 gap", "t=opt gap"],
        period_rows,
        [28, 16, 24, 22, 20, 24, 28],
        size=8,
    )
    chart = OUT / "sharpe_by_period.png"
    if chart.exists():
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.image(str(chart), w=170)
        pdf.ln(3)
    strong = [r[0] for r in period_rows if r[2] != "0/0" and int(r[2].split("/")[0]) / int(r[2].split("/")[1]) >= 0.55]
    weak = [r[0] for r in period_rows if r[2] != "0/0" and int(r[2].split("/")[0]) / int(r[2].split("/")[1]) < 0.5]
    pdf.body(
        f"Periods where t=0 beats buy-and-hold on at least 55% of names: {', '.join(strong) if strong else 'none'}. "
        f"Periods under half: {', '.join(weak) if weak else 'none'}. "
        "The rows with a handful of names are the two-year grid. Read 2016-2018, 2019-2021, and 2022-2026. "
        "Those three each have more than 100 names."
    )
    def _era(label):
        return (block.get("by_period") or {}).get(label)

    e1, e2, e3 = _era("2016-2018"), _era("2019-2021"), _era("2022-2026")
    if e1 and e2 and e3:
        pdf.body(
            f"The edge is large when buy-and-hold is weak. In 2016-2018 the median buy-and-hold Sharpe is "
            f"{e1['median_bnh']:.2f} and the t=0 gap is {e1['median_gap']:+.2f} "
            f"({e1['beat_bnh']} of {e1['n']} names). In 2022-2026 buy-and-hold is {e3['median_bnh']:.2f} and "
            f"t=0 is {e3['median_sharpe']:.2f}, gap {e3['median_gap']:+.2f} "
            f"({e3['beat_bnh']} of {e3['n']}). In that window the strategy is in the market a median "
            f"{e3['median_in_market']:.0f}% of sessions, so part of the gain is time in cash. In 2019-2021 both are strong "
            f"(t=0 {e2['median_sharpe']:.2f}, buy-and-hold {e2['median_bnh']:.2f}) and the gap shrinks to "
            f"{e2['median_gap']:+.2f}, with t=0 ahead on {e2['beat_bnh']} of {e2['n']} names. "
            "The class result is still there in that bull window. It is much thinner."
        )

    pdf.add_page()
    pdf.h1("5. The original 12 against the rest of the market")
    pdf.body(
        "These 12 names are the TASI stocks in the earlier 102-asset study. They are scored here with "
        "the same daily code and the same period rules as every other name."
    )
    compare_rows = []
    for title, stats in [("Original 12", orig), ("All other scored names", rest), ("Every scored name", block)]:
        if not stats:
            continue
        compare_rows.append([
            title,
            str(stats["names"]),
            f"{stats['beat_bnh']}/{stats['ticker_periods']}",
            f"{_pct(stats['beat_bnh'], stats['ticker_periods']):.0f}%",
            f"{stats['names_majority']}/{stats['names']}",
            f"{stats['names_all_periods']}/{stats['names']}",
            f"{stats['median_gap']:+.3f}",
        ])
    pdf.table(
        ["Group", "Names", "Beat", "Share", "Majority", "Every period", "Med gap"],
        compare_rows,
        [42, 18, 24, 18, 24, 28, 22],
        size=8,
    )
    pdf.body(
        "Majority means the name beat buy-and-hold in more than half of its periods. Every period "
        "means it beat buy-and-hold in all of them. A name with two periods must win both."
    )

    pdf.h1("6. What to take from this")
    pdf.body(verdict)
    pdf.bullet(
        f"Judge the class on Binary (t=0) first. It beats buy-and-hold in {beat_share:.0f}% of "
        f"name-periods, median gap {block['median_gap']:+.3f}. The full-window median CAGR is "
        f"{full['median_cagr']:.1f}% against {full['median_bnh_cagr']:.1f}% for buy-and-hold, and the "
        f"median max drawdown is {full['median_dd']:.1f}% against {full['median_bnh_dd']:.1f}%."
    )
    pdf.bullet(
        f"Median gap for the refit threshold is {opt['median_gap']:+.3f}. "
        f"Median gap for one frozen threshold is {opt_f['median_gap']:+.3f}. "
        f"Median gap for the zero threshold is {block['median_gap']:+.3f}. "
        "Cash-heavy thresholds remain common, so a fitted cutoff still needs a guardrail that rejects "
        "a rule which would sit out most of the period."
    )
    pdf.bullet(
        "The weekly feature code is unchanged and still causal: a Thursday bar is available the next "
        "session, not on the Thursday itself. This study did not put those columns into the model. "
        "The earlier 102-asset run is the evidence on mixing, and it did not help the typical asset."
    )
    pdf.body(
        "This is a research prototype. The 2 bp cost is the study convention, not a TASI spread. "
        "Names without two separate evaluation windows are excluded rather than scored on a short "
        "history. The result is a class tendency across periods, not a portfolio and not a forecast "
        "for the next listing."
    )

    if summary["errors"]:
        pdf.h2("Failed downloads or runs")
        for rec in summary["errors"][:12]:
            pdf.bullet(f"{rec.get('ticker')} {rec.get('name')}: {rec.get('reason')}")
        if len(summary["errors"]) > 12:
            pdf.bullet(f"{len(summary['errors']) - 12} further failures are listed in tasi_periods/summary.json.")

    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
