"""
Candlestick chart of SPY weekly data with Order Blocks highlighted.

Bullish OBs are shaded green, Bearish OBs are shaded red.
Each OB zone extends from OB_Low to OB_High.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_with_ob.csv"
OUTPUT_PNG = DATA_DIR / "spy_weekly_ob_chart.png"


def draw_candlesticks(ax, df):
    """Draw OHLC candlesticks on the given axes."""
    dates = mdates.date2num(df.index)
    width = 4.0

    for i in range(len(df)):
        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"

        ax.plot([dates[i], dates[i]], [l, h], color=color, linewidth=0.6, zorder=2)

        body_low = min(o, c)
        body_height = abs(c - o)
        rect = Rectangle(
            (dates[i] - width / 2, body_low), width, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3,
        )
        ax.add_patch(rect)


def draw_ob_zones(ax, df):
    """Overlay Order Block zones as shaded rectangles spanning forward."""
    bull_rows = df[df["OB_Bull"] == 1]
    bear_rows = df[df["OB_Bear"] == 1]

    for idx, row in bull_rows.iterrows():
        x = mdates.date2num(idx)
        zone_width = 120
        rect = Rectangle(
            (x - 10, row["OB_Low"]), zone_width, row["OB_High"] - row["OB_Low"],
            facecolor="#26a69a", edgecolor="#1b5e20", alpha=0.30, linewidth=1.2, zorder=1,
        )
        ax.add_patch(rect)
        ax.annotate(
            "Bull OB", (x, row["OB_High"]),
            fontsize=6, color="#1b5e20", fontweight="bold",
            xytext=(5, 4), textcoords="offset points", zorder=5,
        )

    for idx, row in bear_rows.iterrows():
        x = mdates.date2num(idx)
        zone_width = 120
        rect = Rectangle(
            (x - 10, row["OB_Low"]), zone_width, row["OB_High"] - row["OB_Low"],
            facecolor="#ef5350", edgecolor="#b71c1c", alpha=0.30, linewidth=1.2, zorder=1,
        )
        ax.add_patch(rect)
        ax.annotate(
            "Bear OB", (x, row["OB_Low"]),
            fontsize=6, color="#b71c1c", fontweight="bold",
            xytext=(5, -10), textcoords="offset points", zorder=5,
        )


def main():
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending")
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    print(f"Loaded {len(df)} weekly bars, plotting full history...")

    fig, ax = plt.subplots(figsize=(28, 10))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    draw_candlesticks(ax, df)
    draw_ob_zones(ax, df)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", colors="#cccccc", labelsize=8, rotation=45)
    ax.tick_params(axis="y", colors="#cccccc", labelsize=9)
    ax.yaxis.label.set_color("#cccccc")
    ax.set_ylabel("Price (Adjusted)", fontsize=11)
    ax.set_title("SPY Weekly Candlesticks with Order Blocks (periods=7)", fontsize=14, color="white", pad=15)

    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", color="#333333", linewidth=0.3, alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#26a69a", edgecolor="#1b5e20", alpha=0.5, label="Bullish OB (demand zone)"),
        Patch(facecolor="#ef5350", edgecolor="#b71c1c", alpha=0.5, label="Bearish OB (supply zone)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")

    ax.set_xlim(mdates.date2num(df.index[0]) - 20, mdates.date2num(df.index[-1]) + 20)
    ax.set_ylim(df["Low"].min() * 0.95, df["High"].max() * 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Full chart saved to {OUTPUT_PNG}")

    # --- Zoomed chart: last ~5 years ---
    cutoff = df.index[-1] - pd.DateOffset(years=5)
    df_zoom = df[df.index >= cutoff]

    fig2, ax2 = plt.subplots(figsize=(24, 10))
    fig2.patch.set_facecolor("#1e1e1e")
    ax2.set_facecolor("#1e1e1e")

    draw_candlesticks(ax2, df_zoom)
    draw_ob_zones(ax2, df_zoom)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.tick_params(axis="x", colors="#cccccc", labelsize=8, rotation=45)
    ax2.tick_params(axis="y", colors="#cccccc", labelsize=9)
    ax2.yaxis.label.set_color("#cccccc")
    ax2.set_ylabel("Price (Adjusted)", fontsize=11)
    ax2.set_title("SPY Weekly Candlesticks with Order Blocks — Last 5 Years", fontsize=14, color="white", pad=15)

    for spine in ax2.spines.values():
        spine.set_color("#444444")
    ax2.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.7)
    ax2.grid(axis="x", color="#333333", linewidth=0.3, alpha=0.4)

    ax2.legend(handles=legend_elements, loc="upper left", fontsize=9,
               facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")

    ax2.set_xlim(mdates.date2num(df_zoom.index[0]) - 10, mdates.date2num(df_zoom.index[-1]) + 10)
    ax2.set_ylim(df_zoom["Low"].min() * 0.97, df_zoom["High"].max() * 1.03)

    plt.tight_layout()
    zoom_path = DATA_DIR / "spy_weekly_ob_chart_zoom.png"
    fig2.savefig(zoom_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"Zoomed chart saved to {zoom_path}")


if __name__ == "__main__":
    main()
