"""
Frequency analysis and outlier detection on SPY weekly data.

Produces per-column distribution statistics, normality tests,
and outlier counts (IQR and Z-score methods) to inform
standardization decisions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_ohlcv.csv"
REPORT_CSV = DATA_DIR / "distribution_report.csv"
OUTLIERS_CSV = DATA_DIR / "outliers_detail.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)
    return df


def frequency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution metrics for every column."""
    records = []
    for col in df.columns:
        s = df[col].dropna()
        rec = {
            "Column": col,
            "Count": len(s),
            "Mean": s.mean(),
            "Median": s.median(),
            "Std": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Range": s.max() - s.min(),
            "Q1": s.quantile(0.25),
            "Q3": s.quantile(0.75),
            "IQR": s.quantile(0.75) - s.quantile(0.25),
            "Skewness": s.skew(),
            "Kurtosis": s.kurtosis(),
            "CV": s.std() / s.mean() if s.mean() != 0 else np.nan,
        }

        if len(s) >= 20:
            jb_stat, jb_p = stats.jarque_bera(s)
            rec["JarqueBera_Stat"] = jb_stat
            rec["JarqueBera_p"] = jb_p
            rec["Normal_JB"] = "Yes" if jb_p > 0.05 else "No"

            if len(s) <= 5000:
                sw_stat, sw_p = stats.shapiro(s)
                rec["Shapiro_Stat"] = sw_stat
                rec["Shapiro_p"] = sw_p
                rec["Normal_SW"] = "Yes" if sw_p > 0.05 else "No"

        records.append(rec)

    return pd.DataFrame(records).set_index("Column")


def detect_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers using two methods:
      1. IQR: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
      2. Z-score: |z| > 3

    Returns a summary table and a detail table of flagged rows.
    """
    summary_records = []
    detail_frames = []

    for col in df.columns:
        s = df[col].dropna()

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_mask = (s < lower_iqr) | (s > upper_iqr)

        z = pd.Series(np.abs(stats.zscore(s)), index=s.index)
        z_mask = z > 3

        summary_records.append({
            "Column": col,
            "IQR_Lower_Fence": lower_iqr,
            "IQR_Upper_Fence": upper_iqr,
            "IQR_Outliers": iqr_mask.sum(),
            "IQR_Outlier_Pct": 100 * iqr_mask.sum() / len(s),
            "ZScore_Outliers": z_mask.sum(),
            "ZScore_Outlier_Pct": 100 * z_mask.sum() / len(s),
        })

        combined_mask = iqr_mask | z_mask
        if combined_mask.any():
            flagged = df.loc[combined_mask[combined_mask].index, [col]].copy()
            flagged["Column"] = col
            flagged["Value"] = flagged[col]
            z_series = pd.Series(stats.zscore(s), index=s.index)
            flagged["Z_Score"] = z_series.loc[flagged.index]
            flagged["IQR_Outlier"] = iqr_mask.loc[flagged.index]
            flagged["ZScore_Outlier"] = z_mask.loc[flagged.index]
            flagged = flagged.drop(columns=[col])
            detail_frames.append(flagged)

    summary = pd.DataFrame(summary_records).set_index("Column")
    details = pd.concat(detail_frames) if detail_frames else pd.DataFrame()
    return summary, details


def print_section(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    # --- Frequency / distribution analysis ---
    print_section("FREQUENCY & DISTRIBUTION ANALYSIS")
    dist = frequency_analysis(df)
    print(dist.T.to_string())
    dist.to_csv(REPORT_CSV, float_format="%.6f")
    print(f"\nSaved to {REPORT_CSV}")

    # --- Outlier detection ---
    print_section("OUTLIER DETECTION")
    outlier_summary, outlier_details = detect_outliers(df)
    print(outlier_summary.to_string())

    if not outlier_details.empty:
        outlier_details.to_csv(OUTLIERS_CSV)
        print(f"\n{len(outlier_details)} outlier rows saved to {OUTLIERS_CSV}")
    else:
        print("\nNo outliers detected.")

    # --- Standardization recommendations ---
    print_section("STANDARDIZATION NOTES")
    for col in df.columns:
        row = dist.loc[col]
        skew = row["Skewness"]
        kurt = row["Kurtosis"]
        cv = row["CV"]
        normal_jb = row.get("Normal_JB", "N/A")
        iqr_out = outlier_summary.loc[col, "IQR_Outliers"]

        notes = []
        if abs(skew) > 1:
            notes.append(f"high skew ({skew:.2f})")
        if kurt > 3:
            notes.append(f"heavy tails (kurtosis={kurt:.2f})")
        if normal_jb == "No":
            notes.append("non-normal (Jarque-Bera)")
        if iqr_out > 0:
            notes.append(f"{int(iqr_out)} IQR outliers")

        if abs(skew) > 1 or kurt > 3:
            rec = "RobustScaler or log-transform"
        elif normal_jb == "No":
            rec = "RobustScaler or MinMaxScaler"
        else:
            rec = "StandardScaler (Z-score)"

        flag_str = "; ".join(notes) if notes else "well-behaved"
        print(f"  {col:12s} -> {rec:40s} [{flag_str}]")


if __name__ == "__main__":
    main()
