"""
Hidden Markov Model regime detection for SPY weekly data.

Fits a 3-state Gaussian HMM on observable features (returns + VIX)
using walk-forward estimation to prevent look-ahead bias.

States are post-hoc sorted by mean return:
  0 = Bear,  1 = Sideways,  2 = Bull

Outputs per-week:
  - HMM_Regime           (int 0/1/2)
  - HMM_Regime_Bear      (dummy)
  - HMM_Regime_Sideways  (dummy)
  - HMM_Regime_Bull      (dummy)
  - HMM_Prob_Bear        (posterior probability)
  - HMM_Prob_Sideways
  - HMM_Prob_Bull
  - HMM_Regime_Duration  (weeks in current state)
  - HMM_Regime_Change    (1 if state changed from prior week)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import warnings

N_STATES = 3
MIN_FIT_WEEKS = 104  # 2 years minimum before first prediction
REFIT_EVERY = 26     # refit model every 26 weeks for efficiency


def _sort_states_by_return(model, obs):
    """
    Map raw HMM state labels to bear/sideways/bull based on
    the learned mean of the return dimension (column 0).
    Returns a mapping dict: raw_label -> sorted_label.
    """
    means = model.means_[:, 0]
    order = np.argsort(means)  # ascending: lowest mean = bear
    mapping = {raw: rank for rank, raw in enumerate(order)}
    return mapping


def fit_hmm_walk_forward(df, return_col="Close_PctChg", vix_col="VIX_Close",
                         n_states=N_STATES, min_weeks=MIN_FIT_WEEKS,
                         refit_every=REFIT_EVERY):
    """
    Walk-forward HMM regime detection.

    For week t, the model is fitted on weeks [0, t) only.
    To keep runtime practical, the model is re-estimated every
    `refit_every` weeks and the last fitted model is used in between.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex, must contain return_col and vix_col.
    return_col : column name for weekly returns (percentage).
    vix_col : column name for VIX level.

    Returns
    -------
    DataFrame with HMM features, same index as df.
    """
    if return_col not in df.columns or vix_col not in df.columns:
        raise ValueError(f"Need columns '{return_col}' and '{vix_col}' in df")

    n = len(df)
    returns = df[return_col].values.astype(float)
    vix = df[vix_col].values.astype(float)

    vix_filled = pd.Series(vix).ffill().bfill().values
    ret_filled = np.nan_to_num(returns, nan=0.0)

    obs_full = np.column_stack([ret_filled, vix_filled])

    regimes = np.full(n, np.nan)
    probs = np.full((n, n_states), np.nan)

    current_model = None
    current_mapping = None
    last_fit_idx = -refit_every  # force fit on first eligible week

    for t in range(n):
        if t < min_weeks:
            continue

        need_refit = (t - last_fit_idx) >= refit_every or current_model is None
        if need_refit:
            obs_train = obs_full[:t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",
                        n_iter=100,
                        random_state=42,
                        tol=0.01,
                    )
                    model.fit(obs_train)
                    current_model = model
                    current_mapping = _sort_states_by_return(model, obs_train)
                    last_fit_idx = t
                except Exception:
                    pass

        if current_model is None:
            continue

        obs_up_to_t = obs_full[:t + 1]
        try:
            raw_states = current_model.predict(obs_up_to_t)
            raw_probs = current_model.predict_proba(obs_up_to_t)

            raw_state_t = raw_states[-1]
            mapped_state = current_mapping[raw_state_t]
            regimes[t] = mapped_state

            mapped_probs = np.zeros(n_states)
            for raw_s, rank in current_mapping.items():
                mapped_probs[rank] = raw_probs[-1, raw_s]
            probs[t] = mapped_probs

        except Exception:
            pass

    result = pd.DataFrame(index=df.index)
    result["HMM_Regime"] = regimes

    for i, label in enumerate(["Bear", "Sideways", "Bull"]):
        result[f"HMM_Regime_{label}"] = (regimes == i).astype(float)
        result[f"HMM_Prob_{label}"] = probs[:, i]

    duration = np.zeros(n)
    for t in range(1, n):
        if np.isnan(regimes[t]):
            continue
        if regimes[t] == regimes[t - 1]:
            duration[t] = duration[t - 1] + 1
        else:
            duration[t] = 1
    result["HMM_Regime_Duration"] = duration

    change = np.zeros(n)
    for t in range(1, n):
        if np.isnan(regimes[t]) or np.isnan(regimes[t - 1]):
            continue
        if regimes[t] != regimes[t - 1]:
            change[t] = 1
    result["HMM_Regime_Change"] = change

    nan_mask = np.isnan(regimes)
    for col in result.columns:
        result.loc[nan_mask, col] = np.nan

    print(f"  HMM regime detection: {(~nan_mask).sum()} weeks classified "
          f"(skipped first {min_weeks} weeks)")

    valid = result.dropna(subset=["HMM_Regime"])
    if len(valid) > 0:
        counts = valid["HMM_Regime"].value_counts().sort_index()
        labels = {0: "Bear", 1: "Sideways", 2: "Bull"}
        for state, count in counts.items():
            pct = count / len(valid) * 100
            print(f"    State {int(state)} ({labels.get(int(state), '?')}): "
                  f"{int(count)} weeks ({pct:.1f}%)")

    return result


def add_hmm_features(df, return_col="Close_PctChg", vix_col="VIX_Close"):
    """
    Convenience wrapper: detect regimes and join to df.
    Returns the augmented DataFrame.
    """
    hmm_feats = fit_hmm_walk_forward(df, return_col=return_col, vix_col=vix_col)
    return pd.concat([df, hmm_feats], axis=1)


if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = Path(__file__).parent / "data"
    csv = DATA_DIR / "spy_weekly_enriched.csv"
    df = pd.read_csv(csv, index_col="Week_Ending", parse_dates=True)
    print(f"Loaded {len(df)} rows")
    df = add_hmm_features(df)
    hmm_cols = [c for c in df.columns if c.startswith("HMM_")]
    print(f"\nHMM columns: {hmm_cols}")
    print(df[hmm_cols].dropna().describe())
