"""cv_utils.py -- shared helpers for the tuning scripts (07-10).
GATE G3 GUARD: every loader here returns TRAIN rows only, so no tuning step can see the test set."""
import numpy as np, pandas as pd
from importlib import import_module
ing = import_module("00_ingest"); OUT = ing.OUT

SERIES = ("indicator", "noind")
LAG_BASES = ["y", "rain_sum", "rain_max", "wind_max", "t_mean", "t_max", "t_min", "rain_roll4", "diesel_pct"]   # 9 base series
LAG_COLS = [f"{b}_L{k}" for b in LAG_BASES for k in range(1, 9)]                                              # 72 searchable bits
FIXED_COLS = ["y_roll4_mean", "y_roll4_std", "cpi_veg_mom_L1m", "sin1", "cos1", "sin2", "cos2", "shock_L1"]    # 8 always-on terms
SARIMAX_EXOG = ["sin1", "cos1", "sin2", "cos2", "rain_sum_L1", "diesel_pct_L1", "cpi_veg_mom_L1m"]            # small fixed exogenous set
CV_K, CV_VAL = 4, 36                                                                                           # 4 rolling-origin folds x 36 weeks

def load_train(series):
    P = pd.read_csv(OUT / f"features_panel_{series}.csv", parse_dates=["week_end"])
    P = P[P.split == "train"].sort_values(["commodity", "week_end"]).reset_index(drop=True)
    assert (P.split == "train").all() and len(P) == 3460, "unexpected row count: train rows only"
    return P

def rolling_folds(n, k=CV_K, val=CV_VAL):
    """Expanding-window rolling origin: fold i trains on rows [0:vs) and validates on [vs:ve). Last fold ends at the end of train."""
    s = n - k * val
    return [(s + i * val, s + (i + 1) * val) for i in range(k)]

def pooled_arrays(P, cols):
    """(C, n, F) feature cube and (C, n) delta-y standardized per commodity (scale from the first fold's training window only)."""
    coms = list(P.commodity.unique()); n = (P.commodity == coms[0]).sum()
    vs0 = rolling_folds(n)[0][0]; X, Y, S = [], [], []
    for c in coms:
        d = P[P.commodity == c]; dy = (d.y - d.y_L1).values; s = dy[:vs0].std()
        X.append(d[cols].values); Y.append(dy / s); S.append(s)
    return coms, np.stack(X), np.stack(Y), np.array(S)
