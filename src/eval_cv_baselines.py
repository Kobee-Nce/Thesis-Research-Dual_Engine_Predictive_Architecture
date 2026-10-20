"""eval_cv_baselines.py -- Section 4.4: mean and standard deviation across the four rolling-origin validation folds
for the gated ensemble and baselines B1 (naive), B3 (SARIMAX), B4 (XGBoost) and B5 (fixed 50/50).
Reads the saved out-of-fold forecasts (oof_predictions_<series>.csv, training weeks only) and the frozen weights; fits nothing.
B2 (ARIMA + Fourier) and the ablations were not run on the folds and are therefore reported on the test set only.
Score per fold: geometric mean over the ten commodities of RMSE / naive RMSE (as Eq. 3.4), and the pooled RMSE ratio.
Output: table_4_4_cv_baselines.csv"""
import json, numpy as np, pandas as pd
from importlib import import_module
OUT = import_module("00_ingest").OUT
rows = []
for s in ("indicator", "noind"):
    O = pd.read_csv(OUT / f"oof_predictions_{s}.csv"); W = pd.read_csv(OUT / f"table_3_8_alpha_weights_{s}.csv").set_index("regime").alpha1_sarimax
    a = np.where(O.shock_L1 == 1, W["shock"], W["normal"])
    O["gated"] = a * O.sarimax + (1 - a) * O.xgb; O["blend50"] = 0.5 * O.sarimax + 0.5 * O.xgb
    for f, d in O.groupby("fold"):
        for m in ("naive", "sarimax", "xgb", "blend50", "gated"):
            r = d.groupby("commodity").apply(lambda g: np.sqrt(((g[m] - g.y) ** 2).mean()) / np.sqrt(((g.naive - g.y) ** 2).mean()))
            pooled = np.sqrt(((d[m] - d.y) ** 2).mean()) / np.sqrt(((d.naive - d.y) ** 2).mean())
            rows.append(dict(series=s, fold=f, model=m, rel_rmse_gmean=float(np.exp(np.log(r).mean())), rel_rmse_pooled=float(pooled)))
F = pd.DataFrame(rows)
S = F.groupby(["series", "model"]).agg(mean_gmean=("rel_rmse_gmean", "mean"), sd_gmean=("rel_rmse_gmean", "std"),
                                       mean_pooled=("rel_rmse_pooled", "mean"), sd_pooled=("rel_rmse_pooled", "std")).reset_index()
P = F.pivot_table(index=["series", "model"], columns="fold", values="rel_rmse_gmean").add_prefix("fold").reset_index()
T = S.merge(P, on=["series", "model"]); T.round(4).to_csv(OUT / "table_4_4_cv_baselines.csv", index=False); print(T.round(4).to_string(index=False))
