"""06_smoke_test.py -- NOT thesis results. Confirms the panel trains/evaluates end-to-end at h=1:
naive, SARIMAX(1,1,1)+exog (params fit on train, 1-step filter on test), XGBoost on weekly change, 50/50 blend; pooled by regime."""
import sys, warnings, numpy as np, pandas as pd, xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from importlib import import_module
ing = import_module("00_ingest"); OUT = ing.OUT; warnings.filterwarnings("ignore")
series = sys.argv[1] if len(sys.argv) > 1 else "indicator"
P = pd.read_csv(OUT / f"features_panel_{series}.csv", parse_dates=["week_end"])
feats = [c for c in P.columns if c not in ("week_end", "commodity", "y", "split", "I_shock_t")]
rows = []
for c, d in P.groupby("commodity"):
    d = d.reset_index(drop=True); tr, te = d[d.split == "train"], d[d.split == "test"]
    ex = ["rain_sum_L1", "diesel_pct_L1"]
    m = SARIMAX(tr.y.values, exog=tr[ex].values, order=(1, 1, 1)).fit(disp=False)
    full = m.apply(d.y.values, exog=d[ex].values); pr = full.get_prediction(start=len(tr), dynamic=False).predicted_mean
    X = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    X.fit(tr[feats], tr.y - tr.y_L1); px = te.y_L1.values + X.predict(te[feats])
    o = te[["week_end", "y", "I_shock_t"]].copy(); o["commodity"] = c
    o["naive"] = te.y_L1.values; o["sarimax"] = pr; o["xgb"] = px; o["blend"] = 0.5 * pr + 0.5 * px; rows.append(o)
R = pd.concat(rows); R.to_csv(OUT / f"smoke_predictions_{series}.csv", index=False)
rm = lambda g, k: np.sqrt(((g[k] - g.y) ** 2).mean())
out = []
for lab, g in [("all", R), ("shock", R[R.I_shock_t == 1]), ("normal", R[R.I_shock_t == 0])]:
    out.append({"series": series, "regime": lab, "n_obs": len(g), **{k: round(rm(g, k), 3) for k in ["naive", "sarimax", "xgb", "blend"]}})
print(pd.DataFrame(out).to_string(index=False))
