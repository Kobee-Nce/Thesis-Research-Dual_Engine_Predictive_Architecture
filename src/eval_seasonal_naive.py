"""eval_seasonal_naive.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
52-week seasonal naive: yhat(t) = y(t-52), where the lag is taken from the FULL weekly price history (not only the 90 test rows).
Reports pooled-PHP RMSE ratio and the paper's primary measure (geometric mean over commodities of RMSE/RMSE_naive).
Usage (project root):  python src/eval_seasonal_naive.py"""
import numpy as np, pandas as pd
from pathlib import Path
P = Path("data/processed")
F = pd.read_csv(P / "wf_forecasts_indicator.csv", parse_dates=["week_end"])
W = pd.read_csv(P / "weekly_prices_disaggregated.csv", parse_dates=["week_end"]).melt(id_vars="week_end", var_name="commodity", value_name="price")
W = W.sort_values(["commodity", "week_end"]); W["seasonal_naive"] = W.groupby("commodity")["price"].shift(52)      # lag over the whole history
df = F.merge(W[["commodity", "week_end", "price", "seasonal_naive"]], on=["commodity", "week_end"], how="left")
assert np.allclose(df.price, df.y) and df.seasonal_naive.notna().all(), "history does not cover all test weeks"
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); models = ["naive", "seasonal_naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]; rows = []
for m in models:
    per = {c: rm(g[m] - g.y) / rm(g.naive - g.y) for c, g in df.groupby("commodity")}
    rows.append(dict(Model=m, n=len(df), RMSE_pooled=rm(df[m] - df.y), MAE_pooled=float(np.mean(np.abs(df[m] - df.y))), MAPE_pooled=100 * float(np.mean(np.abs(df[m] - df.y) / df.y)),
                     rel_RMSE_pooled=rm(df[m] - df.y) / rm(df.naive - df.y), rel_RMSE_geomean=float(np.exp(np.mean(np.log(list(per.values()))))), **{f"rel_{c}": v for c, v in per.items()}))
R = pd.DataFrame(rows); R.round(4).to_csv(P / "seasonal_naive_results.csv", index=False); print(R[["Model", "n", "RMSE_pooled", "MAE_pooled", "MAPE_pooled", "rel_RMSE_pooled", "rel_RMSE_geomean"]].round(3).to_string(index=False))
