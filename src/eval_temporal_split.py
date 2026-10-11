"""eval_temporal_split.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
Splits the 90 test weeks into 2024-25 (week_end up to 2025-12-28, 60 weeks) and 2026 (30 weeks).
Reports both the pooled-PHP RMSE ratio and the paper's primary measure (geometric mean over commodities of RMSE/RMSE_naive)."""
import numpy as np, pandas as pd
from pathlib import Path
P = Path("data/processed"); F = pd.read_csv(P / "wf_forecasts_indicator.csv", parse_dates=["week_end"])
F["period"] = np.where(F.week_end.dt.year <= 2025, "2024-2025", "2026"); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
models = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]; rows = []
for per, g in F.groupby("period"):
    for m in models:
        rel = [rm(x[m] - x.y) / rm(x.naive - x.y) for _, x in g.groupby("commodity")]
        rows.append(dict(Period=per, Model=m, weeks=g.week_end.nunique(), shock_gate_weeks=int(g.drop_duplicates("week_end").shock_L1.sum()), RMSE_pooled=rm(g[m] - g.y), MAE_pooled=float(np.mean(np.abs(g[m] - g.y))),
                         rel_RMSE_pooled=rm(g[m] - g.y) / rm(g.naive - g.y), rel_RMSE_geomean=float(np.exp(np.mean(np.log(rel))))))
R = pd.DataFrame(rows); R.round(4).to_csv(P / "temporal_split_results.csv", index=False)
print(R.pivot(index="Model", columns="Period", values="rel_RMSE_geomean").loc[models].round(3).rename(columns=lambda c: f"geomean {c}").to_string())
print(R.pivot(index="Model", columns="Period", values="rel_RMSE_pooled").loc[models].round(3).rename(columns=lambda c: f"pooled {c}").to_string())
print("weeks / shock-gate weeks:", R.drop_duplicates("Period")[["Period", "weeks", "shock_gate_weeks"]].to_dict("records"))
