"""02b_disaggregation_backtest.py -- aggregate-then-reconstruct backtest (RQ3 / Gate G2).
Monthly PSA -> quarterly mean -> disaggregate to weeks -> re-average to months -> compare with true PSA months.
Methods: CL (rain+diesel, rho estimated) | CL-const (no indicators, rho estimated) | Flat (quarter mean) | CL-rho0.5 (previous setting)."""
import numpy as np, pandas as pd
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); dis = import_module("02_disaggregate"); OUT = ing.OUT

if __name__ == "__main__":
    prices = ing.load_prices(); weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks))
    wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    X = dis.build_indicators(wk, fu, n_train)
    pm = pd.Series(weeks.to_period("M"), index=weeks); pq = pd.Series(weeks.to_period("Q"), index=weeks)
    trq = dis.train_periods_of(weeks, n_train, "Q"); first_test_month = pm.iloc[n_train:].min()
    rows = []
    for c in prices.columns:
        y = prices[c].copy(); y.index = y.index.to_period("M")
        yq = y.groupby(y.index.asfreq("Q")).mean(); yq = yq.loc[yq.index != pm.index[-1].to_period("Q")] if False else yq
        yq = yq.iloc[:-1]                                  # drop 2026Q3: only July is in the window (partial quarter)
        months = y.index[y.index.asfreq("Q").isin(yq.index)]
        rec = {}
        for name, Xm, rho in [("CL", X, None), ("CLconst", X[["const"]], None), ("CLrho05", X, 0.5)]:
            p, _, r = dis.chow_lin(yq, Xm, pq, trq, rho=rho); rec[name] = p.groupby(pm).mean().loc[months]
            if name == "CL": rho_q = r
        rec["Flat"] = pd.Series(yq.reindex(months.asfreq("Q")).values, index=months)
        a = y.loc[months]; test = months >= first_test_month
        for sub, mask in [("all", np.ones(len(months), bool)), ("test", test)]:
            row = {"Commodity": c, "subset": sub, "n_months": int(mask.sum()), "rho_q": rho_q}
            for k, s in rec.items():
                e = (s - a).values[mask]; row[f"RMSE_{k}"] = np.sqrt((e ** 2).mean()); row[f"MAE_{k}"] = np.abs(e).mean(); row[f"MAPE_{k}"] = (np.abs(e) / a.values[mask]).mean() * 100
            rows.append(row)
    T = pd.DataFrame(rows).round(2); T.to_csv(OUT / "table_3_3_disaggregation_backtest.csv", index=False)
    for sub in ["all", "test"]:
        t = T[T.subset == sub].set_index("Commodity"); print(f"\n== {sub} months (n={t.n_months.iloc[0]}) ==")
        print(t[[f"RMSE_{k}" for k in ["CL", "CLconst", "Flat", "CLrho05"]] + ["MAPE_CL", "MAPE_Flat"]].to_string())
        print("CL beats Flat on", int((t.RMSE_CL < t.RMSE_Flat).sum()), "/10 | CL beats CL-const on", int((t.RMSE_CL < t.RMSE_CLconst).sum()), "/10 | CL beats old rho=.5 on", int((t.RMSE_CL < t.RMSE_CLrho05).sum()), "/10")
