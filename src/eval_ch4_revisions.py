"""eval_ch4_revisions.py -- two exploratory checks added after the adviser review (Sections 3.8.7 (ix)-(x), 4.4.5 and 4.7.1).
Reads saved outputs only; nothing in the frozen design or the test forecasts is changed.
(ix)  Direction of diesel shocks: the RQ1 volatility regression (Eq. 3.8, no-indicator series, Newey-West 4 lags) with the diesel
      trigger split into large increases (dD >= +5.92%) and large decreases (dD <= -5.92%).
(x)   Week-1 versus monthly-average gap of the monthly-origin evaluation: in the reconstructed weekly series, how far the value of
      the FIRST week of each test month lies from that month's official PSA average (a forecast of week 1 that was exactly right
      would still show this error against the monthly average).
Outputs (data/processed/): table_4_7_rq1_diesel_direction.csv, table_4_4_week1_gap.csv
Usage (project root):  python3 src/eval_ch4_revisions.py"""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

P = Path("data/processed"); N_TRAIN = 357

# ---------------- (ix) diesel increases vs decreases ----------------
g = pd.read_csv(P / "gate_P95.csv", parse_dates=["week_end"]).set_index("week_end")
fuel = pd.read_pickle(P / "fuel_weekly_clean.pkl"); dpct = (fuel.pct_change() * 100).values
assert np.allclose(np.abs(np.nan_to_num(dpct)), g.diesel_abs_pct.values, atol=1e-6), "fuel series does not match the gate"
g["diesel_up"] = ((g.shock_diesel == 1) & (dpct > 0)).astype(int); g["diesel_down"] = ((g.shock_diesel == 1) & (dpct < 0)).astype(int)
W = pd.read_csv(P / "weekly_prices_disaggregated_noind.csv", parse_dates=["week_end"]).set_index("week_end")
v = (np.log(W).diff().abs() * 100).mean(axis=1)
X = pd.concat({**{f"rain_L{k}": g.shock_rain.shift(k) for k in range(4)}, **{f"up_L{k}": g.diesel_up.shift(k) for k in range(4)},
               **{f"down_L{k}": g.diesel_down.shift(k) for k in range(4)}}, axis=1)
D = pd.concat([v.rename("v"), X], axis=1).dropna(); train_end = g.index[N_TRAIN - 1]; rows = []
for lab, sub in [("weekly, training weeks", D[D.index <= train_end]), ("weekly, full sample", D)]:
    gs = g[g.index <= train_end] if lab.startswith("weekly, training") else g
    r = sm.OLS(sub.v, sm.add_constant(sub.drop(columns="v").astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": 4}); ci = r.conf_int()
    for t in r.params.index:
        rows.append(dict(sample=lab, term=t, coef=r.params[t], ci95_low=ci.loc[t, 0], ci95_high=ci.loc[t, 1], p_value=r.pvalues[t], n=int(r.nobs)))
    for grp in ("rain", "up", "down"):
        w = r.wald_test(", ".join(f"{c} = 0" for c in X.columns if c.startswith(grp)), scalar=True)
        rows.append(dict(sample=lab, term=f"JOINT {grp} lags = 0 (Wald)", p_value=float(w.pvalue), n=int(r.nobs)))
    w = r.wald_test(", ".join(f"up_L{k} = down_L{k}" for k in range(4)), scalar=True)
    rows.append(dict(sample=lab, term="EQUAL up = down at every lag (Wald)", p_value=float(w.pvalue), n=int(r.nobs)))
    rows.append(dict(sample=lab, term="count of trigger weeks: diesel increases", coef=float(gs.diesel_up.sum()), n=int(r.nobs)))
    rows.append(dict(sample=lab, term="count of trigger weeks: diesel decreases", coef=float(gs.diesel_down.sum()), n=int(r.nobs)))
T = pd.DataFrame(rows); T.round(4).to_csv(P / "table_4_7_rq1_diesel_direction.csv", index=False)
pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))

# ---------------- (x) week-1 vs monthly-average gap ----------------
M = pd.read_csv(P / "wf_monthly_indicator.csv"); months = sorted(M.month.unique())
pr = pd.read_csv(P / "prices_monthly_clean.csv", parse_dates=["month"]).set_index("month"); pr.index = pr.index.to_period("M")
out = []
for s, fn in [("indicator", "weekly_prices_disaggregated.csv"), ("noind", "weekly_prices_disaggregated_noind.csv")]:
    Wk = pd.read_csv(P / fn, parse_dates=["week_end"]).set_index("week_end"); pm = Wk.index.to_period("M")
    for m in months:
        mp = pd.Period(m, "M"); sub = Wk[pm == mp]
        for c in Wk.columns:
            out.append(dict(series=s, commodity=c, month=m, ape_week1_vs_month=100 * abs(sub[c].iloc[0] - pr.loc[mp, c]) / pr.loc[mp, c]))
G = pd.DataFrame(out)
S = G.groupby(["series", "commodity"]).ape_week1_vs_month.mean().unstack(0)
S.loc["Mean"] = G.groupby("series").ape_week1_vs_month.mean(); S.loc["Median (all commodity-months)"] = G.groupby("series").ape_week1_vs_month.median()
for s in ("indicator", "noind"):
    Mo = pd.read_csv(P / f"wf_monthly_{s}.csv")
    for k in ("psa_naive", "gated"): S.loc[f"MAPE of {k} in the monthly-origin evaluation", s] = 100 * np.mean(np.abs(Mo[k] - Mo.actual_psa) / Mo.actual_psa)
S.round(3).to_csv(P / "table_4_4_week1_gap.csv"); print("\n", S.round(2).to_string())
