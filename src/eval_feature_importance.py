"""eval_feature_importance.py -- Section 4.6: XGBoost feature importance and feature-group removal (TRAINING weeks only).
Uses the frozen design (xgb_best_<series>.json): same features, hyperparameters and 20 trees; nothing is re-selected.
(1) Gain importance: frozen model fitted on all training weeks of each commodity; gain normalised to sum to 1 per commodity,
    then averaged over the ten commodities.
(2) Group removal: on the four rolling-origin folds, the frozen model is refitted without one feature group at a time;
    score = mean over commodities of validation RMSE / naive RMSE (the tuning score of Section 3.7.3, with the frozen 20 trees).
Multi-seed: because the frozen model samples columns (colsample_bytree = 0.47) and rows, a single random_state mixes the effect of
removing a group with sampling noise. Both analyses are therefore repeated for SEEDS (the frozen seed 42 and nine more); the removal
change is computed within each seed (removed minus full model, same seed) and summarised by its mean, SD and the share of seeds in
which removal raises the error.
Outputs: table_4_6_gain_importance_<series>.csv, table_4_6_group_removal_<series>.csv, table_4_6_group_removal_seeds_<series>.csv,
         fig_ch4_gain_importance.png, fig_ch4_group_removal.png
Usage: python eval_feature_importance.py   (runs both series)"""
import json, warnings, numpy as np, pandas as pd, xgboost as xgb, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
cu = import_module("cv_utils"); OUT = cu.OUT; warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

def group_of(f):
    if f.startswith("y_L"): return "Price lags"
    if f.startswith(("rain_sum", "rain_max", "rain_roll4")): return "Rainfall lags"
    if f.startswith("wind"): return "Wind lags"
    if f.startswith(("t_mean", "t_max", "t_min")): return "Temperature lags"
    if f.startswith("diesel"): return "Diesel-change lags"
    if f.startswith("cpi"): return "CPI m/m change"
    if f.startswith(("sin", "cos")): return "Fourier terms"
    if f == "shock_L1": return "Gate state $S_{t-1}$"
    return "Rolling price mean/SD"
LABEL = lambda f: (f.replace("y_roll4_mean", "price mean t-4..t-1").replace("y_roll4_std", "price SD t-4..t-1").replace("cpi_veg_mom_L1m", "CPI m/m (prev. month)")
                   .replace("shock_L1", "gate state S(t-1)").replace("y_L", "price L").replace("diesel_pct_L", "diesel chg L").replace("rain_sum_L", "rain total L")
                   .replace("rain_max_L", "rain max L").replace("rain_roll4_L", "rain 4-wk L").replace("wind_max_L", "wind max L").replace("t_max_L", "T max L")
                   .replace("t_min_L", "T min L").replace("t_mean_L", "T mean L"))

SEEDS = [42] + list(range(1, 10))

def model(B, seed=42):
    p = dict(B["params"]); p.update(B["fixed"]); p["n_estimators"] = B["n_estimators"]; p["random_state"] = seed; return xgb.XGBRegressor(**p)

def cv_score(P, feats, B, seed=42):
    coms = list(P.commodity.unique()); n = (P.commodity == coms[0]).sum(); folds = cu.rolling_folds(n); rel = []
    for c in coms:
        d = P[P.commodity == c]; X = d[feats].values; y = d.y.values; y1 = d.y_L1.values; e, e0 = [], []
        for vs, ve in folds:
            m = model(B, seed).fit(X[:vs], (y - y1)[:vs]); e.append(y1[vs:ve] + m.predict(X[vs:ve]) - y[vs:ve]); e0.append(y1[vs:ve] - y[vs:ve])
        rel.append(np.sqrt(np.mean(np.concatenate(e) ** 2)) / np.sqrt(np.mean(np.concatenate(e0) ** 2)))
    return float(np.mean(rel)), rel

res = {}
EXO = ["Rainfall lags", "Wind lags", "Temperature lags", "Diesel-change lags", "CPI m/m change", "Gate state $S_{t-1}$"]
for s in ("indicator", "noind"):
    B = json.load(open(OUT / f"xgb_best_{s}.json")); feats = B["features"]; P = cu.load_train(s)
    G = []
    for seed in SEEDS:
        for c, d in P.groupby("commodity"):
            m = model(B, seed).fit(d[feats].values, (d.y - d.y_L1).values); g = m.get_booster().get_score(importance_type="gain")
            v = np.array([g.get(f"f{i}", 0.0) for i in range(len(feats))]); G.append(v / v.sum())
    Gm = pd.DataFrame(dict(feature=feats, group=[group_of(f) for f in feats], gain_share=np.mean(G, 0), gain_sd=np.std(G, 0, ddof=1))).sort_values("gain_share", ascending=False)
    Gm.round(4).to_csv(OUT / f"table_4_6_gain_importance_{s}.csv", index=False)
    groups = list(dict.fromkeys(group_of(f) for f in feats)); sets = {g: [f for f in feats if group_of(f) != g] for g in groups}
    sets["All weather, fuel, CPI and gate inputs"] = [f for f in feats if group_of(f) not in EXO]
    rows = []
    for seed in SEEDS:
        full, _ = cv_score(P, feats, B, seed); rows.append(dict(seed=seed, removed="None", n_inputs=len(feats), score=full, change=0.0))
        for g, keep in sets.items():
            sc, _ = cv_score(P, keep, B, seed); rows.append(dict(seed=seed, removed=g, n_inputs=len(keep), score=sc, change=sc - full))
    S = pd.DataFrame(rows); S.round(5).to_csv(OUT / f"table_4_6_group_removal_seeds_{s}.csv", index=False)
    R = S.groupby("removed", sort=False).agg(n_inputs=("n_inputs", "first"), score_mean=("score", "mean"), score_seed42=("score", "first"),
                                             change_mean=("change", "mean"), change_sd=("change", lambda x: x.std(ddof=1)),
                                             share_seeds_worse=("change", lambda x: (x > 0).mean())).reset_index()
    R.loc[R.removed == "None", "removed"] = f"None (frozen {len(feats)}-input model)"
    R.round(4).to_csv(OUT / f"table_4_6_group_removal_{s}.csv", index=False)
    print(f"\n== {s}  ({len(SEEDS)} seeds)\n", Gm.round(3).to_string(index=False), "\n", R.round(4).to_string(index=False)); res[s] = (Gm, R)

fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, s in zip(ax, res):
    Gm = res[s][0].iloc[::-1]; y = np.arange(len(Gm))
    a.barh(y, Gm.gain_share, xerr=Gm.gain_sd, color="#2a78d6", ecolor="0.6", capsize=2)
    a.set_yticks(y); a.set_yticklabels([LABEL(f) for f in Gm.feature], fontsize=8); a.set_xlabel(f"Share of total gain (mean over commodities and {len(SEEDS)} seeds)"); a.set_axisbelow(True)
    a.set_title({"indicator": "Indicator series", "noind": "No-indicator series"}[s], fontsize=10); a.grid(axis="x", color="0.92")
fig.tight_layout(); fig.savefig(OUT / "fig_ch4_gain_importance.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for a, s in zip(ax, res):
    R = res[s][1].iloc[1:].sort_values("change_mean"); y = np.arange(len(R))
    col = np.where(R.share_seeds_worse >= .8, "#eb6834", np.where(R.share_seeds_worse <= .2, "#1baf7a", "0.65"))
    a.barh(y, R.change_mean * 100, xerr=R.change_sd * 100, color=col, ecolor="0.35", capsize=2); a.axvline(0, color="0.3", lw=0.8)
    a.set_yticks(y); a.set_yticklabels(R.removed, fontsize=8.5); a.set_xlabel("Change in score when removed (pp, mean \u00b1 SD over seeds)"); a.set_axisbelow(True)
    a.set_title({"indicator": "Indicator series", "noind": "No-indicator series"}[s] + f" (full model {res[s][1].score_mean.iloc[0]:.3f})", fontsize=10); a.grid(axis="x", color="0.92")
fig.text(.5, .01, "Orange: removal raises the error in at least 80% of seeds. Green: lowers it in at least 80% of seeds. Grey: inconsistent across seeds.", ha="center", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT / "fig_ch4_group_removal.png", dpi=200); plt.close(fig)
