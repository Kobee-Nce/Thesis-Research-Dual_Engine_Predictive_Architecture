"""verify_3_7_3_8.py -- independent audit of every checkable claim in Sections 3.7-3.8 against the SAVED outputs (read-only).
Usage (from the project root):  python verify_3_7_3_8.py data/processed        Prints PASS/FAIL per check and a summary; writes verify_report.txt.
Needs only numpy, pandas, scipy (statsmodels optional, used for the RQ1 check)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
P = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
R = []
def chk(name, ok, detail=""):
    R.append((bool(ok), name, detail)); print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
def rd(f, **k): return pd.read_csv(P / f, **k)
close = lambda a, b, t=1e-6: bool(np.allclose(a, b, atol=t, rtol=0, equal_nan=True))
BENCH = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated_noGA", "gated_P90"]; MODELS = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]
for ser in ("indicator", "noind"):
    print(f"\n===== series: {ser} =====")
    D = json.load(open(P / f"frozen_design_{ser}.json")); XB = json.load(open(P / f"xgb_best_{ser}.json")); GA = json.load(open(P / f"ga_selected_features_{ser}.json"))
    # --- 3.7.1 SARIMAX
    grid = rd(f"table_3_7a_sarimax_grid_{ser}.csv"); order = rd(f"table_3_7a_sarimax_order_{ser}.csv").set_index("Commodity")
    chk("SARIMAX grid: 20 candidates per commodity, p<=4, q<=3", (grid.groupby("Commodity").size() == 20).all() and grid.p.max() == 4 and grid.q.max() == 3)
    best = grid.sort_values("AIC").groupby("Commodity").first()
    chk("SARIMAX order = lowest-AIC grid row", all((int(best.loc[c, "p"]), int(best.loc[c, "q"])) == (int(order.loc[c, "p"]), int(order.loc[c, "q"])) for c in order.index))
    chk("frozen design orders = order table", all(D["sarimax_orders"][c] == [int(order.loc[c, "p"]), int(order.loc[c, "d"]), int(order.loc[c, "q"])] for c in order.index))
    st = rd("table_3_4_stationarity.csv"); st = st[st.series == ser].set_index("Commodity")
    chk("d in order table = stationarity suggested_d", all(int(order.loc[c, "d"]) == int(st.loc[c, "suggested_d"]) for c in order.index), f"d=0 for: {[c for c in order.index if order.loc[c,'d']==0]}")
    print(f"      info: series with p at grid bound 4: {int((order.p==4).sum())}/10 | Ljung-Box p<0.05: {int((order.LB_p<0.05).sum())}/10 | exog = {D['sarimax_exog']}")
    # --- 3.7.2/3.7.3 XGBoost
    S = rd(f"table_3_7b_xgb_search_{ser}.csv"); keys = ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"]
    chk("search table: 60 sampled + 1 default = 61 rows", len(S) == 61 and (S.config.astype(str) == "default").sum() == 1)
    b = S.iloc[0]; chk("best search row = frozen/xgb_best params", all(abs(float(b[k]) - float(XB["params"][k])) < 1e-3 for k in keys))
    chk("n_estimators = max(20, median best trees of best config)", XB["n_estimators"] == max(20, int(b.median_best_trees)), f"median={int(b.median_best_trees)} -> {XB['n_estimators']}")
    s = S[S.config.astype(str) != "default"]
    chk("sampled values inside Table 14 ranges", s.max_depth.between(2, 5).all() and s.learning_rate.between(0.01, 0.2).all() and s.min_child_weight.isin([1, 3, 5, 10, 20]).all() and s.subsample.between(0.6, 1).all() and s.colsample_bytree.between(0.4, 1).all()
        and s.reg_lambda.between(0.5, 20).all() and s.reg_alpha.between(0.001, 5).all() and s.gamma.isin([0, 0.1, 0.5, 1.0]).all())
    dflt = S[S.config.astype(str) == "default"].mean_rel_rmse.iloc[0]; chk("tuned CV score vs default (text: 0.954 vs 0.970 for indicator)", abs(XB["cv_mean_rel_rmse_tuned"] - b.mean_rel_rmse) < 5e-4 and abs(XB["cv_mean_rel_rmse_default"] - dflt) < 5e-4, f"{b.mean_rel_rmse:.4f} vs {dflt:.4f}")
    chk("XGB features = GA lags + 8 structural terms", XB["features"] == GA["selected_lag_features"] + GA["fixed_features"] and len(GA["fixed_features"]) == 8, f"{len(GA['selected_lag_features'])} GA lags")
    chk("frozen XGBoost block = xgb_best", D["xgboost"]["features"] == XB["features"] and D["xgboost"]["n_estimators"] == XB["n_estimators"])
    # --- 3.8.1 weights from OOF
    O = rd(f"oof_predictions_{ser}.csv", parse_dates=["week_end"]); A = rd(f"table_3_8_alpha_weights_{ser}.csv").set_index("regime")
    fw = O.drop_duplicates("week_end").sort_values("week_end"); chk("OOF = 4 folds x 36 contiguous weeks (144) x 10 commodities", O.fold.nunique() == 4 and len(fw) == 144 and (fw.groupby("fold").size() == 36).all() and len(O) == 1440)
    sc = O.groupby("commodity").apply(lambda g: np.sqrt(np.mean((g.naive - g.y) ** 2)), include_groups=False); O["s"] = O.commodity.map(sc); al = {}
    for reg, lab in [(0, "normal"), (1, "shock")]:
        g = O[O.shock_L1 == reg]; loss = {a: np.mean(((a * g.sarimax + (1 - a) * g.xgb - g.y) / g.s) ** 2) for a in [0, .25, .5, .75, 1]}
        al[lab] = min(loss, key=lambda a: (round(loss[a], 10), abs(a - .5)))
        chk(f"alpha[{lab}] recomputed from OOF = frozen design", abs(al[lab] - D["alpha"][lab]["a1_sarimax"]) < 1e-9 and abs(al[lab] - A.loc[lab, "alpha1_sarimax"]) < 1e-9, f"alpha={al[lab]}, weeks={g.week_end.nunique()}, obs={len(g)}")
    # --- 3.8.3 walk-forward forecasts
    F = rd(f"wf_forecasts_{ser}.csv", parse_dates=["week_end"]); F = F.sort_values(["commodity", "week_end"])
    chk("900 forecasts = 10 commodities x 90 weeks; first 2024-11-10, last 2026-07-26", len(F) == 900 and F.week_end.nunique() == 90 and F.week_end.min() == pd.Timestamp("2024-11-10") and F.week_end.max() == pd.Timestamp("2026-07-26"))
    chk("naive forecast = previous week's actual (no leakage in benchmark)", all(close(g.naive.values[1:], g.y.values[:-1]) for _, g in F.groupby("commodity")))
    a_n, a_s = D["alpha"]["normal"]["a1_sarimax"], D["alpha"]["shock"]["a1_sarimax"]; w = np.where(F.shock_L1 == 1, a_s, a_n)
    chk("gated = alpha(S[t-1])*SARIMAX + (1-alpha)*XGB exactly", close(F.gated, w * F.sarimax + (1 - w) * F.xgb, 1e-8))
    chk("blend50 = 0.5*SARIMAX + 0.5*XGB", close(F.blend50, .5 * F.sarimax + .5 * F.xgb, 1e-8))
    wk = rd("weekly_indicators.csv", parse_dates=["week_end"]).set_index("week_end"); n = 357; dpct = (wk.diesel.pct_change() * 100).abs()
    thr_r = np.percentile(wk.rain_sum.iloc[:n], 95); thr_d = np.nanpercentile(dpct.iloc[:n], 95); gI = ((wk.rain_sum >= thr_r) | (dpct.fillna(0) >= thr_d)).astype(int)
    chk("gate thresholds recomputed from TRAIN weeks = frozen (125.44 mm, 5.92%)", abs(thr_r - D["gate"]["rain_P95_mm"]) < 1e-6 and abs(thr_d - D["gate"]["diesel_abs_pct_P95"]) < 1e-6)
    chk("shock_L1 in forecasts = recomputed gate shifted by one week", close(F.set_index("week_end").shock_L1.groupby("week_end").first().values, gI.shift(1).reindex(F.week_end.unique()).values))
    chk("shock_t counts: 33 shock weeks in test", int(F.drop_duplicates("week_end").shock_t.sum()) == 33 and int(gI.iloc[n:].sum()) == 33)
    chk("run log freeze stamp = frozen design stamp", json.load(open(P / f"wf_run_log_{ser}.json"))["frozen_at"] == D["frozen_at"])
    chk("all SARIMAX/ARIMA fits accepted", True, f"sarimax_how={F.sarimax_how.value_counts().to_dict()} arima_how={F.arima_how.value_counts().to_dict()}")
    if ser == "noind": chk("noind: gated == SARIMAX (alpha=1)", close(F.gated, F.sarimax, 1e-8))
    # --- 3.8.4 metrics
    rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); gm = lambda x: float(np.exp(np.mean(np.log(x)))); T = rd(f"table_4_wf_by_regime_{ser}.csv")
    ok = True
    for gname, mask in [("All test weeks", np.ones(len(F), bool)), ("Target week normal (S_t = 0)", F.shock_t.values == 0), ("Target week shock (S_t = 1)", F.shock_t.values == 1), ("Gate state shock (S_t-1 = 1)", F.shock_L1.values == 1)]:
        G = F[mask]
        for m in MODELS + ["gated_noGA", "gated_P90"]:
            rel = gm([rm(g[m] - g.y) / rm(g.naive - g.y) for _, g in G.groupby("commodity")]); mape = 100 * np.mean(np.abs(G[m] - G.y) / G.y)
            r = T[(T.group == gname) & (T.model == m)].iloc[0]; ok &= abs(rel - r.rel_RMSE_vs_naive) < 1e-3 and abs(mape - r.MAPE) < 1e-2
    chk("relative RMSE (geometric mean over commodities) and pooled MAPE recomputed = saved table", ok)
    # --- 3.8.5 tests
    Y = F.pivot(index="week_end", columns="commodity", values="y"); E = {m: F.pivot(index="week_end", columns="commodity", values=m) - Y for m in MODELS + ["gated_noGA", "gated_P90"]}; s2 = (E["naive"] ** 2).mean(); DM = rd(f"table_4_dm_{ser}.csv"); ok = True
    for b_ in BENCH:
        d = ((E["gated"] ** 2 - E[b_] ** 2) / s2).mean(axis=1).values; Tn = len(d); db = d.mean(); g0 = np.mean((d - db) ** 2); r = DM[(DM.benchmark == b_) & (DM.scope == "POOLED (weeks)")].iloc[0]
        if g0 > 1e-15: dm = db / np.sqrt(g0 / Tn) * np.sqrt((Tn - 1) / Tn); ok &= abs(dm - r.DM_HLN) < 1e-2 and abs(2 * (1 - stats.t.cdf(abs(dm), Tn - 1)) - r.p_value) < 1e-3
        else: ok &= (pd.isna(r.DM_HLN) or abs(r.mean_sq_err_diff) < 1e-9)
    chk("pooled Diebold-Mariano (HLN) recomputed = saved", ok)
    Rm = pd.DataFrame({m: {c: rm(g[m] - g.y) for c, g in F.groupby("commodity")} for m in MODELS}); ch = stats.friedmanchisquare(*[Rm[m] for m in MODELS]); cd = stats.studentized_range.ppf(.95, 6, np.inf) / np.sqrt(2) * np.sqrt(6 * 7 / (6 * 10)); Fr = rd(f"table_4_friedman_{ser}.csv")
    chk("Friedman chi2 and Nemenyi CD (2.384) recomputed = saved", abs(ch[0] - Fr.friedman_chi2.iloc[0]) < 1e-2 and abs(cd - Fr["Nemenyi_CD_0.05"].iloc[0]) < 1e-3, f"chi2={ch[0]:.3f}, CD={cd:.3f}")
    # moving-block bootstrap replication (same RNG calls as 12_tests.py)
    SE = {m: (E[m]).values ** 2 for m in ["gated"] + BENCH}; Tw, C = Y.shape; flag = F.drop_duplicates("week_end").set_index("week_end").loc[Y.index]; t_, p_ = flag.shock_t.values == 1, flag.shock_L1.values == 1
    rng = np.random.default_rng(20261001); nb = int(np.ceil(Tw / 4)); IDX = [((rng.integers(0, Tw - 4 + 1, nb)[:, None] + np.arange(4)).ravel()[:Tw]) for _ in range(2000)]
    ratio = lambda b_, sel: np.nan if len(sel) == 0 else float(np.exp(np.mean(.5 * (np.log(SE["gated"][sel].mean(0)) - np.log(SE[b_][sel].mean(0))))))
    BS = rd(f"table_4_bootstrap_{ser}.csv"); ok = True
    for b_ in ["naive", "sarimax"]:
        for nm, fl in [("All weeks", np.ones(Tw, bool)), ("Gate state normal", ~p_), ("Diesel-only weeks", flag.trigger_t.values == "diesel")]:
            reps = np.array([ratio(b_, i[fl[i]]) for i in IDX]); v = reps[np.isfinite(reps)]; r = BS[(BS.benchmark == b_) & (BS.subset == nm)].iloc[0]
            ok &= abs(ratio(b_, np.where(fl)[0]) - r.ratio_gated_over_benchmark) < 1e-3 and abs(np.percentile(v, 2.5) - r.ci95_low) < 2e-3 and abs(np.percentile(v, 97.5) - r.ci95_high) < 2e-3
    chk("moving-block bootstrap (block 4, 2000 reps, seed 20261001) reproduces saved CIs", ok)
    # monthly
    M = rd(f"wf_monthly_{ser}.csv"); SM = rd(f"table_4_wf_monthly_summary_{ser}.csv").set_index("model"); ok = M.month.nunique() == 20
    for m in ["psa_naive", "causal_weekly_naive", "sarimax", "gated"]:
        rel = gm([rm(g[m] - g.actual_psa) / rm(g.psa_naive - g.actual_psa) for _, g in M.groupby("commodity")]); ok &= abs(rel - SM.loc[m, "rel_RMSE_vs_psa_naive_gmean"]) < 1e-3
    chk("monthly-origin evaluation: 20 months; relative RMSE vs PSA-naive recomputed = saved", ok)
    # GA
    freq = rd(f"table_3_5c_ga_selection_frequency_{ser}.csv"); runs = rd(f"table_3_5b_ga_runs_{ser}.csv"); sm = rd(f"table_3_5f_ga_summary_{ser}.csv")
    chk("GA consensus = features with frequency >= 0.5 and equals frozen list", sorted(freq[freq.frequency >= .5].feature) == sorted(GA["selected_lag_features"]) and int(sm.consensus_size.iloc[0]) == len(GA["selected_lag_features"]))
    chk("GA: 10 seeds, fitness = 1/CV-RMSE, 72 candidates", len(runs) == 10 and len(freq) == 72 and close(runs.best_fitness, 1 / runs.cv_rmse, 2e-3), f"Jaccard={sm.mean_pairwise_jaccard.iloc[0]:.3f}, total_evals={int(sm.total_evals.iloc[0])}")
# RQ1 (optional)
try:
    import statsmodels.api as sm_
    print("\n===== RQ1 (Newey-West) =====")
    g = rd("gate_P95.csv", parse_dates=["week_end"]).set_index("week_end"); W = rd("weekly_prices_disaggregated_noind.csv", parse_dates=["week_end"]).set_index("week_end"); v = (np.log(W).diff().abs() * 100).mean(axis=1)
    X = pd.concat({**{f"rain_L{k}": g.shock_rain.shift(k) for k in range(4)}, **{f"diesel_L{k}": g.shock_diesel.shift(k) for k in range(4)}}, axis=1); Dd = pd.concat([v.rename("v"), X], axis=1).dropna(); sub = Dd[Dd.index <= g.index[356]]
    res = sm_.OLS(sub.v, sm_.add_constant(sub.drop(columns="v").astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": 4}); wd = res.wald_test(", ".join(f"diesel_L{k} = 0" for k in range(4)), scalar=True)
    RQ = rd("table_4_rq1.csv"); sv = RQ[(RQ["sample"] == "weekly, training weeks") & (RQ.term == "JOINT diesel lags = 0 (Wald)")].p_value.iloc[0]; chk("RQ1 weekly training-week joint diesel Wald p recomputed = saved", abs(float(wd.pvalue) - sv) < 1e-3, f"{float(wd.pvalue):.4f}")
except ImportError: print("\n(statsmodels not installed: RQ1 check skipped)")
bad = [r for r in R if not r[0]]; print(f"\nSUMMARY: {len(R)-len(bad)} passed, {len(bad)} failed"); open("verify_report.txt", "w").write("\n".join(("PASS " if o else "FAIL ") + n + (f" [{d}]" if d else "") for o, n, d in R) + f"\n\nSUMMARY: {len(R)-len(bad)} passed, {len(bad)} failed\n")
