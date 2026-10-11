"""12_tests.py -- statistical comparison and RQ1 (Section 3.8). Reads the forecasts written by 11_walkforward.py.
  (A) Diebold-Mariano (squared error, h = 1, Harvey-Leybourne-Newbold small-sample correction, Student t with T-1 df): gated ensemble vs each benchmark/ablation,
      per commodity (Holm-adjusted across the 10 commodities) and pooled over weeks (commodity losses scaled by each commodity's naive MSE).
  (B) Friedman test over the 10 commodities for the 6 models (B1-B5 + gated) with Nemenyi critical difference (Demsar, 2006).
  (C) Moving-block bootstrap (block length 4, 2000 resamples, whole weeks resampled so the 10 commodities stay together): 95% CI for the geometric-mean
      RMSE ratio gated/benchmark on all weeks, normal and shock target weeks, by gate state, and by trigger.
  (D) RQ1: mean |weekly log price change| on shock indicators (lags 0-3) with Newey-West CIs, on the indicator-free weekly series, plus the same on official PSA monthly prices.
  (E) Real-data check: do predicted weekly movements agree with real DFTC (Bankerohan) weekly price movements?
SAFETY: exactly one flag: --confirm-test (real test-set forecasts; RQ1 also on the full sample) or --dry-run (reads dryrun_ files; RQ1 on training weeks only).
Usage: python 12_tests.py <indicator|noind> --confirm-test"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from importlib import import_module
ev = import_module("eval_utils"); ing = import_module("00_ingest"); pre = import_module("01_preprocess")
OUT = ing.OUT; warnings.filterwarnings("ignore")
BENCH = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated_noGA", "gated_P90"]
BLOCK, B_REPS, SEED = 4, 2000, 20261001

def dm_test(d):
    """d_t = loss(gated) - loss(benchmark). Returns (HLN-corrected DM statistic, two-sided p, mean difference). h = 1 so the long-run variance is the variance of d."""
    d = np.asarray(d, float); T = len(d); dbar = d.mean(); g0 = np.mean((d - dbar) ** 2)
    if g0 <= 0: return np.nan, np.nan, dbar
    dm = dbar / np.sqrt(g0 / T) * np.sqrt((T - 1) / T); return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), T - 1))), float(dbar)

def wide(F, col): return F.pivot(index="week_end", columns="commodity", values=col).sort_index()

def part_dm(F):
    Y = wide(F, "y"); E = {m: wide(F, m) - Y for m in ev.ALL_MODELS}; s2 = (E["naive"] ** 2).mean(); rows = []
    for b in BENCH:
        P = []
        for c in Y.columns:
            dm, p, dbar = dm_test(E["gated"][c] ** 2 - E[b][c] ** 2); P.append(p)
            rows.append(dict(benchmark=b, scope=c, n=len(Y), mean_sq_err_diff=dbar, DM_HLN=dm, p_value=p, gated_better=bool(dbar < 0)))
        adj = multipletests(np.nan_to_num(P, nan=1.0), method="holm")[1]
        for r, a in zip(rows[-len(P):], adj): r["p_holm_over_commodities"] = float(a)
        dw = ((E["gated"] ** 2 - E[b] ** 2) / s2).mean(axis=1).values; dm, p, dbar = dm_test(dw)
        rows.append(dict(benchmark=b, scope="POOLED (weeks)", n=len(dw), mean_sq_err_diff=dbar, DM_HLN=dm, p_value=p, gated_better=bool(dbar < 0), p_holm_over_commodities=np.nan))
    return pd.DataFrame(rows)

def part_friedman(F):
    R = ev.per_commodity(F, ev.MODELS).pivot(index="Commodity", columns="model", values="RMSE")[ev.MODELS]; N, k = R.shape
    stat, p = stats.friedmanchisquare(*[R[m].values for m in ev.MODELS]); ranks = R.apply(lambda r: stats.rankdata(r.values), axis=1, result_type="expand"); ranks.columns = ev.MODELS
    avg = ranks.mean(); q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2); cd = q * np.sqrt(k * (k + 1) / (6 * N))
    T1 = pd.DataFrame(dict(model=ev.MODELS, avg_rank=avg.values, wins=[(ranks[m] == 1).sum() for m in ev.MODELS])); T1["friedman_chi2"], T1["friedman_p"], T1["N"], T1["k"], T1["Nemenyi_CD_0.05"] = stat, p, N, k, cd
    T2 = pd.DataFrame([dict(model_a=a, model_b=b, rank_diff=avg[a] - avg[b], significant_0_05=bool(abs(avg[a] - avg[b]) > cd)) for i, a in enumerate(ev.MODELS) for b in ev.MODELS[i + 1:]])
    return T1, T2

def part_bootstrap(F):
    Y = wide(F, "y"); SE = {m: (wide(F, m) - Y).values ** 2 for m in ["gated"] + BENCH}; wk = Y.index; T, C = Y.shape
    flag = F.drop_duplicates("week_end").set_index("week_end").loc[wk]; t, p = flag.shock_t.values == 1, flag.shock_L1.values == 1; tr = flag.trigger_t.values
    subsets = [("All weeks", np.ones(T, bool)), ("Normal target week", ~t), ("Shock target week", t), ("Gate state normal", ~p), ("Gate state shock", p), ("Rain-only weeks", tr == "rain"), ("Diesel-only weeks", tr == "diesel")]
    rng = np.random.default_rng(SEED); nb = int(np.ceil(T / BLOCK)); IDX = [((rng.integers(0, T - BLOCK + 1, nb)[:, None] + np.arange(BLOCK)).ravel()[:T]) for _ in range(B_REPS)]
    def ratio(b, sel):
        if len(sel) == 0: return np.nan
        return float(np.exp(np.mean(0.5 * (np.log(SE["gated"][sel].mean(0)) - np.log(SE[b][sel].mean(0))))))
    rows = []
    for b in BENCH:
        for name, flg in subsets:
            if flg.sum() == 0: continue
            pt = ratio(b, np.where(flg)[0]); reps = np.array([ratio(b, i[flg[i]]) for i in IDX]); v = reps[np.isfinite(reps)]
            rows.append(dict(benchmark=b, subset=name, n_weeks=int(flg.sum()), ratio_gated_over_benchmark=pt, ci95_low=float(np.percentile(v, 2.5)), ci95_high=float(np.percentile(v, 97.5)), prob_gated_better=float((v < 1).mean()), valid_resamples=len(v)))
    return pd.DataFrame(rows)

def hac_table(y, X, lags, label):
    res = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}); ci = res.conf_int()
    T = pd.DataFrame(dict(sample=label, term=res.params.index, coef=res.params.values, ci95_low=ci[0].values, ci95_high=ci[1].values, p_value=res.pvalues.values, n=int(res.nobs)))
    for grp in ("rain", "diesel"):
        cols = [c for c in X.columns if c.startswith(grp)]; w = res.wald_test(", ".join(f"{c} = 0" for c in cols), scalar=True)
        T = pd.concat([T, pd.DataFrame([dict(sample=label, term=f"JOINT {grp} lags = 0 (Wald)", coef=np.nan, ci95_low=np.nan, ci95_high=np.nan, p_value=float(w.pvalue), n=int(res.nobs))])])
    return T

def part_rq1(dry):
    g = pd.read_csv(OUT / "gate_P95.csv", parse_dates=["week_end"]).set_index("week_end"); n_tr = 357; train_end = g.index[n_tr - 1]
    W = pd.read_csv(OUT / "weekly_prices_disaggregated_noind.csv", parse_dates=["week_end"]).set_index("week_end"); v = (np.log(W).diff().abs() * 100).mean(axis=1)
    X = pd.concat({**{f"rain_L{k}": g.shock_rain.shift(k) for k in range(4)}, **{f"diesel_L{k}": g.shock_diesel.shift(k) for k in range(4)}}, axis=1); D = pd.concat([v.rename("v"), X], axis=1).dropna()
    out = []
    for lab, sub in [("weekly, training weeks", D[D.index <= train_end])] + ([] if dry else [("weekly, full sample", D)]):
        out.append(hac_table(sub.v, sub.drop(columns="v").astype(float), 4, lab))
    pr = pd.read_csv(OUT / "prices_monthly_clean.csv", parse_dates=["month"]).set_index("month"); pr.index = pr.index.to_period("M"); vm = (np.log(pr).diff().abs() * 100).mean(axis=1)
    cnt = pd.DataFrame({"rain": g.shock_rain.values, "diesel": g.shock_diesel.values}, index=g.index.to_period("M")).groupby(level=0).sum(); last_train_month = g.index[n_tr - 1].to_period("M") - 1
    Xm = pd.concat({"rain_L0": cnt.rain, "rain_L1": cnt.rain.shift(1), "diesel_L0": cnt.diesel, "diesel_L1": cnt.diesel.shift(1)}, axis=1); Dm = pd.concat([vm.rename("v"), Xm], axis=1).dropna()
    for lab, sub in [("monthly PSA, training months", Dm[Dm.index <= last_train_month])] + ([] if dry else [("monthly PSA, full sample", Dm)]):
        out.append(hac_table(sub.v, sub.drop(columns="v").astype(float), 3, lab))
    return pd.concat(out, ignore_index=True)

def part_dftc(F, path):
    if not path.exists(): return None
    D = ev.dftc_weekly(path); rows = []
    for c, g in F.groupby("commodity"):
        s = D[c]; g = g.set_index("week_end"); ch = np.log(s / s.shift(1)); ch = ch[s.index.to_series().diff() == pd.Timedelta(days=7)]
        for m in ["target", "sarimax", "xgb", "blend50", "gated"]:
            pred = np.log(g.y / g.naive) if m == "target" else np.log(g[m] / g.naive); df = pd.concat([pred.rename("p"), ch.rename("a")], axis=1).dropna()
            if len(df) < 8: continue
            nz = df[(df.p != 0) & (df.a != 0)]
            rows.append(dict(Commodity=c + ("*" if c in ev.DFTC_IMPERFECT else ""), series=("reconstructed target" if m == "target" else m), n_weeks=len(df), corr_with_DFTC_change=float(df.p.corr(df.a)),
                             sign_agreement=float((np.sign(nz.p) == np.sign(nz.a)).mean()) if len(nz) else np.nan))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    a = sys.argv[1:]; series = next((x for x in a if not x.startswith("--")), "indicator"); dry, confirm = "--dry-run" in a, "--confirm-test" in a
    if dry == confirm: sys.exit("Exactly one flag is required: --confirm-test (real test-set forecasts) or --dry-run (stand-in window from training weeks).")
    pf = ev.prefix(dry); F = pd.read_csv(OUT / f"{pf}wf_forecasts_{series}.csv", parse_dates=["week_end"]); assert F.groupby("commodity").size().nunique() == 1 and len(F.commodity.unique()) == 10, "need the complete forecast file (10 commodities)"
    print(f"[{series}] {'DRY-RUN' if dry else 'TEST SET'}: {len(F)} forecasts, {F.week_end.nunique()} weeks, {int(F.drop_duplicates('week_end').shock_t.sum())} shock weeks")
    DM = part_dm(F); DM.round(5).to_csv(OUT / f"{pf}table_4_dm_{series}.csv", index=False)
    T1, T2 = part_friedman(F); T1.round(4).to_csv(OUT / f"{pf}table_4_friedman_{series}.csv", index=False); T2.round(3).to_csv(OUT / f"{pf}table_4_nemenyi_{series}.csv", index=False)
    BS = part_bootstrap(F); BS.round(4).to_csv(OUT / f"{pf}table_4_bootstrap_{series}.csv", index=False)
    RQ = part_rq1(dry); RQ.round(4).to_csv(OUT / f"{pf}table_4_rq1.csv", index=False)
    DF = part_dftc(F, ing.ROOT / "data" / "raw" / "Data-Scrape-Final.xlsx")
    if DF is not None: DF.round(3).to_csv(OUT / f"{pf}table_4_dftc_{series}.csv", index=False)
    print("\n(A) Pooled-week Diebold-Mariano: gated vs benchmark (negative mean diff = gated better)"); print(DM[DM.scope == "POOLED (weeks)"][["benchmark", "mean_sq_err_diff", "DM_HLN", "p_value"]].round(4).to_string(index=False))
    print("\n(B) Friedman:", dict(chi2=round(float(T1.friedman_chi2.iloc[0]), 3), p=round(float(T1.friedman_p.iloc[0]), 4), CD=round(float(T1["Nemenyi_CD_0.05"].iloc[0]), 3))); print(T1[["model", "avg_rank", "wins"]].round(2).to_string(index=False))
    print("\n(C) Bootstrap, gated/benchmark RMSE ratio vs naive and sarimax:"); print(BS[BS.benchmark.isin(["naive", "sarimax"])].round(3).to_string(index=False))
    print("\n(D) RQ1 (joint Wald tests):"); print(RQ[RQ.term.str.startswith("JOINT")][["sample", "term", "p_value", "n"]].round(4).to_string(index=False))
    if DF is not None: print("\n(E) DFTC real-data check, mean correlation of predicted vs real weekly change:"); print(DF.groupby("series").corr_with_DFTC_change.mean().round(3).to_string())
    print("\nSaved tables to", OUT)
