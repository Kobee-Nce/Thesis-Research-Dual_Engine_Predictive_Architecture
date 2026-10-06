"""11_walkforward.py -- walk-forward (expanding-window) evaluation, h = 1 week (Sections 3.1.4, 3.2, 3.8).
READS the frozen design (frozen_design_<series>.json, xgb_best_<series>.json); it never re-tunes anything.
For each test week t and commodity: refit the model PARAMETERS on all rows before t, forecast week t using information up to t-1.
Models: B1 naive | B2 ARIMA+Fourier (same (p,d,q) as SARIMAX, Fourier terms only) | B3 SARIMAX (no gate) | B4 XGBoost alone | B5 fixed 50/50
        | gated ensemble (alpha by gate state S_t-1) | A1 gated with no-GA XGBoost (all 72 lags + 8 structural terms) | A3 gated with the P90 gate.
(A2 = run this script on the other weekly series.)  Also runs the LEAK-FREE MONTHLY-ORIGIN evaluation (see monthly_origin()).

SAFETY: exactly one flag is required.
  --confirm-test  evaluate on the real 90-week TEST SET (do this only after the design is frozen; after this the design must not change)
  --dry-run       debugging only: loads TRAIN rows only and uses the last 90 training weeks as a stand-in test set (outputs prefixed dryrun_)
Optional: --skip-monthly   --max-weeks=N   --commodities=A,B   --fresh (ignore saved partial results)
Usage: python 11_walkforward.py <indicator|noind> --confirm-test"""
import sys, json, time, warnings, platform
import numpy as np, pandas as pd, xgboost as xgb, statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from importlib import import_module
cu = import_module("cv_utils"); ev = import_module("eval_utils"); gate = import_module("05_gate"); pre = import_module("01_preprocess")
dis = import_module("02_disaggregate"); ing = import_module("00_ingest")
OUT = ing.OUT; warnings.filterwarnings("ignore")
Y_DERIVED = ["y"] + [f"y_L{k}" for k in range(1, 9)] + ["y_roll4_mean", "y_roll4_std"]
FOURIER = ["sin1", "cos1", "sin2", "cos2"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]      # same coarse grid as Section 3.6.5
N_PSEUDO = 90

def parse():
    a = sys.argv[1:]; series = next((x for x in a if not x.startswith("--")), "indicator")
    opt = {x.split("=")[0]: (x.split("=")[1] if "=" in x else True) for x in a if x.startswith("--")}
    return series, opt

def sarimax_step(y, y1, Z, order, i, prev):
    """Fit on rows [0:i], forecast row i. PRE-SPECIFIED ACCEPTANCE RULE: a fit is accepted only if the optimizer converged AND the forecast is finite AND
    within 10 x the standard deviation of the training window's weekly price changes of the last observed price. Otherwise the previous week's
    parameters are re-used (state-space smoothing) under the same plausibility check; if none are usable, the naive forecast is used.
    Returns (forecast, params, converged, how) with how in {fit, reused_params, naive_fallback}."""
    mod = SARIMAX(y[:i], exog=Z[:i], order=order, trend="c" if order[1] == 0 else "n"); scale = float(np.std(np.diff(y[:i])))
    plausible = lambda f: bool(np.isfinite(f) and abs(f - y1[i]) <= 10 * scale)
    try:
        res = mod.fit(disp=False, maxiter=200, method="lbfgs"); f = float(np.asarray(res.forecast(steps=1, exog=Z[i:i + 1]))[0]); conv = bool(res.mle_retvals.get("converged", True))
        if conv and plausible(f): return f, res.params, True, "fit"
        raise ValueError("fit rejected")
    except Exception:
        if prev is not None:
            try:
                f = float(np.asarray(mod.smooth(prev).forecast(steps=1, exog=Z[i:i + 1]))[0])
                if plausible(f): return f, prev, False, "reused_params"
            except Exception: pass
        return float(y1[i]), prev, False, "naive_fallback"

def xgb_step(X, dy, y1, i, fixed, params, n_trees):
    m = xgb.XGBRegressor(**fixed, **params, n_estimators=n_trees).fit(X[:i], dy[:i]); return float(y1[i] + m.predict(X[i:i + 1])[0])

def tune_alpha(O, regime_col):
    """Same rule as 10_gate_weights.py: pooled squared error / commodity naive RMSE, coarse grid, ties -> closest to 0.5."""
    sc = O.groupby("commodity").apply(lambda x: np.sqrt(np.mean((x.naive - x.y) ** 2)), include_groups=False); O = O.assign(s=O.commodity.map(sc)); res = {}
    for reg in (0, 1):
        x = O[O[regime_col] == reg]; L = {a: np.mean(((a * x.sarimax + (1 - a) * x.xgb - x.y) / x.s) ** 2) for a in ALPHAS}
        res[reg] = float(min(ALPHAS, key=lambda a: (round(L[a], 10), abs(a - 0.5))))
    return res

def run_weekly(P, i0, n, D, XB, a90, G, sig, part_dir, opt):
    orders = {c: tuple(v) for c, v in D["sarimax_orders"].items()}; fixed, prm, nt = XB["fixed"], D["xgboost"]["params"], D["xgboost"]["n_estimators"]
    xf = D["xgboost"]["features"]; a_n, a_s = D["alpha"]["normal"]["a1_sarimax"], D["alpha"]["shock"]["a1_sarimax"]
    coms = sorted(P.commodity.unique()); sel = opt.get("--commodities"); coms = [c for c in coms if c in sel.split(",")] if sel else coms
    stop = i0 + int(opt["--max-weeks"]) if "--max-weeks" in opt else n; frames = []; part_dir.mkdir(exist_ok=True)
    sigf = part_dir / "signature.txt"
    if sigf.exists() and (sigf.read_text() != sig or "--fresh" in opt):
        for f in part_dir.glob("*.csv"): f.unlink()
    sigf.write_text(sig); t0 = time.time()
    for c in coms:
        pf = part_dir / f"{c}.csv"
        if pf.exists() and "--max-weeks" not in opt: frames.append(pd.read_csv(pf, parse_dates=["week_end"])); print(f"  {c}: loaded saved result", flush=True); continue
        d = P[P.commodity == c].reset_index(drop=True).merge(G[["week_end", "shock_L1_P90"]], on="week_end", how="left")
        y, y1 = d.y.values, d.y_L1.values; dy = y - y1; Zr, Zf = d[cu.SARIMAX_EXOG].values, d[FOURIER].values
        Xg, Xa = d[xf].values, d[cu.LAG_COLS + cu.FIXED_COLS].values; rows, ps, pa = [], None, None
        for i in range(i0, stop):
            Z = (Zr - Zr[:i].mean(0)) / Zr[:i].std(0)                       # standardized with statistics of rows before i only
            fs, ps, cs, hs = sarimax_step(y, y1, Z, orders[c], i, ps); fa, pa, ca, ha = sarimax_step(y, y1, Zf, orders[c], i, pa)
            fx = xgb_step(Xg, dy, y1, i, fixed, prm, nt); fx_all = xgb_step(Xa, dy, y1, i, fixed, prm, nt)
            s1 = int(d.shock_L1.values[i]); w = a_s if s1 == 1 else a_n; s9 = int(d.shock_L1_P90.values[i]); w9 = a90[s9]
            rows.append(dict(commodity=c, week_end=d.week_end.values[i], y=y[i], naive=y1[i], arima_fourier=fa, sarimax=fs, xgb=fx, blend50=0.5 * fs + 0.5 * fx,
                             gated=w * fs + (1 - w) * fx, gated_noGA=w * fs + (1 - w) * fx_all, gated_P90=w9 * fs + (1 - w9) * fx, shock_L1=s1, shock_L1_P90=s9,
                             sarimax_how=hs, sarimax_converged=cs, arima_how=ha, arima_converged=ca))
        R = pd.DataFrame(rows); frames.append(R)
        if "--max-weeks" not in opt: R.to_csv(pf, index=False)
        print(f"  {c}: {len(R)} weeks done ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True)

def monthly_origin(P, series, D, XB, i_first_cal, cal_end, coms, opt):
    """LEAK-FREE MONTHLY-ORIGIN EVALUATION. For each target month m (all weeks inside the evaluation window): at the origin (last week of month m-1)
    (1) re-run the Chow-Lin disaggregation using ONLY PSA months <= m-1 and weekly indicators <= origin (parameters beta, rho fixed from training months);
    (2) rebuild the price-based features from that causal weekly series; (3) refit SARIMAX / ARIMA+Fourier / XGBoost on rows up to the origin;
    (4) forecast the FIRST week of month m (one step ahead) and compare it with the official PSA average price of month m.
    Benchmark: last published PSA monthly price (month m-1).  Assumes PSA month m-1 and CPI m-1 are published by the origin."""
    prices = ing.load_prices(); prices.index = prices.index.to_period("M")
    weeks = pre.weekly_calendar(); n_tr, _ = pre.split_indices(len(weeks)); wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    X = dis.build_indicators(wk, fu, n_tr); X = X if series == "indicator" else X[["const"]]; pm = pd.Series(weeks.to_period("M"), index=weeks); trp = dis.train_periods_of(weeks, n_tr)
    orders = {c: tuple(v) for c, v in D["sarimax_orders"].items()}; fixed, prm, nt = XB["fixed"], D["xgboost"]["params"], D["xgboost"]["n_estimators"]; xf = D["xgboost"]["features"]
    a_n, a_s = D["alpha"]["normal"]["a1_sarimax"], D["alpha"]["shock"]["a1_sarimax"]; rows = []; t0 = time.time()
    months = [m for m in prices.index if (pm.values == m).any()]
    for c in coms:
        d0 = P[P.commodity == c].reset_index(drop=True); rho = dis.chow_lin(prices[c], X, pm, trp)[2]
        ps = pa = None
        for m in months:
            idx = np.where(pm.values == m)[0]; f, l = int(idx[0]), int(idx[-1])
            if f < i_first_cal or l > cal_end or f < 13: continue
            o = f - 1; y_low = prices[c][prices.index <= m - 1]
            yc = dis.chow_lin(y_low, X.iloc[:o + 1], pm.iloc[:o + 1], trp, rho=rho)[0].values            # causal weekly series, weeks 0..o
            df = pd.DataFrame({"y": np.r_[yc, np.nan]})
            for k in range(1, 9): df[f"y_L{k}"] = df.y.shift(k)
            df["y_roll4_mean"] = df.y.shift(1).rolling(4).mean(); df["y_roll4_std"] = df.y.shift(1).rolling(4).std()
            D2 = d0.iloc[:f - 10].copy(); assert pd.Timestamp(D2.week_end.iloc[-1]) == pd.Timestamp(weeks[f]); D2[Y_DERIVED] = df.iloc[11:f + 1][Y_DERIVED].values
            i = len(D2) - 1; y, y1 = D2.y.values, D2.y_L1.values; dy = y - y1
            Zr = D2[cu.SARIMAX_EXOG].values; Z = (Zr - Zr[:i].mean(0)) / Zr[:i].std(0)
            fs, ps, _, _ = sarimax_step(y, y1, Z, orders[c], i, ps); fa, pa, _, _ = sarimax_step(y, y1, D2[FOURIER].values, orders[c], i, pa)
            fx = xgb_step(D2[xf].values, dy, y1, i, fixed, prm, nt); w = a_s if int(D2.shock_L1.values[i]) == 1 else a_n
            rows.append(dict(commodity=c, month=str(m), actual_psa=float(prices.loc[m, c]), psa_naive=float(prices.loc[m - 1, c]), causal_weekly_naive=float(y1[i]), arima_fourier=fa,
                             sarimax=fs, xgb=fx, blend50=0.5 * fs + 0.5 * fx, gated=w * fs + (1 - w) * fx))
        print(f"  monthly {c}: done ({time.time()-t0:.0f}s)", flush=True)
    M = pd.DataFrame(rows); mods = ["psa_naive", "causal_weekly_naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]; out = []
    for c, g in M.groupby("commodity"):
        rn = ev.rmse(g.psa_naive - g.actual_psa)
        for mname in mods:
            e = g[mname].values - g.actual_psa.values
            out.append(dict(Commodity=c, model=mname, n_months=len(g), RMSE=ev.rmse(e), MAE=float(np.mean(np.abs(e))), MAPE=100 * float(np.mean(np.abs(e) / g.actual_psa.values)), rel_RMSE_vs_psa_naive=ev.rmse(e) / rn))
    T = pd.DataFrame(out); S = T.groupby("model").agg(n_months=("n_months", "first"), rel_RMSE_vs_psa_naive_gmean=("rel_RMSE_vs_psa_naive", lambda x: ev.gmean(x)), MAPE_mean=("MAPE", "mean")).reindex(mods).reset_index()
    return M, T, S

if __name__ == "__main__":
    series, opt = parse(); dry, confirm = "--dry-run" in opt, "--confirm-test" in opt; t_start = time.time()
    if dry == confirm:
        sys.exit("Exactly one flag is required.\n  --confirm-test : evaluate on the real TEST SET (only after the design is frozen; the design must not change afterwards)\n  --dry-run      : debugging on a stand-in test window taken from TRAINING weeks")
    limited = any(k in opt for k in ("--max-weeks", "--commodities")); pf = ("limited_" if limited else "") + ev.prefix(dry)
    D = json.load(open(OUT / f"frozen_design_{series}.json")); XB = json.load(open(OUT / f"xgb_best_{series}.json"))
    assert D["series"] == series and D["split"]["n_train_weeks"] == 357 and D["split"]["n_test_weeks"] == 90 and D["sarimax_exog"] == cu.SARIMAX_EXOG
    if dry: P = cu.load_train(series)                                              # TRAIN rows only: the test set is never read in a dry run
    else:
        P = pd.read_csv(OUT / f"features_panel_{series}.csv", parse_dates=["week_end"])
    P = P.sort_values(["commodity", "week_end"]).reset_index(drop=True); sizes = P.groupby("commodity").size(); assert sizes.nunique() == 1; n = int(sizes.iloc[0])
    assert P.groupby("commodity").week_end.apply(lambda s: tuple(s)).nunique() == 1, "commodities must share the same weeks"
    c0 = P[P.commodity == P.commodity.iloc[0]]
    if dry: i0 = n - N_PSEUDO
    else:
        i0 = int((c0.split == "train").sum()); assert i0 == 346 and n - i0 == 90 and (c0.split.values[:i0] == "train").all() and (c0.split.values[i0:] == "test").all()
    weeks = pre.weekly_calendar(); n_tr, _ = pre.split_indices(len(weeks)); wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    g95, thr_r, thr_d = gate.compute_gate(wk, fu, n_tr, 95); g90, _, _ = gate.compute_gate(wk, fu, n_tr, 90)
    assert abs(thr_r - D["gate"]["rain_P95_mm"]) < 1e-6 and abs(thr_d - D["gate"]["diesel_abs_pct_P95"]) < 1e-6, "gate thresholds differ from the frozen design"
    trig = np.where((g95.shock_rain == 1) & (g95.shock_diesel == 1), "both", np.where(g95.shock_rain == 1, "rain", np.where(g95.shock_diesel == 1, "diesel", "none")))
    G = pd.DataFrame({"week_end": g95.index, "trigger_t": trig, "shock_t": g95.I_shock.values, "shock_P90_t": g90.I_shock.values}); G["shock_L1_P90"] = G.shock_P90_t.shift(1)
    O = pd.read_csv(OUT / f"oof_predictions_{series}.csv", parse_dates=["week_end"]).merge(G[["week_end", "shock_L1_P90"]], on="week_end", how="left")
    chk = tune_alpha(O, "shock_L1"); assert chk[0] == D["alpha"]["normal"]["a1_sarimax"] and chk[1] == D["alpha"]["shock"]["a1_sarimax"], "OOF predictions do not reproduce the frozen alpha"
    a90 = tune_alpha(O, "shock_L1_P90"); print(f"[{series}] mode={'DRY-RUN (training weeks only)' if dry else 'TEST SET'} | rows/commodity {n} | evaluation weeks {n-i0} | alpha P95 {chk} | alpha P90 {a90}", flush=True)
    sig = f"wf-v2|{D['frozen_at']}|{series}|{i0}|{n}|{dry}"; part = OUT / f"{ev.prefix(dry)}wf_partial_{series}"
    F = run_weekly(P, i0, n, D, XB, a90, G, sig, part, opt).merge(G[["week_end", "trigger_t", "shock_t"]], on="week_end", how="left")
    if not limited:
        assert len(F) == 10 * (n - i0) and F[ev.ALL_MODELS].notna().all().all(), "incomplete forecasts"
        assert (F.shock_t.values == P.set_index(["commodity", "week_end"]).loc[list(zip(F.commodity, F.week_end)), "I_shock_t"].values).all()
    F.to_csv(OUT / f"{pf}wf_forecasts_{series}.csv", index=False)
    ev.per_commodity(F).round(4).to_csv(OUT / f"{pf}table_4_wf_by_commodity_{series}.csv", index=False)
    S = ev.group_summary(F); S.round(4).to_csv(OUT / f"{pf}table_4_wf_by_regime_{series}.csv", index=False)
    print("\nRelative RMSE vs naive (geometric mean over commodities; <1 beats naive) and pooled MAPE:")
    print(S[S.group.isin(["All test weeks", "Target week normal (S_t = 0)", "Target week shock (S_t = 1)"])].pivot_table(index="model", columns="group", values="rel_RMSE_vs_naive").reindex(ev.ALL_MODELS).round(3).to_string())
    print("\nSARIMAX/ARIMA fit outcomes:", F.sarimax_how.value_counts().to_dict(), F.arima_how.value_counts().to_dict(), "| non-converged:", int((~F.sarimax_converged).sum()), int((~F.arima_converged).sum()))
    log = dict(series=series, mode="dry-run" if dry else "test", frozen_at=D["frozen_at"], n_eval_weeks=n - i0, runtime_s=round(time.time() - t_start), python=platform.python_version(),
               numpy=np.__version__, pandas=pd.__version__, statsmodels=statsmodels.__version__, xgboost=xgb.__version__, sarimax_how=F.sarimax_how.value_counts().to_dict(), arima_how=F.arima_how.value_counts().to_dict())
    if "--skip-monthly" not in opt:
        print("\nLeak-free monthly-origin evaluation ...", flush=True)
        cal_start, cal_end = (n_tr - N_PSEUDO, n_tr - 1) if dry else (n_tr, len(weeks) - 1)
        coms = sorted(F.commodity.unique()); M, T, SM = monthly_origin(P, series, D, XB, cal_start, cal_end, coms, opt)
        M.round(4).to_csv(OUT / f"{pf}wf_monthly_{series}.csv", index=False); T.round(4).to_csv(OUT / f"{pf}table_4_wf_monthly_by_commodity_{series}.csv", index=False); SM.round(4).to_csv(OUT / f"{pf}table_4_wf_monthly_summary_{series}.csv", index=False)
        print(SM.round(3).to_string(index=False)); log["monthly_months"] = int(M.month.nunique())
    json.dump(log, open(OUT / f"{pf}wf_run_log_{series}.json", "w"), indent=1)
    print(f"\nDone in {time.time()-t_start:.0f}s." + ("" if dry else "\nTEST SET HAS NOW BEEN EVALUATED: do not change orders, features, hyperparameters, thresholds or weights."))
