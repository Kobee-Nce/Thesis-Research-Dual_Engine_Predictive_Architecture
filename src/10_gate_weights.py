"""10_gate_weights.py -- gate weights alpha (Section 3.6) and the FROZEN DESIGN file (Gate G3). TRAIN weeks only.
Out-of-fold one-step forecasts from SARIMAX (order from 07) and XGBoost (features from 08, params from 09) on the same rolling-origin folds.
Blend = a1*SARIMAX + a2*XGBoost, a1+a2=1, a1 on a 0..1 grid (grid {0, 0.25, 0.5, 0.75, 1}), chosen SEPARATELY for the normal and shock regime (regime = I_shock at t-1).
Loss = squared error divided by each commodity's naive RMSE (scale-free), pooled. Usage: python 10_gate_weights.py [indicator|noind]"""
import sys, json, time, warnings, numpy as np, pandas as pd, xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from importlib import import_module
cu = import_module("cv_utils"); gate = import_module("05_gate"); pre = import_module("01_preprocess"); OUT = cu.OUT; warnings.filterwarnings("ignore")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # coarse, pre-specified grid (Section 3.6.5)

def sarimax_oof(y, X, order, vs, ve):
    mu, sd = X[:vs].mean(0), X[:vs].std(0); Z = (X - mu) / sd
    m = SARIMAX(y[:vs], exog=Z[:vs], order=order, trend="c" if order[1] == 0 else "n").fit(disp=False, maxiter=200, method="lbfgs")
    full = m.apply(y[:ve], exog=Z[:ve]); return np.asarray(full.get_prediction(start=vs, end=ve - 1, dynamic=False).predicted_mean)

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"; t0 = time.time()
    P = cu.load_train(series); xb = json.load(open(OUT / f"xgb_best_{series}.json")); ga = json.load(open(OUT / f"ga_selected_features_{series}.json"))
    so = pd.read_csv(OUT / f"table_3_7a_sarimax_order_{series}.csv").set_index("Commodity"); n = (P.commodity == P.commodity.iloc[0]).sum(); folds = cu.rolling_folds(n); rows = []
    for c in P.commodity.unique():
        d = P[P.commodity == c].reset_index(drop=True); y, y1 = d.y.values, d.y_L1.values; order = (int(so.loc[c, "p"]), int(so.loc[c, "d"]), int(so.loc[c, "q"]))
        Xs, Xg = d[cu.SARIMAX_EXOG].values, d[xb["features"]].values
        for k, (vs, ve) in enumerate(folds):
            try: ps = sarimax_oof(y, Xs, order, vs, ve)
            except Exception: ps = y1[vs:ve]
            m = xgb.XGBRegressor(**{**xb["fixed"], **xb["params"]}, n_estimators=xb["n_estimators"]).fit(Xg[:vs], (y - y1)[:vs]); px = y1[vs:ve] + m.predict(Xg[vs:ve])
            rows.append(pd.DataFrame(dict(commodity=c, fold=k + 1, week_end=d.week_end[vs:ve].values, y=y[vs:ve], naive=y1[vs:ve], sarimax=ps, xgb=px, shock_L1=d.shock_L1[vs:ve].values)))
        print(f"  {c} done ({time.time()-t0:.0f}s)", flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / f"oof_predictions_{series}.csv", index=False)
    scale = O.groupby("commodity").apply(lambda g: np.sqrt(np.mean((g.naive - g.y) ** 2)), include_groups=False); O["s"] = O.commodity.map(scale)
    res, curve, alpha = [], [], {}
    for reg, lab in [(0, "normal"), (1, "shock")]:
        g = O[O.shock_L1 == reg]; loss = {a: np.mean(((a * g.sarimax + (1 - a) * g.xgb - g.y) / g.s) ** 2) for a in ALPHAS}
        best = min(ALPHAS, key=lambda a: (round(loss[a], 10), abs(a - 0.5))); alpha[lab] = float(best)
        for a in ALPHAS: curve.append(dict(regime=lab, alpha_sarimax=a, rel_rmse=np.sqrt(loss[a])))
        f = lambda col: np.sqrt(np.mean(((g[col] - g.y) / g.s) ** 2))
        res.append(dict(regime=lab, n_obs=len(g), n_distinct_weeks=g.week_end.nunique(), alpha1_sarimax=best, alpha2_xgb=round(1 - best, 2), rel_rmse_blend=np.sqrt(loss[best]),
                        rel_rmse_sarimax=np.sqrt(loss[1.0]), rel_rmse_xgb=np.sqrt(loss[0.0]), rel_rmse_5050=np.sqrt(loss[0.5]), rel_rmse_naive=f("naive")))
    T = pd.DataFrame(res).round(4); T.to_csv(OUT / f"table_3_8_alpha_weights_{series}.csv", index=False); pd.DataFrame(curve).round(4).to_csv(OUT / f"alpha_loss_curve_{series}.csv", index=False)
    weeks = pre.weekly_calendar(); n_tr, _ = pre.split_indices(len(weeks)); wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    _, thr_r, thr_d = gate.compute_gate(wk, fu, n_tr, 95)
    json.dump(dict(frozen_at=pd.Timestamp.now().isoformat(timespec="seconds"), series=series, split=dict(n_train_weeks=n_tr, n_test_weeks=len(weeks) - n_tr),
                   gate=dict(rain_P95_mm=float(thr_r), diesel_abs_pct_P95=float(thr_d)), alpha=dict(shock=dict(a1_sarimax=alpha["shock"], a2_xgb=round(1 - alpha["shock"], 2)), normal=dict(a1_sarimax=alpha["normal"], a2_xgb=round(1 - alpha["normal"], 2))),
                   sarimax_orders={c: [int(so.loc[c, "p"]), int(so.loc[c, "d"]), int(so.loc[c, "q"])] for c in so.index}, sarimax_exog=cu.SARIMAX_EXOG,
                   xgboost=dict(params=xb["params"], n_estimators=xb["n_estimators"], features=xb["features"]), ga_selected=ga["selected_lag_features"],
                   note="GATE G3: design frozen. Do not change after the test set is evaluated."), open(OUT / f"frozen_design_{series}.json", "w"), indent=1)
    print(T.to_string(index=False)); print("alpha:", alpha, f"| {time.time()-t0:.0f}s")
