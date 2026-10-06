"""07_sarimax_order.py -- SARIMAX order selection (Section 3.7). TRAIN weeks only.
d comes from 03_stationarity (ADF/KPSS). (p,q) in {0..4}x{0..3} chosen by AIC (grid widened after a first pass hit the upper bound). Annual seasonality = Fourier terms (K=2), not s=52.
Exogenous set (fixed, z-scored with training statistics): Fourier x4, rain_sum_L1, diesel_pct_L1, cpi_veg_mom_L1m.
Usage: python 07_sarimax_order.py [indicator|noind]"""
import sys, warnings, itertools, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from importlib import import_module
cu = import_module("cv_utils"); OUT = cu.OUT; warnings.filterwarnings("ignore")

def fit_sarimax(y, X, order):
    d = order[1]; trend = "c" if d == 0 else "n"
    return SARIMAX(y, exog=X, order=order, trend=trend).fit(disp=False, maxiter=200, method="lbfgs")

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"
    P = cu.load_train(series); st = pd.read_csv(OUT / "table_3_4_stationarity.csv"); st = st[st.series == series].set_index("Commodity")
    grid_rows, best_rows = [], []
    for c in P.commodity.unique():
        d = P[P.commodity == c]; y = d.y.values; X = d[cu.SARIMAX_EXOG]; X = ((X - X.mean()) / X.std()).values
        dd = int(st.loc[c, "suggested_d"]); res = []
        for p, q in itertools.product(range(5), range(4)):
            try:
                m = fit_sarimax(y, X, (p, dd, q)); ok = bool(m.mle_retvals.get("converged", True))
                lb = acorr_ljungbox(m.resid[dd:], lags=[15], model_df=p + q, return_df=True).lb_pvalue.iloc[0]
                res.append(dict(series=series, Commodity=c, p=p, d=dd, q=q, AIC=m.aic, BIC=m.bic, LB_p=lb, converged=ok))
            except Exception as e:
                res.append(dict(series=series, Commodity=c, p=p, d=dd, q=q, AIC=np.nan, BIC=np.nan, LB_p=np.nan, converged=False))
        R = pd.DataFrame(res).sort_values("AIC").reset_index(drop=True); grid_rows.append(R)
        b = R.iloc[0].copy(); b["dAIC_next"] = R.AIC.iloc[1] - R.AIC.iloc[0]; best_rows.append(b)
        print(f"{c:14s} d={dd} best (p,q)=({int(b.p)},{int(b.q)})  AIC={b.AIC:.1f}  dAIC_next={b.dAIC_next:.2f}  LB_p={b.LB_p:.3f}", flush=True)
    pd.concat(grid_rows).round(3).to_csv(OUT / f"table_3_7a_sarimax_grid_{series}.csv", index=False)
    B = pd.DataFrame(best_rows).round(3); B.to_csv(OUT / f"table_3_7a_sarimax_order_{series}.csv", index=False)
    print("Boundary hits (p=4 or q=3):", int(((B.p == 4) | (B.q == 3)).sum()), "of", len(B))
    print("\nOrders:", {r.Commodity: (int(r.p), int(r.d), int(r.q)) for r in B.itertuples()})
