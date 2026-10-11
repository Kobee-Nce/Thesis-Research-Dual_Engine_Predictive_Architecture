"""eval_error_diagnostics.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
One-step forecast errors e = y - forecast (positive = under-forecast). Per model and commodity:
  mean error and a Newey-West (Bartlett, 3 lags) t-test of E(e) = 0 (Holm over the 10 commodities per model);
  Ljung-Box Q on e (10 lags);  ARCH-LM test on e (4 lags) for changing error variance.
Usage (project root):  python src/eval_error_diagnostics.py [indicator|noind]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ser = sys.argv[1] if len(sys.argv) > 1 else "indicator"; P = Path("data/processed"); MODELS = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]; L_NW, LB_LAGS, ARCH_Q = 3, 10, 4
F = pd.read_csv(P / f"wf_forecasts_{ser}.csv", parse_dates=["week_end"]).sort_values(["commodity", "week_end"])
def nw_t(e):
    n = len(e); m = e.mean(); u = e - m; v = np.sum(u * u) / n
    for l in range(1, L_NW + 1): v += 2 * (1 - l / (L_NW + 1)) * np.sum(u[l:] * u[:-l]) / n
    t = m / np.sqrt(v / n) if v > 0 else np.nan; return m, t, (2 * (1 - stats.t.cdf(abs(t), n - 1)) if np.isfinite(t) else np.nan)
def ljung_box(e, h=LB_LAGS):
    n = len(e); u = e - e.mean(); den = np.sum(u * u)
    if den <= 1e-18: return np.nan, np.nan
    r = np.array([np.sum(u[k:] * u[:-k]) / den for k in range(1, h + 1)]); q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1))); return q, float(1 - stats.chi2.cdf(q, h))
def arch_lm(e, q=ARCH_Q):
    y = (e - e.mean()) ** 2
    if y.std() < 1e-12: return np.nan, np.nan
    X = np.column_stack([np.ones(len(y) - q)] + [y[q - k:len(y) - k] for k in range(1, q + 1)]); yy = y[q:]; b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r2 = 1 - np.sum((yy - X @ b) ** 2) / np.sum((yy - yy.mean()) ** 2); s = (len(yy)) * r2; return float(s), float(1 - stats.chi2.cdf(s, q))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), 2, p)); adj = np.full(len(p), np.nan); run = 0
    for r, i in enumerate(o):
        if np.isnan(p[i]): continue
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1)
    return adj
rows = []
for m in MODELS:
    for c, g in F.groupby("commodity"):
        e = (g.y - g[m]).values; me, t, pb = nw_t(e); q, plb = ljung_box(e); s, pa = arch_lm(e)
        rows.append(dict(model=m, Commodity=c, n=len(e), mean_error=me, mean_error_pct_of_price=100 * me / g.y.mean(), NW_t=t, p_bias=pb, LB_Q=q, p_LB=plb, ARCH_LM=s, p_ARCH=pa))
D = pd.DataFrame(rows); D["p_bias_holm"] = D.groupby("model").p_bias.transform(lambda x: pd.Series(holm(x.values), index=x.index)); D.round(4).to_csv(P / f"error_diagnostics_{ser}_by_commodity.csv", index=False)
S = D.groupby("model", sort=False).agg(mean_error_PHP=("mean_error", "mean"), mean_error_pct=("mean_error_pct_of_price", "mean"), bias_sig_raw=("p_bias", lambda x: int((x < .05).sum())), bias_sig_holm=("p_bias_holm", lambda x: int((x < .05).sum())),
                                       LB_reject_5pct=("p_LB", lambda x: int((x < .05).sum())), ARCH_reject_5pct=("p_ARCH", lambda x: int((x < .05).sum()))).reindex(MODELS)
S.round(4).to_csv(P / f"error_diagnostics_{ser}_summary.csv"); pd.set_option("display.width", 200); print(f"[{ser}] error diagnostics (counts are out of 10 commodities; e = actual - forecast)"); print(S.round(3).to_string())
