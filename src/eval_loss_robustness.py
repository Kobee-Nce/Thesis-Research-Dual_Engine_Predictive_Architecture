"""eval_loss_robustness.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
Does the gated-vs-benchmark conclusion depend on the loss function?  (a) Diebold-Mariano (HLN) with squared AND absolute error: pooled over weeks
(commodity losses scaled by the naive loss) and per commodity with Holm correction.  (b) Wilcoxon signed-rank test across the 10 commodities on the
log ratio of RMSE (and of MAE) gated/benchmark (scale-free; with n = 10 the smallest possible two-sided p is about 0.002).
Usage (project root):  python src/eval_loss_robustness.py [indicator|noind]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore"); ser = sys.argv[1] if len(sys.argv) > 1 else "indicator"; P = Path("data/processed")
BENCH = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated_noGA", "gated_P90"]
F = pd.read_csv(P / f"wf_forecasts_{ser}.csv", parse_dates=["week_end"]); wide = lambda col: F.pivot(index="week_end", columns="commodity", values=col).sort_index()
Y = wide("y"); E = {m: wide(m) - Y for m in ["naive", "gated"] + BENCH}; com = list(Y.columns)
def dm(d):
    d = np.asarray(d, float); T = len(d); db = d.mean(); g0 = np.mean((d - db) ** 2)
    if g0 <= 1e-18: return np.nan, np.nan, db
    s = db / np.sqrt(g0 / T) * np.sqrt((T - 1) / T); return float(s), float(2 * (1 - stats.t.cdf(abs(s), T - 1))), float(db)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), 2, p)); adj = np.full(len(p), np.nan); run = 0
    for r, i in enumerate(o):
        if np.isnan(p[i]): continue
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1)
    return adj
rows = []
for kind in ("squared", "absolute"):
    fn = (lambda e: e ** 2) if kind == "squared" else (lambda e: e.abs()); sc = fn(E["naive"]).mean()
    for b in BENCH:
        dpool = ((fn(E["gated"]) - fn(E[b])) / sc).mean(axis=1).values; st, p, db = dm(dpool); per = [dm(fn(E["gated"][c]) - fn(E[b][c])) for c in com]; ph = holm([x[1] for x in per])
        mt = (lambda e: np.sqrt(np.mean(e ** 2))) if kind == "squared" else (lambda e: np.mean(np.abs(e)))
        lr = np.array([np.log(mt(E["gated"][c].values) / mt(E[b][c].values)) for c in com])
        try: w = stats.wilcoxon(lr, zero_method="wilcox", alternative="two-sided"); pw = float(w.pvalue)
        except ValueError: pw = np.nan
        rows.append(dict(loss=kind, benchmark=b, pooled_mean_loss_diff=db, pooled_DM_HLN=st, pooled_p=p, commodities_gated_lower=int((lr < 0).sum()), commodities_sig_holm=int(np.nansum(ph < .05)),
                         Wilcoxon_p_log_ratio=pw, median_log_ratio=float(np.median(lr))))
R = pd.DataFrame(rows); R.round(4).to_csv(P / f"loss_robustness_{ser}.csv", index=False); pd.set_option("display.width", 220)
print(f"[{ser}] gated vs benchmark; negative difference / log ratio = gated better"); print(R.round(4).to_string(index=False))
