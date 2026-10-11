"""eval_model_confidence_set.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
Hansen-Lunde-Nason Model Confidence Set (MCS), range (T_R) and semi-quadratic (T_max) statistics, moving-block bootstrap (block 4, 5000 resamples, seed 20261001).
Weekly loss = mean over commodities of the error divided by that commodity's naive loss (squared and absolute versions). Identical forecasts are merged into one model.
Models in the (1-alpha) set are those with MCS p-value >= alpha. Usage (project root):  python src/eval_model_confidence_set.py [indicator|noind] [--all]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
a = sys.argv[1:]; ser = next((x for x in a if not x.startswith("--")), "indicator"); P = Path("data/processed")
MODELS = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"] + (["gated_noGA", "gated_P90"] if "--all" in a else []); B, BLOCK, SEED = 5000, 4, 20261001
F = pd.read_csv(P / f"wf_forecasts_{ser}.csv", parse_dates=["week_end"]); wide = lambda col: F.pivot(index="week_end", columns="commodity", values=col).sort_index()
Y = wide("y"); E = {m: wide(m) - Y for m in MODELS}
def losses(kind):
    sc = (E["naive"] ** 2).mean() if kind == "squared" else E["naive"].abs().mean()
    return {m: ((E[m] ** 2 if kind == "squared" else E[m].abs()) / sc).mean(axis=1).values for m in MODELS}
def mcs(L, names):
    T, m = L.shape; rng = np.random.default_rng(SEED); nb = int(np.ceil(T / BLOCK)); idx = np.array([((rng.integers(0, T - BLOCK + 1, nb)[:, None] + np.arange(BLOCK)).ravel()[:T]) for _ in range(B)])
    mean = L.mean(0); bm = L[idx].mean(axis=1); out = {}
    for stat in ("TR", "Tmax"):
        alive = list(range(m)); running = 0.0; pv = {}
        while len(alive) > 1:
            al = np.array(alive); k = len(al); dbar = mean[al][:, None] - mean[al][None, :]; dst = bm[:, al][:, :, None] - bm[:, al][:, None, :]; eye = np.eye(k, dtype=bool)
            if stat == "TR":
                var = dst.var(axis=0, ddof=1); var[eye] = 1.0; t = dbar / np.sqrt(var); t[eye] = -np.inf; tobs = np.abs(t[~eye]).max()
                tb = (np.abs((dst - dbar) / np.sqrt(var)) * (~eye)).max(axis=(1, 2)); p = float((tb >= tobs).mean()); worst = int(np.argmax(t.max(axis=1)))
            else:
                di = dbar.sum(1) / (k - 1); dsi = dst.sum(2) / (k - 1); var = dsi.var(axis=0, ddof=1); t = di / np.sqrt(var); tobs = t.max()
                tb = ((dsi - di) / np.sqrt(var)).max(axis=1); p = float((tb >= tobs).mean()); worst = int(np.argmax(t))
            running = max(running, p); pv[names[alive[worst]]] = running; alive.pop(worst)
        pv[names[alive[0]]] = 1.0; out[stat] = pv
    return out
rows = []
for kind in ("squared", "absolute"):
    Ld = losses(kind); names, cols = [], []
    for mname in MODELS:                                   # merge identical forecasts (zero variance of the difference)
        for j, c in enumerate(cols):
            if np.allclose(Ld[mname], c, atol=1e-12): names[j] += " = " + mname; break
        else: names.append(mname); cols.append(Ld[mname])
    res = mcs(np.column_stack(cols), names)
    for n in names: rows.append(dict(loss=kind, model=n, mean_scaled_loss=float(np.mean(cols[names.index(n)])), MCS_p_TR=res["TR"][n], MCS_p_Tmax=res["Tmax"][n]))
R = pd.DataFrame(rows); R["in_90pct_set"] = (R.MCS_p_TR >= .10) & (R.MCS_p_Tmax >= .10); R["in_75pct_set"] = (R.MCS_p_TR >= .25) & (R.MCS_p_Tmax >= .25)
R.round(4).to_csv(P / f"model_confidence_set_{ser}.csv", index=False); pd.set_option("display.width", 200); print(f"[{ser}] Model Confidence Set (B={B}, block={BLOCK})")
for kind, g in R.groupby("loss", sort=False): print(f"\nloss = {kind}"); print(g.drop(columns="loss").sort_values("mean_scaled_loss").round(3).to_string(index=False))
