"""eval_ch4_ga.py -- Section 4.2 figures for the Genetic Algorithm feature search (RQ4).
Reads saved outputs of 08_ga.py only (training weeks); fits nothing.
Outputs: fig_ch4_ga_convergence.png, fig_ch4_ga_selection_heatmap.png, fig_ch4_ga_comparison.png, table_4_2_ga_comparison.csv"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from importlib import import_module
OUT = import_module("00_ingest").OUT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
SER = {"indicator": "Indicator series", "noind": "No-indicator series"}
BASES = ["y", "rain_sum", "rain_max", "rain_roll4", "wind_max", "t_mean", "t_max", "t_min", "diesel_pct"]
NAMES = {"y": "Weekly price", "rain_sum": "Total weekly rainfall", "rain_max": "Max daily rainfall", "rain_roll4": "Four-week rainfall",
         "wind_max": "Max daily wind speed", "t_mean": "Mean temperature", "t_max": "Max temperature", "t_min": "Min temperature",
         "diesel_pct": "Diesel price change"}

# ---- convergence: best CV-RMSE (= 1 / best fitness) by generation, one line per seed ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for a, s in zip(ax, SER):
    G = pd.read_csv(OUT / f"ga_convergence_{s}.csv"); naive = pd.read_csv(OUT / f"table_3_5f_ga_summary_{s}.csv").naive_cv_rmse.iloc[0]
    for seed, g in G.groupby("seed"):
        a.plot(g.generation + 1, 1 / g.best_fitness, color="#2a78d6", lw=1.2, alpha=0.55)
        a.scatter(g.generation.iloc[-1] + 1, 1 / g.best_fitness.iloc[-1], s=14, color="#2a78d6", zorder=3)
    a.axhline(naive, color="0.45", ls=":", lw=1.2, label=f"Naive (zero change) = {naive:.3f}")
    a.axvline(20, color="#e34948", ls="--", lw=0.9, label="Generation limit (20)")
    a.set_title(SER[s], fontsize=10); a.set_xlabel("Generation"); a.xaxis.set_major_locator(MaxNLocator(integer=True)); a.legend(fontsize=8, loc="upper right"); a.grid(color="0.92")
ax[0].set_ylabel("Best $\\mathrm{RMSE}_{CV}$ (standardized)"); fig.tight_layout(); fig.savefig(OUT / "fig_ch4_ga_convergence.png", dpi=200); plt.close(fig)

# ---- selection-frequency heatmap, consensus genes outlined ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for a, s in zip(ax, SER):
    F = pd.read_csv(OUT / f"table_3_5c_ga_selection_frequency_{s}.csv")
    M = F.pivot(index="base", columns="lag", values="frequency").loc[BASES]; C = F.pivot(index="base", columns="lag", values="in_consensus").loc[BASES]
    im = a.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]; a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else "black")
            if C.values[i, j]: a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e34948", lw=2))
    a.set_xticks(range(8)); a.set_xticklabels(range(1, 9)); a.set_xlabel("Lag (weeks)"); a.set_title(SER[s], fontsize=10)
    a.set_yticks(range(len(BASES))); a.set_yticklabels([NAMES[b] for b in BASES], fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Share of the 10 GA runs selecting the gene")
fig.savefig(OUT / "fig_ch4_ga_selection_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- GA vs alternatives: selection folds (relative to naive) and hold-out fold 4 (relative to naive) ----
rows = []
for s in SER:
    D = pd.read_csv(OUT / f"table_3_5d_ga_vs_baselines_{s}.csv")
    H = pd.read_csv(OUT / f"table_3_5e_ga_holdout_fold_{s}.csv", index_col=0).iloc[:, 0]; hn = H.filter(like="Naive").iloc[0]
    for _, r in D.iterrows(): rows.append(dict(series=s, check="Selection folds 1-4", method=r.method, n_lags=r.n_lag_features, rmse=r.cv_rmse_std, rel_to_naive=r.rel_to_naive))
    for m, v in H.items():
        if "Naive" not in m: rows.append(dict(series=s, check="Hold-out fold 4", method=m, n_lags=np.nan, rmse=v, rel_to_naive=v / hn))
T = pd.DataFrame(rows); T.round(4).to_csv(OUT / "table_4_2_ga_comparison.csv", index=False); print(T.round(3).to_string(index=False))
short = lambda m: (m.replace("(>=50% of seeds)", "").replace("(within-fold)", "").replace("(folds 1-3 only)", "(folds 1–3)").replace(" by |corr|", "").strip())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, s in zip(ax, SER):
    d = T[(T.series == s) & (T.check == "Selection folds 1-4")].iloc[::-1]; h = T[(T.series == s) & (T.check == "Hold-out fold 4")].iloc[::-1]
    y = np.arange(len(d)); a.scatter(d.rel_to_naive, y, s=46, color="#2a78d6", label="Selection folds 1–4 (GA chose on these)", zorder=3)
    a.set_yticks(y); a.set_yticklabels([short(m) for m in d.method], fontsize=8.5)
    yh = np.arange(len(h)) + len(d) + 0.8; a.scatter(h.rel_to_naive, yh, s=46, color="#eb6834", marker="D", label="Hold-out fold 4 (unseen by GA)", zorder=3)
    a.set_yticks(list(y) + list(yh)); a.set_yticklabels([short(m) for m in d.method] + [short(m) for m in h.method], fontsize=8.5)
    a.axvline(1, color="0.45", ls=":", lw=1); a.grid(axis="x", color="0.92"); a.set_title(SER[s], fontsize=10)
    a.set_xlabel("RMSE relative to the naive (zero-change) forecast")
h_, l_ = ax[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False); fig.savefig(OUT / "fig_ch4_ga_comparison.png", dpi=200); plt.close(fig)
