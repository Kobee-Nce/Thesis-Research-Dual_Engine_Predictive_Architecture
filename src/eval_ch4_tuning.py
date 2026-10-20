"""eval_ch4_tuning.py -- Section 4.3 figures for model training and hyperparameter optimisation (training folds only).
Reads saved outputs only (09_xgb_tuning.py, 10_gate_weights.py, eval_learning_curves.py); fits nothing.
Outputs: fig_ch4_learning_curves.png, fig_ch4_tuned_vs_reference.png, fig_ch4_alpha_loss.png"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
OUT = import_module("00_ingest").OUT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
SER = {"indicator": "Indicator series", "noind": "No-indicator series"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# ---- learning curves: training / early-stopping hold-out / validation loss by number of trees ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, s in zip(ax, SER):
    L = pd.read_csv(OUT / f"table_4_learning_curve_{s}.csv", index_col=0); nt = json.load(open(OUT / f"xgb_best_{s}.json"))["n_estimators"]
    a.plot(L.index, L.train, color=BLUE, lw=2, label="Training")
    a.plot(L.index, L.es, color=AQUA, lw=2, ls="--", label="Early-stopping hold-out")
    a.plot(L.index, L.val, color=ORANGE, lw=2, label="Validation (36-week folds)")
    a.axhline(L.naive_val.iloc[0], color="0.45", ls=":", lw=1.2, label="Naive, validation")
    a.axvline(nt, color="#e34948", ls="--", lw=0.9, label=f"Frozen model ({nt} trees)")
    k = int(L.val.idxmin()); a.scatter(k, L.val.min(), s=40, color=ORANGE, zorder=4)
    a.text(0.97, 0.55, f"Validation minimum {L.val.min():.3f} at {k} trees\nAt {nt} trees: training {L.train[nt]:.3f}, validation {L.val[nt]:.3f}\nAt 300 trees: training {L.train[300]:.3f}, validation {L.val[300]:.3f}",
           transform=a.transAxes, ha="right", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    a.set_title(SER[s], fontsize=10); a.set_xlabel("Number of trees (boosting rounds)"); a.grid(color="0.92")
ax[0].set_ylabel("Scaled RMSE of weekly price change"); h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
fig.savefig(OUT / "fig_ch4_learning_curves.png", dpi=200); plt.close(fig)

# ---- per-commodity validation score, tuned vs reference configuration ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for a, s in zip(ax, SER):
    C = pd.read_csv(OUT / f"table_3_7c_xgb_cv_by_commodity_{s}.csv")
    bp = a.boxplot([C.default, C.tuned], widths=0.45, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.5))
    for p, c in zip(bp["boxes"], [YELLOW, BLUE]): p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor(c)
    for _, r in C.iterrows(): a.plot([1, 2], [r.default, r.tuned], color="0.75", lw=0.8, zorder=1)
    a.scatter(np.ones(len(C)), C.default, s=26, color=YELLOW, edgecolor="white", zorder=3); a.scatter(2 * np.ones(len(C)), C.tuned, s=26, color=BLUE, edgecolor="white", zorder=3)
    for _, r in C.iterrows():
        if r.tuned < 0.85 or r.tuned > 1.05: a.text(2.2, r.tuned, r.Commodity, fontsize=7.5, va="center")
    a.axhline(1, color="0.45", ls=":", lw=1); a.set_xticks([1, 2]); a.set_xticklabels(["Reference\nconfiguration", "Tuned\n(config 56)"])
    a.set_title(SER[s], fontsize=10); a.grid(axis="y", color="0.92"); a.set_xlim(0.6, 2.6)
ax[0].set_ylabel("Validation RMSE relative to naive, per commodity"); fig.tight_layout(); fig.savefig(OUT / "fig_ch4_tuned_vs_reference.png", dpi=200); plt.close(fig)

# ---- ensemble-weight loss over the grid, by regime ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for a, s in zip(ax, SER):
    A = pd.read_csv(OUT / f"alpha_loss_curve_{s}.csv"); W = pd.read_csv(OUT / f"table_3_8_alpha_weights_{s}.csv").set_index("regime")
    for reg, c, m, lab in [("normal", BLUE, "o", "Normal regime ($S_{t-1}=0$)"), ("shock", ORANGE, "s", "Shock regime ($S_{t-1}=1$)")]:
        d = A[A.regime == reg]; a.plot(d.alpha_sarimax, d.rel_rmse, color=c, lw=2, marker=m, ms=6, label=lab)
        b = W.loc[reg, "alpha1_sarimax"]; a.scatter(b, d.set_index("alpha_sarimax").rel_rmse[b], s=140, facecolor="none", edgecolor=c, lw=1.8, zorder=4)
        a.axhline(W.loc[reg, "rel_rmse_naive"], color=c, ls=":", lw=1)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xlabel("SARIMAX weight $\\alpha_1$ (XGBoost weight $1-\\alpha_1$)")
    a.set_title(SER[s], fontsize=10); a.grid(color="0.92")
ax[0].set_ylabel("Scaled RMSE on validation folds"); h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
fig.savefig(OUT / "fig_ch4_alpha_loss.png", dpi=200); plt.close(fig)
print("saved fig_ch4_learning_curves.png, fig_ch4_tuned_vs_reference.png, fig_ch4_alpha_loss.png")
