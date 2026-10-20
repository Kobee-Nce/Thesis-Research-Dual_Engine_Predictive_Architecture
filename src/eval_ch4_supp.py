"""eval_ch4_supp.py -- figures for Section 4.8 (Supplementary Analyses). Reads saved outputs only; nothing is re-estimated.
Inputs (data/processed/): temporal_split_results.csv, loo_shock_episode_indicator.csv, error_diagnostics_indicator_by_commodity.csv
Outputs: fig_ch4_temporal_split.png, fig_ch4_loo_episodes.png, table_4_8_errdiag_gated_sarimax.csv
Usage (project root):  python3 src/eval_ch4_supp.py"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = Path("data/processed"); plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"arima_fourier": "#eda100", "sarimax": "#1baf7a", "xgb": "#eb6834", "blend50": "#6250d6", "gated": "#2a78d6"}
NAME = {"arima_fourier": "B2 ARIMA + Fourier", "sarimax": "B3 SARIMAX", "xgb": "B4 XGBoost", "blend50": "B5 Fixed 50/50", "gated": "Gated ensemble"}

# ---- 1. temporal split ----
T = pd.read_csv(P / "temporal_split_results.csv"); per = list(dict.fromkeys(T.Period)); M = list(COL)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, col, ttl in zip(axs, ["rel_RMSE_geomean", "rel_RMSE_pooled"], ["Geometric-mean RelRMSE", "Pooled RMSE ratio (PHP/kg)"]):
    x = np.arange(len(per)); w = 0.16
    for i, m in enumerate(M):
        v = [T[(T.Period == p) & (T.Model == m)][col].iloc[0] for p in per]
        a.bar(x + (i - 2) * w, v, w * 0.92, color=COL[m], label=NAME[m])
    lab = [f"{p}\n({int(T[T.Period == p].weeks.iloc[0])} weeks, {int(T[T.Period == p].shock_gate_weeks.iloc[0])} shock-gate)" for p in per]
    a.set_xticks(x); a.set_xticklabels(lab); a.axhline(1, color="0.4", ls=":", lw=1); a.set_title(ttl, fontsize=9, loc="left")
    a.grid(axis="y", color="0.92"); a.set_axisbelow(True)
axs[0].set_ylabel("RMSE relative to naive"); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(P / "fig_ch4_temporal_split.png", dpi=200); plt.close(fig)

# ---- 2. leave-one-shock-episode-out ----
L = pd.read_csv(P / "loo_shock_episode_indicator.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, pan, ttl in zip(axs, ["A: all test weeks", "B: shock-state weeks only"], ["All test weeks", "Shock-state weeks only"]):
    d = L[L.panel == pan]; full = d[d.omitted == "none"].ratio_gated_over_sarimax.iloc[0]; d = d[d.omitted != "none"]
    y = np.arange(len(d))[::-1]; c = np.where(d.p_value < .10, "#e34948", "#2a78d6")
    a.scatter(d.ratio_gated_over_sarimax, y, color=c, s=34, zorder=3, edgecolor="white", linewidth=1)
    a.axvline(full, color="0.3", lw=1, label=f"no episode omitted ({full:.3f})"); a.axvline(1, color="0.4", ls=":", lw=1)
    a.set_yticks(y); a.set_yticklabels([f"{o.replace('episode', 'Ep.')}  {str(dt)[:10]} ({k} wk)" for o, dt, k in zip(d.omitted, d.episode_dates, 90 - d.weeks_kept if pan.startswith("A") else 33 - d.weeks_kept)], fontsize=7.5)
    a.set_title(ttl, fontsize=9, loc="left"); a.set_xlabel("RMSE ratio, gated / SARIMAX, with the episode omitted"); a.legend(fontsize=8, frameon=False, loc="lower right")
    a.grid(axis="x", color="0.92"); a.set_axisbelow(True)
fig.text(.5, .005, "Red: pooled Diebold-Mariano p < 0.10 after omission. Ratio < 1 = gated ensemble more accurate.", ha="center", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(P / "fig_ch4_loo_episodes.png", dpi=200); plt.close(fig)

# ---- 3. error diagnostics: per-commodity summary for gated and SARIMAX ----
E = pd.read_csv(P / "error_diagnostics_indicator_by_commodity.csv")
S = E[E.model.isin(["naive", "sarimax", "blend50", "gated"])].pivot(index="Commodity", columns="model", values=["mean_error_pct_of_price", "p_LB", "p_ARCH"])
S.round(4).to_csv(P / "table_4_8_errdiag_gated_sarimax.csv"); print(S.round(3).to_string())
print("Ljung-Box rejects (p<0.05):", {m: int((E[E.model == m].p_LB < .05).sum()) for m in E.model.unique()})
