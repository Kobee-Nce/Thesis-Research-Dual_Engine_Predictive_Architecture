"""eval_ch4_reconstruction.py -- Section 4.1 summary table and figures for the weekly price reconstruction (RQ2, RQ3).
Reads saved outputs only (02_disaggregate.py, 02b_disaggregation_backtest.py, 02c_dftc_weekly_validation.py); fits nothing.
Outputs: table_4_1_backtest_summary.csv, fig_ch4_backtest_boxplot.png, fig_ch4_dftc_within_month.png"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
OUT = import_module("00_ingest").OUT
COL = {"CL": "#2a78d6", "CLrho05": "#eda100", "CLconst": "#eb6834", "Flat": "#1baf7a"}
LAB = {"CL": "Chow–Lin\nrain + diesel\n(ρ estimated)", "CLrho05": "Chow–Lin\nrain + diesel\n(ρ = 0.5)",
       "CLconst": "Chow–Lin\nconstant only\n(ρ estimated)", "Flat": "Flat\nallocation\n(quarter mean)"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6})

B = pd.read_csv(OUT / "table_3_3_disaggregation_backtest.csv"); rows = []
for sub in ("all", "test"):
    d = B[B.subset == sub]
    for k in COL:
        rows.append(dict(subset=sub, method=k, n_months=int(d.n_months.iloc[0]), mean_RMSE=d[f"RMSE_{k}"].mean(), sd_RMSE=d[f"RMSE_{k}"].std(),
                         mean_MAPE=d[f"MAPE_{k}"].mean(), sd_MAPE=d[f"MAPE_{k}"].std(),
                         n_lower_RMSE_than_CL=int((d[f"RMSE_{k}"] < d.RMSE_CL).sum()) if k != "CL" else np.nan))
S = pd.DataFrame(rows); S.round(3).to_csv(OUT / "table_4_1_backtest_summary.csv", index=False); print(S.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for a, sub, title in [(ax[0], "all", "All 102 months (2018Q1–2026Q2)"), (ax[1], "test", "Test-period months only (20)")]:
    d = B[B.subset == sub]; data = [d[f"MAPE_{k}"].values for k in COL]
    bp = a.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="black", lw=1.5), showfliers=False)
    for p, k in zip(bp["boxes"], COL): p.set_facecolor(COL[k]); p.set_alpha(0.35); p.set_edgecolor(COL[k])
    rng = np.random.default_rng(0)
    for i, k in enumerate(COL):
        a.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d[f"MAPE_{k}"], s=22, color=COL[k], edgecolor="white", lw=0.6, zorder=3)
    a.set_xticks(range(1, 5)); a.set_xticklabels([LAB[k] for k in COL], fontsize=8); a.set_title(title, fontsize=10)
ax[0].set_ylabel("Backtest MAPE (%) per commodity"); fig.tight_layout(); fig.savefig(OUT / "fig_ch4_backtest_boxplot.png", dpi=200); plt.close(fig)

V = pd.read_csv(OUT / "table_dftc_weekly_validation.csv").sort_values("sd_dev_true"); y = np.arange(len(V))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for col, lab, c, m in [("sd_dev_true", "Observed (DFTC)", "0.25", "o"), ("sd_dev_recon_CL", "Reconstructed, rain + diesel", COL["CL"], "s"),
                       ("sd_dev_recon_CLnoind", "Reconstructed, constant only", COL["CLconst"], "D")]:
    ax[0].scatter(V[col], y, s=48, color=c, marker=m, edgecolor="white", lw=0.8, label=lab, zorder=3)
ax[0].hlines(y, V[["sd_dev_recon_CL", "sd_dev_recon_CLnoind", "sd_dev_true"]].min(axis=1), V.sd_dev_true, color="0.85", lw=1.5, zorder=1)
ax[0].set_yticks(y); ax[0].set_yticklabels(V.Commodity); ax[0].set_xlabel("SD of within-month log deviation"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("(a) Size of within-month variation", fontsize=10)
h = 0.36
ax[1].barh(y - h / 2, V.corr_dev_CL, height=h, color=COL["CL"], label="Rain + diesel")
ax[1].barh(y + h / 2, V.corr_dev_CLnoind, height=h, color=COL["CLconst"], label="Constant only")
ax[1].axvline(0, color="0.4", lw=0.8); ax[1].set_xlabel("Correlation with observed DFTC deviation"); ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("(b) Agreement in timing of within-month movement", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_ch4_dftc_within_month.png", dpi=200); plt.close(fig)
