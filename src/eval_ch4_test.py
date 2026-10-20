"""eval_ch4_test.py -- Section 4.4 figures and summary table for the test-set evaluation (reads saved walk-forward outputs only).
Outputs: table_4_4_test_summary_<series>.csv, fig_ch4_test_index.png, fig_ch4_test_by_commodity.png,
         fig_ch4_test_by_regime.png, fig_ch4_cv_folds.png, fig_ch4_monthly_origin.png
Requires table_4_4_cv_baselines.csv (eval_cv_baselines.py)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
OUT = import_module("00_ingest").OUT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
MODELS = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]
NAME = {"naive": "B1 Naive", "arima_fourier": "B2 ARIMA + Fourier", "sarimax": "B3 SARIMAX", "xgb": "B4 XGBoost", "blend50": "B5 Fixed 50/50",
        "gated": "Gated ensemble", "gated_noGA": "A1 Gated, no GA", "gated_P90": "A3 Gated, P90"}
COL = {"naive": "0.55", "arima_fourier": "#eda100", "sarimax": "#1baf7a", "xgb": "#eb6834", "blend50": "#6250d6", "gated": "#2a78d6"}
MK = {"naive": "o", "arima_fourier": "v", "sarimax": "s", "xgb": "^", "blend50": "D", "gated": "o"}

# ---- summary table (pooled metrics, geometric-mean relative RMSE, commodities won) ----
for s in ("indicator", "noind"):
    F = pd.read_csv(OUT / f"wf_forecasts_{s}.csv"); C = pd.read_csv(OUT / f"table_4_wf_by_commodity_{s}.csv")
    nv = C[C.model == "naive"].set_index("Commodity").RMSE; C["rel"] = C.RMSE / C.Commodity.map(nv)
    T = C.pivot(index="Commodity", columns="model", values="rel"); won = T[MODELS].idxmin(axis=1).value_counts()
    rows = []
    for m in MODELS + ["gated_noGA", "gated_P90"]:
        e = F[m] - F.y
        rows.append(dict(model=m, RMSE=np.sqrt((e ** 2).mean()), MAE=e.abs().mean(), MAPE=(e.abs() / F.y).mean() * 100,
                         rel_RMSE_gmean=float(np.exp(np.log(T[m]).mean())), commodities_best_of_B1_B5_gated=int(won.get(m, 0)) if m in MODELS else np.nan))
    pd.DataFrame(rows).round(4).to_csv(OUT / f"table_4_4_test_summary_{s}.csv", index=False)
    T.round(4).to_csv(OUT / f"table_4_4_test_rel_by_commodity_{s}.csv")

# ---- average price index over the test weeks (indicator series) ----
F = pd.read_csv(OUT / "wf_forecasts_indicator.csv", parse_dates=["week_end"]); base = F.sort_values("week_end").groupby("commodity").y.first()
I = pd.DataFrame({m: (F[m] / F.commodity.map(base) * 100).groupby(F.week_end).mean() for m in ["y", "naive", "sarimax", "xgb", "gated"]})
fig, a = plt.subplots(figsize=(11, 4.2))
a.plot(I.index, I.y, color="black", lw=2.2, label="Actual (reconstructed weekly price)")
for m, ls in [("naive", ":"), ("sarimax", "-"), ("xgb", "-"), ("gated", "-")]: a.plot(I.index, I[m], color=COL[m], lw=1.3, ls=ls, label=NAME[m])
a.set_ylabel("Average price index (first test week = 100)"); a.legend(fontsize=8, ncol=3, loc="upper left"); a.grid(color="0.92")
fig.tight_layout(); fig.savefig(OUT / "fig_ch4_test_index.png", dpi=200); plt.close(fig)

# ---- relative RMSE by commodity ----
T = pd.read_csv(OUT / "table_4_4_test_rel_by_commodity_indicator.csv", index_col=0).sort_values("gated")
fig, a = plt.subplots(figsize=(9, 5.2)); y = np.arange(len(T))
for k, m in enumerate(MODELS[1:]):
    a.scatter(T[m], y + (k - 2) * 0.12, s=36, color=COL[m], marker=MK[m], edgecolor="white", lw=0.5, label=NAME[m], zorder=3)
a.axvline(1, color="0.45", ls=":", lw=1); a.set_yticks(y); a.set_yticklabels(T.index); a.grid(axis="x", color="0.92")
a.set_xlabel("Test-set RMSE relative to the naive forecast (B1 = 1)"); a.legend(fontsize=8, loc="lower right"); fig.tight_layout()
fig.savefig(OUT / "fig_ch4_test_by_commodity.png", dpi=200); plt.close(fig)

# ---- by regime ----
R = pd.read_csv(OUT / "table_4_wf_by_regime_indicator.csv")
groups = [("All test weeks", "All test\nweeks (90)"), ("Target week normal (S_t = 0)", "Normal\ntarget (57)"), ("Target week shock (S_t = 1)", "Shock\ntarget (33)"),
          ("Rain-only shock weeks", "Rain-only\nshock (9)"), ("Diesel-only shock weeks", "Diesel-only\nshock (22)")]
fig, a = plt.subplots(figsize=(11, 4.4)); w = 0.15
for k, m in enumerate(MODELS[1:]):
    v = [R[(R.group == g) & (R.model == m)].rel_RMSE_vs_naive.iloc[0] for g, _ in groups]
    a.bar(np.arange(len(groups)) + (k - 2) * w, v, width=w * 0.92, color=COL[m], label=NAME[m])
a.axhline(1, color="0.45", ls=":", lw=1); a.set_xticks(range(len(groups))); a.set_xticklabels([l for _, l in groups], fontsize=9)
a.set_ylabel("Relative RMSE (geometric mean, naive = 1)"); a.legend(fontsize=8, ncol=5, loc="upper left"); a.set_ylim(0, 1.15); a.grid(axis="y", color="0.92")
fig.tight_layout(); fig.savefig(OUT / "fig_ch4_test_by_regime.png", dpi=200); plt.close(fig)

# ---- validation folds ----
V = pd.read_csv(OUT / "table_4_4_cv_baselines.csv"); V = V[V.series == "indicator"].set_index("model")
fig, a = plt.subplots(figsize=(7.5, 4))
for m in ["naive", "sarimax", "xgb", "blend50", "gated"]:
    a.plot(range(1, 5), V.loc[m, ["fold1", "fold2", "fold3", "fold4"]].values, color=COL[m], marker=MK[m], lw=1.8, label=f"{NAME[m]} ({V.loc[m,'mean_gmean']:.3f} ± {V.loc[m,'sd_gmean']:.3f})")
a.set_xticks(range(1, 5)); a.set_xticklabels(["Fold 1\nFeb–Oct 2022", "Fold 2\nOct 2022–Jun 2023", "Fold 3\nJun 2023–Feb 2024", "Fold 4\nMar–Nov 2024"], fontsize=8)
a.set_ylabel("Relative RMSE (geometric mean)"); a.grid(color="0.92"); h, l = a.get_legend_handles_labels()
fig.set_size_inches(8, 5); fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.savefig(OUT / "fig_ch4_cv_folds.png", dpi=200); plt.close(fig)

# ---- monthly-origin evaluation against official PSA prices ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, s in zip(ax, ("indicator", "noind")):
    M = pd.read_csv(OUT / f"table_4_wf_monthly_summary_{s}.csv").set_index("model")
    order = ["causal_weekly_naive"] + MODELS[1:]; v = M.loc[order, "rel_RMSE_vs_psa_naive_gmean"]
    a.bar(range(len(order)), v, color=["0.55"] + [COL[m] for m in MODELS[1:]]); a.axhline(1, color="black", ls="--", lw=1)
    for i, x in enumerate(v): a.text(i, x + 0.005, f"{x:.3f}", ha="center", fontsize=7.5)
    a.set_xticks(range(len(order))); a.set_xticklabels(["Causal weekly\nnaive", "B2", "B3", "B4", "B5", "Gated"], fontsize=8.5)
    a.set_ylim(0.9, 1.2); a.set_title({"indicator": "Indicator series", "noind": "No-indicator series (A2)"}[s], fontsize=10); a.grid(axis="y", color="0.92")
ax[0].set_ylabel("RMSE relative to last PSA price"); fig.tight_layout(); fig.savefig(OUT / "fig_ch4_monthly_origin.png", dpi=200); plt.close(fig)
print(pd.read_csv(OUT / "table_4_4_test_summary_indicator.csv").to_string(index=False)); print(pd.read_csv(OUT / "table_4_4_test_summary_noind.csv").to_string(index=False))
