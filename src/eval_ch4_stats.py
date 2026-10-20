"""eval_ch4_stats.py -- figures and summary tables for Section 4.5 (Statistical Validation).
Reads the saved test-set outputs only (no refitting): table_4_dm_indicator.csv, table_4_friedman_indicator.csv,
table_4_nemenyi_indicator.csv, table_4_bootstrap_indicator.csv and wf_forecasts_indicator.csv.
Outputs (data/processed/): fig_ch4_dm_heatmap.png, fig_ch4_cd_diagram.png, fig_ch4_bootstrap_ci.png,
table_4_5_dm_summary.csv, table_4_5_abs_dm_per_commodity.csv
Usage (project root):  python3 src/eval_ch4_stats.py"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

P = Path("data/processed")
NAME = {"naive": "B1 Naive", "arima_fourier": "B2 ARIMA+Fourier", "sarimax": "B3 SARIMAX", "xgb": "B4 XGBoost",
        "blend50": "B5 Fixed 50/50", "gated_noGA": "A1 No GA", "gated_P90": "A3 P90 gate", "gated": "Gated ensemble"}
BENCH = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated_noGA", "gated_P90"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- 1. Diebold-Mariano heat map (indicator series) ----------------
D = pd.read_csv(P / "table_4_dm_indicator.csv")
com = [c for c in D.scope.unique() if not c.startswith("POOLED")]
S = D.pivot(index="benchmark", columns="scope", values="DM_HLN").loc[BENCH, com + ["POOLED (weeks)"]]
H = D.pivot(index="benchmark", columns="scope", values="p_holm_over_commodities").loc[BENCH, com]
pp = D[D.scope.str.startswith("POOLED")].set_index("benchmark").loc[BENCH, "p_value"]
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
fig, a = plt.subplots(figsize=(10, 3.9)); lim = 6
a.imshow(S.clip(-lim, lim).values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i, b in enumerate(BENCH):
    for j, c in enumerate(S.columns):
        v = S.loc[b, c]; sig = (H.loc[b, c] < .05) if c in com else (pp[b] < .05)
        a.text(j, i, f"{v:.1f}" + ("*" if sig else ""), ha="center", va="center", fontsize=8,
               color="white" if abs(v) > 4 else "#222", fontweight="bold" if sig else "normal")
a.set_xticks(range(len(S.columns))); a.set_xticklabels([c if c in com else "Pooled" for c in S.columns], rotation=35, ha="right")
a.set_yticks(range(len(BENCH))); a.set_yticklabels([NAME[b] for b in BENCH]); a.axvline(len(com) - .5, color="white", lw=3)
for s in a.spines.values(): s.set_visible(False)
a.tick_params(length=0); a.set_title("Gated ensemble vs benchmark: DM statistic (HLN). Negative (blue) = gated ensemble more accurate", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(P / "fig_ch4_dm_heatmap.png", dpi=200); plt.close(fig)

# summary table: pooled DM, commodities where gated is lower, commodities significant after Holm
rows = []
for b in BENCH:
    d = D[D.benchmark == b]; po = d[d.scope.str.startswith("POOLED")].iloc[0]; pc = d[~d.scope.str.startswith("POOLED")]
    rows.append(dict(benchmark=b, pooled_DM_HLN=po.DM_HLN, pooled_p=po.p_value, commodities_gated_lower=int(pc.gated_better.sum()),
                     sig_holm_gated_better=int(((pc.p_holm_over_commodities < .05) & pc.gated_better).sum()),
                     sig_holm_gated_worse=int(((pc.p_holm_over_commodities < .05) & ~pc.gated_better).sum())))
T = pd.DataFrame(rows); T.round(4).to_csv(P / "table_4_5_dm_summary.csv", index=False); print(T.round(3).to_string(index=False))

# ---------------- 2. Critical-difference diagram (Friedman / Nemenyi) ----------------
F = pd.read_csv(P / "table_4_friedman_indicator.csv").sort_values("avg_rank"); cd = F.Nemenyi_CD_0_05.iloc[0] if "Nemenyi_CD_0_05" in F else F["Nemenyi_CD_0.05"].iloc[0]
k = len(F); fig, a = plt.subplots(figsize=(8, 2.9)); a.set_xlim(.6, k + .4); a.set_ylim(-.15, 1.25); a.axis("off")
a.hlines(1, 1, k, color="#222", lw=1)
for r in range(1, k + 1): a.vlines(r, .97, 1.03, color="#222", lw=1); a.text(r, 1.08, str(r), ha="center", fontsize=9)
a.text((k + 1) / 2, 1.2, "Average rank across the ten commodities (1 = lowest RMSE)", ha="center", fontsize=8.5, color="#555")
a.plot([1, 1 + cd], [.05, .05], color="#222", lw=2); a.vlines([1, 1 + cd], .02, .08, color="#222", lw=1.5)
a.text(1 + cd / 2, -.05, f"CD = {cd:.3f}", ha="center", va="top", fontsize=8.5)
left, right = F.iloc[: (k + 1) // 2], F.iloc[(k + 1) // 2:]
for i, (_, r) in enumerate(left.iterrows()):
    y = .8 - i * .16; a.plot([r.avg_rank, r.avg_rank, .75], [1, y, y], color="#444", lw=.9)
    a.text(.73, y, f"{NAME[r.model]} ({r.avg_rank:.1f})", ha="right", va="center", fontsize=8.5)
for i, (_, r) in enumerate(right.iloc[::-1].iterrows()):
    y = .8 - i * .16; a.plot([r.avg_rank, r.avg_rank, k + .25], [1, y, y], color="#444", lw=.9)
    a.text(k + .27, y, f"({r.avg_rank:.1f}) {NAME[r.model]}", ha="left", va="center", fontsize=8.5)
# cliques: maximal groups of consecutive models whose rank range is below CD
rk = F.avg_rank.values; cl = []
for i in range(k):
    j = max(j for j in range(i, k) if rk[j] - rk[i] < cd)
    if j > i and not any(s <= i and j <= e for s, e in cl): cl.append((i, j))
for n, (i, j) in enumerate(cl):
    y = .9 - n * .07; a.plot([rk[i] - .04, rk[j] + .04], [y, y], color="#2a78d6", lw=3.5, solid_capstyle="round")
fig.subplots_adjust(left=.2, right=.8); fig.savefig(P / "fig_ch4_cd_diagram.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("cliques (models joined, not significantly different):", [[F.model.iloc[x] for x in range(i, j + 1)] for i, j in cl])

# ---------------- 3. Bootstrap confidence intervals (forest plot) ----------------
B = pd.read_csv(P / "table_4_bootstrap_indicator.csv")
SUB = ["All weeks", "Normal target week", "Shock target week", "Gate state normal", "Gate state shock", "Rain-only weeks", "Diesel-only weeks"]
PAN = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated_noGA"]
fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
for a, b in zip(axs.flat, PAN):
    d = B[B.benchmark == b].set_index("subset").loc[SUB]; y = np.arange(len(SUB))[::-1]
    col = np.where(d.ci95_high < 1, "#2a78d6", np.where(d.ci95_low > 1, "#e34948", "0.6"))
    a.hlines(y, d.ci95_low, d.ci95_high, color=col, lw=2)
    a.scatter(d.ratio_gated_over_benchmark, y, s=36, color=col, zorder=3, edgecolor="white", linewidth=1)
    a.axvline(1, color="0.35", ls=":", lw=1); a.set_title(f"vs {NAME[b]}", fontsize=9, loc="left"); a.grid(axis="x", color="0.92")
    a.set_yticks(y); a.set_yticklabels([f"{s} ({int(n)})" for s, n in zip(SUB, d.n_weeks)])
for a in axs[1]: a.set_xlabel("RMSE ratio, gated / benchmark (95% CI)")
fig.text(.5, .005, "Blue: gated ensemble significantly more accurate (interval below 1). Red: significantly less accurate (interval above 1). Grey: interval includes 1.", ha="center", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(P / "fig_ch4_bootstrap_ci.png", dpi=200); plt.close(fig)

# ---------------- 4. Absolute-loss DM per commodity, gated vs SARIMAX (for the text) ----------------
W = pd.read_csv(P / "wf_forecasts_indicator.csv"); w = lambda c: W.pivot(index="week_end", columns="commodity", values=c).sort_index()
Y = w("y"); r = []
for b in ["sarimax", "blend50"]:
    for c in Y.columns:
        dd = ((w("gated") - Y)[c].abs() - (w(b) - Y)[c].abs()).values; n = len(dd)
        st = dd.mean() / np.sqrt(np.mean((dd - dd.mean()) ** 2) / n) * np.sqrt((n - 1) / n); r.append(dict(benchmark=b, commodity=c, DM_HLN=st, p=2 * (1 - stats.t.cdf(abs(st), n - 1))))
A = pd.DataFrame(r)
A["p_holm"] = A.groupby("benchmark").p.transform(lambda p: np.minimum(1, np.maximum.accumulate(np.sort(p.values) * np.arange(len(p), 0, -1))[np.argsort(np.argsort(p.values))]))
A.round(4).to_csv(P / "table_4_5_abs_dm_per_commodity.csv", index=False); print(A[A.p_holm < .05].round(4).to_string(index=False))
