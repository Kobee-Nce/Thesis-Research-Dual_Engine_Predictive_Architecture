"""04c_eda.py -- exploratory data analysis for Section 3.3.1 and the diesel-gap figure of Section 3.4.
Reads only saved outputs of scripts 01-05 (no model is fitted). Correlations use TRAINING weeks only.
Outputs (data/processed/):
  table_3_3_descriptive.csv        Table 6  descriptive statistics of the weekly series (full sample)
  table_3_3_eda_correlation.csv    Table 7  correlation of each candidate series with the standardized weekly price change
  fig_eda_weekly_prices.png        Figure 7 weekly series by commodity, split marked
  fig_eda_boxplot.png              Figure 8 distribution by commodity, ordered by median
  fig_eda_correlation.png          Figure 9 correlation matrix (lag 1, training weeks)
  fig_eda_gate_inputs.png          Figure 10 training distributions of the two gate inputs with P90/P95 lines
  fig_diesel_gaps.png              Figure 13 filled diesel weeks (training: interpolated; test: carried forward)"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT

weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks)); split = weeks[n_train]
W = pd.read_pickle(OUT / "weekly_prices_disaggregated.pkl")
I = pd.read_csv(OUT / "weekly_indicators.csv", parse_dates=["week_end"], index_col="week_end")
P = pd.read_csv(OUT / "features_panel_indicator.csv", parse_dates=["week_end"])
G = pd.read_csv(OUT / "gate_P95.csv", parse_dates=["week_end"], index_col="week_end")

# ---- Table 6: descriptive statistics (full sample, reconstructed weekly series) ----
D = pd.DataFrame({"Mean": W.mean(), "SD": W.std(), "Min": W.min(), "Max": W.max()}); D["CV"] = D.SD / D.Mean
D.round(3).rename_axis("Commodity").to_csv(OUT / "table_3_3_descriptive.csv"); print(D.round(3).to_string())

# ---- Table 7: correlation with the standardized weekly price change (training weeks only) ----
SER = {"y": "Weekly retail price (standardized level)", "rain_sum": "Total weekly rainfall", "rain_max": "Maximum daily rainfall",
       "wind_max": "Maximum daily wind speed", "t_mean": "Mean temperature", "t_max": "Maximum temperature",
       "t_min": "Minimum temperature", "rain_roll4": "Four-week rainfall total", "diesel_pct": "Weekly diesel price change"}
T = P[P.split == "train"].copy()
T["dy"] = T.y - T.y_L1
sd_dy = T.groupby("commodity").dy.transform("std"); T["target"] = T.dy / sd_dy          # within-commodity scale
for k in range(1, 9):                                                                     # price level standardized within commodity
    g = T.groupby("commodity")[f"y_L{k}"]; T[f"y_L{k}"] = (T[f"y_L{k}"] - g.transform("mean")) / g.transform("std")
rows = []
for s, lab in SER.items():
    r = {k: T[f"{s}_L{k}"].corr(T.target) for k in range(1, 9)}
    kbest = max(r, key=lambda k: abs(r[k]))
    rows.append({"Series": lab, "r_lag1": r[1], "max_abs_r": abs(r[kbest]), "lag_of_max": kbest, "r_at_max": r[kbest]})
rows.append({"Series": "Vegetable CPI, m/m change (previous month)", "r_lag1": T.cpi_veg_mom_L1m.corr(T.target),
             "max_abs_r": abs(T.cpi_veg_mom_L1m.corr(T.target)), "lag_of_max": "1 month", "r_at_max": T.cpi_veg_mom_L1m.corr(T.target)})
C = pd.DataFrame(rows).sort_values("max_abs_r", ascending=False).reset_index(drop=True); C.index += 1
C.round(3).rename_axis("Rank").to_csv(OUT / "table_3_3_eda_correlation.csv"); print(C.round(3).to_string())
print("Training commodity-weeks used:", len(T))

# ---- Figure 7: weekly series by commodity ----
fig, ax = plt.subplots(5, 2, figsize=(11, 11), sharex=True)
for a, c in zip(ax.ravel(), W.columns):
    a.plot(W.index, W[c], lw=0.9, color="tab:blue"); a.axvline(split, color="red", ls="--", lw=0.8)
    a.set_title(c, fontsize=9); a.set_ylabel("PHP/kg", fontsize=8); a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_weekly_prices.png", dpi=160); plt.close(fig)

# ---- Figure 8: boxplot ordered by median ----
order = W.median().sort_values().index
fig, a = plt.subplots(figsize=(10, 4.5)); a.boxplot([W[c] for c in order], tick_labels=list(order), showfliers=True)
a.set_ylabel("PHP/kg"); a.tick_params(axis="x", rotation=30, labelsize=8); fig.tight_layout()
fig.savefig(OUT / "fig_eda_boxplot.png", dpi=160); plt.close(fig)

# ---- Figure 9: correlation matrix at lag 1 (training weeks) ----
cols = [f"{s}_L1" for s in SER] + ["cpi_veg_mom_L1m", "target"]
names = ["Price (std.)", "Rain total", "Rain max", "Wind max", "T mean", "T max", "T min", "Rain 4-wk", "Diesel %", "CPI m/m", "Next-week\nprice change"]
M = T[cols].corr().values
fig, a = plt.subplots(figsize=(8, 7)); im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=60, ha="right", fontsize=8); a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)): a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(OUT / "fig_eda_correlation.png", dpi=160); plt.close(fig)

# ---- Figure 10: training distributions of the gate inputs ----
tr = G.iloc[:n_train]; d = tr.diesel_abs_pct.iloc[1:]           # first week has no defined change
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, x, lab in [(ax[0], tr.rain_sum, "Weekly total rainfall (mm)"), (ax[1], d, "Absolute weekly diesel price change (%)")]:
    a.hist(x, bins=40, color="lightsteelblue", edgecolor="grey")
    for p, ls in [(90, ":"), (95, "--")]:
        v = np.percentile(x, p); a.axvline(v, color="red", ls=ls, lw=1.2, label=f"P{p} = {v:.2f}")
    a.set_xlabel(lab); a.set_ylabel("Training weeks"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_gate_inputs.png", dpi=160); plt.close(fig)
print("P95 rain %.2f mm, P95 diesel %.3f%%; P90 rain %.2f mm, P90 diesel %.3f%%" % (np.percentile(tr.rain_sum, 95), np.percentile(d, 95), np.percentile(tr.rain_sum, 90), np.percentile(d, 90)))

# ---- Figure 13: filled diesel weeks ----
f = I.diesel_filled.astype(bool); trn = np.arange(len(f)) < n_train
# two-panel design adapted from the team's 04b_figure_3_6_v2.py: price series with filled weeks marked + filled weeks per year
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [2.4, 1]})
a.plot(I.index, I.diesel, lw=1.1, color="#3C3489", label="Weekly diesel price (cleaned)")
a.scatter(I.index[f.values & trn], I.diesel[f.values & trn], color="tab:orange", s=18, zorder=5, label=f"Training gap, linear interpolation ({int((f.values & trn).sum())})")
a.scatter(I.index[f.values & ~trn], I.diesel[f.values & ~trn], color="tab:purple", marker="s", s=18, zorder=5, label=f"Test gap, last value carried forward ({int((f.values & ~trn).sum())})")
a.axvline(split, color="red", ls="--", lw=0.8); a.set_ylabel("PHP/liter"); a.legend(fontsize=8, loc="upper left")
per_year = f.groupby(f.index.year).sum()
bars = b.bar(per_year.index.astype(str), per_year.values, color="0.6"); b.set_ylim(0, per_year.max() * 1.35)
for r, v in zip(bars, per_year.values):
    if v > 0: b.text(r.get_x() + r.get_width() / 2, v + 0.3, str(int(v)), ha="center", fontsize=8)
b.set_ylabel("Filled\nweeks"); b.set_xlabel("Year"); fig.tight_layout()
fig.savefig(OUT / "fig_diesel_gaps.png", dpi=160); plt.close(fig)
print("Filled diesel weeks by year:", f.groupby(f.index.year).sum().to_dict())
