"""eval_ch4_rq.py -- figures and summary tables for Section 4.7 (RQ1 and RQ2). Reads saved outputs only; nothing is re-estimated.
Inputs (data/processed/): table_4_rq1.csv, gate_P95.csv, weekly_prices_disaggregated_noind.csv, prices_monthly_clean.csv,
        table_dftc_weekly_validation.csv, table_4_dftc_indicator.csv, table_4_dftc_noind.csv
Outputs: fig_ch4_rq1_coefficients.png, fig_ch4_volatility_shocks.png,
         table_4_7_volatility_by_regime.csv, table_4_7_monthly_vs_within.csv, table_4_7_dftc_forecast_summary.csv
Usage (project root):  python3 src/eval_ch4_rq.py"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = Path("data/processed"); N_TRAIN = 357
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- 1. RQ1 coefficient plot ----------------
R = pd.read_csv(P / "table_4_rq1.csv"); R = R[~R.term.str.startswith(("const", "JOINT"))]
SAMPLES = [("weekly, training weeks", "Weekly, training"), ("weekly, full sample", "Weekly, full sample"),
           ("monthly PSA, training months", "Monthly PSA, training"), ("monthly PSA, full sample", "Monthly PSA, full sample")]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.8), sharey=False)
for a, (key, title) in zip(axs, SAMPLES):
    d = R[R["sample"] == key].iloc[::-1]; y = np.arange(len(d))
    col = np.where(d.p_value < .05, np.where(d.term.str.startswith("rain"), "#2a78d6", "#eb6834"), "0.65")
    a.hlines(y, d.ci95_low, d.ci95_high, color=col, lw=2); a.scatter(d.coef, y, color=col, s=34, zorder=3, edgecolor="white", linewidth=1)
    a.axvline(0, color="0.35", ls=":", lw=1); a.set_yticks(y); a.set_yticklabels([t.replace("rain_L", "Rain, lag ").replace("diesel_L", "Diesel, lag ") for t in d.term])
    a.set_title(f"{title} (n = {int(d.n.iloc[0])})", fontsize=9, loc="left"); a.grid(axis="x", color="0.92"); a.set_axisbelow(True)
    a.set_xlabel("pp" + (" of mean |weekly log change|" if key.startswith("weekly") else " of mean |monthly log change|"), fontsize=8)
fig.text(.5, .005, "Coloured: p < 0.05 (blue rainfall, orange diesel). Grey: not significant. Bars: 95% CI with Newey-West standard errors.", ha="center", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(P / "fig_ch4_rq1_coefficients.png", dpi=200); plt.close(fig)

# ---------------- 2. Volatility series and shocks ----------------
g = pd.read_csv(P / "gate_P95.csv", parse_dates=["week_end"]).set_index("week_end")
W = pd.read_csv(P / "weekly_prices_disaggregated_noind.csv", parse_dates=["week_end"]).set_index("week_end"); v = (np.log(W).diff().abs() * 100).mean(axis=1)
M = pd.read_csv(P / "prices_monthly_clean.csv", parse_dates=["month"]).set_index("month"); vm = (np.log(M).diff().abs() * 100).mean(axis=1)
fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
a = axs[0]; a.plot(v.index, v, color="#2a78d6", lw=1.1, label="Weekly volatility $v_t$ (no-indicator series)")
for col, lab, c in [("shock_rain", "Rainfall trigger", "#1baf7a"), ("shock_diesel", "Diesel trigger", "#eb6834")]:
    t = g.index[g[col] == 1]; a.scatter(t, np.full(len(t), v.max() * (1.08 if col == "shock_rain" else 1.16)), marker="|", s=60, color=c, label=lab)
a.axvline(g.index[N_TRAIN - 1], color="0.3", ls="--", lw=1); a.text(g.index[N_TRAIN - 1], v.max() * 0.98, " test period", fontsize=8, va="top")
a.set_ylim(0, v.max() * 1.22); a.set_ylabel("Mean |weekly log change| (%)"); a.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False); a.grid(color="0.93"); a.set_axisbelow(True)
b = axs[1]; b.plot(vm.index, vm, color="#6250d6", lw=1.3, marker="o", ms=2.5, label="Monthly volatility (official PSA prices)")
b.axvline(g.index[N_TRAIN - 1], color="0.3", ls="--", lw=1); b.text(g.index[N_TRAIN - 1], vm.max() * 0.98, " test period", fontsize=8, va="top"); b.set_ylabel("Mean |monthly log change| (%)"); b.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False); b.grid(color="0.93"); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(P / "fig_ch4_volatility_shocks.png", dpi=200); plt.close(fig)

# mean weekly volatility k weeks after a trigger (descriptive companion to the regression)
D = pd.concat([v.rename("v"), g[["shock_rain", "shock_diesel"]]], axis=1).dropna(); rows = []
for lab, sub in [("training", D[D.index <= g.index[N_TRAIN - 1]]), ("full sample", D)]:
    base = sub.v[(sub.shock_rain == 0) & (sub.shock_diesel == 0)].mean()
    for trig in ["shock_rain", "shock_diesel"]:
        for k in range(4):
            m = sub.v[sub[trig].shift(k) == 1]; rows.append(dict(sample=lab, trigger=trig, lag=k, n=len(m), mean_v=m.mean(), mean_v_no_trigger_weeks=base))
T = pd.DataFrame(rows); T.round(3).to_csv(P / "table_4_7_volatility_by_regime.csv", index=False); print(T.round(3).to_string(index=False))

# ---------------- 3. RQ2: within-month variation (DFTC) against month-to-month variation (PSA) ----------------
V = pd.read_csv(P / "table_dftc_weekly_validation.csv"); V["Commodity"] = V.Commodity.str.rstrip("*"); V = V.set_index("Commodity")
lm = np.log(M).diff(); lm = lm[(lm.index >= "2024-08-01") & (lm.index <= "2026-07-31")]  # same period as the DFTC comparison (Aug 2024 - Jul 2026)
per = pd.DataFrame({"sd_monthly_log_change_PSA": lm.std(), "mean_abs_monthly_log_change_PSA": lm.abs().mean()})
per["sd_within_month_DFTC"] = V.sd_dev_true; per["sd_within_month_CL"] = V.sd_dev_recon_CL; per["sd_within_month_const"] = V.sd_dev_recon_CLnoind
per["within_over_monthly_sd"] = per.sd_within_month_DFTC / per.sd_monthly_log_change_PSA
per = per.sort_values("sd_within_month_DFTC", ascending=False); per.loc["Mean"] = per.mean()
per.round(3).to_csv(P / "table_4_7_monthly_vs_within.csv"); print(per.round(3).to_string())

# ---------------- 4. Real-data plausibility check: forecast changes vs DFTC weekly changes ----------------
out = []
for s in ("indicator", "noind"):
    F = pd.read_csv(P / f"table_4_dftc_{s}.csv"); F["exact_match"] = ~F.Commodity.str.endswith("*")
    for m, d in F.groupby("series", sort=False):
        out.append(dict(series=s, model=m, mean_corr_all10=d.corr_with_DFTC_change.mean(), mean_corr_8_exact=d[d.exact_match].corr_with_DFTC_change.mean(),
                        mean_sign_agreement=d.sign_agreement.mean(), commodities_corr_positive=int((d.corr_with_DFTC_change > 0).sum())))
O = pd.DataFrame(out); O.round(3).to_csv(P / "table_4_7_dftc_forecast_summary.csv", index=False); print(O.round(3).to_string(index=False))
