"""04b_shock_timeline.py -- shock-week timeline figure for Section 3.6 (weekly rainfall, weekly diesel change, I_shock).
Reads the pipeline's own outputs (weather_weekly.pkl, fuel_weekly_clean.pkl, fuel_filled_flag.pkl) so figure and tables always agree.
Usage: python 04b_shock_timeline.py   -> data/processed/fig_shock_timeline.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); gate = import_module("05_gate"); OUT = ing.OUT

weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks))
wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl"); filled = pd.read_pickle(OUT / "fuel_filled_flag.pkl")
g, thr_r, thr_d = gate.compute_gate(wk, fu, n_train, 95); dchg = fu.pct_change() * 100
x = g.index; split = x[n_train]; r_on, d_on = g.shock_rain.values == 1, g.shock_diesel.values == 1
fig, ax = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.1]})
for a in ax: a.axvspan(split, x[-1] + pd.Timedelta(days=7), color="#e8e8e8", alpha=.7, lw=0); a.grid(alpha=.25, lw=.5)
ax[0].bar(x, g.rain_sum, width=6, color=np.where(r_on, "#1f5fa8", "#9db7d5")); ax[0].axhline(thr_r, color="#b22222", ls="--", lw=1.2, label=f"Training P95 = {thr_r:.1f} mm")
ax[0].set_ylabel("Weekly rainfall (mm)"); ax[0].legend(loc="upper center", fontsize=8)
ax[1].bar(x, dchg, width=6, color=np.where(d_on, "#d2691e", "#e5c3a5")); ax[1].axhline(thr_d, color="#b22222", ls="--", lw=1.2, label=f"\u00b1Training P95 = {thr_d:.2f}%"); ax[1].axhline(-thr_d, color="#b22222", ls="--", lw=1.2)
ax[1].scatter(x[filled.values], np.zeros(int(filled.sum())), marker="|", s=40, color="k", lw=.8, label="Gap-filled diesel week"); ax[1].set_ylabel("Weekly diesel change (%)"); ax[1].legend(loc="upper center", fontsize=8, ncol=2)
sd = x[g.I_shock.values == 1]; cols = ["#6a3d9a" if (r_on[x.get_loc(d)] and d_on[x.get_loc(d)]) else "#1f5fa8" if r_on[x.get_loc(d)] else "#d2691e" for d in sd]
ax[2].vlines(sd, 0, 1, colors=cols, lw=2); ax[2].set_yticks([]); ax[2].set_ylabel("$I_{shock}$")
ax[2].legend(handles=[Line2D([0], [0], color=c, lw=3, label=l) for c, l in (("#1f5fa8", "Rainfall"), ("#d2691e", "Diesel"), ("#6a3d9a", "Both"))], loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.25), frameon=False)
ax[0].text(x[6], ax[0].get_ylim()[1] * 0.9, f"Training set ({n_train} weeks)", fontsize=9); ax[0].text(split + pd.Timedelta(days=15), ax[0].get_ylim()[1] * 0.9, f"Test set ({len(x) - n_train} weeks)", fontsize=9)
ax[2].set_xlim(x[0], x[-1] + pd.Timedelta(days=7)); fig.tight_layout(); fig.savefig(OUT / "fig_shock_timeline.png", dpi=200); plt.close(fig)
print(f"Saved fig_shock_timeline.png | rain P95 {thr_r:.1f} mm, diesel P95 {thr_d:.2f}% | shock weeks: total {int(g.I_shock.sum())}, train {int(g.I_shock[:n_train].sum())}, test {int(g.I_shock[n_train:].sum())}")
