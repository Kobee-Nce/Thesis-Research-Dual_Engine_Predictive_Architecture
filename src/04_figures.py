"""04_figures.py -- Fig: monthly PSA vs reconstructed weekly series (indicator and indicator-free), train/test split marked."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT
prices = ing.load_prices(); W = pd.read_pickle(OUT / "weekly_prices_disaggregated.pkl"); W0 = pd.read_pickle(OUT / "weekly_prices_disaggregated_noind.pkl")
weeks = pre.weekly_calendar(); split = weeks[pre.split_indices(len(weeks))[0]]
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
for a, c in zip(ax, ["Cabbage", "Tomato"]):
    a.step(prices.index + pd.offsets.MonthEnd(0), prices[c], where="mid", color="grey", lw=1.2, label="PSA monthly (actual)")
    a.plot(W.index, W[c], color="tab:blue", lw=1, label="Weekly, Chow-Lin (rain + diesel)"); a.plot(W0.index, W0[c], color="tab:orange", lw=1, label="Weekly, Chow-Lin (no indicators)")
    a.axvline(split, color="red", ls="--", lw=1); a.set_ylabel(f"{c} (PHP/kg)"); a.legend(fontsize=8, loc="upper left")
ax[0].text(split, ax[0].get_ylim()[1] * 0.97, " test ->", color="red", fontsize=8, va="top")
plt.tight_layout(); plt.savefig(OUT / "fig_weekly_vs_monthly.png", dpi=160)
