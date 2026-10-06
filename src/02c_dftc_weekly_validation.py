"""02c_dftc_weekly_validation.py -- supplementary validation of the reconstructed weekly series against real
DAILY DFTC Bankerohan retail prices (Aug 2024 - Jul 2026, the fully covered period).
Levels differ (DFTC ~0.6-0.7 x PSA), so we compare scale-free WITHIN-MONTH log deviations and weekly log changes."""
import numpy as np, pandas as pd, sys
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT
DFTC = sys.argv[1] if len(sys.argv) > 1 else str(ing.ROOT / "data" / "raw" / "Data-Scrape-Final.xlsx")   # default: data/raw/
MAP = {"ampalaya(galaxy)": "Ampalaya", "kalabasa(suprema)": "Squash", "kamatis(diamante big)": "Tomato", "patola": "Patola",
       "talong(banate king)": "Eggplant", "upo(mayumi)": "Upo", "carrots(big)": "Carrot", "repolyo(wakamini)": "Cabbage",
       "sibuyas(native)": "Red onion*", "ahos(imported)": "Native garlic*"}   # * imperfect PSA match (different variety)

d = pd.read_excel(DFTC, sheet_name="Format B"); d = d[d.Price_Type == "retail"].copy()
d["Date"] = pd.to_datetime(d.Date, errors="coerce"); d["p"] = pd.to_numeric(d.Price_PHP, errors="coerce")
d = d[(d.Date >= "2024-08-05") & (d.Date <= "2026-07-26") & d.p.notna()]
weeks = pre.weekly_calendar(); out = []
for lab, c in MAP.items():
    s = d[d.Commodity == lab].drop_duplicates("Date", keep="first").set_index("Date").p.sort_index()
    s = s[(s > 0) & (s < 3 * s.median())]; s = s[s > 0.35 * s.median()]            # drop zero/misplaced-zero typos
    wk = s.resample("W-SUN").agg(["mean", "count"]); wk = wk[wk["count"] >= 4]["mean"]
    row = {"Commodity": c, "n_weeks": len(wk)}
    for tag, f in [("CL", "weekly_prices_disaggregated.pkl"), ("CLnoind", "weekly_prices_disaggregated_noind.pkl")]:
        W = pd.read_pickle(OUT / f)[c.rstrip("*")].reindex(wk.index)
        lw, lr = np.log(wk), np.log(W); pm = wk.index.to_period("M")
        dev_w = lw - lw.groupby(pm).transform("mean"); dev_r = lr - lr.groupby(pm).transform("mean")
        row[f"corr_dev_{tag}"] = dev_w.corr(dev_r); row[f"corr_dchg_{tag}"] = lw.diff().corr(lr.diff())
        row[f"sd_dev_recon_{tag}"] = dev_r.std(); row["sd_dev_true"] = dev_w.std()
    out.append(row)
T = pd.DataFrame(out).round(2); T.to_csv(OUT / "table_dftc_weekly_validation.csv", index=False); print(T.to_string(index=False))
