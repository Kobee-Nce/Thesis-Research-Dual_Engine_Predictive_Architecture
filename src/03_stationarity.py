"""03_stationarity.py -- ADF / KPSS on TRAIN weekly series (levels and first differences) -> SARIMAX d."""
import pandas as pd, numpy as np, warnings
from statsmodels.tsa.stattools import adfuller, kpss
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT; warnings.filterwarnings("ignore")
if __name__ == "__main__":
    n_train, _ = pre.split_indices(len(pre.weekly_calendar())); rows = []
    for tag, f in [("indicator", "weekly_prices_disaggregated.pkl"), ("noind", "weekly_prices_disaggregated_noind.pkl")]:
        tr = pd.read_pickle(OUT / f).iloc[:n_train]
        for c in tr.columns:
            r = {"series": tag, "Commodity": c}
            for lab, s in [("lvl", tr[c]), ("diff", tr[c].diff().dropna())]:
                a = adfuller(s, autolag="AIC"); k = kpss(s, regression="c", nlags="auto"); r[f"ADF_stat_{lab}"], r[f"ADF_p_{lab}"] = a[0], a[1]; r[f"KPSS_stat_{lab}"], r[f"KPSS_p_{lab}"] = k[0], k[1]
            r["suggested_d"] = 0 if (r["ADF_p_lvl"] < .05 and r["KPSS_p_lvl"] > .05) else 1
            rows.append(r)
    T = pd.DataFrame(rows).round(3); T.to_csv(OUT / "table_3_4_stationarity.csv", index=False); print(T.to_string(index=False))
