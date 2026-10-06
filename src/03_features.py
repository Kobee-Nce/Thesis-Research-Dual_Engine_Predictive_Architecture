"""03_features.py -- leakage-safe feature panel. Every predictor in row t uses information from week t-1 or earlier
(or is a deterministic calendar term). Usage: python 03_features.py [indicator|noind]"""
import sys, numpy as np, pandas as pd
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); gate = import_module("05_gate"); OUT = ing.OUT
MAXLAG = 8
DYN = ["rain_sum", "rain_max", "wind_max", "t_mean", "t_max", "t_min", "rain_roll4", "diesel_pct"]   # 8 weekly drivers x lags 1..8

def build_panel(series="indicator"):
    weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks))
    wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    W = pd.read_pickle(OUT / ("weekly_prices_disaggregated.pkl" if series == "indicator" else "weekly_prices_disaggregated_noind.pkl"))
    g, _, _ = gate.compute_gate(wk, fu, n_train)
    base = wk[["rain_sum", "rain_max", "wind_max", "t_mean", "t_max", "t_min"]].copy()
    base["rain_roll4"] = wk.rain_sum.rolling(4).sum(); base["diesel_pct"] = fu.pct_change() * 100
    shared = pd.DataFrame(index=weeks)
    for f in DYN:
        for k in range(1, MAXLAG + 1): shared[f"{f}_L{k}"] = base[f].shift(k)
    cpi = ing.load_cpi(); cpi.index = cpi.index.to_period('M'); cmom = cpi.pct_change() * 100                       # monthly CPI (vegetables) m/m %, never disaggregated
    per = weeks.to_period("M"); shared["cpi_veg_mom_L1m"] = [cmom.get(p - 1, np.nan) for p in per]   # previous completed month only
    doy = weeks.dayofyear.values
    for k in (1, 2): shared[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25); shared[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    shared["shock_L1"] = g.I_shock.shift(1); shared["I_shock_t"] = g.I_shock.values   # I_shock_t is for REGIME REPORTING only, never a predictor
    shared["split"] = g.split.values
    frames = []
    for c in W.columns:
        y = W[c]; d = shared.copy(); d.insert(0, "y", y); d.insert(0, "commodity", c)
        for k in range(1, MAXLAG + 1): d[f"y_L{k}"] = y.shift(k)
        d["y_roll4_mean"] = y.shift(1).rolling(4).mean(); d["y_roll4_std"] = y.shift(1).rolling(4).std()
        frames.append(d.iloc[MAXLAG:].dropna())                             # first 8 weeks lost to the longest lag
    P = pd.concat(frames).rename_axis("week_end").reset_index()
    return P

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"
    P = build_panel(series); P.to_csv(OUT / f"features_panel_{series}.csv", index=False)
    cols = [c for c in P.columns if c not in ("week_end", "commodity", "y", "split", "I_shock_t")]
    print(series, "| rows", len(P), "| model features", len(cols), "| NaN", int(P.isna().sum().sum()))
    print(P.groupby("split").size().to_dict(), "| rows per commodity:", P.groupby("commodity").size().unique())
    print("Shock weeks (I_shock_t) by split, per commodity:", P[P.commodity == "Cabbage"].groupby("split").I_shock_t.sum().to_dict())
