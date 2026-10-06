"""eval_utils.py -- shared helpers for 11_walkforward.py and 12_tests.py: metric definitions, regime groups, DFTC weekly extraction.
Metric conventions (Section 3.8): RMSE and MAPE are computed ONCE over all forecast steps of a group (never averaged per step).
RMSE is not scale-free across commodities, so cross-commodity summaries use the geometric mean of RMSE(model)/RMSE(naive)."""
import numpy as np, pandas as pd
from importlib import import_module
ing = import_module("00_ingest"); OUT = ing.OUT

MODELS = ["naive", "arima_fourier", "sarimax", "xgb", "blend50", "gated"]                  # B1-B5 + the proposed gated ensemble (6 models)
ABLATIONS = ["gated_noGA", "gated_P90"]                                                      # A1 and A3 (A2 = the other weekly series, run separately)
ALL_MODELS = MODELS + ABLATIONS
LABEL = {"naive": "B1 Naive (last week)", "arima_fourier": "B2 ARIMA + Fourier", "sarimax": "B3 SARIMAX (no gate)", "xgb": "B4 XGBoost alone",
         "blend50": "B5 Fixed 50/50", "gated": "Gated ensemble (proposed)", "gated_noGA": "A1 Gated, no GA", "gated_P90": "A3 Gated, P90 gate"}

def prefix(dry): return "dryrun_" if dry else ""
def rmse(e): return float(np.sqrt(np.mean(np.square(np.asarray(e, float)))))
def gmean(x): x = np.asarray(x, float); return float(np.exp(np.mean(np.log(x))))

def per_commodity(F, models=ALL_MODELS):
    rows = []
    for c, g in F.groupby("commodity"):
        for m in models:
            e = g[m].values - g.y.values
            rows.append(dict(Commodity=c, model=m, n=len(g), RMSE=rmse(e), MAE=float(np.mean(np.abs(e))), MAPE=100 * float(np.mean(np.abs(e) / g.y.values))))
    return pd.DataFrame(rows)

def group_masks(F):
    t, p = F.shock_t.values == 1, F.shock_L1.values == 1
    trig = F.trigger_t.values
    return [("All test weeks", np.ones(len(F), bool)), ("Target week normal (S_t = 0)", ~t), ("Target week shock (S_t = 1)", t),
            ("Gate state normal (S_t-1 = 0)", ~p), ("Gate state shock (S_t-1 = 1)", p),
            ("Rain-only shock weeks", trig == "rain"), ("Diesel-only shock weeks", trig == "diesel"), ("Both triggers", trig == "both"), ("Neither trigger", trig == "none")]

def group_summary(F, models=ALL_MODELS):
    """Per group and model: geometric mean over commodities of RMSE/RMSE(naive), pooled MAPE, number of weeks and commodity-week observations."""
    rows = []
    for gname, mask in group_masks(F):
        G = F[mask]
        if len(G) == 0: continue
        nw = G.week_end.nunique()
        rn = {c: rmse(g.naive.values - g.y.values) for c, g in G.groupby("commodity")}
        for m in models:
            rel = [rmse(g[m].values - g.y.values) / rn[c] for c, g in G.groupby("commodity") if rn[c] > 0]
            rows.append(dict(group=gname, model=m, n_weeks=nw, n_obs=len(G), rel_RMSE_vs_naive=gmean(rel), MAPE=100 * float(np.mean(np.abs(G[m].values - G.y.values) / G.y.values))))
    return pd.DataFrame(rows)

# ---- DFTC weekly prices (real weekly ground truth; identical cleaning rules to 02c_dftc_weekly_validation.py) ----
DFTC_MAP = {"ampalaya(galaxy)": "Ampalaya", "kalabasa(suprema)": "Squash", "kamatis(diamante big)": "Tomato", "patola": "Patola", "talong(banate king)": "Eggplant",
            "upo(mayumi)": "Upo", "carrots(big)": "Carrot", "repolyo(wakamini)": "Cabbage", "sibuyas(native)": "Red onion", "ahos(imported)": "Native garlic"}
DFTC_IMPERFECT = {"Red onion", "Native garlic"}   # DFTC variety differs from the PSA series

def dftc_weekly(path, start="2023-10-02", end="2026-07-26"):
    d = pd.read_excel(path, sheet_name="Format B"); d = d[d.Price_Type == "retail"].copy()
    d["Date"] = pd.to_datetime(d.Date, errors="coerce"); d["p"] = pd.to_numeric(d.Price_PHP, errors="coerce")
    d = d[(d.Date >= start) & (d.Date <= end) & d.p.notna()]; out = {}
    for lab, c in DFTC_MAP.items():
        s = d[d.Commodity == lab].drop_duplicates("Date", keep="first").set_index("Date").p.sort_index()
        s = s[(s > 0) & (s < 3 * s.median())]; s = s[s > 0.35 * s.median()]
        wk = s.resample("W-SUN").agg(["mean", "count"]); out[c] = wk[wk["count"] >= 4]["mean"]
    return out
