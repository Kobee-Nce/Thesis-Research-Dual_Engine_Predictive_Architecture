"""05_gate.py -- regime gate I_shock = rain >= train P95  OR  |weekly diesel % change| >= train P95 (thresholds from TRAIN only).
The model uses I_shock(t-1); evaluation splits use I_shock(t)."""
import numpy as np, pandas as pd
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT

def compute_gate(wk, fuel, n_train, pct=95):
    dpct = (fuel.pct_change() * 100).abs()   # first week has no defined change (NaN), excluded from the percentile
    tr_r, tr_d = np.percentile(wk.rain_sum.iloc[:n_train], pct), np.nanpercentile(dpct.iloc[:n_train], pct)
    g = pd.DataFrame({"rain_sum": wk.rain_sum, "diesel_abs_pct": dpct.fillna(0)}); g["shock_rain"] = (g.rain_sum >= tr_r).astype(int)
    g["shock_diesel"] = (g.diesel_abs_pct >= tr_d).astype(int); g["I_shock"] = ((g.shock_rain + g.shock_diesel) > 0).astype(int)
    g["split"] = ["train"] * n_train + ["test"] * (len(g) - n_train)
    return g, tr_r, tr_d

if __name__ == "__main__":
    weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks))
    wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    rows = []
    for pct in (95, 90):
        g, r, d = compute_gate(wk, fu, n_train, pct)
        for lab, col in [("Rain", "shock_rain"), ("Diesel", "shock_diesel"), ("Rain OR Diesel", "I_shock")]:
            rows.append((f"P{pct}", lab, round(r, 1) if lab == "Rain" else (round(d, 2) if lab == "Diesel" else f"{r:.1f} mm / {d:.2f}%"),
                         int(g[col].sum()), int(g[g.split == "train"][col].sum()), int(g[g.split == "test"][col].sum())))
        if pct == 95:
            g.rename_axis("week_end").to_csv(OUT / "gate_P95.csv")
            def tmat(v):
                M = np.zeros((2, 2))
                for a, b in zip(v[:-1], v[1:]): M[a, b] += 1
                return M
            for sp in ("train", "test"):   # TRAIN matrix informs design; TEST matrix is descriptive only (Chapter 4)
                M = tmat(g[g.split == sp].I_shock.values.astype(int)); P = M / M.sum(1, keepdims=True)
                out = pd.DataFrame({"from": ["Normal (S_t-1=0)", "Shock (S_t-1=1)"], "to_normal_n": M[:, 0].astype(int), "to_shock_n": M[:, 1].astype(int),
                                    "to_normal_p": P[:, 0].round(3), "to_shock_p": P[:, 1].round(3), "from_weeks": M.sum(1).astype(int)})
                out.to_csv(OUT / f"table_transition_matrix_{sp}.csv", index=False); print(f"Transition matrix ({sp}, {int(M.sum())} transitions):"); print(out.to_string(index=False))
            print("Rain-only / diesel-only / both in test:", int(((g.shock_rain == 1) & (g.shock_diesel == 0) & (g.split == "test")).sum()),
                  int(((g.shock_rain == 0) & (g.shock_diesel == 1) & (g.split == "test")).sum()), int(((g.shock_rain == 1) & (g.shock_diesel == 1) & (g.split == "test")).sum()))
    T = pd.DataFrame(rows, columns=["Percentile", "Trigger", "Threshold", "Total", "Train", "Test"]); T.to_csv(OUT / "table_3_6_gate_counts.csv", index=False); print(T.to_string(index=False))
