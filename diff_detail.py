"""diff_detail.py -- size of the differences between two runs (for files compare_runs.py marks DIFF).
Usage (project root, venv active):
  python diff_detail.py "/Users/kobee/Thesis Data/Thesis_verify/data/processed" data/processed > logs_m4/diff_detail.txt
Prints, per file: number of differing cells, max absolute and max relative difference per numeric column,
and every differing non-numeric value (e.g. the 'converged' flag)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

A, B = Path(sys.argv[1]), Path(sys.argv[2])
pats = ["table_3_7*_*.csv", "table_3_8_*.csv", "alpha_loss_curve_*.csv", "oof_predictions_*.csv", "wf_forecasts_*.csv",
        "wf_monthly_*.csv", "table_4_*.csv"]
overall = 0.0
for pat in pats:
    for fa in sorted(A.glob(pat)):
        fb = B / fa.name
        if not fb.exists(): continue
        a, b = pd.read_csv(fa), pd.read_csv(fb)
        a = a.drop(columns=[c for c in a.columns if c == "seconds"]); b = b.drop(columns=[c for c in b.columns if c == "seconds"])
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f"\n### {fa.name}: SHAPE/COLUMNS DIFFER {a.shape} vs {b.shape}"); continue
        lines = []
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]):
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                d = np.abs(x - y); bad = ~np.isclose(x, y, atol=1e-6, rtol=1e-6, equal_nan=True)
                if bad.any():
                    rel = np.nanmax(d[bad] / np.maximum(np.abs(x[bad]), 1e-12))
                    lines.append(f"  {c:32s} cells={int(bad.sum()):5d}  max_abs={np.nanmax(d[bad]):.3g}  max_rel={rel:.3g}")
                    if fa.name.startswith("table_"): overall = max(overall, float(np.nanmax(d[bad])))
            else:
                bad = a[c].astype(str) != b[c].astype(str)
                if bad.any():
                    lines.append(f"  {c:32s} cells={int(bad.sum()):5d}  (non-numeric)")
                    key = [k for k in ["series", "Commodity", "commodity", "p", "q", "week_end", "model", "benchmark", "scope"] if k in a.columns]
                    for i in np.where(bad)[0][:15]:
                        lines.append(f"      row {i}: { {k: str(a.at[i, k]) for k in key} }  {a.at[i, c]} -> {b.at[i, c]}")
        if lines: print(f"\n### {fa.name}"); print("\n".join(lines))

for f in ["xgb_best_indicator.json"]:
    if not ((A / f).exists() and (B / f).exists()):
        print(f"\n### {f}: missing in one folder -- skipped"); continue
    ja, jb = json.load(open(A / f)), json.load(open(B / f))
    print(f"\n### {f}")
    for k in sorted(set(ja) | set(jb)):
        if json.dumps(ja.get(k), sort_keys=True) != json.dumps(jb.get(k), sort_keys=True):
            print(f"  {k}: {ja.get(k)} -> {jb.get(k)}")
print(f"\nLargest absolute difference in any table_* file: {overall:.3g}")
