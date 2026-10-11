"""compare_runs.py -- compare a re-run (scratch copy) with the original saved outputs.
Usage: python compare_runs.py /path/to/original/data/processed /path/to/scratch/data/processed
CSV files are compared numerically (tolerance 1e-6, ignoring the seconds column); JSON files are compared ignoring 'frozen_at'."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
A, B = Path(sys.argv[1]), Path(sys.argv[2]); ok = bad = 0
pats = ["table_3_5*_*.csv", "table_3_7*_*.csv", "table_3_8_*.csv", "ga_convergence_*.csv", "alpha_loss_curve_*.csv", "oof_predictions_*.csv", "wf_forecasts_*.csv", "wf_monthly_*.csv", "table_4_*.csv",
        "ga_selected_features_*.json", "xgb_best_*.json", "frozen_design_*.json"]
for pat in pats:
    for fa in sorted(A.glob(pat)):
        fb = B / fa.name
        if not fb.exists(): print("MISSING in re-run:", fa.name); bad += 1; continue
        if fa.suffix == ".json":
            ja, jb = json.load(open(fa)), json.load(open(fb)); ja.pop("frozen_at", None); jb.pop("frozen_at", None); same = ja == jb
            same = json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True)
        else:
            a, b = pd.read_csv(fa), pd.read_csv(fb); a = a.drop(columns=[c for c in a.columns if c == "seconds"]); b = b.drop(columns=[c for c in b.columns if c == "seconds"])
            same = a.shape == b.shape and list(a.columns) == list(b.columns)
            if same:
                for c in a.columns:
                    if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]): same &= bool(np.allclose(a[c], b[c], atol=1e-6, rtol=1e-6, equal_nan=True))
                    else: same &= bool((a[c].astype(str) == b[c].astype(str)).all())
        print(("SAME  " if same else "DIFF  ") + fa.name); ok += same; bad += (not same)
print(f"\n{ok} identical, {bad} different")
