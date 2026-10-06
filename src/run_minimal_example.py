"""run_minimal_example.py -- minimal working example (about 30 seconds, no model training).
Runs ingest -> weekly calendar/split -> Chow-Lin disaggregation -> shock gate, then checks the key numbers reported in Chapter 3.
Usage (from the project root, virtual environment active):  python src/run_minimal_example.py"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent; SRC = ROOT / "src"; OUT = ROOT / "data" / "processed"
STEPS = ["00_ingest.py", "01_preprocess.py", "02_disaggregate.py", "05_gate.py"]

for s in STEPS:
    print(f"Running {s} ...", flush=True)
    r = subprocess.run([sys.executable, str(SRC / s)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); sys.exit(f"FAILED at {s}. Check that the raw files are in data/raw/ (see README).")

results = []
def check(name, ok): results.append(ok); print(("PASS  " if ok else "FAIL  ") + name)

wi = pd.read_csv(OUT / "weekly_indicators.csv", parse_dates=["week_end"])
check("447 weekly observations (Mon-Sun weeks, Jan 2018 - Jul 2026)", len(wi) == 447)
check("chronological split: 357 train / 90 test weeks", (wi.split == "train").sum() == 357 and (wi.split == "test").sum() == 90)
check("45 diesel weeks filled (39 train, 6 test)", int(wi.diesel_filled.sum()) == 45 and int(wi[wi.split == "train"].diesel_filled.sum()) == 39)
W = pd.read_csv(OUT / "weekly_prices_disaggregated.csv", parse_dates=["week_end"])
check("weekly price table: 447 weeks x 10 commodities, all positive", W.shape == (447, 11) and (W.drop(columns="week_end") > 0).all().all())
pr = pd.read_csv(OUT / "prices_monthly_clean.csv", parse_dates=["month"]).set_index("month"); pr.index = pr.index.to_period("M")
m = W.drop(columns="week_end").groupby(W.week_end.dt.to_period("M").values).mean()
check("weekly prices average back to the 103 official PSA monthly prices (error < 1e-4)", len(m) == 103 and float((m - pr.loc[m.index]).abs().max().max()) < 1e-4)
g = pd.read_csv(OUT / "table_3_6_gate_counts.csv"); row = g[(g.Percentile == "P95") & (g.Trigger == "Rain OR Diesel")].iloc[0]
check("shock gate (P95 rain OR diesel): 34 train / 33 test shock weeks", int(row.Train) == 34 and int(row.Test) == 33)
print("\nALL CHECKS PASSED - environment and data are set up correctly." if all(results) else "\nSOME CHECKS FAILED - see above.")
sys.exit(0 if all(results) else 1)
