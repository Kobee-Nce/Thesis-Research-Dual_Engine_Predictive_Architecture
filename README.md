# Predictive Modeling of Agricultural Commodity Prices in Davao City Using Weather and Market Data

Code and data for the undergraduate thesis of Basong, Solano and Soriano (Ateneo de Davao University).
The pipeline reconstructs weekly vegetable prices from monthly PSA records (Chow–Lin disaggregation),
selects lagged weather and fuel features with a Genetic Algorithm, and forecasts prices one week ahead
with a regime-gated SARIMAX + XGBoost ensemble, evaluated walk-forward on a 90-week test set.

## 1. Requirements

- Python **3.12** (tested with 3.12.14 and 3.12.15; pandas 3 needs Python ≥ 3.11)
- About 1 GB of free disk space; no GPU is needed
- **macOS only:** XGBoost needs the OpenMP runtime. Install it with `brew install libomp` before running scripts 08 onward
- macOS, Linux or Windows

## 2. Installation

```bash
git clone https://github.com/Kobee-Nce/Thesis-Research-Dual_Engine_Predictive_Architecture.git
cd Thesis-Research-Dual_Engine_Predictive_Architecture

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` pins every package version used for the reported results
(key libraries: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0,
xgboost 3.4.1, matplotlib 3.11.2, openpyxl 3.1.5, xlrd 2.0.2, pyarrow 25.0.1).

## 3. Data

The raw files must be in these folders (relative to the project root):

```
data/raw/
    psa/      2018-2026 Carrots.xlsx, 2018-2026 Fruit Vegetables.xlsx,
              2018-2026 Garlic Onion.xlsx, 2018-2026 Leafy Vegetables.xlsx
    pagasa/   Thesis - Weather Data.xls        (Open-Meteo export; folder name is historical)
    doe/      Doe_fuel_cleaned_data.xlsx       (DOE weekly diesel prices, Davao City)
    cpi/      2018-2026 CPI.xlsm
    Data-Scrape-Final.xlsx                     (DFTC daily retail prices; validation only; NOT in this repository, see below)
```

All outputs (tables, figures, forecasts, frozen design files) are written to `data/processed/`.

### Data sources and terms of use

| Data | Source | Terms |
|---|---|---|
| Monthly retail prices (Davao City) | [PSA OpenSTAT](https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2M__2018NEW/) | CC BY 4.0, credit the Philippine Statistics Authority ([terms](https://openstat.psa.gov.ph/Terms)) |
| Consumer Price Index (vegetables) | [PSA OpenSTAT](https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2M__PI__CPI__2018NEW/) | CC BY 4.0, credit the Philippine Statistics Authority |
| Weekly diesel pump prices (Davao City) | [DOE Mindanao pump prices](https://doe.gov.ph/data-and-prices/liquid-fuels/retail-pump-prices/mindanao-pump-prices) | Public domain unless otherwise stated (DOE website); weekly reports compiled into one spreadsheet |
| Daily weather | [Open-Meteo](https://open-meteo.com/) | CC BY 4.0, "Weather data by Open-Meteo.com" ([licence](https://open-meteo.com/en/licence)) |
| DFTC daily retail prices per kg (validation only) | Daily price posts on the official Davao Food Terminal Complex Facebook page, compiled by the researchers (2 Oct 2023 – 5 Sep 2026) | No reuse terms stated, so **not redistributed here**; available from the authors on request |

The files are the versions downloaded for the reported run. Data downloaded again may differ, because sources revise past values.
Use of these data does not imply endorsement by the source agencies.

**Without the DFTC file**, everything runs except `02c_dftc_weekly_validation.py` (skip it) and the DFTC comparison in `12_tests.py`, which is skipped automatically. The minimal working example does not need it.


## 4. Minimal working example (about 30 seconds)

From the project root, with the virtual environment active:

```bash
python src/run_minimal_example.py
```

This runs ingestion, the weekly calendar and split, Chow–Lin disaggregation and the shock gate,
then checks six numbers reported in Chapter 3. Expected output ends with:

```
PASS  447 weekly observations (Mon-Sun weeks, Jan 2018 - Jul 2026)
PASS  chronological split: 357 train / 90 test weeks
PASS  45 diesel weeks filled (39 train, 6 test)
PASS  weekly price table: 447 weeks x 10 commodities, all positive
PASS  weekly prices average back to the 103 official PSA monthly prices (error < 1e-4)
PASS  shock gate (P95 rain OR diesel): 34 train / 33 test shock weeks

ALL CHECKS PASSED - environment and data are set up correctly.
```

## 5. Full pipeline

> **Important:** run the full pipeline in a **copy** of the project. `10_gate_weights.py` rewrites the
> time stamp in `data/processed/frozen_design_*.json`, which records when the design was frozen
> before the test set was evaluated.

```bash
cd src
python 00_ingest.py
python 01_preprocess.py
python 02_disaggregate.py
python 02b_disaggregation_backtest.py
python 02c_dftc_weekly_validation.py
python 03_stationarity.py
python 03_features.py indicator && python 03_features.py noind
python 04_figures.py
python 04b_shock_timeline.py
python 05_gate.py

for s in indicator noind; do
  python 07_sarimax_order.py $s
  python 08_ga.py $s
  python 09_xgb_tuning.py $s
  python 10_gate_weights.py $s
  python 11_walkforward.py $s --confirm-test     # test set: only after the design is frozen
  python 12_tests.py $s --confirm-test
done

# supplementary (post-hoc) analyses on the frozen forecasts
cd .. 
python src/eval_temporal_split.py
python src/eval_seasonal_naive.py
for s in indicator noind; do
  python src/eval_loo_shock_episode.py $s
  python src/eval_model_confidence_set.py $s
  python src/eval_error_diagnostics.py $s
  python src/eval_loss_robustness.py $s
done
python src/forecast_encompassing_changes.py indicator   # indicator series only (gated = SARIMAX for noind)
```

`11_walkforward.py` and `12_tests.py` refuse to run without a flag: `--confirm-test` uses the real
test set; `--dry-run` uses a stand-in window from the training weeks for debugging.

Approximate run time of scripts 00–12: about 43 min on an Apple M1 (8 GB) and 28 min on an Apple M4 (16 GB); the walk-forward step takes 14 + 18 min (M1) or 9 + 12 min (M4).

## 6. Checking a rerun

```bash
# compare a rerun (copy) with the original saved outputs
python compare_runs.py /path/to/original/data/processed /path/to/copy/data/processed
python diff_detail.py   /path/to/original/data/processed /path/to/copy/data/processed   # size of any differences

# audit every checkable number in Sections 3.7-3.8 against the saved outputs
python verify_3_7_3_8.py data/processed
```

A rerun on the original Apple M1 reproduced 57 of 57 result files identically. On a different processor (Apple M4), all design files and
30 of 57 files are identical and the rest differ only in floating-point digits (no reported value or test decision changes);
the audit script passes 62 of 62 checks on both machines.

## 7. Repository layout

| Path | Contents |
|---|---|
| `src/` | Pipeline scripts, numbered in run order |
| `data/raw/` | Raw input files (see Section 3) |
| `data/processed/` | All pipeline outputs |
| `logs/`, `logs_m4/` | Console output, run times, hardware and comparison results of the M1 and M4 reruns |
| `verify_3_7_3_8.py` | Read-only audit of Sections 3.7–3.8 |
| `compare_runs.py`, `diff_detail.py` | File-by-file comparison of two runs, and the size of any differences |
| `requirements.txt` | Pinned package versions |

## 8. Random seeds

GA runs: 1000–1009 (hold-out check: 2000–2004); XGBoost search and fitting: 42;
moving-block bootstrap and Model Confidence Set: 20261001. XGBoost runs single-threaded
(`n_jobs=1`) so results are deterministic.
