"""01_preprocess.py -- weekly calendar, chronological split, causal fuel-gap cleaning, weather aggregation."""
import pandas as pd, numpy as np
from importlib import import_module
ing = import_module("00_ingest")
OUT = ing.OUT
MAX_GAP = 4  # longest tolerated run of interpolated diesel weeks (observed maximum is 4)

def weekly_calendar(start="2018-01-01", end="2026-07-31"):
    return pd.date_range(start, end, freq="W-SUN")  # Monday-Sunday weeks, labeled by their Sunday

def split_indices(n_weeks, train_frac=0.8):
    n_train = int(n_weeks * train_frac)  # floor -> 357 train / 90 test (Table 3.2)
    return n_train, n_weeks - n_train

def clean_fuel(fuel_raw, weeks, n_train):
    """Weekly mean of readings. Train weeks: linear interpolation using training data only.
    Test weeks: last-observation-carried-forward (causal, no future values). Gap length is checked."""
    fw = fuel_raw.resample("W-SUN").mean().reindex(weeks)
    flag = fw.isna()
    runs = flag.groupby((~flag).cumsum()).sum().max()
    assert runs <= MAX_GAP, f"diesel gap of {runs} weeks exceeds cap {MAX_GAP}"
    tr = fw.iloc[:n_train].interpolate(limit_direction="both")
    te = fw.iloc[n_train:].copy()
    te.iloc[0] = te.iloc[0] if pd.notna(te.iloc[0]) else tr.iloc[-1]
    te = te.ffill()
    return pd.concat([tr, te]), flag

def aggregate_weather(weather_daily, weeks):
    w = weather_daily.resample("W-SUN").agg(
        rain_sum=("precip", "sum"), rain_max=("precip", "max"), wind_max=("wind_max", "max"),
        t_mean=("t_mean", "mean"), t_max=("t_max", "max"), t_min=("t_min", "min")).reindex(weeks)
    return w

if __name__ == "__main__":
    weeks = weekly_calendar(); n_train, n_test = split_indices(len(weeks))
    print(f"Weeks: {len(weeks)} | train {n_train} ({weeks[0].date()}..{weeks[n_train-1].date()}) | test {n_test} ({weeks[n_train].date()}..{weeks[-1].date()})")
    fuel_clean, flag = clean_fuel(ing.load_fuel(), weeks, n_train)
    print(f"Diesel: {int(flag.sum())} of {len(weeks)} weeks filled (train {int(flag[:n_train].sum())}, test {int(flag[n_train:].sum())})")
    wk = aggregate_weather(ing.load_weather(), weeks)
    assert not wk.isna().any().any() and not fuel_clean.isna().any()
    wk.to_pickle(OUT / "weather_weekly.pkl"); fuel_clean.to_pickle(OUT / "fuel_weekly_clean.pkl"); flag.to_pickle(OUT / "fuel_filled_flag.pkl")
    out = wk.copy(); out["diesel"] = fuel_clean; out["diesel_filled"] = flag
    out["split"] = ["train"] * n_train + ["test"] * n_test
    out.rename_axis("week_end").to_csv(OUT / "weekly_indicators.csv")
