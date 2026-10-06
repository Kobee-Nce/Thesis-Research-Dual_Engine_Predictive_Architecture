"""
00_ingest.py -- parse raw PSA/Open-Meteo/DOE/CPI files directly from the VS Code workspace.

Expected project layout (this script lives in <project>/src/; results are written to <project>/data/processed/):

    <project>/data/raw/
        psa/      2018-2026 Carrots.xlsx, 2018-2026 Fruit Vegetables.xlsx,
                  2018-2026 Garlic Onion.xlsx, 2018-2026 Leafy Vegetables.xlsx
        pagasa/   Thesis - Weather Data.xls      (Open-Meteo export; folder name is historical)
        doe/      Doe_fuel_cleaned_data.xlsx
        cpi/      2018-2026 CPI.xlsm

Only the four "2018-2026 *.xlsx" files above are needed from the Vegetable Data
folder -- see the file-by-file breakdown printed at the bottom of this script.
"""
import pandas as pd, numpy as np, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the Thesis project folder (src/ sits inside it)
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------------------------------

VEG_DIR = ROOT / "data" / "raw" / "psa"
WEATHER_DIR = ROOT / "data" / "raw" / "pagasa"
FUEL_DIR = ROOT / "data" / "raw" / "doe"
CPI_DIR = ROOT / "data" / "raw" / "cpi"

WINDOW = ("2018-01-01", "2026-07-31")
MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]

# Commodity -> (source file, exact PSA row label)
COMMODITY_SOURCES = {
    "Cabbage":       ("2018-2026 Leafy Vegetables.xlsx", "CABBAGE, 1 KG"),
    "Carrot":        ("2018-2026 Carrots.xlsx",          "CARROT, 1 KG"),
    "Tomato":        ("2018-2026 Fruit Vegetables.xlsx", "TOMATO, 1 KG"),
    "Eggplant":      ("2018-2026 Fruit Vegetables.xlsx", "EGGPLANT, LONG, PURPLE, 1 KG"),
    "Ampalaya":      ("2018-2026 Fruit Vegetables.xlsx", "BITTER GOURD (AMPALAYA), FRUIT, 1 KG"),
    "Squash":        ("2018-2026 Fruit Vegetables.xlsx", "SQUASH, 1 KG"),
    "Patola":        ("2018-2026 Fruit Vegetables.xlsx", "PATOLA, 1 KG"),
    "Upo":           ("2018-2026 Fruit Vegetables.xlsx", "COMMON GOURD (UPO), 1 KG"),
    "Red onion":     ("2018-2026 Garlic Onion.xlsx",     "ONION RED CREOLE (BERMUDA RED), 1 KG"),
    "Native garlic": ("2018-2026 Garlic Onion.xlsx",     "GARLIC, NATIVE, 1 KG"),
}


def _parse_psa_sheet(path):
    """Parse a PSA-format workbook (title row, year row, month row, area/commodity rows) into long form."""
    d = pd.read_excel(path, header=None)
    yrs = d.iloc[2].ffill()
    mon = d.iloc[3]
    out = []
    r, area = 4, None
    while r < d.shape[0] and (pd.notna(d.iloc[r, 1]) or pd.notna(d.iloc[r, 0])):
        if pd.notna(d.iloc[r, 0]):
            area = str(d.iloc[r, 0])
        name = d.iloc[r, 1]
        if pd.notna(name):
            for c in range(2, d.shape[1]):
                if mon.iloc[c] in MONTHS:
                    val = pd.to_numeric(d.iloc[r, c], errors="coerce")
                    out.append((area, str(name).strip(), int(float(yrs.iloc[c])),
                                MONTHS.index(mon.iloc[c]) + 1, val))
        r += 1
    return pd.DataFrame(out, columns=["area", "commodity", "year", "month", "price"])


def load_prices():
    frames = {}
    cache = {}
    for commodity, (fname, label) in COMMODITY_SOURCES.items():
        if fname not in cache:
            cache[fname] = _parse_psa_sheet(VEG_DIR / fname)
        df = cache[fname]
        s = df[df.commodity == label].dropna(subset=["price"])
        s = s.set_index(pd.to_datetime(dict(year=s.year, month=s.month, day=1))).price
        frames[commodity] = s
    piv = pd.DataFrame(frames)
    return piv.loc[WINDOW[0]:WINDOW[1]]


def load_weather():
    raw = pd.read_excel(WEATHER_DIR / "Thesis - Weather Data.xls", header=None)
    w = raw.iloc[5:].copy()
    w.columns = ["date", "precip", "wind_max", "t_mean", "t_max", "t_min"]
    w["date"] = pd.to_datetime(w.date, errors="coerce")
    for c in w.columns[1:]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["date"]).set_index("date")
    return w.loc[WINDOW[0]:WINDOW[1]]


def load_fuel():
    M = {m: i + 1 for i, m in enumerate(
        ["january","february","march","april","may","june","july",
         "august","september","october","november","december"])}

    def parse_month_field(s):
        s = str(s).strip().lower().replace("feb-march", "february - march")
        parts = [p.strip() for p in re.split(r"\s*-\s*", s)]
        return M.get(parts[0]), (M.get(parts[1]) if len(parts) > 1 else None)

    df = pd.read_excel(FUEL_DIR / "Doe_fuel_cleaned_data.xlsx", header=None).iloc[1:, :4]
    df.columns = ["Month", "Date", "Year", "Diesel"]
    sec, recs = None, []
    global FUEL_DROPPED; FUEL_DROPPED = []
    for _, r in df.iterrows():
        if re.fullmatch(r"\d{4}", str(r.Month).strip()) and pd.isna(r.Date):
            sec = int(r.Month)
            continue
        a, b = parse_month_field(r.Month)
        day_match = re.findall(r"\d+", str(r.Date))
        if a is None or not day_match:
            FUEL_DROPPED.append((str(r.Month), str(r.Date), r.Year, r.Diesel, 'unparsed')); continue
        y = int(r.Year) if pd.notna(r.Year) else sec
        if b == 1 and a == 12:
            y -= 1
        try:
            dt = pd.Timestamp(year=y, month=a, day=int(day_match[0]))
        except Exception:
            FUEL_DROPPED.append((str(r.Month), str(r.Date), r.Year, r.Diesel, 'bad date')); continue
        recs.append((dt, pd.to_numeric(r.Diesel, errors="coerce")))
    f = pd.DataFrame(recs, columns=["start", "diesel"]).sort_values("start")
    f = f.set_index("start").diesel
    for d, v in f[(f.index < WINDOW[0]) | (f.index > WINDOW[1])].items():
        FUEL_DROPPED.append((str(d.date()), "", "", v, "outside window"))
    return f.loc[WINDOW[0]:WINDOW[1]]


def load_cpi():
    MONTHS_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df = pd.read_excel(CPI_DIR / "2018-2026 CPI.xlsm", header=None)
    yrs = df.iloc[2].ffill()
    mon = df.iloc[3]
    target = "01.1.7 - Vegetables, tubers, plantains, cooking bananas and pulses (ND)"
    out = []
    for r in range(4, df.shape[0]):
        name = df.iloc[r, 1]
        if pd.notna(name) and str(name).strip() == target:
            for c in range(2, df.shape[1]):
                if mon.iloc[c] in MONTHS_ABBR:
                    val = pd.to_numeric(df.iloc[r, c], errors="coerce")
                    out.append((int(float(yrs.iloc[c])), MONTHS_ABBR.index(mon.iloc[c]) + 1, val))
            break
    s = pd.DataFrame(out, columns=["year", "month", "cpi"])
    s = s.set_index(pd.to_datetime(dict(year=s.year, month=s.month, day=1))).cpi.sort_index()
    return s.loc[WINDOW[0]:WINDOW[1]]


def export_clean():
    """Write the four clean CSVs required by the Chapter 3 brief."""
    prices, weather, fuel, cpi = load_prices(), load_weather(), load_fuel(), load_cpi()
    prices.rename_axis("month").to_csv(OUT / "prices_monthly_clean.csv")
    weather.rename_axis("date").to_csv(OUT / "weather_daily_clean.csv")
    fuel.rename_axis("week_start").to_csv(OUT / "fuel_diesel_clean.csv")
    cpi.rename_axis("month").to_csv(OUT / "cpi_veg_monthly_clean.csv")
    pd.DataFrame(FUEL_DROPPED, columns=["Month", "Date", "Year", "Diesel", "reason"]).to_csv(OUT / "fuel_rows_not_used.csv", index=False)
    return prices, weather, fuel, cpi


if __name__ == "__main__":
    prices, weather, fuel, cpi = export_clean()
    print("Prices:", prices.shape, "| % missing:", round(prices.isna().mean().mean() * 100, 2))
    print("Weather:", weather.shape, "| % missing:", round(weather.isna().mean().mean() * 100, 2))
    print("Fuel obs:", len(fuel), "| blank diesel:", int(fuel.isna().sum()), "| gaps handled in 01_preprocess.py")
    print("Fuel rows not used:", len(FUEL_DROPPED), "->", pd.Series([d[4] for d in FUEL_DROPPED]).value_counts().to_dict())
    print("CPI:", cpi.shape, "| % missing:", round(cpi.isna().mean() * 100, 2))
