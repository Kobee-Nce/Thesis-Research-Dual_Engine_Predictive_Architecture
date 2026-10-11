"""02_disaggregate.py -- Chow-Lin GLS temporal disaggregation (monthly PSA -> weekly), AR(1) rho ESTIMATED
by profile likelihood on training months only; indicators standardized with training statistics only."""
import numpy as np, pandas as pd
from scipy.linalg import toeplitz
from importlib import import_module
ing = import_module("00_ingest"); pre = import_module("01_preprocess"); OUT = ing.OUT
RHO_GRID = np.round(np.arange(0.0, 0.991, 0.01), 2)

def build_indicators(weather_wk, fuel_wk, n_train):
    z = lambda s: (s - s.iloc[:n_train].mean()) / s.iloc[:n_train].std()   # train-only standardization
    return pd.DataFrame({"const": 1.0, "rain": z(weather_wk.rain_sum), "diesel": z(fuel_wk)}, index=weather_wk.index)

def agg_matrix(week_ord, periods):
    C = np.zeros((len(periods), len(week_ord)))
    for i, p in enumerate(periods):
        w = week_ord == p.ordinal; C[i, w] = 1.0 / w.sum()
    return C

def _gls(y, Xl, V):
    Vi_X = np.linalg.solve(V, Xl); Vi_y = np.linalg.solve(V, y)
    b = np.linalg.solve(Xl.T @ Vi_X, Xl.T @ Vi_y); u = y - Xl @ b
    n = len(y); s2 = u @ np.linalg.solve(V, u) / n; ld = np.linalg.slogdet(V)[1]
    return b, -0.5 * (n * np.log(s2) + ld)

def chow_lin(y_low, X, week_periods, train_periods, rho=None):
    """y_low: Series indexed by Period (low-frequency average values). X: weekly indicators (n_weeks x k).
    train_periods: set of Periods fully inside the training window (used for beta and rho)."""
    wo = np.array([p.ordinal for p in week_periods]); periods = list(y_low.index)
    C = agg_matrix(wo, periods); Xv = X.values; n = len(wo)
    tr = np.array([p in train_periods for p in periods]); idx = np.arange(n)
    def fit(r):
        S = toeplitz(r ** idx); Ct = C[tr]
        return _gls(y_low.values[tr], Ct @ Xv, Ct @ S @ Ct.T + 1e-9 * np.eye(tr.sum()))
    if rho is None:
        lls = [fit(r)[1] for r in RHO_GRID]; rho = float(RHO_GRID[int(np.argmax(lls))])
    b, _ = fit(rho); S = toeplitz(rho ** idx); V = C @ S @ C.T + 1e-9 * np.eye(len(periods))
    resid = y_low.values - C @ Xv @ b
    p_hat = Xv @ b + S @ C.T @ np.linalg.solve(V, resid)
    return pd.Series(p_hat, index=X.index), b, rho

def train_periods_of(weeks, n_train, freq="M"):
    pw = pd.Series(weeks.to_period(freq), index=weeks)
    test_p = set(pw.iloc[n_train:]); return {p for p in pw.iloc[:n_train] if p not in test_p}  # drop split-straddling period

if __name__ == "__main__":
    prices = ing.load_prices(); weeks = pre.weekly_calendar(); n_train, _ = pre.split_indices(len(weeks))
    wk = pd.read_pickle(OUT / "weather_weekly.pkl"); fu = pd.read_pickle(OUT / "fuel_weekly_clean.pkl")
    X = build_indicators(wk, fu, n_train); pw = pd.Series(weeks.to_period("M"), index=weeks); trp = train_periods_of(weeks, n_train)
    res, res0, pars, chk = {}, {}, [], []
    for c in prices.columns:
        y = prices[c].copy(); y.index = y.index.to_period("M")
        p, b, rho = chow_lin(y, X, pw, trp); res[c] = p
        res0[c] = chow_lin(y, X[["const"]], pw, trp)[0]   # indicator-free variant (alternative disaggregation, ablation A2)
        pars.append((c, rho, *b)); m = p.groupby(pw).mean(); chk.append((m - y.loc[m.index]).abs().max())
    W = pd.DataFrame(res); assert max(chk) < 1e-4 and (W > 0).all().all()
    W.rename_axis("week_end").to_csv(OUT / "weekly_prices_disaggregated.csv"); W.to_pickle(OUT / "weekly_prices_disaggregated.pkl")
    W0 = pd.DataFrame(res0); W0.rename_axis("week_end").to_csv(OUT / "weekly_prices_disaggregated_noind.csv"); W0.to_pickle(OUT / "weekly_prices_disaggregated_noind.pkl")
    P = pd.DataFrame(pars, columns=["Commodity", "rho", "b_const", "b_rain_z", "b_diesel_z"]).round(3)
    P.to_csv(OUT / "table_disagg_parameters.csv", index=False); print(P.to_string(index=False))
    print("Max |monthly mean of weekly - PSA monthly| =", f"{max(chk):.2e}")
