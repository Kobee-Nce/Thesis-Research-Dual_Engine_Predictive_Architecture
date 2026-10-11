import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ser = sys.argv[1] if len(sys.argv) > 1 else "indicator"
P = Path("data/processed")
F = pd.read_csv(P / f"wf_forecasts_{ser}.csv", parse_dates=["week_end"])

def nw_ols(y, X, L=1):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    A = np.linalg.inv(X.T @ X)

    S = (X * e[:, None]).T @ (X * e[:, None])

    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)

    se = np.sqrt(np.diag(A @ S @ A))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - X.shape[1]))

    return b, se, p

def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0

    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1)

    return adj

rows = []

for c, g in F.sort_values(["commodity", "week_end"]).groupby("commodity"):
    dy = (g.y - g.naive).values
    ds = (g.sarimax - g.naive).values
    dx = (g.xgb - g.naive).values

    b, se, p = nw_ols(
        dy,
        np.column_stack([ds, dx])
    )

    rows.append(dict(
        Commodity=c,
        n=len(g),
        b_SARIMAX=b[1],
        p_SARIMAX=p[1],
        b_XGB=b[2],
        p_XGB=p[2],
        corr_changes=np.corrcoef(ds, dx)[0, 1]
    ))

T = pd.DataFrame(rows)

T["p_SARIMAX_holm"] = holm(T.p_SARIMAX)
T["p_XGB_holm"] = holm(T.p_XGB)

OUT = P / f"forecast_encompassing_changes_{ser}.csv"
T.round(4).to_csv(OUT, index=False)

print(T.round(3).to_string(index=False))

print(
    f"\nSARIMAX adds information beyond XGB: "
    f"{int((T.p_SARIMAX_holm < .05).sum())}/10 after Holm"
    f" | XGB adds information beyond SARIMAX: "
    f"{int((T.p_XGB_holm < .05).sum())}/10 after Holm"
)

print(f"\nSaved: {OUT}")
