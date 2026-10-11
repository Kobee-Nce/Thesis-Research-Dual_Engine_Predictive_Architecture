"""eval_loo_shock_episode.py -- SECONDARY post-hoc analysis on the FROZEN test forecasts (no retraining).
Leave-one-shock-episode-out: weeks whose gate state is shock (S[t-1] = 1) are grouped into contiguous episodes. Each episode is dropped in turn and the
gated ensemble is re-scored against SARIMAX and naive, (A) on all remaining test weeks and (B) on the remaining shock-state weeks only.
Statistic: geometric mean over commodities of RMSE ratio (the paper's primary measure) + pooled Diebold-Mariano (HLN) test of gated vs SARIMAX.
Usage (project root):  python src/eval_loo_shock_episode.py [indicator|noind]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ser = sys.argv[1] if len(sys.argv) > 1 else "indicator"; P = Path("data/processed")
F = pd.read_csv(P / f"wf_forecasts_{ser}.csv", parse_dates=["week_end"]).sort_values(["commodity", "week_end"])
wide = lambda col: F.pivot(index="week_end", columns="commodity", values=col).sort_index()
Y = wide("y"); E = {m: wide(m) - Y for m in ["naive", "sarimax", "gated"]}; s2 = (E["naive"] ** 2).mean(); weeks = Y.index
flag = F.drop_duplicates("week_end").set_index("week_end").loc[weeks]; S1 = flag.shock_L1.values == 1; trig_prev = np.r_[["none"], flag.trigger_t.values[:-1]]
def dm(d):
    d = np.asarray(d, float); T = len(d); db = d.mean(); g0 = np.mean((d - db) ** 2)
    if T < 5 or g0 <= 1e-18: return np.nan, np.nan
    s = db / np.sqrt(g0 / T) * np.sqrt((T - 1) / T); return float(s), float(2 * (1 - stats.t.cdf(abs(s), T - 1)))
def score(keep):
    r = {}
    for b in ["sarimax", "naive"]:
        r[f"gated/{b}"] = float(np.exp(np.mean([np.log(np.sqrt(np.mean(E["gated"][c].values[keep] ** 2)) / np.sqrt(np.mean(E[b][c].values[keep] ** 2))) for c in Y.columns])))
    d = ((E["gated"] ** 2 - E["sarimax"] ** 2) / s2).mean(axis=1).values[keep]; stat, p = dm(d)
    return dict(weeks_kept=int(keep.sum()), ratio_gated_over_sarimax=r["gated/sarimax"], ratio_gated_over_naive=r["gated/naive"], mean_loss_diff_gated_minus_sarimax=float(d.mean()), DM_HLN=stat, p_value=p)
idx = np.where(S1)[0]; eps = []; st = pr = idx[0]
for i in idx[1:]:
    if i != pr + 1: eps.append((st, pr)); st = i
    pr = i
eps.append((st, pr)); rows = []
print(f"[{ser}] {len(idx)} shock-state weeks in {len(eps)} episodes")
for k, (a, b) in enumerate(eps, 1):
    print(f"  episode {k:2d}: {weeks[a].date()} to {weeks[b].date()}  ({b-a+1} wk)  trigger of the preceding week(s): {sorted(set(trig_prev[a:b+1]))}")
for panel, base in [("A: all test weeks", np.ones(len(weeks), bool)), ("B: shock-state weeks only", S1)]:
    rows.append(dict(panel=panel, omitted="none", episode_dates="", **score(base)))
    for k, (a, b) in enumerate(eps, 1):
        keep = base.copy(); keep[a:b + 1] = False
        if keep.sum() < 5: continue
        rows.append(dict(panel=panel, omitted=f"episode {k}", episode_dates=f"{weeks[a].date()} to {weeks[b].date()}", **score(keep)))
R = pd.DataFrame(rows); R.round(4).to_csv(P / f"loo_shock_episode_{ser}.csv", index=False)
pd.set_option("display.width", 200); print(R[["panel", "omitted", "weeks_kept", "ratio_gated_over_sarimax", "ratio_gated_over_naive", "DM_HLN", "p_value"]].round(3).to_string(index=False))
for panel in R.panel.unique():
    x = R[(R.panel == panel) & (R.omitted != "none")]
    print(f"\n{panel}: gated/SARIMAX ratio across leave-one-out runs: min {x.ratio_gated_over_sarimax.min():.3f}, max {x.ratio_gated_over_sarimax.max():.3f}; below 1 in {int((x.ratio_gated_over_sarimax < 1).sum())} of {len(x)} runs; "
          f"gated/naive max {x.ratio_gated_over_naive.max():.3f}")
