"""08_ga.py -- STANDARD Genetic Algorithm for feature x lag selection (Section 3.5, RQ4). TRAIN weeks only.
Chromosome: 72 bits = 9 base series (price, rain_sum, rain_max, wind_max, t_mean, t_max, t_min, rain_roll4, diesel_pct) x lags 1..8.
Eight structural terms (rolling price mean/std, lagged CPI, Fourier x4, lagged gate) are always included.
Fitness = 1 / RMSE of a light XGBoost wrapper on rolling-origin CV (4 expanding folds x 36 weeks), pooled over the 10 commodities,
target = weekly price change standardized per commodity. 10 independent seeds -> selection frequency and stability.
Usage: python 08_ga.py [indicator|noind]"""
import sys, json, time, warnings, numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from importlib import import_module
cu = import_module("cv_utils"); OUT = cu.OUT; warnings.filterwarnings("ignore")

GA = dict(pop=24, generations=20, tournament=3, p_crossover=0.8, p_mutation=0.02, elitism=2, patience=6, init_prob=0.25, n_seeds=10)
SURROGATE = dict(n_estimators=40, max_depth=3, learning_rate=0.15, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, max_bin=32,
                 tree_method="hist", n_jobs=1, random_state=42, verbosity=0)
NB = len(cu.LAG_COLS); FIXED_IDX = np.arange(NB, NB + len(cu.FIXED_COLS))

class Fitness:
    def __init__(self, X, Y, folds): self.X, self.Y, self.folds, self.cache, self.evals = X, Y, folds, {}, 0
    def rmse(self, bits):
        key = bits.tobytes()
        if key not in self.cache:
            idx = np.r_[np.where(bits)[0], FIXED_IDX]; err = []
            for vs, ve in self.folds:
                m = xgb.XGBRegressor(**SURROGATE).fit(self.X[:, :vs][:, :, idx].reshape(-1, len(idx)), self.Y[:, :vs].reshape(-1))
                err.append(m.predict(self.X[:, vs:ve][:, :, idx].reshape(-1, len(idx))) - self.Y[:, vs:ve].reshape(-1))
            self.cache[key] = float(np.sqrt(np.mean(np.concatenate(err) ** 2))); self.evals += 1
        return self.cache[key]
    def fit(self, bits): return 1.0 / self.rmse(bits)

def run_ga(fit, seed, g=GA):
    rng = np.random.default_rng(seed); pop = rng.random((g["pop"], NB)) < g["init_prob"]
    for b in pop:
        if not b.any(): b[rng.integers(NB)] = True
    hist, best, stall = [], -1, 0
    for gen in range(g["generations"]):
        f = np.array([fit.fit(b) for b in pop]); order = np.argsort(-f)
        hist.append(dict(seed=seed, generation=gen, best_fitness=f.max(), mean_fitness=f.mean(), n_evals=fit.evals))
        if f.max() > best + 1e-9: best, stall, bb = f.max(), 0, pop[order[0]].copy()
        else: stall += 1
        if stall >= g["patience"]: break
        nxt = [pop[i].copy() for i in order[:g["elitism"]]]
        while len(nxt) < g["pop"]:
            pa = [pop[max(rng.integers(0, g["pop"], g["tournament"]), key=lambda i: f[i])] for _ in range(2)]
            c1, c2 = pa[0].copy(), pa[1].copy()
            if rng.random() < g["p_crossover"]:
                sw = rng.random(NB) < 0.5; c1[sw], c2[sw] = pa[1][sw], pa[0][sw]
            for c in (c1, c2):
                c ^= rng.random(NB) < g["p_mutation"]
                if not c.any(): c[rng.integers(NB)] = True
                if len(nxt) < g["pop"]: nxt.append(c)
        pop = np.array(nxt)
    return bb, pd.DataFrame(hist)

def filter_rmse(X, Y, folds, K):
    """Baseline selector: top-K lag features by mean |corr| with the target, chosen INSIDE each fold's training window."""
    err = []
    for vs, ve in folds:
        sc = np.array([[abs(np.corrcoef(X[c, :vs, j], Y[c, :vs])[0, 1]) for j in range(NB)] for c in range(X.shape[0])])
        idx = np.r_[np.argsort(-np.nan_to_num(sc.mean(0)))[:K], FIXED_IDX]
        m = xgb.XGBRegressor(**SURROGATE).fit(X[:, :vs][:, :, idx].reshape(-1, len(idx)), Y[:, :vs].reshape(-1))
        err.append(m.predict(X[:, vs:ve][:, :, idx].reshape(-1, len(idx))) - Y[:, vs:ve].reshape(-1))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))

def consensus(freq, thr=0.5):
    bits = freq >= thr
    return bits if bits.any() else (np.argsort(-freq).argsort() < 10)

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"; t0 = time.time()
    P = cu.load_train(series); coms, X, Y, S = cu.pooled_arrays(P, cu.LAG_COLS + cu.FIXED_COLS); folds = cu.rolling_folds(X.shape[1])
    fit = Fitness(X, Y, folds); runs, hist, masks = [], [], []
    for s in range(GA["n_seeds"]):
        b, h = run_ga(fit, seed=1000 + s); hist.append(h); masks.append(b)
        runs.append(dict(seed=1000 + s, cv_rmse=fit.rmse(b), best_fitness=fit.fit(b), n_selected=int(b.sum()), generations_run=len(h), cumulative_evals=fit.evals))
        print(f"seed {1000+s}: CV-RMSE {runs[-1]['cv_rmse']:.4f}  selected {int(b.sum()):2d}  gens {len(h)}  ({time.time()-t0:.0f}s)", flush=True)
    M = np.array(masks); freq = M.mean(0); cons = consensus(freq)
    jac = [(M[i] & M[j]).sum() / max((M[i] | M[j]).sum(), 1) for i in range(len(M)) for j in range(i + 1, len(M))]
    pd.DataFrame(runs).round(4).to_csv(OUT / f"table_3_5b_ga_runs_{series}.csv", index=False)
    pd.concat(hist).to_csv(OUT / f"ga_convergence_{series}.csv", index=False)
    SF = pd.DataFrame(dict(feature=cu.LAG_COLS, base=[c.rsplit("_L", 1)[0] for c in cu.LAG_COLS], lag=[int(c.rsplit("_L", 1)[1]) for c in cu.LAG_COLS], frequency=freq, in_consensus=cons))
    SF.round(3).to_csv(OUT / f"table_3_5c_ga_selection_frequency_{series}.csv", index=False)
    sel = [c for c, b in zip(cu.LAG_COLS, cons) if b]; json.dump(dict(series=series, selected_lag_features=sel, fixed_features=cu.FIXED_COLS, ga=GA), open(OUT / f"ga_selected_features_{series}.json", "w"), indent=1)
    # baselines (same CV folds). Note: GA/consensus were selected on these folds -> in-CV numbers are optimistic for GA.
    K = int(cons.sum()); rng = np.random.default_rng(7); rnd = [fit.rmse(np.isin(np.arange(NB), rng.choice(NB, K, replace=False))) for _ in range(30)]
    price_only = np.array([c.startswith("y_L") for c in cu.LAG_COLS]); allb = np.ones(NB, bool)
    rows = [("GA consensus (>=50% of seeds)", fit.rmse(cons), K), ("GA best single seed", min(r["cv_rmse"] for r in runs), int(np.mean([r["n_selected"] for r in runs]))),
            ("All 72 lag features", fit.rmse(allb), 72), ("Price lags only", fit.rmse(price_only), 8), (f"Filter top-{K} by |corr| (within-fold)", filter_rmse(X, Y, folds, K), K),
            (f"Random {K}-subsets (mean of 30)", float(np.mean(rnd)), K), (f"Random {K}-subsets (best of 30)", float(np.min(rnd)), K)]
    naive = float(np.sqrt(np.mean(np.concatenate([Y[:, vs:ve].reshape(-1) for vs, ve in folds]) ** 2)))
    # honest holdout: GA on folds 1-3 only (5 seeds), judged on fold 4 vs baselines
    f3, last = folds[:3], folds[3:]; fit3 = Fitness(X, Y, f3); m3 = np.array([run_ga(fit3, 2000 + s)[0] for s in range(5)]); c3 = consensus(m3.mean(0)); K3 = int(c3.sum())
    fit4 = Fitness(X, Y, last); ho = {"GA consensus (folds 1-3 only)": fit4.rmse(c3), "All 72 lag features": fit4.rmse(allb), "Price lags only": fit4.rmse(price_only),
          f"Filter top-{K3} by |corr|": filter_rmse(X, Y, last, K3), "Naive (zero change)": float(np.sqrt(np.mean(Y[:, last[0][0]:last[0][1]] ** 2)))}
    T = pd.DataFrame(rows, columns=["method", "cv_rmse_std", "n_lag_features"]); T["rel_to_naive"] = T.cv_rmse_std / naive
    T.round(4).to_csv(OUT / f"table_3_5d_ga_vs_baselines_{series}.csv", index=False)
    pd.Series(ho, name="fold4_rmse_std").round(4).to_csv(OUT / f"table_3_5e_ga_holdout_fold_{series}.csv")
    pd.DataFrame([dict(series=series, mean_pairwise_jaccard=np.mean(jac), consensus_size=K, naive_cv_rmse=naive, mean_runs_cv_rmse=np.mean([r["cv_rmse"] for r in runs]),
                       sd_runs_cv_rmse=np.std([r["cv_rmse"] for r in runs]), total_evals=fit.evals, seconds=time.time() - t0)]).round(4).to_csv(OUT / f"table_3_5f_ga_summary_{series}.csv", index=False)
    pd.DataFrame([GA | dict(surrogate="XGBoost 40 trees depth 3", folds=f"{len(folds)} x {cu.CV_VAL} weeks", encoding="72-bit mask")]).T.rename(columns={0: "value"}).to_csv(OUT / f"table_3_5a_ga_params_{series}.csv")
    # figures
    H = pd.concat(hist); fig, ax = plt.subplots(figsize=(6, 3.5))
    for s, h in H.groupby("seed"): ax.plot(h.generation, 1 / h.best_fitness, color="tab:blue", alpha=.5)
    ax.set_xlabel("Generation"); ax.set_ylabel("Best CV-RMSE (standardized)"); ax.set_title("GA convergence, 10 seeds"); plt.tight_layout(); plt.savefig(OUT / f"fig_ga_convergence_{series}.png", dpi=160); plt.close()
    HM = SF.pivot(index="base", columns="lag", values="frequency").loc[cu.LAG_BASES]; fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(HM.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_xticklabels(range(1, 9)); ax.set_yticks(range(len(HM))); ax.set_yticklabels(HM.index); ax.set_xlabel("Lag (weeks)"); plt.colorbar(im, label="Selection frequency"); plt.tight_layout(); plt.savefig(OUT / f"fig_ga_heatmap_{series}.png", dpi=160); plt.close()
    print("\nCONSENSUS:", K, "features:", sel); print(T.round(4).to_string(index=False)); print("\nHoldout fold 4:", {k: round(v, 4) for k, v in ho.items()})
    print(f"Mean pairwise Jaccard across seeds: {np.mean(jac):.3f} | naive CV RMSE {naive:.4f} | total {time.time()-t0:.0f}s")
