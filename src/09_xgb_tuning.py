"""09_xgb_tuning.py -- XGBoost hyperparameter search (Section 3.7). TRAIN weeks only.
Method: RANDOM SEARCH (60 configs, seed 42) scored by rolling-origin CV (4 expanding folds x 36 weeks).
Per fold the last 20% of the fold's training window is held out chronologically for EARLY STOPPING (patience 20, max 500 trees).
Target: weekly price change; forecast = last week's price + predicted change. Objective: mean over the 10 commodities of
RMSE / RMSE(naive last-week forecast), computed once over all validation steps (lower is better; 1.0 = naive).
Features: GA-selected lag features + 8 structural terms. Usage: python 09_xgb_tuning.py [indicator|noind]"""
import sys, json, time, warnings, numpy as np, pandas as pd, xgboost as xgb
from importlib import import_module
cu = import_module("cv_utils"); OUT = cu.OUT; warnings.filterwarnings("ignore")

FIXED = dict(objective="reg:squarederror", tree_method="hist", n_estimators=500, early_stopping_rounds=20, random_state=42, n_jobs=1, verbosity=0)
N_CONFIGS, ES_FRAC = 60, 0.20
DEFAULT = dict(max_depth=3, learning_rate=0.05, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)

def sample_configs(n, seed=42):
    r = np.random.default_rng(seed)
    return [dict(max_depth=int(r.choice([2, 3, 4, 5])), learning_rate=float(np.exp(r.uniform(np.log(0.01), np.log(0.2)))), min_child_weight=int(r.choice([1, 3, 5, 10, 20])),
                 subsample=float(r.uniform(0.6, 1.0)), colsample_bytree=float(r.uniform(0.4, 1.0)), reg_lambda=float(np.exp(r.uniform(np.log(0.5), np.log(20)))),
                 reg_alpha=float(np.exp(r.uniform(np.log(1e-3), np.log(5)))), gamma=float(r.choice([0, 0.1, 0.5, 1.0]))) for _ in range(n)]

def fit_early_stop(Xtr, dtr, params):
    k = int(len(Xtr) * (1 - ES_FRAC)); m = xgb.XGBRegressor(**FIXED, **params)
    m.fit(Xtr[:k], dtr[:k], eval_set=[(Xtr[k:], dtr[k:])], verbose=False); return m

def cv_eval(P, feats, params):
    coms = list(P.commodity.unique()); n = (P.commodity == coms[0]).sum(); folds = cu.rolling_folds(n); rel, its = {}, []
    for c in coms:
        d = P[P.commodity == c]; X = d[feats].values; y = d.y.values; y1 = d.y_L1.values; dy = y - y1; e, e0 = [], []
        for vs, ve in folds:
            m = fit_early_stop(X[:vs], dy[:vs], params); its.append(int(m.best_iteration) + 1)
            e.append(y1[vs:ve] + m.predict(X[vs:ve]) - y[vs:ve]); e0.append(y1[vs:ve] - y[vs:ve])
        rel[c] = np.sqrt(np.mean(np.concatenate(e) ** 2)) / np.sqrt(np.mean(np.concatenate(e0) ** 2))
    return rel, its

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"; t0 = time.time()
    P = cu.load_train(series); ga = json.load(open(OUT / f"ga_selected_features_{series}.json")); feats = ga["selected_lag_features"] + ga["fixed_features"]
    print(f"{len(feats)} features ({len(ga['selected_lag_features'])} GA-selected + {len(ga['fixed_features'])} structural)")
    rows = []
    for i, p in enumerate([DEFAULT] + sample_configs(N_CONFIGS)):
        rel, its = cv_eval(P, feats, p); rows.append(dict(config=("default" if i == 0 else i), mean_rel_rmse=np.mean(list(rel.values())), median_best_trees=int(np.median(its)), **p))
        if i % 10 == 0: print(f"  config {i:2d}/{N_CONFIGS}  mean rel-RMSE {rows[-1]['mean_rel_rmse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows).sort_values("mean_rel_rmse").reset_index(drop=True); R.round(4).to_csv(OUT / f"table_3_7b_xgb_search_{series}.csv", index=False)
    best = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v)) for k, v in R.iloc[0][list(DEFAULT)].items()}
    n_trees = int(max(20, R.iloc[0].median_best_trees)); rel, _ = cv_eval(P, feats, best)
    reld, _ = cv_eval(P, feats, DEFAULT)
    pd.DataFrame(dict(tuned=rel, default=reld)).round(4).rename_axis("Commodity").to_csv(OUT / f"table_3_7c_xgb_cv_by_commodity_{series}.csv")
    json.dump(dict(series=series, features=feats, params=best, n_estimators=n_trees, fixed={k: v for k, v in FIXED.items() if k not in ("n_estimators", "early_stopping_rounds")},
                   cv_mean_rel_rmse_tuned=float(np.mean(list(rel.values()))), cv_mean_rel_rmse_default=float(np.mean(list(reld.values())))), open(OUT / f"xgb_best_{series}.json", "w"), indent=1)
    print("\nBEST:", best, "| trees:", n_trees); print(f"CV mean rel-RMSE: tuned {np.mean(list(rel.values())):.4f} | default {np.mean(list(reld.values())):.4f} | naive = 1.0")
    print(pd.DataFrame(dict(tuned=rel, default=reld)).round(3).to_string()); print(f"{time.time()-t0:.0f}s")
