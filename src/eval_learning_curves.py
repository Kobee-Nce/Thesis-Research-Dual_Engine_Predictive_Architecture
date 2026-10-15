"""eval_learning_curves.py -- training/validation loss curves and hyperparameter sensitivity of the FROZEN XGBoost design.
TRAIN weeks only (cv_utils.load_train asserts this); no design setting is changed or re-selected.
For each commodity and each of the four rolling-origin folds (Section 3.5.2), the frozen configuration (xgb_best_<series>.json)
is fitted with up to 300 trees and the loss is recorded after every tree on
  (a) the fitting part of the fold's training window (first 80%)      -> training loss
  (b) the last 20% of that window (the early-stopping hold-out)       -> early-stopping loss
  (c) the fold's 36 validation weeks                                  -> validation loss
Loss = RMSE of the weekly price change, divided by the commodity's SD of weekly changes in the first fold's training window,
pooled over commodities and folds (1.0 ~ predicting the training-mean change; the naive zero-change forecast is drawn for reference).
Also: Spearman correlation of each searched hyperparameter with the CV score over the 61 random-search configurations.
Outputs: table_4_learning_curve_<series>.csv, table_4_hparam_sensitivity_<series>.csv,
         fig_learning_curve_<series>.png, fig_hparam_sensitivity_<series>.png
Usage: python eval_learning_curves.py [indicator|noind]"""
import sys, json, warnings, numpy as np, pandas as pd, xgboost as xgb, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from importlib import import_module
cu = import_module("cv_utils"); OUT = cu.OUT; warnings.filterwarnings("ignore")
MAX_TREES, ES_FRAC = 300, 0.20
HP = ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"]

def rmse_path(m, X, y):
    return np.array([np.sqrt(np.mean((m.predict(X, iteration_range=(0, r)) - y) ** 2)) for r in range(1, MAX_TREES + 1)])

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indicator"
    B = json.load(open(OUT / f"xgb_best_{series}.json")); feats, params = B["features"], B["params"]
    fixed = dict(B["fixed"]); fixed["n_estimators"] = MAX_TREES
    P = cu.load_train(series); coms = list(P.commodity.unique()); n = (P.commodity == coms[0]).sum(); folds = cu.rolling_folds(n)
    acc = {k: [] for k in ("train", "es", "val")}; naive = []
    for c in coms:
        d = P[P.commodity == c]; X = d[feats].values; dy = (d.y - d.y_L1).values; s = dy[:folds[0][0]].std()
        for vs, ve in folds:
            k = int(vs * (1 - ES_FRAC)); m = xgb.XGBRegressor(**fixed, **params).fit(X[:k], dy[:k])
            for key, (a, b) in dict(train=(0, k), es=(k, vs), val=(vs, ve)).items():
                acc[key].append((rmse_path(m, X[a:b], dy[a:b]) / s) ** 2)
            naive.append(np.mean((dy[vs:ve] / s) ** 2))
    L = pd.DataFrame({k: np.sqrt(np.mean(v, axis=0)) for k, v in acc.items()}, index=pd.RangeIndex(1, MAX_TREES + 1, name="trees"))
    L["naive_val"] = np.sqrt(np.mean(naive)); L.round(4).to_csv(OUT / f"table_4_learning_curve_{series}.csv")
    best_val = int(L.val.idxmin()); best_es = int(L.es.idxmin())
    print(f"{series}: min validation loss {L.val.min():.4f} at {best_val} trees; min early-stopping loss at {best_es} trees; "
          f"at frozen {B['n_estimators']} trees: train {L.train[B['n_estimators']]:.4f}, val {L.val[B['n_estimators']]:.4f}; "
          f"at {MAX_TREES} trees: train {L.train[MAX_TREES]:.4f}, val {L.val[MAX_TREES]:.4f}; naive {L.naive_val.iloc[0]:.4f}")

    fig, a = plt.subplots(figsize=(8, 4.5))
    a.plot(L.index, L.train, label="Training (fitting part of fold window)", color="tab:blue")
    a.plot(L.index, L.es, label="Early-stopping hold-out (last 20% of window)", color="tab:green", ls="--")
    a.plot(L.index, L.val, label="Validation (36-week fold)", color="tab:orange")
    a.axhline(L.naive_val.iloc[0], color="grey", ls=":", label="Naive (zero change), validation")
    a.axvline(B["n_estimators"], color="red", ls="--", lw=0.9, label=f"Frozen model ({B['n_estimators']} trees)")
    a.set_xlabel("Number of trees (boosting rounds)"); a.set_ylabel("Scaled RMSE of weekly price change"); a.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / f"fig_learning_curve_{series}.png", dpi=160); plt.close(fig)

    R = pd.read_csv(OUT / f"table_3_7b_xgb_search_{series}.csv"); R = R[R.config != "default"]
    S = pd.DataFrame([dict(hyperparameter=h, spearman_rho=spearmanr(R[h], R.mean_rel_rmse)[0], p_value=spearmanr(R[h], R.mean_rel_rmse)[1],
                           range_min=R[h].min(), range_max=R[h].max()) for h in HP]).sort_values("spearman_rho", key=abs, ascending=False)
    S.round(4).to_csv(OUT / f"table_4_hparam_sensitivity_{series}.csv", index=False); print(S.round(3).to_string(index=False))
    fig, ax = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for a, h in zip(ax.ravel(), HP):
        a.scatter(R[h], R.mean_rel_rmse, s=14, color="tab:blue", alpha=0.7); a.scatter(R[h].iloc[0], R.mean_rel_rmse.iloc[0], s=40, color="red", zorder=5)
        if h in ("learning_rate", "reg_lambda", "reg_alpha"): a.set_xscale("log")
        rho = S.set_index("hyperparameter").spearman_rho[h]; a.set_title(f"{h} (Spearman ρ = {rho:.2f})", fontsize=9)
    for a in ax[:, 0]: a.set_ylabel("Mean relative RMSE (CV)")
    fig.tight_layout(); fig.savefig(OUT / f"fig_hparam_sensitivity_{series}.png", dpi=160); plt.close(fig)
