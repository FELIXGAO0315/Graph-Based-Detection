import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore")
SEED = 42
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed


class Winsorizer(BaseEstimator, TransformerMixin):
    """clip at 1% / 99% quantiles, only variables with > 10 distinct values"""
    def __init__(self, lower=0.01, upper=0.99, min_unique=10):
        self.lower, self.upper, self.min_unique = lower, upper, min_unique

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.on_ = np.array([np.unique(c[~np.isnan(c)]).size > self.min_unique for c in X.T])
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        X[:, self.on_] = np.clip(X[:, self.on_], self.lo_[self.on_], self.hi_[self.on_])
        return X


def Prep():
    return make_pipeline(Winsorizer(), SimpleImputer(strategy="median", keep_empty_features=True))


class AdaptiveLassoScaler(BaseEstimator, TransformerMixin):
    """Fit ridge weights only on the rows/labels passed to this pipeline's fit."""
    def fit(self, X, y):
        self.ridge_ = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced", max_iter=5000).fit(X, y)
        self.scale_ = np.abs(self.ridge_.coef_.ravel()) + 1e-6
        return self

    def transform(self, X):
        return np.asarray(X, float) * self.scale_


def selector_pipeline(estimator, adaptive=False):
    # tune_and_fit fits a fresh pipeline on each inner training fold, then refits
    # the chosen C on the complete outer training set. Validation is transform-only.
    steps = [Prep(), StandardScaler()]
    if adaptive:
        steps.append(AdaptiveLassoScaler())
    return make_pipeline(*steps, estimator)


C_GRID = np.logspace(-3, 1, 12)

# helpers
def inner_splits(y, g, k=4, seed=SEED):
    """grouped, stratified inner CV splits"""
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, g))


def cv_score(make_model, X, y, splits):
    """mean PR-AUC over the splits"""
    s = []
    for tr, va in splits:
        if y[va].sum() == 0 or y[tr].sum() == 0:
            continue
        m = make_model().fit(X[tr], y[tr])
        sc = m.decision_function(X[va]) if hasattr(m, "decision_function") else m.predict_proba(X[va])[:, 1]
        s.append(average_precision_score(y[va], sc))
    return np.mean(s) if s else -np.inf


def tune_and_fit(make_model_C, X, y, splits, grid=C_GRID, one_se=True):
    """Tune a complete selector pipeline on raw X; refit the chosen C on all X.
    one_se: smallest C within 1 SE of the best.
    """
    res = []
    for C in grid:
        per = []
        for tr, va in splits:
            if y[va].sum() == 0:
                continue
            m = make_model_C(C).fit(X[tr], y[tr])
            per.append(average_precision_score(y[va], m.decision_function(X[va])))
        res.append((C, np.mean(per), np.std(per) / np.sqrt(len(per))))
    res = pd.DataFrame(res, columns=["C", "mean", "se"])
    best = res.loc[res["mean"].idxmax()]
    C_star = res.loc[res["mean"] >= best["mean"] - best["se"], "C"].min() if one_se else best["C"]
    return make_model_C(C_star).fit(X, y), C_star, res


def nonzero(coef, names, tol=1e-8):
    return [n for n, b in zip(names, np.ravel(coef)) if abs(b) > tol]


# variable selection: 4 penalized models, >= 3/4 vote
def select_variables(Xtr, ytr, gtr, names, verbose=True, seed=SEED):
    sp = inner_splits(ytr, gtr, seed=seed)
    out, info = {}, {}

    # Pass raw Xtr into tuning: every candidate owns its preprocessing.
    # lasso
    mk = lambda C: selector_pipeline(LogisticRegression(
        penalty="l1", C=C, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))
    m, C, _ = tune_and_fit(mk, Xtr, ytr, sp)
    out["lasso_logit"] = nonzero(m[-1].coef_, names); info["lasso_logit"] = f"C={C:.4g}"

    # elastic net, tune C and l1_ratio
    best = None
    for l1r in [0.2, 0.5, 0.8]:
        mk = lambda C, l1r=l1r: selector_pipeline(LogisticRegression(
            penalty="elasticnet", l1_ratio=l1r, C=C, solver="saga",
            class_weight="balanced", max_iter=3000, tol=1e-3, random_state=seed))
        m, C, res = tune_and_fit(mk, Xtr, ytr, sp)
        score = res["mean"].max()
        if best is None or score > best[0]:
            best = (score, m, C, l1r)
    out["elastic_net_logit"] = nonzero(best[1][-1].coef_, names)
    info["elastic_net_logit"] = f"C={best[2]:.4g}, l1_ratio={best[3]}"

    # adaptive lasso: ridge weights are fitted inside each inner training fold
    mk = lambda C: selector_pipeline(LogisticRegression(
        penalty="l1", C=C, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed),
        adaptive=True)
    m, C, _ = tune_and_fit(mk, Xtr, ytr, sp)
    out["adaptive_lasso_logit"] = nonzero(m[-1].coef_, names)
    info["adaptive_lasso_logit"] = f"C={C:.4g}"

    # L1 SVM
    mk = lambda C: selector_pipeline(LinearSVC(
        penalty="l1", loss="squared_hinge", dual=False, C=C,
        class_weight="balanced", max_iter=20000, random_state=seed))
    m, C, _ = tune_and_fit(mk, Xtr, ytr, sp, grid=np.logspace(-4, 0, 15))
    out["l1_svm"] = nonzero(m[-1].coef_, names); info["l1_svm"] = f"C={C:.4g}"

    sets = [set(v) for v in out.values()]
    inter = sorted(set.intersection(*sets))
    votes = pd.Series({n: sum(n in s for s in sets) for n in names})
    vote3 = sorted(votes[votes >= 3].index)
    if verbose:
        for k, v in out.items():
            print(f"  {k:22s} [{info[k]}] -> {len(v):3d} variables")
        print(f"  INTERSECTION of 4 -> {len(inter)} variables; >=3-of-4 vote -> {len(vote3)}")
    return out, inter, vote3, votes, info


# models 1-5
MODELS = ["1_linear_regression", "2_logistic_regression", "3_decision_tree",
          "4_random_forest", "5_gradient_boosting"]


def make_model(name, depth=None, seed=SEED):
    if name == "1_linear_regression":
        return make_pipeline(Prep(), StandardScaler(), LinearRegression())
    if name == "2_logistic_regression":
        return make_pipeline(Prep(), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if name == "3_decision_tree":
        return make_pipeline(Prep(), DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5,
                                                            class_weight="balanced", random_state=seed))
    if name == "4_random_forest":
        return make_pipeline(Prep(), RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=1))
    if name == "5_gradient_boosting":
        return make_pipeline(Prep(), HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=seed))
    raise ValueError(name)


def get_score(model, X):
    # P(y=1), y_hat for linear regression
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def fit_model(name, X, y, g, seed=SEED):
    if name != "3_decision_tree":
        return make_model(name, seed=seed).fit(X, y)
    # tree: pick max_depth by inner CV
    sp = inner_splits(y, g, seed=seed)
    cv = {d: np.mean([average_precision_score(y[va], get_score(make_model(name, d, seed).fit(X[tr], y[tr]), X[va]))
                      for tr, va in sp]) for d in [2, 3, 4, 5, 6]}
    best = max(cv, key=cv.get)
    m = make_model(name, best, seed).fit(X, y)
    m.best_depth_, m.depth_cv_ = best, cv
    return m
