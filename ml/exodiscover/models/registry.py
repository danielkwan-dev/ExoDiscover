"""The candidate model ladder.

Every model is scored identically under the same grouped CV, so any reported
gain over the baseline is attributable to the model rather than to differences
in how it was measured. The DummyClassifier is not decoration: it is the
number every other row has to beat.

Missing values arrive as NaN, because the feature builder no longer invents
them. The boosted families and HistGradientBoosting learn a split direction for
NaN directly from the training data, which beats imputing a median and -- more
importantly here -- means SHAP explains the exact matrix the model scored.
LogisticRegression and RandomForest cannot take NaN, so those two carry an
explicit median imputer inside a pipeline; being fitted per fold, it never sees
the fold it is evaluated on.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

#: Families worth an Optuna search. dummy and logreg have nothing to tune.
TUNABLE: frozenset[str] = frozenset(
    {"random_forest", "hist_gb", "xgboost", "lightgbm", "catboost"}
)


#: Step name given to the learner inside every pipeline in this module, so a
#: caller can address its parameters as ``clf__<name>``.
FINAL_STEP = "clf"


def final_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """The learner itself, with any preprocessing pipeline stripped away."""
    return estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator


def param_prefix(estimator: BaseEstimator) -> str:
    """Prefix that addresses the learner's own parameters on ``estimator``."""
    return f"{FINAL_STEP}__" if isinstance(estimator, Pipeline) else ""


def candidates(seed: int) -> dict[str, BaseEstimator]:
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logreg": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        # The forest is the one tree family that cannot take NaN, so it gets an
        # explicit imputer. Fitted inside the pipeline, it learns its medians
        # from the training fold only and is persisted with the model, which is
        # what keeps scoring independent of the batch a row arrives in.
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed
                    ),
                ),
            ]
        ),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.05,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        ),
    }


#: Members of the soft-voting ensemble: the three boosted families, which sit
#: within 0.003 PR-AUC of each other on this data.
ENSEMBLE_MEMBERS: tuple[str, ...] = ("catboost", "xgboost", "lightgbm")


def ensembles(seed: int) -> dict[str, BaseEstimator]:
    """Soft-voting over the boosted families.

    Only soft voting, deliberately. A stacked ensemble would need an inner
    cross-validation to build its meta-features, and scikit-learn's
    StackingClassifier cannot be told to group that inner split by host star
    without metadata routing -- so it would quietly reintroduce exactly the
    split contamination this project exists to remove. Soft voting has no inner
    CV and therefore no such hole. The members are also highly correlated
    (Pearson r above 0.99 on held-out predictions), so a learned combiner has
    very little left to learn.
    """
    base = candidates(seed)
    members = [(name, base[name]) for name in ENSEMBLE_MEMBERS]
    return {
        "soft_vote": VotingClassifier(estimators=members, voting="soft", n_jobs=1),
    }
