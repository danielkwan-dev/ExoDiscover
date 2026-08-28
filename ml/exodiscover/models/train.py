"""Model selection, tuning, and calibration.

Everything here runs under star-grouped cross-validation. Where scikit-learn
cannot route ``groups`` through a meta-estimator (``CalibratedClassifierCV``
is the notable case), the grouped folds are materialised into an explicit list
of index pairs and passed as ``cv``, which keeps the grouping guarantee intact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_validate

from exodiscover.config import settings
from exodiscover.data.splits import grouped_train_test_split, make_cv
from exodiscover.models.registry import (
    TUNABLE,
    candidates,
    ensembles,
    final_estimator,
    param_prefix,
)

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}


@dataclass
class TrainResult:
    name: str
    estimator: BaseEstimator
    cv_scores: dict[str, float] = field(default_factory=dict)


def score_estimator(estimator: BaseEstimator, X, y, groups) -> dict[str, float]:
    """Grouped-CV scores for one estimator, using the ladder's exact protocol."""
    return _score(estimator, X, y, groups)


def _score(estimator: BaseEstimator, X, y, groups) -> dict[str, float]:
    """Grouped-CV scores, with the fold-to-fold spread alongside each mean.

    The spread is not decoration. The boosted families land within a few
    thousandths of each other, and without the standard deviation there is no
    way to say whether picking the top row means anything.
    """
    out = cross_validate(
        estimator, X, y, groups=groups, cv=make_cv(), scoring=SCORING, n_jobs=1
    )
    scores: dict[str, float] = {}
    for metric in SCORING:
        fold_values = out[f"test_{metric}"]
        # cross_validate negates Brier so higher is better; restore the natural
        # orientation, where lower is better.
        if metric == "brier":
            fold_values = -fold_values
        scores[metric] = float(np.mean(fold_values))
        scores[f"{metric}_std"] = float(np.std(fold_values))
    return scores


def fit_candidates(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, *, include_ensembles: bool = True
) -> list[TrainResult]:
    """Score the whole ladder under grouped CV, best PR-AUC first."""
    pool = dict(candidates(settings.random_seed))
    if include_ensembles:
        pool.update(ensembles(settings.random_seed))

    results = [
        TrainResult(name=name, estimator=est, cv_scores=_score(est, X, y, groups))
        for name, est in pool.items()
    ]
    return sorted(results, key=lambda r: r.cv_scores["pr_auc"], reverse=True)


def tune_best(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, name: str, n_trials: int = 40
) -> BaseEstimator:
    """Optuna search over the winning family, scored on grouped PR-AUC."""
    pool = {**candidates(settings.random_seed), **ensembles(settings.random_seed)}
    base = pool[name]
    if name not in TUNABLE:
        # dummy, logreg, and the ensembles have nothing worth searching; the
        # ensemble's members are tuned via their own entries in the ladder.
        return base

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Address the learner's own parameters. Where a pipeline wraps it, every
    # name is prefixed, so matching the bare names against the pipeline's
    # parameter set would quietly match nothing and "tune" the default model.
    tunable_params = set(final_estimator(base).get_params())
    prefix = param_prefix(base)

    def _applicable(proposed: Mapping[str, Any]) -> dict[str, Any]:
        return {f"{prefix}{k}": v for k, v in proposed.items() if k in tunable_params}

    def objective(trial: optuna.Trial) -> float:
        proposed = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        }
        est = clone(base)
        est.set_params(**_applicable(proposed))
        return _score(est, X, y, groups)["pr_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = clone(base)
    best.set_params(**_applicable(study.best_params))
    return best


def build_calibrated(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    method: str | None = None,
) -> tuple[CalibratedClassifierCV, dict]:
    """Calibrate on a star-disjoint holdout, choosing the method by Brier score.

    Calibration matters here because the discovery ranking consumes
    probabilities rather than hard classes: an uncalibrated score orders
    candidates but cannot be read as "how likely is this a planet".

    Fitting the base model once and calibrating it on data it never saw --
    rather than the k-fold variant, which stores one model per fold -- keeps
    the deployed artifact a single estimator (0.1 MB rather than 25 MB) and
    keeps the calibration set genuinely unseen, since the split is grouped by
    host star.

    Isotonic is flexible but saturates: as a step function it collapses its top
    bin to a single value, which ties candidates exactly where the ranking
    matters. Sigmoid (Platt) is smooth and never ties, but assumes a shape the
    scores may not have. Rather than assert which is right, fit both and keep
    whichever scores better on a third, disjoint slice.

    Returns the calibrated model and a report of the comparison.
    """
    X_fit, rest_X, y_fit, rest_y = grouped_train_test_split(X, y, groups, test_size=0.4)
    rest_groups = groups.loc[rest_X.index]
    X_cal, X_val, y_cal, y_val = grouped_train_test_split(
        rest_X, rest_y, rest_groups, test_size=0.5
    )

    base = clone(estimator).fit(X_fit, y_fit)
    frozen = FrozenEstimator(base)

    trials: dict[str, float] = {}
    fitted: dict[str, CalibratedClassifierCV] = {}
    for candidate_method in ("isotonic", "sigmoid"):
        model = CalibratedClassifierCV(frozen, method=candidate_method)
        model.fit(X_cal, y_cal)
        probability = model.predict_proba(X_val)[:, 1]
        trials[candidate_method] = float(brier_score_loss(y_val, probability))
        fitted[candidate_method] = model

    chosen = method or min(trials, key=lambda k: trials[k])
    report = {
        "chosen": chosen,
        "brier_by_method": trials,
        "n_calibration": int(len(y_cal)),
        "n_validation": int(len(y_val)),
        "distinct_probabilities": {
            name: int(pd.Series(m.predict_proba(X_val)[:, 1]).nunique())
            for name, m in fitted.items()
        },
    }
    return fitted[chosen], report
