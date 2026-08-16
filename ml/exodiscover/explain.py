"""SHAP explanations, global and per-row.

The per-row explanations are what make the discovery shortlist actionable: a
ranked list without reasons asks a reviewer to trust the model, whereas a
ranked list with contributions lets them check its reasoning.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline


class Contribution(TypedDict):
    """One feature's signed contribution to a single prediction."""

    feature: str
    value: float
    shap: float


class Importance(TypedDict):
    feature: str
    mean_abs_shap: float


def unwrap(model):
    """Return the underlying tree model.

    SHAP's TreeExplainer needs the tree itself, not the calibration wrapper or
    the preprocessing pipeline around it.
    """
    if isinstance(model, CalibratedClassifierCV):
        return unwrap(model.calibrated_classifiers_[0].estimator)
    if isinstance(model, FrozenEstimator):
        return unwrap(model.estimator)
    if isinstance(model, Pipeline):
        return unwrap(model.steps[-1][1])
    return model


def _tree_shap(model, X: pd.DataFrame) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):  # older API: one array per class
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # newer API: (n_rows, n_features, n_classes)
        values = values[:, :, -1]
    return values


def shap_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive class, shaped (n_rows, n_features).

    A soft-voting ensemble predicts the mean of its members' probabilities.
    SHAP attributions are additive per model, so the mean of the members'
    attributions is the attribution of that mean -- which makes averaging the
    member matrices the exactly corresponding explanation, not an approximation
    of convenience. TreeExplainer cannot take the VotingClassifier itself, so
    the members are explained individually and combined here.
    """
    base = unwrap(model)
    if isinstance(base, VotingClassifier):
        members = [_tree_shap(unwrap(est), X) for est in base.estimators_]
        return np.mean(members, axis=0)
    return _tree_shap(base, X)


def global_importance(model, X: pd.DataFrame) -> list[Importance]:
    mean_abs = np.abs(shap_matrix(model, X)).mean(axis=0)
    rows: list[Importance] = [
        {"feature": str(f), "mean_abs_shap": float(v)}
        for f, v in zip(X.columns, mean_abs, strict=True)
    ]
    return sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)


def _row_contributions(columns, values, row: pd.Series, top_n: int) -> list[Contribution]:
    rows: list[Contribution] = [
        {"feature": str(f), "value": float(row[f]), "shap": float(s)}
        for f, s in zip(columns, values, strict=True)
    ]
    return sorted(rows, key=lambda r: abs(r["shap"]), reverse=True)[:top_n]


def explain_row(model, X: pd.DataFrame, index: int = 0, top_n: int = 6) -> list[Contribution]:
    """Signed feature contributions for one row, largest magnitude first."""
    values = shap_matrix(model, X.iloc[[index]])[0]
    return _row_contributions(X.columns, values, X.iloc[index], top_n)


def explain_rows(model, X: pd.DataFrame, top_n: int = 3) -> list[list[Contribution]]:
    """Per-row contributions for a whole frame, computed in a single pass."""
    matrix = shap_matrix(model, X)
    return [
        _row_contributions(X.columns, matrix[i], X.iloc[i], top_n) for i in range(len(X))
    ]
