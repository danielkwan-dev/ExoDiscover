"""Evidence that the headline score is earned rather than memorised.

A ROC-AUC of 0.98 invites the obvious suspicion, and the honest answer is not
an argument but two measurements.

The first is the generalisation gap. Overfitting is a *difference* -- what the
model scores on rows it trained on, minus what it scores on stars it has never
seen. A model that memorised its training set approaches 1.0 on the left and
falls away sharply on the right. A small gap means the score transfers.

The second is the learning curve. A memorising model needs every row it can
get and degrades badly when starved; a model that has found real structure
saturates. Measuring the same held-out stars against progressively smaller
training sets separates the two cases, and it also answers the question that
follows -- whether more data would help, or whether the ceiling belongs to the
task rather than the model.

Neither is affected by the train/test ratio, which is the usual first guess:
changing 80/20 to 70/30 moves this dataset's score by under 0.001, because the
split size was never what produced the number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from exodiscover.config import settings

#: Fractions of the training *stars* used for each learning-curve point.
FRACTIONS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)

#: Gap above which the model is judged to have memorised rather than learned.
#: Held deliberately tight: gradient boosting on tabular data of this size
#: routinely fits its training set to within a couple of points, so a gap of
#: five is already worth explaining.
OVERFIT_GAP = 0.05


def default_model() -> HistGradientBoostingClassifier:
    """The diagnostic's stand-in model: fast, and handles NaN natively."""
    return HistGradientBoostingClassifier(random_state=settings.random_seed)


def generalisation_gap(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Score the same fitted model on its training rows and on held-out stars."""
    model = clone(estimator).fit(X_train, y_train)
    train_auc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    test_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    gap = train_auc - test_auc
    return {
        "train_roc_auc": train_auc,
        "test_roc_auc": test_auc,
        "gap": gap,
        "threshold": OVERFIT_GAP,
        "overfit": bool(gap > OVERFIT_GAP),
        "n_train_rows": int(len(y_train)),
        "n_test_rows": int(len(y_test)),
    }


def learning_curve(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fractions: tuple[float, ...] = FRACTIONS,
) -> list[dict]:
    """Held-out score against training-set size, on one fixed held-out set.

    Subsampling is by *star*, not by row, for the same reason the splits are:
    dropping half the rows while keeping every star would leave each star's
    siblings partly present and understate how much the model leans on them.
    """
    rng = np.random.default_rng(settings.random_seed)
    stars = pd.unique(groups_train)
    rows: list[dict] = []

    for fraction in sorted(fractions):
        n_stars = max(2, int(round(len(stars) * fraction)))
        chosen = set(rng.choice(stars, size=n_stars, replace=False))
        mask = groups_train.isin(chosen).to_numpy()

        y_subset = y_train[mask]
        if y_subset.nunique() < 2:
            continue  # a single-class subsample cannot be scored

        model = clone(estimator).fit(X_train[mask], y_subset)
        rows.append(
            {
                "fraction": float(fraction),
                "n_train_stars": int(n_stars),
                "n_train_rows": int(mask.sum()),
                "n_test_rows": int(len(y_test)),
                "roc_auc": float(
                    roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                ),
            }
        )
    return rows
