"""Star-grouped splitting.

Kepler holds 9,564 KOI rows across only 8,214 stars, with up to seven KOIs on
a single star. A random row split puts sibling KOIs from the same star on both
sides, so every split in this project is grouped on ``kepid``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exodiscover.config import settings


def make_cv(n_splits: int | None = None) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=n_splits or settings.cv_folds,
        shuffle=True,
        random_state=settings.random_seed,
    )


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out whole stars.

    Implemented as one fold of a grouped K-fold so that stratification and
    grouping are honoured at the same time.
    """
    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed or settings.random_seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
