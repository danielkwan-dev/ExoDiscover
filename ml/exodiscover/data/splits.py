"""Star-grouped splitting.

Kepler holds 9,564 KOI rows across only 8,214 stars, with up to seven KOIs on
a single star. A random row split puts sibling KOIs from the same star on both
sides, so every split in this project is grouped on ``kepid``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out whole stars, honouring ``test_size`` exactly.

    This was one fold of a grouped K-fold, which can only produce fractions of
    the form ``1/k``: a 70/30 split was unreachable, and a request for 0.4 came
    back as 0.5 because ``round(2.5)`` is 2. GroupShuffleSplit takes the
    fraction directly.

    The trade is that scikit-learn has no splitter that stratifies *and*
    groups at an arbitrary ratio. Grouping is the constraint that must not
    bend -- a star spanning the split leaks sibling KOIs across it, which is
    the error this project exists to avoid -- so grouping is exact and balance
    is left to fall where it does. With 6,639 stars it lands within a point of
    the overall rate; test_class_balance_survives_the_split holds that line.
    """
    fraction = settings.test_size if test_size is None else test_size
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"test_size must lie in (0, 1), got {fraction}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed or settings.random_seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
