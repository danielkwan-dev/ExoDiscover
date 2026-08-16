import numpy as np
import pandas as pd

from exodiscover.data.splits import grouped_train_test_split, make_cv


def _frame(n_stars: int = 200):
    rng = np.random.default_rng(0)
    rows = []
    for star in range(n_stars):
        for _ in range(rng.integers(1, 4)):
            rows.append({"kepid": star, "feat": rng.normal(), "y": int(star % 2)})
    df = pd.DataFrame(rows)
    return df[["feat"]], df["y"], df["kepid"]


def test_no_star_appears_in_both_train_and_test():
    X, y, groups = _frame()
    X_tr, X_te, _, _ = grouped_train_test_split(X, y, groups, test_size=0.2, seed=42)
    train_stars = set(groups.loc[X_tr.index])
    test_stars = set(groups.loc[X_te.index])
    assert train_stars & test_stars == set(), "a kepid spans the split"


def test_split_is_deterministic_under_a_fixed_seed():
    X, y, groups = _frame()
    a = grouped_train_test_split(X, y, groups, seed=42)[1].index.tolist()
    b = grouped_train_test_split(X, y, groups, seed=42)[1].index.tolist()
    assert a == b


def test_test_size_is_approximately_honoured():
    X, y, groups = _frame()
    _, X_te, _, _ = grouped_train_test_split(X, y, groups, test_size=0.2, seed=42)
    assert 0.13 <= len(X_te) / len(X) <= 0.27


def test_cv_folds_never_span_a_star():
    X, y, groups = _frame()
    for train_idx, test_idx in make_cv(5).split(X, y, groups):
        assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set()


def test_real_catalog_split_holds_out_whole_stars(koi_binary):
    X, y, groups = koi_binary
    X_tr, X_te, _, _ = grouped_train_test_split(X, y, groups)
    assert set(groups.loc[X_tr.index]) & set(groups.loc[X_te.index]) == set()
