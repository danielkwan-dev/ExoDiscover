import pytest
from sklearn.calibration import CalibratedClassifierCV

from exodiscover import discover, explain
from exodiscover.data.splits import make_cv
from exodiscover.models.registry import candidates


@pytest.fixture
def fitted(koi_binary):
    X, y, _ = koi_binary
    return candidates(42)["lightgbm"].fit(X, y), X


def test_global_importance_is_sorted_and_complete(fitted):
    model, X = fitted
    rows = explain.global_importance(model, X)
    assert len(rows) == X.shape[1]
    values = [r["mean_abs_shap"] for r in rows]
    assert values == sorted(values, reverse=True)


def test_explain_row_returns_top_n_signed_contributions(fitted):
    model, X = fitted
    rows = explain.explain_row(model, X, index=0, top_n=6)
    assert len(rows) == 6
    assert set(rows[0]) == {"feature", "value", "shap"}
    magnitudes = [abs(r["shap"]) for r in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_rows_matches_explain_row(fitted):
    model, X = fitted
    batch = explain.explain_rows(model, X.iloc[:3], top_n=4)
    assert len(batch) == 3
    single = explain.explain_row(model, X.iloc[:3], index=1, top_n=4)
    assert [r["feature"] for r in batch[1]] == [r["feature"] for r in single]


def test_ensemble_shap_is_the_mean_of_member_shap(koi_binary):
    """Soft voting averages member probabilities, so the corresponding
    attribution is the mean of the members' attributions -- not an
    approximation. Pin that, since the explanation would otherwise silently
    become wrong if the combination rule changed."""
    import numpy as np

    from exodiscover.explain import _tree_shap, shap_matrix, unwrap
    from exodiscover.models.registry import ensembles

    X, y, _ = koi_binary
    model = ensembles(42)["soft_vote"].fit(X, y)

    combined = shap_matrix(model, X.iloc[:20])
    members = [_tree_shap(unwrap(est), X.iloc[:20]) for est in model.estimators_]
    assert np.allclose(combined, np.mean(members, axis=0))
    assert combined.shape == (20, X.shape[1])


def test_unwrap_reaches_the_tree_through_a_calibration_wrapper(koi_binary):
    X, y, groups = koi_binary
    folds = list(make_cv().split(X, y, groups))
    cal = CalibratedClassifierCV(candidates(42)["lightgbm"], cv=folds).fit(X, y)
    inner = explain.unwrap(cal)
    assert inner.__class__.__name__ == "LGBMClassifier"


def test_only_candidates_are_ranked(fitted, koi_sample):
    model, _ = fitted
    ranked = discover.rank_candidates(model, koi_sample, top_n=10)
    candidate_names = set(koi_sample[koi_sample["koi_disposition"] == "CANDIDATE"]["kepoi_name"])
    assert set(ranked["kepoi_name"]) <= candidate_names


def test_ranking_is_ordered_by_descending_probability(fitted, koi_sample):
    model, _ = fitted
    ranked = discover.rank_candidates(model, koi_sample, top_n=10)
    probs = ranked["probability"].tolist()
    assert probs == sorted(probs, reverse=True)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert all(ranked["top_reasons"].str.len() > 0)


def test_ranking_breaks_calibration_ties(koi_binary, koi_sample):
    """Isotonic calibration is a step function and ties candidates at the top.

    Ranking must use the finer-grained base score so the order is meaningful,
    while the reported probability stays the calibrated one.
    """
    from exodiscover.models import train as train_mod

    X, y, groups = koi_binary
    model, _ = train_mod.build_calibrated(candidates(42)["lightgbm"], X, y, groups)
    ranked = discover.rank_candidates(model, koi_sample, top_n=20)

    # Calibrated probabilities may tie; the ordering must still be stable and
    # non-increasing, and distinct rows must not be dropped.
    probs = ranked["probability"].tolist()
    assert probs == sorted(probs, reverse=True)
    assert ranked["kepoi_name"].nunique() == len(ranked)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_ranking_an_empty_pool_returns_an_empty_frame(fitted, koi_sample):
    model, _ = fitted
    no_candidates = koi_sample[koi_sample["koi_disposition"] != "CANDIDATE"]
    ranked = discover.rank_candidates(model, no_candidates, top_n=10)
    assert ranked.empty
    assert list(ranked.columns) == discover.RANKING_COLUMNS


def test_validation_counts_candidates_since_confirmed():
    import pandas as pd

    ranked = pd.DataFrame(
        {"kepoi_name": ["K1.01", "K2.01", "K3.01"], "probability": [0.9, 0.8, 0.7]}
    )
    newer = pd.DataFrame(
        {
            "kepoi_name": ["K1.01", "K2.01", "K3.01"],
            "koi_disposition": ["CONFIRMED", "CANDIDATE", "CONFIRMED"],
        }
    )
    result = discover.validate_against_snapshot(ranked, newer)
    assert result["n_checked"] == 3
    assert result["n_since_confirmed"] == 2
    assert result["hit_rate"] == pytest.approx(2 / 3)
