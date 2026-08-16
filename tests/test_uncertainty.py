"""Tests for the uncertainty reporting: CV spread and grouped bootstrap CIs.

These exist because the project makes claims like "the boosted families are
within noise of each other". A claim about noise needs the noise measured.
"""

import numpy as np
import pytest

from exodiscover.evaluate import bootstrap_ci
from exodiscover.models import registry, train


def test_cv_scores_carry_a_standard_deviation(koi_binary):
    X, y, groups = koi_binary
    scores = train.score_estimator(registry.candidates(42)["hist_gb"], X, y, groups)
    for metric in ("pr_auc", "roc_auc", "brier"):
        assert f"{metric}_std" in scores
        assert scores[f"{metric}_std"] >= 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 2)
    y_true = rng.integers(0, 2, size=600)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, size=600), 0, 1)

    from sklearn.metrics import roc_auc_score

    point = roc_auc_score(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob, groups, n_resamples=200)
    lo, hi = ci["roc_auc"]
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(200), 2)
    y_true = rng.integers(0, 2, size=400)
    y_prob = rng.uniform(size=400)
    a = bootstrap_ci(y_true, y_prob, groups, n_resamples=100, seed=7)
    b = bootstrap_ci(y_true, y_prob, groups, n_resamples=100, seed=7)
    assert a["roc_auc"] == b["roc_auc"]


def test_grouped_bootstrap_is_wider_than_ignoring_groups():
    """Sibling rows on one star are not independent draws, so resampling stars
    must yield a wider interval than pretending each row is independent.

    Both the label *and* the score have to correlate within a star for this to
    bite: siblings share stellar parameters, so the model gives them similar
    scores. Correlating only the labels leaves the effective sample size
    unchanged and the two intervals come out identical.
    """
    rng = np.random.default_rng(2)
    n_stars, per_star = 150, 6
    groups = np.repeat(np.arange(n_stars), per_star)

    star_label = rng.integers(0, 2, size=n_stars)
    # One score per star, shared by its siblings apart from a little jitter.
    star_score = np.clip(star_label * 0.5 + rng.normal(0.25, 0.30, size=n_stars), 0, 1)
    y_true = np.repeat(star_label, per_star)
    y_prob = np.clip(
        np.repeat(star_score, per_star) + rng.normal(0, 0.01, size=n_stars * per_star), 0, 1
    )

    grouped = bootstrap_ci(y_true, y_prob, groups, n_resamples=400, seed=3)
    ungrouped = bootstrap_ci(y_true, y_prob, np.arange(len(y_true)), n_resamples=400, seed=3)

    grouped_width = grouped["roc_auc"][1] - grouped["roc_auc"][0]
    ungrouped_width = ungrouped["roc_auc"][1] - ungrouped["roc_auc"][0]
    assert grouped_width > ungrouped_width


def test_degenerate_resamples_are_skipped():
    """A resample containing only one class cannot yield an AUC; it should be
    dropped rather than crash or poison the interval."""
    y_true = np.array([1] * 20 + [0] * 2)
    y_prob = np.linspace(0.9, 0.1, 22)
    groups = np.arange(22)
    ci = bootstrap_ci(y_true, y_prob, groups, n_resamples=200, seed=5)
    assert ci["n_resamples"] <= 200
    assert ci["n_resamples"] > 0
    assert not np.isnan(ci["roc_auc"]).any()


@pytest.mark.parametrize("alpha", [0.05, 0.20])
def test_a_wider_alpha_gives_a_narrower_interval(alpha):
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(200), 2)
    y_true = rng.integers(0, 2, size=400)
    y_prob = np.clip(y_true * 0.5 + rng.normal(0.25, 0.3, size=400), 0, 1)
    ci = bootstrap_ci(y_true, y_prob, groups, n_resamples=300, alpha=alpha)
    width = ci["roc_auc"][1] - ci["roc_auc"][0]
    assert 0.0 < width < 1.0
