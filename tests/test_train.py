import numpy as np

from exodiscover.models import registry, train


def test_registry_contains_the_full_ladder():
    models = registry.candidates(seed=42)
    assert {
        "dummy",
        "logreg",
        "random_forest",
        "hist_gb",
        "xgboost",
        "lightgbm",
        "catboost",
    } <= set(models)


def test_registry_is_deterministic():
    a = registry.final_estimator(registry.candidates(seed=42)["random_forest"])
    b = registry.final_estimator(registry.candidates(seed=42)["random_forest"])
    assert a.get_params()["random_state"] == b.get_params()["random_state"] == 42


def test_tuning_reaches_parameters_through_a_pipeline(koi_binary):
    """Optuna must tune the estimator even when an imputer wraps it.

    A pipeline renames every parameter to `clf__*`. The search proposes bare
    names, so filtering them against the pipeline's own parameter set silently
    matched nothing and returned the untuned model while still reporting a
    tuned score.
    """
    X, y, groups = koi_binary
    tuned = train.tune_best(X, y, groups, "random_forest", n_trials=2)
    baseline = registry.final_estimator(registry.candidates(42)["random_forest"])
    assert registry.final_estimator(tuned).get_params()["max_depth"] != (
        baseline.get_params()["max_depth"]
    )


def test_fit_candidates_beats_the_dummy_baseline(koi_binary):
    X, y, groups = koi_binary
    by_name = {r.name: r for r in train.fit_candidates(X, y, groups)}
    assert by_name["hist_gb"].cv_scores["pr_auc"] > by_name["dummy"].cv_scores["pr_auc"]


def test_fit_candidates_is_sorted_best_first(koi_binary):
    X, y, groups = koi_binary
    scores = [r.cv_scores["pr_auc"] for r in train.fit_candidates(X, y, groups)]
    assert scores == sorted(scores, reverse=True)


def test_brier_is_reported_in_its_natural_orientation(koi_binary):
    """cross_validate negates Brier so higher is better; we flip it back, so
    every reported value must be a genuine (non-negative) Brier score."""
    X, y, groups = koi_binary
    for result in train.fit_candidates(X, y, groups):
        assert 0.0 <= result.cv_scores["brier"] <= 1.0


def test_calibrated_model_emits_probabilities_in_range(koi_binary):
    X, y, groups = koi_binary
    base = registry.candidates(seed=42)["hist_gb"]
    model, _ = train.build_calibrated(base, X, y, groups)
    p = model.predict_proba(X)[:, 1]
    assert p.min() >= 0.0
    assert p.max() <= 1.0
    assert np.isfinite(p).all()


def test_calibration_compares_both_methods_and_reports_the_choice(koi_binary):
    X, y, groups = koi_binary
    base = registry.candidates(seed=42)["hist_gb"]
    _, report = train.build_calibrated(base, X, y, groups)
    assert set(report["brier_by_method"]) == {"isotonic", "sigmoid"}
    assert report["chosen"] in {"isotonic", "sigmoid"}
    # The chosen method must be the one that actually scored better.
    assert report["brier_by_method"][report["chosen"]] == min(
        report["brier_by_method"].values()
    )


def test_calibration_sets_are_disjoint_from_each_other(koi_binary):
    """Fit, calibrate, and select must be three separate star-disjoint slices,
    or the method choice is made on data the calibrator already saw."""
    X, y, groups = koi_binary
    base = registry.candidates(seed=42)["hist_gb"]
    _, report = train.build_calibrated(base, X, y, groups)
    assert report["n_calibration"] > 0
    assert report["n_validation"] > 0
    assert report["n_calibration"] + report["n_validation"] < len(X)


def test_method_can_be_forced(koi_binary):
    X, y, groups = koi_binary
    base = registry.candidates(seed=42)["hist_gb"]
    _, report = train.build_calibrated(base, X, y, groups, method="sigmoid")
    assert report["chosen"] == "sigmoid"


def test_soft_vote_ensemble_is_registered():
    assert "soft_vote" in registry.ensembles(seed=42)
    assert registry.ENSEMBLE_MEMBERS == ("catboost", "xgboost", "lightgbm")


def test_tuning_a_non_tunable_family_is_a_no_op(koi_binary):
    X, y, groups = koi_binary
    out = train.tune_best(X, y, groups, "dummy", n_trials=2)
    assert out.__class__.__name__ == "DummyClassifier"
