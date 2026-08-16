import json

import numpy as np

from exodiscover import evaluate


def test_precision_at_50_counts_only_the_top_ranked():
    y_true = np.array([1] * 50 + [0] * 950)
    y_prob = np.concatenate([np.linspace(0.99, 0.90, 50), np.full(950, 0.10)])
    assert evaluate.precision_at_k(y_true, y_prob, k=50) == 1.0


def test_precision_at_k_handles_k_larger_than_n():
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8])
    assert evaluate.precision_at_k(y_true, y_prob, k=50) == 2 / 3


def test_precision_at_k_penalises_a_bad_ranking():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.linspace(0.99, 0.90, 50), np.full(50, 0.10)])
    assert evaluate.precision_at_k(y_true, y_prob, k=50) == 0.0


def test_reliability_curve_is_diagonal_for_a_perfect_forecaster():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, 20_000)
    y_true = (rng.uniform(0, 1, 20_000) < y_prob).astype(int)
    curve = evaluate.reliability_curve(y_true, y_prob, n_bins=10)
    assert np.allclose(curve["bin_centers"], curve["observed"], atol=0.05)


def test_reliability_curve_detects_an_overconfident_forecaster():
    y_prob = np.full(1000, 0.95)
    y_true = np.zeros(1000, dtype=int)
    curve = evaluate.reliability_curve(y_true, y_prob, n_bins=10)
    assert curve["observed"][0] == 0.0
    assert curve["bin_centers"][0] > 0.9


def test_write_metrics_round_trips(tmp_path):
    path = evaluate.write_metrics({"roc_auc": 0.87}, tmp_path / "metrics.json")
    assert json.loads(path.read_text())["roc_auc"] == 0.87


def test_plot_all_writes_three_figures(tmp_path):
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, 200)
    y_true = (y_prob > 0.5).astype(int)
    written = evaluate.plot_all(y_true, y_prob, tmp_path)
    assert len(written) == 3
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
