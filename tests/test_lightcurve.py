import numpy as np
import pytest

from exodiscover.features import lightcurve as lc


def _with_transit(n=1024, depth=0.01, period=200, width=10):
    flux = np.ones(n)
    for start in range(50, n - width, period):
        flux[start : start + width] -= depth
    return flux


def test_normalize_sets_median_to_one():
    out = lc.normalize(np.array([100.0, 200.0, 300.0]))
    assert np.median(out) == pytest.approx(1.0)


def test_normalize_drops_non_finite_samples():
    out = lc.normalize(np.array([1.0, np.nan, 2.0, np.inf]))
    assert len(out) == 2
    assert np.isfinite(out).all()


def test_flatten_removes_a_linear_trend_but_keeps_the_dip():
    n = 1024
    trend = np.linspace(1.0, 1.5, n)
    flux = trend.copy()
    flux[500:510] -= 0.02
    flat = lc.flatten(flux)
    # Away from the dip the flattened curve should sit near 1.
    assert np.median(np.delete(flat, slice(495, 515))) == pytest.approx(1.0, abs=0.01)
    # The dip must survive.
    assert flat[505] < 0.995


def test_flatten_handles_a_series_shorter_than_the_window():
    out = lc.flatten(np.ones(50))
    assert len(out) == 50
    assert np.isfinite(out).all()


def test_segment_shape_and_overlap():
    windows = lc.segment(np.arange(1000.0), window=256, stride=128)
    assert windows.shape[1] == 256
    assert windows.shape[0] == len(range(0, 1000 - 256 + 1, 128))
    # 50% overlap: the second half of window 0 equals the first half of window 1.
    assert np.array_equal(windows[0][128:], windows[1][:128])


def test_segment_returns_empty_when_series_is_too_short():
    assert lc.segment(np.ones(10), window=256).shape == (0, 256)


def test_preprocess_returns_all_four_stages():
    stages = lc.preprocess(_with_transit())
    assert set(stages) == {"raw", "normalized", "flattened", "windows"}
    assert stages["windows"].shape[1] == lc.DEFAULT_WINDOW


def test_window_features_detect_a_synthetic_transit():
    flux = np.ones(256)
    flux[100:110] = 0.99
    feats = lc.window_features(flux)
    assert feats["num_dips"] >= 1
    assert feats["max_dip_depth"] == pytest.approx(0.01, abs=0.005)


def test_flat_lightcurve_has_no_dips():
    feats = lc.window_features(np.ones(256))
    assert feats["num_dips"] == 0
    assert feats["dip_fraction"] == 0.0


def test_deeper_transit_reports_greater_depth():
    shallow = lc.window_features(np.concatenate([np.ones(240), np.full(16, 0.995)]))
    deep = lc.window_features(np.concatenate([np.ones(240), np.full(16, 0.95)]))
    assert deep["max_dip_depth"] > shallow["max_dip_depth"]
