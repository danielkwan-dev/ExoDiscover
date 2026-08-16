"""Light-curve preprocessing: the four stages that turn raw photometry into
model input.

Ported from the original ``lightcurve_preprocessing.py``, which was a linear
script requiring a network download. These are pure functions over arrays, so
they are testable offline and reusable by the API's explainer endpoint.

The pipeline is:

1. **normalize** - divide by the median so flux is dimensionless and
   comparable across stars and instruments.
2. **flatten** - remove slow stellar variability with a Savitzky-Golay filter,
   leaving the short-timescale dips a transit produces.
3. **segment** - cut into fixed-length overlapping windows, the shape a
   sequence model expects.
4. **describe** - summary statistics used for classical models and for the UI.

.. warning::
   The cached window arrays shipped with the original project record no star
   identity, so they cannot support a leakage-free evaluation. See
   ``docs/MODEL_CARD.md``. These functions are used for visualisation and
   explanation only; no light-curve classifier is shipped.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

DEFAULT_WINDOW = 256
DEFAULT_STRIDE = 128


def normalize(flux: np.ndarray) -> np.ndarray:
    """Scale flux to a median of 1, dropping non-finite samples."""
    flux = np.asarray(flux, dtype=float)
    flux = flux[np.isfinite(flux)]
    if flux.size == 0:
        return flux
    median = np.median(flux)
    if median == 0:
        return flux
    return flux / median


def flatten(flux: np.ndarray, window_length: int = 401, polyorder: int = 2) -> np.ndarray:
    """Divide out slow trends with a Savitzky-Golay filter.

    Stellar rotation and instrumental drift live on timescales far longer than
    a transit, so dividing by the smoothed curve preserves dips while removing
    the baseline they sit on.
    """
    flux = np.asarray(flux, dtype=float)
    # savgol_filter requires an odd window no longer than the series.
    window_length = min(window_length, len(flux))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        return flux
    trend = signal.savgol_filter(flux, window_length, polyorder)
    trend = np.where(trend == 0, 1.0, trend)
    return flux / trend


def segment(
    flux: np.ndarray, window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE
) -> np.ndarray:
    """Cut into fixed-length overlapping windows, shape ``(n_windows, window)``."""
    flux = np.asarray(flux, dtype=float)
    if len(flux) < window:
        return np.empty((0, window))
    starts = range(0, len(flux) - window + 1, stride)
    return np.array([flux[i : i + window] for i in starts])


def preprocess(flux: np.ndarray) -> dict[str, np.ndarray]:
    """Run all stages, returning each one for display."""
    normalized = normalize(flux)
    flattened = flatten(normalized)
    return {
        "raw": np.asarray(flux, dtype=float),
        "normalized": normalized,
        "flattened": flattened,
        "windows": segment(flattened),
    }


def window_features(window: np.ndarray) -> dict[str, float]:
    """Summary statistics for a single window.

    Retains the statistical, dip-detection, frequency, and signal-to-noise
    features from the original ``extract_features``; the per-sample Python loop
    that computed transit run lengths is dropped, since it cost O(n) per window
    and duplicated ``dip_fraction``.
    """
    w = np.asarray(window, dtype=float)
    mean = float(np.mean(w))
    std = float(np.std(w))

    threshold = mean - 2 * std
    dips = w < threshold
    # Count falling edges: a 0 -> 1 transition starts a new dip.
    num_dips = int(np.sum(np.diff(np.concatenate(([0], dips.astype(int), [0]))) == 1))

    fft = np.fft.rfft(w)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(w))
    peaks, _ = signal.find_peaks(power, height=float(np.mean(power)))
    if len(peaks) > 0:
        primary = peaks[int(np.argmax(power[peaks]))]
        primary_period = float(1.0 / freqs[primary]) if freqs[primary] != 0 else 0.0
    else:
        primary_period = 0.0

    noise = float(np.var(np.diff(w)) / 2)

    # skew/kurtosis are undefined for a constant series and scipy warns about
    # catastrophic cancellation; a flat window has neither, by definition.
    has_variance = std > 0
    skewness = float(stats.skew(w)) if has_variance else 0.0
    kurtosis = float(stats.kurtosis(w)) if has_variance else 0.0

    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(w)),
        "range": float(np.ptp(w)),
        "num_dips": float(num_dips),
        "dip_fraction": float(np.mean(dips)),
        "max_dip_depth": float(mean - np.min(w)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "num_periods": float(len(peaks)),
        "primary_period": primary_period,
        "snr": float(np.var(w) / noise) if noise > 0 else 0.0,
        "smoothness": float(np.mean(np.abs(np.diff(w)))),
    }
