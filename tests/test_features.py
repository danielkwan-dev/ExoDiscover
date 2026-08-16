"""Feature tests.

The physics functions are validated against Kepler-10's published values, so a
wrong constant or exponent fails loudly instead of quietly degrading the model.
If one of these fails, fix the formula — never loosen the tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from exodiscover.features import tabular

# NASA Exoplanet Archive, Kepler-10 system:
#   R* = 1.065 Rsun, logg = 4.35 (cgs)  ->  rho* ~ 1.04 g/cm3
# Kepler-10 b:
#   P = 0.837495 d, Rp = 1.47 Rearth, b ~ 0.30
#   observed duration = 1.811 h, observed depth = 152 ppm, a/R* ~ 3.4
KEPLER10_LOGG = 4.35
KEPLER10_SRAD = 1.065
KEPLER10B_PERIOD = 0.837495
KEPLER10B_PRAD = 1.47
KEPLER10B_IMPACT = 0.30


def _kepler10_rho() -> np.ndarray:
    return tabular.stellar_density(np.array([KEPLER10_LOGG]), np.array([KEPLER10_SRAD]))


def test_stellar_density_matches_kepler10():
    assert _kepler10_rho()[0] == pytest.approx(1.04, rel=0.10)


def test_a_over_rstar_matches_kepler10b():
    a_r = tabular.a_over_rstar(np.array([KEPLER10B_PERIOD]), _kepler10_rho())
    assert a_r[0] == pytest.approx(3.4, rel=0.10)


def test_expected_duration_matches_kepler10b():
    dur = tabular.expected_duration_hours(
        np.array([KEPLER10B_PERIOD]), _kepler10_rho(), np.array([KEPLER10B_IMPACT])
    )
    assert dur[0] == pytest.approx(1.811, rel=0.15)


def test_expected_depth_matches_kepler10b():
    depth = tabular.expected_depth_ppm(
        np.array([KEPLER10B_PRAD]), np.array([KEPLER10_SRAD])
    )
    assert depth[0] == pytest.approx(152.0, rel=0.20)


def test_denser_star_implies_shorter_transit():
    """A tighter orbit around a denser star crosses the disc faster."""
    period = np.array([10.0, 10.0])
    rho = np.array([0.5, 4.0])
    dur = tabular.expected_duration_hours(period, rho, np.array([0.0, 0.0]))
    assert dur[0] > dur[1]


def test_grazing_transit_yields_zero_duration_not_nan():
    dur = tabular.expected_duration_hours(np.array([10.0]), np.array([1.0]), np.array([1.5]))
    assert dur[0] == 0.0


def test_build_features_output_is_clean_and_aligned(koi_sample):
    X = tabular.build_features(koi_sample)
    assert len(X) == len(koi_sample)
    assert list(X.index) == list(koi_sample.index)
    assert list(X.columns) == tabular.FEATURE_COLUMNS


def test_build_features_strips_leakage_even_though_input_has_it(koi_sample):
    """The raw catalog contains koi_score and the fpflag family. The feature
    matrix must not, no matter what the input carries."""
    assert "koi_score" in koi_sample.columns, "fixture should exercise the guard"
    X = tabular.build_features(koi_sample)
    assert not set(X.columns) & {
        "koi_score",
        "koi_disposition",
        "koi_pdisposition",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
    }


def test_uncertainty_features_stay_excluded(koi_sample):
    """Regression guard on a deliberate modelling decision.

    rel_err_* encodes follow-up refinement: a confirmed planet's parameters
    were tightened by the observations that confirmed it, so the uncertainty
    reflects the label's consequence rather than the signal. Dropping them
    costs 0.003 ROC-AUC. See docs/LEAKAGE.md before reinstating them.
    """
    X = tabular.build_features(koi_sample)
    assert not set(X.columns) & set(tabular.EXCLUDED_UNCERTAINTY_FEATURES)
    assert not any(c.startswith("rel_err") for c in tabular.FEATURE_COLUMNS)


def test_build_features_emits_only_finite_values(koi_sample):
    X = tabular.build_features(koi_sample)
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_multiplicity_counts_kois_per_star():
    df = pd.DataFrame(
        {
            "kepid": [1, 1, 1, 2],
            "koi_period": [1.0, 2.0, 3.0, 4.0],
            "koi_srad": [1.0] * 4,
            "koi_slogg": [4.5] * 4,
            "koi_prad": [1.0] * 4,
            "koi_depth": [100.0] * 4,
            "koi_duration": [2.0] * 4,
            "koi_impact": [0.3] * 4,
        }
    )
    X = tabular.build_features(df)
    assert X["n_kois_on_star"].tolist() == [3.0, 3.0, 3.0, 1.0]


def test_missing_optional_columns_do_not_crash():
    """TESS frames lack several KOI columns; the builder must tolerate that."""
    df = pd.DataFrame(
        {
            "kepid": [1],
            "koi_period": [3.0],
            "koi_depth": [500.0],
            "koi_prad": [2.0],
            "koi_duration": [2.0],
            "koi_srad": [1.0],
            "koi_slogg": [4.4],
        }
    )
    X = tabular.build_features(df)
    assert list(X.columns) == tabular.FEATURE_COLUMNS
    assert np.isfinite(X.to_numpy(dtype=float)).all()
