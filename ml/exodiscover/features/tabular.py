"""Physics-informed features for KOI classification.

The discriminative power here comes from consistency checks rather than raw
values. A genuine planetary transit must satisfy relationships between period,
stellar density, duration, and depth; eclipsing binaries and blended
background stars violate them in characteristic ways.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exodiscover.data.schema import LEAKY_COLUMNS, assert_no_leakage

G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
R_SUN_CM = 6.957e10
R_EARTH_CM = 6.371e8
SEC_PER_DAY = 86_400.0

FEATURE_COLUMNS: list[str] = [
    "log_period",
    "log_depth",
    "log_prad",
    "log_insol",
    "koi_impact",
    "koi_model_snr",
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "koi_tce_plnt_num",
    "rho_star",
    "a_over_rstar",
    "duration_ratio",
    "depth_ratio",
    "n_kois_on_star",
]

#: Deliberately excluded. Relative measurement uncertainty (err/value) looked
#: like legitimate signal, and the plan originally included it. The
#: Confirmed-vs-Candidate diagnostic showed otherwise: rel_err_depth was the
#: single strongest feature separating confirmed planets from candidates
#: (mean|SHAP| 0.93, with the three rel_err_* terms totalling 1.75 - more than
#: transit physics or brightness/multiplicity selection). The reason is
#: temporal: a confirmed planet's parameters were *refined by the follow-up
#: observations that confirmed it*, so the uncertainty encodes the label's
#: consequence rather than the signal's nature. Dropping them costs the binary
#: model 0.003 ROC-AUC (0.9865 -> 0.9835, as measured when the decision was
#: taken). See docs/LEAKAGE.md.
EXCLUDED_UNCERTAINTY_FEATURES: list[str] = [
    "rel_err_period",
    "rel_err_depth",
    "rel_err_duration",
]


def stellar_density(logg_cgs: np.ndarray, srad_rsun: np.ndarray) -> np.ndarray:
    """Mean stellar density in g/cm^3.

    From ``g = GM/R^2`` and ``rho = M / (4/3 pi R^3)`` it follows that
    ``rho = 3g / (4 pi G R)``.
    """
    g = np.power(10.0, np.asarray(logg_cgs, dtype=float))
    r = np.asarray(srad_rsun, dtype=float) * R_SUN_CM
    return 3.0 * g / (4.0 * np.pi * G_CGS * r)


def a_over_rstar(period_days: np.ndarray, rho_star: np.ndarray) -> np.ndarray:
    """Scaled semi-major axis, ``a/R* = (G rho P^2 / 3 pi)^(1/3)``.

    This is Kepler's third law rewritten so that only the stellar density and
    the orbital period are needed — neither the stellar mass nor the radius
    appears on its own.
    """
    p = np.asarray(period_days, dtype=float) * SEC_PER_DAY
    return np.cbrt(G_CGS * np.asarray(rho_star, dtype=float) * p**2 / (3.0 * np.pi))


def expected_duration_hours(
    period_days: np.ndarray, rho_star: np.ndarray, impact: np.ndarray
) -> np.ndarray:
    """Small-angle transit duration, ``T = (P / pi) * sqrt(1 - b^2) / (a/R*)``.

    A grazing or geometrically impossible transit (``b >= 1``) yields 0 rather
    than NaN, so downstream ratios stay finite.
    """
    a_r = a_over_rstar(period_days, rho_star)
    chord = np.sqrt(np.clip(1.0 - np.asarray(impact, dtype=float) ** 2, 0.0, None))
    return np.asarray(period_days, dtype=float) * 24.0 / (np.pi * a_r) * chord


def expected_depth_ppm(prad_earth: np.ndarray, srad_rsun: np.ndarray) -> np.ndarray:
    """``(Rp / R*)^2`` expressed in parts per million."""
    k = (np.asarray(prad_earth, dtype=float) * R_EARTH_CM) / (
        np.asarray(srad_rsun, dtype=float) * R_SUN_CM
    )
    return k**2 * 1e6


MULTIPLICITY_COLUMN = "koi_multiplicity"


def add_multiplicity(koi: pd.DataFrame) -> pd.DataFrame:
    """Attach the KOI count per host star, computed over the *whole* catalog.

    This must happen before any label-based filtering. Counting within a
    filtered subset would make a star that hosts one confirmed planet and two
    unvetted candidates look like a single-planet system during training, while
    a user scoring that same star would correctly report three — a train/serve
    skew. Multi-planet systems are rarely false positives, so the count carries
    real signal and has to mean the same thing in both places.
    """
    out = koi.copy()
    if "kepid" in out.columns:
        out[MULTIPLICITY_COLUMN] = out.groupby("kepid")["kepid"].transform("size").astype(float)
    else:
        out[MULTIPLICITY_COLUMN] = 1.0
    return out


def _numeric(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    """Fetch a column as floats, tolerating its absence."""
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _safe_log10(x: pd.Series) -> pd.Series:
    return np.log10(x.clip(lower=1e-9))


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def build_features(koi: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from a raw KOI frame.

    Leaky columns are dropped on the way in and the result is asserted clean on
    the way out, so an engineered feature can never reintroduce one under a
    surviving name.
    """
    df = koi.drop(columns=list(LEAKY_COLUMNS), errors="ignore")
    out = pd.DataFrame(index=df.index)

    period = _numeric(df, "koi_period")
    depth = _numeric(df, "koi_depth")
    prad = _numeric(df, "koi_prad")
    duration = _numeric(df, "koi_duration")

    out["log_period"] = _safe_log10(period)
    out["log_depth"] = _safe_log10(depth)
    out["log_prad"] = _safe_log10(prad)
    # No default. A default of 1.0 here meant an absent koi_insol column became
    # log_insol 0.0 while a column holding null became NaN -- the same object
    # scored two ways depending on how the caller expressed "I don't have this".
    out["log_insol"] = _safe_log10(_numeric(df, "koi_insol"))

    for col in (
        "koi_impact",
        "koi_model_snr",
        "koi_teq",
        "koi_steff",
        "koi_slogg",
        "koi_srad",
        "koi_kepmag",
        "koi_tce_plnt_num",
    ):
        out[col] = _numeric(df, col)

    rho = stellar_density(out["koi_slogg"].to_numpy(), out["koi_srad"].to_numpy())
    out["rho_star"] = rho
    out["a_over_rstar"] = a_over_rstar(period.to_numpy(), rho)

    expected_dur = expected_duration_hours(
        period.to_numpy(), rho, out["koi_impact"].fillna(0.0).to_numpy()
    )
    out["duration_ratio"] = _safe_ratio(duration, pd.Series(expected_dur, index=df.index))

    expected_dep = expected_depth_ppm(prad.to_numpy(), out["koi_srad"].to_numpy())
    out["depth_ratio"] = _safe_ratio(depth, pd.Series(expected_dep, index=df.index))

    # Multiplicity must be counted over the whole catalog by add_multiplicity
    # and handed in. Counting within whatever frame arrives here would make the
    # answer depend on the batch: the same star reads as 1 KOI when a row is
    # scored alone and as 5 when scored beside its siblings. Unknown means NaN,
    # which the model handles, rather than a 1.0 that asserts a lone KOI.
    out["n_kois_on_star"] = _numeric(df, MULTIPLICITY_COLUMN)

    # Infinities are arithmetic accidents -- a division by a zero radius -- and
    # become NaN. Genuine NaN is left alone: filling it here used the median of
    # whatever frame happened to be passed, which made a row's features depend
    # on its batch (0.0 when served alone, the catalog median when served with
    # the catalog) and computed test-fold statistics from the test fold itself.
    # Imputation belongs to the estimator, where it is fitted on training data
    # only and travels with the persisted model. See models/registry.py.
    out = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    assert_no_leakage(out, context="feature matrix")
    return out
