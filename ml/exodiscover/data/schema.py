"""Column contracts for the NASA exoplanet catalogs.

``LEAKY_COLUMNS`` is the single source of truth for features that encode the
target. ``koi_score`` is the Robovetter's own disposition score and the
``koi_fpflag_*`` family are its false-positive vetting decisions; a model
given any of them reproduces the label instead of learning the physics.
"""

from __future__ import annotations

import pandas as pd

LEAKY_COLUMNS: frozenset[str] = frozenset(
    {
        "koi_disposition",
        "koi_pdisposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_tce_delivname",
        "tfopwg_disp",
        "disposition",
    }
)

KOI_LABEL_MAP: dict[str, int] = {
    "FALSE POSITIVE": 0,
    "CANDIDATE": 1,
    "CONFIRMED": 2,
}

# TESS Follow-up Working Group dispositions.
#   CP  = Confirmed Planet          KP  = Known Planet (also confirmed)
#   PC  = Planet Candidate          APC = Ambiguous Planet Candidate
#   FP  = False Positive            FA  = False Alarm
# The original preprocess_merge.py mapped KP to Candidate and dropped APC/FA
# entirely; both errors are corrected here.
TOI_LABEL_MAP: dict[str, int] = {
    "CP": 2,
    "KP": 2,
    "PC": 1,
    "APC": 1,
    "FP": 0,
    "FA": 0,
}

#: TOI dispositions whose planet status is actually resolved. PC and APC are
#: unvetted, so they carry no ground truth and must not become labels.
TOI_RESOLVED: tuple[str, ...] = ("CP", "KP", "FP", "FA")


#: Highest rank correlation a shipped feature may have with any Robovetter
#: column. Measured on the full catalog, the strongest *legitimate* pairing is
#: log_prad against koi_fpflag_ss at 0.553 -- eclipsing binaries are both large
#: and flagged as such, so real physics and the vetting flag agree without
#: either causing the other. A renamed koi_score scores 1.000, and one halved
#: and mixed with noise still scores 0.885. 0.80 sits clear of the legitimate
#: ceiling and below both leaks.
MAX_LEAK_CORRELATION = 0.80


class LeakageError(ValueError):
    """Raised when a target-encoding column reaches a feature matrix."""


def assert_no_leakage(df: pd.DataFrame, *, context: str = "dataframe") -> None:
    """Raise if a leaky column is present *by name*, naming every offender.

    This catches the original defect -- hand-picked feature lists that included
    koi_score -- and nothing subtler. A leaky column renamed on the way in
    passes it, and because build_features restricts its output to
    FEATURE_COLUMNS, the check can only ever fire there if that list itself is
    edited. `assert_no_derived_leakage` is the one that inspects values.
    """
    offenders = sorted(set(df.columns) & LEAKY_COLUMNS)
    if offenders:
        raise LeakageError(
            f"{len(offenders)} leaky column(s) present in {context}: "
            f"{', '.join(offenders)}. These encode the target; see docs/LEAKAGE.md."
        )


def find_derived_leakage(
    features: pd.DataFrame,
    source: pd.DataFrame,
    *,
    threshold: float = MAX_LEAK_CORRELATION,
) -> list[tuple[str, str, float]]:
    """Feature/leaky-column pairs whose |Spearman| exceeds ``threshold``.

    Rank correlation rather than Pearson, because a leak reintroduced through a
    monotone transform -- a log, a rescaling, a rank itself -- is still a leak,
    and Spearman is blind to the transform in a way Pearson is not.

    Returns the offenders strongest-first rather than raising, so callers can
    report the whole picture; ``assert_no_derived_leakage`` wraps it.
    """
    offenders: list[tuple[str, str, float]] = []
    for name in sorted(LEAKY_COLUMNS & set(source.columns)):
        target = pd.to_numeric(source[name], errors="coerce")
        if target.nunique(dropna=True) < 2:
            continue  # constant or unparseable: no correlation to measure
        for feature in features.columns:
            values = pd.to_numeric(features[feature], errors="coerce")
            if values.nunique(dropna=True) < 2:
                continue
            rho = values.corr(target, method="spearman")
            if pd.notna(rho) and abs(rho) > threshold:
                offenders.append((str(feature), name, float(rho)))
    return sorted(offenders, key=lambda row: -abs(row[2]))


def assert_no_derived_leakage(
    features: pd.DataFrame,
    source: pd.DataFrame,
    *,
    threshold: float = MAX_LEAK_CORRELATION,
    context: str = "feature matrix",
) -> None:
    """Raise if any feature tracks a Robovetter column too closely.

    This is a training-time audit, not a serving check: it needs many rows to
    mean anything, and it costs a correlation per feature per leaky column.
    """
    offenders = find_derived_leakage(features, source, threshold=threshold)
    if offenders:
        detail = "; ".join(f"{feat} vs {leak} rho={rho:+.3f}" for feat, leak, rho in offenders)
        raise LeakageError(
            f"{len(offenders)} feature(s) in {context} correlate with a "
            f"target-encoding column above {threshold}: {detail}. A leaky column "
            "may have been reintroduced under another name; see docs/LEAKAGE.md."
        )
