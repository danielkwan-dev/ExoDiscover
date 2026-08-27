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


class LeakageError(ValueError):
    """Raised when a target-encoding column reaches a feature matrix."""


def assert_no_leakage(df: pd.DataFrame, *, context: str = "dataframe") -> None:
    """Raise if any leaky column is present, naming every offender."""
    offenders = sorted(set(df.columns) & LEAKY_COLUMNS)
    if offenders:
        raise LeakageError(
            f"{len(offenders)} leaky column(s) present in {context}: "
            f"{', '.join(offenders)}. These encode the target; see docs/LEAKAGE.md."
        )
