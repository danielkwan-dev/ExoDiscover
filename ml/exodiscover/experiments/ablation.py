"""Quantifies the failure modes found in the original pipeline.

Two studies live here. The leakage study isolates the effect of the
Robovetter columns and of splitting rows instead of stars. The framing study
asks whether the three-class target is the right one at all, by scoring every
framing on the single decision they all make in common.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from exodiscover.config import settings
from exodiscover.data.splits import grouped_train_test_split
from exodiscover.evaluate import binary_scores, precision_at_k
from exodiscover.features.tabular import (
    MULTIPLICITY_COLUMN,
    add_multiplicity,
    build_features,
)

#: The Robovetter's own verdict columns, re-added only to measure their effect.
LEAKY_EXTRAS = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


def collapse_to_planetlike(y: np.ndarray) -> np.ndarray:
    """Map three-class labels onto the planet-like vs false-positive decision.

    Confirmed and Candidate both mean "the vetting pipeline believes this is a
    planet", so both collapse to 1. This is what makes the three-class and
    binary framings comparable.
    """
    return (np.asarray(y) > 0).astype(int)


def with_multiplicity(koi: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-star KOI count unless the caller already has.

    Counting must happen over the whole catalog and before any label filter, so
    a caller that has done it properly -- `exo train` reads the catalog through
    add_multiplicity -- keeps its own count rather than having it recomputed
    over whatever subset arrives here.
    """
    return koi if MULTIPLICITY_COLUMN in koi.columns else add_multiplicity(koi)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features for an ablation, refusing a matrix with an empty column.

    build_features leaves missing inputs as NaN rather than inventing values,
    which means a column absent from the source frame arrives entirely empty.
    That is always a caller error, and it surfaces a long way from its cause:
    HistGradientBoosting's binner needs two distinct values and otherwise
    raises "window shape cannot be larger than input array shape" from inside
    numpy's stride tricks. Naming the column here is cheaper than tracing that.
    """
    X = build_features(frame)
    empty = [column for column in X.columns if X[column].notna().sum() == 0]
    if empty:
        raise ValueError(
            f"feature(s) {', '.join(empty)} are entirely missing for these "
            f"{len(X)} rows, so no model can bin them. The source frame is "
            "missing the columns they derive from, or the per-star count was "
            "never attached -- see with_multiplicity."
        )
    return X


def _model() -> HistGradientBoostingClassifier:
    # Handles NaN natively, so the leaky columns can be fed in unimputed.
    return HistGradientBoostingClassifier(random_state=settings.random_seed)


def _binary_frame(koi: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].reset_index(drop=True)
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df, y


def _fit_score(X_tr, X_te, y_tr, y_te) -> dict[str, float]:
    model = _model().fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        **binary_scores(y_te, y_prob),
        "precision_at_50": precision_at_k(np.asarray(y_te), y_prob, 50),
        "n_test": int(len(y_te)),
    }


def run_leakage_ablation(koi: pd.DataFrame) -> list[dict]:
    """Three setups, one variable changed at a time."""
    koi = with_multiplicity(koi)
    df, y = _binary_frame(koi)
    clean = prepare_features(df)

    leaky = clean.copy()
    for col in LEAKY_EXTRAS:
        if col in df.columns:
            leaky[col] = pd.to_numeric(df[col], errors="coerce")

    seed = settings.random_seed
    rows: list[dict] = []

    X_tr, X_te, y_tr, y_te = train_test_split(
        leaky, y, test_size=0.2, random_state=seed, stratify=y
    )
    rows.append(
        {
            "setup": "leaky_random_split",
            **_fit_score(X_tr, X_te, y_tr, y_te),
            "note": "koi_score and the fpflag family present, rows split at random "
            "(reproduces the original methodology)",
        }
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        clean, y, test_size=0.2, random_state=seed, stratify=y
    )
    rows.append(
        {
            "setup": "clean_random_split",
            **_fit_score(X_tr, X_te, y_tr, y_te),
            "note": "leakage removed, but sibling KOIs from one star still span the split",
        }
    )

    X_tr, X_te, y_tr, y_te = grouped_train_test_split(clean, y, df["kepid"])
    rows.append(
        {
            "setup": "clean_grouped_split",
            **_fit_score(X_tr, X_te, y_tr, y_te),
            "note": "leakage removed and whole stars held out - the honest number",
        }
    )

    return rows


def run_framing_ablation(koi: pd.DataFrame) -> list[dict]:
    """Score all three task framings on the planet-like decision."""
    koi = with_multiplicity(koi)
    rows: list[dict] = []

    # A: flat three-class, collapsed at inference so it is comparable.
    df3 = koi[koi["koi_disposition"].isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])]
    df3 = df3.reset_index(drop=True)
    y3 = df3["koi_disposition"].map({"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2})
    X3 = prepare_features(df3)
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(X3, y3, df3["kepid"])
    model = _model().fit(X_tr, y_tr)
    prob_planetlike = model.predict_proba(X_te)[:, 1:].sum(axis=1)
    rows.append(
        {
            "framing": "3class",
            "target": "FP / Candidate / Confirmed, collapsed to planet-like at inference",
            **binary_scores(collapse_to_planetlike(y_te), prob_planetlike),
            "n_train": int(len(y_tr)),
        }
    )

    # B: binary confirmed vs false positive; candidates withheld for discovery.
    dfb, yb = _binary_frame(koi)
    Xb = prepare_features(dfb)
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(Xb, yb, dfb["kepid"])
    model = _model().fit(X_tr, y_tr)
    rows.append(
        {
            "framing": "binary",
            "target": "Confirmed vs False Positive",
            **binary_scores(y_te, model.predict_proba(X_te)[:, 1]),
            "n_train": int(len(y_tr)),
        }
    )

    # C: diagnostic. If this separates well on brightness, multiplicity, and
    # period rather than transit shape, the third class is encoding follow-up
    # selection rather than physics.
    dfd = koi[koi["koi_disposition"].isin(["CONFIRMED", "CANDIDATE"])].reset_index(drop=True)
    yd = (dfd["koi_disposition"] == "CONFIRMED").astype(int)
    Xd = prepare_features(dfd)
    X_tr, X_te, y_tr, y_te = grouped_train_test_split(Xd, yd, dfd["kepid"])
    model = _model().fit(X_tr, y_tr)
    rows.append(
        {
            "framing": "diagnostic",
            "target": "Confirmed vs Candidate (both already vetted as planet-like)",
            **binary_scores(y_te, model.predict_proba(X_te)[:, 1]),
            "n_train": int(len(y_tr)),
        }
    )

    return rows
