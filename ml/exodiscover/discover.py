"""Ranks unvetted KOI candidates by calibrated planet probability.

This is the project's actual output. The classifier is trained on resolved
dispositions only, so the 1,979 CANDIDATE rows are genuinely unseen: scoring
them produces a shortlist a vetting team could work, each row carrying the
reasons the model produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exodiscover.explain import Contribution, explain_rows, unwrap
from exodiscover.features.tabular import build_features

RANKING_COLUMNS = [
    "kepoi_name",
    "kepler_name",
    "kepid",
    "probability",
    "score",
    "rank",
    "top_reasons",
]


def _ranking_score(model, X: pd.DataFrame) -> np.ndarray:
    """Fine-grained score used only for ordering, never for display."""
    base = unwrap(model)
    if base is not model and hasattr(base, "predict_proba"):
        return np.asarray(base.predict_proba(X)[:, 1])
    return np.asarray(model.predict_proba(X)[:, 1])


def _format_reasons(contributions: list[Contribution]) -> str:
    return "; ".join(
        f"{c['feature']}={c['value']:.3g} ({c['shap']:+.3f})" for c in contributions
    )


def rank_candidates(model, koi: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Score every CANDIDATE row and return the top_n by probability."""
    pool = koi[koi["koi_disposition"] == "CANDIDATE"].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    X = build_features(pool)
    probability = model.predict_proba(X)[:, 1]

    # Rank on the uncalibrated score, report the calibrated one.
    #
    # Isotonic regression is a step function: it maps every candidate in its top
    # bin to the same value, so the calibrated probabilities tie (15 of the top
    # 50 land on exactly 1.0). Ties make the ordering arbitrary precisely where
    # the ordering matters most. The base model's raw score is strictly finer
    # grained and monotonically related to it, so it breaks the ties correctly
    # without changing which candidates are at the top.
    score = _ranking_score(model, X)
    order = score.argsort()[::-1][:top_n]
    top = pool.iloc[order].reset_index(drop=True)
    X_top = X.iloc[order].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "kepoi_name": top.get("kepoi_name", pd.Series(dtype=object)),
            "kepler_name": top.get("kepler_name", pd.Series(dtype=object)),
            "kepid": top["kepid"],
            "probability": probability[order],
            # Reported alongside the calibrated value because isotonic ties at
            # the top: the raw score is what actually distinguishes these rows.
            "score": score[order],
        }
    )
    out["rank"] = range(1, len(out) + 1)
    out["top_reasons"] = [_format_reasons(c) for c in explain_rows(model, X_top, top_n=3)]
    return out[RANKING_COLUMNS]


def validate_against_snapshot(ranked: pd.DataFrame, newer_koi: pd.DataFrame) -> dict:
    """Check how many ranked candidates a later archive snapshot confirms.

    This is the honest test of the shortlist: not "does the model score well"
    but "did the things it flagged turn out to be planets".
    """
    status = newer_koi.drop_duplicates("kepoi_name").set_index("kepoi_name")["koi_disposition"]
    checked = ranked[ranked["kepoi_name"].isin(status.index)]
    n = int(len(checked))
    confirmed = int((status.loc[checked["kepoi_name"]] == "CONFIRMED").sum()) if n else 0
    return {
        "n_checked": n,
        "n_since_confirmed": confirmed,
        "hit_rate": float(confirmed / n) if n else 0.0,
    }
