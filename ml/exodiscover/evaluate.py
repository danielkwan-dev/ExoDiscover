"""Evaluation metrics, calibration diagnostics, and figures.

Accuracy is deliberately absent from the headline set. The catalog is
imbalanced and the operational question is "which candidates are worth
follow-up time", so ranking quality (PR-AUC, precision@k) and probability
quality (Brier, reliability) are the metrics that mean anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from exodiscover.config import settings  # noqa: E402


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 50) -> float:
    """Fraction of the k highest-ranked items that are true positives.

    This is what a vetting team actually experiences: they work a finite
    shortlist, not the whole catalog.
    """
    y_true = np.asarray(y_true)
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    top = np.argsort(np.asarray(y_prob))[::-1][:k]
    return float(np.mean(y_true[top]))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Observed frequency against predicted probability, per bin."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    centers, observed, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centers.append(float(np.mean(y_prob[mask])))
        observed.append(float(np.mean(y_true[mask])))
        counts.append(int(mask.sum()))
    return {"bin_centers": centers, "observed": observed, "counts": counts}


def binary_scores(y_true, y_prob) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def bootstrap_ci(
    y_true, y_prob, groups, *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict[str, Any]:
    """Confidence intervals for ROC-AUC and PR-AUC, resampling *stars*.

    Resampling rows would understate the interval: sibling KOIs on one star
    share stellar parameters, so they are not independent draws. Resampling
    whole host stars respects that dependence, which is the same reasoning that
    makes the splits grouped.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)

    unique_stars = np.unique(groups)
    index_by_star = {star: np.flatnonzero(groups == star) for star in unique_stars}
    rng = np.random.default_rng(seed)

    roc: list[float] = []
    pr: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique_stars, size=len(unique_stars), replace=True)
        idx = np.concatenate([index_by_star[s] for s in drawn])
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue  # a degenerate resample carries no information
        roc.append(roc_auc_score(sample_true, y_prob[idx]))
        pr.append(average_precision_score(sample_true, y_prob[idx]))

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "roc_auc": [float(np.percentile(roc, lo)), float(np.percentile(roc, hi))],
        "pr_auc": [float(np.percentile(pr, lo)), float(np.percentile(pr, hi))],
        "n_resamples": len(roc),
    }


def evaluate_binary(model, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        **binary_scores(y_test, y_prob),
        "precision_at_50": precision_at_k(np.asarray(y_test), y_prob, 50),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(
            y_test,
            y_pred,
            target_names=["false positive", "planet"],
            output_dict=True,
            zero_division=0,
        ),
        "n_test": int(len(y_test)),
    }


def write_metrics(payload: dict, path: Path | None = None) -> Path:
    target = path or settings.metrics_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def plot_all(y_true, y_prob, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    written: list[Path] = []

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#4f46e5")
    ax.axhline(y_true.mean(), ls="--", color="grey", lw=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall")
    ax.legend(loc="lower left", fontsize=8)
    path = out_dir / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    curve = reliability_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.plot(curve["bin_centers"], curve["observed"], marker="o", color="#4f46e5")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration")
    ax.legend(loc="upper left", fontsize=8)
    path = out_dir / "reliability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["FP", "planet"])
    ax.set_yticks([0, 1], ["FP", "planet"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix")
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written
