"""Probability-quality metrics shared by training and evaluation.

Accuracy is nearly useless for this problem. A model that never predicts a draw
can still score ~53% -- roughly what the market itself achieves -- while being
worthless for betting, because value depends on the *probability*, not the
argmax. Everything here is therefore a proper scoring rule or a calibration
diagnostic, with accuracy reported only for context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTCOMES

# Guard against log(0) blowing up log loss on a confidently wrong prediction.
_EPS = 1e-15


def _one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def log_loss(y: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass cross-entropy. The primary metric for this project.

    Reference points for 3-way football: predicting the base rates alone scores
    about 1.05; a sharp bookmaker's closing line scores about 0.95-1.02.
    """
    proba = np.clip(proba, _EPS, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y)), y.astype(int)])))


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error summed over classes."""
    return float(np.mean(np.sum((proba - _one_hot(y, proba.shape[1])) ** 2, axis=1)))


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean(np.argmax(proba, axis=1) == y.astype(int)))


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE over the predicted-class confidence.

    Bins predictions by how confident the model was in its top choice, and
    measures the average gap between that confidence and how often it was right.
    """
    confidence = proba.max(axis=1)
    correct = (np.argmax(proba, axis=1) == y.astype(int)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_gap, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        total_gap += in_bin.sum() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total_gap / n)


def per_class_calibration(
    y: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability-diagram data, one row per (outcome, bin).

    Reported per class because the draw is the interesting failure case: models
    routinely under-predict draws, and a single pooled curve hides that.
    """
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    onehot = _one_hot(y, proba.shape[1])

    for k, outcome in enumerate(OUTCOMES):
        p_k = proba[:, k]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (p_k > lo) & (p_k <= hi)
            if not in_bin.any():
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": int(in_bin.sum()),
                    "mean_predicted": float(p_k[in_bin].mean()),
                    "observed_rate": float(onehot[in_bin, k].mean()),
                }
            )
    return pd.DataFrame(rows)


def summary(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """All scalar metrics at once, for logging to MLflow."""
    return {
        "log_loss": log_loss(y, proba),
        "brier": brier(y, proba),
        "accuracy": accuracy(y, proba),
        "ece": expected_calibration_error(y, proba, n_bins),
    }
