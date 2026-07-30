"""Scoring-rule tests, checked against hand-computed values.

These are the numbers every conclusion in the writeup is stated in, so they are
verified against arithmetic done by hand rather than against another library.
"""

from __future__ import annotations

import numpy as np

from src import metrics


def test_log_loss_matches_hand_computation():
    y = np.array([0, 1, 2])
    proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    expected = -np.mean([np.log(0.5), np.log(0.6), np.log(0.6)])
    assert np.isclose(metrics.log_loss(y, proba), expected)


def test_log_loss_is_finite_for_a_confidently_wrong_prediction():
    """A zero on the true class must be clipped, not produce inf.

    This is exactly the isotonic-calibration failure that phase 1 uncovered.
    """
    y = np.array([0])
    proba = np.array([[0.0, 0.5, 0.5]])
    assert np.isfinite(metrics.log_loss(y, proba))


def test_perfect_predictions_score_zero():
    y = np.array([0, 1, 2])
    proba = np.eye(3)
    assert metrics.log_loss(y, proba) < 1e-10
    assert metrics.brier(y, proba) < 1e-10


def test_brier_matches_hand_computation():
    y = np.array([0])
    proba = np.array([[0.7, 0.2, 0.1]])
    expected = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert np.isclose(metrics.brier(y, proba), expected)


def test_base_rate_predictions_score_near_the_entropy_of_the_base_rates():
    """Predicting the base rates should score close to their entropy (~1.05 for
    football's 44/25/31 split). This is the reference point the README quotes."""
    rng = np.random.default_rng(0)
    base = np.array([0.44, 0.25, 0.31])
    y = rng.choice(3, size=20000, p=base)
    proba = np.tile(base, (len(y), 1))
    assert np.isclose(metrics.log_loss(y, proba), -(base * np.log(base)).sum(), atol=0.02)


def test_accuracy_uses_the_argmax():
    y = np.array([0, 1])
    proba = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    assert np.isclose(metrics.accuracy(y, proba), 0.5)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    """Half the samples correct at a constant 0.5 confidence -> zero gap."""
    y = np.array([0, 1, 0, 1] * 50)
    proba = np.tile([0.5, 0.5, 0.0], (len(y), 1))
    assert metrics.expected_calibration_error(y, proba, n_bins=10) < 1e-9


def test_ece_detects_overconfidence():
    """Always claiming 0.9 while being right half the time is a 0.4 gap."""
    y = np.array([0, 1] * 100)
    proba = np.tile([0.9, 0.1, 0.0], (len(y), 1))
    assert np.isclose(metrics.expected_calibration_error(y, proba, n_bins=10), 0.4, atol=0.01)


def test_per_class_calibration_covers_every_outcome():
    rng = np.random.default_rng(1)
    y = rng.choice(3, size=500)
    proba = rng.dirichlet([2, 2, 2], size=500)
    table = metrics.per_class_calibration(y, proba, n_bins=10)
    assert set(table["outcome"]) == {"H", "D", "A"}
    assert (table["n"] > 0).all()


def test_summary_reports_every_metric():
    y = np.array([0, 1, 2])
    proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    assert set(metrics.summary(y, proba)) == {"log_loss", "brier", "accuracy", "ece"}
