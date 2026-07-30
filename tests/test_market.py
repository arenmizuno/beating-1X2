"""Devigging tests.

The market probabilities are the project's benchmark, so an error here would
shift the one number every conclusion is measured against.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.market import devig_multiplicative, devig_shin, select_odds


def _implied(odds_frame):
    odds, _ = select_odds(odds_frame)
    return 1.0 / odds.to_numpy(dtype=float)


def test_devigged_probabilities_sum_to_one(odds_frame):
    implied = _implied(odds_frame)
    assert np.allclose(devig_multiplicative(implied).sum(axis=1), 1.0)
    probs, _ = devig_shin(implied)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_devig_removes_the_known_overround(odds_frame):
    """The fixture is built from true probabilities inflated by exactly 5%.

    Multiplicative devigging assumes the margin is spread proportionally, which
    is precisely how the fixture was constructed, so it must recover the
    original probabilities almost exactly.
    """
    implied = _implied(odds_frame)
    assert np.allclose(implied.sum(axis=1), 1.05, atol=1e-9)

    expected = np.array(
        [[0.45, 0.27, 0.28], [0.60, 0.25, 0.15], [0.20, 0.26, 0.54], [0.34, 0.33, 0.33]]
    )
    assert np.allclose(devig_multiplicative(implied), expected, atol=1e-9)


def test_shin_shifts_mass_away_from_longshots(odds_frame):
    """Shin's method exists because bookmakers load their margin onto longshots.

    Relative to proportional normalization it should therefore assign LESS
    probability to the least likely outcome of a lopsided match.
    """
    implied = _implied(odds_frame)
    multiplicative = devig_multiplicative(implied)
    shin, z = devig_shin(implied)

    # Row 1 is the most lopsided (0.60 / 0.25 / 0.15).
    longshot = np.argmin(multiplicative[1])
    assert shin[1, longshot] < multiplicative[1, longshot]
    assert np.all(z >= 0.0) and np.all(z < 1.0)


def test_shin_falls_back_when_there_is_no_margin():
    """A book with no overround makes Shin degenerate; it must not blow up."""
    fair = np.array([[0.45, 0.27, 0.28]])
    probs, z = devig_shin(fair)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()
    assert np.allclose(probs, fair, atol=1e-6)


def test_select_odds_prefers_pinnacle_closing(odds_frame):
    """PSC must win when present -- it is the sharpest published line and the
    benchmark the whole evaluation is built on."""
    frame = odds_frame.copy()
    frame["odds_AvgC_H"] = 2.0
    frame["odds_AvgC_D"] = 3.0
    frame["odds_AvgC_A"] = 4.0

    odds, source = select_odds(frame)
    assert (source == "PSC").all()
    assert np.allclose(odds["H"].to_numpy(), frame["odds_PSC_H"].to_numpy())


def test_select_odds_falls_back_when_pinnacle_missing(odds_frame):
    frame = odds_frame.copy()
    frame[["odds_PSC_H", "odds_PSC_D", "odds_PSC_A"]] = np.nan
    frame["odds_AvgC_H"], frame["odds_AvgC_D"], frame["odds_AvgC_A"] = 2.0, 3.5, 4.0

    _, source = select_odds(frame)
    assert (source == "AvgC").all()


@pytest.mark.parametrize("bad", [0.0, 1.0, -2.0])
def test_invalid_prices_are_not_selected(odds_frame, bad):
    """Odds at or below 1.0 are data errors, not prices."""
    frame = odds_frame.copy()
    frame.loc[0, ["odds_PSC_H", "odds_PSC_D", "odds_PSC_A"]] = bad
    _, source = select_odds(frame)
    assert source.isna().iloc[0]
