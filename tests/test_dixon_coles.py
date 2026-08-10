"""Dixon-Coles rating tests.

Three jobs, mirroring test_features.py's leakage-regression philosophy:
  1. the fit recovers sensible relative strength and stays on the documented
     natural (not log) scale,
  2. time-decay actually changes the fitted ratings,
  3. the guard that protects `walk_forward_ratings`'s checkpoint boundary
     actually fires on a broken cutoff -- a guard that never fires on a broken
     build is worse than no guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dixon_coles import (
    assert_home_away_consistent,
    expected_goals,
    fit_ratings,
    match_probabilities,
    walk_forward_ratings,
)


def _matches(rows: list[tuple[str, str, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "home_team": h,
                "away_team": a,
                "fthg": hg,
                "ftag": ag,
                "date": pd.Timestamp(date),
                "league": "E0",
            }
            for h, a, hg, ag, date in rows
        ]
    )


def test_fit_ratings_recovers_relative_strength():
    """Strong scores heavily and concedes little; Weak does the opposite."""
    matches = _matches(
        [
            ("Strong", "Weak", 4, 0, "2020-01-01"),
            ("Weak", "Strong", 0, 3, "2020-01-02"),
            ("Strong", "Mid1", 3, 1, "2020-01-03"),
            ("Mid1", "Strong", 0, 2, "2020-01-04"),
            ("Strong", "Mid2", 3, 0, "2020-01-05"),
            ("Mid2", "Strong", 1, 3, "2020-01-06"),
            ("Weak", "Mid1", 0, 2, "2020-01-07"),
            ("Mid1", "Weak", 3, 0, "2020-01-08"),
            ("Weak", "Mid2", 1, 3, "2020-01-09"),
            ("Mid2", "Weak", 2, 0, "2020-01-10"),
            ("Mid1", "Mid2", 1, 1, "2020-01-11"),
            ("Mid2", "Mid1", 1, 1, "2020-01-12"),
        ]
    )
    cutoff = matches["date"].max() + pd.Timedelta(days=1)
    ratings = fit_ratings(matches, cutoff, xi=0.0, l2_penalty=0.02, rho_bounds=(-0.3, 0.3)).set_index(
        "team"
    )

    assert ratings.loc["Strong", "attack"] > ratings.loc["Weak", "attack"]
    # Lower "defense" is the stronger defense: lambda_away = away_attack * home_defense,
    # so a small defense value means opponents score less against this team.
    assert ratings.loc["Strong", "defense"] < ratings.loc["Weak", "defense"]


def test_ratings_are_natural_scale_not_log():
    matches = _matches(
        [
            ("A", "B", 2, 1, "2020-01-01"),
            ("B", "A", 0, 1, "2020-01-02"),
            ("A", "C", 1, 1, "2020-01-03"),
            ("C", "A", 2, 0, "2020-01-04"),
        ]
    )
    cutoff = matches["date"].max() + pd.Timedelta(days=1)
    ratings = fit_ratings(matches, cutoff, xi=0.001, l2_penalty=0.05, rho_bounds=(-0.3, 0.3))

    assert (ratings["attack"] > 0).all()
    assert (ratings["defense"] > 0).all()
    assert (ratings["home_advantage"] > 0).all()
    # A log-scale value fitted with a small L2 penalty would typically sit near
    # zero and could easily be negative; natural-scale strengths cluster near 1.
    assert ratings["attack"].between(0.1, 10).all()
    assert ratings["home_advantage"].between(0.1, 10).all()


def test_time_decay_downweights_old_matches():
    """An old blowout should matter less under strong decay than under none."""
    rows = [("TeamA", "Filler", 6, 0, "2015-01-01")]  # a distant blowout
    date = pd.Timestamp("2023-01-01")
    for i in range(8):  # recent, unremarkable form
        rows.append(("TeamA", "Filler", 1, 1, str((date + pd.Timedelta(days=i)).date())))
        rows.append(("Filler", "TeamA", 1, 1, str((date + pd.Timedelta(days=i + 1)).date())))
    matches = _matches(rows)
    cutoff = matches["date"].max() + pd.Timedelta(days=1)

    no_decay = fit_ratings(matches, cutoff, xi=0.0, l2_penalty=0.02, rho_bounds=(-0.3, 0.3))
    strong_decay = fit_ratings(matches, cutoff, xi=0.01, l2_penalty=0.02, rho_bounds=(-0.3, 0.3))

    a_no_decay = no_decay.set_index("team").loc["TeamA", "attack"]
    a_decay = strong_decay.set_index("team").loc["TeamA", "attack"]
    assert a_decay < a_no_decay, "the old blowout should count for less once it is decayed away"


def test_fit_ratings_rejects_a_match_after_its_own_cutoff():
    """The guard that protects walk_forward_ratings's checkpoint boundary."""
    matches = _matches([("A", "B", 1, 0, "2020-01-10")])
    cutoff = pd.Timestamp("2020-01-05")  # strictly before the match's own date
    with pytest.raises(RuntimeError, match="after its own cutoff"):
        fit_ratings(matches, cutoff, xi=0.001, l2_penalty=0.05, rho_bounds=(-0.3, 0.3))


def test_walk_forward_ratings_runs_on_realistic_data(matches):
    """End-to-end smoke test on the shared 8-team/2-season fixture."""
    ratings = walk_forward_ratings(
        matches,
        refit_every_matches=20,
        min_matches_for_fit=20,
        xi=0.0018,
        l2_penalty=0.05,
        rho_bounds=(-0.3, 0.3),
    )
    assert not ratings.empty
    assert set(ratings.columns) == {
        "team",
        "league",
        "attack",
        "defense",
        "home_advantage",
        "rho",
        "from_date",
    }
    assert (ratings["attack"] > 0).all()
    # Every snapshot's from_date must be strictly before the first match that
    # could ever look it up (that lookup itself happens at date - 1 day in
    # attach_dixon_coles, so this is a coarser but still meaningful sanity check).
    assert (ratings["from_date"] <= matches["date"].max()).all()


def test_expected_goals_and_match_probabilities_are_valid_distributions():
    home_attack = np.array([1.4, 0.8])
    home_defense = np.array([0.9, 1.1])
    away_attack = np.array([1.0, 1.2])
    away_defense = np.array([1.0, 0.9])
    home_advantage = np.array([1.3, 1.3])

    lambda_home, lambda_away = expected_goals(
        home_attack, home_defense, away_attack, away_defense, home_advantage
    )
    assert (lambda_home > 0).all()
    assert (lambda_away > 0).all()

    p_h, p_d, p_a = match_probabilities(lambda_home, lambda_away, rho=np.array([-0.1, 0.05]))
    totals = p_h + p_d + p_a
    assert np.allclose(totals, 1.0, atol=1e-6)
    for p in (p_h, p_d, p_a):
        assert (p >= 0).all()
        assert (p <= 1).all()


def test_assert_home_away_consistent_raises_on_mismatch():
    wide = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_dc_rho": [0.1, 0.1],
            "away_dc_rho": [0.1, 0.2],
        }
    )
    with pytest.raises(RuntimeError, match="disagree"):
        assert_home_away_consistent(wide, "dc_rho")


def test_assert_home_away_consistent_passes_when_equal():
    wide = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_dc_rho": [0.1, 0.1],
            "away_dc_rho": [0.1, 0.1],
        }
    )
    assert_home_away_consistent(wide, "dc_rho")  # must not raise
