"""Leakage regression tests -- the most important tests in this repo.

The entire project rests on the claim that no feature can see its own match. If
that ever silently breaks, every metric becomes meaningless and the headline
result ("the market beats us") could flip for entirely spurious reasons.

These tests do three distinct jobs:
  1. verify the shift-then-roll features really only use prior matches,
  2. verify Elo is read strictly before kickoff,
  3. verify the in-pipeline `leakage_check` actually FAILS on leaky input --
     a guard that never fires on a broken build is worse than no guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_lagged_rolling,
    attach_elo,
    build_feature_frame,
    build_long_frame,
    leakage_check,
    pivot_to_wide,
)


def test_rolling_features_use_only_prior_matches(matches, elo):
    """Recompute every populated 5-match form value by brute force.

    This deliberately does NOT call the pipeline's own helpers -- it filters by
    date and averages the tail directly, so it is an independent check rather
    than a restatement of the implementation.
    """
    long = attach_elo(add_lagged_rolling(build_long_frame(matches)), elo)
    wide = pivot_to_wide(long, matches)

    checked = 0
    for row in wide.itertuples(index=False):
        for side in ("home", "away"):
            actual = getattr(row, f"{side}_points_r5")
            if pd.isna(actual):
                continue
            team = getattr(row, f"{side}_team")
            prior = long[(long["team"] == team) & (long["date"] < row.date)]
            prior = prior.sort_values(["date", "match_id"], kind="stable")
            expected = prior["points"].tail(5).mean()
            assert np.isclose(actual, expected), f"{row.match_id} {side} {team}"
            checked += 1

    assert checked > 100, "too few populated values to be a meaningful test"


def test_current_match_result_never_enters_its_own_features(matches, elo):
    """Change one match's score and confirm its own feature row is unaffected.

    A feature that moved would mean the outcome had leaked into the predictors.
    Later matches for those teams SHOULD move -- that is legitimate history.
    """
    long = attach_elo(add_lagged_rolling(build_long_frame(matches)), elo)
    baseline = pivot_to_wide(long, matches).set_index("match_id")

    target_id = baseline.index[len(baseline) // 2]
    tampered = matches.copy()
    mask = tampered["match_id"] == target_id
    tampered.loc[mask, ["fthg", "ftag"]] = [9, 0]
    tampered.loc[mask, ["home_xg", "away_xg"]] = [9.9, 0.1]
    tampered.loc[mask, "ftr"] = "H"

    long2 = attach_elo(add_lagged_rolling(build_long_frame(tampered)), elo)
    after = pivot_to_wide(long2, tampered).set_index("match_id")

    feature_cols = [c for c in baseline.columns if c.endswith(("_r5", "_r10"))]
    pd.testing.assert_series_equal(
        baseline.loc[target_id, feature_cols],
        after.loc[target_id, feature_cols],
        check_names=False,
    )


def test_elo_is_read_before_kickoff(matches, elo):
    """The 2019-01-02 rating epoch must not be used for 2018 fixtures."""
    long = attach_elo(add_lagged_rolling(build_long_frame(matches)), elo)

    early = long[long["date"] < pd.Timestamp("2019-01-01")]
    late = long[long["date"] > pd.Timestamp("2019-06-01")]

    # Ratings are ~1500 in the first epoch and ~2000 in the second, so the value
    # alone identifies which epoch was selected.
    assert early["elo"].max() < 1600.0, "a future Elo epoch leaked into 2018 matches"
    assert late["elo"].min() >= 2000.0


def test_leakage_check_rejects_leaky_features(matches, elo):
    """The guard must fail when handed genuinely leaky values.

    Build a wide frame whose rolling column includes the CURRENT match, which is
    exactly the bug that rolling-then-shifting (instead of shifting-then-rolling)
    would introduce, and assert the check catches it.
    """
    long = attach_elo(add_lagged_rolling(build_long_frame(matches)), elo)
    wide = pivot_to_wide(long, matches)

    # Sanity: the honest frame passes.
    leakage_check(wide, long, n_samples=50)

    leaky = wide.copy()
    populated = leaky["home_points_r5"].notna()
    leaky.loc[populated, "home_points_r5"] = leaky.loc[populated, "home_points_r5"] + 1.0

    with pytest.raises(RuntimeError, match="leakage_check failed"):
        leakage_check(leaky, long, n_samples=50)


def test_scoring_mode_matches_training_path(matches, elo):
    """Features for an unplayed fixture must equal what the training path would
    produce for that same fixture.

    This is the training/serving skew guarantee. We hold out the final match,
    build features for it as an *unplayed* fixture, and require them to be
    identical to the features the ordinary pipeline computed when it was a
    completed match -- since in both cases only prior matches are visible.
    """
    full = build_feature_frame(matches, elo, run_leakage_check=False).set_index("match_id")

    last_id = matches.sort_values("date").iloc[-1]["match_id"]
    history = matches[matches["match_id"] != last_id]
    upcoming = matches[matches["match_id"] == last_id][
        ["match_id", "league", "season", "date", "home_team", "away_team"]
    ]

    scored = build_feature_frame(history, elo, upcoming=upcoming).set_index("match_id")

    assert scored.loc[last_id, "feature_complete"]
    feature_cols = [c for c in full.columns if c.endswith(("_r5", "_r10"))] + [
        "home_elo",
        "away_elo",
        "elo_diff",
    ]
    pd.testing.assert_series_equal(
        full.loc[last_id, feature_cols],
        scored.loc[last_id, feature_cols],
        check_names=False,
    )
