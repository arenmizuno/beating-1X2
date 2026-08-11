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

from src.dixon_coles import walk_forward_ratings
from src.features import (
    add_derived_form,
    add_lagged_rolling,
    attach_dixon_coles,
    attach_elo,
    build_feature_frame,
    build_long_frame,
    leakage_check,
    pivot_to_wide,
)


def _build_long(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """The same attach chain build_feature_frame runs internally, exposed here
    so tests can probe the intermediate `long` frame directly. Small
    refit/min-match values keep this fast on the compact synthetic fixture."""
    long = add_derived_form(attach_elo(add_lagged_rolling(build_long_frame(matches)), elo))
    dc_ratings = walk_forward_ratings(
        matches,
        refit_every_matches=20,
        min_matches_for_fit=20,
        xi=0.0018,
        l2_penalty=0.05,
        rho_bounds=(-0.3, 0.3),
    )
    return attach_dixon_coles(long, dc_ratings)


def test_rolling_features_use_only_prior_matches(matches, elo):
    """Recompute every populated 5-match form value by brute force.

    This deliberately does NOT call the pipeline's own helpers -- it filters by
    date and averages the tail directly, so it is an independent check rather
    than a restatement of the implementation.
    """
    long = _build_long(matches, elo)
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
    long = _build_long(matches, elo)
    baseline = pivot_to_wide(long, matches).set_index("match_id")

    target_id = baseline.index[len(baseline) // 2]
    tampered = matches.copy()
    mask = tampered["match_id"] == target_id
    tampered.loc[mask, ["fthg", "ftag"]] = [9, 0]
    tampered.loc[mask, ["home_xg", "away_xg"]] = [9.9, 0.1]
    tampered.loc[mask, "ftr"] = "H"

    long2 = _build_long(tampered, elo)
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
    long = _build_long(matches, elo)
    wide = pivot_to_wide(long, matches)

    # Sanity: the honest frame passes.
    leakage_check(wide, long, n_samples=50)

    leaky = wide.copy()
    populated = leaky["home_points_r5"].notna()
    leaky.loc[populated, "home_points_r5"] = leaky.loc[populated, "home_points_r5"] + 1.0

    with pytest.raises(RuntimeError, match="leakage_check failed"):
        leakage_check(leaky, long, n_samples=50)


def test_venue_split_form_uses_only_prior_same_venue_matches(matches, elo):
    """points_venue_r5 must equal the mean points over a team's last 5 prior
    matches AT THE SAME VENUE -- an independent brute-force recomputation of the
    (team, is_home) grouping in add_derived_form."""
    long = _build_long(matches, elo)
    wide = pivot_to_wide(long, matches)

    checked = 0
    for row in wide.itertuples(index=False):
        for side, is_home in (("home", 1), ("away", 0)):
            actual = getattr(row, f"{side}_points_venue_r5")
            if pd.isna(actual):
                continue
            team = getattr(row, f"{side}_team")
            prior = long[
                (long["team"] == team)
                & (long["date"] < row.date)
                & (long["is_home"] == is_home)
            ].sort_values(["date", "match_id"], kind="stable")
            expected = prior["points"].tail(5).mean()
            assert np.isclose(actual, expected), f"{row.match_id} {side} {team}"
            checked += 1

    assert checked > 20, "too few populated venue-split values to be meaningful"


def test_promoted_flag_marks_newly_appearing_teams():
    """is_promoted is NaN in the first season (no prior-season info to compare
    against), 0 for a returning team, and 1 for a team new to the league."""
    teams_by_season = {2018: ["A", "B", "C", "D"], 2019: ["A", "B", "C", "E"]}
    rows = []
    for season, teams in teams_by_season.items():
        start = pd.Timestamp(f"{season}-08-10")
        i = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append(
                    {
                        "match_id": f"E0_{season}_{home}_{away}",
                        "league": "E0",
                        "season": season,
                        "date": start + pd.Timedelta(days=3 * i),
                        "home_team": home,
                        "away_team": away,
                        "fthg": 1,
                        "ftag": 0,
                        "home_xg": 1.1,
                        "away_xg": 0.4,
                        "ftr": "H",
                    }
                )
                i += 1
    m = pd.DataFrame(rows)
    m["date"] = pd.to_datetime(m["date"])
    all_teams = sorted({t for teams in teams_by_season.values() for t in teams})
    elo = pd.DataFrame(
        {
            "team": all_teams,
            "league": "E0",
            "level": 1,
            "elo": [1500.0 + i for i in range(len(all_teams))],
            "from_date": pd.to_datetime("2017-01-01"),
            "to_date": pd.to_datetime("2030-01-01"),
        }
    )
    elo["from_date"] = elo["from_date"].astype("datetime64[ns]")
    elo["to_date"] = elo["to_date"].astype("datetime64[ns]")

    long = add_derived_form(attach_elo(add_lagged_rolling(build_long_frame(m)), elo))

    assert long[long["season"] == 2018]["is_promoted"].isna().all()
    new_team = long[(long["season"] == 2019) & (long["team"] == "E")]
    returning = long[(long["season"] == 2019) & (long["team"] == "A")]
    assert (new_team["is_promoted"] == 1.0).all()
    assert (returning["is_promoted"] == 0.0).all()


def test_dc_features_do_not_use_current_match_result(matches, elo):
    """Same tamper test as test_current_match_result_never_enters_its_own_features,
    applied to the Dixon-Coles columns: changing one match's score must not move
    that match's own dc_* feature row, since the rating snapshot it looks up was
    fit using only strictly earlier matches."""
    baseline = build_feature_frame(matches, elo, run_leakage_check=False).set_index("match_id")

    target_id = baseline.index[len(baseline) // 2]
    tampered = matches.copy()
    mask = tampered["match_id"] == target_id
    tampered.loc[mask, ["fthg", "ftag"]] = [9, 0]
    tampered.loc[mask, ["home_xg", "away_xg"]] = [9.9, 0.1]
    tampered.loc[mask, "ftr"] = "H"

    after = build_feature_frame(tampered, elo, run_leakage_check=False).set_index("match_id")

    dc_cols = [
        "home_dc_attack",
        "away_dc_attack",
        "home_dc_defense",
        "away_dc_defense",
        "dc_lambda_home",
        "dc_lambda_away",
        "dc_p_H",
        "dc_p_D",
        "dc_p_A",
    ]
    pd.testing.assert_series_equal(
        baseline.loc[target_id, dc_cols],
        after.loc[target_id, dc_cols],
        check_names=False,
    )


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
