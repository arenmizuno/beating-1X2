"""Stage 4: build the leakage-safe, pre-kickoff feature table.

Leakage is the one failure mode that would invalidate this entire project, so
the construction here is deliberately paranoid.

The central rule is **shift, then roll**. Every rolling statistic is computed on
a series that has already been lagged by one match within its team, so a team's
feature row for match N can only ever see matches 1..N-1. Rolling first and
lagging afterwards would produce the same column name and a subtly leaky column,
which is exactly the kind of bug that survives code review, so the lag is
applied as its own explicit, named step.

Three specific hazards, and how each is handled:

  xG is measured after the match.  It enters only through the lagged rolling
      means, never as a same-match value. The raw home_xg/away_xg columns are
      dropped from the output entirely so they cannot be selected by accident.

  Elo ratings move after results.  ClubElo publishes ratings as [from, to]
      validity ranges. We look up the rating in force on the day *before*
      kickoff, so a rating that already absorbed the match result can never be
      selected.

  Season boundaries.  Rolling windows deliberately span seasons -- a team's form
      in May is informative in August, and resetting would discard five matches
      per team per season. Promoted clubs simply have no prior top-flight
      history and are dropped by the min_prior_matches filter until they do.

`leakage_check` re-derives a random sample of feature values from scratch, using
only rows strictly earlier than the match in question, and asserts they match
the pipeline output. It runs as part of the stage, not as an optional test.

Output: data/processed/features.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import dixon_coles
from src.config import (
    FEATURES_PATH,
    INTERIM_DIR,
    MARKET_PATH,
    MATCHES_PATH,
    OUTCOME_TO_IDX,
    PARAMS,
    SEED,
    ensure_dirs,
    get_logger,
)

log = get_logger("features")

CLUBELO_CANONICAL_PATH = INTERIM_DIR / "clubelo_canonical.parquet"

_F = PARAMS["features"]
FORM_WINDOWS: list[int] = _F["form_windows"]
MIN_PRIOR_MATCHES: int = _F["min_prior_matches"]
MAX_REST_DAYS: int = _F["max_rest_days"]
LEAKAGE_SAMPLE: int = _F["leakage_check_sample"]

_DC = PARAMS["dixon_coles"]
DC_XI: float = _DC["xi"]
DC_L2_PENALTY: float = _DC["l2_penalty"]
DC_REFIT_EVERY_MATCHES: int = _DC["refit_every_matches"]
DC_MIN_MATCHES_FOR_FIT: int = _DC["min_matches_for_fit"]
DC_SCORE_GRID_MAX: int = _DC["score_grid_max"]
DC_RHO_BOUNDS: tuple[float, float] = tuple(_DC["rho_bounds"])

# Per-match team statistics that get lagged and rolled.
ROLLING_STATS = ["points", "goals_for", "goals_against", "xg_for", "xg_against"]


# ---------------------------------------------------------------------------
# Long team-match frame
# ---------------------------------------------------------------------------
def build_long_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) -- two rows per fixture.

    Working in long form is what makes the lag correct: a team's history is a
    single contiguous series regardless of whether it played home or away.
    """
    base = ["match_id", "league", "season", "date"]

    home = matches[base + ["home_team", "away_team", "fthg", "ftag", "home_xg", "away_xg"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "fthg": "goals_for",
            "ftag": "goals_against",
            "home_xg": "xg_for",
            "away_xg": "xg_against",
        }
    )
    home["is_home"] = 1

    away = matches[base + ["away_team", "home_team", "ftag", "fthg", "away_xg", "home_xg"]].copy()
    away.columns = base + ["team", "opponent", "goals_for", "goals_against", "xg_for", "xg_against"]
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long["goals_for"] > long["goals_against"], long["goals_for"] == long["goals_against"]],
        [3, 1],
        default=0,
    )
    return long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def add_lagged_rolling(long: pd.DataFrame) -> pd.DataFrame:
    """Add lagged rolling means, rest days, and prior-match counts.

    Step 1 lags every statistic by one match within its team. Step 2 rolls the
    *already lagged* series. Keeping these as two visible steps is the whole
    point -- see the module docstring.
    """
    out = long.copy()
    grouped = out.groupby("team", sort=False)

    # --- Step 1: lag ------------------------------------------------------
    for stat in ROLLING_STATS:
        out[f"_lagged_{stat}"] = grouped[stat].shift(1)

    # --- Step 2: roll the lagged series -----------------------------------
    lagged_groups = out.groupby("team", sort=False)
    for stat in ROLLING_STATS:
        for window in FORM_WINDOWS:
            rolled = (
                lagged_groups[f"_lagged_{stat}"]
                .rolling(window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
            out[f"{stat}_r{window}"] = rolled

    # --- Rest days and history depth --------------------------------------
    previous_date = grouped["date"].shift(1)
    rest = (out["date"] - previous_date).dt.days
    # A team back from the summer break is not meaningfully "90 days rested".
    out["rest_days"] = rest.clip(upper=MAX_REST_DAYS)

    out["n_prior_matches"] = grouped.cumcount()
    out["n_prior_in_season"] = out.groupby(["team", "season"], sort=False).cumcount()

    return out.drop(columns=[f"_lagged_{s}" for s in ROLLING_STATS])


# ---------------------------------------------------------------------------
# Elo as of the day before kickoff
# ---------------------------------------------------------------------------
def attach_elo(long: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Look up each team's Elo rating in force the day BEFORE the match.

    ClubElo rows are [from_date, to_date] validity ranges. Using the day before
    kickoff guarantees we never pick up a rating that has already absorbed the
    result we are trying to predict.
    """
    left = long.copy()
    left["_asof_date"] = left["date"] - pd.Timedelta(days=1)
    left = left.sort_values("_asof_date", kind="stable")

    right = elo[["team", "elo", "from_date"]].sort_values("from_date", kind="stable")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_asof_date",
        right_on="from_date",
        by="team",
        direction="backward",
    )
    merged = merged.drop(columns=["_asof_date", "from_date"])
    return merged.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def attach_dixon_coles(long: pd.DataFrame, dc_ratings: pd.DataFrame) -> pd.DataFrame:
    """Look up each team's Dixon-Coles rating in force the day BEFORE the match.

    Same as-of-kickoff pattern as `attach_elo`: `dc_ratings` rows are
    [from_date, +inf) validity snapshots (see `dixon_coles.walk_forward_ratings`),
    so this can never pick up a snapshot fit using the very match it is
    labeling, or any match after it.
    """
    left = long.copy()
    left["_asof_date"] = left["date"] - pd.Timedelta(days=1)
    left = left.sort_values("_asof_date", kind="stable")

    right = (
        dc_ratings[["team", "attack", "defense", "home_advantage", "rho", "from_date"]]
        .rename(
            columns={
                "attack": "dc_attack",
                "defense": "dc_defense",
                "home_advantage": "dc_home_advantage",
                "rho": "dc_rho",
            }
        )
        .sort_values("from_date", kind="stable")
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="_asof_date",
        right_on="from_date",
        by="team",
        direction="backward",
    )
    merged = merged.drop(columns=["_asof_date", "from_date"])
    return merged.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Back to one row per match
# ---------------------------------------------------------------------------
def _feature_columns() -> list[str]:
    cols = [
        "elo",
        "rest_days",
        "n_prior_matches",
        "n_prior_in_season",
        "dc_attack",
        "dc_defense",
        "dc_home_advantage",
        "dc_rho",
    ]
    for stat in ROLLING_STATS:
        for window in FORM_WINDOWS:
            cols.append(f"{stat}_r{window}")
    return cols


def pivot_to_wide(long: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _feature_columns()

    home = long[long["is_home"] == 1][["match_id"] + feature_cols]
    home = home.rename(columns={c: f"home_{c}" for c in feature_cols})

    away = long[long["is_home"] == 0][["match_id"] + feature_cols]
    away = away.rename(columns={c: f"away_{c}" for c in feature_cols})

    spine = matches[["match_id", "league", "season", "date", "home_team", "away_team", "ftr"]].copy()
    wide = spine.merge(home, on="match_id", validate="one_to_one")
    wide = wide.merge(away, on="match_id", validate="one_to_one")

    # --- Differential features -------------------------------------------
    # Trees can derive these themselves, but logistic regression cannot, and
    # they are the quantities football analytics actually reasons about.
    wide["elo_diff"] = wide["home_elo"] - wide["away_elo"]
    wide["rest_diff"] = wide["home_rest_days"] - wide["away_rest_days"]

    for window in FORM_WINDOWS:
        wide[f"points_diff_r{window}"] = (
            wide[f"home_points_r{window}"] - wide[f"away_points_r{window}"]
        )
        # Net xG: attacking output minus what the team conceded, home minus away.
        home_net_xg = wide[f"home_xg_for_r{window}"] - wide[f"home_xg_against_r{window}"]
        away_net_xg = wide[f"away_xg_for_r{window}"] - wide[f"away_xg_against_r{window}"]
        wide[f"xg_diff_r{window}"] = home_net_xg - away_net_xg

        home_net_goals = wide[f"home_goals_for_r{window}"] - wide[f"home_goals_against_r{window}"]
        away_net_goals = wide[f"away_goals_for_r{window}"] - wide[f"away_goals_against_r{window}"]
        wide[f"goal_diff_r{window}"] = home_net_goals - away_net_goals

        # xG minus actual goals is the standard regression-to-mean signal:
        # a team overperforming its xG is usually about to stop.
        wide[f"xg_overperformance_r{window}"] = (
            home_net_goals - home_net_xg
        ) - (away_net_goals - away_net_xg)

    # --- Dixon-Coles differentials and standalone match probabilities -------
    wide["dc_attack_diff"] = wide["home_dc_attack"] - wide["away_dc_attack"]
    wide["dc_defense_diff"] = wide["home_dc_defense"] - wide["away_dc_defense"]

    # home_advantage and rho are global-per-fit, not per-team, so both sides of
    # a match must resolve to the identical snapshot value. A mismatch means
    # the two as-of lookups landed on different checkpoints -- a real bug.
    dixon_coles.assert_home_away_consistent(wide, "dc_home_advantage")
    dixon_coles.assert_home_away_consistent(wide, "dc_rho")

    lambda_home, lambda_away = dixon_coles.expected_goals(
        wide["home_dc_attack"].to_numpy(),
        wide["home_dc_defense"].to_numpy(),
        wide["away_dc_attack"].to_numpy(),
        wide["away_dc_defense"].to_numpy(),
        wide["home_dc_home_advantage"].to_numpy(),
    )
    wide["dc_lambda_home"] = lambda_home
    wide["dc_lambda_away"] = lambda_away
    wide["dc_goal_diff_expected"] = lambda_home - lambda_away

    # dc_p_H/D/A is the Dixon-Coles model's OWN standalone 1X2 probability.
    # It is reported downstream (evaluate.py) as a third benchmark alongside
    # market and model, and is deliberately EXCLUDED from the ML feature set
    # (see feature_columns() in train.py) -- feeding a model its own
    # comparison benchmark as an input would make "model beats Dixon-Coles"
    # close to tautological.
    p_h, p_d, p_a = dixon_coles.match_probabilities(
        lambda_home, lambda_away, wide["home_dc_rho"].to_numpy(), max_goals=DC_SCORE_GRID_MAX
    )
    wide["dc_p_H"], wide["dc_p_D"], wide["dc_p_A"] = p_h, p_d, p_a

    wide["target"] = wide["ftr"].map(OUTCOME_TO_IDX)
    return wide


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------
def leakage_check(wide: pd.DataFrame, long: pd.DataFrame, n_samples: int = LEAKAGE_SAMPLE) -> None:
    """Re-derive sampled features from scratch using only strictly prior matches.

    This is an independent reimplementation, not a re-run of the pipeline code:
    it filters the long frame by date and averages the tail directly. If the
    shift-then-roll logic ever regresses, this fails loudly.
    """
    rng = np.random.default_rng(SEED)
    window = FORM_WINDOWS[0]
    checked = 0
    failures: list[str] = []

    # Only sample rows that actually have a value to compare against.
    candidates = wide[wide[f"home_points_r{window}"].notna()]
    if candidates.empty:
        raise RuntimeError("leakage_check found no populated feature rows to verify")

    sample_size = min(n_samples, len(candidates))
    sample = candidates.iloc[rng.choice(len(candidates), size=sample_size, replace=False)]

    # NOTE: written as an explicit comprehension on purpose. ruff's C416 wants
    # dict(...) here, but a pandas GroupBy does not convert that way -- it
    # raises TypeError rather than yielding {name: group}.
    long_by_team = {team: chunk for team, chunk in long.groupby("team", sort=False)}  # noqa: C416

    for row in sample.itertuples(index=False):
        for side in ("home", "away"):
            team = getattr(row, f"{side}_team")
            history = long_by_team.get(team)
            if history is None:
                continue

            prior = history[history["date"] < row.date].sort_values(
                ["date", "match_id"], kind="stable"
            )
            if len(prior) < window:
                continue

            expected = prior["points"].tail(window).mean()
            actual = getattr(row, f"{side}_points_r{window}")

            if pd.isna(actual) or not np.isclose(expected, actual, atol=1e-9):
                failures.append(
                    f"{row.match_id} {side}={team}: pipeline={actual!r} recomputed={expected!r}"
                )
            checked += 1

    if failures:
        for failure in failures[:10]:
            log.error("  LEAKAGE CHECK FAILED: %s", failure)
        raise RuntimeError(
            f"leakage_check failed on {len(failures)}/{checked} team-match values"
        )

    log.info("leakage_check passed: %d team-match values re-derived independently", checked)


# ---------------------------------------------------------------------------
# Shared feature construction (used by BOTH training and serving)
# ---------------------------------------------------------------------------
def build_feature_frame(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    *,
    upcoming: pd.DataFrame | None = None,
    run_leakage_check: bool = True,
) -> pd.DataFrame:
    """Build the wide feature table from completed matches, optionally scoring
    unplayed fixtures with the identical code path.

    When `upcoming` is supplied it is appended to the completed-match history as
    rows whose outcome columns are missing, the ordinary pipeline runs over the
    combined frame, and only the appended rows are returned.

    This is the single most important design decision in the serving layer:
    **inference does not reimplement feature engineering.** Recomputing rolling
    form and Elo separately for live fixtures is the classic source of
    training/serving skew, and here it would quietly void the leakage guarantees
    the whole project rests on.

    Appending is safe precisely because the feature logic is shift-then-roll: it
    only ever looks backward, so an unplayed fixture draws on prior completed
    matches and contributes nothing to any earlier row.

    One caveat, handled explicitly: if a team appears in more than one unplayed
    fixture, the second one's rolling window would reach back through the first
    -- whose result is unknown -- and yield missing features. Those rows are
    flagged via `feature_complete` rather than silently imputed into nonsense.
    """
    scoring_mode = upcoming is not None and len(upcoming) > 0

    if scoring_mode:
        history = matches.copy()
        pending = upcoming.copy()
        # Outcome columns are unknown for unplayed fixtures. They must be absent
        # rather than zero-filled, so nothing downstream can mistake them for
        # observed results.
        for col in ("fthg", "ftag", "home_xg", "away_xg", "ftr"):
            pending[col] = np.nan
        combined = pd.concat([history, pending], ignore_index=True)
        log.info(
            "scoring mode: %d completed matches + %d unplayed fixtures",
            len(history),
            len(pending),
        )
    else:
        combined = matches

    dc_ratings = dixon_coles.walk_forward_ratings(
        combined,
        refit_every_matches=DC_REFIT_EVERY_MATCHES,
        min_matches_for_fit=DC_MIN_MATCHES_FOR_FIT,
        xi=DC_XI,
        l2_penalty=DC_L2_PENALTY,
        rho_bounds=DC_RHO_BOUNDS,
    )

    long = build_long_frame(combined)
    log.info("long team-match frame: %d rows", len(long))

    long = add_lagged_rolling(long)
    long = attach_elo(long, elo)
    log.info("Elo coverage on team-matches: %.2f%%", 100 * long["elo"].notna().mean())

    long = attach_dixon_coles(long, dc_ratings)
    log.info(
        "Dixon-Coles coverage on team-matches: %.2f%%", 100 * long["dc_attack"].notna().mean()
    )

    wide = pivot_to_wide(long, combined)

    if scoring_mode:
        wide = wide[wide["match_id"].isin(upcoming["match_id"])].copy()
        feature_cols = [c for c in wide.columns if c.endswith(tuple(f"_r{w}" for w in FORM_WINDOWS))]
        wide["feature_complete"] = wide[feature_cols + ["home_elo", "away_elo"]].notna().all(axis=1)
        incomplete = int((~wide["feature_complete"]).sum())
        if incomplete:
            log.warning(
                "%d/%d fixtures have incomplete features (team likely appears in "
                "more than one unplayed fixture, or lacks top-flight history)",
                incomplete,
                len(wide),
            )
        log.info("built features for %d unplayed fixtures", len(wide))
        return wide

    log.info("wide feature table before filtering: %d rows", len(wide))
    if run_leakage_check:
        # Verify BEFORE filtering, so the check sees the widest possible sample.
        leakage_check(wide, long)
    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    ensure_dirs()

    matches = pd.read_parquet(MATCHES_PATH)
    elo = pd.read_parquet(CLUBELO_CANONICAL_PATH)
    market = pd.read_parquet(MARKET_PATH)
    log.info("loaded matches=%d elo_rows=%d market=%d", len(matches), len(elo), len(market))

    wide = build_feature_frame(matches, elo)

    # --- Filters -----------------------------------------------------------
    enough_history = (wide["home_n_prior_matches"] >= MIN_PRIOR_MATCHES) & (
        wide["away_n_prior_matches"] >= MIN_PRIOR_MATCHES
    )
    log.info(
        "dropping %d rows with fewer than %d prior matches for either side",
        int((~enough_history).sum()),
        MIN_PRIOR_MATCHES,
    )
    wide = wide[enough_history]

    # Attach the market probabilities. Left join: a match with no closing odds
    # is still a valid training row, it just cannot be evaluated for value.
    wide = wide.merge(market, on="match_id", how="left", validate="one_to_one")
    log.info("market coverage after join: %.2f%%", 100 * wide["p_market_H"].notna().mean())

    feature_cols = [
        c
        for c in wide.columns
        if c.startswith(
            ("home_", "away_", "elo_", "points_diff", "xg_diff", "goal_diff", "xg_over", "rest_", "dc_")
        )
        and c not in ("home_team", "away_team")
    ]
    missing_rates = wide[feature_cols].isna().mean().sort_values(ascending=False)
    if (missing_rates > 0).any():
        log.info("features with missing values:\n%s", (100 * missing_rates[missing_rates > 0]).round(2).to_string())

    wide = wide.sort_values(["date", "league"], kind="stable").reset_index(drop=True)
    wide.to_parquet(FEATURES_PATH, index=False)
    log.info("wrote %s (%d rows, %d columns)", FEATURES_PATH, len(wide), wide.shape[1])

    per_season = wide.groupby("season").size()
    log.info("rows per season:\n%s", per_season.to_string())

    base_rates = wide["ftr"].value_counts(normalize=True).reindex(["H", "D", "A"])
    log.info("base rates: H=%.1f%% D=%.1f%% A=%.1f%%", *(100 * base_rates.values))
    return wide


if __name__ == "__main__":
    main()
