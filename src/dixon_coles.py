"""Dixon-Coles (1997) goal model, with Dixon & Robinson (1998) time-decay.

A second, independent rating source alongside ClubElo: instead of an external
win/loss/margin-based rating, this fits each team's attack and defense
strength directly from goal counts via a (weighted) Poisson maximum-likelihood
fit, with the low-score correlation correction Dixon & Coles introduced to fix
the small but real excess of 0-0/1-0/0-1/1-1 results independent Poisson
underpredicts.

Unlike Elo, there is no external service continuously publishing this rating,
so it has to be refit in-house. Refitting after every single match (the
maximally leakage-faithful choice) is computationally infeasible across
~14,000 matches, so `walk_forward_ratings` refits periodically instead,
producing a ratings-history table shaped exactly like the ClubElo table
`attach_elo` already consumes: one row per (team, snapshot), with a `from_date`
marking when that snapshot's fit used ONLY matches strictly on or before it.
`fit_ratings` asserts this itself -- a match dated after its own cutoff would
mean a rating snapshot silently absorbed a result it is later used to predict.

Two things a naive implementation gets wrong, both handled explicitly here:

  Identifiability.  attack_i * defense_j is invariant to rescaling attack up
      and defense down by the same factor. Rather than a hard sum-to-zero
      constraint (which needs constrained optimization), attack/defense are
      fit in log space with an L2 penalty toward zero -- this fixes the scale
      AND shrinks teams with few matches (new promotions) toward league-average
      strength, instead of letting a small sample swing to an extreme rating.

  Parameter scale.  attack/defense/home_advantage are stored on their NATURAL
      scale (already exponentiated), not the log scale the optimizer works in
      internally -- so `expected_goals` below is a plain multiplication, not a
      log-sum-exp, and there is exactly one place (the return of `fit_ratings`)
      where the scale conversion happens.

Output: a ratings-history table with columns
  team, league, attack, defense, home_advantage, rho, from_date
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.config import get_logger

log = get_logger("dixon_coles")

RATINGS_COLUMNS = ["team", "league", "attack", "defense", "home_advantage", "rho", "from_date"]


# ---------------------------------------------------------------------------
# One fit
# ---------------------------------------------------------------------------
def fit_ratings(
    matches: pd.DataFrame,
    cutoff_date,
    *,
    xi: float,
    l2_penalty: float,
    rho_bounds: tuple[float, float],
    warm_start: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weighted-MLE Dixon-Coles fit on one league's matches up to `cutoff_date`.

    `matches` must contain exactly one league's completed fixtures, each with
    home_team/away_team/fthg/ftag/date. Every match must be dated on or before
    `cutoff_date` -- this is asserted, not assumed, because it is the one
    invariant that makes the resulting snapshot leakage-safe to attach via
    `attach_dixon_coles`'s as-of-kickoff lookup.

    `warm_start`, when given the previous checkpoint's ratings for the same
    league, both speeds convergence and stabilizes the fit (teams seen before
    start near their last estimate rather than at league-average).
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    if (matches["date"] > cutoff_date).any():
        raise RuntimeError(
            "dixon_coles.fit_ratings received a match dated after its own cutoff "
            f"({cutoff_date.date()}) -- this would leak a future result into the "
            "rating snapshot"
        )

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = matches["home_team"].map(idx).to_numpy()
    away_idx = matches["away_team"].map(idx).to_numpy()
    fthg = matches["fthg"].to_numpy(dtype=float)
    ftag = matches["ftag"].to_numpy(dtype=float)

    # Exponential time-decay: matches on the cutoff date itself get full
    # weight (days_before=0), older matches count for less.
    days_before = (cutoff_date - matches["date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * days_before)

    theta0 = np.zeros(2 * n + 2)
    if warm_start is not None and len(warm_start):
        ws = warm_start.set_index("team")
        for t, i in idx.items():
            if t in ws.index:
                theta0[i] = np.log(ws.loc[t, "attack"])
                theta0[n + i] = np.log(ws.loc[t, "defense"])
        theta0[2 * n] = np.log(ws["home_advantage"].iloc[0])
        theta0[2 * n + 1] = ws["rho"].iloc[0]

    def unpack(theta):
        return theta[:n], theta[n : 2 * n], theta[2 * n], theta[2 * n + 1]

    def neg_log_likelihood(theta):
        a, d, log_gamma, rho = unpack(theta)
        lambda_home = np.exp(a[home_idx] + d[away_idx] + log_gamma)
        lambda_away = np.exp(a[away_idx] + d[home_idx])

        ll = (
            fthg * np.log(lambda_home)
            - lambda_home
            + ftag * np.log(lambda_away)
            - lambda_away
        )

        # Dixon-Coles low-score correction, applied only to the four cells
        # where independent Poisson measurably underfits football scorelines.
        tau = np.ones_like(ll)
        tau = np.where((fthg == 0) & (ftag == 0), 1 - lambda_home * lambda_away * rho, tau)
        tau = np.where((fthg == 1) & (ftag == 0), 1 + lambda_away * rho, tau)
        tau = np.where((fthg == 0) & (ftag == 1), 1 + lambda_home * rho, tau)
        tau = np.where((fthg == 1) & (ftag == 1), 1 - rho, tau)
        # tau can go slightly non-positive for extreme parameter values visited
        # mid-optimization (never at the converged optimum for football-scale
        # lambdas); clip so log() stays finite and the optimizer keeps moving.
        tau = np.clip(tau, 1e-8, None)

        penalty = l2_penalty * (np.sum(a**2) + np.sum(d**2))
        return -float(np.sum(weights * (ll + np.log(tau)))) + penalty

    bounds = [(None, None)] * (2 * n + 1) + [rho_bounds]
    result = minimize(neg_log_likelihood, theta0, method="L-BFGS-B", bounds=bounds)

    a, d, log_gamma, rho = unpack(result.x)
    return pd.DataFrame(
        {
            "team": teams,
            "attack": np.exp(a),
            "defense": np.exp(d),
            "home_advantage": np.exp(log_gamma),
            "rho": rho,
        }
    )


# ---------------------------------------------------------------------------
# Walk-forward refit, per league
# ---------------------------------------------------------------------------
def walk_forward_ratings(
    matches: pd.DataFrame,
    *,
    refit_every_matches: int,
    min_matches_for_fit: int,
    xi: float,
    l2_penalty: float,
    rho_bounds: tuple[float, float],
) -> pd.DataFrame:
    """Leakage-safe ratings history: refit periodically, per league.

    Checkpoints are aligned to whole calendar dates (never splitting a single
    matchday's fixtures across the boundary) -- the same "never bisect the
    atomic unit" principle `splits.py` applies to seasons. A checkpoint's
    `from_date` is the last date included in its fit; because
    `attach_dixon_coles` looks up ratings as-of the day BEFORE kickoff, a match
    played ON `from_date` itself still resolves to the PRIOR (older) snapshot,
    never the one that was just fit partly on that date's own results.

    Only matches with a known result are ever used to fit -- unplayed fixtures
    (NaN fthg/ftag, present when `features.py` is building live-scoring
    features) are dropped here but still receive a rating via `attach_dixon_coles`'s
    as-of lookup against whatever the latest real checkpoint is.
    """
    completed = matches.dropna(subset=["fthg", "ftag"]).copy()
    completed["date"] = pd.to_datetime(completed["date"])

    all_ratings = []
    for league, league_matches in completed.groupby("league", sort=False):
        league_matches = league_matches.sort_values("date", kind="stable").reset_index(drop=True)
        unique_dates = sorted(league_matches["date"].unique())

        warm_start = None
        cumulative = 0
        next_target = min_matches_for_fit
        n_checkpoints = 0

        for d in unique_dates:
            cumulative += int((league_matches["date"] == d).sum())
            if cumulative < next_target:
                continue

            cutoff_date = pd.Timestamp(d)
            prior = league_matches[league_matches["date"] <= cutoff_date]
            ratings = fit_ratings(
                prior,
                cutoff_date,
                xi=xi,
                l2_penalty=l2_penalty,
                rho_bounds=rho_bounds,
                warm_start=warm_start,
            )
            ratings["league"] = league
            ratings["from_date"] = cutoff_date
            all_ratings.append(ratings)
            warm_start = ratings
            n_checkpoints += 1
            next_target = cumulative + refit_every_matches

        log.info(
            "dixon_coles %s: %d checkpoints over %d completed matches",
            league,
            n_checkpoints,
            len(league_matches),
        )

    if not all_ratings:
        return pd.DataFrame(columns=RATINGS_COLUMNS)
    return pd.concat(all_ratings, ignore_index=True)[RATINGS_COLUMNS]


# ---------------------------------------------------------------------------
# Match-level outputs (pure functions -- called from features.py once ratings
# are attached per side)
# ---------------------------------------------------------------------------
def expected_goals(home_attack, home_defense, away_attack, away_defense, home_advantage):
    """Natural-scale attack/defense/home_advantage in, expected goals out.

    lambda_home gets the home-advantage multiplier; lambda_away does not --
    the standard Dixon-Coles convention.
    """
    lambda_home = home_attack * away_defense * home_advantage
    lambda_away = away_attack * home_defense
    return lambda_home, lambda_away


def match_probabilities(lambda_home, lambda_away, rho, max_goals: int = 10):
    """Tau-adjusted bivariate Poisson pmf, summed into 1X2 probabilities.

    Vectorized over arrays of matches. NaN lambda/rho (cold-start matches with
    no rating snapshot yet) propagate to NaN probabilities, same as every
    other early-history feature in this pipeline.
    """
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)
    rho = np.asarray(rho, dtype=float)

    goals = np.arange(max_goals + 1)
    pmf_home = poisson.pmf(goals[None, :], lambda_home[:, None])  # (n, G+1)
    pmf_away = poisson.pmf(goals[None, :], lambda_away[:, None])  # (n, G+1)
    joint = pmf_home[:, :, None] * pmf_away[:, None, :]  # (n, G+1, G+1): [i, x, y]

    tau = np.ones_like(joint)
    tau[:, 0, 0] = 1 - lambda_home * lambda_away * rho
    tau[:, 1, 0] = 1 + lambda_away * rho
    tau[:, 0, 1] = 1 + lambda_home * rho
    tau[:, 1, 1] = 1 - rho
    joint = joint * np.clip(tau, 1e-8, None)

    totals = joint.sum(axis=(1, 2), keepdims=True)
    joint = joint / totals

    home_goals = goals[:, None]
    away_goals = goals[None, :]
    win_mask = (home_goals > away_goals).astype(float)
    draw_mask = (home_goals == away_goals).astype(float)
    away_mask = (home_goals < away_goals).astype(float)

    p_h = (joint * win_mask[None, :, :]).sum(axis=(1, 2))
    p_d = (joint * draw_mask[None, :, :]).sum(axis=(1, 2))
    p_a = (joint * away_mask[None, :, :]).sum(axis=(1, 2))
    return p_h, p_d, p_a


# ---------------------------------------------------------------------------
# Leakage guard: home/away snapshot consistency
# ---------------------------------------------------------------------------
def assert_home_away_consistent(wide: pd.DataFrame, column: str, atol: float = 1e-9) -> None:
    """Both teams in a match must resolve to the same league-wide snapshot.

    home_advantage and rho are global-per-fit, not per-team, so `home_<column>`
    and `away_<column>` should always agree for a given match. If they don't,
    it means the two sides' as-of lookups landed on different checkpoints --
    a real bug, not a benign edge case, so this raises rather than warns.
    """
    home_col, away_col = f"home_{column}", f"away_{column}"
    both_present = wide[home_col].notna() & wide[away_col].notna()

    # np.isclose returns an array shaped to the (filtered) subset, not the full
    # frame, so it has to be scattered back by index before combining with
    # `both_present` -- ANDing them directly would misalign whenever any row
    # is missing a value.
    close = pd.Series(True, index=wide.index)
    close.loc[both_present] = np.isclose(
        wide.loc[both_present, home_col], wide.loc[both_present, away_col], atol=atol
    )
    mismatched = both_present & ~close
    if mismatched.any():
        bad = wide.loc[mismatched, ["match_id", home_col, away_col]]
        raise RuntimeError(
            f"dixon_coles: home/away {column} disagree for {int(mismatched.sum())} "
            f"match(es) -- both teams should share one league-wide snapshot; "
            f"first few:\n{bad.head(10).to_string(index=False)}"
        )
