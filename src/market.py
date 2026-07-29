"""Stage 3: turn bookmaker closing odds into vig-free market probabilities.

This module produces the project's central benchmark. Decimal odds do not sum to
a probability -- the book's margin ("overround") makes the raw implied
probabilities sum to roughly 1.03-1.08. Removing that margin is the whole job,
and *how* you remove it measurably changes which fixtures look like value.

We compute two devigging methods and keep both:

  multiplicative  p_i = pi_i / sum(pi)
      Simple proportional normalization. Assumes the book spreads its margin
      evenly across outcomes. Easy, and wrong in a known direction: it
      systematically overstates longshot probabilities.

  shin            Shin (1993)
      Models the margin as the book's defence against insider trading, which
      concentrates it on longshots. Solves for the insider-trading proportion z
      such that the recovered probabilities sum to 1. Better calibrated on the
      away-win and draw tails, which is exactly where our value flags will fire.

The primary method is set by `market.primary_devig` in params.yaml.

Raw closing prices are carried through unchanged, because closing-line value
(CLV) in src/evaluate.py needs the actual price, not the probability.

Output: data/interim/market.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    MARKET_PATH,
    MATCHES_PATH,
    OUTCOMES,
    PARAMS,
    ensure_dirs,
    get_logger,
)

log = get_logger("market")

_M = PARAMS["market"]
ODDS_PREFERENCE: list[str] = _M["odds_preference"]
PRIMARY_DEVIG: str = _M["primary_devig"]


def select_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pick the best available closing-odds triple for each match.

    Walks `odds_preference` in order and takes the first source that has all
    three prices present and greater than 1.0. Returns the chosen odds and a
    label recording which source won, so the writeup can report coverage.
    """
    odds = pd.DataFrame(
        np.nan, index=df.index, columns=OUTCOMES, dtype=float
    )
    source = pd.Series(pd.NA, index=df.index, dtype="object")

    for prefix in ODDS_PREFERENCE:
        cols = [f"odds_{prefix}_{o}" for o in OUTCOMES]
        if not all(c in df.columns for c in cols):
            continue

        candidate = df[cols]
        # Odds of exactly 1.0 or below are data errors, not prices.
        usable = candidate.notna().all(axis=1) & (candidate > 1.0).all(axis=1)
        fill = usable & source.isna()
        if not fill.any():
            continue

        odds.loc[fill, OUTCOMES] = candidate.loc[fill].to_numpy()
        source.loc[fill] = prefix
        log.info("  %-6s supplied %5d matches", prefix, int(fill.sum()))

    return odds, source


def devig_multiplicative(implied: np.ndarray) -> np.ndarray:
    """Proportional normalization. implied is (n, 3) of 1/odds."""
    return implied / implied.sum(axis=1, keepdims=True)


def devig_shin(implied: np.ndarray, *, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Shin's method, solved by vectorized bisection over z.

    For overround Pi = sum(pi) and insider proportion z:

        p_i(z) = [sqrt(z^2 + 4(1-z) * pi_i^2 / Pi) - z] / (2(1-z))

    sum(p_i) decreases monotonically in z, from sqrt(Pi) > 1 at z=0 to
    sum(pi_i^2)/Pi < 1 as z -> 1, so a unique root exists in (0, 1) whenever the
    book has a positive margin. Bisection is used rather than a scalar solver
    because it vectorizes across all ~14k matches at once and cannot diverge.

    Returns (probabilities, z).
    """
    overround = implied.sum(axis=1, keepdims=True)

    def sum_p(z: np.ndarray) -> np.ndarray:
        z = z.reshape(-1, 1)
        inner = z**2 + 4.0 * (1.0 - z) * implied**2 / overround
        p = (np.sqrt(np.maximum(inner, 0.0)) - z) / (2.0 * (1.0 - z))
        return p.sum(axis=1)

    n = implied.shape[0]
    lo = np.full(n, 1e-10)
    hi = np.full(n, 1.0 - 1e-10)

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        # sum_p is decreasing in z: if it is still above 1, z must be larger.
        too_high = sum_p(mid) > 1.0
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)

    z = 0.5 * (lo + hi)
    zc = z.reshape(-1, 1)
    inner = zc**2 + 4.0 * (1.0 - zc) * implied**2 / overround
    probs = (np.sqrt(np.maximum(inner, 0.0)) - zc) / (2.0 * (1.0 - zc))

    # Guard against the degenerate no-margin case (arbitrage or bad data),
    # where Shin is undefined; fall back to proportional normalization.
    degenerate = (overround.ravel() <= 1.0) | ~np.isfinite(probs).all(axis=1)
    if degenerate.any():
        log.warning("%d matches had no usable overround; using multiplicative devig", int(degenerate.sum()))
        probs[degenerate] = devig_multiplicative(implied[degenerate])
        z[degenerate] = 0.0

    # Renormalize away residual bisection error (order 1e-10).
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs, z


def main() -> pd.DataFrame:
    ensure_dirs()
    matches = pd.read_parquet(MATCHES_PATH)
    log.info("loaded %d matches", len(matches))

    odds, source = select_odds(matches)
    has_odds = source.notna()
    log.info("odds coverage: %.2f%% (%d/%d)", 100 * has_odds.mean(), has_odds.sum(), len(matches))

    priced = matches.loc[has_odds].copy()
    odds_arr = odds.loc[has_odds].to_numpy(dtype=float)
    implied = 1.0 / odds_arr
    overround = implied.sum(axis=1)

    p_mult = devig_multiplicative(implied)
    p_shin, shin_z = devig_shin(implied)

    out = pd.DataFrame({"match_id": priced["match_id"].values})
    out["odds_source"] = source.loc[has_odds].values
    out["overround"] = overround
    out["shin_z"] = shin_z

    for i, outcome in enumerate(OUTCOMES):
        out[f"odds_{outcome}"] = odds_arr[:, i]
        out[f"p_mult_{outcome}"] = p_mult[:, i]
        out[f"p_shin_{outcome}"] = p_shin[:, i]

    # Opening prices, carried through for closing-line value in evaluation.
    #
    # CLV asks whether the line moved toward you between placing a bet and
    # kickoff, so it needs two prices at different times. football-data ships
    # both Pinnacle series -- PS is the early price, PSC the closing one -- so a
    # bet struck at PS can be scored against PSC. Without these columns CLV
    # would not be computable from closing odds alone.
    open_cols = [f"odds_PS_{o}" for o in OUTCOMES]
    if all(c in matches.columns for c in open_cols):
        opening = priced[open_cols].to_numpy(dtype=float)
        valid_open = np.isfinite(opening).all(axis=1) & (opening > 1.0).all(axis=1)
        opening[~valid_open] = np.nan
        for i, outcome in enumerate(OUTCOMES):
            out[f"odds_open_{outcome}"] = opening[:, i]
        log.info("opening prices available for %.2f%% of priced matches", 100 * valid_open.mean())
    else:
        log.warning("no Pinnacle opening columns found; CLV will be unavailable")

    # The primary set downstream stages actually use.
    for outcome in OUTCOMES:
        out[f"p_market_{outcome}"] = out[f"p_{PRIMARY_DEVIG}_{outcome}"]

    out.to_parquet(MARKET_PATH, index=False)
    log.info("wrote %s (%d priced matches)", MARKET_PATH, len(out))

    log.info("source mix:\n%s", out["odds_source"].value_counts().to_string())
    log.info(
        "overround: mean=%.4f median=%.4f p95=%.4f",
        overround.mean(),
        np.median(overround),
        np.percentile(overround, 95),
    )
    log.info("shin z: mean=%.4f median=%.4f", shin_z.mean(), np.median(shin_z))

    mean_probs = {o: out[f"p_market_{o}"].mean() for o in OUTCOMES}
    log.info(
        "mean market probability: H=%.3f D=%.3f A=%.3f (should track the base rates)",
        *[mean_probs[o] for o in OUTCOMES],
    )

    # How much the two devig methods disagree -- context for the value threshold.
    max_gap = np.abs(p_shin - p_mult).max(axis=1)
    log.info(
        "shin vs multiplicative, max per-match probability gap: mean=%.4f p95=%.4f",
        max_gap.mean(),
        np.percentile(max_gap, 95),
    )
    return out


if __name__ == "__main__":
    main()
