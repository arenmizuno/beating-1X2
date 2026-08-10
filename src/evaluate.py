"""Stage 7: statistical and economic evaluation against the market baseline.

Two questions, and they are not the same question:

  Statistical -- are our probabilities better than the closing line's?
      Answered with proper scoring rules (log loss, Brier) and calibration
      diagnostics, always computed on identical rows for model and market.

  Economic -- would acting on the disagreements have made money?
      Answered by backtesting the value-bet rule with confidence intervals, and
      by closing-line value.

The confidence intervals are not decoration. A few hundred flagged bets at ~2.5
average odds produce enormously noisy ROI: swings of +/-10% arise from chance
alone, so a point estimate of "+4% ROI" without an interval is not evidence of
anything. Every economic number here is reported with a bootstrap CI, and the
honest reading is usually that the interval spans zero.

CLV caveat: value is flagged against the closing-derived market probability,
then CLV asks whether the opening price on that same selection was better than
the close. It answers "does the model pick sides the market later moves toward"
-- genuine evidence of signal -- but it is not a clean simulation of betting at
open, because the flag itself used closing information.

Outputs (all under reports/):
  model_vs_market.{csv,md}     headline statistical comparison
  breakdown_by_league.csv      per-league, per-season detail
  calibration_data.csv         reliability-diagram data
  calibration_*.png            reliability diagrams
  value_backtest.{csv,md}      ROI, Kelly, CLV with bootstrap CIs
  roi_by_threshold.png         sensitivity to the edge threshold
  summary.md                   the writeup-ready digest
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metrics
from src.config import (
    HOLDOUT_SEASON,
    OUTCOMES,
    PARAMS,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    season_label,
)

log = get_logger("evaluate")

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

_V = PARAMS["value"]
EDGE_THRESHOLD: float = _V["edge_threshold"]
RELATIVE_THRESHOLD: float = _V["relative_threshold"]
KELLY_FRACTION: float = _V["kelly_fraction"]
FLAT_STAKE: float = _V["flat_stake"]
BOOTSTRAP_ITERATIONS: int = _V["bootstrap_iterations"]
BOOTSTRAP_CI: float = _V["bootstrap_ci"]
N_BINS: int = PARAMS["evaluate"]["calibration_bins"]
MIN_CALIBRATION_BIN_N: int = PARAMS["evaluate"]["min_calibration_bin_n"]

P_COLS = [f"p_{o}" for o in OUTCOMES]
P_MARKET_COLS = [f"p_market_{o}" for o in OUTCOMES]
P_DC_COLS = [f"dc_p_{o}" for o in OUTCOMES]
ODDS_COLS = [f"odds_{o}" for o in OUTCOMES]
ODDS_OPEN_COLS = [f"odds_open_{o}" for o in OUTCOMES]


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------
def compare_to_market(predictions: pd.DataFrame) -> pd.DataFrame:
    """Model vs market vs pure Dixon-Coles on every (track, model, fold).

    Dixon-Coles is a third, independent benchmark: a goals-based rating that
    never sees a bookmaker price, scored on the exact same rows as model and
    market. It is NOT a model input (see feature_columns() in train.py), so
    "model beats Dixon-Coles" is a genuine comparison, not a tautology.
    """
    rows = []
    for (track, model, fold), chunk in predictions.groupby(["track", "model", "fold"]):
        y = chunk["y_true"].to_numpy()
        model_scores = metrics.summary(y, chunk[P_COLS].to_numpy(), N_BINS)
        market_scores = metrics.summary(y, chunk[P_MARKET_COLS].to_numpy(), N_BINS)
        dc_scores = metrics.summary(y, chunk[P_DC_COLS].to_numpy(), N_BINS)
        rows.append(
            {
                "track": track,
                "model": model,
                "fold": fold,
                "n": len(chunk),
                **{f"model_{k}": v for k, v in model_scores.items()},
                **{f"market_{k}": v for k, v in market_scores.items()},
                **{f"dc_{k}": v for k, v in dc_scores.items()},
                "logloss_gap": model_scores["log_loss"] - market_scores["log_loss"],
                "dc_logloss_gap": dc_scores["log_loss"] - market_scores["log_loss"],
            }
        )
    return pd.DataFrame(rows).sort_values(["track", "model", "fold"])


def breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-league and per-season detail, walk-forward folds only.

    Aggregate numbers can hide a model that is fine in four leagues and broken
    in the fifth, which is exactly the kind of thing a drift dashboard would
    later need to catch.
    """
    rows = []
    walk_forward = predictions[~predictions["fold"].str.startswith("holdout")]
    for (track, model, league, season), chunk in walk_forward.groupby(
        ["track", "model", "league", "season"]
    ):
        if len(chunk) < 50:
            continue
        y = chunk["y_true"].to_numpy()
        rows.append(
            {
                "track": track,
                "model": model,
                "league": league,
                "season": season,
                "n": len(chunk),
                "model_log_loss": metrics.log_loss(y, chunk[P_COLS].to_numpy()),
                "market_log_loss": metrics.log_loss(y, chunk[P_MARKET_COLS].to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    out["logloss_gap"] = out["model_log_loss"] - out["market_log_loss"]
    return out


# ---------------------------------------------------------------------------
# Value bets and backtesting
# ---------------------------------------------------------------------------
def flag_value_bets(
    chunk: pd.DataFrame,
    threshold: float = EDGE_THRESHOLD,
    *,
    mode: str = "absolute",
    relative_threshold: float = RELATIVE_THRESHOLD,
) -> pd.DataFrame:
    """One row per flagged (match, outcome) selection.

    Two selection rules:

      "absolute"  p_model - p_market > threshold. The rule as specified in the
                  project proposal. Measured to be heavily longshot-biased --
                  see the note on `value.relative_threshold` in params.yaml.

      "relative"  requires BOTH the absolute margin above AND
                  p_model / p_market - 1 > relative_threshold, which scales the
                  requirement with the price and stops the rule collapsing into
                  "bet every longshot".

    All three outcomes are tested independently, so one match can in principle
    produce more than one selection.
    """
    proba = chunk[P_COLS].to_numpy()
    market = chunk[P_MARKET_COLS].to_numpy()
    odds = chunk[ODDS_COLS].to_numpy()
    has_open = all(c in chunk.columns for c in ODDS_OPEN_COLS)
    odds_open = chunk[ODDS_OPEN_COLS].to_numpy() if has_open else np.full_like(odds, np.nan)

    edge = proba - market
    qualifies = edge > threshold
    if mode == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_edge = np.where(market > 0, proba / market - 1.0, np.inf)
        qualifies &= relative_edge > relative_threshold
    elif mode != "absolute":
        raise ValueError(f"unknown value-bet mode: {mode}")

    rows_idx, class_idx = np.where(qualifies)
    if len(rows_idx) == 0:
        return pd.DataFrame()

    selected = chunk.iloc[rows_idx]
    won = (selected["y_true"].to_numpy() == class_idx).astype(float)
    price = odds[rows_idx, class_idx]

    bets = pd.DataFrame(
        {
            "match_id": selected["match_id"].values,
            "league": selected["league"].values,
            "season": selected["season"].values,
            "outcome": [OUTCOMES[k] for k in class_idx],
            "p_model": proba[rows_idx, class_idx],
            "p_market": market[rows_idx, class_idx],
            "edge": edge[rows_idx, class_idx],
            "odds": price,
            "odds_open": odds_open[rows_idx, class_idx],
            "won": won,
        }
    )

    # Flat staking: one unit per selection.
    bets["flat_stake"] = FLAT_STAKE
    bets["flat_profit"] = np.where(won == 1, FLAT_STAKE * (price - 1.0), -FLAT_STAKE)

    # Fractional Kelly. Full Kelly is far too aggressive when the probability
    # estimate is itself uncertain -- and ours demonstrably is, since it loses
    # to the market on log loss -- so the stake is scaled down and never
    # allowed to go negative (that would be backing the other side).
    kelly_full = (bets["p_model"] * price - 1.0) / (price - 1.0)
    bets["kelly_stake"] = (KELLY_FRACTION * kelly_full).clip(lower=0.0)
    bets["kelly_profit"] = np.where(
        won == 1, bets["kelly_stake"] * (price - 1.0), -bets["kelly_stake"]
    )

    # CLV: did we take a better price than the market closed at?
    bets["clv"] = bets["odds_open"] / bets["odds"] - 1.0
    return bets


def bootstrap_roi(
    profit: np.ndarray, stake: np.ndarray, iterations: int = BOOTSTRAP_ITERATIONS
) -> tuple[float, float, float]:
    """Bootstrap CI for ROI, resampling whole bets.

    Returns (roi, lo, hi). Resampling bets rather than matches keeps each
    draw's stake/return pairing intact.
    """
    if len(profit) == 0 or stake.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(SEED)
    n = len(profit)
    draws = rng.integers(0, n, size=(iterations, n))
    sampled_roi = profit[draws].sum(axis=1) / np.maximum(stake[draws].sum(axis=1), 1e-12)

    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    return (
        float(profit.sum() / stake.sum()),
        float(np.quantile(sampled_roi, alpha)),
        float(np.quantile(sampled_roi, 1.0 - alpha)),
    )


def backtest(predictions: pd.DataFrame, threshold: float = EDGE_THRESHOLD) -> pd.DataFrame:
    rows = []
    for (track, model), chunk in predictions.groupby(["track", "model"]):
        for scope, subset in (
            ("walk_forward", chunk[~chunk["fold"].str.startswith("holdout")]),
            ("holdout", chunk[chunk["fold"].str.startswith("holdout")]),
        ):
            for mode in ("absolute", "relative"):
                bets = flag_value_bets(subset, threshold, mode=mode)
                if bets.empty:
                    rows.append(
                        {
                            "track": track,
                            "model": model,
                            "scope": scope,
                            "mode": mode,
                            "n_bets": 0,
                        }
                    )
                    continue

                flat_roi, flat_lo, flat_hi = bootstrap_roi(
                    bets["flat_profit"].to_numpy(), bets["flat_stake"].to_numpy()
                )
                kelly_roi, kelly_lo, kelly_hi = bootstrap_roi(
                    bets["kelly_profit"].to_numpy(), bets["kelly_stake"].to_numpy()
                )
                clv = bets["clv"].dropna()

                rows.append(
                    {
                        "track": track,
                        "model": model,
                        "scope": scope,
                        "mode": mode,
                        "n_bets": len(bets),
                        "bet_rate": len(bets) / len(subset),
                        "mean_odds": bets["odds"].mean(),
                        "mean_edge": bets["edge"].mean(),
                        "strike_rate": bets["won"].mean(),
                        "flat_roi": flat_roi,
                        "flat_roi_lo": flat_lo,
                        "flat_roi_hi": flat_hi,
                        "flat_profit_units": bets["flat_profit"].sum(),
                        "kelly_roi": kelly_roi,
                        "kelly_roi_lo": kelly_lo,
                        "kelly_roi_hi": kelly_hi,
                        "mean_clv": clv.mean() if len(clv) else float("nan"),
                        "positive_clv_rate": (clv > 0).mean() if len(clv) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def roi_by_threshold(predictions: pd.DataFrame) -> pd.DataFrame:
    """ROI across a sweep of edge thresholds.

    A rule that only looks profitable at one hand-picked threshold is a rule
    fitted to noise. This is the sensitivity check for that.
    """
    rows = []
    thresholds = np.arange(0.02, 0.21, 0.01)
    walk_forward = predictions[~predictions["fold"].str.startswith("holdout")]

    for (track, model), chunk in walk_forward.groupby(["track", "model"]):
        for threshold in thresholds:
            bets = flag_value_bets(chunk, float(threshold))
            if bets.empty:
                rows.append(
                    {"track": track, "model": model, "threshold": threshold, "n_bets": 0}
                )
                continue
            roi, lo, hi = bootstrap_roi(
                bets["flat_profit"].to_numpy(), bets["flat_stake"].to_numpy()
            )
            rows.append(
                {
                    "track": track,
                    "model": model,
                    "threshold": float(threshold),
                    "n_bets": len(bets),
                    "flat_roi": roi,
                    "roi_lo": lo,
                    "roi_hi": hi,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    """Reliability diagrams, model against market, per track/model."""
    all_data = []
    walk_forward = predictions[~predictions["fold"].str.startswith("holdout")]

    for (track, model), chunk in walk_forward.groupby(["track", "model"]):
        y = chunk["y_true"].to_numpy()
        model_cal = metrics.per_class_calibration(y, chunk[P_COLS].to_numpy(), N_BINS)
        market_cal = metrics.per_class_calibration(y, chunk[P_MARKET_COLS].to_numpy(), N_BINS)
        model_cal["source"], market_cal["source"] = "model", "market"
        combined = pd.concat([model_cal, market_cal])
        combined["track"], combined["model"] = track, model
        all_data.append(combined)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
        for ax, outcome in zip(axes, OUTCOMES, strict=True):
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="perfect")
            for source, style in (("model", "o-"), ("market", "s--")):
                part = combined[(combined["source"] == source) & (combined["outcome"] == outcome)]
                # Bins holding a handful of matches are noise, and plotting them
                # invites reading a 1-of-1 bin as a calibration failure. The draw
                # panel is where this bites: a draw is almost never priced above
                # ~0.35, so the top bins hold single matches. Everything stays in
                # calibration_data.csv; only the plot is filtered.
                part = part[part["n"] >= MIN_CALIBRATION_BIN_N]
                ax.plot(part["mean_predicted"], part["observed_rate"], style, ms=4, label=source)
            ax.set_title(f"{outcome}  ({['home win', 'draw', 'away win'][OUTCOMES.index(outcome)]})")
            ax.set_xlabel("predicted probability")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("observed frequency")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(
            f"Reliability -- {track} / {model} (walk-forward folds; "
            f"bins with n < {MIN_CALIBRATION_BIN_N} omitted)"
        )
        fig.tight_layout()
        path = REPORTS_DIR / f"calibration_{track}_{model}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info("wrote %s", path)

    return pd.concat(all_data, ignore_index=True)


def plot_roi_curve(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (track, model), chunk in sweep.groupby(["track", "model"]):
        chunk = chunk[chunk["n_bets"] > 30]
        if chunk.empty:
            continue
        ax.plot(chunk["threshold"], 100 * chunk["flat_roi"], "o-", ms=4, label=f"{track}/{model}")
        ax.fill_between(
            chunk["threshold"], 100 * chunk["roi_lo"], 100 * chunk["roi_hi"], alpha=0.12
        )
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("edge threshold (model probability minus market probability)")
    ax.set_ylabel("flat-stake ROI (%)")
    ax.set_title("Value-bet ROI by threshold, with 95% bootstrap intervals\n(walk-forward folds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = REPORTS_DIR / "roi_by_threshold.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(comparison: pd.DataFrame, bets: pd.DataFrame) -> None:
    walk_forward = comparison[~comparison["fold"].str.startswith("holdout")]
    holdout = comparison[comparison["fold"].str.startswith("holdout")]

    wf_mean = (
        walk_forward.groupby(["track", "model"])[
            [
                "model_log_loss",
                "market_log_loss",
                "dc_log_loss",
                "model_brier",
                "model_ece",
                "model_accuracy",
            ]
        ]
        .mean()
        .reset_index()
    )
    wf_mean["gap"] = wf_mean["model_log_loss"] - wf_mean["market_log_loss"]
    wf_mean = wf_mean.sort_values("model_log_loss")

    lines = [
        "# beating-1X2 -- Results Summary",
        "",
        "Generated by `python -m src.evaluate`.",
        "",
        "## 1. Probability quality vs the market and Dixon-Coles (walk-forward mean)",
        "",
        "The market column is the vig-free closing line; dc_log_loss is a pure",
        "Dixon-Coles goals model (never sees a bookmaker price, and is excluded",
        "from the model's own features) -- both scored on exactly the same matches",
        "as the model. Lower log loss is better.",
        "",
        wf_mean.round(4).to_markdown(index=False),
        "",
        f"**Market baseline: {walk_forward['market_log_loss'].mean():.4f} log loss. "
        f"Dixon-Coles baseline: {walk_forward['dc_log_loss'].mean():.4f} log loss.**",
        "",
        "## 2. Holdout season (" + season_label(HOLDOUT_SEASON) + ", never used in training)",
        "",
        holdout[
            [
                "track",
                "model",
                "n",
                "model_log_loss",
                "market_log_loss",
                "dc_log_loss",
                "logloss_gap",
                "dc_logloss_gap",
            ]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
        "## 3. Value-bet backtest",
        "",
        "ROI intervals are 95% bootstrap over resampled bets. An interval spanning",
        "zero means the result is indistinguishable from chance.",
        "",
        bets.round(4).to_markdown(index=False),
        "",
    ]
    path = REPORTS_DIR / "summary.md"
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)


def main() -> None:
    ensure_dirs()
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    log.info("loaded %d prediction rows", len(predictions))

    comparison = compare_to_market(predictions)
    comparison.to_csv(REPORTS_DIR / "model_vs_market.csv", index=False)
    (REPORTS_DIR / "model_vs_market.md").write_text(comparison.round(4).to_markdown(index=False))
    log.info("model vs market:\n%s", comparison.round(4).to_string(index=False))

    detail = breakdown(predictions)
    detail.to_csv(REPORTS_DIR / "breakdown_by_league.csv", index=False)
    by_league = detail.groupby(["track", "model", "league"])["logloss_gap"].mean().unstack()
    log.info("mean log-loss gap vs market, by league:\n%s", by_league.round(4).to_string())

    calibration = plot_calibration(predictions)
    calibration.to_csv(REPORTS_DIR / "calibration_data.csv", index=False)

    bets = backtest(predictions)
    bets.to_csv(REPORTS_DIR / "value_backtest.csv", index=False)
    (REPORTS_DIR / "value_backtest.md").write_text(bets.round(4).to_markdown(index=False))
    log.info("value-bet backtest:\n%s", bets.round(4).to_string(index=False))

    sweep = roi_by_threshold(predictions)
    sweep.to_csv(REPORTS_DIR / "roi_by_threshold.csv", index=False)
    plot_roi_curve(sweep)

    write_summary(comparison, bets)


if __name__ == "__main__":
    main()
