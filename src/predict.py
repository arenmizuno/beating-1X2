"""Inference: score fixtures with the registered champion model.

This is the one place that turns a fixture into a prediction, used by both the
FastAPI service and the Prefect scoring flow, so the API and any batch job
cannot drift apart.

Three properties worth stating plainly, because each is a decision:

**The model is resolved by alias, never by name.** Everything loads
`models:/beating-1x2@champion` and reads its `model_contract.json` for the
feature list and exact design-matrix column order. Swapping in a different
algorithm later is a re-registration; nothing in this file changes.

**Features come from the training code path.** `build_feature_frame` appends the
unplayed fixtures to the completed-match history and runs the same
shift-then-roll pipeline, so serving features cannot diverge from training
features. See src/features.py for why appending is leakage-safe.

**Live prices are not closing prices.** The backtest benchmarks against Pinnacle
closing odds; an unplayed fixture has no close yet. Live value flags therefore
compare against a systematically softer line and will fire more readily than the
backtest implies. Every response carries `odds_source` and an explicit
`benchmark` note so this is visible at the point of use rather than buried.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from src.config import (
    CHAMPION_ALIAS,
    INTERIM_DIR,
    MATCHES_PATH,
    MLRUNS_DIR,
    OUTCOMES,
    PARAMS,
    REGISTERED_MODEL_NAME,
    get_logger,
)
from src.features import build_feature_frame
from src.market import devig_multiplicative, devig_shin, select_odds
from src.train import apply_probability_floor, make_design

log = get_logger("predict")

CLUBELO_CANONICAL_PATH = INTERIM_DIR / "clubelo_canonical.parquet"
EDGE_THRESHOLD: float = PARAMS["value"]["edge_threshold"]
PRIMARY_DEVIG: str = PARAMS["market"]["primary_devig"]


@dataclass
class Champion:
    """The loaded model plus the contract describing how to feed it."""

    model: object
    version: str
    model_semver: str
    run_id: str
    track: str
    algorithm: str
    feature_columns: list[str]
    design_columns: list[str]
    train_seasons: list[int]


def load_champion() -> Champion:
    """Resolve and load models:/<name>@champion from the local MLflow store."""
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    client = mlflow.tracking.MlflowClient()

    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    # Resolution is by alias; the artifact *location* is then rebased onto the
    # local store root instead of trusting the path MLflow recorded.
    #
    # The file-backed registry bakes an absolute host path into the version's
    # `source` (and into each run's `artifact_uri`) at registration time. Loading
    # `models:/<name>@champion` directly follows that path, which exists only on
    # the machine that trained the model -- so the container, which mounts the
    # same store at /app/mlruns, fails at startup with "No such file or
    # directory". Rebuilding the path from MLRUNS_DIR makes the store relocatable
    # without changing how the champion is selected.
    run = client.get_run(version.run_id)
    artifacts = MLRUNS_DIR / run.info.experiment_id / version.run_id / "artifacts"
    model = mlflow.sklearn.load_model(str(artifacts / "model"))
    contract = json.loads((artifacts / "model_contract.json").read_text())

    log.info(
        "loaded %s v%s (semver %s, %s/%s, %d features)",
        REGISTERED_MODEL_NAME,
        version.version,
        contract["model_semver"],
        contract["track"],
        contract["model"],
        len(contract["feature_columns"]),
    )
    return Champion(
        model=model,
        # MLflow hands back an int here; every consumer treats registry versions
        # as opaque labels, so normalize to str once at the boundary.
        version=str(version.version),
        model_semver=contract["model_semver"],
        run_id=version.run_id,
        track=contract["track"],
        algorithm=contract["model"],
        feature_columns=contract["feature_columns"],
        design_columns=contract["design_columns"],
        train_seasons=contract["train_seasons"],
    )


def attach_market_probabilities(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Devig whatever live odds the fixture feed carried.

    Reuses src/market.py so the live path and the historical path remove the
    bookmaker margin identically.
    """
    out = fixtures.copy()
    for col in [f"p_market_{o}" for o in OUTCOMES] + [f"odds_{o}" for o in OUTCOMES]:
        out[col] = np.nan
    out["odds_source"] = None

    odds, source = select_odds(out)
    priced = source.notna()
    if not priced.any():
        log.warning("no usable live odds on any fixture")
        return out

    implied = 1.0 / odds.loc[priced].to_numpy(dtype=float)
    probs = (
        devig_shin(implied)[0] if PRIMARY_DEVIG == "shin" else devig_multiplicative(implied)
    )

    for i, outcome in enumerate(OUTCOMES):
        out.loc[priced, f"odds_{outcome}"] = odds.loc[priced].to_numpy()[:, i]
        out.loc[priced, f"p_market_{outcome}"] = probs[:, i]
    out.loc[priced, "odds_source"] = source[priced]

    log.info("devigged live odds for %d/%d fixtures", int(priced.sum()), len(out))
    return out


def score_fixtures(
    fixtures: pd.DataFrame,
    champion: Champion | None = None,
    *,
    matches: pd.DataFrame | None = None,
    elo: pd.DataFrame | None = None,
    threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Score unplayed fixtures and flag value against the live market price."""
    if fixtures.empty:
        return pd.DataFrame()

    champion = champion or load_champion()
    matches = matches if matches is not None else pd.read_parquet(MATCHES_PATH)
    elo = elo if elo is not None else pd.read_parquet(CLUBELO_CANONICAL_PATH)

    priced = attach_market_probabilities(fixtures)

    features = build_feature_frame(
        matches,
        elo,
        upcoming=priced[["match_id", "league", "season", "date", "home_team", "away_team"]],
    )
    # The market columns live on the fixture rows, not on the historical spine,
    # so they are joined back on after feature construction.
    market_cols = [f"p_market_{o}" for o in OUTCOMES] + [f"odds_{o}" for o in OUTCOMES]
    features = features.merge(
        priced[["match_id", "odds_source", *market_cols]], on="match_id", how="left"
    )

    usable = features["feature_complete"]
    if champion.track == "market_aware":
        # A market-aware champion cannot score a fixture with no live price.
        usable &= features["p_market_H"].notna()
    if not usable.any():
        log.warning("no fixture has both complete features and a usable price")
        return pd.DataFrame()

    scorable = features[usable].copy()
    design = make_design(scorable, champion.feature_columns)
    if list(design.columns) != champion.design_columns:
        raise RuntimeError(
            "design matrix does not match the model contract -- the champion was "
            "trained on a different feature set than this pipeline produces"
        )

    proba, n_floored = apply_probability_floor(champion.model.predict_proba(design))
    if n_floored:
        log.info("%d probabilities hit the floor", n_floored)

    result = scorable[
        ["match_id", "league", "season", "date", "home_team", "away_team", "odds_source"]
    ].copy()
    for i, outcome in enumerate(OUTCOMES):
        result[f"p_{outcome}"] = proba[:, i]
        result[f"p_market_{outcome}"] = scorable[f"p_market_{outcome}"].to_numpy()
        result[f"odds_{outcome}"] = scorable[f"odds_{outcome}"].to_numpy()
        result[f"edge_{outcome}"] = result[f"p_{outcome}"] - result[f"p_market_{outcome}"]

    edges = result[[f"edge_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    best = np.nanargmax(np.where(np.isnan(edges), -np.inf, edges), axis=1)
    result["best_edge_outcome"] = [OUTCOMES[i] for i in best]
    result["best_edge"] = edges[np.arange(len(edges)), best]
    result["value_flag"] = result["best_edge"] > threshold
    result["model_version"] = champion.version
    result["model_semver"] = champion.model_semver
    result["model_track"] = champion.track

    log.info(
        "scored %d fixtures; %d flagged as value at threshold %.2f",
        len(result),
        int(result["value_flag"].sum()),
        threshold,
    )
    return result


def main() -> pd.DataFrame:
    from src.ingest_fixtures import load_fixtures
    from src.prediction_markets import attach_prediction_market_prices

    fixtures = load_fixtures()
    if fixtures.empty:
        log.warning("no upcoming fixtures to score")
        return pd.DataFrame()

    fixtures = attach_prediction_market_prices(fixtures)
    result = score_fixtures(fixtures)
    if not result.empty:
        log.info("\n%s", result.head(20).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
