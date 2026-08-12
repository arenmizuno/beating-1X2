"""FastAPI service for beating-1X2.

Serves the model registered as `models:/beating-1x2@champion`. Nothing here
names an algorithm: the champion's `model_contract.json` supplies the feature
list and the exact design-matrix column order, so replacing the model is a
re-registration plus a restart, not a code change.

Run locally:
    uvicorn src.api:app --reload
Interactive docs at /docs.

A note that appears in the API's own responses, not just here: the backtest
benchmarks against Pinnacle *closing* odds, but an unplayed fixture has no
closing price. Live value flags are therefore computed against a softer,
earlier line and will fire more readily than the backtest implies. Every
prediction carries the `odds_source` actually used.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    CHAMPION_ALIAS,
    LEAGUE_CODES,
    MATCHES_PATH,
    OUTCOMES,
    PARAMS,
    REGISTERED_MODEL_NAME,
    get_logger,
)
from src.predict import (
    CLUBELO_CANONICAL_PATH,
    Champion,
    load_champion,
    score_fixtures,
)
from src.train import apply_probability_floor, make_design

log = get_logger("api")

# Re-exported so tests can build a design-column list matching make_design()
# without importing config separately.
LEAGUE_CODES_FOR_TESTS = LEAGUE_CODES

EDGE_THRESHOLD: float = PARAMS["value"]["edge_threshold"]

# Process-local state, populated once at startup.
STATE: dict[str, Any] = {
    "champion": None,
    "matches": None,
    "elo": None,
    "started_at": None,
    "counters": {"predict": 0, "predict_matches": 0, "predict_upcoming": 0, "errors": 0},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion and the historical data once, at startup.

    The history is needed on every scoring request -- rolling-form features are
    computed from it -- so loading it per request would dominate latency.
    """
    # Reset first. STATE is module-level, so without this an application restart
    # whose model load FAILS would keep serving the previously loaded champion
    # while /health still reported "ok" -- a stale model presented as a live one.
    STATE["champion"] = None
    STATE["matches"] = None
    STATE["elo"] = None
    STATE["error"] = None
    STATE["started_at"] = time.time()

    try:
        STATE["champion"] = load_champion()
        STATE["matches"] = pd.read_parquet(MATCHES_PATH)
        STATE["elo"] = pd.read_parquet(CLUBELO_CANONICAL_PATH)
        log.info(
            "ready: champion v%s, %d historical matches",
            STATE["champion"].version,
            len(STATE["matches"]),
        )
    except Exception as exc:  # noqa: BLE001
        # Start anyway so /health can report *why* the service is unusable,
        # which is far more debuggable than a container that exits immediately.
        log.error("startup failed: %s", exc)
        STATE["error"] = str(exc)
    yield


app = FastAPI(
    title="beating-1X2",
    description=(
        "Match-outcome probabilities for the top-5 European leagues, with "
        "value flags against the betting market. Benchmarked against the "
        "closing line, which it does not beat."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    # pydantic v2 reserves the `model_` prefix; these fields describe the ML
    # model, not pydantic's, so the namespace guard is disabled deliberately.
    model_config = {"protected_namespaces": ()}

    status: str
    model_name: str
    model_version: str | None = None
    model_semver: str | None = None
    model_alias: str
    uptime_seconds: float
    detail: str | None = None


class ModelInfo(BaseModel):
    name: str
    version: str
    model_semver: str
    alias: str
    run_id: str
    track: str
    algorithm: str
    n_features: int
    feature_columns: list[str]
    train_seasons: list[int]


class FeaturePayload(BaseModel):
    """One fixture's features, keyed exactly as the model contract names them."""

    league: str = Field(..., description="Division code, e.g. E0")
    features: dict[str, float]


class PredictRequest(BaseModel):
    rows: list[FeaturePayload]


class Prediction(BaseModel):
    p_home: float
    p_draw: float
    p_away: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    model_semver: str
    predictions: list[Prediction]


class MatchRequest(BaseModel):
    match_ids: list[str]


class FixturePrediction(BaseModel):
    match_id: str
    league: str
    date: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    p_market_home: float | None = None
    p_market_draw: float | None = None
    p_market_away: float | None = None
    best_edge_outcome: str | None = None
    best_edge: float | None = None
    value_flag: bool = False
    odds_source: str | None = None


class UpcomingResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    n_fixtures: int
    model_version: str | None = None
    model_semver: str | None = None
    benchmark: str
    note: str | None = None
    predictions: list[FixturePrediction] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _champion() -> Champion:
    champion = STATE.get("champion")
    if champion is None:
        STATE["counters"]["errors"] += 1
        raise HTTPException(
            status_code=503,
            detail=(
                f"no champion loaded: {STATE.get('error', 'unknown')}. "
                "Run `python -m src.train` to register one."
            ),
        )
    return champion


def _to_fixture_predictions(scored: pd.DataFrame) -> list[FixturePrediction]:
    out = []
    for row in scored.itertuples(index=False):
        out.append(
            FixturePrediction(
                match_id=row.match_id,
                league=row.league,
                date=str(pd.Timestamp(row.date).date()),
                home_team=row.home_team,
                away_team=row.away_team,
                p_home=float(row.p_H),
                p_draw=float(row.p_D),
                p_away=float(row.p_A),
                p_market_home=_opt(row.p_market_H),
                p_market_draw=_opt(row.p_market_D),
                p_market_away=_opt(row.p_market_A),
                best_edge_outcome=row.best_edge_outcome,
                best_edge=_opt(row.best_edge),
                value_flag=bool(row.value_flag),
                odds_source=row.odds_source,
            )
        )
    return out


def _opt(value) -> float | None:
    return None if pd.isna(value) else float(value)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    champion = STATE.get("champion")
    return HealthResponse(
        status="ok" if champion else "degraded",
        model_name=REGISTERED_MODEL_NAME,
        model_version=champion.version if champion else None,
        model_semver=champion.model_semver if champion else None,
        model_alias=CHAMPION_ALIAS,
        uptime_seconds=round(time.time() - STATE["started_at"], 1),
        detail=None if champion else STATE.get("error", "model not loaded"),
    )


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    champion = _champion()
    return ModelInfo(
        name=REGISTERED_MODEL_NAME,
        version=champion.version,
        model_semver=champion.model_semver,
        alias=CHAMPION_ALIAS,
        run_id=champion.run_id,
        track=champion.track,
        algorithm=champion.algorithm,
        n_features=len(champion.feature_columns),
        feature_columns=champion.feature_columns,
        train_seasons=champion.train_seasons,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score explicit feature payloads.

    The escape hatch for callers who already hold features -- backtesting
    harnesses, notebooks, what-if analysis. Missing features are left as NaN
    rather than zero-filled, so the model's own imputation handles them.
    """
    champion = _champion()
    if not request.rows:
        raise HTTPException(status_code=400, detail="no rows supplied")

    frame = pd.DataFrame(
        [{"league": r.league, **r.features} for r in request.rows]
    )
    for col in champion.feature_columns:
        if col not in frame.columns:
            frame[col] = float("nan")

    design = make_design(frame, champion.feature_columns)
    if list(design.columns) != champion.design_columns:
        raise HTTPException(status_code=500, detail="design matrix contract mismatch")

    proba, _ = apply_probability_floor(champion.model.predict_proba(design))
    STATE["counters"]["predict"] += 1
    return PredictResponse(
        model_version=champion.version,
        model_semver=champion.model_semver,
        predictions=[
            Prediction(p_home=float(p[0]), p_draw=float(p[1]), p_away=float(p[2]))
            for p in proba
        ],
    )


@app.post("/predict/matches", response_model=UpcomingResponse)
def predict_matches(request: MatchRequest) -> UpcomingResponse:
    """Re-score historical matches by id.

    Useful for demonstrating the service without waiting for a live matchday,
    and for spot-checking that the served model reproduces training-time
    predictions.
    """
    champion = _champion()
    matches = STATE["matches"]

    selected = matches[matches["match_id"].isin(request.match_ids)]
    if selected.empty:
        raise HTTPException(status_code=404, detail="no matching match_id found")

    history = matches[matches["date"] < selected["date"].min()]
    scored = score_fixtures(selected, champion, matches=history, elo=STATE["elo"])
    STATE["counters"]["predict_matches"] += 1

    return UpcomingResponse(
        n_fixtures=len(scored),
        model_version=champion.version,
        model_semver=champion.model_semver,
        benchmark="historical closing odds (the benchmark the model was evaluated against)",
        predictions=_to_fixture_predictions(scored) if not scored.empty else [],
    )


@app.get("/predict/upcoming", response_model=UpcomingResponse)
def predict_upcoming(threshold: float = EDGE_THRESHOLD) -> UpcomingResponse:
    """Fetch upcoming fixtures, score them, and flag value.

    Returns an empty list between seasons -- that is a legitimate result, not an
    error, and it is what this endpoint returns from late May to mid-August.
    """
    from src.ingest_fixtures import load_fixtures
    from src.prediction_markets import attach_prediction_market_prices

    champion = _champion()
    STATE["counters"]["predict_upcoming"] += 1

    fixtures = load_fixtures()
    if fixtures.empty:
        return UpcomingResponse(
            n_fixtures=0,
            model_version=champion.version,
            model_semver=champion.model_semver,
            benchmark="live pre-match odds (softer than the closing line used in backtesting)",
            note=(
                "No upcoming fixtures in the top-5 European leagues. These "
                "seasons run mid-August to late May; between them the fixture "
                "feed is legitimately empty."
            ),
        )

    fixtures = attach_prediction_market_prices(fixtures)
    scored = score_fixtures(
        fixtures, champion, matches=STATE["matches"], elo=STATE["elo"], threshold=threshold
    )

    return UpcomingResponse(
        n_fixtures=len(scored),
        model_version=champion.version,
        model_semver=champion.model_semver,
        benchmark="live pre-match odds (softer than the closing line used in backtesting)",
        note=(
            "Value flags compare against a pre-match price, not a closing price. "
            "The model loses to the closing line in backtesting, so treat flags "
            "as research output rather than betting advice."
        ),
        predictions=_to_fixture_predictions(scored) if not scored.empty else [],
    )


@app.get("/metrics")
def metrics() -> dict:
    """Lightweight counters for the dashboard."""
    champion = STATE.get("champion")
    return {
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
        "model_loaded": champion is not None,
        "model_version": champion.version if champion else None,
        "model_semver": champion.model_semver if champion else None,
        "requests": STATE["counters"],
        "outcomes": OUTCOMES,
    }
