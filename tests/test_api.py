"""API contract tests, run against a stub champion.

No MLflow store, no parquet files, no network. The point is to pin the HTTP
contract -- status codes, response shapes, and the degraded-mode behaviour --
independently of whether a model happens to be registered on this machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api
from src.predict import Champion


class StubModel:
    """Returns a fixed distribution, so assertions are about plumbing."""

    def predict_proba(self, X):
        return np.tile([0.5, 0.3, 0.2], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    """A client whose startup succeeds with a stub champion and tiny history."""
    feature_columns = ["home_elo", "away_elo", "elo_diff"]
    design_columns = feature_columns + [f"league_{c}" for c in api.LEAGUE_CODES_FOR_TESTS]

    champion = Champion(
        model=StubModel(),
        version="7",
        model_semver="1.0.0",
        run_id="stub-run",
        track="market_blind",
        algorithm="logistic",
        feature_columns=feature_columns,
        design_columns=design_columns,
        train_seasons=[2018, 2019],
    )

    monkeypatch.setattr(api, "load_champion", lambda: champion)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame(
        {"match_id": [], "date": pd.to_datetime([]), "league": [], "season": []}
    ))
    with TestClient(api.app) as c:
        yield c


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_name"] == "beating-1x2"
    assert body["model_version"] == "7"
    assert body["model_semver"] == "1.0.0"
    assert body["model_alias"] == "champion"


def test_model_endpoint_exposes_the_contract(client):
    body = client.get("/model").json()
    assert body["track"] == "market_blind"
    assert body["algorithm"] == "logistic"
    assert body["model_semver"] == "1.0.0"
    assert body["n_features"] == 3
    assert body["feature_columns"] == ["home_elo", "away_elo", "elo_diff"]


def test_predict_returns_a_distribution_per_row(client):
    body = client.post(
        "/predict",
        json={
            "rows": [
                {"league": "E0", "features": {"home_elo": 1800, "away_elo": 1700, "elo_diff": 100}},
                {"league": "SP1", "features": {"home_elo": 1600, "away_elo": 1900, "elo_diff": -300}},
            ]
        },
    ).json()

    assert body["model_version"] == "7"
    assert body["model_semver"] == "1.0.0"
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        total = prediction["p_home"] + prediction["p_draw"] + prediction["p_away"]
        assert total == pytest.approx(1.0)


def test_predict_tolerates_missing_features(client):
    """Absent features become NaN for the model's own imputer, rather than
    being silently zero-filled -- zero is a meaningful Elo-difference, NaN is
    not."""
    response = client.post(
        "/predict", json={"rows": [{"league": "E0", "features": {"home_elo": 1800}}]}
    )
    assert response.status_code == 200


def test_predict_rejects_an_empty_request(client):
    assert client.post("/predict", json={"rows": []}).status_code == 400


def test_unknown_match_id_is_a_404(client):
    response = client.post("/predict/matches", json={"match_ids": ["does_not_exist"]})
    assert response.status_code == 404


def test_metrics_counts_requests(client):
    client.post(
        "/predict",
        json={"rows": [{"league": "E0", "features": {"home_elo": 1800, "away_elo": 1700}}]},
    )
    assert client.get("/metrics").json()["requests"]["predict"] >= 1


def test_service_degrades_rather_than_dying_without_a_model(monkeypatch):
    """If no champion is registered, /health must explain why and scoring must
    return 503 -- a container that exits on startup tells the operator nothing."""

    def explode():
        raise RuntimeError("no registered model named beating-1x2")

    monkeypatch.setattr(api, "load_champion", explode)
    with TestClient(api.app) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded"
        assert "beating-1x2" in health["detail"]
        assert c.get("/model").status_code == 503
