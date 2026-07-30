"""Tests for the inference path: fixtures, prediction markets, scoring.

All hermetic. The prediction-market functions are exercised through their pure
parsing/matching helpers and through the empty-input path, so no test touches
Kalshi or Polymarket -- an outage at either venue must never redden CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest_fixtures import season_of
from src.prediction_markets import (
    LEAGUE_TO_KALSHI_SERIES,
    _best_match,
    _mid_price,
    _normalize,
    attach_prediction_market_prices,
)


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2026-08-15", 2026),  # opening weekend belongs to the new season
        ("2026-12-26", 2026),  # Boxing Day, same season
        ("2027-02-01", 2026),  # spring still belongs to the season that began in August
        ("2026-05-24", 2025),  # final matchday of the previous season
        ("2026-07-01", 2026),  # July is the cutover
    ],
)
def test_season_of_handles_the_calendar_straddle(date, expected):
    """European seasons span two calendar years; getting this wrong would put
    fixtures in the wrong season and break the match_id join."""
    assert season_of(pd.Timestamp(date)) == expected


def test_every_league_has_a_kalshi_series():
    """All five leagues must map to a full-time 1X2 ("Game") series."""
    from src.config import LEAGUE_CODES

    assert set(LEAGUE_TO_KALSHI_SERIES) == set(LEAGUE_CODES)
    assert all(t.endswith("GAME") for t in LEAGUE_TO_KALSHI_SERIES.values())


def test_normalize_removes_the_prediction_market_spread():
    """Prediction-market prices carry their own spread and will not sum to 1."""
    out = _normalize({"H": 0.50, "D": 0.28, "A": 0.30})
    assert out is not None
    assert abs(sum(out.values()) - 1.0) < 1e-12
    # Proportions are preserved.
    assert out["H"] == pytest.approx(0.50 / 1.08)


def test_normalize_rejects_incomplete_or_absurd_triples():
    assert _normalize({"H": 0.5, "D": 0.3}) is None            # missing away
    assert _normalize({"H": 0.01, "D": 0.01, "A": 0.01}) is None  # implausible total
    assert _normalize({"H": 0.9, "D": 0.9, "A": 0.9}) is None


def test_mid_price_prefers_the_bid_ask_midpoint():
    """Kalshi quotes in cents; a 40/44 market is a 0.42 probability."""
    assert _mid_price({"yes_bid": 40, "yes_ask": 44}) == pytest.approx(0.42)
    # Falls back to last traded price when the book is one-sided.
    assert _mid_price({"last_price": 37}) == pytest.approx(0.37)
    assert _mid_price({}) is None


def test_best_match_requires_both_clubs_to_match():
    candidates = [
        {"title": "Arsenal vs. Chelsea", "sides": ["arsenal to win", "chelsea to win"]},
        {"title": "Everton vs. Fulham", "sides": ["everton to win", "fulham to win"]},
    ]
    assert _best_match("Arsenal", "Chelsea", candidates)["title"] == "Arsenal vs. Chelsea"
    # One club matching is not enough -- that would pair the wrong fixture.
    assert _best_match("Arsenal", "Tottenham", candidates) is None


def test_prediction_market_enrichment_is_a_no_op_on_empty_input(tmp_path, monkeypatch):
    """The between-seasons case: no fixtures, no network, no exception."""
    monkeypatch.setattr("src.prediction_markets.COVERAGE_PATH", tmp_path / "coverage.json")
    empty = pd.DataFrame(columns=["match_id", "league", "home_team", "away_team"])
    out = attach_prediction_market_prices(empty)
    assert out.empty
    assert (tmp_path / "coverage.json").exists()


def test_enrichment_leaves_unmatched_fixtures_null(monkeypatch, tmp_path):
    """A fixture with no market keeps NaN prices so the caller falls back to
    bookmaker odds, rather than being dropped or erroring."""
    monkeypatch.setattr("src.prediction_markets.COVERAGE_PATH", tmp_path / "coverage.json")
    monkeypatch.setattr("src.prediction_markets._kalshi_prices", lambda league: [])
    monkeypatch.setattr("src.prediction_markets._polymarket_prices", lambda: [])

    fixtures = pd.DataFrame(
        {
            "match_id": ["E0_2026_Arsenal_Chelsea"],
            "league": ["E0"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    out = attach_prediction_market_prices(fixtures)
    assert len(out) == 1
    assert out["pm_p_H"].isna().all()
    assert out["pm_source"].isna().all()


def test_enrichment_attaches_a_matched_market(monkeypatch, tmp_path):
    monkeypatch.setattr("src.prediction_markets.COVERAGE_PATH", tmp_path / "coverage.json")
    monkeypatch.setattr(
        "src.prediction_markets._kalshi_prices",
        lambda league: [
            {
                "source": "kalshi",
                "event": "KXEPLGAME-26AUG15ARSCHE",
                "title": "Arsenal vs. Chelsea",
                "sides": ["arsenal to win", "chelsea to win"],
                "pm_p_H": 0.5,
                "pm_p_D": 0.25,
                "pm_p_A": 0.25,
            }
        ],
    )
    monkeypatch.setattr("src.prediction_markets._polymarket_prices", lambda: [])

    fixtures = pd.DataFrame(
        {
            "match_id": ["E0_2026_Arsenal_Chelsea"],
            "league": ["E0"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    out = attach_prediction_market_prices(fixtures)
    assert out["pm_source"].iloc[0] == "kalshi"
    assert out["pm_p_H"].iloc[0] == pytest.approx(0.5)
