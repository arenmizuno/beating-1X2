"""Best-effort prediction-market prices for upcoming fixtures.

The original project proposal named Polymarket/Kalshi as the market source.
Phase 1 established they cannot supply *history* -- there is no per-match 1X2
archive back to 2018 -- so bookmaker closing odds became the training and
backtesting benchmark. This module restores prediction markets in the role they
can actually fill: a live, inference-time price to compare against.

Coverage was measured before this was written, and it is better than expected:

  Kalshi carries full-time 1X2 match series for every one of our five leagues
  -- KXEPLGAME, KXLALIGAGAME, KXBUNDESLIGAGAME, KXSERIEAGAME, KXLIGUE1GAME.

  Polymarket's soccer coverage skews to season-long and novelty markets
  (Ballon d'Or, "2027 Champion", transfers) with only occasional per-match
  events, so it is queried as a secondary source.

Everything here is best-effort by design. A prediction-market outage, a schema
change, or simply no market existing for a fixture must degrade to "no price
found" and never break a prediction request. Coverage is counted and written to
reports/prediction_market_coverage.json so the real hit rate is reported rather
than assumed.

NOTE ON VALIDATION: the parsing below is written against each venue's documented
response shape and is exercised structurally by the test suite. It could not be
validated against live 1X2 markets at build time, because the top-5 European
leagues are between seasons (every series returned zero open markets in late
July). The coverage report is the mechanism for confirming it once play resumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
import requests
from rapidfuzz import fuzz

from src.config import (
    OUTCOMES,
    REPORTS_DIR,
    USER_AGENT,
    ensure_dirs,
    get_logger,
)

log = get_logger("prediction_markets")

COVERAGE_PATH = REPORTS_DIR / "prediction_market_coverage.json"

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Full-time match-winner ("Game") series, one per league we cover.
LEAGUE_TO_KALSHI_SERIES = {
    "E0": "KXEPLGAME",
    "SP1": "KXLALIGAGAME",
    "D1": "KXBUNDESLIGAGAME",
    "I1": "KXSERIEAGAME",
    "F1": "KXLIGUE1GAME",
}

# Deliberately short. This is an optional enrichment on a request path; it must
# never be the reason a prediction is slow.
TIMEOUT_S = 8
NAME_MATCH_THRESHOLD = 82


@dataclass
class Coverage:
    """Tallies how often a prediction-market price was actually found."""

    fixtures: int = 0
    matched: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record(self, source: str) -> None:
        self.by_source[source] = self.by_source.get(source, 0) + 1
        self.matched += 1

    def as_dict(self) -> dict:
        return {
            "fixtures_queried": self.fixtures,
            "fixtures_with_prediction_market_price": self.matched,
            "coverage_rate": (self.matched / self.fixtures) if self.fixtures else 0.0,
            "by_source": self.by_source,
            "errors": self.errors[:20],
        }


def _get(url: str, params: dict) -> dict | list | None:
    """Single best-effort GET. Any failure is a None, never an exception."""
    try:
        response = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001 -- enrichment must never break serving
        log.debug("request failed %s: %s", url, exc)
        return None


def _mid_price(market: dict) -> float | None:
    """Kalshi quotes in cents. Prefer the bid/ask midpoint, fall back to last."""
    bid, ask = market.get("yes_bid"), market.get("yes_ask")
    if bid and ask:
        return (float(bid) + float(ask)) / 200.0
    last = market.get("last_price")
    return float(last) / 100.0 if last else None


def _normalize(probs: dict[str, float]) -> dict[str, float] | None:
    """Prediction-market prices carry their own spread, so a 1X2 triple will not
    sum to 1. Normalize, which is the multiplicative devig of src/market.py."""
    if set(probs) != set(OUTCOMES):
        return None
    total = sum(probs.values())
    if not (0.5 < total < 2.0):
        return None
    return {k: v / total for k, v in probs.items()}


def _kalshi_prices(league: str) -> list[dict]:
    """Open 1X2 markets for one league, grouped into per-event triples."""
    series = LEAGUE_TO_KALSHI_SERIES.get(league)
    if not series:
        return []

    payload = _get(
        KALSHI_MARKETS_URL, {"series_ticker": series, "status": "open", "limit": 200}
    )
    if not isinstance(payload, dict):
        return []

    events: dict[str, dict] = {}
    for market in payload.get("markets", []):
        event = market.get("event_ticker")
        if not event:
            continue
        price = _mid_price(market)
        if price is None:
            continue
        # Kalshi splits a 3-way game into one binary contract per outcome; the
        # sub-title names which one ("Home", "Draw", "<Team> to win").
        label = f"{market.get('yes_sub_title') or ''} {market.get('title') or ''}".lower()
        bucket = events.setdefault(event, {"title": market.get("title", ""), "probs": {}})
        if "draw" in label or "tie" in label:
            bucket["probs"]["D"] = price
        else:
            bucket.setdefault("_sides", []).append((label, price))

    out = []
    for event, bucket in events.items():
        sides = bucket.get("_sides", [])
        if len(sides) != 2:
            continue
        # Kalshi orders a game's contracts home-then-away within the event.
        bucket["probs"]["H"] = sides[0][1]
        bucket["probs"]["A"] = sides[1][1]
        normalized = _normalize(bucket["probs"])
        if normalized:
            out.append(
                {
                    "source": "kalshi",
                    "event": event,
                    "title": bucket["title"],
                    "sides": [s[0] for s in sides],
                    **{f"pm_p_{k}": v for k, v in normalized.items()},
                }
            )
    return out


def _polymarket_prices() -> list[dict]:
    """Open soccer events on Polymarket, kept as a secondary source."""
    payload = _get(POLYMARKET_EVENTS_URL, {"closed": "false", "limit": 200, "tag_slug": "soccer"})
    if not isinstance(payload, list):
        return []

    out = []
    for event in payload:
        title = event.get("title", "")
        # Per-match events are titled "<Home> vs. <Away>".
        if " vs" not in title.lower():
            continue
        probs: dict[str, float] = {}
        for market in event.get("markets", []):
            question = (market.get("question") or market.get("groupItemTitle") or "").lower()
            try:
                price = float(json.loads(market.get("outcomePrices", "[]"))[0])
            except Exception:  # noqa: BLE001
                continue
            if "draw" in question or "tie" in question:
                probs["D"] = price
            elif "H" not in probs:
                probs["H"] = price
            else:
                probs["A"] = price
        normalized = _normalize(probs)
        if normalized:
            out.append(
                {
                    "source": "polymarket",
                    "event": event.get("slug", ""),
                    "title": title,
                    **{f"pm_p_{k}": v for k, v in normalized.items()},
                }
            )
    return out


def _best_match(home: str, away: str, candidates: list[dict]) -> dict | None:
    """Pair a fixture with a market by name similarity on both clubs."""
    best, best_score = None, 0.0
    for candidate in candidates:
        text = f"{candidate.get('title', '')} {' '.join(candidate.get('sides', []))}"
        score = min(fuzz.partial_ratio(home.lower(), text.lower()),
                    fuzz.partial_ratio(away.lower(), text.lower()))
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= NAME_MATCH_THRESHOLD else None


def attach_prediction_market_prices(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach pm_p_H/D/A and pm_source to each fixture, where a market exists.

    Fixtures with no matching market keep NaN prices -- the caller falls back to
    bookmaker odds. Never raises.
    """
    ensure_dirs()
    coverage = Coverage(fixtures=len(fixtures))
    enriched = fixtures.copy()
    for col in [f"pm_p_{o}" for o in OUTCOMES] + ["pm_source", "pm_event"]:
        enriched[col] = None

    if fixtures.empty:
        log.info("no fixtures to enrich")
        _write_coverage(coverage)
        return enriched

    candidates: list[dict] = []
    for league in sorted(fixtures["league"].unique()):
        found = _kalshi_prices(league)
        log.info("kalshi %-4s (%s): %d open 1X2 markets", league, LEAGUE_TO_KALSHI_SERIES.get(league), len(found))
        candidates.extend(found)

    polymarket = _polymarket_prices()
    log.info("polymarket: %d per-match soccer events", len(polymarket))
    candidates.extend(polymarket)

    if not candidates:
        log.warning(
            "no prediction markets found for any fixture -- expected between "
            "seasons, or when a venue has not listed these matches"
        )
        _write_coverage(coverage)
        return enriched

    for idx, row in enriched.iterrows():
        match = _best_match(row["home_team"], row["away_team"], candidates)
        if not match:
            continue
        for outcome in OUTCOMES:
            enriched.at[idx, f"pm_p_{outcome}"] = match[f"pm_p_{outcome}"]
        enriched.at[idx, "pm_source"] = match["source"]
        enriched.at[idx, "pm_event"] = match["event"]
        coverage.record(match["source"])

    log.info(
        "prediction-market coverage: %d/%d fixtures (%.1f%%) %s",
        coverage.matched,
        coverage.fixtures,
        100 * coverage.matched / max(coverage.fixtures, 1),
        coverage.by_source,
    )
    _write_coverage(coverage)
    return enriched


def _write_coverage(coverage: Coverage) -> None:
    COVERAGE_PATH.write_text(json.dumps(coverage.as_dict(), indent=2))
    log.info("wrote %s", COVERAGE_PATH)


if __name__ == "__main__":
    from src.ingest_fixtures import load_fixtures

    attach_prediction_market_prices(load_fixtures())
