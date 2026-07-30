"""Upcoming fixtures with live pre-match odds.

football-data.co.uk publishes a rolling `fixtures.csv` of matches not yet
played, carrying the same bookmaker columns as its historical season files.
Because it comes from the same publisher, team names are already in our
canonical spelling -- no alias mapping is needed here, unlike src/harmonize.py.

Two honest caveats, both surfaced rather than hidden:

**These are pre-match prices, not closing prices.** The whole evaluation
benchmarks against Pinnacle *closing* odds (`PSC`), which are the sharpest and
hardest line to beat. A live fixture has no closing price yet, so we fall back
to the current Pinnacle price (`PS`). That line is systematically softer, which
means live value flags are computed against an easier benchmark than the one the
backtest used, and will fire more readily. `src/predict.py` reports the odds
source on every prediction so this is visible at the point of use.

**The feed is seasonal.** Between late May and mid-August the top-5 European
leagues are not playing, and the file legitimately contains no rows for them.
That is an empty result, not an error.

Output: data/interim/fixtures.parquet
"""

from __future__ import annotations

import io

import pandas as pd

from src.config import (
    INTERIM_DIR,
    LEAGUE_CODES,
    RAW_DIR,
    ensure_dirs,
    get_logger,
)
from src.fetch import fetch_bytes
from src.ingest_footballdata import ODDS_PREFIXES

log = get_logger("ingest.fixtures")

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
OUTPUT_PATH = INTERIM_DIR / "fixtures.parquet"


def season_of(date: pd.Timestamp) -> int:
    """European seasons straddle the calendar year and start in July/August.

    A fixture in August 2026 belongs to season 2026 (i.e. 2026-27); one in
    February 2026 belongs to season 2025.
    """
    return int(date.year if date.month >= 7 else date.year - 1)


def load_fixtures(*, use_cache: bool = False) -> pd.DataFrame:
    """Fetch and normalize upcoming fixtures for our five leagues.

    Defaults to bypassing the cache: unlike historical seasons, this file
    changes constantly and a stale copy would silently serve yesterday's prices.
    """
    ensure_dirs()
    cache_path = RAW_DIR / "fixtures.csv"
    body = fetch_bytes(FIXTURES_URL, cache_path, use_cache=use_cache)

    df = pd.read_csv(io.BytesIO(body), encoding="latin-1", on_bad_lines="skip")

    # Unlike the historical season files, this one is served with a UTF-8 BOM.
    # Decoded as latin-1 (which we need for accented club names) the BOM survives
    # as a literal prefix on the first header, turning "Div" into "ï»¿Div".
    df.columns = [str(c).lstrip("﻿").replace("ï»¿", "").strip() for c in df.columns]

    df = df.dropna(subset=["Div", "HomeTeam", "AwayTeam"])
    log.info("fixtures feed: %d rows across %d divisions", len(df), df["Div"].nunique())

    df = df[df["Div"].isin(LEAGUE_CODES)].copy()
    if df.empty:
        log.warning(
            "no upcoming fixtures for %s -- the top-5 European leagues are "
            "between seasons (they resume in mid-August)",
            ", ".join(LEAGUE_CODES),
        )
        return pd.DataFrame(
            columns=["match_id", "league", "season", "date", "home_team", "away_team"]
        )

    dates = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    missing = dates.isna()
    if missing.any():
        dates.loc[missing] = pd.to_datetime(
            df.loc[missing, "Date"], format="%d/%m/%y", errors="coerce"
        )

    out = pd.DataFrame(
        {
            "league": df["Div"].values,
            "date": dates.values,
            "home_team": df["HomeTeam"].str.strip().values,
            "away_team": df["AwayTeam"].str.strip().values,
            "kickoff_time": df["Time"].values if "Time" in df.columns else None,
        }
    ).dropna(subset=["date"])

    out["season"] = out["date"].map(season_of)

    # Same construction as src/harmonize.py, so a fixture's id matches the id it
    # will carry once the result is published and it enters the history.
    out["match_id"] = (
        out["league"]
        + "_"
        + out["season"].astype(str)
        + "_"
        + out["home_team"].str.replace(" ", "", regex=False)
        + "_"
        + out["away_team"].str.replace(" ", "", regex=False)
    )

    # Carry whatever odds prefixes the feed happens to publish, normalized to the
    # same odds_<PREFIX>_<H|D|A> schema src/market.py consumes. Note the feed has
    # no closing (`*C`) columns -- nothing has closed yet.
    present = []
    for prefix in ODDS_PREFIXES:
        cols = [f"{prefix}{s}" for s in ("H", "D", "A")]
        if all(c in df.columns for c in cols):
            for col, suffix in zip(cols, ("H", "D", "A"), strict=True):
                out[f"odds_{prefix}_{suffix}"] = pd.to_numeric(
                    df[col], errors="coerce"
                ).values
            present.append(prefix)

    log.info(
        "%d upcoming fixtures in our leagues | odds: %s",
        len(out),
        ", ".join(present) or "NONE",
    )
    if len(out):
        log.info(
            "date range %s to %s",
            out["date"].min().date(),
            out["date"].max().date(),
        )
    return out.sort_values(["date", "league"], kind="stable").reset_index(drop=True)


def main() -> pd.DataFrame:
    fixtures = load_fixtures()
    fixtures.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s (%d fixtures)", OUTPUT_PATH, len(fixtures))
    return fixtures


if __name__ == "__main__":
    main()
