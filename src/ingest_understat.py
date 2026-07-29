"""Stage 1b: expected-goals (xG) data from Understat.

Understat has no documented public API. Its league pages used to inline the
fixture list as a hex-escaped `var datesData = JSON.parse('...')` blob; the site
now loads that same payload over AJAX from /getLeagueData/<league>/<season>,
which returns {teams, players, dates} as JSON. We read that endpoint directly.
Responses are cached on disk and throttled (see src/fetch.py) so re-running the
pipeline costs nothing.

IMPORTANT -- xG is a POST-MATCH measurement. Nothing in this file may be used as
a same-match feature. src/features.py consumes it strictly as lagged rolling
aggregates. We also deliberately discard Understat's `forecast` field: it is a
model output derived from that match's own xG, so it is doubly unusable.

Understat timestamps kickoffs in a different timezone than football-data.co.uk
records match dates, so the two disagree by a day on late kickoffs. We keep the
full timestamp here and let src/harmonize.py join with a date tolerance.

Output: data/interim/understat.parquet
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import (
    INTERIM_DIR,
    LEAGUES,
    RAW_UNDERSTAT_DIR,
    SEASONS,
    UNDERSTAT_URL,
    ensure_dirs,
    get_logger,
    season_to_understat,
)
from src.fetch import fetch_text

log = get_logger("ingest.understat")

OUTPUT_PATH = INTERIM_DIR / "understat.parquet"

# The endpoint is the site's own AJAX route; it expects to be called as one.
_AJAX_HEADERS = {"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"}


def load_season(league: str, season: int) -> pd.DataFrame | None:
    understat_league = LEAGUES[league]["understat"]
    us_season = season_to_understat(season)
    url = UNDERSTAT_URL.format(league=understat_league, season=us_season)
    cache_path = RAW_UNDERSTAT_DIR / f"{understat_league}_{us_season}.json"

    try:
        payload = json.loads(fetch_text(url, cache_path, headers=_AJAX_HEADERS))
    except (RuntimeError, ValueError) as exc:
        log.warning("skipping %s %s: %s", understat_league, us_season, exc)
        return None

    records = payload.get("dates")
    if not records:
        log.warning("no `dates` payload for %s %s", understat_league, us_season)
        return None

    rows = []
    for rec in records:
        # Fixtures not yet played carry isResult=False with null goals/xG.
        if not rec.get("isResult"):
            continue
        try:
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "understat_datetime": pd.to_datetime(rec["datetime"]),
                    "home_team_raw": rec["h"]["title"].strip(),
                    "away_team_raw": rec["a"]["title"].strip(),
                    "home_goals": int(rec["goals"]["h"]),
                    "away_goals": int(rec["goals"]["a"]),
                    "home_xg": float(rec["xG"]["h"]),
                    "away_xg": float(rec["xG"]["a"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("  malformed record skipped: %s", exc)

    if not rows:
        log.warning("no completed fixtures for %s %s", understat_league, us_season)
        return None

    df = pd.DataFrame(rows)
    df["date"] = df["understat_datetime"].dt.normalize()

    log.info("%-12s %s: %4d matches with xG", understat_league, us_season, len(df))
    return df


def main() -> pd.DataFrame:
    ensure_dirs()
    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            frame = load_season(league, season)
            if frame is not None and not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("no Understat seasons ingested -- check connectivity")

    xg = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "league", "home_team_raw"], kind="stable"
    )
    xg = xg.reset_index(drop=True)

    xg.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s (%d matches)", OUTPUT_PATH, len(xg))

    counts = xg.groupby(["season", "league"]).size().unstack(fill_value=0)
    log.info("xG matches per season/league:\n%s", counts.to_string())
    return xg


if __name__ == "__main__":
    main()
