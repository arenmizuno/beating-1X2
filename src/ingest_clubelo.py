"""Stage 1c: club strength ratings from ClubElo.

ClubElo exposes two CSV endpoints:

  /<YYYY-MM-DD>  every club's rating on that date
  /<ClubName>    one club's full rating history, as [From, To] validity ranges

We use the date endpoint only to *discover* which clubs were in our five top
divisions (two snapshots per season catches promotions and mid-season movement),
then pull each club's full history once. That is ~130 requests total, versus one
request per match date if we queried snapshots directly.

The [From, To] range form is what makes this leakage-safe downstream: a rating
row is valid over a closed date interval, so src/features.py can ask for the
rating in force *before* a given kickoff rather than after it.

Output: data/interim/clubelo.parquet
"""

from __future__ import annotations

import io

import pandas as pd

from src.config import (
    CLUBELO_DATE_URL,
    CLUBELO_TEAM_URL,
    INTERIM_DIR,
    RAW_CLUBELO_DIR,
    REPORTS_DIR,
    SEASONS,
    ensure_dirs,
    get_logger,
)
from src.fetch import fetch_bytes

log = get_logger("ingest.clubelo")

OUTPUT_PATH = INTERIM_DIR / "clubelo.parquet"

# ClubElo's country codes for our five leagues.
LEAGUE_TO_COUNTRY = {"E0": "ENG", "SP1": "ESP", "D1": "GER", "I1": "ITA", "F1": "FRA"}
COUNTRIES = set(LEAGUE_TO_COUNTRY.values())
COUNTRY_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_COUNTRY.items()}


def _club_to_url_name(club: str) -> str:
    """ClubElo's per-club route strips spaces: 'Man City' -> 'ManCity'."""
    return club.replace(" ", "")


def _read_clubelo_csv(body: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(body))
    # Rank is the literal string "None" for clubs outside the top division.
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Elo"] = pd.to_numeric(df["Elo"], errors="coerce")
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df["From"] = pd.to_datetime(df["From"], errors="coerce")
    df["To"] = pd.to_datetime(df["To"], errors="coerce")
    return df


def discover_clubs() -> dict[str, str]:
    """Return {club_name: country} for top-division clubs across all seasons.

    Snapshots are taken at the start of each season and again mid-season, which
    is enough to catch every club that appears in our fixture list.
    """
    clubs: dict[str, str] = {}
    snapshot_dates = []
    for season in SEASONS:
        snapshot_dates.append(f"{season}-08-01")
        snapshot_dates.append(f"{season + 1}-01-15")

    for date in snapshot_dates:
        url = CLUBELO_DATE_URL.format(date=date)
        cache_path = RAW_CLUBELO_DIR / "snapshots" / f"{date}.csv"
        try:
            df = _read_clubelo_csv(fetch_bytes(url, cache_path))
        except (RuntimeError, ValueError) as exc:
            log.warning("snapshot %s failed: %s", date, exc)
            continue

        top = df[(df["Country"].isin(COUNTRIES)) & (df["Level"] == 1)]
        for club, country in zip(top["Club"], top["Country"]):
            clubs.setdefault(str(club).strip(), country)
        log.info("snapshot %s: %3d top-division clubs (running total %d)", date, len(top), len(clubs))

    return clubs


def load_club_history(club: str, country: str) -> pd.DataFrame | None:
    url = CLUBELO_TEAM_URL.format(team=_club_to_url_name(club))
    cache_path = RAW_CLUBELO_DIR / "clubs" / f"{_club_to_url_name(club)}.csv"

    try:
        df = _read_clubelo_csv(fetch_bytes(url, cache_path))
    except (RuntimeError, ValueError) as exc:
        log.warning("club history failed for %-24s: %s", club, exc)
        return None

    if df.empty:
        log.warning("empty history for %s", club)
        return None

    # Trim to our study window (plus a season of lead-in) rather than carrying
    # ratings back to 1946.
    lo = pd.Timestamp(f"{SEASONS[0] - 1}-01-01")
    hi = pd.Timestamp(f"{SEASONS[-1] + 2}-01-01")
    df = df[(df["To"] >= lo) & (df["From"] <= hi)]

    if df.empty:
        log.warning("no in-window ratings for %s", club)
        return None

    return pd.DataFrame(
        {
            "club_raw": club,
            "country": country,
            "league": COUNTRY_TO_LEAGUE[country],
            "level": df["Level"].values,
            "elo": df["Elo"].values,
            "from_date": df["From"].values,
            "to_date": df["To"].values,
        }
    )


def main() -> pd.DataFrame:
    ensure_dirs()

    clubs = discover_clubs()
    if not clubs:
        raise RuntimeError("no clubs discovered from ClubElo -- check connectivity")
    log.info("discovered %d distinct clubs across %d seasons", len(clubs), len(SEASONS))

    frames, failed = [], []
    for i, (club, country) in enumerate(sorted(clubs.items()), start=1):
        frame = load_club_history(club, country)
        if frame is None:
            failed.append({"club": club, "country": country})
            continue
        frames.append(frame)
        if i % 25 == 0:
            log.info("  %d/%d clubs fetched", i, len(clubs))

    if not frames:
        raise RuntimeError("no ClubElo histories retrieved")

    elo = pd.concat(frames, ignore_index=True).sort_values(
        ["club_raw", "from_date"], kind="stable"
    )
    elo = elo.reset_index(drop=True)
    elo.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s (%d rating rows, %d clubs)", OUTPUT_PATH, len(elo), elo["club_raw"].nunique())

    if failed:
        failed_path = REPORTS_DIR / "clubelo_failed_clubs.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        log.warning("%d clubs could not be fetched -- see %s", len(failed), failed_path)

    per_league = elo.groupby("league")["club_raw"].nunique()
    log.info("clubs per league:\n%s", per_league.to_string())
    return elo


if __name__ == "__main__":
    main()
