"""Stage 1a: results and closing odds from football-data.co.uk.

This is the backbone of the dataset. It supplies the label (full-time result)
and -- importantly -- the market prices. The project proposal originally named
Polymarket/Kalshi as the market source; neither has per-match 3-way pricing back
to 2018, whereas football-data has shipped closing odds for these exact leagues
and seasons all along. See README for the full rationale.

Schema drift is the main hazard here. football-data changed its aggregate-odds
columns around 2019-20 (the `Bb*` family was replaced by `Max*`/`Avg*`), so
rather than assume a fixed layout we probe for every known odds prefix and
record which ones each season actually carried.

Output: data/interim/footballdata.parquet
"""

from __future__ import annotations

import io

import pandas as pd

from src.config import (
    FOOTBALLDATA_URL,
    INTERIM_DIR,
    LEAGUE_CODES,
    RAW_FOOTBALLDATA_DIR,
    SEASONS,
    ensure_dirs,
    get_logger,
    season_to_footballdata,
)
from src.fetch import fetch_bytes

log = get_logger("ingest.footballdata")

OUTPUT_PATH = INTERIM_DIR / "footballdata.parquet"

# Odds column prefixes we know football-data has used. A prefix P means the
# triple (P+"H", P+"D", P+"A") holds decimal odds for home/draw/away.
#
#   PSC   Pinnacle closing        <- sharpest, preferred
#   PS    Pinnacle opening
#   B365C Bet365 closing
#   AvgC  market average closing
#   MaxC  market maximum closing
#   BbAv  Betbrain average        <- pre-2019-20 only
#   BbMx  Betbrain maximum        <- pre-2019-20 only
#
# We extract whatever is present; src/market.py applies the preference order.
ODDS_PREFIXES = ["PSC", "PS", "B365C", "B365", "AvgC", "Avg", "MaxC", "Max", "BbAv", "BbMx"]

# Columns we need from every season regardless of vintage.
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _parse_dates(raw: pd.Series) -> pd.Series:
    """football-data writes dates as dd/mm/yy in older files and dd/mm/yyyy in
    newer ones, sometimes mixing both within a single season file."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(raw[missing], format="%d/%m/%y", errors="coerce")
        parsed.loc[missing] = fallback
    return parsed


def load_season(league: str, season: int) -> pd.DataFrame | None:
    """Download (or read from cache) and normalize one league-season CSV."""
    fd_season = season_to_footballdata(season)
    url = FOOTBALLDATA_URL.format(season=fd_season, league=league)
    cache_path = RAW_FOOTBALLDATA_DIR / f"{league}_{fd_season}.csv"

    try:
        body = fetch_bytes(url, cache_path)
    except RuntimeError as exc:
        # A season that has not started yet legitimately 404s. Warn, don't fail.
        log.warning("skipping %s %s: %s", league, fd_season, exc)
        return None

    # latin-1 rather than utf-8: these files carry accented club names and are
    # not utf-8 encoded.
    df = pd.read_csv(io.BytesIO(body), encoding="latin-1", on_bad_lines="skip")

    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        log.warning("skipping %s %s: missing core columns %s", league, fd_season, missing_core)
        return None

    # Trailing all-blank rows are common at the end of these files.
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).copy()

    out = pd.DataFrame(
        {
            "league": league,
            "season": season,
            "date": _parse_dates(df["Date"]),
            "home_team_raw": df["HomeTeam"].str.strip(),
            "away_team_raw": df["AwayTeam"].str.strip(),
            "fthg": pd.to_numeric(df["FTHG"], errors="coerce"),
            "ftag": pd.to_numeric(df["FTAG"], errors="coerce"),
            "ftr": df["FTR"].str.strip().str.upper(),
        }
    )

    present_prefixes = []
    for prefix in ODDS_PREFIXES:
        cols = [f"{prefix}{suffix}" for suffix in ("H", "D", "A")]
        if all(c in df.columns for c in cols):
            for col, suffix in zip(cols, ("H", "D", "A")):
                out[f"odds_{prefix}_{suffix}"] = pd.to_numeric(df[col], errors="coerce")
            present_prefixes.append(prefix)

    out = out.dropna(subset=["date"])
    out = out[out["ftr"].isin(["H", "D", "A"])]

    log.info(
        "%-4s %s: %4d matches | odds: %s",
        league,
        fd_season,
        len(out),
        ", ".join(present_prefixes) or "NONE",
    )
    if not present_prefixes:
        log.warning("  no usable odds columns for %s %s", league, fd_season)

    return out


def main() -> pd.DataFrame:
    ensure_dirs()
    frames = []
    for league in LEAGUE_CODES:
        for season in SEASONS:
            frame = load_season(league, season)
            if frame is not None and not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("no football-data seasons ingested -- check connectivity")

    matches = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "league", "home_team_raw"], kind="stable"
    )
    matches = matches.reset_index(drop=True)

    matches.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s (%d matches, %d columns)", OUTPUT_PATH, len(matches), matches.shape[1])

    counts = matches.groupby(["season", "league"]).size().unstack(fill_value=0)
    log.info("matches per season/league:\n%s", counts.to_string())

    result_mix = matches["ftr"].value_counts(normalize=True).reindex(["H", "D", "A"])
    log.info(
        "outcome base rates: H=%.1f%% D=%.1f%% A=%.1f%%",
        *(100 * result_mix.values),
    )
    return matches


if __name__ == "__main__":
    main()
