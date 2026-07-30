"""Stage 2: reconcile team names across sources and join into one match table.

This is the highest-risk stage in the pipeline. The three sources spell clubs
differently ("Man United" / "Manchester United", "Ath Bilbao" / "Athletic Club",
"Espanol" / "Espanyol"), and a silent partial join here would quietly poison
every downstream result. So the design is deliberately loud and deliberately
reviewable.

Two ideas do most of the work:

1. **Optimal assignment, not greedy matching.** Within a single league-season
   both sources describe the *same* set of ~20 clubs. That makes alias
   resolution a bipartite assignment problem, not a series of independent
   lookups. Solving it globally (Hungarian algorithm over a rapidfuzz score
   matrix) means the hard pairs resolve correctly once the easy ones are
   claimed -- "Ath Bilbao" finds "Athletic Club" because nothing else is left.

2. **Ordered pairs are unique keys.** Every top-5 league here is a double
   round-robin, so within a league-season the ordered pair (home, away) plays
   exactly once. We can therefore join on the pair alone and treat the date as
   an *assertion* rather than a join condition -- which is what we want, since
   Understat timestamps kickoffs in a different timezone and legitimately
   disagrees with football-data by a day on late fixtures.

Input:
  data/mappings/team_aliases_manual.csv   hand-verified overrides (committed)

Outputs:
  data/interim/matches.parquet            football-data spine + xG
  data/interim/clubelo_canonical.parquet  Elo history under canonical names
  data/interim/team_aliases_resolved.csv  full alias table (manual + auto)
  reports/unmatched_teams.csv             anything needing a human eye
  reports/date_disagreements.csv          pairs joined but dated far apart
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from src.config import (
    INTERIM_DIR,
    MANUAL_ALIASES_PATH,
    MAPPINGS_DIR,
    MATCHES_PATH,
    PARAMS,
    REPORTS_DIR,
    RESOLVED_ALIASES_PATH,
    ensure_dirs,
    get_logger,
)

log = get_logger("harmonize")

CLUBELO_CANONICAL_PATH = INTERIM_DIR / "clubelo_canonical.parquet"

_H = PARAMS["harmonize"]
AUTO_ACCEPT_SCORE: float = _H["auto_accept_score"]
REVIEW_THRESHOLD: float = _H["review_threshold"]
DATE_TOLERANCE_DAYS: int = _H["date_tolerance_days"]
MIN_JOIN_RATE: float = _H["min_join_rate"]


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------
def _score(a: str, b: str) -> float:
    """Blend two rapidfuzz scorers.

    WRatio handles abbreviation ("Man United" vs "Manchester United"); the
    token-set ratio handles dropped qualifiers ("Betis" vs "Real Betis"). Taking
    the max of the two is more forgiving than either alone, and the assignment
    step below is what keeps that forgiveness from causing mismatches.
    """
    return max(fuzz.WRatio(a, b), fuzz.token_set_ratio(a, b))


def assign_names(source_names: list[str], canon_names: list[str]) -> list[tuple[str, str, float]]:
    """Optimally pair every source name with a distinct canonical name.

    Returns (source_name, canonical_name, score) triples. If the two sets differ
    in size, the smaller one is matched exhaustively and the surplus is left
    unpaired (reported by the caller).
    """
    if not source_names or not canon_names:
        return []

    scores = np.array([[_score(s, c) for c in canon_names] for s in source_names], dtype=float)
    # linear_sum_assignment minimises, so negate to maximise total similarity.
    rows, cols = linear_sum_assignment(-scores)
    return [(source_names[r], canon_names[c], float(scores[r, c])) for r, c in zip(rows, cols, strict=True)]


def load_manual_aliases() -> pd.DataFrame:
    """Hand-verified overrides, treated as ground truth.

    Fuzzy string similarity cannot resolve everything: "Athletic Club" and
    "Ath Bilbao" are the same club but share almost no characters, and
    "SPAL 2013" versus "Spal" scores lower than genuinely unrelated names. Those
    cases get decided once, by a human, and recorded here.
    """
    if not MANUAL_ALIASES_PATH.exists():
        log.warning("no manual alias file at %s", MANUAL_ALIASES_PATH)
        return pd.DataFrame(columns=["source", "league", "raw_name", "canonical_name"])

    manual = pd.read_csv(MANUAL_ALIASES_PATH)
    manual = manual[["source", "league", "raw_name", "canonical_name"]].copy()
    manual["score"] = 100.0
    manual["status"] = "manual"
    manual["n_distinct_targets"] = 1
    manual["seasons"] = "manual"
    log.info("loaded %d manual alias override(s) from %s", len(manual), MANUAL_ALIASES_PATH.name)
    return manual


def resolve_source_aliases(
    canon_by_group: dict[tuple[str, int], set[str]],
    source_by_group: dict[tuple[str, int], set[str]],
    source_label: str,
    manual: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve aliases per (league, season), then consolidate across seasons.

    A club keeps the same alias across seasons, so we resolve independently in
    each season -- where the assignment constraint is strongest -- and then take
    the highest-scoring resolution per (league, raw_name).

    Manually overridden names are removed from both sides of the assignment
    first. Leaving them in would let the matcher re-derive (and possibly
    contradict) a decision a human has already made, and taking their canonical
    targets out of the pool makes the remaining assignment easier.
    """
    pinned: dict[tuple[str, str], str] = {}
    if manual is not None and not manual.empty:
        source_manual = manual[manual["source"] == source_label]
        pinned = {
            (row.league, row.raw_name): row.canonical_name
            for row in source_manual.itertuples(index=False)
        }

    records = []
    for group, canon_names in canon_by_group.items():
        source_names = source_by_group.get(group)
        if not source_names:
            log.warning("%s: no teams for %s %s", source_label, *group)
            continue

        league, season = group

        # Remove already-decided names, and the canonical targets they claim.
        free_source = sorted(n for n in source_names if (league, n) not in pinned)
        claimed = {pinned[(league, n)] for n in source_names if (league, n) in pinned}
        free_canon = sorted(n for n in canon_names if n not in claimed)

        pairs = assign_names(free_source, free_canon)
        for raw, canonical, score in pairs:
            records.append(
                {
                    "source": source_label,
                    "league": league,
                    "season": season,
                    "raw_name": raw,
                    "canonical_name": canonical,
                    "score": score,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["source", "league", "raw_name", "canonical_name", "score", "seasons", "status"]
        )

    per_season = pd.DataFrame(records)

    # Drop implausible pairings BEFORE consolidating.
    #
    # The assignment is exhaustive: if a season's source set is larger than the
    # canonical set -- which happens when a relegated club's level-1 ClubElo
    # rating still overlaps the season window -- the surplus names get forced
    # onto whatever targets are left, at near-zero scores. Those forced pairings
    # are not evidence of an ambiguous club name, they are non-matches, and
    # letting them survive would corrupt the cross-season consistency check
    # below. (Observed: Mallorca scored 100 in 2019 and 2021 but was forced onto
    # "Ath Madrid" at 33 in 2020, which alone made it look ambiguous.)
    forced = per_season["score"] < REVIEW_THRESHOLD
    if forced.any():
        log.info(
            "%s: discarding %d forced pairing(s) scoring below %d",
            source_label,
            int(forced.sum()),
            REVIEW_THRESHOLD,
        )
        per_season = per_season[~forced]

    if per_season.empty:
        return pd.DataFrame(
            columns=["source", "league", "raw_name", "canonical_name", "score", "seasons", "status"]
        )

    # Consolidate: one row per (source, league, raw_name), keeping the mapping
    # that scored best and recording how consistent it was across seasons.
    consolidated = (
        per_season.sort_values("score", ascending=False)
        .groupby(["source", "league", "raw_name"], as_index=False)
        .first()
        .drop(columns=["season"])
    )

    seasons_seen = (
        per_season.groupby(["source", "league", "raw_name"])["season"]
        .agg(lambda s: ",".join(str(x) for x in sorted(s)))
        .rename("seasons")
    )
    consolidated = consolidated.merge(seasons_seen, on=["source", "league", "raw_name"])

    # Flag disagreement: the same raw name resolving to different canonical
    # names in different seasons is a red flag even if each score was high.
    distinct_targets = (
        per_season.groupby(["source", "league", "raw_name"])["canonical_name"]
        .nunique()
        .rename("n_distinct_targets")
    )
    consolidated = consolidated.merge(distinct_targets, on=["source", "league", "raw_name"])

    consolidated["status"] = np.where(
        (consolidated["score"] >= AUTO_ACCEPT_SCORE) & (consolidated["n_distinct_targets"] == 1),
        "auto",
        np.where(consolidated["score"] >= REVIEW_THRESHOLD, "review", "unresolved"),
    )
    return consolidated


# ---------------------------------------------------------------------------
# Group helpers
# ---------------------------------------------------------------------------
def _teams_by_group(df: pd.DataFrame, home_col: str, away_col: str) -> dict[tuple[str, int], set[str]]:
    groups: dict[tuple[str, int], set[str]] = {}
    for (league, season), chunk in df.groupby(["league", "season"]):
        groups[(league, int(season))] = set(chunk[home_col]) | set(chunk[away_col])
    return groups


def _clubelo_teams_by_group(elo: pd.DataFrame, seasons: list[int]) -> dict[tuple[str, int], set[str]]:
    """Which ClubElo clubs were top-division in each league-season.

    A season runs roughly August to May, so we ask which clubs held level 1 with
    a rating valid inside that window.
    """
    groups: dict[tuple[str, int], set[str]] = {}
    for season in seasons:
        lo = pd.Timestamp(f"{season}-08-01")
        hi = pd.Timestamp(f"{season + 1}-05-31")
        in_window = elo[(elo["to_date"] >= lo) & (elo["from_date"] <= hi) & (elo["level"] == 1)]
        for league, chunk in in_window.groupby("league"):
            groups[(league, season)] = set(chunk["club_raw"])
    return groups


def _apply_aliases(
    df: pd.DataFrame, aliases: pd.DataFrame, source_label: str, cols: list[str]
) -> pd.DataFrame:
    """Map raw name columns to canonical names using the resolved alias table."""
    usable = aliases[(aliases["source"] == source_label) & (aliases["status"] != "unresolved")]
    lookup = {
        (row.league, row.raw_name): row.canonical_name for row in usable.itertuples(index=False)
    }
    out = df.copy()
    for col in cols:
        canonical_col = col.replace("_raw", "")
        out[canonical_col] = [
            lookup.get((league, raw)) for league, raw in zip(out["league"], out[col], strict=True)
        ]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    ensure_dirs()

    fd = pd.read_parquet(INTERIM_DIR / "footballdata.parquet")
    us = pd.read_parquet(INTERIM_DIR / "understat.parquet")
    elo = pd.read_parquet(INTERIM_DIR / "clubelo.parquet")
    log.info("loaded footballdata=%d understat=%d clubelo=%d rows", len(fd), len(us), len(elo))

    seasons = sorted(fd["season"].unique().tolist())

    # football-data supplies the canonical spelling: it carries the label and the
    # odds, so every other source is mapped onto its names.
    canon_groups = _teams_by_group(fd, "home_team_raw", "away_team_raw")
    us_groups = _teams_by_group(us, "home_team_raw", "away_team_raw")
    elo_groups = _clubelo_teams_by_group(elo, seasons)

    manual = load_manual_aliases()

    aliases = pd.concat(
        [
            manual,
            resolve_source_aliases(canon_groups, us_groups, "understat", manual),
            resolve_source_aliases(canon_groups, elo_groups, "clubelo", manual),
        ],
        ignore_index=True,
    )

    status_counts = aliases["status"].value_counts().to_dict()
    log.info("alias resolution: %s", status_counts)

    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    aliases.sort_values(["source", "league", "raw_name"]).to_csv(RESOLVED_ALIASES_PATH, index=False)
    log.info("wrote %s (%d aliases)", RESOLVED_ALIASES_PATH, len(aliases))

    needs_review = aliases[~aliases["status"].isin(["auto", "manual"])].sort_values("score")
    review_path = REPORTS_DIR / "unmatched_teams.csv"
    needs_review.to_csv(review_path, index=False)
    if not needs_review.empty:
        log.warning(
            "%d alias(es) below auto-accept -- REVIEW %s before trusting results",
            len(needs_review),
            review_path,
        )
        log.warning("\n%s", needs_review.head(25).to_string(index=False))

    # --- Join Understat xG onto the football-data spine ---------------------
    us_canon = _apply_aliases(us, aliases, "understat", ["home_team_raw", "away_team_raw"])
    us_canon = us_canon.dropna(subset=["home_team", "away_team"])

    key = ["league", "season", "home_team", "away_team"]

    fd = fd.rename(columns={"home_team_raw": "home_team", "away_team_raw": "away_team"})

    # The ordered pair must be unique per league-season for the join to be safe.
    for name, frame in (("football-data", fd), ("understat", us_canon)):
        dupes = frame.duplicated(subset=key).sum()
        if dupes:
            log.warning("%s: %d duplicate (league, season, home, away) rows", name, dupes)

    merged = fd.merge(
        us_canon[key + ["home_xg", "away_xg", "date"]].rename(columns={"date": "date_understat"}),
        on=key,
        how="left",
        validate="one_to_one",
    )

    # Date is an assertion, not a join key. Sources may differ by a day.
    date_gap = (merged["date"] - merged["date_understat"]).abs()
    disagreements = merged[date_gap > pd.Timedelta(days=DATE_TOLERANCE_DAYS)]
    if not disagreements.empty:
        gap_path = REPORTS_DIR / "date_disagreements.csv"
        disagreements[key + ["date", "date_understat"]].to_csv(gap_path, index=False)
        log.warning(
            "%d matches joined but dated >%dd apart -- see %s",
            len(disagreements),
            DATE_TOLERANCE_DAYS,
            gap_path,
        )

    merged["match_id"] = (
        merged["league"]
        + "_"
        + merged["season"].astype(str)
        + "_"
        + merged["home_team"].str.replace(" ", "", regex=False)
        + "_"
        + merged["away_team"].str.replace(" ", "", regex=False)
    )
    if merged["match_id"].duplicated().any():
        raise RuntimeError("match_id is not unique -- the ordered-pair assumption broke")

    # --- Join-rate gate ----------------------------------------------------
    merged["has_xg"] = merged["home_xg"].notna()
    join_rate = merged.groupby(["season", "league"])["has_xg"].mean().unstack()
    log.info("xG join rate by season/league:\n%s", (100 * join_rate).round(1).to_string())

    overall = merged["has_xg"].mean()
    log.info("overall xG join rate: %.2f%% (%d/%d)", 100 * overall, merged["has_xg"].sum(), len(merged))

    worst = join_rate.min().min()
    if worst < MIN_JOIN_RATE:
        offenders = join_rate.stack()
        offenders = offenders[offenders < MIN_JOIN_RATE]
        raise RuntimeError(
            f"xG join rate below {MIN_JOIN_RATE:.0%} for:\n{(100 * offenders).round(1).to_string()}\n"
            f"Review {review_path} and re-run."
        )

    merged = merged.drop(columns=["date_understat"]).sort_values(["date", "league"], kind="stable")
    merged = merged.reset_index(drop=True)
    merged.to_parquet(MATCHES_PATH, index=False)
    log.info("wrote %s (%d matches, %d columns)", MATCHES_PATH, len(merged), merged.shape[1])

    # --- ClubElo under canonical names -------------------------------------
    elo_canon = _apply_aliases(elo, aliases, "clubelo", ["club_raw"])
    unmapped = elo_canon["club"].isna().sum()
    if unmapped:
        log.warning("%d ClubElo rating rows have no canonical team; dropping", unmapped)
    elo_canon = elo_canon.dropna(subset=["club"]).rename(columns={"club": "team"})
    elo_canon = elo_canon[["team", "league", "level", "elo", "from_date", "to_date"]]
    elo_canon = elo_canon.sort_values(["team", "from_date"], kind="stable").reset_index(drop=True)
    elo_canon.to_parquet(CLUBELO_CANONICAL_PATH, index=False)
    log.info(
        "wrote %s (%d rows, %d canonical clubs)",
        CLUBELO_CANONICAL_PATH,
        len(elo_canon),
        elo_canon["team"].nunique(),
    )

    return merged


if __name__ == "__main__":
    main()
