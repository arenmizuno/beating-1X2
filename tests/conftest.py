"""Shared synthetic fixtures.

Every test runs against generated data, never against data/. That keeps the
suite fast, hermetic, and runnable in CI with no network -- which matters
because the real pipeline depends on three external sources that would make CI
both slow and flaky.

The synthetic league is a double round-robin over 8 teams across 2 seasons, so
each team plays 14 matches per season. That is deliberately just enough history
for the 10-match rolling windows to populate in the second season while staying
small enough to reason about by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = [f"Team{c}" for c in "ABCDEFGH"]
SEASONS = [2018, 2019]
LEAGUE = "E0"


def _round_robin(teams: list[str]) -> list[tuple[str, str]]:
    """Every ordered pair exactly once -- home and away legs, as in a real
    European league season. This is what makes (season, home, away) a unique
    key, which src/harmonize.py relies on for its join."""
    return [(h, a) for h in teams for a in teams if h != a]


def _build_matches(seasons: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        fixtures = _round_robin(TEAMS)
        start = pd.Timestamp(f"{season}-08-10")
        for i, (home, away) in enumerate(fixtures):
            fthg = int(rng.integers(0, 4))
            ftag = int(rng.integers(0, 4))
            rows.append(
                {
                    "match_id": f"{LEAGUE}_{season}_{home}_{away}",
                    "league": LEAGUE,
                    "season": season,
                    # Spread fixtures a few days apart so rest-day gaps are real.
                    "date": start + pd.Timedelta(days=3 * i),
                    "home_team": home,
                    "away_team": away,
                    "fthg": fthg,
                    "ftag": ftag,
                    "home_xg": float(fthg) + rng.normal(0, 0.3),
                    "away_xg": float(ftag) + rng.normal(0, 0.3),
                    "ftr": "H" if fthg > ftag else ("D" if fthg == ftag else "A"),
                }
            )
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


@pytest.fixture
def matches() -> pd.DataFrame:
    """A synthetic completed-match table with the columns src/features.py needs."""
    return _build_matches(SEASONS)


@pytest.fixture
def matches_four_seasons() -> pd.DataFrame:
    """Four seasons on top of `matches`'s 2, for train.py tests that need a
    genuine train/calibrate/meta three-way split (stacking needs 3 training
    seasons) PLUS a separate eval season left over (2018-2020 train, 2021
    eval) -- more history than the base 2-season fixture provides on purpose
    (see `matches`'s own use in test_splits.py, which relies on a 2020 season
    NOT existing)."""
    return _build_matches([2018, 2019, 2020, 2021])


@pytest.fixture
def elo() -> pd.DataFrame:
    """Elo history as [from_date, to_date] validity ranges, matching ClubElo."""
    rows = []
    for i, team in enumerate(TEAMS):
        # Two rating epochs per team so the as-of lookup has something to choose
        # between rather than trivially matching a single row.
        #
        # The two epochs are deliberately far apart (~1500 vs ~2000) with only
        # small within-epoch spread, so a test can tell "which epoch was used"
        # from the value alone. Overlapping ranges would make the assertion
        # ambiguous and the test worthless.
        rows.append(
            {
                "team": team,
                "league": LEAGUE,
                "level": 1,
                "elo": 1500.0 + i,
                "from_date": pd.Timestamp("2017-01-01"),
                "to_date": pd.Timestamp("2019-01-01"),
            }
        )
        rows.append(
            {
                "team": team,
                "league": LEAGUE,
                "level": 1,
                "elo": 2000.0 + i,
                "from_date": pd.Timestamp("2019-01-02"),
                "to_date": pd.Timestamp("2030-01-01"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def odds_frame() -> pd.DataFrame:
    """Decimal odds carrying a realistic ~5% overround."""
    true_probs = np.array(
        [[0.45, 0.27, 0.28], [0.60, 0.25, 0.15], [0.20, 0.26, 0.54], [0.34, 0.33, 0.33]]
    )
    overround = 1.05
    odds = 1.0 / (true_probs * overround)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(len(odds))],
            "odds_PSC_H": odds[:, 0],
            "odds_PSC_D": odds[:, 1],
            "odds_PSC_A": odds[:, 2],
        }
    )
