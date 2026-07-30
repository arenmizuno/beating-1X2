"""Shared configuration: paths, constants, and the params.yaml loader.

This is the only module that reads params.yaml. Everything else imports from
here, so there is exactly one place where a path or a tunable is defined.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Per-source raw landing zones.
RAW_FOOTBALLDATA_DIR = RAW_DIR / "footballdata"
RAW_CLUBELO_DIR = RAW_DIR / "clubelo"
RAW_UNDERSTAT_DIR = RAW_DIR / "understat"

REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Serving resolves models:/<REGISTERED_MODEL_NAME>@<CHAMPION_ALIAS>, never a
# concrete version or algorithm. Swapping the model later is a re-registration,
# not a code change in the API, dashboard, or drift stage.
REGISTERED_MODEL_NAME = "beating-1x2"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "model"
CHAMPION_PATH = REPORTS_DIR / "champion.json"

# Hand-verified team-name overrides. This is a committed SOURCE INPUT, not a
# derived artifact: the fuzzy matcher consults it first and never overwrites it.
# Anything listed here is treated as ground truth.
MANUAL_ALIASES_PATH = MAPPINGS_DIR / "team_aliases_manual.csv"

# The full resolved alias table (manual + auto-matched). Derived, so it lives in
# interim/ and is regenerated on every harmonize run.
RESOLVED_ALIASES_PATH = INTERIM_DIR / "team_aliases_resolved.csv"

# Stage outputs.
MATCHES_PATH = INTERIM_DIR / "matches.parquet"
MARKET_PATH = INTERIM_DIR / "market.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

ALL_DIRS = [
    RAW_FOOTBALLDATA_DIR,
    RAW_CLUBELO_DIR,
    RAW_UNDERSTAT_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MAPPINGS_DIR,
    REPORTS_DIR,
]


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# params.yaml
# ---------------------------------------------------------------------------
def load_params() -> dict[str, Any]:
    with PARAMS_PATH.open() as fh:
        return yaml.safe_load(fh)


PARAMS: dict[str, Any] = load_params()

SEED: int = PARAMS["seed"]
LEAGUES: dict[str, dict[str, str]] = PARAMS["leagues"]
LEAGUE_CODES: list[str] = list(LEAGUES.keys())

SEASON_START: int = PARAMS["seasons"]["start"]
SEASON_END: int = PARAMS["seasons"]["end"]
SEASONS: list[int] = list(range(SEASON_START, SEASON_END + 1))
HOLDOUT_SEASON: int = PARAMS["holdout_season"]

# Seasons available for training and model selection. The holdout is excluded
# here rather than at the call site so it cannot be used by accident.
DEV_SEASONS: list[int] = [s for s in SEASONS if s < HOLDOUT_SEASON]

# ---------------------------------------------------------------------------
# Outcome encoding
# ---------------------------------------------------------------------------
# football-data.co.uk encodes full-time result in the FTR column as H/D/A.
# We keep that ordering everywhere: index 0 = home win, 1 = draw, 2 = away win.
OUTCOMES = ["H", "D", "A"]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}
OUTCOME_NAMES = {"H": "home win", "D": "draw", "A": "away win"}


# ---------------------------------------------------------------------------
# Season identifier conversions
# ---------------------------------------------------------------------------
# A season is identified internally by its STARTING year (2018 = 2018-19).
# Each source spells it differently.
def season_to_footballdata(season: int) -> str:
    """2018 -> '1819', matching football-data.co.uk's mmz4281/<season>/ paths."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_to_understat(season: int) -> str:
    """2018 -> '2018'. Understat keys a season by its starting year."""
    return str(season)


def season_label(season: int) -> str:
    """2018 -> '2018-19', for reports and plot titles."""
    return f"{season}-{(season + 1) % 100:02d}"


# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
FOOTBALLDATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Understat used to inline its fixture list in the league page as a hex-escaped
# `var datesData = JSON.parse(...)` blob. It now loads that data over AJAX from
# this endpoint instead, which returns {teams, players, dates} as JSON.
UNDERSTAT_URL = "https://understat.com/getLeagueData/{league}/{season}"

CLUBELO_TEAM_URL = "http://api.clubelo.com/{team}"
CLUBELO_DATE_URL = "http://api.clubelo.com/{date}"

# Understat is scraped, so identify ourselves rather than masquerading.
USER_AGENT = (
    "ADSP32021-final-project/1.0 (UChicago MS Applied Data Science coursework; "
    "academic use)"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Consistent stage logging. Each src/ module calls this at import."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
