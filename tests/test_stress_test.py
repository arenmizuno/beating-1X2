"""Tests for the drift stress-test primitives.

These exercise the corruption functions and the input-drift detector directly on
small synthetic data, so no model load or MLflow store is needed. The end-to-end
run() (which loads the champion and scores) is covered by running the module
against the pipeline, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.stress_test import (
    corrupt_out_of_bounds,
    corrupt_schema,
    corrupt_swapped,
    input_drift,
    performance,
)

FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_xg_for_r5",
    "away_xg_for_r5",
    "p_market_H",
    "p_market_D",
    "p_market_A",
]


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "home_elo": rng.normal(1700, 80, n),
            "away_elo": rng.normal(1600, 80, n),
            "elo_diff": rng.normal(100, 50, n),
            "home_xg_for_r5": rng.normal(1.4, 0.3, n),
            "away_xg_for_r5": rng.normal(1.2, 0.3, n),
            "p_market_H": rng.uniform(0.3, 0.6, n),
            "p_market_D": rng.uniform(0.2, 0.3, n),
            "p_market_A": rng.uniform(0.2, 0.4, n),
            "league": "E0",
            "ftr": rng.choice(["H", "D", "A"], n),
        }
    )


def test_out_of_bounds_pushes_every_feature_past_its_range():
    clean = _frame()
    corrupted = corrupt_out_of_bounds(clean, FEATURES, np.random.default_rng(1))
    for col in FEATURES:
        assert corrupted[col].min() > clean[col].max()


def test_swapped_exchanges_home_away_and_negates_diffs():
    clean = _frame()
    corrupted = corrupt_swapped(clean, FEATURES)
    # home/away ratings exchanged
    np.testing.assert_array_equal(corrupted["home_elo"], clean["away_elo"])
    np.testing.assert_array_equal(corrupted["away_elo"], clean["home_elo"])
    # market home/away price flipped
    np.testing.assert_array_equal(corrupted["p_market_H"], clean["p_market_A"])
    np.testing.assert_array_equal(corrupted["p_market_A"], clean["p_market_H"])
    # differentials negated
    np.testing.assert_allclose(corrupted["elo_diff"], -clean["elo_diff"])


def test_schema_break_nulls_dropped_columns_only():
    clean = _frame()
    dropped = [c for c in FEATURES if "xg" in c]
    corrupted = corrupt_schema(clean, dropped)
    for col in dropped:
        assert corrupted[col].isna().all()
    for col in set(FEATURES) - set(dropped):
        assert not corrupted[col].isna().any()


def test_input_drift_flags_out_of_bounds_and_ignores_identity():
    clean = _frame()
    corrupted = corrupt_out_of_bounds(clean, FEATURES, np.random.default_rng(2))
    drift = input_drift(clean, corrupted, FEATURES)
    assert bool(drift["drifted"].all())

    same = input_drift(clean, clean, FEATURES)
    assert int(same["drifted"].sum()) == 0
    assert not same["missing"].any()


def test_input_drift_marks_missing_features():
    clean = _frame()
    dropped = [c for c in FEATURES if "xg" in c]
    corrupted = corrupt_schema(clean, dropped)
    drift = input_drift(clean, corrupted, FEATURES).set_index("feature")
    for col in dropped:
        assert bool(drift.loc[col, "missing"])
        assert not bool(drift.loc[col, "drifted"])


def test_performance_reports_gap_to_market():
    clean = _frame()
    y = clean["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    proba = np.full((len(clean), 3), 1 / 3)
    market = clean[["p_market_H", "p_market_D", "p_market_A"]].to_numpy()
    market = market / market.sum(axis=1, keepdims=True)
    perf = performance(proba, y, market)
    assert set(perf) == {"log_loss", "ece", "accuracy", "market_log_loss", "gap"}
    assert perf["gap"] == perf["log_loss"] - perf["market_log_loss"]
