"""Tests for the newly added train.py logic: hyperparameter search, stacking,
and champion selection over the enlarged candidate set.

The existing (track, model) fold logic predates this file and is exercised
indirectly through these tests; the focus here is what's new, using small
synthetic data so LightGBM/logistic fit in milliseconds and the holdout-safety
properties are checkable directly rather than by eyeballing a real run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import build_feature_frame
from src.splits import Split
from src.train import (
    StackingEnsemble,
    run_fold_stacking,
    tune_lightgbm,
)

SEARCH_SPACE = {
    "num_leaves": (7, 15),
    "learning_rate": (0.05, 0.15),
    "n_estimators": (20, 40),
    "min_child_samples": (5, 10),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
    "reg_lambda": (0.0, 1.0),
}


@pytest.fixture
def features_frame(matches_four_seasons, elo) -> pd.DataFrame:
    """A real, leakage-safe features table (via build_feature_frame) over four
    synthetic seasons, with a synthetic vig-free market probability merged in
    -- everything train.py's fold logic needs, built the same way features.py
    itself builds it rather than hand-rolled."""
    wide = build_feature_frame(matches_four_seasons, elo, run_leakage_check=False)
    enough_history = (wide["home_n_prior_matches"] >= 5) & (wide["away_n_prior_matches"] >= 5)
    wide = wide[enough_history].reset_index(drop=True)

    rng = np.random.default_rng(7)
    probs = rng.dirichlet([3, 2, 3], size=len(wide))
    wide["p_market_H"] = probs[:, 0]
    wide["p_market_D"] = probs[:, 1]
    wide["p_market_A"] = probs[:, 2]
    wide["odds_H"] = 1.0 / wide["p_market_H"]
    wide["odds_D"] = 1.0 / wide["p_market_D"]
    wide["odds_A"] = 1.0 / wide["p_market_A"]
    return wide


def test_tune_lightgbm_only_uses_walk_forward_folds(features_frame):
    wf_fold = Split(name="wf_2020", train_seasons=(2018, 2019), eval_seasons=(2020,))

    best_params, trials = tune_lightgbm(
        features_frame, "market_blind", [wf_fold], n_trials=2, search_space=SEARCH_SPACE
    )

    assert set(best_params) == set(SEARCH_SPACE)
    assert len(trials) == 2
    assert trials["mean_wf_log_loss"].notna().all()
    # The best row really is the minimum -- tune_lightgbm must not silently
    # return an arbitrary trial.
    assert best_params["num_leaves"] == int(trials.iloc[0]["num_leaves"])


def test_tune_lightgbm_refuses_a_holdout_fold(features_frame):
    """Scoring a trial against the holdout would turn the untouched test season
    into a tuning set -- the same contamination register_champion's
    walk-forward-only selection exists to prevent one level up."""
    holdout_fold = Split(name="holdout_2020", train_seasons=(2018, 2019), eval_seasons=(2020,))
    with pytest.raises(ValueError, match="holdout"):
        tune_lightgbm(features_frame, "market_blind", [holdout_fold], n_trials=1, search_space=SEARCH_SPACE)


def test_stacking_ensemble_predict_proba_shape_and_sums_to_one():
    class DummyModel:
        def __init__(self, probs):
            self._probs = probs

        def predict_proba(self, X):
            return np.tile(self._probs, (len(X), 1))

    base_models = {
        "logistic": DummyModel([0.5, 0.2, 0.3]),
        "lightgbm": DummyModel([0.4, 0.3, 0.3]),
    }
    meta_X = np.array(
        [
            [0.5, 0.2, 0.3, 0.4, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.3, 0.3, 0.4],
            [0.2, 0.6, 0.2, 0.3, 0.5, 0.2],
        ]
    )
    meta_y = np.array([0, 2, 1])  # all three classes, or predict_proba returns < 3 columns
    meta = LogisticRegression(max_iter=200).fit(meta_X, meta_y)

    ensemble = StackingEnsemble(base_models, meta, cols=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    proba = ensemble.predict_proba(X)

    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(ensemble.predict(X), np.argmax(proba, axis=1))


def test_run_fold_stacking_scores_a_fold_with_enough_seasons(features_frame):
    long_fold = Split(name="wf_2021", train_seasons=(2018, 2019, 2020), eval_seasons=(2021,))
    predictions, model_metrics, market_metrics, fitted = run_fold_stacking(
        features_frame, long_fold, "market_blind"
    )

    assert not predictions.empty
    assert isinstance(fitted, StackingEnsemble)
    proba_sum = predictions[["p_H", "p_D", "p_A"]].sum(axis=1)
    assert np.allclose(proba_sum, 1.0, atol=1e-6)
    assert "log_loss" in model_metrics and "log_loss" in market_metrics


def test_run_fold_stacking_skips_a_fold_too_short_for_three_slices(features_frame):
    """Only 2 training seasons: too short for fit + calibration + meta (needs
    calibration_seasons + meta_seasons + 1 = 3 by default). Other model_names
    run this same fold fine -- only stacking needs the extra season."""
    short_fold = Split(name="wf_2020", train_seasons=(2018, 2019), eval_seasons=(2020,))
    with pytest.raises(ValueError, match="too short"):
        run_fold_stacking(features_frame, short_fold, "market_blind")


def test_register_champion_selects_best_of_enlarged_candidate_set(tmp_path, monkeypatch):
    """The larger candidate pool (lightgbm_tuned, stacking added on top of
    logistic/lightgbm) must fall out of the SAME unmodified selection rule:
    walk-forward mean log loss only, never holdout."""
    import src.train as train_module

    class FakeVersion:
        version = "7"

    class FakeClient:
        def set_registered_model_alias(self, *args, **kwargs):
            pass

        def set_model_version_tag(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train_module.mlflow, "register_model", lambda *a, **k: FakeVersion())
    monkeypatch.setattr(train_module, "MlflowClient", FakeClient)
    monkeypatch.setattr(train_module, "CHAMPION_PATH", tmp_path / "champion.json")

    wf_summary = {
        ("market_blind", "logistic"): {"wf_log_loss": 1.02, "wf_market_log_loss": 0.97, "wf_gap": 0.05},
        ("market_blind", "lightgbm"): {"wf_log_loss": 1.05, "wf_market_log_loss": 0.97, "wf_gap": 0.08},
        ("market_aware", "lightgbm_tuned"): {
            "wf_log_loss": 0.999,
            "wf_market_log_loss": 0.97,
            "wf_gap": 0.029,
        },
        ("market_aware", "stacking"): {"wf_log_loss": 1.5, "wf_market_log_loss": 0.97, "wf_gap": 0.53},
    }
    holdout_runs = {
        key: {
            "run_id": "fake-run-id",
            "model_metrics": {"log_loss": v["wf_log_loss"]},
            "market_metrics": {"log_loss": v["wf_market_log_loss"]},
            "feature_columns": ["f1", "f2"],
            "train_seasons": [2018, 2019, 2020],
        }
        for key, v in wf_summary.items()
    }

    train_module.register_champion(wf_summary, holdout_runs)

    written = json.loads(train_module.CHAMPION_PATH.read_text())
    assert written["track"] == "market_aware"
    assert written["algorithm"] == "lightgbm_tuned"
    assert set(written["all_configurations"]) == {
        "market_blind/logistic",
        "market_blind/lightgbm",
        "market_aware/lightgbm_tuned",
        "market_aware/stacking",
    }
