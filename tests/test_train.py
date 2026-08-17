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
    assert_semver_available,
    build_model,
    fit_deployment_model,
    run_fold,
    run_fold_stacking,
    select_champion,
    tune_lightgbm,
    tune_model,
    validate_model_semver,
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

# Small per-algorithm spaces so the multi-algorithm search fits in milliseconds.
XGBOOST_SEARCH = {
    "max_depth": (2, 4),
    "learning_rate": (0.05, 0.15),
    "n_estimators": (20, 40),
    "min_child_weight": (1, 5),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
    "reg_lambda": (0.0, 1.0),
}
CATBOOST_SEARCH = {
    "depth": (2, 4),
    "learning_rate": (0.05, 0.15),
    "iterations": (20, 40),
    "l2_leaf_reg": (1.0, 5.0),
    "subsample": (0.8, 1.0),
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


@pytest.mark.parametrize(
    "algorithm,search_space,int_keys",
    [
        ("xgboost", XGBOOST_SEARCH, ("max_depth", "n_estimators", "min_child_weight")),
        ("catboost", CATBOOST_SEARCH, ("depth", "iterations")),
    ],
)
def test_tune_model_searches_each_algorithm(features_frame, algorithm, search_space, int_keys):
    """The generic search drives xgboost and catboost the same way it drives
    lightgbm: best_params covers exactly the space, and integer-ranged
    hyperparameters come back as ints (not numpy floats that would break the
    estimator constructor)."""
    wf_fold = Split(name="wf_2020", train_seasons=(2018, 2019), eval_seasons=(2020,))
    best_params, trials = tune_model(
        features_frame, "market_blind", [wf_fold], algorithm, n_trials=2, search_space=search_space
    )
    assert set(best_params) == set(search_space)
    assert len(trials) == 2
    assert trials["mean_wf_log_loss"].notna().all()
    for key in int_keys:
        assert isinstance(best_params[key], int)


def test_tune_model_refuses_a_holdout_fold(features_frame):
    holdout_fold = Split(name="holdout_2020", train_seasons=(2018, 2019), eval_seasons=(2020,))
    with pytest.raises(ValueError, match="holdout"):
        tune_model(
            features_frame, "market_blind", [holdout_fold], "xgboost", n_trials=1, search_space=XGBOOST_SEARCH
        )


@pytest.mark.parametrize("name", ["logistic", "lightgbm", "xgboost", "catboost", "mlp"])
def test_build_model_emits_three_class_probabilities(features_frame, name):
    """Every model family build_model knows must fit and return a normalized
    3-way probability -- the one interface the fold logic, calibration, stacking
    and serving all depend on."""
    from src.train import feature_columns, make_design

    cols = feature_columns(features_frame, "market_blind")
    X = make_design(features_frame, cols)
    y = features_frame["target"].to_numpy()

    model = build_model(name)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(features_frame), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


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


def test_candidate_fold_runners_reject_the_holdout(features_frame):
    holdout = Split(name="holdout_2021", train_seasons=(2018, 2019, 2020), eval_seasons=(2021,))
    with pytest.raises(ValueError, match="sealed holdout"):
        run_fold(features_frame, holdout, "market_blind", "logistic")
    with pytest.raises(ValueError, match="sealed holdout"):
        run_fold_stacking(features_frame, holdout, "market_blind")


def test_deployment_fit_never_reads_holdout_rows(features_frame, monkeypatch):
    import src.train as train_module

    split = Split(name="holdout_2021", train_seasons=(2018, 2019, 2020), eval_seasons=(2021,))
    seen = {}

    def fake_fit(model_name, train, cols, received_split, overrides=None):
        seen["seasons"] = set(train["season"])
        seen["split"] = received_split
        return object(), (), ()

    monkeypatch.setattr(train_module, "holdout_split", lambda: split)
    monkeypatch.setattr(train_module, "fit_calibrated", fake_fit)
    _, _, train_seasons = fit_deployment_model(
        features_frame, "market_blind", "logistic", None
    )

    assert seen["seasons"] == {2018, 2019, 2020}
    assert 2021 not in seen["seasons"]
    assert seen["split"] == split
    assert train_seasons == [2018, 2019, 2020]


@pytest.mark.parametrize("value", ["1", "1.0", "01.0.0", "v1.0.0", "1.0.0.0"])
def test_model_semver_must_be_strict(value):
    with pytest.raises(ValueError, match="semantic version"):
        validate_model_semver(value)


def test_duplicate_model_semver_is_rejected():
    class Existing:
        version = "4"
        tags = {"model_semver": "1.0.0"}

    class FakeClient:
        def search_model_versions(self, _filter):
            return [Existing()]

    with pytest.raises(ValueError, match="already assigned"):
        assert_semver_available(FakeClient(), "1.0.0")


def test_register_champion_selects_best_of_enlarged_candidate_set(tmp_path, monkeypatch):
    """The larger candidate pool (lightgbm_tuned, stacking added on top of
    logistic/lightgbm) must fall out of the SAME unmodified selection rule:
    walk-forward mean log loss only, never holdout."""
    import src.train as train_module

    class FakeVersion:
        version = "7"

    class FakeClient:
        tags = {}

        def search_model_versions(self, _filter):
            return []

        def set_registered_model_alias(self, *args, **kwargs):
            pass

        def set_model_version_tag(self, _name, _version, key, value):
            self.tags[key] = value

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
    best_key = select_champion(wf_summary)
    run_info = {
        "run_id": "fake-run-id",
        "feature_columns": ["f1", "f2"],
        "train_seasons": [2018, 2019, 2020],
    }

    train_module.register_champion(wf_summary, best_key, run_info)

    written = json.loads(train_module.CHAMPION_PATH.read_text())
    assert written["track"] == "market_aware"
    assert written["algorithm"] == "lightgbm_tuned"
    # Compared against the configured value, not a literal: the assertion is
    # that register_champion writes and tags whatever semver params.yaml
    # declares, which stays true across intentional version bumps.
    assert written["model_semver"] == train_module.MODEL_SEMVER
    assert "holdout" not in written
    assert FakeClient.tags["model_semver"] == train_module.MODEL_SEMVER
    assert set(written["all_configurations"]) == {
        "market_blind/logistic",
        "market_blind/lightgbm",
        "market_aware/lightgbm_tuned",
        "market_aware/stacking",
    }
