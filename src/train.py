"""Stage 6: train and evaluate models across the walk-forward folds.

Two tracks, and the contrast between them is the actual research question:

  market_blind   Features only. This is the only track that can be honestly
                 compared against the closing line, because it never sees it.

  market_aware   The same features plus the vig-free market probability. If it
                 scores no better than market_blind, our xG and Elo signal adds
                 nothing the market had not already priced -- which is a real
                 finding, and the more likely one. If it beats the market alone,
                 the features carry residual information.

The competing configurations per track live in `train.models` (params.yaml):

  logistic         Multinomial logistic regression. Not a throwaway baseline --
                   on football outcome data a well-specified linear model on
                   Elo/form differentials is genuinely competitive with
                   boosting, and it is naturally well calibrated.

  lightgbm         Gradient boosting, for any non-linearity the linear model
  xgboost          misses. Three independent gradient-boosting implementations,
  catboost         each with fixed hyperparameters from params.yaml, so their
                   inductive biases can be compared head to head.

  *_tuned          The same algorithm with hyperparameters chosen by a random
                   search scored on walk-forward folds ONLY (see `tune_model`)
                   -- never the holdout, which would turn the untouched test
                   season into a tuning set. Each tuned entry is kept SEPARATE
                   from its fixed counterpart so the fixed-vs-tuned comparison
                   stays visible in champion.json.

  mlp              A neural net (sklearn multilayer perceptron), imputed and
                   scaled like logistic since a dense net tolerates neither NaN
                   nor unscaled features.

  stacking         A meta-learner over the calibrated base models named in
                   `train.stacking_base_models` (see `run_fold_stacking`),
                   trained on a THIRD temporal slice of the training window that
                   neither base model nor its calibrator ever sees.

Every configuration is selected purely by `register_champion`'s existing rule
(walk-forward mean log loss, never holdout) -- a larger candidate pool changes
nothing about how the winner is chosen.

All models get identical treatment: fitted on the same seasons, then calibrated
on the same held-out later season. Calibrating an already-calibrated model is
harmless, and equal treatment keeps the comparison clean.

Every fold also scores the market on exactly the same rows, so the benchmark is
never computed over a different subset than the model it is being compared to.

Outputs:
  data/processed/predictions.parquet   per-fold out-of-sample predictions
  mlruns/                              MLflow tracking store
"""

from __future__ import annotations

import json
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import metrics
from src.config import (
    CHAMPION_ALIAS,
    CHAMPION_PATH,
    FEATURES_PATH,
    LEAGUE_CODES,
    MLRUNS_DIR,
    MODEL_ARTIFACT_PATH,
    OUTCOMES,
    PARAMS,
    PROCESSED_DIR,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
)
from src.splits import Split, calibration_split, holdout_split, stacking_split, walk_forward_splits

log = get_logger("train")

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

_T = PARAMS["train"]
TRACKS: list[str] = _T["tracks"]
MODELS: list[str] = _T["models"]
CALIBRATION_METHOD: str = _T["calibration_method"]
PROBABILITY_FLOOR: float = _T["probability_floor"]
EXPERIMENT_NAME: str = _T["mlflow_experiment"]
STACKING_BASE_MODELS: list[str] = _T["stacking_base_models"]
N_BINS: int = PARAMS["evaluate"]["calibration_bins"]

MARKET_FEATURES = [f"p_market_{o}" for o in OUTCOMES]
DC_FEATURES = [f"dc_p_{o}" for o in OUTCOMES]

# Columns that are identifiers, labels, or market data -- never plain features.
NON_FEATURE_COLUMNS = {
    "match_id",
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "ftr",
    "target",
    "odds_source",
    "overround",
    "shin_z",
}


def feature_columns(df: pd.DataFrame, track: str) -> list[str]:
    """Feature list for a track.

    Anything derived from odds is excluded by prefix rather than by name, so a
    new odds column added upstream cannot silently leak into the market_blind
    track. `dc_p_` (the Dixon-Coles model's own standalone 1X2 probability) is
    excluded from BOTH tracks for a different reason: it is reported as a
    benchmark alongside market and model (see evaluate.py), and feeding a model
    its own comparison benchmark as an input would make "model beats
    Dixon-Coles" nearly tautological. The underlying dc_attack/dc_defense/
    dc_lambda_*/dc_goal_diff_expected columns are NOT excluded -- those are the
    actual new signal being tested.
    """
    excluded = set(NON_FEATURE_COLUMNS)
    for col in df.columns:
        if col.startswith(("odds_", "p_mult_", "p_shin_", "p_market_", "dc_p_")):
            excluded.add(col)

    cols = [c for c in df.columns if c not in excluded]

    if track == "market_aware":
        cols += MARKET_FEATURES
    elif track != "market_blind":
        raise ValueError(f"unknown track: {track}")

    return cols


def make_design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric design matrix, with league one-hot encoded.

    The league dummies are built from the fixed league list in config rather
    than from whatever happens to be present in this fold, so every fold and
    both tracks produce identically-shaped, identically-ordered matrices.
    """
    X = df[cols].copy()
    for code in LEAGUE_CODES:
        X[f"league_{code}"] = (df["league"] == code).astype(float)
    return X


def build_model(name: str, overrides: dict | None = None):
    """Construct an unfitted estimator by name.

    `overrides` replaces the fixed params.yaml hyperparameters for that
    algorithm -- used by `tune_model` to score a search trial without mutating
    global config. Every estimator exposes the sklearn predict_proba interface,
    which is all the fold logic, calibration, stacking and serving require.

    Preprocessing differs by family. logistic and the neural net are wrapped in
    an impute+scale pipeline: neither tolerates NaN, and both are sensitive to
    feature scale (Elo ~1500, rest days ~7). The gradient-boosting ensembles
    route NaN down their own branch and are scale-invariant, so they take the
    raw design matrix -- a median fill for "team has no xG history yet" would
    throw away information the tree can use.
    """
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=_T["logistic"]["C"],
                        max_iter=_T["logistic"]["max_iter"],
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if name == "mlp":
        params = dict(_T["mlp"])
        if overrides:
            params.update(overrides)
        hidden = tuple(params.pop("hidden_layer_sizes"))
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=hidden,
                        early_stopping=True,
                        random_state=SEED,
                        **params,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        params = dict(_T["lightgbm"])
        if overrides:
            params.update(overrides)
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=SEED,
            verbose=-1,
            **params,
        )
    if name == "xgboost":
        # num_class is set automatically by the sklearn wrapper from the label
        # count; passing it explicitly raises, so only the objective is named.
        params = dict(_T["xgboost"])
        if overrides:
            params.update(overrides)
        return XGBClassifier(
            objective="multi:softprob",
            random_state=SEED,
            verbosity=0,
            **params,
        )
    if name == "catboost":
        # bootstrap_type is pinned to Bernoulli (not exposed to the search)
        # because CatBoost's default Bayesian bootstrap rejects the `subsample`
        # parameter both the fixed config and the search set. allow_writing_files
        # is off so fitting does not litter a catboost_info/ directory.
        params = dict(_T["catboost"])
        if overrides:
            params.update(overrides)
        return CatBoostClassifier(
            loss_function="MultiClass",
            bootstrap_type="Bernoulli",
            random_seed=SEED,
            allow_writing_files=False,
            verbose=False,
            **params,
        )
    raise ValueError(f"unknown model: {name}")


def fit_calibrated(
    model_name: str, train: pd.DataFrame, cols: list[str], split: Split, overrides: dict | None = None
):
    """Fit on the earlier seasons, calibrate on the most recent one.

    The calibration split is temporal, not random: calibrating on rows drawn
    from the seasons the model trained on would fit the calibrator to in-sample
    confidence and hide exactly the overconfidence we want to correct.
    """
    fit_seasons, calib_seasons = calibration_split(split.train_seasons)

    fit_rows = train[train["season"].isin(fit_seasons)]
    calib_rows = train[train["season"].isin(calib_seasons)]

    base = build_model(model_name, overrides)
    base.fit(make_design(fit_rows, cols), fit_rows["target"].to_numpy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv="prefit")
        calibrated.fit(make_design(calib_rows, cols), calib_rows["target"].to_numpy())

    return calibrated, fit_seasons, calib_seasons


def market_probabilities(df: pd.DataFrame) -> np.ndarray:
    return df[MARKET_FEATURES].to_numpy(dtype=float)


def apply_probability_floor(proba: np.ndarray) -> tuple[np.ndarray, int]:
    """Clamp probabilities away from zero, then renormalize.

    This is an output contract, not a cosmetic fix. A predicted probability of
    exactly zero asserts that an outcome is impossible; when it then happens,
    log loss is undefined and any Kelly stake sized against it is unbounded. No
    honest 3-way football model should ever emit one -- even the longest
    longshots price around 1-2%.

    Returns the floored probabilities and the number of entries that hit the
    floor, which is logged per fold so a regression to degenerate calibration
    output is visible rather than silently absorbed.
    """
    n_floored = int((proba < PROBABILITY_FLOOR).sum())
    floored = np.maximum(proba, PROBABILITY_FLOOR)
    return floored / floored.sum(axis=1, keepdims=True), n_floored


def _build_predictions_frame(
    scored: pd.DataFrame, split: Split, track: str, model_name: str, y: np.ndarray, proba: np.ndarray
) -> pd.DataFrame:
    """Assemble the per-row prediction record shared by every model path.

    Carries market AND Dixon-Coles probabilities through alongside the model's
    own, so evaluate.py can score model vs. market vs. Dixon-Coles on
    identical rows without re-deriving anything.
    """
    predictions = pd.DataFrame(
        {
            "match_id": scored["match_id"].values,
            "league": scored["league"].values,
            "season": scored["season"].values,
            "date": scored["date"].values,
            "fold": split.name,
            "track": track,
            "model": model_name,
            "y_true": y,
        }
    )
    for i, outcome in enumerate(OUTCOMES):
        predictions[f"p_{outcome}"] = proba[:, i]
        predictions[f"p_market_{outcome}"] = scored[f"p_market_{outcome}"].values
        predictions[f"dc_p_{outcome}"] = scored[f"dc_p_{outcome}"].values
        predictions[f"odds_{outcome}"] = scored[f"odds_{outcome}"].values
        # Carried for closing-line value; may be NaN where no opening price.
        if f"odds_open_{outcome}" in scored.columns:
            predictions[f"odds_open_{outcome}"] = scored[f"odds_open_{outcome}"].values
    return predictions


def run_fold(
    features: pd.DataFrame, split: Split, track: str, model_name: str, overrides: dict | None = None
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], object]:
    """Train one (track, model, fold) and score it against the market.

    `model_name` selects the algorithm ("logistic", "lightgbm", "xgboost",
    "catboost", "mlp"); the actual label written to `predictions` is whatever
    the caller passes as `model_name` unmodified, so a "<algo>_tuned" config
    (same algorithm, tuned hyperparameters via `overrides`) is tracked as its
    own configuration.

    Returns the fitted (calibrated) estimator alongside the metrics so the
    holdout fold's model -- the one trained on the most data, and therefore the
    one worth deploying -- can be registered without refitting.
    """
    cols = feature_columns(features, track)
    train, evaluation = split.apply(features)
    algorithm = model_name[: -len("_tuned")] if model_name.endswith("_tuned") else model_name

    # market_aware needs a price to train on, so drop unpriced rows from its
    # training set. market_blind keeps everything it can use.
    if track == "market_aware":
        train = train.dropna(subset=MARKET_FEATURES)

    model, fit_seasons, calib_seasons = fit_calibrated(algorithm, train, cols, split, overrides)

    # Score model and market on IDENTICAL rows -- only those with a closing
    # price -- so the benchmark is never advantaged by a different subset.
    scored = evaluation.dropna(subset=MARKET_FEATURES).copy()
    proba, n_floored = apply_probability_floor(
        model.predict_proba(make_design(scored, cols))
    )
    y = scored["target"].to_numpy()

    model_metrics = metrics.summary(y, proba, N_BINS)
    model_metrics["floored_fraction"] = n_floored / proba.size
    market_metrics = metrics.summary(y, market_probabilities(scored), N_BINS)

    predictions = _build_predictions_frame(scored, split, track, model_name, y, proba)

    log.info(
        "  %-14s %-12s %-14s n=%5d | logloss %.4f (market %.4f, %+.4f) | ece %.4f | floored %.3f%%",
        model_name,
        track,
        split.name,
        len(scored),
        model_metrics["log_loss"],
        market_metrics["log_loss"],
        model_metrics["log_loss"] - market_metrics["log_loss"],
        model_metrics["ece"],
        100 * model_metrics["floored_fraction"],
    )
    return predictions, model_metrics, market_metrics, model


# ---------------------------------------------------------------------------
# Hyperparameter search (walk-forward folds only)
# ---------------------------------------------------------------------------
# Hyperparameters whose search range is a range of integers. Everything else in
# a search space is sampled as a uniform float. Kept as one set across all
# algorithms so `_sample_params` needs no per-algorithm special-casing.
_INTEGER_HYPERPARAMS = {
    "num_leaves",
    "n_estimators",
    "min_child_samples",
    "max_depth",
    "min_child_weight",
    "depth",
    "iterations",
}


def _sample_params(rng: np.random.Generator, search_space: dict) -> dict:
    """Draw one hyperparameter set from a `{name: [lo, hi]}` search space.

    Names in `_INTEGER_HYPERPARAMS` are sampled as inclusive integers, the rest
    as uniform floats. A `n_trials` key (the search budget, not a
    hyperparameter) is ignored if present.
    """
    out = {}
    for key, bounds in search_space.items():
        if key == "n_trials":
            continue
        lo, hi = bounds
        if key in _INTEGER_HYPERPARAMS:
            out[key] = int(rng.integers(lo, hi + 1))
        else:
            out[key] = float(rng.uniform(lo, hi))
    return out


def tune_model(
    features: pd.DataFrame,
    track: str,
    folds: list[Split],
    algorithm: str,
    n_trials: int,
    search_space: dict,
) -> tuple[dict, pd.DataFrame]:
    """Random search over one algorithm's hyperparameters, walk-forward folds ONLY.

    Generalizes across every tunable algorithm (lightgbm, xgboost, catboost):
    the only algorithm-specific input is `search_space`, and the estimator being
    scored comes straight from `build_model(algorithm, params)`.

    `folds` must never include the holdout split -- scoring a trial against it
    would turn the untouched test season into a tuning set, exactly the
    contamination `register_champion`'s walk-forward-only selection rule already
    exists to prevent one level up. No new dependency: a plain
    `np.random.default_rng(SEED)` random search, not Optuna -- these spaces are
    small enough that a sample-efficient sampler buys little.

    Returns (best_params, trials) so the full search is auditable, the same way
    `register_champion` makes its own decision auditable.
    """
    if any(split.name.startswith("holdout") for split in folds):
        raise ValueError("tune_model must never be scored against the holdout split")

    cols = feature_columns(features, track)
    rng = np.random.default_rng(SEED)

    trials = []
    for trial in range(n_trials):
        params = _sample_params(rng, search_space)
        fold_logloss = []
        for split in folds:
            train, evaluation = split.apply(features)
            if track == "market_aware":
                train = train.dropna(subset=MARKET_FEATURES)
            model, _, _ = fit_calibrated(algorithm, train, cols, split, params)
            scored = evaluation.dropna(subset=MARKET_FEATURES).copy()
            proba, _ = apply_probability_floor(model.predict_proba(make_design(scored, cols)))
            fold_logloss.append(metrics.log_loss(scored["target"].to_numpy(), proba))
        mean_log_loss = float(np.mean(fold_logloss))
        trials.append({"trial": trial, "mean_wf_log_loss": mean_log_loss, **params})
        log.info(
            "  %s_tuned/%s trial %2d/%d: wf logloss %.4f",
            algorithm,
            track,
            trial + 1,
            n_trials,
            mean_log_loss,
        )

    trials_df = pd.DataFrame(trials).sort_values("mean_wf_log_loss").reset_index(drop=True)
    best = trials_df.iloc[0]
    best_params = {
        k: (int(v) if k in _INTEGER_HYPERPARAMS else float(v))
        for k, v in best.items()
        if k not in ("trial", "mean_wf_log_loss")
    }
    return best_params, trials_df


def tune_lightgbm(
    features: pd.DataFrame,
    track: str,
    folds: list[Split],
    n_trials: int,
    search_space: dict,
) -> tuple[dict, pd.DataFrame]:
    """LightGBM-specific entry point to `tune_model`, retained for callers and
    tests that predate the multi-algorithm search."""
    return tune_model(features, track, folds, "lightgbm", n_trials, search_space)


# ---------------------------------------------------------------------------
# Stacking ensemble (walk-forward folds only, meta-learner on its own slice)
# ---------------------------------------------------------------------------
class StackingEnsemble:
    """A fitted meta-learner over several already-calibrated base models.

    The base learners are configurable (`train.stacking_base_models` in
    params.yaml); by default calibrated logistic + lightgbm + xgboost +
    catboost. Exposes only `.predict_proba(X)` / `.predict(X)`, which is all
    `apply_probability_floor`, `mlflow.sklearn.log_model`'s signature inference,
    and the serving layer (`api.py`/`predict.py`, which only ever call
    `champion.model.predict_proba(design)` generically) require -- so nothing
    downstream needs to change for stacking to become the registered champion.

    `base_names` fixes the order in which the base models' probability vectors
    are concatenated; it MUST match the order the meta-learner was trained on.
    It defaults to the base_models dict's insertion order.
    """

    def __init__(
        self,
        base_models: dict[str, object],
        meta_model: LogisticRegression,
        cols: list[str],
        base_names: list[str] | None = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cols = cols
        self.base_names = list(base_names) if base_names is not None else list(base_models)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        stacked = np.hstack(
            [self.base_models[name].predict_proba(X) for name in self.base_names]
        )
        return self.meta_model.predict_proba(stacked)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def run_fold_stacking(
    features: pd.DataFrame, split: Split, track: str
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], object]:
    """Stack the calibrated base models in STACKING_BASE_MODELS via a meta-learner.

    Three temporally disjoint slices of the training window (`stacking_split`):
    fit_seasons trains the base models, calib_seasons calibrates them
    (identical to `fit_calibrated`), and meta_seasons -- seen by neither the
    base models nor their calibrators -- trains the meta-learner. Training the
    meta-learner on calib_seasons instead would double-dip the exact
    in-sample-confidence problem calibration is meant to avoid one level up.

    Raises ValueError (propagated from `stacking_split`) on folds whose
    training window is too short for a three-way split; callers should skip
    and log those folds rather than treating the error as fatal.
    """
    cols = feature_columns(features, track)
    train, evaluation = split.apply(features)
    if track == "market_aware":
        train = train.dropna(subset=MARKET_FEATURES)

    fit_seasons, calib_seasons, meta_seasons = stacking_split(split.train_seasons)
    fit_rows = train[train["season"].isin(fit_seasons)]
    calib_rows = train[train["season"].isin(calib_seasons)]
    meta_rows = train[train["season"].isin(meta_seasons)]

    base_models = {}
    for name in STACKING_BASE_MODELS:
        base = build_model(name)
        base.fit(make_design(fit_rows, cols), fit_rows["target"].to_numpy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv="prefit")
            calibrated.fit(make_design(calib_rows, cols), calib_rows["target"].to_numpy())
        base_models[name] = calibrated

    meta_design = np.hstack(
        [base_models[name].predict_proba(make_design(meta_rows, cols)) for name in STACKING_BASE_MODELS]
    )
    meta_model = LogisticRegression(max_iter=2000, random_state=SEED)
    meta_model.fit(meta_design, meta_rows["target"].to_numpy())

    ensemble = StackingEnsemble(base_models, meta_model, cols, STACKING_BASE_MODELS)

    scored = evaluation.dropna(subset=MARKET_FEATURES).copy()
    proba, n_floored = apply_probability_floor(ensemble.predict_proba(make_design(scored, cols)))
    y = scored["target"].to_numpy()

    model_metrics = metrics.summary(y, proba, N_BINS)
    model_metrics["floored_fraction"] = n_floored / proba.size
    market_metrics = metrics.summary(y, market_probabilities(scored), N_BINS)

    predictions = _build_predictions_frame(scored, split, track, "stacking", y, proba)

    log.info(
        "  %-14s %-12s %-14s n=%5d | logloss %.4f (market %.4f, %+.4f) | ece %.4f | floored %.3f%%",
        "stacking",
        track,
        split.name,
        len(scored),
        model_metrics["log_loss"],
        market_metrics["log_loss"],
        model_metrics["log_loss"] - market_metrics["log_loss"],
        model_metrics["ece"],
        100 * model_metrics["floored_fraction"],
    )
    return predictions, model_metrics, market_metrics, ensemble


def _stacking_split_ok(train_seasons: tuple[int, ...]) -> bool:
    """Whether `stacking_split` would succeed on this training window."""
    try:
        stacking_split(train_seasons)
        return True
    except ValueError:
        return False


def _mlflow_model_params(model_name: str, overrides: dict | None) -> dict:
    """Flatten a configuration's hyperparameters for MLflow logging.

    Each key is algorithm-prefixed so tracks and configurations stay comparable
    in the MLflow UI. `*_tuned` configs log the searched `overrides`; fixed
    configs log their params.yaml block; stacking logs its base-model list.
    """
    if model_name.endswith("_tuned"):
        algorithm = model_name[: -len("_tuned")]
        return {f"{algorithm}_tuned_{k}": v for k, v in (overrides or {}).items()}
    if model_name == "stacking":
        return {"base_models": ",".join(STACKING_BASE_MODELS)}
    if model_name in _T:
        prefix = "logreg" if model_name == "logistic" else model_name
        return {f"{prefix}_{k}": v for k, v in _T[model_name].items()}
    return {}


def main() -> pd.DataFrame:
    ensure_dirs()
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    features = pd.read_parquet(FEATURES_PATH)
    log.info("loaded %d feature rows, %d columns", len(features), features.shape[1])

    wf_folds = walk_forward_splits()
    folds = wf_folds + [holdout_split()]
    log.info("running %d folds (%d walk-forward + 1 holdout)", len(folds), len(folds) - 1)

    # "stacking" needs 3 training seasons (fit + calibration + meta); the
    # earliest walk-forward fold may not have that many. Champion selection
    # must compare every candidate over the SAME walk-forward folds -- scoring
    # configurations on different subsets would let one that skips a hard fold
    # (here, the 2020 COVID season) look artificially strong. This is exactly
    # the unfair comparison run_fold's own docstring warns against one level
    # up (model vs market scored on identical rows only); it applies model vs
    # model here. `common_wf_names` is the set every candidate, including
    # stacking, can actually cover.
    common_wf_names = {
        split.name
        for split in wf_folds
        if _stacking_split_ok(split.train_seasons)
    }
    if len(common_wf_names) < len(wf_folds):
        log.warning(
            "champion selection uses %d/%d walk-forward folds for every candidate "
            "(excluding %s -- too few training seasons for stacking's 3-way split)",
            len(common_wf_names),
            len(wf_folds),
            sorted(s.name for s in wf_folds if s.name not in common_wf_names),
        )

    all_predictions = []
    # Keyed by (track, model): walk-forward scores drive champion selection,
    # holdout run ids point at the persisted model artifact to register.
    wf_summary: dict[tuple[str, str], dict[str, float]] = {}
    holdout_runs: dict[tuple[str, str], dict] = {}

    for track in TRACKS:
        cols = feature_columns(features, track)
        log.info("track=%s uses %d features", track, len(cols))

        for model_name in MODELS:
            overrides: dict | None = None
            if model_name.endswith("_tuned"):
                # Tuned ONCE per (track, algorithm), scored on walk-forward folds
                # only -- never `folds` (which appends the holdout).
                algorithm = model_name[: -len("_tuned")]
                search_space = _T[f"{algorithm}_search"]
                overrides, search_trials = tune_model(
                    features,
                    track,
                    walk_forward_splits(),
                    algorithm,
                    search_space["n_trials"],
                    search_space,
                )
                search_path = REPORTS_DIR / f"{algorithm}_search_{track}.csv"
                search_trials.to_csv(search_path, index=False)
                log.info("track=%s %s best: %s", track, model_name, overrides)
                with mlflow.start_run(run_name=f"{track}__{model_name}_search"):
                    mlflow.log_params(
                        {"track": track, "algorithm": algorithm, "n_trials": search_space["n_trials"]}
                    )
                    mlflow.log_metric(
                        "best_wf_log_loss", float(search_trials["mean_wf_log_loss"].min())
                    )
                    mlflow.log_artifact(str(search_path))

            model_params = _mlflow_model_params(model_name, overrides)

            with mlflow.start_run(run_name=f"{track}__{model_name}"):
                mlflow.log_params(
                    {
                        "track": track,
                        "model": model_name,
                        "n_features": len(cols),
                        "calibration": CALIBRATION_METHOD,
                        "seed": SEED,
                        **model_params,
                    }
                )

                fold_scores_by_name: dict[str, dict] = {}
                market_scores_by_name: dict[str, dict] = {}

                for split in folds:
                    if model_name == "stacking":
                        try:
                            predictions, model_metrics, market_metrics, fitted = run_fold_stacking(
                                features, split, track
                            )
                        except ValueError as exc:
                            log.info("  skipping stacking on %s: %s", split.name, exc)
                            continue
                    else:
                        predictions, model_metrics, market_metrics, fitted = run_fold(
                            features, split, track, model_name, overrides
                        )
                    all_predictions.append(predictions)
                    fold_scores_by_name[split.name] = model_metrics
                    market_scores_by_name[split.name] = market_metrics

                    is_holdout = split.name.startswith("holdout")
                    prefix = "holdout" if is_holdout else f"fold_{split.eval_seasons[0]}"
                    with mlflow.start_run(run_name=split.name, nested=True) as fold_run:
                        mlflow.log_params(
                            {
                                "train_seasons": str(list(split.train_seasons)),
                                "eval_seasons": str(list(split.eval_seasons)),
                            }
                        )
                        mlflow.log_metrics({f"model_{k}": v for k, v in model_metrics.items()})
                        mlflow.log_metrics({f"market_{k}": v for k, v in market_metrics.items()})

                        # Persist the holdout model: it is trained on every
                        # development season, so it is the one that would
                        # actually be deployed. Earlier folds exist to measure,
                        # not to serve, and logging all of them would bloat the
                        # store for no benefit.
                        if is_holdout:
                            sample = make_design(features.head(5), cols)
                            signature = infer_signature(sample, fitted.predict_proba(sample))
                            # Explicit cloudpickle: mlflow's default "skops"
                            # serialization refuses to load any type it does not
                            # already recognize, which includes StackingEnsemble
                            # -- verified to fail loudly ("untrusted types") with
                            # skops and to round-trip correctly with cloudpickle.
                            # Applied to every model, not just stacking, so there
                            # is one serialization path for the whole registry.
                            mlflow.sklearn.log_model(
                                fitted,
                                artifact_path=MODEL_ARTIFACT_PATH,
                                signature=signature,
                                input_example=sample,
                                serialization_format="cloudpickle",
                            )
                            # The serving layer must rebuild the design matrix in
                            # exactly this column order, so the feature list
                            # travels WITH the model rather than being re-derived
                            # from a params file that may have moved on.
                            mlflow.log_dict(
                                {
                                    "track": track,
                                    "model": model_name,
                                    "feature_columns": cols,
                                    "design_columns": list(sample.columns),
                                    "train_seasons": list(split.train_seasons),
                                    "probability_floor": PROBABILITY_FLOOR,
                                },
                                "model_contract.json",
                            )
                            holdout_runs[(track, model_name)] = {
                                "run_id": fold_run.info.run_id,
                                "model_metrics": model_metrics,
                                "market_metrics": market_metrics,
                                "feature_columns": cols,
                                "train_seasons": list(split.train_seasons),
                            }

                    mlflow.log_metrics(
                        {f"{prefix}_{k}": v for k, v in model_metrics.items()}
                    )

                # Walk-forward means for champion selection use ONLY the folds
                # common to every candidate in this track (common_wf_names) --
                # never a config-specific subset. The holdout is reported alone,
                # never averaged in.
                wf_model = pd.DataFrame(
                    [fold_scores_by_name[name] for name in common_wf_names]
                ).mean()
                wf_market = pd.DataFrame(
                    [market_scores_by_name[name] for name in common_wf_names]
                ).mean()
                mlflow.log_metrics({f"wf_mean_model_{k}": v for k, v in wf_model.items()})
                mlflow.log_metrics({f"wf_mean_market_{k}": v for k, v in wf_market.items()})
                mlflow.log_metric(
                    "wf_logloss_vs_market", wf_model["log_loss"] - wf_market["log_loss"]
                )

                log.info(
                    "%s/%s walk-forward mean: logloss %.4f vs market %.4f (%+.4f)",
                    track,
                    model_name,
                    wf_model["log_loss"],
                    wf_market["log_loss"],
                    wf_model["log_loss"] - wf_market["log_loss"],
                )
                wf_summary[(track, model_name)] = {
                    "wf_log_loss": float(wf_model["log_loss"]),
                    "wf_market_log_loss": float(wf_market["log_loss"]),
                    "wf_gap": float(wf_model["log_loss"] - wf_market["log_loss"]),
                }

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    log.info("wrote %s (%d prediction rows)", PREDICTIONS_PATH, len(predictions))

    register_champion(wf_summary, holdout_runs)
    return predictions


def register_champion(
    wf_summary: dict[tuple[str, str], dict[str, float]],
    holdout_runs: dict[tuple[str, str], dict],
) -> None:
    """Promote the best configuration to the MLflow Model Registry.

    Selection is by walk-forward mean log loss -- never by holdout performance,
    which would turn the untouched test season into a model-selection set and
    quietly invalidate it.

    The decision is written to reports/champion.json so promotion is auditable
    rather than implicit: which configuration won, by how much, and how it
    compares to the market baseline it still loses to.
    """
    best_key = min(wf_summary, key=lambda k: wf_summary[k]["wf_log_loss"])
    track, model_name = best_key
    run_info = holdout_runs[best_key]

    log.info(
        "champion: %s/%s (walk-forward log loss %.4f, market %.4f)",
        track,
        model_name,
        wf_summary[best_key]["wf_log_loss"],
        wf_summary[best_key]["wf_market_log_loss"],
    )

    client = MlflowClient()
    model_uri = f"runs:/{run_info['run_id']}/{MODEL_ARTIFACT_PATH}"
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    # The registry versions integers; semantic meaning lives in tags and the
    # alias. Serving resolves models:/<name>@champion, so swapping the model
    # later is a re-registration, not a code change anywhere downstream.
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    for key, value in {
        "track": track,
        "algorithm": model_name,
        "calibration": CALIBRATION_METHOD,
        "selected_by": "walk_forward_mean_log_loss",
        "beats_market": str(wf_summary[best_key]["wf_gap"] < 0),
    }.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)

    champion = {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": version.version,
        "alias": CHAMPION_ALIAS,
        "run_id": run_info["run_id"],
        "track": track,
        "algorithm": model_name,
        "selected_by": "walk_forward_mean_log_loss",
        "train_seasons": run_info["train_seasons"],
        "n_features": len(run_info["feature_columns"]),
        "walk_forward": wf_summary[best_key],
        "holdout": {
            "model": run_info["model_metrics"],
            "market": run_info["market_metrics"],
        },
        "all_configurations": {f"{t}/{m}": v for (t, m), v in wf_summary.items()},
    }
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2))
    log.info(
        "registered %s v%s as @%s -- see %s",
        REGISTERED_MODEL_NAME,
        version.version,
        CHAMPION_ALIAS,
        CHAMPION_PATH,
    )


if __name__ == "__main__":
    main()
