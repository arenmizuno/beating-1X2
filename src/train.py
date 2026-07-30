"""Stage 6: train and evaluate models across the walk-forward folds.

Two tracks, and the contrast between them is the actual research question:

  market_blind   Features only. This is the only track that can be honestly
                 compared against the closing line, because it never sees it.

  market_aware   The same features plus the vig-free market probability. If it
                 scores no better than market_blind, our xG and Elo signal adds
                 nothing the market had not already priced -- which is a real
                 finding, and the more likely one. If it beats the market alone,
                 the features carry residual information.

Two models per track:

  logistic       Multinomial logistic regression. Not a throwaway baseline --
                 on football outcome data a well-specified linear model on
                 Elo/form differentials is genuinely competitive with boosting,
                 and it is naturally well calibrated.

  lightgbm       Gradient boosting, for any non-linearity the linear model misses.

Both models get identical treatment: fitted on the same seasons, then calibrated
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
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    SEED,
    ensure_dirs,
    get_logger,
)
from src.splits import Split, calibration_split, holdout_split, walk_forward_splits

log = get_logger("train")

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

_T = PARAMS["train"]
TRACKS: list[str] = _T["tracks"]
CALIBRATION_METHOD: str = _T["calibration_method"]
PROBABILITY_FLOOR: float = _T["probability_floor"]
EXPERIMENT_NAME: str = _T["mlflow_experiment"]
N_BINS: int = PARAMS["evaluate"]["calibration_bins"]

MARKET_FEATURES = [f"p_market_{o}" for o in OUTCOMES]

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
    track.
    """
    excluded = set(NON_FEATURE_COLUMNS)
    for col in df.columns:
        if col.startswith(("odds_", "p_mult_", "p_shin_", "p_market_")):
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


def build_model(name: str) -> Pipeline | LGBMClassifier:
    if name == "logistic":
        # Impute and scale: logistic regression cannot take NaN, and the
        # features are on wildly different scales (Elo ~1500, rest days ~7).
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
    if name == "lightgbm":
        # No imputation: LightGBM routes NaN down its own branch, which is more
        # informative than a median fill for "team has no xG history yet".
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=SEED,
            verbose=-1,
            **_T["lightgbm"],
        )
    raise ValueError(f"unknown model: {name}")


def fit_calibrated(model_name: str, train: pd.DataFrame, cols: list[str], split: Split):
    """Fit on the earlier seasons, calibrate on the most recent one.

    The calibration split is temporal, not random: calibrating on rows drawn
    from the seasons the model trained on would fit the calibrator to in-sample
    confidence and hide exactly the overconfidence we want to correct.
    """
    fit_seasons, calib_seasons = calibration_split(split.train_seasons)

    fit_rows = train[train["season"].isin(fit_seasons)]
    calib_rows = train[train["season"].isin(calib_seasons)]

    base = build_model(model_name)
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


def run_fold(
    features: pd.DataFrame, split: Split, track: str, model_name: str
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], object]:
    """Train one (track, model, fold) and score it against the market.

    Returns the fitted (calibrated) estimator alongside the metrics so the
    holdout fold's model -- the one trained on the most data, and therefore the
    one worth deploying -- can be registered without refitting.
    """
    cols = feature_columns(features, track)
    train, evaluation = split.apply(features)

    # market_aware needs a price to train on, so drop unpriced rows from its
    # training set. market_blind keeps everything it can use.
    if track == "market_aware":
        train = train.dropna(subset=MARKET_FEATURES)

    model, fit_seasons, calib_seasons = fit_calibrated(model_name, train, cols, split)

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
        predictions[f"odds_{outcome}"] = scored[f"odds_{outcome}"].values
        # Carried for closing-line value; may be NaN where no opening price.
        if f"odds_open_{outcome}" in scored.columns:
            predictions[f"odds_open_{outcome}"] = scored[f"odds_open_{outcome}"].values

    log.info(
        "  %-9s %-12s %-14s n=%5d | logloss %.4f (market %.4f, %+.4f) | ece %.4f | floored %.3f%%",
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


def main() -> pd.DataFrame:
    ensure_dirs()
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    features = pd.read_parquet(FEATURES_PATH)
    log.info("loaded %d feature rows, %d columns", len(features), features.shape[1])

    folds = walk_forward_splits() + [holdout_split()]
    log.info("running %d folds (%d walk-forward + 1 holdout)", len(folds), len(folds) - 1)

    all_predictions = []
    # Keyed by (track, model): walk-forward scores drive champion selection,
    # holdout run ids point at the persisted model artifact to register.
    wf_summary: dict[tuple[str, str], dict[str, float]] = {}
    holdout_runs: dict[tuple[str, str], dict] = {}

    for track in TRACKS:
        cols = feature_columns(features, track)
        log.info("track=%s uses %d features", track, len(cols))

        for model_name in ("logistic", "lightgbm"):
            if model_name == "lightgbm":
                model_params = {f"lgbm_{k}": v for k, v in _T["lightgbm"].items()}
            else:
                model_params = {f"logreg_{k}": v for k, v in _T["logistic"].items()}

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

                fold_scores, market_scores = [], []

                for split in folds:
                    predictions, model_metrics, market_metrics, fitted = run_fold(
                        features, split, track, model_name
                    )
                    all_predictions.append(predictions)
                    fold_scores.append(model_metrics)
                    market_scores.append(market_metrics)

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
                            mlflow.sklearn.log_model(
                                fitted,
                                artifact_path=MODEL_ARTIFACT_PATH,
                                signature=signature,
                                input_example=sample,
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

                # Walk-forward means exclude the holdout, which is reported alone.
                wf_model = pd.DataFrame(fold_scores[:-1]).mean()
                wf_market = pd.DataFrame(market_scores[:-1]).mean()
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
