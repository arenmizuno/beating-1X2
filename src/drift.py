"""Stage 8: drift monitoring.

Three kinds of drift matter here, and they fail in different ways:

  Feature drift    Have the inputs moved? Measured per feature with a
                   Kolmogorov-Smirnov statistic and Population Stability Index
                   against a reference window.

  Target drift     Has the thing we predict moved? For football this is the
                   home/draw/away mix, and it is not hypothetical: 2020-21 was
                   played in empty stadiums and home-win rate fell to 39.8%
                   from 44-45% either side. That season is the worst fold for
                   every model configuration in this project, so it serves as
                   the worked example rather than injected synthetic noise.

  Calibration decay  The one that actually matters in production. A model can
                   show no feature drift and still quietly stop being
                   calibrated. Tracked as per-season log loss and ECE, and --
                   more informatively -- as the gap to the market baseline,
                   which controls for seasons that were simply harder for
                   everyone.

The statistics are computed directly rather than delegated, so the dashboard
always has numbers regardless of which Evidently version is installed. An
Evidently HTML report is generated additionally when the library is available.

Outputs:
  reports/drift/drift_metrics.json     machine-readable summary
  reports/drift/feature_drift.csv      per-feature, per-season detail
  reports/drift/calibration_decay.csv  per-season model and market scores
  reports/drift/evidently_*.html       full Evidently reports, when available
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import metrics
from src.config import (
    DEV_SEASONS,
    FEATURES_PATH,
    OUTCOMES,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_dirs,
    get_logger,
    season_label,
)

log = get_logger("drift")

DRIFT_DIR = REPORTS_DIR / "drift"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

# A feature is called drifted when the KS test is significant AND the PSI is
# materially large. Requiring both suppresses the false positives that pure
# significance testing produces on tens of thousands of rows, where a trivially
# small shift is still "significant".
KS_ALPHA = 0.01
PSI_THRESHOLD = 0.2

# Reference window: the first two development seasons (2018-19 and 2019-20).
#
# Deliberately stops BEFORE 2020-21. A reference window has to predate the shift
# you want to detect -- including the COVID season here would fold the empty-
# stadium home-advantage collapse into the baseline and make it undetectable by
# construction, which is exactly the mistake that makes real monitoring useless.
REFERENCE_SEASONS = DEV_SEASONS[:2]


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two samples, binned on the reference quantiles.

    Rule of thumb: < 0.1 no real shift, 0.1-0.2 moderate, > 0.2 significant.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 50 or len(current) < 50:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)

    # Floor both to keep the log finite when a bin empties out entirely.
    eps = 1e-6
    ref_frac = np.maximum(ref_frac, eps)
    cur_frac = np.maximum(cur_frac, eps)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def feature_drift(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """KS statistic and PSI per feature, for every season after the reference."""
    reference = features[features["season"].isin(REFERENCE_SEASONS)]
    log.info(
        "reference window: %s (%d rows)",
        ", ".join(season_label(s) for s in REFERENCE_SEASONS),
        len(reference),
    )

    rows = []
    for season in sorted(features["season"].unique()):
        if season in REFERENCE_SEASONS:
            continue
        current = features[features["season"] == season]
        for column in feature_columns:
            ref_values = reference[column].to_numpy(dtype=float)
            cur_values = current[column].to_numpy(dtype=float)
            ref_values = ref_values[np.isfinite(ref_values)]
            cur_values = cur_values[np.isfinite(cur_values)]
            if len(ref_values) < 50 or len(cur_values) < 50:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            psi = population_stability_index(ref_values, cur_values)
            rows.append(
                {
                    "season": int(season),
                    "feature": column,
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(p_value),
                    "psi": psi,
                    "drifted": bool(p_value < KS_ALPHA and psi > PSI_THRESHOLD),
                }
            )
    return pd.DataFrame(rows)


def target_drift(features: pd.DataFrame) -> pd.DataFrame:
    """Outcome-mix drift per season, against the reference window."""
    reference = features[features["season"].isin(REFERENCE_SEASONS)]
    ref_counts = reference["ftr"].value_counts().reindex(OUTCOMES).fillna(0)
    ref_rate = ref_counts / ref_counts.sum()

    rows = []
    for season in sorted(features["season"].unique()):
        current = features[features["season"] == season]
        counts = current["ftr"].value_counts().reindex(OUTCOMES).fillna(0)
        expected = ref_rate * counts.sum()
        chi2 = float(((counts - expected) ** 2 / expected.clip(lower=1)).sum())
        p_value = float(1 - stats.chi2.cdf(chi2, df=2))

        row = {
            "season": int(season),
            "n": int(counts.sum()),
            "chi2": chi2,
            "p_value": p_value,
            "drifted": bool(p_value < KS_ALPHA and season not in REFERENCE_SEASONS),
        }
        for outcome in OUTCOMES:
            row[f"rate_{outcome}"] = float(counts[outcome] / counts.sum())
            row[f"reference_rate_{outcome}"] = float(ref_rate[outcome])
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_decay(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-season model quality, and the gap to the market.

    The gap is the useful signal: a season where everyone did worse is a hard
    season, whereas a season where only WE did worse is model decay.
    """
    rows = []
    for (track, model, season), chunk in predictions.groupby(["track", "model", "season"]):
        y = chunk["y_true"].to_numpy()
        model_scores = metrics.summary(y, chunk[[f"p_{o}" for o in OUTCOMES]].to_numpy())
        market_scores = metrics.summary(
            y, chunk[[f"p_market_{o}" for o in OUTCOMES]].to_numpy()
        )
        rows.append(
            {
                "track": track,
                "model": model,
                "season": int(season),
                "n": len(chunk),
                "model_log_loss": model_scores["log_loss"],
                "market_log_loss": market_scores["log_loss"],
                "logloss_gap": model_scores["log_loss"] - market_scores["log_loss"],
                "model_ece": model_scores["ece"],
            }
        )
    return pd.DataFrame(rows).sort_values(["track", "model", "season"])


def write_evidently_reports(features: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Generate Evidently HTML, if the library is present and its API matches.

    Evidently's public API has changed substantially across releases, so this is
    strictly additive: a failure here logs and returns, because the JSON/CSV
    metrics above are the artifacts the dashboard actually depends on.
    """
    written = []
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception as exc:  # noqa: BLE001
        log.warning("Evidently unavailable or API changed (%s); skipping HTML reports", exc)
        return written

    reference = features[features["season"].isin(REFERENCE_SEASONS)][feature_columns]
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    for season in sorted(features["season"].unique()):
        if season in REFERENCE_SEASONS:
            continue
        current = features[features["season"] == season][feature_columns]
        try:
            report = Report(metrics=[DataDriftPreset()])
            evaluation = report.run(reference_data=reference, current_data=current)
            path = DRIFT_DIR / f"evidently_{season}.html"
            evaluation.save_html(str(path))
            written.append(str(path))
            log.info("wrote %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Evidently report failed for %s: %s", season, exc)
            break
    return written


def main() -> dict:
    ensure_dirs()
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    feature_columns = [
        c
        for c in features.columns
        if c.endswith(("_r5", "_r10")) or c in ("home_elo", "away_elo", "elo_diff", "rest_diff")
    ]
    log.info("monitoring %d features across %d seasons", len(feature_columns), features["season"].nunique())

    drift = feature_drift(features, feature_columns)
    drift.to_csv(DRIFT_DIR / "feature_drift.csv", index=False)

    targets = target_drift(features)
    targets.to_csv(DRIFT_DIR / "target_drift.csv", index=False)

    decay = calibration_decay(predictions)
    decay.to_csv(DRIFT_DIR / "calibration_decay.csv", index=False)

    drifted_by_season = (
        drift.groupby("season")["drifted"].sum().astype(int).to_dict() if not drift.empty else {}
    )
    log.info("features flagged as drifted, by season: %s", drifted_by_season)

    log.info(
        "outcome mix by season:\n%s",
        targets[["season", "rate_H", "rate_D", "rate_A", "p_value", "drifted"]]
        .round(4)
        .to_string(index=False),
    )

    evidently_reports = write_evidently_reports(features, feature_columns)

    summary = {
        "reference_seasons": REFERENCE_SEASONS,
        "n_features_monitored": len(feature_columns),
        "thresholds": {"ks_alpha": KS_ALPHA, "psi": PSI_THRESHOLD},
        "drifted_features_by_season": drifted_by_season,
        "target_drift_seasons": targets[targets["drifted"]]["season"].tolist(),
        "worst_calibration_gap": (
            decay.loc[decay["logloss_gap"].idxmax(), ["track", "model", "season", "logloss_gap"]]
            .to_dict()
            if not decay.empty
            else {}
        ),
        "evidently_reports": evidently_reports,
    }
    (DRIFT_DIR / "drift_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", DRIFT_DIR / "drift_metrics.json")
    return summary


if __name__ == "__main__":
    main()
