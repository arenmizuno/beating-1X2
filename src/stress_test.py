"""Stage 8b: drift stress test against the deployed model.

The COVID analysis in `src/drift.py` catches a *real* distribution shift in the
historical data. This module does the complementary thing the assignment asks
for explicitly: it takes the sealed holdout as a clean "test set", establishes a
serving baseline, then deliberately **corrupts** that test set and sends the
corrupted rows back through the *same served scoring path* to show the monitor
catching the fault.

The workflow mirrors the four required steps:

  1. Baseline validation  Score the clean holdout through the model and record
                           its log loss, ECE and predicted-outcome mix -- the
                           baseline every scenario is measured against.
  2. Drift simulation      Three injected faults, each a corruption the rubric
                           names by example:
                             out_of_bounds  every feature replaced with values
                                            far outside its observed range
                             swapped        home/away feature columns swapped and
                                            the market's home/away price flipped
                             schema_break   the entire xG feature family dropped,
                                            as if the upstream feed went dark
  3. Anomaly verification  Each corrupted set is scored through the model and
                           compared to the baseline: per-feature KS+PSI input
                           drift, a PSI on the predicted home-win probability,
                           and the holdout log-loss / ECE hit.
  4. Alerting              A scenario raises an alert when features go missing,
                           a material share of features drift, the prediction
                           distribution shifts, or log loss degrades past a
                           threshold. The dashboard renders the alerts in red.

Scoring goes through the live FastAPI service and fails closed if it is not
reachable. Unit tests and local diagnostics may opt into the identical
in-process code path explicitly with `--in-process`; production-validation
artifacts can therefore never silently claim a deployment was exercised.

Outputs:
  reports/drift/stress_test.json               baseline + per-scenario summary
  reports/drift/stress_test_feature_drift.csv  per-feature detail, per scenario
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import requests
from scipy import stats

from src import metrics
from src.config import (
    FEATURES_PATH,
    HOLDOUT_SEASON,
    OUTCOME_TO_IDX,
    OUTCOMES,
    REPORTS_DIR,
    ensure_dirs,
    get_logger,
    season_label,
)
from src.drift import KS_ALPHA, PSI_THRESHOLD, population_stability_index
from src.predict import Champion, load_champion
from src.train import apply_probability_floor, make_design

log = get_logger("stress_test")

DRIFT_DIR = REPORTS_DIR / "drift"
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# A scenario alerts when any of these trip. The thresholds are deliberately
# lenient enough that the clean baseline (measured against itself) never fires,
# and blunt enough that a real corruption always does.
LOGLOSS_TOLERANCE = 0.10   # absolute log-loss increase over the baseline
PRED_PSI_TOLERANCE = 0.20  # PSI on the predicted home-win probability
DRIFT_SHARE = 0.20         # share of monitored features flagged as drifted


# ---------------------------------------------------------------------------
# Corruptions -- each returns a full copy of the feature frame with every
# feature column still present, so the served design matrix keeps its shape.
# ---------------------------------------------------------------------------
def corrupt_out_of_bounds(clean: pd.DataFrame, feature_columns: list[str], rng) -> pd.DataFrame:
    """Replace every feature with random values far above its observed range."""
    out = clean.copy()
    for col in feature_columns:
        values = clean[col].to_numpy(dtype=float)
        lo, hi = np.nanmin(values), np.nanmax(values)
        span = (hi - lo) or 1.0
        out[col] = rng.uniform(hi + 5 * span, hi + 15 * span, size=len(out))
    return out


def corrupt_swapped(clean: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Swap home/away feature columns, flip the market price, negate diffs.

    The model receives a mirror image of reality: the home team's ratings and
    form arrive in the away slots and vice versa, the market's home and away
    prices are exchanged, and every explicit differential flips sign.
    """
    out = clean.copy()
    for col in feature_columns:
        if col.startswith("home_"):
            twin = "away_" + col[len("home_") :]
            if twin in feature_columns:
                out[col] = clean[twin].to_numpy()
                out[twin] = clean[col].to_numpy()
    if "p_market_H" in feature_columns and "p_market_A" in feature_columns:
        out["p_market_H"] = clean["p_market_A"].to_numpy()
        out["p_market_A"] = clean["p_market_H"].to_numpy()
    for col in feature_columns:
        if col.endswith("_diff") or "_diff_" in col:
            out[col] = -clean[col].to_numpy()
    return out


def corrupt_schema(clean: pd.DataFrame, dropped: list[str]) -> pd.DataFrame:
    """Drop feature columns entirely -- an upstream feed going dark."""
    out = clean.copy()
    for col in dropped:
        out[col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Serving + monitoring primitives
# ---------------------------------------------------------------------------
def score(
    frame: pd.DataFrame,
    feature_columns: list[str],
    champion: Champion,
    api_url: str | None,
    *,
    in_process: bool = False,
) -> tuple[np.ndarray, str]:
    """Return probabilities from the required API or explicit diagnostic path."""
    if in_process:
        proba, _ = apply_probability_floor(
            champion.model.predict_proba(make_design(frame, feature_columns))
        )
        return proba, "in-process (explicit diagnostic mode)"

    if not api_url:
        raise ValueError("api_url is required unless in_process=True")

    rows = []
    for _, row in frame.iterrows():
        feats = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
        rows.append({"league": row["league"], "features": feats})
    try:
        resp = requests.post(f"{api_url}/predict", json={"rows": rows}, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        preds = payload["predictions"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"deployed API validation failed at {api_url}/predict: {exc}") from exc

    if len(preds) != len(frame):
        raise RuntimeError(
            f"deployed API returned {len(preds)} predictions for {len(frame)} input rows"
        )
    proba = np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in preds])
    return proba, f"{api_url}/predict"


def input_drift(
    baseline: pd.DataFrame, current: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Per-feature KS + PSI of the corrupted set against the clean baseline."""
    rows = []
    for col in feature_columns:
        ref = baseline[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        missing = len(cur) == 0
        if missing or len(ref) < 50 or len(cur) < 50:
            rows.append(
                {
                    "feature": col,
                    "ks_statistic": float("nan"),
                    "ks_p_value": float("nan"),
                    "psi": float("nan"),
                    "drifted": False,
                    "missing": missing,
                }
            )
            continue
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        rows.append(
            {
                "feature": col,
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(p_value),
                "psi": psi,
                "drifted": bool(p_value < KS_ALPHA and psi > PSI_THRESHOLD),
                "missing": False,
            }
        )
    return pd.DataFrame(rows)


def performance(proba: np.ndarray, y_true: np.ndarray, market_proba: np.ndarray) -> dict:
    """Holdout scores for a set of predictions, plus the gap to the market."""
    model = metrics.summary(y_true, proba)
    market = metrics.summary(y_true, market_proba)
    return {
        "log_loss": model["log_loss"],
        "ece": model["ece"],
        "accuracy": model["accuracy"],
        "market_log_loss": market["log_loss"],
        "gap": model["log_loss"] - market["log_loss"],
    }


def outcome_mix(proba: np.ndarray) -> dict:
    """Mean predicted probability per outcome -- the prediction distribution."""
    return {o: float(proba[:, i].mean()) for i, o in enumerate(OUTCOMES)}


def select_holdout_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Return the complete sealed season without complete-case filtering."""
    return features[features["season"] == HOLDOUT_SEASON].reset_index(drop=True)


# ---------------------------------------------------------------------------
def run(api_url: str | None, *, in_process: bool = False) -> dict:
    ensure_dirs()
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    champion = load_champion()
    feature_columns = champion.feature_columns

    features = pd.read_parquet(FEATURES_PATH)
    clean = select_holdout_rows(features)
    y_true = clean["ftr"].map(OUTCOME_TO_IDX).to_numpy()
    market_proba = clean[[f"p_market_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    log.info(
        "clean holdout: %s, %d rows scored via %s",
        season_label(HOLDOUT_SEASON),
        len(clean),
        "explicit in-process mode" if in_process else api_url,
    )

    # 1. Baseline -----------------------------------------------------------
    base_proba, source = score(
        clean, feature_columns, champion, api_url, in_process=in_process
    )
    base_perf = performance(base_proba, y_true, market_proba)
    base_mix = outcome_mix(base_proba)

    dropped_schema = [c for c in feature_columns if "xg" in c]
    scenarios_spec = [
        (
            "out_of_bounds",
            "Every feature replaced with values 5-15x above its observed range.",
            corrupt_out_of_bounds(clean, feature_columns, rng),
        ),
        (
            "swapped_columns",
            "Home/away feature columns swapped, market home/away price flipped, "
            "differentials negated.",
            corrupt_swapped(clean, feature_columns),
        ),
        (
            f"schema_break ({len(dropped_schema)} xG fields dropped)",
            "The entire xG feature family removed, as if the Understat feed went dark.",
            corrupt_schema(clean, dropped_schema),
        ),
    ]

    # 2-4. Simulate, verify, alert -----------------------------------------
    scenarios = []
    feature_rows = []
    for name, description, corrupted in scenarios_spec:
        proba, _ = score(
            corrupted, feature_columns, champion, api_url, in_process=in_process
        )
        drift = input_drift(clean, corrupted, feature_columns)
        perf = performance(proba, y_true, market_proba)
        pred_psi = population_stability_index(base_proba[:, 0], proba[:, 0])

        n_total = len(feature_columns)
        n_drifted = int(drift["drifted"].sum())
        n_missing = int(drift["missing"].sum())
        delta_log_loss = perf["log_loss"] - base_perf["log_loss"]

        reasons = []
        if n_missing:
            reasons.append(f"{n_missing} feature(s) missing (schema break)")
        if n_drifted >= max(5, int(DRIFT_SHARE * n_total)):
            reasons.append(f"{n_drifted}/{n_total} features drifted (KS+PSI)")
        if np.isfinite(pred_psi) and pred_psi > PRED_PSI_TOLERANCE:
            reasons.append(f"prediction PSI {pred_psi:.2f}")
        if delta_log_loss > LOGLOSS_TOLERANCE:
            reasons.append(f"log loss +{delta_log_loss:.3f} vs baseline")

        scenarios.append(
            {
                "name": name,
                "description": description,
                "n_features_total": n_total,
                "n_features_drifted": n_drifted,
                "n_features_missing": n_missing,
                "prediction_psi": None if not np.isfinite(pred_psi) else round(pred_psi, 4),
                "log_loss": round(perf["log_loss"], 4),
                "delta_log_loss": round(delta_log_loss, 4),
                "ece": round(perf["ece"], 4),
                "accuracy": round(perf["accuracy"], 4),
                "gap_vs_market": round(perf["gap"], 4),
                "outcome_mix": {o: round(v, 4) for o, v in outcome_mix(proba).items()},
                "alert": bool(reasons),
                "reasons": reasons,
            }
        )
        drift = drift.assign(scenario=name)
        feature_rows.append(drift)
        log.info(
            "%-32s drifted %2d/%d | missing %2d | pred PSI %s | logloss %.4f (%+.4f) | ALERT=%s",
            name,
            n_drifted,
            n_total,
            n_missing,
            "n/a" if not np.isfinite(pred_psi) else f"{pred_psi:.2f}",
            perf["log_loss"],
            delta_log_loss,
            bool(reasons),
        )

    summary = {
        "holdout_season": HOLDOUT_SEASON,
        "n_rows": len(clean),
        "scored_via": source,
        "champion": {
            "version": champion.version,
            "model_semver": champion.model_semver,
            "track": champion.track,
            "algorithm": champion.algorithm,
        },
        "thresholds": {
            "logloss_tolerance": LOGLOSS_TOLERANCE,
            "prediction_psi_tolerance": PRED_PSI_TOLERANCE,
            "drift_share": DRIFT_SHARE,
            "ks_alpha": KS_ALPHA,
            "psi": PSI_THRESHOLD,
        },
        "baseline": {
            "log_loss": round(base_perf["log_loss"], 4),
            "market_log_loss": round(base_perf["market_log_loss"], 4),
            "gap_vs_market": round(base_perf["gap"], 4),
            "ece": round(base_perf["ece"], 4),
            "accuracy": round(base_perf["accuracy"], 4),
            "outcome_mix": {o: round(v, 4) for o, v in base_mix.items()},
        },
        "scenarios": scenarios,
        "n_alerts": sum(s["alert"] for s in scenarios),
    }
    (DRIFT_DIR / "stress_test.json").write_text(json.dumps(summary, indent=2))
    pd.concat(feature_rows, ignore_index=True).to_csv(
        DRIFT_DIR / "stress_test_feature_drift.csv", index=False
    )
    write_html_report(summary)
    log.info(
        "wrote %s (%d/%d scenarios alerted)",
        DRIFT_DIR / "stress_test.json",
        summary["n_alerts"],
        len(scenarios),
    )
    return summary


def write_html_report(summary: dict) -> None:
    """Render a self-contained HTML monitoring report from the summary.

    Static HTML + CSS with no JavaScript, so it renders on any clone and
    screenshots cleanly -- the artifact behind the deck's Drift Analysis slide.
    The Streamlit dashboard shows the same numbers interactively.
    """
    base = summary["baseline"]
    n_alerts, total = summary["n_alerts"], len(summary["scenarios"])
    banner_class = "alert" if n_alerts else "ok"
    banner_text = (
        f"MONITOR ALERT — {n_alerts}/{total} injected faults caught"
        if n_alerts
        else "No anomalies detected"
    )
    max_delta = max((s["delta_log_loss"] for s in summary["scenarios"]), default=1.0) or 1.0

    def mix(m: dict) -> str:
        return f"{m['H']:.3f} / {m['D']:.3f} / {m['A']:.3f}"

    scenario_blocks = []
    for s in summary["scenarios"]:
        reasons = "".join(f"<li>{r}</li>" for r in s["reasons"]) or "<li>within tolerance</li>"
        bar_pct = max(2.0, 100.0 * s["delta_log_loss"] / max_delta)
        psi = "n/a" if s["prediction_psi"] is None else f"{s['prediction_psi']:.2f}"
        state = "alert" if s["alert"] else "ok"
        scenario_blocks.append(
            f"""
        <div class="card {state}">
          <div class="card-head">
            <span class="dot {state}"></span>
            <h3>{s['name']}</h3>
            <span class="pill {state}">{'ALERT' if s['alert'] else 'OK'}</span>
          </div>
          <p class="desc">{s['description']}</p>
          <div class="grid">
            <div><span class="k">features drifted</span><span class="v">{s['n_features_drifted']}/{s['n_features_total']}</span></div>
            <div><span class="k">features missing</span><span class="v">{s['n_features_missing']}</span></div>
            <div><span class="k">prediction PSI</span><span class="v">{psi}</span></div>
            <div><span class="k">holdout log loss</span><span class="v">{s['log_loss']:.4f}</span></div>
            <div><span class="k">Δ vs baseline</span><span class="v big">+{s['delta_log_loss']:.4f}</span></div>
            <div><span class="k">predicted H/D/A</span><span class="v">{mix(s['outcome_mix'])}</span></div>
          </div>
          <div class="barwrap"><div class="bar {state}" style="width:{bar_pct:.1f}%"></div></div>
          <ul class="reasons">{reasons}</ul>
        </div>"""
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>beating-1X2 — drift stress test</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 32px 40px; color: #1a1a2e; background: #fff; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color: #555; font-size: 13px; margin-bottom: 20px; }}
  .sub code {{ background: #f1f1f4; padding: 1px 6px; border-radius: 4px; }}
  .banner {{ font-size: 20px; font-weight: 700; padding: 14px 20px; border-radius: 10px;
            margin-bottom: 22px; }}
  .banner.alert {{ background: #fdecec; color: #b3202a; border: 1px solid #f2b8bc; }}
  .banner.ok {{ background: #eafaf0; color: #1c7a45; border: 1px solid #b8e6cb; }}
  .baseline {{ display: flex; gap: 26px; padding: 16px 20px; background: #f7f7fa;
              border: 1px solid #e6e6ee; border-radius: 10px; margin-bottom: 26px; }}
  .baseline div span {{ display: block; }}
  .baseline .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #777; }}
  .baseline .v {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }}
  .card {{ border: 1px solid #e6e6ee; border-radius: 10px; padding: 16px 18px; }}
  .card.alert {{ border-color: #f2b8bc; background: #fffafa; }}
  .card.ok {{ border-color: #b8e6cb; background: #fbfffc; }}
  .card-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .card-head h3 {{ font-size: 14px; margin: 0; flex: 1; font-family: ui-monospace, Menlo, monospace; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .dot.alert {{ background: #d3202a; }} .dot.ok {{ background: #1c7a45; }}
  .pill {{ font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }}
  .pill.alert {{ background: #d3202a; color: #fff; }} .pill.ok {{ background: #1c7a45; color: #fff; }}
  .desc {{ font-size: 12px; color: #555; margin: 4px 0 12px; min-height: 48px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; margin-bottom: 12px; }}
  .grid div span {{ display: block; }}
  .grid .k {{ font-size: 10px; text-transform: uppercase; letter-spacing: .03em; color: #888; }}
  .grid .v {{ font-size: 15px; font-weight: 600; }}
  .grid .v.big {{ color: #b3202a; }}
  .barwrap {{ height: 8px; background: #eee; border-radius: 6px; overflow: hidden; margin-bottom: 10px; }}
  .bar {{ height: 100%; }} .bar.alert {{ background: #d3202a; }} .bar.ok {{ background: #1c7a45; }}
  .reasons {{ margin: 0; padding-left: 18px; font-size: 12px; color: #444; }}
  .reasons li {{ margin: 2px 0; }}
  .foot {{ margin-top: 24px; font-size: 11px; color: #888; }}
</style></head><body>
  <h1>beating-1X2 — drift stress test</h1>
  <div class="sub">
    Champion <b>v{summary['champion']['version']}</b>
    (semver {summary['champion']['model_semver']})
    ({summary['champion']['track']}/{summary['champion']['algorithm']}) ·
    clean holdout {season_label(summary['holdout_season'])}, {summary['n_rows']} rows ·
    scored via <code>{summary['scored_via']}</code>
  </div>
  <div class="banner {banner_class}">{banner_text}</div>
  <div class="baseline">
    <div><span class="k">Baseline log loss</span><span class="v">{base['log_loss']:.4f}</span></div>
    <div><span class="k">vs market</span><span class="v">{base['market_log_loss']:.4f} ({base['gap_vs_market']:+.4f})</span></div>
    <div><span class="k">ECE</span><span class="v">{base['ece']:.4f}</span></div>
    <div><span class="k">Accuracy</span><span class="v">{base['accuracy']:.4f}</span></div>
    <div><span class="k">Predicted H/D/A</span><span class="v">{mix(base['outcome_mix'])}</span></div>
  </div>
  <div class="cards">{''.join(scenario_blocks)}</div>
  <div class="foot">
    Each scenario corrupts the clean holdout and re-scores it through the deployed model.
    Alert rule: a feature goes missing, ≥{int(100*summary['thresholds']['drift_share'])}% of
    features drift (KS p&lt;{summary['thresholds']['ks_alpha']} and PSI&gt;{summary['thresholds']['psi']}),
    prediction PSI&gt;{summary['thresholds']['prediction_psi_tolerance']},
    or log loss rises &gt;{summary['thresholds']['logloss_tolerance']} over baseline.
  </div>
</body></html>"""
    (DRIFT_DIR / "stress_test.html").write_text(html)
    log.info("wrote %s", DRIFT_DIR / "stress_test.html")


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Required deployed API base URL (default: API_URL or http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Explicit diagnostic mode that bypasses the deployed API; never use for final evidence.",
    )
    args = parser.parse_args()
    return run(args.api_url, in_process=args.in_process)


if __name__ == "__main__":
    main()
