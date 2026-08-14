"""Streamlit monitoring dashboard for beating-1X2.

    streamlit run dashboard/app.py

Reads the committed artifacts under reports/, so it renders on a fresh clone
without running the pipeline. The live tab additionally talks to the FastAPI
service if one is reachable, and degrades to an explanation if not -- the
dashboard must never be the reason a demo fails.
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTCOMES, REPORTS_DIR, season_label  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DRIFT_DIR = REPORTS_DIR / "drift"
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Beating the 1X2 · Model Monitor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_visual_system() -> None:
    """Load the dashboard's local, dependency-free visual system."""
    st.markdown(
        f"<style>{(APP_DIR / 'style.css').read_text()}</style>",
        unsafe_allow_html=True,
    )


def render_hero(model: dict | None) -> None:
    """Introduce the product, its state, and the current model release."""
    if model:
        gap = float(model["walk_forward"]["wf_gap"])
        outcome = (
            f"Market leads by {gap:+.4f} log loss"
            if gap > 0
            else f"Model leads by {abs(gap):.4f} log loss"
        )
        outcome_class = "warning" if gap > 0 else "positive"
        release = f"SemVer {escape(str(model['model_semver']))} · MLflow v{model['version']}"
    else:
        outcome = "Model report unavailable"
        outcome_class = "warning"
        release = "Run the pipeline to publish a release"

    st.markdown(
        f"""
        <section class="hero-shell" aria-labelledby="dashboard-title">
          <div class="hero-copy">
            <span class="eyebrow">Production model monitor</span>
            <h1 class="hero-title" id="dashboard-title">Beating the 1X2</h1>
            <p class="hero-subtitle">
              Match-outcome intelligence for Europe’s top five leagues, evaluated
              against the closing betting market and monitored from training to serving.
            </p>
            <div class="status-row" aria-label="Model status">
              <span class="status-chip {outcome_class}">
                <span class="status-dot" aria-hidden="true"></span>{escape(outcome)}
              </span>
              <span class="meta-chip">{release}</span>
            </div>
          </div>
          <div class="hero-mark" aria-hidden="true">1X2</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_intro(eyebrow: str, title: str, description: str) -> None:
    """Create consistent wayfinding at the start of every dashboard view."""
    st.markdown(
        f"""
        <header class="section-heading">
          <span class="eyebrow">{escape(eyebrow)}</span>
          <h2>{escape(title)}</h2>
          <p>{escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary(model: dict) -> None:
    """Show the champion metadata in a responsive, glanceable grid."""
    walk_forward = model["walk_forward"]
    algorithm = str(model["algorithm"]).replace("_", " ").title()
    track = str(model["track"]).replace("_", " ")
    st.markdown(
        f"""
        <section class="metric-grid" aria-label="Champion model summary">
          <div class="metric-card">
            <span class="metric-card-label">Release</span>
            <span class="metric-card-value">v{escape(str(model['model_semver']))}</span>
            <span class="metric-card-detail">MLflow v{model['version']}</span>
          </div>
          <div class="metric-card">
            <span class="metric-card-label">Configuration</span>
            <span class="metric-card-value">{escape(algorithm)}</span>
            <span class="metric-card-detail">{escape(track)}</span>
          </div>
          <div class="metric-card">
            <span class="metric-card-label">CV log loss</span>
            <span class="metric-card-value">{walk_forward['wf_log_loss']:.4f}</span>
            <span class="metric-card-detail">Four common walk-forward folds</span>
          </div>
          <div class="metric-card">
            <span class="metric-card-label">Market benchmark</span>
            <span class="metric-card-value">{walk_forward['wf_market_log_loss']:.4f}</span>
            <span class="metric-card-detail warning">
              Gap {walk_forward['wf_gap']:+.4f} · lower is better
            </span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


apply_visual_system()


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def missing(name: str) -> None:
    st.warning(f"`{name}` not found. Run the pipeline first — see the README.")


champion = load_json(REPORTS_DIR / "champion.json")
render_hero(champion)
if champion:
    render_model_summary(champion)

results_tab, calibration_tab, drift_tab, stress_tab, live_tab = st.tabs(
    ["Performance", "Calibration", "Data drift", "Stress tests", "Live scoring"]
)

# ---------------------------------------------------------------------------
with results_tab:
    section_intro(
        "Performance",
        "Model versus market",
        "Compare every walk-forward configuration with the vig-free closing line. Lower log loss is better.",
    )
    comparison = load_csv(REPORTS_DIR / "model_vs_market.csv")
    if comparison is None:
        missing("reports/model_vs_market.csv")
    else:
        walk_forward = comparison[~comparison["fold"].str.startswith("holdout")]
        summary = (
            walk_forward.groupby(["track", "model"])[
                ["model_log_loss", "market_log_loss", "model_brier", "model_ece"]
            ]
            .mean()
            .reset_index()
        )
        summary["gap_vs_market"] = summary["model_log_loss"] - summary["market_log_loss"]
        st.dataframe(
            summary.sort_values("model_log_loss").round(4), width="stretch"
        )
        st.caption(
            "Every configuration loses to the closing line. The market_aware track "
            "is handed the market's own probability as a feature and still degrades it."
        )

        st.markdown("**Per-fold log loss**")
        chart = walk_forward.copy()
        chart["config"] = chart["track"] + "/" + chart["model"]
        st.line_chart(
            chart.pivot_table(index="fold", columns="config", values="model_log_loss")
            .join(
                chart.groupby("fold")["market_log_loss"].mean().rename("market (benchmark)")
            )
        )

    backtest = load_csv(REPORTS_DIR / "value_backtest.csv")
    if backtest is not None:
        st.subheader("Value-bet backtest")
        st.dataframe(
            backtest[
                [
                    "track", "model", "scope", "mode", "n_bets", "bet_rate",
                    "mean_odds", "strike_rate", "flat_roi", "flat_roi_lo",
                    "flat_roi_hi", "mean_clv",
                ]
            ].round(4),
            width="stretch",
        )
        st.caption(
            "ROI intervals are 95% bootstrap. Every interval spans or sits below "
            "zero, and closing-line value is negative throughout."
        )

# ---------------------------------------------------------------------------
with calibration_tab:
    section_intro(
        "Calibration",
        "Probability reliability",
        "Inspect whether predicted probabilities match observed outcome rates across home wins, draws, and away wins.",
    )
    calibration = load_csv(REPORTS_DIR / "calibration_data.csv")
    if calibration is None:
        missing("reports/calibration_data.csv")
    else:
        configs = sorted((calibration["track"] + "/" + calibration["model"]).unique())
        chosen = st.selectbox("Configuration", configs)
        track, model = chosen.split("/")
        subset = calibration[(calibration["track"] == track) & (calibration["model"] == model)]

        min_n = st.slider("Minimum matches per bin", 1, 100, 30)
        columns = st.columns(3)
        names = {"H": "home win", "D": "draw", "A": "away win"}
        for column, outcome in zip(columns, OUTCOMES, strict=True):
            with column:
                st.markdown(f"**{outcome} — {names[outcome]}**")
                part = subset[(subset["outcome"] == outcome) & (subset["n"] >= min_n)]
                if part.empty:
                    st.info("No bins meet the threshold.")
                    continue
                st.line_chart(
                    part.pivot_table(
                        index="mean_predicted", columns="source", values="observed_rate"
                    )
                )
        st.caption(
            "A perfectly calibrated line is the diagonal. Note the draw panel: "
            "neither the model nor the market ever prices a draw above ~0.42."
        )

# ---------------------------------------------------------------------------
with drift_tab:
    section_intro(
        "Data drift",
        "Distribution monitoring",
        "Track outcome, feature, and calibration shifts against a pre-COVID reference window.",
    )
    drift_summary = load_json(DRIFT_DIR / "drift_metrics.json")
    targets = load_csv(DRIFT_DIR / "target_drift.csv")

    if drift_summary is None or targets is None:
        missing("reports/drift/ — run `python -m src.drift`")
    else:
        reference = ", ".join(season_label(s) for s in drift_summary["reference_seasons"])
        st.info(
            f"Reference window: **{reference}** — deliberately stopping before "
            "2020-21 so the COVID shift is detectable rather than baked into the baseline."
        )

        st.markdown("**Target drift — outcome mix by season**")
        display = targets.copy()
        display["season"] = display["season"].map(season_label)
        st.dataframe(
            display[["season", "n", "rate_H", "rate_D", "rate_A", "p_value", "drifted"]].round(4),
            width="stretch",
        )
        st.line_chart(display.set_index("season")[["rate_H", "rate_D", "rate_A"]])
        st.caption(
            "2020-21 is flagged: home-win rate fell to 39.8% from 44–45% either "
            "side. Matches were played in empty stadiums — a genuine distribution "
            "shift, and the worst fold for every model configuration."
        )

        decay = load_csv(DRIFT_DIR / "calibration_decay.csv")
        if decay is not None:
            st.markdown("**Calibration decay — gap to the market by season**")
            decay["config"] = decay["track"] + "/" + decay["model"]
            st.line_chart(
                decay.pivot_table(index="season", columns="config", values="logloss_gap")
            )
            st.caption(
                "Gap to market, not raw log loss: a season where everyone did "
                "worse was a hard season, whereas a season where only we did "
                "worse is model decay."
            )

        drifted = drift_summary.get("drifted_features_by_season", {})
        if drifted:
            st.markdown("**Feature drift — features flagged per season**")
            st.bar_chart(pd.Series(drifted, name="drifted features"))

# ---------------------------------------------------------------------------
with stress_tab:
    section_intro(
        "Stress tests",
        "Known faults versus the deployed model",
        "Corrupt the sealed holdout in controlled ways and verify that independent monitoring signals catch each failure.",
    )
    stress = load_json(DRIFT_DIR / "stress_test.json")

    if stress is None:
        missing("reports/drift/stress_test.json — run `python -m src.stress_test`")
    else:
        base = stress["baseline"]
        st.info(
            f"Baseline: the clean {season_label(stress['holdout_season'])} holdout "
            f"({stress['n_rows']} rows) scored through the model "
            f"(**{stress['scored_via']}**). Each scenario below corrupts that same "
            "test set and re-scores it. A scenario alerts when features go missing, "
            "a fifth of features drift, the prediction mix shifts, or log loss "
            "degrades past the tolerance."
        )

        n_alerts = stress["n_alerts"]
        total = len(stress["scenarios"])
        if n_alerts == total:
            st.success(f"Verification passed — all {total}/{total} injected faults were caught.")
        elif n_alerts:
            st.warning(f"Partial verification — {n_alerts}/{total} injected faults were caught.")
        else:
            st.error("Verification failed — none of the injected faults produced an alert.")

        a, b, c, d = st.columns(4)
        a.metric("Baseline log loss", f"{base['log_loss']:.4f}")
        b.metric(
            "Baseline vs market",
            f"{base['market_log_loss']:.4f}",
            delta=f"{base['gap_vs_market']:+.4f}",
            delta_color="inverse",
        )
        c.metric("Baseline ECE", f"{base['ece']:.4f}")
        d.metric("Faults caught", f"{n_alerts}/{total}")

        table = pd.DataFrame(stress["scenarios"])
        table["features_drifted"] = (
            table["n_features_drifted"].astype(str) + "/" + table["n_features_total"].astype(str)
        )
        view = table[
            [
                "name", "alert", "features_drifted", "n_features_missing",
                "prediction_psi", "log_loss", "delta_log_loss", "ece", "accuracy",
            ]
        ].rename(columns={"name": "scenario", "n_features_missing": "missing"})

        def _flag_alert(row):
            colour = "background-color: rgba(255,75,75,0.20)" if row["alert"] else ""
            return [colour] * len(row)

        st.dataframe(
            view.style.apply(_flag_alert, axis=1),
            width="stretch",
            hide_index=True,
        )

        chart = table.set_index("name")[["delta_log_loss"]].rename(
            columns={"delta_log_loss": "log-loss increase vs baseline"}
        )
        st.markdown("**Model degradation — log-loss increase over the clean baseline**")
        st.bar_chart(chart)

        for scenario in stress["scenarios"]:
            marker = "🔴" if scenario["alert"] else "🟢"
            with st.expander(f"{marker} {scenario['name']}"):
                st.write(scenario["description"])
                if scenario["reasons"]:
                    st.markdown("**Alerts raised:**")
                    for reason in scenario["reasons"]:
                        st.markdown(f"- {reason}")
                else:
                    st.write("Within tolerance on every monitored signal.")
                st.caption(
                    f"Predicted outcome mix H/D/A: "
                    f"{scenario['outcome_mix']['H']:.3f} / "
                    f"{scenario['outcome_mix']['D']:.3f} / "
                    f"{scenario['outcome_mix']['A']:.3f}  "
                    f"(baseline {base['outcome_mix']['H']:.3f} / "
                    f"{base['outcome_mix']['D']:.3f} / {base['outcome_mix']['A']:.3f})"
                )
        st.caption(
            "This complements the Drift tab: that one detects a real historical "
            "shift (COVID); this one injects known faults into the served test set "
            "and verifies the monitor catches each."
        )

# ---------------------------------------------------------------------------
with live_tab:
    section_intro(
        "Live scoring",
        "Upcoming fixtures",
        "Check serving health, score the latest fixtures, and surface value flags against currently available prices.",
    )
    st.caption(f"Serving endpoint: `{API_URL}`")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(
            f"API {health['status']} — {health['model_name']} "
            f"{health['model_semver']} (MLflow v{health['model_version']}) "
            f"@{health['model_alias']}"
        )
        upcoming = requests.get(f"{API_URL}/predict/upcoming", timeout=60).json()

        if upcoming.get("note"):
            st.info(upcoming["note"])
        st.caption(f"Benchmark: {upcoming.get('benchmark', 'n/a')}")

        predictions = upcoming.get("predictions", [])
        if predictions:
            frame = pd.DataFrame(predictions)
            flagged = int(frame["value_flag"].sum())
            left, right = st.columns(2)
            left.metric("Fixtures scored", len(frame))
            right.metric("Flagged as value", flagged)
            st.dataframe(frame.round(4), width="stretch")
        else:
            st.write("No fixtures to score right now.")
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not reach the API at {API_URL} ({exc}).\n\n"
            "Start it with `uvicorn src.api:app` or `docker compose up`."
        )
