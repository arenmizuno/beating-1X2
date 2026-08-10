# Beating the 1X2 — Presentation Outline

ADSP 32021 MLOps Final Project. Slide-ready content with real values from
`reports/summary.md`, `reports/champion.json`, and `reports/drift/drift_metrics.json`.
14 slides, sized for a 10-15 minute presentation.

---

## Slide 1 — Title

**On slide:**
- **Beating the 1X2** — Can an Expected-Goals Model Beat Football's Betting Market?
- ADSP 32021: MLOps Final Project · Professor Abado · Summer 2026
- Aren Mizuno · Nick Dhaliwal · Arthur Acker · Nick Mikhail
- github.com/arenmizuno/beating-1X2

**Speaker note:** "We built a full MLOps pipeline to test whether a machine-learning model can beat the soccer betting market. Spoiler: it can't — and proving that credibly is the deliverable."

---

## Slide 2 — The Team & Contributions

**On slide (one row each):**
- **Aren Mizuno** — data ingestion + harmonization, leakage discipline, feature pipeline
- **Nick Dhaliwal** — model training, MLflow tracking + registry, AutoML/model selection
- **Arthur Acker** — Docker + FastAPI serving, deployment, drift monitoring & Evidently
- **Nick Mikhail** — evaluation, value-bet backtest, statistical honesty (bootstrap CIs)

**Speaker note:** *Adjust the split to match who actually did what — the assignment requires each member to state distinct engineering contributions.*

---

## Slide 3 — Problem Statement & EDA

**On slide:**
- **Task:** multiclass classification of match outcome — Home / Draw / Away — from pre-kickoff features
- **Value bet:** where model probability diverges from market-implied probability beyond a threshold
- **Data & scope:** top-5 European leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1), 2018-19 -> 2025-26
  - **14,285** matches ingested -> **13,786** usable (require prior match history for both sides)
  - Sources: football-data.co.uk (odds), Understat (xG), ClubElo (ratings)
  - Cross-source join rate: **99.99%** (14,284 / 14,285) after hand-verified team-name mapping
- **Class balance:** home win ~44%, draw ~26%, away win ~30%

**Speaker note:** "Three sources, three different spellings for every club — harmonizing them cleanly was a real engineering problem, solved as a global assignment, not fuzzy lookups."

---

## Slide 4 — Evaluation Metric & Two Baselines

**On slide:**
- **Primary metric: multiclass log loss** (proper scoring rule, punishes overconfident probabilities — the right lens for betting)
- Secondary: Brier score, ECE (calibration), accuracy
- **Two baselines the model must beat:**
  - Vig-free closing market line (Shin devig): **0.9691** log loss
  - Dixon-Coles Poisson goals model (never sees a price): **0.9926** log loss
- **Split discipline:** 5-fold **walk-forward** CV + **2025-26 holdout sealed** until final validation; no random K-fold anywhere

**Speaker note:** "A sharp closing line should land around 0.95-1.02 log loss. Anything that beats it by a lot would be a leakage bug, not skill — so the market baseline doubles as our leakage tripwire."

---

## Slide 5 — System Architecture

**On slide (diagram + labels):**
- **Training pipeline:** ingest (3 sources, cached/throttled) -> harmonize (Hungarian algorithm team matching) -> market devig -> **features (shift-then-roll, leakage-guarded)** -> walk-forward splits -> **train (2 tracks x 4 models)** -> MLflow tracking + Model Registry
- **Serving pipeline:** Docker + FastAPI -> resolve `models:/beating-1x2@champion` -> `model_contract.json` (feature list + column order) -> predictions + value flags -> Streamlit + Evidently monitoring
- **Two orchestrators, two jobs:**
  - **DVC** = reproducible build DAG (`dvc repro` rebuilds only what changed)
  - **Prefect** = scheduler/runtime (concurrent ingestion, retries, observability)

**Speaker note:** "Training and serving share ONE feature code path — the classic source of train/serve skew is avoided by appending upcoming fixtures to history and running the exact same pipeline."

---

## Slide 6 — Experiment Tracking & Model Selection

**On slide:**
- **MLflow** tracks every run's params, metrics, and artifacts; Model Registry holds `beating-1x2` with `@champion` alias
- **Search space: 2 tracks x 9 configurations**
  - Tracks: `market_blind` (no odds) vs `market_aware` (closing line as a feature)
  - Model families: logistic · LightGBM · XGBoost · CatBoost · neural net (MLP) · stacking
  - Three boosters carry both a fixed and a hyperparameter-tuned variant (`*_tuned`); the stack combines calibrated logistic + LightGBM + XGBoost + CatBoost
- **Champion = `market_aware / catboost_tuned`** — 54 features, registered v2
  - Selected by **walk-forward mean log loss (0.9740)**, never by holdout (which would burn the sealed test set)
  - Hyperparameters chosen by a walk-forward-only random search (25 trials); the untouched holdout never scores a trial
- **Calibration:** sigmoid (Platt), not isotonic — isotonic emitted exact zeros and lost on all 10 fold/model combos

**Speaker note:** "Show the MLflow UI here — the run comparison across the 18 configurations, with tuned CatBoost on top. Tuned boosting beat the stack, which beat logistic; the neural net underperformed at this sample size."

---

## Slide 7 — Model Results (the core evidence)

**On slide — Walk-forward mean log loss, `market_aware` track (lower is better):**

| Model | Log loss | Gap vs market |
|---|---|---|
| **catboost_tuned (champion)** | **0.9760** | **+0.0067** |
| stacking | 0.9765 | +0.0090 |
| xgboost_tuned | 0.9793 | +0.0100 |
| catboost | 0.9809 | +0.0116 |
| xgboost | 0.9940 | +0.0247 |
| logistic | 0.9946 | +0.0253 |
| lightgbm_tuned | 0.9961 | +0.0268 |
| mlp (neural net) | 1.0034 | +0.0341 |
| lightgbm | 1.0052 | +0.0359 |
| **Market (vig-free close)** | **0.9691** | — |
| Dixon-Coles baseline | 0.9926 | — |

*(The `market_blind` track shows the same ordering, ~0.010-0.030 worse per model — put it in backup.)*

**Three takeaways (callout box):**
- **The market wins everywhere** — every fold, every league, both tracks; the smallest gap anywhere is the champion's **+0.0067**, still positive
- **`market_aware` still loses** — even handed the closing line, the model *degrades* it; the xG/Elo/form features carry nothing the market hasn't priced
- **Tuned boosting > stacking > logistic > neural net**; nine model types all sit **between** the two baselines. The neural net underperforms at ~14k rows — deep learning does not pay off here

**Speaker note:** "Champion holdout: 0.9851 vs market 0.9784 — the gap narrows on holdout but never closes. Consistent story. We threw nine model families at it, tuned three of them, and the market still won."

---

## Slide 8 — Value-Bet Backtest (the economic verdict)

**On slide:**
- ROI reported with **95% bootstrap CIs** — a few hundred bets at ~5.0 odds are enormously noisy
- **Champion (market_aware/stacking), walk-forward, absolute edge:**
  - 3,154 bets · strike rate 30.7% · **flat ROI -0.01%** · 95% CI **[-7.0%, +7.0%]** · mean CLV **-1.4%**
- **No configuration is profitable; no threshold (0.02-0.20 sweep) produces an ROI whose 95% CI excludes zero**
- **Closing-line value (CLV) is negative throughout** — only **31-47%** of selections beat the close
- Best-looking point estimate: **+6.5% ROI** — but CI **[-17.8%, +34.2%]** on 362 bets -> noise, not signal

**Speaker note:** "One holdout slice shows +0.6% ROI (+3.43 units on 544 bets), but its CI is [-12%, +14%]. Reporting that as a win would be noise-mining — which is exactly the mistake the CIs are there to prevent."

---

## Slide 9 — Deployment (Containerized API)

**On slide:**
- **Docker + FastAPI**, `docker compose up --build` -> API at `:8000/docs`, dashboard at `:8501`
- Model is **swappable by alias** — replacing it is a re-registration + restart, zero serving-code changes

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + loaded model version |
| `GET /model` | champion metadata + feature contract |
| `POST /predict` | score explicit feature payloads |
| `GET /predict/upcoming` | fetch fixtures, score, flag value |
| `GET /metrics` | request counters |

- **Live prices != closing prices:** upcoming fixtures use current (softer) odds, so live value flags fire more readily than the backtest — stated in every response
- **Kalshi** carries live 1X2 markets for all five leagues (`KXEPLGAME`, etc.)

**Speaker note:** "Live demo: hit `/predict/upcoming` — note that May-August returns `n_fixtures: 0` because the leagues aren't playing. That's correct behavior, not an error."

---

## Slide 10 — Production Monitoring (baseline + real drift)

**On slide:**
- **Stack:** custom KS + PSI statistics (version-proof) + **Evidently** HTML reports per season (2020-2025)
- **Reference window: 2018-19 + 2019-20 only** — a baseline must predate the shift you want to detect
- **Thresholds:** KS alpha = 0.01, PSI = 0.20 · **32 features monitored**
- **Real detected drift — 2020-21 COVID season:**
  - **Target drift at p < 0.0001** — home-win rate fell to **39.8%** from 44-45% either side (empty stadiums)
  - Worst calibration gap: **market_blind/logistic, 2020, +0.094** log loss vs market
  - Also the worst-scoring fold for every configuration
- Feature drift flagged: 2 features in 2023, 2 in 2025

**Speaker note:** "This is drift we detected in real historical data, not injected noise. Calibration decay is tracked as gap-to-market, so a hard season for everyone doesn't false-alarm as model decay."

---

## Slide 11 — Drift Simulation & Anomaly Verification (required stress test)

**On slide:**
- **Corrupt the sealed test set three ways:**
  - Out-of-bounds random values (e.g. Elo = 50,000)
  - Swap feature columns (home <-> away)
  - Alter schema (drop / rename a required column)
- **Send drifted data to the deployed API ->**
  - Schema-invalid payloads rejected at the contract boundary (`model_contract.json`)
  - Numeric corruption breaches KS/PSI -> dashboard flags drifted features and alerts
- **Before/after screenshot:** green baseline run vs red corrupted run

**Speaker note:** "Baseline validation first — clean holdout passes and matches the monitoring baseline. Then we break it and confirm the dashboard lights up. This is the anomaly-verification step the rubric grades."

---

## Slide 12 — Conclusions

**On slide:**
- **The negative result IS the deliverable** — and it's well-evidenced:
  - No model beats the market on any fold, league, or track — nine model families, three of them hyperparameter-tuned
  - Even a principled Dixon-Coles goals model (0.9926) and tuned CatBoost, the champion (0.9760), fall short of 0.9691
  - No value threshold yields a profit whose CI excludes zero; CLV is negative throughout
- **Interpretation:** the top-5 closing line is near-efficient — there is no exploitable pre-kickoff edge in public xG/Elo/form data
- **What we actually delivered:** a reproducible, leakage-proof, containerized, monitored MLOps pipeline that establishes a negative result *credibly* — the professional-standard verdict

**Speaker note:** "A pipeline that let us fool ourselves into a fake edge would be worse than useless. The engineering value is that we can trust the 'no.'"

---

## Slide 13 — Limitations & Future Work

**On slide — Limitations:**
- **Longshot bias:** at a 0.05 absolute edge the rule flags 74-89% of fixtures at mean odds >4.7; a relative-edge filter makes it *worse*, not better
- **2020-21** is a genuine distribution shift — no honest tuning fixes it
- **CLV caveat:** value is flagged against closing-derived probability, so it's not a clean bet-at-open simulation
- **Understat is scraped** (no documented API) — schema already changed once mid-project
- **Cut from scope:** historical injury data (no free feed); prediction markets as a *historical* training source

**On slide — Future work:**
- Hosted deployment + scheduled Prefect server (currently local by design)
- Live prediction-market prices (Kalshi/Polymarket) as an inference-time input
- Semantic versioning surfaced via registry tags (MLflow versions integers)

**Speaker note:** "We flagged these three risks in the proposal before building; all three resolved as predicted."

---

## Slide 14 — Repository & Questions

**On slide:**
- **github.com/arenmizuno/beating-1X2** (public)
- Reproduce locally: `pip install -r requirements.txt` -> run stages -> `docker compose up`; cold run < 10 min
- **53 hermetic tests** (~2.5s, no network) · CI **mutation-tests the leakage guard** (flips `shift(1)`->`shift(0)`, fails if tests stay green)
- Questions?

**Speaker note:** "The leakage mutation test is the detail worth landing: a guard that never fails on a broken build is no guard at all."
