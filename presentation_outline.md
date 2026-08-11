# Beating the 1X2 — Presentation Outline

ADSP 32021 MLOps Final Project. Slide-ready content with real values from
`reports/summary.md`, `reports/champion.json`, and `reports/drift/drift_metrics.json`.
16 slides, sized for a 10-15 minute presentation.

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

## Slide 3 — Problem Statement

**On slide:**
- **Research question:** can a machine-learning model using public pre-kickoff data (xG, Elo, form) beat the soccer betting market's closing line?
- **Task:** multiclass classification of match outcome — Home / Draw / Away — from pre-kickoff features only
- **Value bet:** flag a wager where model probability exceeds the market-implied probability by more than a threshold
- **What "beating the market" means here:** a lower probabilistic **log loss** than the vig-free closing line — a genuine informational edge, not just accuracy
- **Why it's hard:** the closing line aggregates the sharpest money, so beating it is the standard test of market efficiency

**Speaker note:** "The whole project is one question — is there an exploitable pre-kickoff edge the market hasn't already priced? Everything downstream exists to answer that credibly."

---

## Slide 4 — Data & EDA

**On slide:**
- **Scope:** top-5 European leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1), seasons **2018-19 -> 2025-26**
- **Volume:** **14,285** matches ingested -> **13,786** usable (both sides need prior match history for rolling features)
- **Three sources, one row per match:** football-data.co.uk (odds / results) · Understat (expected goals) · ClubElo (team ratings)
- **Cross-source join rate: 99.99%** (14,284 / 14,285) after hand-verified team-name mapping
- **Class balance:** home win **~44%**, draw **~26%**, away win **~30%** — persistent home advantage
- **EDA highlight:** home-win rate holds at 44-45% every season *except* **2020-21**, where it fell to **39.8%** (empty stadiums) — foreshadows the drift slide

**Speaker note:** "Three sources spell every club differently — harmonizing them cleanly was a real engineering problem, solved as a global assignment, not fuzzy lookups. The home-advantage dip in 2020-21 is real, and it comes back in monitoring."

---

## Slide 5 — Evaluation Metric & Two Baselines

**On slide:**
- **Primary metric: multiclass log loss** — a proper scoring rule that punishes overconfident probabilities; the right lens for betting, where *calibrated probability* matters more than raw accuracy
- Secondary: Brier score, ECE (calibration), accuracy
- **Two baselines, each chosen to answer a different question:**
  - **Vig-free closing market line** (Shin devig): **0.9691** — *the bar to clear.* The sharpest public probability there is; beating it is the definition of an edge. Doubles as a **leakage tripwire** — a sharp line lands ~0.95-1.02, so beating it by a lot means a bug, not skill.
  - **Dixon-Coles Poisson goals model** (never sees a price): **0.9926** — *the "reasonable model" control.* Shows what a principled, price-blind approach achieves, isolating whether our ML adds anything over a classic goals model. Kept out of the feature set so it stays an independent yardstick.
- **Split discipline:** 5-fold **walk-forward** CV + **2025-26 holdout sealed** until final validation; no random K-fold anywhere

**Speaker note:** "We picked two baselines on purpose. The market line is the thing to beat and our leakage alarm. Dixon-Coles is the sanity control — if the fancy models can't even out-predict a textbook Poisson goals model, that's worth knowing. Our models land between the two."

---

## Slide 6 — System Architecture (overview)

**On slide (one end-to-end diagram — boxes + arrows, minimal prose):**
- **Training:** ingest (3 sources) -> harmonize -> market devig -> features (leakage-guarded) -> walk-forward splits -> train (2 tracks x 9 configs) -> MLflow tracking + Model Registry
- **Serving:** Docker + FastAPI -> resolve `models:/beating-1x2@champion` -> predictions + value flags -> Streamlit + Evidently monitoring
- **Two orchestrators, two jobs:** **DVC** = reproducible build DAG (`dvc repro`) · **Prefect** = scheduler/runtime (concurrent ingestion, retries, observability)
- *The next four slides zoom into each subsystem: data + features, training/selection, deployment, monitoring.*

**Speaker note:** "This is the map. One thing to flag up front — training and serving share ONE feature code path, so there is no train/serve skew. The following slides drill into each box."

---

## Slide 7 — Ingestion & Feature Engineering

**On slide:**
- **Ingestion:** three third-party sources pulled concurrently with retries (Prefect), cached and throttled to stay polite
- **Harmonization:** every club is spelled three ways; team-name matching is solved as a **global assignment (Hungarian algorithm)**, not fuzzy per-row lookups -> **99.99%** join
- **Market devig:** Shin method strips the bookmaker margin -> vig-free probabilities
- **54 features:** rolling form (r5/r10), Elo ratings, rest days, and Dixon-Coles-derived attack/defense strengths
- **Leakage discipline (the centerpiece): shift-then-roll** — every rolling window is shifted by one match so a game never sees itself; only prior matches feed a fixture; enforced by the walk-forward splits
- **Mutation-tested guard:** CI flips `shift(1)` -> `shift(0)` and **fails the build if the leakage tests stay green**
- **One feature code path** for training and serving -> no train/serve skew

**Speaker note:** "This is where most of the engineering risk lived. The leakage guard is the detail to land — and it is itself tested by mutation, so a guard that silently breaks can't pass CI. The same code builds features at train time and at inference."

---

## Slide 8 — Experiment Tracking & Model Selection

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

## Slide 9 — Model Results (the core evidence)

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

## Slide 10 — Value-Bet Backtest (the economic verdict)

**On slide:**
- ROI reported with **95% bootstrap CIs** — a few hundred bets at ~6.1 odds are enormously noisy
- **Champion (market_aware/catboost_tuned), walk-forward, absolute edge:**
  - 2,329 bets · strike rate 31.8% · **flat ROI +0.08%** · 95% CI **[-9.3%, +9.6%]** · mean CLV **-3.3%**
- **No configuration is profitable** — across the **0.02-0.20 threshold sweep**, not one produces an ROI whose 95% CI excludes zero
- **Closing-line value is negative** in every walk-forward configuration — only **21-48%** of selections beat the close
- Best point estimate anywhere: **+0.6% ROI** (market_aware/logistic, holdout, +3.43 units on 544 bets) — but CI **[-12.0%, +14.4%]** -> noise, not signal

**Speaker note:** "The single best-looking slice is +0.6% ROI on 544 holdout bets — and its CI is [-12%, +14%]. Reporting that as a win would be noise-mining, which is exactly what the CIs are there to prevent. Zero of our configurations clear zero with confidence."

---

## Slide 11 — Deployment (Containerized API)

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

## Slide 12 — Production Monitoring (baseline + real drift)

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

## Slide 13 — Drift Simulation & Anomaly Verification (required stress test)

**Screenshot:** `reports/drift/stress_test.png` — the red "MONITOR ALERT — 3/3
injected faults caught" report. Same view in the dashboard's **Stress test** tab.

**On slide — baseline + three injected faults, scored through the deployed API:**
- **Baseline:** clean 2025-26 holdout (1,702 rows) -> log loss **0.9828**, within tolerance of the monitoring baseline
- **Each fault is caught by a *different* detector** (the whole point):

| Fault | Corruption | What caught it |
|---|---|---|
| out_of_bounds | every feature 5-15x past its range | **54/54** features drift + prediction PSI **12.4** |
| swapped_columns | home/away swapped, market price flipped | log loss **+0.386** (performance) — only 3/54 marginals move |
| schema_break | xG feature family dropped | **12 missing** features (schema) — log loss barely moves |

**Speaker note:** "Baseline validation first — the clean holdout passes. Then we
inject faults and send them to the *deployed model*. The lesson is the table's
right column: marginal input-drift monitoring alone would miss the column swap,
and performance monitoring alone would miss the schema break. You need all three
signals. This is the anomaly-verification step the rubric grades — and it
complements Slide 12, which is a *real* detected shift rather than an injected one."

---

## Slide 14 — Conclusions

**On slide:**
- **The negative result IS the deliverable** — and it's well-evidenced:
  - No model beats the market on any fold, league, or track — nine model families, three of them hyperparameter-tuned
  - Even a principled Dixon-Coles goals model (0.9926) and tuned CatBoost, the champion (0.9760), fall short of 0.9691
  - No value threshold yields a profit whose CI excludes zero; CLV is negative throughout
- **Interpretation:** the top-5 closing line is near-efficient — there is no exploitable pre-kickoff edge in public xG/Elo/form data
- **What we actually delivered:** a reproducible, leakage-proof, containerized, monitored MLOps pipeline that establishes a negative result *credibly* — the professional-standard verdict

**Speaker note:** "A pipeline that let us fool ourselves into a fake edge would be worse than useless. The engineering value is that we can trust the 'no.'"

---

## Slide 15 — Limitations & Future Work

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

## Slide 16 — Repository & Questions

**On slide:**
- **github.com/arenmizuno/beating-1X2** (public)
- Reproduce locally: `pip install -r requirements.txt` -> run stages -> `docker compose up`; cold run < 10 min
- **84 hermetic tests** (no network) · CI **mutation-tests the leakage guard** (flips `shift(1)`->`shift(0)`, fails if tests stay green)
- Questions?

**Speaker note:** "The leakage mutation test is the detail worth landing: a guard that never fails on a broken build is no guard at all."
