# Beating the 1X2 — Presentation Outline

ADSP 32021 MLOps Final Project. Slide-ready content with real values from
`reports/summary.md`, `reports/champion.json`, and `reports/drift/drift_metrics.json`.
17 slides, sized for a 10-15 minute presentation.

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
- *The following slides zoom into each subsystem: data + features, models, training/selection, deployment, monitoring.*

**Speaker note:** "This is the map. One thing to flag up front — training and serving share ONE feature code path, so there is no train/serve skew. The following slides drill into each box."

---

## Slide 7 — Ingestion & Feature Engineering

**On slide:**
- **Ingestion:** three third-party sources pulled concurrently with retries (Prefect), cached and throttled to stay polite
- **Harmonization:** every club is spelled three ways; team-name matching is solved as a **global assignment (Hungarian algorithm)**, not fuzzy per-row lookups -> **99.99%** join
- **Market devig:** Shin method strips the bookmaker margin -> vig-free probabilities
- **86 features** across two generations:
  - *Base:* rolling form (r5/r10), Elo ratings, rest days, Dixon-Coles attack/defense strengths
  - *Engineered (Tier-1):* strength-of-schedule + Elo over-performance, home/away-split form, recency-weighted (EWMA) form, fixture congestion, season points-per-game, promoted-team flag
- **Honest result:** the engineered features left the gap to the market **unchanged** — it already prices team strength, form and schedule (see Conclusions). Kept as evidence, not spin.
- **Leakage discipline (the centerpiece): shift-then-roll** — every rolling window is shifted by one match so a game never sees itself; only prior matches feed a fixture; enforced by the walk-forward splits
- **Mutation-tested guard:** CI flips `shift(1)` -> `shift(0)` and **fails the build if the leakage tests stay green**
- **One feature code path** for training and serving -> no train/serve skew

**Speaker note:** "This is where most of the engineering risk lived. The leakage guard is the detail to land — and it is itself tested by mutation, so a guard that silently breaks can't pass CI. The same code builds features at train time and at inference."

---

## Slide 8 — Models & Modeling Approach

**On slide:**
- **Two experimental tracks** (the core design question):
  - `market_blind` — features only, no odds: *can we predict outcomes from the game itself?*
  - `market_aware` — the closing line added as a feature: *given the market's own number, can we improve on it?*
- **Six model families, climbing the complexity ladder:**
  - **Logistic regression** — linear, interpretable floor
  - **LightGBM · XGBoost · CatBoost** — gradient-boosted trees (the workhorses)
  - **MLP** — a neural net, to test whether deep learning helps at this scale
  - **Stacking** — a logistic meta-learner over the calibrated boosters
- **Tuned variants:** each booster also runs a hyperparameter-tuned version (`*_tuned`) -> **9 configs per track, 18 total**
- **All models are probability-calibrated** — betting needs trustworthy probabilities, not just correct rankings

**Speaker note:** "The ladder is the point. If a tuned ensemble of four models plus a neural net still can't beat the closing line, that's strong evidence the edge isn't there — not that we picked the wrong model. Climbing from linear to deep to ensemble is how we falsify 'we just needed a better model.'"

---

## Slide 9 — Experiment Tracking & Model Selection

**On slide:**
- **MLflow** tracks every run's params, metrics, and artifacts; Model Registry holds `beating-1x2` with `@champion` alias + tags
- **18 runs** (2 tracks x 9 configs) compared on a single metric
- **Champion = `market_aware / catboost_tuned`** — 86 features, registered v3
  - Selected by **walk-forward mean log loss (0.9749)**, never by holdout (which would burn the sealed test set)
  - Hyperparameters chosen by a **walk-forward-only** random search (25 trials); the untouched holdout never scores a trial
- **Calibration choice:** sigmoid (Platt) over isotonic — isotonic emitted exact zeros and lost on all 10 fold/model combos

**Speaker note:** "Show the MLflow UI here — the run comparison across the 18 configurations, with tuned CatBoost on top. Tuned boosting beat the stack, which beat logistic; the neural net underperformed at this sample size. Nothing is selected on the holdout — only walk-forward mean."

---

## Slide 10 — Model Results (the core evidence)

**On slide — Walk-forward mean log loss, `market_aware` track (lower is better):**

| Model | Log loss | Gap vs market |
|---|---|---|
| **catboost_tuned (champion)** | **0.9766** | **+0.0073** |
| stacking | 0.9779 | +0.0104 |
| xgboost_tuned | 0.9790 | +0.0097 |
| catboost | 0.9826 | +0.0133 |
| xgboost | 0.9942 | +0.0249 |
| lightgbm_tuned | 0.9962 | +0.0269 |
| mlp (neural net) | 1.0000 | +0.0307 |
| logistic | 1.0012 | +0.0319 |
| lightgbm | 1.0048 | +0.0356 |
| **Market (vig-free close)** | **0.9691** | — |
| Dixon-Coles baseline | 0.9926 | — |

*(The `market_blind` track shows the same ordering, ~0.010-0.030 worse per model — put it in backup.)*

**Three takeaways (callout box):**
- **The market wins everywhere** — every fold, every league, both tracks; the smallest gap anywhere is the champion's **+0.0073**, still positive
- **`market_aware` still loses** — even handed the closing line, the model *degrades* it; and a round of engineered features (opponent strength, venue, congestion, standing) left the gap unchanged — nothing the market hasn't already priced
- **Tuned boosting > stacking > logistic > neural net**; nine model types all sit **between** the two baselines. The neural net underperforms at ~14k rows — deep learning does not pay off here

**Speaker note:** "Champion holdout: 0.9853 vs market 0.9784 — the gap narrows on holdout but never closes. Consistent story. We threw nine model families at it, tuned three of them, engineered a dozen more features, and the market still won."

---

## Slide 11 — Value-Bet Backtest (the economic verdict)

**On slide:**
- ROI reported with **95% bootstrap CIs** — a few hundred bets at ~6.0 odds are enormously noisy
- **Champion (market_aware/catboost_tuned), walk-forward, absolute edge:**
  - 2,429 bets · strike rate 31.2% · **flat ROI -1.6%** · 95% CI **[-10.0%, +7.8%]** · mean CLV **-3.4%**
- **No configuration is profitable** — across the **0.02-0.20 threshold sweep**, not one produces an ROI whose 95% CI excludes zero
- **Closing-line value is negative** in every walk-forward configuration — only **23-46%** of selections beat the close
- Best point estimate anywhere: a flat **+0.03% ROI** (market_aware/catboost_tuned, walk-forward relative edge, +0.4 units on 1,319 bets) — CI **[-14.9%, +15.5%]** -> pure noise

**Speaker note:** "The best-looking slice across every model and threshold is +0.03% ROI — essentially zero — and its CI spans -15% to +15%. There is no configuration whose profit clears zero with confidence, and closing-line value is negative throughout."

---

## Slide 12 — Deployment (Containerized API)

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

## Slide 13 — Production Monitoring (baseline + real drift)

**On slide:**
- **Stack:** custom KS + PSI statistics (version-proof) + **Evidently** HTML reports per season (2020-2025)
- **Reference window: 2018-19 + 2019-20 only** — a baseline must predate the shift you want to detect
- **Thresholds:** KS alpha = 0.01, PSI = 0.20 · **54 features monitored**
- **Real detected drift — 2020-21 COVID season:**
  - **Target drift at p < 0.0001** — home-win rate fell to **39.8%** from 44-45% either side (empty stadiums)
  - Worst calibration gap: **market_blind/logistic, 2020, +0.095** log loss vs market
  - Also the worst-scoring fold for every configuration
- Feature drift flagged: 2-6 of 54 features in recent seasons (roster/schedule churn)

**Speaker note:** "This is drift we detected in real historical data, not injected noise. Calibration decay is tracked as gap-to-market, so a hard season for everyone doesn't false-alarm as model decay."

---

## Slide 14 — Drift Simulation & Anomaly Verification (required stress test)

**Screenshot:** `reports/drift/stress_test.png` — the red "MONITOR ALERT — 3/3
injected faults caught" report. Same view in the dashboard's **Stress test** tab.

**On slide — baseline + three injected faults, scored through the deployed API:**
- **Baseline:** clean 2025-26 holdout (1,609 rows) -> log loss **0.9796**, within tolerance of the monitoring baseline
- **Each fault is caught by a *different* detector** (the whole point):

| Fault | Corruption | What caught it |
|---|---|---|
| out_of_bounds | every feature 5-15x past its range | **84/86** features drift + prediction PSI **12.5** |
| swapped_columns | home/away swapped, market price flipped | log loss **+0.405** (performance) — only 13/86 marginals move |
| schema_break | xG feature family dropped | **12 missing** features (schema) — log loss barely moves |

**Speaker note:** "Baseline validation first — the clean holdout passes. Then we
inject faults and send them to the *deployed model*. The lesson is the table's
right column: marginal input-drift monitoring alone would miss the column swap,
and performance monitoring alone would miss the schema break. You need all three
signals. This is the anomaly-verification step the rubric grades — and it
complements Slide 13, which is a *real* detected shift rather than an injected one."

---

## Slide 15 — Conclusions

**On slide:**
- **The negative result IS the deliverable** — and it's well-evidenced:
  - No model beats the market on any fold, league, or track — nine model families, three of them hyperparameter-tuned
  - Even a principled Dixon-Coles goals model (0.9926) and tuned CatBoost, the champion (0.9766), fall short of 0.9691
  - A round of theory-driven feature engineering (opponent strength, venue-split form, congestion, standing) **left the gap unchanged** — further evidence the signal isn't there
  - No value threshold yields a profit whose CI excludes zero; CLV is negative throughout
- **Interpretation:** the top-5 closing line is near-efficient — there is no exploitable pre-kickoff edge in public xG/Elo/form data
- **What we actually delivered:** a reproducible, leakage-proof, containerized, monitored MLOps pipeline that establishes a negative result *credibly* — the professional-standard verdict

**Speaker note:** "A pipeline that let us fool ourselves into a fake edge would be worse than useless. The engineering value is that we can trust the 'no.'"

---

## Slide 16 — Limitations & Future Work

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

## Slide 17 — Repository & Questions

**On slide:**
- **github.com/arenmizuno/beating-1X2** (public)
- Reproduce locally: `pip install -r requirements.txt` -> run stages -> `docker compose up`; cold run < 10 min
- **84 hermetic tests** (no network) · CI **mutation-tests the leakage guard** (flips `shift(1)`->`shift(0)`, fails if tests stay green)
- Questions?

**Speaker note:** "The leakage mutation test is the detail worth landing: a guard that never fails on a broken build is no guard at all."
