# beating-1X2

## ADSP 32021 – MLOps
### Final Project – University of Chicago

**Instructor:** Shaddy Abado

**Term:** Summer 2026

---

## Team Members

- Aren Mizuno
- Nick Dhaliwal
- Arthur Acker
- Nick Mikhail

---

# Project Overview

beating-1X2 is an end-to-end MLOps pipeline that asks whether a machine-learning
model can beat football's 1X2 betting market — predicting match outcomes (home
win / draw / away win) from pre-kickoff signals and comparing them against the
market's own implied probabilities.

The system ingests, models, serves and monitors match data to:
- predict 1X2 outcomes from leakage-safe pre-kickoff features,
- devig bookmaker closing odds into a vig-free market baseline to beat,
- flag value bets where model probability exceeds the market by a threshold,
- serve the champion model through a containerized API and dashboard,
- and monitor feature, target and calibration drift in production.

The pipeline is built to establish its result *credibly* — walk-forward splits,
a sealed holdout scored only through the deployed API, a market baseline as a
leakage tripwire, and bootstrap confidence intervals on every economic claim —
so the finding can be trusted whichever way it lands.

**Scope:** top-5 European leagues (Premier League, La Liga, Bundesliga, Serie A,
Ligue 1), seasons 2018-19 through 2025-26. 14,285 matches ingested, 13,786 after
requiring sufficient match history for both sides.

---

## Headline result

**The model does not beat the market, and the value-bet rule does not make
money.** That is the finding, not a failure of the build — and the pipeline is
designed to establish it credibly rather than to hide it.

Multiclass log loss over the **four common walk-forward folds** — the
equal-comparison protocol used to select the champion, scored on identical
matches for model and market (lower is better):

| track | model | log loss | vs market |
|---|---|---|---|
| **market_aware** | **catboost_tuned (champion)** | **0.9749** | **+0.0074** |
| market_aware | xgboost_tuned | 0.9751 | +0.0076 |
| market_aware | stacking | 0.9779 | +0.0104 |
| market_aware | logistic | 0.9850 | +0.0175 |
| market_blind | catboost_tuned | 0.9863 | +0.0188 |
| — | **market (vig-free closing line)** | **0.9675** | — |
| — | Dixon-Coles goals baseline | 0.9908 | — |

Across the full walk-forward (each configuration on all its available folds) the
market baseline is 0.9693 and Dixon-Coles 0.9926; see `reports/summary.md` for
that per-configuration detail. On the sealed **2025-26 holdout**, scored once
through the deployed API, the champion posts 0.9853 against the market's 0.9784 —
a stable +0.0069 gap.

Three things worth drawing out:

- **The market wins everywhere**, on every fold, every league, and both tracks —
  across six model families and nine configurations per track (18 runs in total),
  three of them hyperparameter-tuned.
- **`market_aware` still loses.** Even when handed the closing-line probability
  as an input feature, the model degrades it, and a round of engineered features
  (opponent strength, venue-split form, congestion, standing) left the gap
  unchanged. The features carry nothing the market has not already priced.
- **Tuned boosting is the best of the field** — tuned CatBoost and XGBoost lead,
  the neural net and untuned LightGBM trail — yet every configuration still sits
  between the market and the Dixon-Coles baseline.

Economically, no configuration is profitable. The champion's flat-stakes ROI is
−1.6% over 2,429 walk-forward bets (95% bootstrap CI [−10.0%, +7.8%], 31.2%
strike rate), and no edge threshold anywhere in a 0.02–0.20 sweep produces an ROI
whose 95% confidence interval excludes zero. The best point estimate across every
model and threshold is +0.03% — the champion on a relative-edge rule, worth +0.4
units on 1,319 bets with a CI of [−14.9%, +15.5%], which is pure noise. Closing-line
value is negative throughout (the champion's mean CLV is −3.4%; only 23–46% of
selections beat the close), the professional-standard verdict that there is no
exploitable signal here.

---

## Repository structure

```
.
├── params.yaml                  # every tunable; a run = this file + the commit
├── dvc.yaml                     # the reproducible stage DAG
├── flows.py                     # Prefect orchestration
├── Dockerfile / docker-compose.yml
├── src/
│   ├── config.py                 # paths, constants, params loader
│   ├── fetch.py                  # cached, throttled HTTP for all sources
│   ├── ingest_footballdata.py    # stage 1a: results + closing/opening odds
│   ├── ingest_understat.py       # stage 1b: expected goals
│   ├── ingest_clubelo.py         # stage 1c: club Elo ratings
│   ├── harmonize.py              # stage 2: team-name matching + match join
│   ├── market.py                 # stage 3: odds -> Shin vig-free probabilities
│   ├── features.py               # stage 4: leakage-safe feature table
│   ├── splits.py                 # stage 5: walk-forward temporal splits
│   ├── metrics.py                # scoring rules + calibration diagnostics
│   ├── train.py                  # stage 6: both tracks, nine configs, registry
│   ├── evaluate.py               # stage 7: statistical + economic evaluation
│   ├── drift.py                  # stage 8: feature/target/calibration drift
│   ├── stress_test.py            # stage 8b: injected-fault drift stress test
│   ├── ingest_fixtures.py        # upcoming fixtures + live pre-match odds
│   ├── prediction_markets.py     # Kalshi/Polymarket enrichment (best-effort)
│   ├── predict.py                # shared scoring path (API + batch)
│   └── api.py                    # FastAPI service
├── dashboard/app.py             # Streamlit monitoring dashboard
├── tests/                       # 98 hermetic tests, no network
├── data/
│   ├── mappings/team_aliases_manual.csv   # committed, hand-verified
│   ├── raw/ interim/ processed/  # gitignored; rebuilt by the pipeline
├── reports/                      # committed outputs, incl. drift/
└── mlruns/                       # gitignored MLflow store + model registry
```

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

Activate it before running anything else:

```bash
source .venv/bin/activate
```

This matters more than the usual boilerplate. The DVC and Prefect entry points
further down invoke each stage as a bare `python -m src.<stage>`, so an
unactivated shell runs them under whatever interpreter is first on `PATH`. On a
machine with conda that is the base environment, which happens to have pandas
and pyarrow but not catboost -- ingestion through `features` succeeds, and the
pipeline then dies at `train` with `ModuleNotFoundError: No module named
'catboost'`.

The orchestration and monitoring layer (DVC, Prefect, Evidently, Streamlit) is
installed separately, so the served container does not carry a workflow engine:

```bash
pip install -r requirements-ops.txt
```

Then, in order (each stage is independently runnable and caches its downloads).
The explicit `./.venv/bin/python` prefix below is redundant once the environment
is active; it is spelled out so each command also works on its own:

```bash
./.venv/bin/python -m src.ingest_footballdata
```
```bash
./.venv/bin/python -m src.ingest_understat
```
```bash
./.venv/bin/python -m src.ingest_clubelo
```
```bash
./.venv/bin/python -m src.harmonize && ./.venv/bin/python -m src.market
```
```bash
./.venv/bin/python -m src.features && ./.venv/bin/python -m src.train && ./.venv/bin/python -m src.evaluate
```

Inspect experiment tracking with:

```bash
./.venv/bin/mlflow ui --backend-store-uri mlruns --port 5001
```

A full cold run takes under ten minutes, most of it polite rate-limiting on the
three data sources (ClubElo alone is ~155 requests, about 5.5 minutes). Once the
raw cache exists, everything from `harmonize` onward reruns in about 90 seconds.

---

## Design decisions worth defending

### The market data source changed from the proposal

The proposal named the Polymarket/Kalshi APIs as the source of market-implied
probabilities. **They cannot serve that role.** Neither venue has per-match
three-way pricing for top-5 European leagues going back to 2018 — Kalshi's sports
contracts only expanded meaningfully in 2024-25, and Polymarket's football
coverage skews to tournaments and marquee fixtures with no historical archive for
league play. Pairing a market price to all ~14,000 labelled matches from those
APIs is not possible.

football-data.co.uk, already in our source list for results, ships bookmaker
odds for exactly these leagues and seasons. It is now the market source. In this
dataset Pinnacle closing prices (`PSC`) cover 94% of matches directly and 100%
after fallbacks — Pinnacle being the sharpest widely-published line and the
standard academic benchmark.

Prediction markets are better positioned as a *live inference-time* input in a
later phase, which is a stronger story anyway: backtested against bookmaker
closing lines, served against live prediction-market prices.

### Injury and availability flags were dropped

The proposal listed them as features. No reliable free historical injury feed
exists for 2018-2026 across five leagues, and confirmed lineups only publish
about an hour before kickoff, so even a live source could not retro-fill the
training set. Rather than ship a feature we could not source, it is cut and
recorded here.

### Odds column schema drifts across seasons

football-data changed its aggregate-odds columns around 2019-20: the `Bb*`
(Betbrain) family disappeared and `Max*`/`Avg*` replaced it. Ingestion therefore
probes for every known odds prefix and records which were present per season,
rather than assuming a fixed layout. Pinnacle closing is present throughout,
which is why it is the primary.

### Team-name matching is an assignment problem, not a lookup

The three sources spell clubs differently: `Man United` / `Manchester United`,
`Ath Bilbao` / `Athletic Club`, `Espanol` / `Espanyol`, `SPAL 2013` / `Spal`.
Within one league-season all sources describe the *same* ~20 clubs, which makes
this a bipartite assignment, not a series of independent fuzzy lookups. Solving
it globally (Hungarian algorithm over a rapidfuzz score matrix) lets the hard
pairs resolve once the easy ones are claimed.

Two safeguards proved necessary in practice:

- **Forced pairings are discarded.** When a season's source set is larger than
  the canonical set — which happens when a relegated club's top-division Elo
  rating still overlaps the season window — the assignment is exhaustive and
  shoves surplus names onto whatever is left at near-zero scores. Observed:
  `Mallorca` matched itself at score 100 in 2019 and 2021 but was forced onto
  `Ath Madrid` at score 33 in 2020. Anything scoring below the review threshold
  is treated as a non-match.
- **Hand-verified overrides are a committed source input.**
  `data/mappings/team_aliases_manual.csv` is consulted first and never
  overwritten by the pipeline; the derived full table is written to
  `data/interim/`. Fuzzy similarity cannot resolve `Athletic Club` →
  `Ath Bilbao`, and that decision should be made once, by a person, and version
  controlled.

**The join-rate gate earned its keep.** The first harmonize run failed loudly at
a 90.0% xG join rate in La Liga — exactly 2 of 20 clubs unmapped. After curation
the rate is **99.99%** (14,284 / 14,285; the single gap is one Ligue 1 2025-26
fixture Understat does not carry).

### Ordered pairs are the join key; the date is an assertion

Every league here is a double round-robin, so within a league-season the ordered
pair (home, away) plays exactly once. Joining on the pair makes the join robust
and lets the date become a *check* rather than a join condition — which matters,
because Understat timestamps kickoffs in a different timezone and legitimately
disagrees with football-data by a day on late fixtures.

This surfaced three genuine reschedulings rather than three bugs, including
Udinese–Roma, abandoned on 2024-04-14 after Evan Ndicka collapsed and completed
on 2024-04-25. See `reports/date_disagreements.csv`.

### Leakage discipline

The single failure mode that would invalidate everything, so the handling is
deliberately paranoid:

- **Shift, then roll.** Every rolling statistic is computed on a series already
  lagged by one match within its team. Rolling first and lagging after produces
  an identically-named, subtly leaky column, so the lag is a separate visible
  step.
- **xG is post-match.** It enters only through lagged rolling means; the raw
  same-match columns are dropped from the feature table entirely so they cannot
  be selected by accident. Understat's own `forecast` field is discarded too —
  it is a model output derived from that match's xG.
- **Elo as of the day before kickoff.** ClubElo publishes ratings as
  `[from, to]` validity ranges, so a rating that already absorbed the result can
  never be picked up.
- **Odds excluded by prefix, not by name**, so a new odds column added upstream
  cannot silently leak into the `market_blind` track.
- **`leakage_check` runs as part of the feature stage**, not as an optional
  test. It re-derives sampled feature values from scratch using only strictly
  earlier rows and asserts they match — an independent reimplementation, not a
  re-run of the pipeline code.
- **No random K-fold anywhere.** All splits are forward in time, and the
  2025-26 season is held out entirely.

The strongest evidence that this worked: the market baseline scores 0.9693 log
loss, squarely inside the 0.95–1.02 band a sharp closing line should occupy. A
model that beat that by a wide margin would be evidence of leakage, not skill.

### Calibration: sigmoid, and it was measured

Isotonic regression was the initial choice. With only one season (~1,700 rows)
of calibration data it maps whole score regions to exactly 0.0 — in the worst
fold it assigned probability zero to the outcome that actually occurred in 61
matches, which alone produced more than half of that fold's log loss (2.15 vs a
typical ~1.01). Sigmoid (Platt) scaling beat it on **all ten** fold-model
combinations and cannot produce a degenerate 0 or 1.

A probability floor is also applied as an explicit output contract: a predicted
zero asserts an outcome is impossible, makes log loss undefined when it happens,
and implies an unbounded Kelly stake. The floor-hit rate is logged per fold; with
sigmoid it never binds (0.000%).

### Why the value rule fires on longshots

At the proposal's 0.05 absolute edge threshold the rule flags **74–89% of all
fixtures** at mean odds above 4.7. A fixed 0.05 gap means very different things
at different prices: against a market probability of 0.20 it is a 25% relative
overestimate, against 0.67 only 7.5%. Since the model is less sharp than the
market, it holds more probability mass on longshots nearly everywhere.

Adding a relative-edge filter on top does **not** fix this — it makes it
marginally worse (mean odds rose 5.28 → 5.67), because once a 0.05 absolute edge
is required a longshot at p=0.10 clears a 20% relative bar automatically, so the
relative test only bites favourites. Both variants are reported side by side.

The deeper point: no selection rule rescues a model less accurate than the
market. That is why the threshold sweep is in the deliverable.

### Statistical honesty in the backtest

Confidence intervals are not decoration here. A thousand-odd flagged bets at
mean odds near 6 produce enormously noisy ROI — the sweep's best point estimate
is +0.03%, with a 95% interval of [−14.9%, +15.5%] on 1,319 bets and negative
results at neighbouring thresholds. Reporting that number without its interval
would be noise-mining. Every economic figure carries a bootstrap CI.

---

## Known limitations

- **2020-21 is a genuine distribution shift.** Home-win rate fell to 39.8% from
  44–45% either side — the COVID empty-stadium season. It is the worst fold for
  every configuration, and no amount of tuning fixes that honestly.
- **CLV is not a clean bet-at-open simulation.** Value is flagged against the
  closing-derived market probability, then CLV asks whether the opening price on
  that selection was better. It answers "does the model pick sides the market
  later moves toward", which is real evidence, but the flag itself used closing
  information.
- **Rolling windows span seasons**, so a promoted club carries no top-flight
  history and is dropped by the `min_prior_matches` filter until it has one.
- **Understat is scraped**, not served through a documented API, and it has
  already changed shape once during this project (the fixture list moved from an
  inlined `datesData` blob to an AJAX endpoint). It could change again.

## Serving and operations

### The model is swappable by design

Nothing downstream names an algorithm. `src/train.py` registers the best
configuration — currently `market_aware / catboost_tuned`, registry version 5,
`model_semver` 1.0.1, selected by **walk-forward** mean log loss without scoring
the holdout — to the MLflow Model Registry as `beating-1x2` with a `@champion`
alias. The sealed test season is first scored later through the deployed API. A
`model_contract.json` artifact travels with the model carrying its feature list
and exact design-matrix column order.

The API, the batch scorer, and the dashboard all resolve
`models:/beating-1x2@champion`. Replacing the model is a re-registration plus a
restart; no serving code changes. `reports/champion.json` records which
configuration won and by how much, so promotion is auditable.

### Training and serving share one feature code path

The obvious way to serve this model is to recompute rolling form and Elo for
live fixtures — and that is the classic route to training/serving skew, which
here would silently void the leakage guarantees the project rests on.

Instead, `build_feature_frame(matches, elo, upcoming=...)` **appends** unplayed
fixtures to the completed-match history, runs the unchanged pipeline, and
extracts the appended rows. This is safe precisely because the feature logic is
shift-then-roll: it only ever looks backward. Serving and training are one
implementation, not two kept in sync by discipline. A test asserts that features
for a held-out fixture are identical whether it is scored as "upcoming" or
processed as history.

### Live prices are not closing prices

The backtest benchmarks against Pinnacle *closing* odds — the sharpest and
hardest line to beat. An unplayed fixture has no close, so live scoring falls
back to the current price, which is systematically softer. **Live value flags
will therefore fire more readily than the backtest implies.** Every prediction
carries the `odds_source` actually used and the API states the benchmark in its
response.

### Prediction markets are in, and coverage is real

Phase 1 concluded prediction markets could not supply history. That remains
true. But checking their *live* coverage before writing the integration turned
up something better than expected: **Kalshi carries full-time 1X2 match series
for all five of our leagues** — `KXEPLGAME`, `KXLALIGAGAME`, `KXBUNDESLIGAGAME`,
`KXSERIEAGAME`, `KXLIGUE1GAME`. Polymarket's soccer coverage skews to season-long
and novelty markets, so it is a secondary source.

The integration is best-effort throughout: short timeouts, every failure
downgraded to "no price found", and a measured hit rate written to
`reports/prediction_market_coverage.json` rather than an assumed one.

### Running the service

```bash
docker compose up --build
```

API docs at `http://localhost:8000/docs`, dashboard at `http://localhost:8501`,
MLflow at `http://localhost:5001`. The MLflow service publishes container port
5000 as host 5001 because macOS Control Center (AirPlay Receiver) holds 5000 and
silently shadows the UI.

Or run them directly:

```bash
uvicorn src.api:app --reload
```
```bash
streamlit run dashboard/app.py
```

| endpoint | purpose |
|---|---|
| `GET /health` | liveness plus the loaded model version |
| `GET /model` | champion metadata and feature contract |
| `POST /predict` | score explicit feature payloads |
| `POST /predict/matches` | re-score historical matches by id |
| `GET /predict/upcoming` | fetch fixtures, score, flag value |
| `GET /metrics` | request counters |

Between late May and mid-August `/predict/upcoming` returns an empty list. That
is correct: the top-5 leagues are not playing. It is reported as
`n_fixtures: 0` with an explanatory note, not as an error.

### Orchestration: what DVC does and what Prefect does

A fair criticism of the original proposal was that it named two things that look
like orchestrators. They do different jobs:

- **DVC** (`dvc.yaml`) is the build system. It declares each stage's inputs and
  outputs so `dvc repro` rebuilds only what changed — editing a model
  hyperparameter reruns training and evaluation, but not the hour of ingestion
  above it. Run `dvc dag` to see the graph.
- **Prefect** (`flows.py`) is the scheduler and runtime. It handles retries
  against flaky third-party sources, concurrency, and run observability. The
  three ingestion tasks run concurrently with retries; everything downstream is
  sequential.

Both shell out to `python -m src.<stage>`, so activate the virtualenv first --
see [Running it](#running-it) for what happens when you do not.

```bash
dvc repro
```

The six committed hyperparameter-search tables are audited training inputs and
are reused by default (`train.reuse_search_results: true`). Set that parameter
to `false` when intentionally rerunning the full Optuna search; the selected
parameters are still evaluated solely on walk-forward folds.

```bash
python flows.py            # full pipeline
```
```bash
python flows.py --scoring  # just refresh and score fixtures
```

Generated artifacts are pushed to the DVC remote declared in `.dvc/config`, so a
fresh clone can recover the exact data a run produced instead of rebuilding it:

```bash
dvc push
```
```bash
dvc pull
```

Note that `train` is not idempotent by design. It refuses to register a second
registry version under a `model_semver` that is already claimed, so re-running a
completed pipeline unchanged fails at that stage -- deliberately, since an
identical model is not a new release. Bump `model_semver` in `params.yaml` when
you intend one.

### Drift monitoring

`src/drift.py` tracks three things, and computes the statistics directly so the
dashboard has numbers regardless of Evidently's version; Evidently HTML reports
are generated additionally.

The reference window is **2018-19 and 2019-20 only** — deliberately stopping
before 2020-21. A reference window has to predate the shift you want to detect,
and folding the COVID season into the baseline would make it undetectable by
construction. With the window set correctly, 2020-21 flags as target drift at
p < 0.0001: home-win rate fell to 39.8% from 44–45% either side, matches having
been played in empty stadiums. It is also the worst fold for every model
configuration. That is a real detected shift, not injected synthetic noise.

Feature drift is judged across **54 monitored features** at KS α = 0.01 and
PSI > 0.20 (a feature must trip both). The seasons after the reference flag 2–6
features each — 2 in 2021-22, 6 in 2024-25 — consistent with roster and schedule
churn rather than a structural break.

Calibration decay is tracked as the **gap to the market**, not raw log loss: a
season where everyone scored worse was a hard season; a season where only we
scored worse is model decay. The worst gap is +0.095 (market_blind / logistic,
2020-21), the same COVID season that trips the target-drift alarm.

### Drift stress test

`src/drift.py` detects a *real* historical shift. `src/stress_test.py` does the
complementary thing: it validates a serving baseline, then injects **known
faults** and confirms the monitor catches each. It first scores the clean
2025-26 holdout (1,727 rows) through the deployed API (`API_URL`, defaulting to
localhost), posting 0.9853 log loss against the market's 0.9784 (a +0.0069 gap),
and fails closed if the service is unavailable. Only after that baseline passes
does it re-score three corrupted copies. An explicit `--in-process` mode exists
only for tests and local diagnostics:

| Fault | Corruption | Caught by |
|---|---|---|
| `out_of_bounds` | every feature pushed 5–15× past its range | 84/86 features drift (KS+PSI), prediction PSI 10.71 |
| `swapped_columns` | home/away columns swapped, market price flipped | log loss +0.395, prediction PSI 0.36 — **performance**, not input drift (only 13/86 marginals move) |
| `schema_break` | the xG feature family dropped | 12 missing features — **schema**, though log loss barely moves (+0.001) |

Each fault trips a *different* detector, which is the point: marginal input-drift
monitoring alone would miss the column swap, and performance monitoring alone
would miss the schema break. Outputs land in `reports/drift/stress_test.{json,csv}`
plus a self-contained `stress_test.html` report (and `stress_test.png`), rendered
interactively in the dashboard's **Stress test** tab.

```bash
python -m src.stress_test --api-url http://localhost:8000
```

### Tests and CI

98 hermetic tests, no network — they run on synthetic frames so an outage
at any data source can never redden the build.

The leakage tests are the centrepiece, and they are themselves verified by
mutation: CI changes `shift(1)` to `shift(0)` in the feature code and **fails the
build if the tests stay green**. A guard that never fires on a broken build is
worse than no guard.

```bash
pytest
```
```bash
ruff check .
```

## Still not done

Hosted deployment (the container runs locally by design), and a scheduled
Prefect deployment against a Prefect server rather than ad-hoc flow runs.

MLflow's registry still assigns integer versions, while the release identifier
`model_semver` is validated from `params.yaml`, stored in the model contract and
registry tags, and returned by every model-bearing API response.

## Responsible use

Coursework for UChicago ADSP 32021, for academic purposes only. Nothing here is
betting advice, and the results specifically indicate no exploitable edge. Data
is used under each source's terms: football-data.co.uk and ClubElo publish for
free public use; Understat is accessed at a deliberately throttled rate with
aggressive local caching.
