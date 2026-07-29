# ADSP 32021 — Final Project: Soccer Value-Bet Detector

Multiclass prediction of association-football match outcomes (home win / draw /
away win) from pre-kickoff features, compared against the betting market's
implied probabilities to flag fixtures where the two disagree.

**Team:** Aren Mizuno · Nick Dhaliwal · Arthur Acker · Nick Mikhail

**Scope:** top-5 European leagues (Premier League, La Liga, Bundesliga, Serie A,
Ligue 1), seasons 2018-19 through 2025-26. 14,285 matches ingested, 13,786 after
requiring sufficient match history for both sides.

---

## Headline result

**The model does not beat the market, and the value-bet rule does not make
money.** That is the finding, not a failure of the build — and the pipeline is
designed to establish it credibly rather than to hide it.

Walk-forward mean multiclass log loss (lower is better), scored on identical
matches for model and market:

| track | model | log loss | vs market |
|---|---|---|---|
| market_blind | logistic | 1.0078 | +0.0385 |
| market_blind | lightgbm | 1.0191 | +0.0498 |
| **market_aware** | **logistic** | **0.9944** | **+0.0251** |
| market_aware | lightgbm | 1.0069 | +0.0376 |
| — | **market (vig-free closing line)** | **0.9693** | — |

Three things worth drawing out:

- **The market wins everywhere**, on every fold, every league, and both tracks.
- **`market_aware` still loses.** Even when handed the closing-line probability
  as an input feature, the model degrades it. The features do not carry
  information the market has not already priced.
- **Logistic beats LightGBM** in both tracks. On this kind of data a linear
  model over Elo and form differentials is genuinely competitive with boosting.

Economically, no configuration is profitable, and no edge threshold anywhere in
a 0.02–0.20 sweep produces an ROI whose 95% confidence interval excludes zero.
Closing-line value is negative throughout (mean −1% to −3%; only 31–47% of
selections beat the close), which is the professional-standard verdict that
there is no exploitable signal here.

See `reports/summary.md` for the generated detail.

---

## Repository structure

```
.
├── params.yaml                  # every tunable; a run = this file + the commit
├── requirements.txt
├── src/
│   ├── config.py                 # paths, constants, params loader
│   ├── fetch.py                  # cached, throttled HTTP for all sources
│   ├── ingest_footballdata.py    # stage 1a: results + closing/opening odds
│   ├── ingest_understat.py       # stage 1b: expected goals
│   ├── ingest_clubelo.py         # stage 1c: club Elo ratings
│   ├── harmonize.py              # stage 2: team-name matching + match join
│   ├── market.py                 # stage 3: odds -> vig-free probabilities
│   ├── features.py               # stage 4: leakage-safe feature table
│   ├── splits.py                 # stage 5: walk-forward temporal splits
│   ├── metrics.py                # scoring rules + calibration diagnostics
│   ├── train.py                  # stage 6: both tracks, both models, MLflow
│   └── evaluate.py               # stage 7: statistical + economic evaluation
├── data/
│   ├── mappings/team_aliases_manual.csv   # committed, hand-verified
│   ├── raw/                      # gitignored; rebuilt by the ingest stages
│   ├── interim/                  # gitignored
│   └── processed/                # gitignored
├── reports/                      # committed outputs
└── mlruns/                       # gitignored MLflow store
```

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

Then, in order (each stage is independently runnable and caches its downloads):

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
./.venv/bin/mlflow ui --backend-store-uri mlruns
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

Confidence intervals are not decoration here. A few hundred flagged bets at ~5.0
average odds produce enormously noisy ROI — the sweep's best point estimate is
+6.5%, with a 95% interval of [−17.8%, +34.2%] on 362 bets and negative results
at neighbouring thresholds. Reporting that number without its interval would be
noise-mining. Every economic figure carries a bootstrap CI.

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

## Deferred to phase 2

Prefect orchestration, DVC data versioning, FastAPI + Docker serving, an
Evidently-backed drift dashboard, GitHub Actions CI, and live Polymarket/Kalshi
pricing at inference. Note that the MLflow Model Registry versions integers, not
semver — semantic versions belong in tags alongside stage aliases.

## Responsible use

Coursework for UChicago ADSP 32021, for academic purposes only. Nothing here is
betting advice, and the results specifically indicate no exploitable edge. Data
is used under each source's terms: football-data.co.uk and ClubElo publish for
free public use; Understat is accessed at a deliberately throttled rate with
aggressive local caching.
