"""Prefect flows for beating-1X2.

    python flows.py                 # run the full pipeline once
    python flows.py --scoring       # run only the recurring scoring path

**On the overlap between Prefect and DVC**, which is a fair thing to ask about
when a project uses both:

    DVC     owns the reproducible data DAG. `dvc repro` rebuilds only the
            stages whose declared inputs actually changed, and versions the
            artifacts. It answers "is this result reproducible?"

    Prefect owns execution: scheduling, retries against flaky third-party
            sources, concurrency, and run observability. It answers "did
            tonight's run succeed, and if not, where?"

They are not two orchestrators competing. DVC is the build system; Prefect is
the scheduler that invokes it and watches it.

Retries are concentrated on the ingestion tasks, because that is where the
real-world failures are: three third-party sources, one of which (Understat) is
scraped rather than served through a documented API and has already changed
shape once during this project.
"""

from __future__ import annotations

import sys

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner


# ---------------------------------------------------------------------------
# Ingestion -- retried, because these are third-party network calls
# ---------------------------------------------------------------------------
@task(name="ingest-football-data", retries=3, retry_delay_seconds=30, log_prints=True)
def ingest_football_data() -> int:
    from src.ingest_footballdata import main

    return len(main())


@task(name="ingest-understat", retries=3, retry_delay_seconds=30, log_prints=True)
def ingest_understat() -> int:
    from src.ingest_understat import main

    return len(main())


@task(name="ingest-clubelo", retries=3, retry_delay_seconds=30, log_prints=True)
def ingest_clubelo() -> int:
    from src.ingest_clubelo import main

    return len(main())


# ---------------------------------------------------------------------------
# Transform and model -- deterministic, so a retry would just fail again
# ---------------------------------------------------------------------------
@task(name="harmonize", log_prints=True)
def harmonize() -> int:
    from src.harmonize import main

    return len(main())


@task(name="market", log_prints=True)
def market() -> int:
    from src.market import main

    return len(main())


@task(name="features", log_prints=True)
def features() -> int:
    from src.features import main

    return len(main())


@task(name="train", log_prints=True)
def train() -> int:
    from src.train import main

    return len(main())


@task(name="evaluate", log_prints=True)
def evaluate() -> None:
    from src.evaluate import main

    main()


@task(name="drift", log_prints=True)
def drift() -> dict:
    from src.drift import main

    return main()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@task(name="score-upcoming", retries=2, retry_delay_seconds=60, log_prints=True)
def score_upcoming() -> int:
    from src.predict import main

    return len(main())


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
@flow(
    name="beating-1x2-full-pipeline",
    task_runner=ThreadPoolTaskRunner(max_workers=3),
    log_prints=True,
)
def full_pipeline() -> None:
    """Ingest, transform, train, evaluate, monitor.

    The three ingestion tasks hit different hosts and share no state, so they
    are submitted concurrently. Everything after harmonize is strictly
    sequential -- each stage consumes the previous stage's output.
    """
    logger = get_run_logger()

    football_data = ingest_football_data.submit()
    understat = ingest_understat.submit()
    clubelo = ingest_clubelo.submit()

    logger.info(
        "ingested football-data=%d understat=%d clubelo=%d",
        football_data.result(),
        understat.result(),
        clubelo.result(),
    )

    n_matches = harmonize()
    n_priced = market()
    n_features = features()
    logger.info("harmonized=%d priced=%d features=%d", n_matches, n_priced, n_features)

    train()
    evaluate()
    summary = drift()
    logger.info("drift: target-drifted seasons %s", summary.get("target_drift_seasons"))


@flow(name="beating-1x2-scoring", log_prints=True)
def scoring_flow() -> None:
    """The recurring path: refresh fixtures and score them.

    Cheap enough to schedule daily. Between late May and mid-August it will
    legitimately find nothing to score -- the top-5 leagues are not playing.
    """
    logger = get_run_logger()
    n = score_upcoming()
    logger.info("scored %d upcoming fixtures", n)


if __name__ == "__main__":
    if "--scoring" in sys.argv:
        scoring_flow()
    else:
        full_pipeline()
