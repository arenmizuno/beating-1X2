"""Stage 5: temporal splits.

Random K-fold is the wrong tool for this problem and would flatter every result.
Matches are ordered in time and teams persist across them, so a random fold puts
May's fixtures in the training set and August's in validation -- the model then
"predicts" a season it has already seen the end of. Every split here is
therefore strictly forward in time.

Two kinds of split:

  walk_forward_splits   Expanding window over the development seasons. Train on
                        every season before S, validate on S. This is what
                        model selection and hyperparameter choices use.

  holdout_split         Train on all development seasons, test on the holdout
                        season. Touched exactly once, at the very end.

Within a training window, `calibration_split` further carves off the most recent
season(s) to fit probability calibration on. That inner split is temporal too --
calibrating on randomly sampled rows from the same seasons the model trained on
would produce a calibrator fitted to in-sample confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import DEV_SEASONS, HOLDOUT_SEASON, PARAMS, get_logger

log = get_logger("splits")

MIN_TRAIN_SEASONS: int = PARAMS["split"]["min_train_seasons"]
CALIBRATION_SEASONS: int = PARAMS["train"]["calibration_seasons"]
STACKING_META_SEASONS: int = PARAMS["train"]["stacking_meta_seasons"]


@dataclass(frozen=True)
class Split:
    """One temporal fold, described by seasons rather than row indices.

    Seasons are the natural unit here: they make the split human-checkable in
    logs and reports, and they keep a fold from ever bisecting a season.
    """

    name: str
    train_seasons: tuple[int, ...]
    eval_seasons: tuple[int, ...]

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return df["season"].isin(self.train_seasons)

    def eval_mask(self, df: pd.DataFrame) -> pd.Series:
        return df["season"].isin(self.eval_seasons)

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return df[self.train_mask(df)].copy(), df[self.eval_mask(df)].copy()

    def __str__(self) -> str:
        train = f"{min(self.train_seasons)}-{max(self.train_seasons)}"
        evals = ",".join(str(s) for s in self.eval_seasons)
        return f"{self.name}: train[{train}] -> eval[{evals}]"


def walk_forward_splits(
    dev_seasons: list[int] | None = None, min_train_seasons: int = MIN_TRAIN_SEASONS
) -> list[Split]:
    """Expanding-window folds over the development seasons."""
    seasons = sorted(dev_seasons if dev_seasons is not None else DEV_SEASONS)

    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"need more than {min_train_seasons} development seasons to build a "
            f"walk-forward split; got {len(seasons)}"
        )

    splits = []
    for i in range(min_train_seasons, len(seasons)):
        eval_season = seasons[i]
        splits.append(
            Split(
                name=f"wf_{eval_season}",
                train_seasons=tuple(seasons[:i]),
                eval_seasons=(eval_season,),
            )
        )
    return splits


def holdout_split(
    dev_seasons: list[int] | None = None, holdout_season: int = HOLDOUT_SEASON
) -> Split:
    """Final evaluation: everything before the holdout, tested on the holdout."""
    seasons = sorted(dev_seasons if dev_seasons is not None else DEV_SEASONS)
    if holdout_season in seasons:
        raise ValueError(
            f"holdout season {holdout_season} must not appear in the development seasons"
        )
    return Split(
        name=f"holdout_{holdout_season}",
        train_seasons=tuple(seasons),
        eval_seasons=(holdout_season,),
    )


def calibration_split(
    train_seasons: tuple[int, ...], calibration_seasons: int = CALIBRATION_SEASONS
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a training window into (fit seasons, calibration seasons).

    The calibrator is fitted on the most recent season(s) of the window, which
    the model itself never sees. Returns (fit, calibration).
    """
    seasons = sorted(train_seasons)
    if len(seasons) <= calibration_seasons:
        raise ValueError(
            f"training window {seasons} is too short to reserve "
            f"{calibration_seasons} season(s) for calibration"
        )
    cut = len(seasons) - calibration_seasons
    return tuple(seasons[:cut]), tuple(seasons[cut:])


def stacking_split(
    train_seasons: tuple[int, ...],
    calibration_seasons: int = CALIBRATION_SEASONS,
    meta_seasons: int = STACKING_META_SEASONS,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split a training window into (fit, calibration, meta) seasons.

    The stacking meta-learner needs a slice the base models' calibrators never
    touched -- training it on `calibration_seasons` (the same rows
    `calibration_split` already reserves for calibrating the base models)
    would double-dip the exact in-sample-confidence problem calibration_split's
    own docstring warns about one level up. This carves TWO trailing slices
    instead of one: the middle slice still calibrates the base models exactly
    as before, and the most recent slice -- which neither the base models nor
    their calibrators ever see -- trains the meta-learner.

    Returns (fit, calibration, meta), sorted and disjoint. Raises ValueError if
    the window is too short, exactly like `calibration_split`.
    """
    seasons = sorted(train_seasons)
    reserved = calibration_seasons + meta_seasons
    if len(seasons) <= reserved:
        raise ValueError(
            f"training window {seasons} is too short to reserve "
            f"{calibration_seasons} calibration + {meta_seasons} meta season(s)"
        )
    calib_cut = len(seasons) - reserved
    meta_cut = len(seasons) - meta_seasons
    return (
        tuple(seasons[:calib_cut]),
        tuple(seasons[calib_cut:meta_cut]),
        tuple(seasons[meta_cut:]),
    )


def describe(splits: list[Split]) -> str:
    return "\n".join(f"  {s}" for s in splits)


if __name__ == "__main__":
    folds = walk_forward_splits()
    log.info("development seasons: %s", DEV_SEASONS)
    log.info("walk-forward folds:\n%s", describe(folds))
    log.info("final: %s", holdout_split())
    for fold in folds:
        fit, calib = calibration_split(fold.train_seasons)
        log.info("  %s -> fit%s calibrate%s", fold.name, list(fit), list(calib))
