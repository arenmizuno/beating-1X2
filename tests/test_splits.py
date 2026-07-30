"""Temporal-split tests.

A split that leaks the future is indistinguishable from a good model, so these
assert the properties that make the evaluation trustworthy: folds never overlap,
they only ever move forward, and the holdout season is genuinely untouched.
"""

from __future__ import annotations

import pytest

from src.config import HOLDOUT_SEASON
from src.splits import calibration_split, holdout_split, walk_forward_splits


def test_train_and_eval_never_overlap():
    for split in walk_forward_splits():
        assert not set(split.train_seasons) & set(split.eval_seasons)


def test_training_is_always_strictly_earlier():
    """Every training season must precede every evaluation season."""
    for split in walk_forward_splits():
        assert max(split.train_seasons) < min(split.eval_seasons)


def test_training_window_expands():
    splits = walk_forward_splits()
    for earlier, later in zip(splits, splits[1:], strict=False):
        assert set(earlier.train_seasons) < set(later.train_seasons)


def test_holdout_season_never_appears_in_any_training_window():
    """The single most important property of the whole evaluation."""
    for split in walk_forward_splits():
        assert HOLDOUT_SEASON not in split.train_seasons
        assert HOLDOUT_SEASON not in split.eval_seasons

    final = holdout_split()
    assert HOLDOUT_SEASON not in final.train_seasons
    assert final.eval_seasons == (HOLDOUT_SEASON,)


def test_holdout_rejects_a_contaminated_development_set():
    with pytest.raises(ValueError, match="must not appear"):
        holdout_split(dev_seasons=[2018, 2019, HOLDOUT_SEASON])


def test_calibration_split_is_temporal_and_disjoint():
    """Calibration seasons must be the LATEST ones, and never also fit seasons.

    Calibrating on rows the model trained on would fit the calibrator to
    in-sample confidence, hiding the overconfidence it exists to correct.
    """
    for split in walk_forward_splits():
        fit, calib = calibration_split(split.train_seasons)
        assert not set(fit) & set(calib)
        assert max(fit) < min(calib)
        assert set(fit) | set(calib) == set(split.train_seasons)


def test_calibration_rejects_too_short_a_window():
    with pytest.raises(ValueError, match="too short"):
        calibration_split((2018,), calibration_seasons=1)


def test_split_masks_select_the_right_rows(matches):
    """With min_train_seasons=2 the first fold trains on the first two seasons."""
    split = walk_forward_splits(dev_seasons=[2018, 2019, 2020], min_train_seasons=2)[0]
    assert split.train_seasons == (2018, 2019)
    assert split.eval_seasons == (2020,)

    train, evaluation = split.apply(matches)
    assert set(train["season"]) == {2018, 2019}
    # The synthetic fixture has no 2020 season, so evaluation is legitimately
    # empty -- what matters is that no training row leaked into it.
    assert evaluation.empty
