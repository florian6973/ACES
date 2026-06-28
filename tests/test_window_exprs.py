"""Pin the lazy window-summary twins to the eager aggregators in aces.aggregate.

These are the highest-risk pieces of the compiled engine (especially the event-bound
``mode`` x ``closed`` x ``offset`` matrix), so we check them directly against the legacy
functions on both fixed and randomized inputs, independent of the full query pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from aces._window_exprs import summarize_event_bound_window, summarize_temporal_window
from aces.aggregate import aggregate_event_bound_window, aggregate_temporal_window
from aces.types import TemporalWindowBounds, ToEventWindowBounds

_FIXED = pl.DataFrame(
    {
        "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        "timestamp": [
            datetime(1989, 12, 1, 12, 3),
            datetime(1989, 12, 3, 13, 14),
            datetime(1989, 12, 5, 15, 17),
            datetime(1989, 12, 2, 12, 3),
            datetime(1989, 12, 4, 13, 14),
            datetime(1989, 12, 6, 15, 17),
            datetime(1989, 12, 8, 16, 22),
            datetime(1989, 12, 10, 3, 7),
        ],
        "is_A": [1, 0, 1, 1, 1, 1, 0, 0],
        "is_B": [0, 1, 0, 1, 0, 1, 1, 1],
        "is_C": [0, 1, 0, 0, 0, 1, 0, 1],
    }
)


def _random_df(seed: int, n_subjects: int = 6, events: int = 25) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_subjects * events
    sid = np.repeat(np.arange(1, n_subjects + 1), events)
    gaps = np.cumsum(rng.exponential(10, size=(n_subjects, events)) + 1e-3, axis=1).reshape(-1)
    ts = [datetime(2010, 1, 1) + timedelta(hours=float(h)) for h in gaps]
    data = {"subject_id": list(sid), "timestamp": ts}
    for c in ("is_A", "is_B", "is_C"):
        data[c] = list((rng.random(n) < 0.3).astype(np.int64))
    return pl.DataFrame(data).sort("subject_id", "timestamp")


def _assert_temporal(df, bounds):
    eager = aggregate_temporal_window(df, bounds)
    lazy = summarize_temporal_window(df.lazy(), bounds).collect()
    assert_frame_equal(eager, lazy, check_dtypes=True)


def _assert_event(df, bounds):
    eager = aggregate_event_bound_window(df, bounds)
    lazy = summarize_event_bound_window(df.lazy(), bounds).collect()
    assert_frame_equal(eager, lazy, check_dtypes=True)


TEMPORAL_BOUNDS = [
    TemporalWindowBounds(True, timedelta(days=7), True, None),
    TemporalWindowBounds(True, timedelta(days=1), False, timedelta(0)),
    TemporalWindowBounds(False, timedelta(days=2), False, timedelta(minutes=1)),
    TemporalWindowBounds(False, timedelta(days=-1), True, timedelta(days=1)),
    TemporalWindowBounds(True, timedelta(days=-1), False, timedelta(days=1)),
    TemporalWindowBounds(False, timedelta(hours=12), False, timedelta(hours=12)),
]


@pytest.mark.parametrize("bounds", TEMPORAL_BOUNDS, ids=lambda b: str(tuple(b)))
@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_temporal_twin_matches_legacy(bounds, seed):
    df = _FIXED if seed is None else _random_df(seed)
    _assert_temporal(df, bounds)


@pytest.mark.parametrize("end_event", ["is_C", "-is_A", "_RECORD_END", "-_RECORD_START"])
@pytest.mark.parametrize("left_inc", [True, False])
@pytest.mark.parametrize("right_inc", [True, False])
@pytest.mark.parametrize("offset", [None, timedelta(days=3), timedelta(days=-3)])
@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_event_bound_twin_matches_legacy(end_event, left_inc, right_inc, offset, seed):
    df = _FIXED if seed is None else _random_df(seed)
    _assert_event(df, ToEventWindowBounds(left_inc, end_event, right_inc, offset))
