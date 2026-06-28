"""Shared, lazy window-summary expression builders for the compiled engine.

This module is the single source of truth for *how a window is summarized* -- both the
legacy interpreter (:mod:`aces.aggregate`) and the compiled engine (:mod:`aces.compile`)
should agree with what is built here. Keeping the semantics in one place is what prevents
the two engines from drifting apart.

Unlike :mod:`aces.aggregate`, whose functions take and return eager ``pl.DataFrame``
objects, the builders here operate on ``pl.LazyFrame`` inputs and return ``pl.LazyFrame``
outputs so the whole task can be fused into one query plan and executed by the Polars
streaming engine.

.. note::
    Bodies are implemented in phase 2 (temporal) and phase 3 (event-bound). See
    ``docs/source/compiled_engine.md``.
"""

from __future__ import annotations

import polars as pl

from .types import TemporalWindowBounds, ToEventWindowBounds


def summarize_temporal_window(lf: pl.LazyFrame, endpoint_expr: TemporalWindowBounds) -> pl.LazyFrame:
    """Lazy analogue of :func:`aces.aggregate.aggregate_temporal_window`.

    Returns one row per input row, keyed by ``(subject_id, timestamp)``, with
    ``timestamp_at_start`` / ``timestamp_at_end`` and the per-window predicate sums.
    """
    raise NotImplementedError("Implemented in phase 2 (temporal compiler).")


def summarize_event_bound_window(lf: pl.LazyFrame, endpoint_expr: ToEventWindowBounds) -> pl.LazyFrame:
    """Lazy analogue of :func:`aces.aggregate.aggregate_event_bound_window`.

    Returns one row per input row, keyed by ``(subject_id, timestamp)``, with
    ``timestamp_at_start`` / ``timestamp_at_end`` and the per-window predicate sums.
    """
    raise NotImplementedError("Implemented in phase 3 (event-bound compiler).")
