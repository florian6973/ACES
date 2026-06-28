"""Compiled-engine entry point: a drop-in alternative to :func:`aces.query.query`.

``lazy_query(cfg, predicates_df)`` mirrors the signature and output columns of
:func:`aces.query.query`, but executes via the compiled Polars plan from
:mod:`aces.compile` rather than the recursive interpreter. It is opt-in; the legacy
engine remains the default until parity is proven (see ``docs/source/compiled_engine.md``).
"""

from __future__ import annotations

import logging

import polars as pl

from .compile import compile_query
from .config import TaskExtractorConfig

logger = logging.getLogger(__name__)


def lazy_query(
    cfg: TaskExtractorConfig,
    predicates_df: pl.DataFrame,
    *,
    streaming: bool = True,
) -> pl.DataFrame:
    """Query a task by compiling ``cfg`` to a fused Polars plan and collecting it.

    Args:
        cfg: The parsed task configuration.
        predicates_df: The predicates dataframe (same schema as :func:`aces.query.query`).
        streaming: Collect with the Polars streaming engine (falls back to in-memory if
            an op is unsupported).

    Returns:
        A result dataframe equal to ``query(cfg, predicates_df)`` after a canonical sort.

    Raises:
        TypeError: If ``predicates_df`` is not a ``polars.DataFrame``.
        ValueError: If the ``(subject_id, timestamp)`` columns are not unique.
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got: {type(predicates_df)}.")

    is_unique = predicates_df.n_unique(subset=["subject_id", "timestamp"]) == predicates_df.shape[0]
    if not is_unique:
        raise ValueError("The (subject_id, timestamp) columns must be unique.")

    plan = compile_query(cfg)
    result_lf = plan(predicates_df.lazy())

    if not streaming:
        return result_lf.collect(engine="in-memory")
    try:
        return result_lf.collect(engine="streaming")
    except pl.exceptions.PolarsError:  # pragma: no cover - streaming fallback
        logger.warning("Streaming collect failed; falling back to in-memory collect.")
        return result_lf.collect(engine="in-memory")
