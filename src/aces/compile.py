"""Compile a :class:`~aces.config.TaskExtractorConfig` into a fused Polars query plan.

The legacy engine (:func:`aces.query.query` -> :func:`aces.extract_subtree.extract_subtree`)
interprets the window tree recursively, materializing an intermediate ``DataFrame`` at
every node. Because the window tree is statically known and each anchor realizes at most
one child anchor, that recursion can instead be *unrolled at compile time* into a single
``pl.LazyFrame`` graph and handed to the Polars streaming engine.

``compile_query(cfg)`` returns a pure function ``pl.LazyFrame -> pl.LazyFrame``; binding
it to data and collecting happens in :func:`aces.lazy_query.lazy_query`.

.. note::
    The compiler body is implemented in phases 2-3 (see
    ``docs/source/compiled_engine.md``). This module currently defines the public surface.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .config import TaskExtractorConfig

CompiledPlan = Callable[[pl.LazyFrame], pl.LazyFrame]


def compile_query(cfg: TaskExtractorConfig) -> CompiledPlan:
    """Compile ``cfg`` into a function mapping a predicates ``LazyFrame`` to a result plan.

    The returned plan, when applied to a sorted predicates ``LazyFrame`` and collected,
    produces a frame equal to ``aces.query.query(cfg, predicates_df)`` (after a canonical
    sort).

    Args:
        cfg: The parsed task configuration.

    Returns:
        A pure ``pl.LazyFrame -> pl.LazyFrame`` plan builder.
    """
    raise NotImplementedError("Implemented in phases 2-3 (compiled engine).")
