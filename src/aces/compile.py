"""Compile a :class:`~aces.config.TaskExtractorConfig` into a fused Polars query plan.

The legacy engine (:func:`aces.query.query` -> :func:`aces.extract_subtree.extract_subtree`)
interprets the window tree recursively, materializing an intermediate ``DataFrame`` at
every node. Because the window tree is statically known and each anchor realizes at most
one child anchor, that recursion can instead be *unrolled at compile time* into a single
``pl.LazyFrame`` graph and handed to the Polars streaming engine.

``_compile_subtree`` mirrors :func:`aces.extract_subtree.extract_subtree` step-for-step
(same joins, constraint filters, anchor remapping, and summary structs) but composes
``LazyFrame`` operations instead of eager ones; ``compile_query`` wraps it with the
trigger-anchor, static-variable, label, and index-timestamp logic from
:func:`aces.query.query`. Keeping the structure identical is what makes the compiled
output equal to the legacy output by construction.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import timedelta

import polars as pl
from bigtree import Node, preorder_iter

from ._window_exprs import summarize_event_bound_window, summarize_temporal_window
from .config import TaskExtractorConfig
from .types import ANY_EVENT_COLUMN, TemporalWindowBounds, ToEventWindowBounds

CompiledPlan = Callable[[pl.LazyFrame], pl.LazyFrame]


def _constraint_keep_expr(constraints: dict[str, tuple[int | None, int | None]]) -> pl.Expr:
    """Boolean keep-expression equivalent to :func:`aces.constraints.check_constraints`.

    Mirrors the eager validation (rejecting empty / inverted bounds) and the ``*`` ->
    ``_ANY_EVENT`` aliasing, but returns a filter expression instead of a filtered frame
    (no ``.item()`` logging side effect, so it stays lazy).
    """
    should_drop = pl.lit(False)
    for col, (valid_min_inc, valid_max_inc) in constraints.items():
        if (valid_min_inc is None and valid_max_inc is None) or (
            valid_min_inc is not None and valid_max_inc is not None and valid_max_inc < valid_min_inc
        ):
            raise ValueError(f"Invalid constraint for '{col}': {valid_min_inc} - {valid_max_inc}")

        if col == "*":
            col = ANY_EVENT_COLUMN

        drop_expr = pl.lit(False)
        if valid_min_inc is not None:
            drop_expr = drop_expr | (pl.col(col) < valid_min_inc)
        if valid_max_inc is not None:
            drop_expr = drop_expr | (pl.col(col) > valid_max_inc)
        should_drop = should_drop | drop_expr

    return ~should_drop


def _check_static_variables_lazy(demographics: list[str], lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy analogue of :func:`aces.constraints.check_static_variables`."""
    schema_names = lf.collect_schema().names()
    constraints = []
    for demographic in demographics:
        if demographic not in schema_names:
            raise ValueError(f"Static predicate '{demographic}' not found in the predicates dataframe.")
        constraints.append(
            (pl.col("timestamp").is_null() & (pl.col(demographic) > 0)).any().over("subject_id")
        )
    return lf.filter(pl.all_horizontal(constraints)).drop_nulls(subset=["timestamp"]).drop(demographics)


def _accumulate_offset(endpoint_expr, offset: timedelta):
    """Fold the accumulated subtree-root offset into ``endpoint_expr`` without mutation."""
    if type(endpoint_expr) is tuple:
        return (*endpoint_expr, offset)
    return dataclasses.replace(endpoint_expr, offset=endpoint_expr.offset + offset)


def _compile_subtree(
    subtree: Node,
    anchor_lf: pl.LazyFrame,
    predicates_lf: pl.LazyFrame,
    predicate_cols: list[str],
    offset: timedelta = timedelta(0),
) -> pl.LazyFrame:
    """Lazy, compile-time unrolling of :func:`aces.extract_subtree.extract_subtree`.

    ``anchor_lf`` is keyed by ``(subject_id, subtree_anchor_timestamp)``. Returns a
    ``LazyFrame`` keyed the same way, with one ``<window>_summary`` struct column per
    descendant window.
    """
    if not subtree.children:
        return anchor_lf

    recursive_results: list[pl.LazyFrame] = []
    for child in subtree.children:
        endpoint_expr = _accumulate_offset(child.endpoint_expr, offset)

        # Step 1: summarize the window from the subtree root to this child.
        if isinstance(endpoint_expr, TemporalWindowBounds):
            child_root_offset = offset + endpoint_expr.window_size
            window_summary = (
                summarize_temporal_window(predicates_lf, endpoint_expr)
                .with_columns(
                    pl.col("timestamp").alias("subtree_anchor_timestamp"),
                    pl.col("timestamp").alias("child_anchor_timestamp"),
                )
                .drop("timestamp")
            )
        elif isinstance(endpoint_expr, ToEventWindowBounds):
            # The child root is an extant event, so it is its own anchor (zero offset).
            child_root_offset = timedelta(0)
            child_anchor_time = (
                "timestamp_at_start" if endpoint_expr.end_event.startswith("-") else "timestamp_at_end"
            )
            window_summary = (
                summarize_event_bound_window(predicates_lf, endpoint_expr)
                .with_columns(
                    pl.col("timestamp").alias("subtree_anchor_timestamp"),
                    pl.col(child_anchor_time).alias("child_anchor_timestamp"),
                )
                .drop("timestamp")
            )
        else:  # pragma: no cover - guarded by config parsing
            raise ValueError(f"Invalid endpoint expression: '{endpoint_expr}'")

        # Step 2: restrict to valid subtree anchors.
        window_summary = window_summary.join(
            anchor_lf, on=["subject_id", "subtree_anchor_timestamp"], how="inner"
        )

        # Step 3: enforce this window's constraints.
        if child.constraints:
            window_summary = window_summary.filter(_constraint_keep_expr(child.constraints))

        # Step 4: the surviving child-anchor timestamps become the next subtree anchors.
        child_anchor_realizations = window_summary.select(
            "subject_id",
            pl.col("child_anchor_timestamp").alias("subtree_anchor_timestamp"),
        ).unique(maintain_order=True)

        # Step 5: recurse.
        recursive_result = _compile_subtree(
            child, child_anchor_realizations, predicates_lf, predicate_cols, child_root_offset
        )

        # Step 6.1: lift the recursive result back into this subtree's anchor space.
        recursive_result = (
            recursive_result.rename({"subtree_anchor_timestamp": "child_anchor_timestamp"})
            .join(
                window_summary.select(
                    "subject_id", "subtree_anchor_timestamp", "child_anchor_timestamp"
                ),
                on=["subject_id", "child_anchor_timestamp"],
                how="left",
            )
            .drop("child_anchor_timestamp")
        )

        # Step 6.2: attach this window's summary struct.
        for_return = window_summary.select(
            "subject_id",
            "subtree_anchor_timestamp",
            pl.struct(
                pl.lit(child.name).alias("window_name"),
                "timestamp_at_start",
                "timestamp_at_end",
                *predicate_cols,
            ).alias(f"{child.name}_summary"),
        )
        recursive_results.append(
            recursive_result.join(for_return, on=["subject_id", "subtree_anchor_timestamp"], how="left")
        )

    # Step 7: a valid realization requires every child branch to succeed.
    all_children = recursive_results[0]
    for df in recursive_results[1:]:
        all_children = all_children.join(df, on=["subject_id", "subtree_anchor_timestamp"], how="inner")
    return all_children


def compile_query(cfg: TaskExtractorConfig) -> CompiledPlan:
    """Compile ``cfg`` into a function mapping a predicates ``LazyFrame`` to a result plan.

    The returned plan, when applied to a predicates ``LazyFrame`` and collected, produces a
    frame equal to ``aces.query.query(cfg, predicates_df)`` (after a canonical sort).
    """

    def plan(predicates_lf: pl.LazyFrame) -> pl.LazyFrame:
        static_variables = [pred for pred in cfg.predicates if cfg.predicates[pred].static]
        if static_variables:
            predicates_lf = _check_static_variables_lazy(static_variables, predicates_lf)
        else:
            predicates_lf = predicates_lf.drop_nulls(subset=["subject_id", "timestamp"])

        predicate_cols = [
            c for c in predicates_lf.collect_schema().names() if c not in {"subject_id", "timestamp"}
        ]

        root_anchors = predicates_lf.filter(
            _constraint_keep_expr({cfg.trigger.predicate: (1, None)})
        ).select("subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp"))

        result = _compile_subtree(cfg.window_tree, root_anchors, predicates_lf, predicate_cols)
        result = result.rename({"subtree_anchor_timestamp": "trigger"})

        to_return_cols = [
            "subject_id",
            "trigger",
            *[f"{node.node_name}_summary" for node in preorder_iter(cfg.window_tree)][1:],
        ]

        if cfg.label_window:
            label_col = "end" if cfg.windows[cfg.label_window].root_node == "start" else "start"
            result = result.with_columns(
                pl.col(f"{cfg.label_window}.{label_col}_summary")
                .struct.field(cfg.windows[cfg.label_window].label)
                .alias("label")
            )
            to_return_cols.insert(1, "label")

        if cfg.index_timestamp_window:
            index_timestamp_col = (
                "end" if cfg.windows[cfg.index_timestamp_window].root_node == "start" else "start"
            )
            result = result.with_columns(
                pl.col(f"{cfg.index_timestamp_window}.{index_timestamp_col}_summary")
                .struct.field(f"timestamp_at_{cfg.windows[cfg.index_timestamp_window].index_timestamp}")
                .alias("index_timestamp")
            )
            to_return_cols.insert(1, "index_timestamp")

        return result.select(to_return_cols)

    return plan
