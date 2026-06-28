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


def _window_summary_struct(name: str, predicate_cols: list[str]) -> pl.Expr:
    return pl.struct(
        pl.lit(name).alias("window_name"),
        "timestamp_at_start",
        "timestamp_at_end",
        *predicate_cols,
    ).alias(f"{name}_summary")


def _process_children(
    node: Node,
    cur: pl.LazyFrame,
    predicates_lf: pl.LazyFrame,
    predicate_cols: list[str],
    offset: timedelta = timedelta(0),
) -> pl.LazyFrame:
    """Compile every edge in ``node``'s subtree onto ``cur``, *flattening* where possible.

    ``cur`` is keyed by ``(subject_id, subtree_anchor_timestamp)`` in ``node``'s anchor
    space (the timestamps that realize ``node``), and already carries the ``<window>_summary``
    structs for edges processed so far. The return value extends it with a summary column
    for every descendant edge, still keyed in ``node``'s anchor space, with rows restricted
    to anchors for which the whole subtree has a valid realization.

    The key optimization over a faithful unrolling of
    :func:`aces.extract_subtree.extract_subtree`: a **temporal** child keeps the same anchor
    timestamp as its parent, so its window can be joined on *flat* and its own children
    processed in the same anchor space — no nested anchor remap. Only **event-bound** edges
    (which move the anchor to a realized boundary event) require the nested
    remap-back-to-parent logic. This turns the deep join cascade the naive compile produced
    on long temporal chains (super-linear) back into a linear, flat sequence of joins.
    """
    for child in node.children:
        endpoint_expr = _accumulate_offset(child.endpoint_expr, offset)

        if isinstance(endpoint_expr, TemporalWindowBounds):
            # Temporal edge: anchor unchanged -> flatten. Join the window summary directly
            # onto cur, filter constraints in place, attach the struct, and recurse into the
            # child's own children in this same anchor space (offset accumulates window_size).
            child_offset = offset + endpoint_expr.window_size
            summary = summarize_temporal_window(predicates_lf, endpoint_expr).rename(
                {"timestamp": "subtree_anchor_timestamp"}
            )
            cur = cur.join(summary, on=["subject_id", "subtree_anchor_timestamp"], how="inner")
            if child.constraints:
                cur = cur.filter(_constraint_keep_expr(child.constraints))
            cur = cur.with_columns(_window_summary_struct(child.name, predicate_cols)).drop(
                "timestamp_at_start", "timestamp_at_end", *predicate_cols
            )
            cur = _process_children(child, cur, predicates_lf, predicate_cols, child_offset)

        elif isinstance(endpoint_expr, ToEventWindowBounds):
            # Event-bound edge: anchor moves to a realized boundary event -> nest. Build the
            # child subtree in child-anchor space, then remap it back to this anchor space.
            child_anchor_time = (
                "timestamp_at_start" if endpoint_expr.end_event.startswith("-") else "timestamp_at_end"
            )
            # Restrict the (expensive) event-bound summary to the current anchor timestamps.
            anchor_ts = cur.select(
                "subject_id", pl.col("subtree_anchor_timestamp").alias("timestamp")
            ).unique(maintain_order=True)
            ws = (
                summarize_event_bound_window(predicates_lf, endpoint_expr, anchors=anchor_ts)
                .with_columns(
                    pl.col("timestamp").alias("subtree_anchor_timestamp"),
                    pl.col(child_anchor_time).alias("child_anchor_timestamp"),
                )
                .drop("timestamp")
            )
            if child.constraints:
                ws = ws.filter(_constraint_keep_expr(child.constraints))

            child_base = ws.select(
                "subject_id", pl.col("child_anchor_timestamp").alias("subtree_anchor_timestamp")
            ).unique(maintain_order=True)
            child_res = _process_children(child, child_base, predicates_lf, predicate_cols, timedelta(0))

            # Lift the child subtree back into this anchor space and attach the edge's summary.
            child_res = (
                child_res.rename({"subtree_anchor_timestamp": "child_anchor_timestamp"})
                .join(
                    ws.select("subject_id", "subtree_anchor_timestamp", "child_anchor_timestamp"),
                    on=["subject_id", "child_anchor_timestamp"],
                    how="left",
                )
                .drop("child_anchor_timestamp")
                .join(
                    ws.select(
                        "subject_id",
                        "subtree_anchor_timestamp",
                        _window_summary_struct(child.name, predicate_cols),
                    ),
                    on=["subject_id", "subtree_anchor_timestamp"],
                    how="left",
                )
            )
            cur = cur.join(child_res, on=["subject_id", "subtree_anchor_timestamp"], how="inner")

        else:  # pragma: no cover - guarded by config parsing
            raise ValueError(f"Invalid endpoint expression: '{endpoint_expr}'")

    return cur


def compile_query(cfg: TaskExtractorConfig, materialize_base: bool = True) -> CompiledPlan:
    """Compile ``cfg`` into a function mapping a predicates ``LazyFrame`` to a result plan.

    The returned plan, when applied to a predicates ``LazyFrame`` and collected, produces a
    frame equal to ``aces.query.query(cfg, predicates_df)`` (after a canonical sort).

    Args:
        cfg: The parsed task configuration.
        materialize_base: Collect the prepared predicates frame once before fanning out to the
            windows, so the shared base prep is not re-evaluated per window (see the barrier
            note in ``plan``). Almost always a win; exposed mainly so benchmarks can compare.
    """

    def plan(predicates_lf: pl.LazyFrame) -> pl.LazyFrame:
        static_variables = [pred for pred in cfg.predicates if cfg.predicates[pred].static]
        if static_variables:
            predicates_lf = _check_static_variables_lazy(static_variables, predicates_lf)
        else:
            predicates_lf = predicates_lf.drop_nulls(subset=["subject_id", "timestamp"])

        # Materialization barrier: every window summary and the trigger filter read from this
        # same prepared frame, but Polars' common-subplan elimination does not dedupe it across
        # the join tree, so without this it re-runs the (expensive) static-variable filter and
        # base scan once per window. Collecting once mirrors the per-node materialization the
        # legacy interpreter gets for free. (No-op cost when the input is already a small frame.)
        if materialize_base:
            predicates_lf = predicates_lf.collect(engine="in-memory").lazy()

        predicate_cols = [
            c for c in predicates_lf.collect_schema().names() if c not in {"subject_id", "timestamp"}
        ]

        root_anchors = predicates_lf.filter(
            _constraint_keep_expr({cfg.trigger.predicate: (1, None)})
        ).select("subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp"))

        result = _process_children(cfg.window_tree, root_anchors, predicates_lf, predicate_cols)
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
