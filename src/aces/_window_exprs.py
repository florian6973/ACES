"""Shared, lazy window-summary expression builders for the compiled engine.

This module is the single source of truth for *how a window is summarized*. The builders
here are faithful, lazy ports of the eager functions in :mod:`aces.aggregate`
(:func:`aggregate_temporal_window` and :func:`boolean_expr_bound_sum`): they reproduce the
exact same sequence of Polars operations, but operate on ``pl.LazyFrame`` inputs and
return ``pl.LazyFrame`` outputs so the whole task can be fused into one streaming plan.

Because the operation sequences mirror the legacy code line-for-line, the compiled engine
is equal to the recursive interpreter *by construction*; ``tests/test_window_exprs.py``
pins that equivalence against :mod:`aces.aggregate` directly.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from .types import PRED_CNT_TYPE, TemporalWindowBounds, ToEventWindowBounds

_KEY_COLS = ("subject_id", "timestamp")


def _predicate_cols(lf: pl.LazyFrame) -> list[str]:
    return [c for c in lf.collect_schema().names() if c not in set(_KEY_COLS)]


def summarize_temporal_window(lf: pl.LazyFrame, endpoint_expr: TemporalWindowBounds) -> pl.LazyFrame:
    """Lazy analogue of :func:`aces.aggregate.aggregate_temporal_window` (multi-row path).

    Returns one row per input row keyed by ``(subject_id, timestamp)`` with
    ``timestamp_at_start`` / ``timestamp_at_end`` and the per-window predicate sums.

    The eager function has a special ``<= 1 total row`` singleton branch; that case only
    triggers when the *entire* frame has at most one row, which never occurs on the
    multi-subject predicate frames the compiled engine runs on, so it is omitted here.
    """
    if not isinstance(endpoint_expr, TemporalWindowBounds):
        endpoint_expr = TemporalWindowBounds(*endpoint_expr)

    cols = _predicate_cols(lf)
    return (
        lf.rolling(
            index_column="timestamp",
            group_by="subject_id",
            **endpoint_expr.polars_gp_rolling_kwargs,
        )
        .agg(*[pl.col(c).sum().cast(PRED_CNT_TYPE).alias(c) for c in cols])
        .sort(by=["subject_id", "timestamp"])
        .select(
            "subject_id",
            "timestamp",
            (pl.col("timestamp") + endpoint_expr.offset).alias("timestamp_at_start"),
            (pl.col("timestamp") + endpoint_expr.offset + endpoint_expr.window_size).alias(
                "timestamp_at_end"
            ),
            *cols,
        )
        .fill_null(0)
    )


def summarize_event_bound_window(
    lf: pl.LazyFrame,
    endpoint_expr: ToEventWindowBounds,
    anchors: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Lazy analogue of :func:`aces.aggregate.aggregate_event_bound_window`.

    If ``anchors`` (a frame with ``subject_id``/``timestamp``) is given, the output is
    restricted to those rows. This is *not* just a post-filter: it shrinks the expensive
    concat+sort inside the boundary-sum from O(all events) to O(anchors + boundary events),
    which is exact because non-anchor "real" rows never affect an anchor's nearest boundary
    (only boundary rows propagate via the fill). See ``_boolean_expr_bound_sum_lazy``.
    """
    if not isinstance(endpoint_expr, ToEventWindowBounds):
        endpoint_expr = ToEventWindowBounds(*endpoint_expr)
    return _boolean_expr_bound_sum_lazy(
        lf, **endpoint_expr.boolean_expr_bound_sum_kwargs, anchors=anchors
    )


def _boolean_expr_bound_sum_lazy(
    lf: pl.LazyFrame,
    boundary_expr: pl.Expr,
    mode: str,
    closed: str,
    offset: timedelta = timedelta(0),
    anchors: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Faithful lazy port of :func:`aces.aggregate.boolean_expr_bound_sum`.

    See that function's docstring for the full semantics of ``mode`` / ``closed`` /
    ``offset``. This port keeps the same expression construction step-for-step.
    """
    if mode not in ("bound_to_row", "row_to_bound"):
        raise ValueError(f"Mode '{mode}' invalid!")
    if closed not in ("both", "none", "left", "right"):
        raise ValueError(f"Closed '{closed}' invalid!")

    aggd_over_offset = None
    if offset != timedelta(0):
        if offset > timedelta(0):
            left_inclusive = False
            if mode == "row_to_bound":
                right_inclusive = closed not in ("left", "both")
            else:
                right_inclusive = closed in ("right", "both")
        else:
            right_inclusive = False
            if mode == "row_to_bound":
                left_inclusive = closed in ("left", "both")
            else:
                left_inclusive = closed not in ("right", "both")

        aggd_over_offset = summarize_temporal_window(
            lf,
            TemporalWindowBounds(
                left_inclusive=left_inclusive,
                window_size=offset,
                right_inclusive=right_inclusive,
                offset=None,
            ),
        )

    cols = _predicate_cols(lf)

    cumsum_cols = {c: pl.col(c).cum_sum().over("subject_id").alias(f"{c}_cumsum_at_row") for c in cols}
    lf = lf.with_columns(*cumsum_cols.values())

    cumsum_at_boundary = {c: pl.col(f"{c}_cumsum_at_row").alias(f"{c}_cumsum_at_boundary") for c in cols}

    if (mode == "bound_to_row" and closed in ("left", "both")) or (
        mode == "row_to_bound" and closed not in ("right", "both")
    ):
        cumsum_at_boundary = {
            c: (expr - pl.col(c)).alias(f"{c}_cumsum_at_boundary") for c, expr in cumsum_at_boundary.items()
        }

    timestamp_offset = pl.col("timestamp") - offset
    if mode == "bound_to_row":
        if closed in ("left", "both"):
            timestamp_offset -= timedelta(seconds=1e-6)
        else:
            timestamp_offset += timedelta(seconds=1e-6)

        fill_strategy = "forward"
        sum_exprs = {
            c: (
                pl.col(f"{c}_cumsum_at_row")
                - pl.col(f"{c}_cumsum_at_boundary").fill_null(strategy=fill_strategy).over("subject_id")
            ).alias(c)
            for c in cols
        }
        if (closed in ("left", "none") and offset <= timedelta(0)) or offset < timedelta(0):
            sum_exprs = {c: expr - pl.col(c) for c, expr in sum_exprs.items()}
    else:
        if closed in ("right", "both"):
            timestamp_offset += timedelta(seconds=1e-6)
        else:
            timestamp_offset -= timedelta(seconds=1e-6)

        fill_strategy = "backward"
        sum_exprs = {
            c: (
                pl.col(f"{c}_cumsum_at_boundary").fill_null(strategy=fill_strategy).over("subject_id")
                - pl.col(f"{c}_cumsum_at_row")
            ).alias(c)
            for c in cols
        }
        if (closed in ("left", "both") and offset <= timedelta(0)) or offset < timedelta(0):
            sum_exprs = {c: expr + pl.col(c) for c, expr in sum_exprs.items()}

    at_boundary_df = lf.filter(boundary_expr).select(
        "subject_id",
        pl.col("timestamp").alias("timestamp_at_boundary"),
        timestamp_offset.alias("timestamp"),
        *cumsum_at_boundary.values(),
        pl.lit(False).alias("is_real"),
    )

    # Only the anchor rows need to appear in the output, and dropping non-anchor real rows
    # does not change any anchor's nearest boundary (only boundary rows carry the fill), so
    # this restriction is exact -- it just shrinks the concat+sort below. The cumsum columns
    # were already computed over the full frame above, so per-row counts stay correct.
    real_rows = lf
    if anchors is not None:
        real_rows = lf.join(anchors, on=["subject_id", "timestamp"], how="semi")

    with_at_boundary_events = (
        pl.concat([real_rows.with_columns(pl.lit(True).alias("is_real")), at_boundary_df], how="diagonal")
        .sort(by=["subject_id", "timestamp"])
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp_at_boundary").fill_null(strategy=fill_strategy).over("subject_id"),
            *sum_exprs.values(),
            "is_real",
        )
        .filter("is_real")
        .drop("is_real")
    )

    if mode == "bound_to_row":
        st_timestamp_expr = pl.col("timestamp_at_boundary")
        end_timestamp_expr = pl.when(pl.col("timestamp_at_boundary").is_not_null()).then(
            pl.col("timestamp") + offset
        )
    else:
        st_timestamp_expr = pl.when(pl.col("timestamp_at_boundary").is_not_null()).then(
            pl.col("timestamp") + offset
        )
        end_timestamp_expr = pl.col("timestamp_at_boundary")

    if offset == timedelta(0):
        return with_at_boundary_events.select(
            "subject_id",
            "timestamp",
            st_timestamp_expr.alias("timestamp_at_start"),
            end_timestamp_expr.alias("timestamp_at_end"),
            *(pl.col(c).cast(PRED_CNT_TYPE).fill_null(0).alias(c) for c in cols),
        )

    if mode == "bound_to_row" and offset > timedelta(0):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) + pl.col(f"{c}_in_offset_period")

    elif (mode == "bound_to_row" and offset < timedelta(0)) or (
        mode == "row_to_bound" and offset > timedelta(0)
    ):

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) - pl.col(f"{c}_in_offset_period")

    else:  # mode == "row_to_bound" and offset < timedelta(0)

        def agg_offset_fn(c: str) -> pl.Expr:
            return pl.col(c) + pl.col(f"{c}_in_offset_period")

    return with_at_boundary_events.join(
        aggd_over_offset,
        on=["subject_id", "timestamp"],
        how="left",
        suffix="_in_offset_period",
    ).select(
        "subject_id",
        "timestamp",
        st_timestamp_expr.alias("timestamp_at_start"),
        end_timestamp_expr.alias("timestamp_at_end"),
        *(agg_offset_fn(c).cast(PRED_CNT_TYPE, strict=False).fill_null(0).alias(c) for c in cols),
    )
