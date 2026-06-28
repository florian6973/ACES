"""Contains utilities for validating that windows satisfy a set of constraints."""

import logging

import polars as pl

from .types import ANY_EVENT_COLUMN

logger = logging.getLogger(__name__)


def check_constraints(
    window_constraints: dict[str, tuple[int | None, int | None]], summary_df: pl.DataFrame
) -> pl.DataFrame:
    """Checks the constraints on the counts of predicates in the summary dataframe.

    Args:
        window_constraints: constraints on counts of predicates that must
            be satisfied, organized as a dictionary from predicate column name to the lowerbound and upper
            bound range required for that constraint to be satisfied.
        summary_df: A dataframe containing a row for every possible prospective window to be analyzed. The
            only columns expected are predicate columns within the ``window_constraints`` dictionary.

    Returns: A filtered dataframe containing only the rows that satisfy the constraints.

    Raises:
        ValueError: If the constraint for a column is empty.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=2, hour=5,  minute=17),
        ...         datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=6, hour=11, minute=0),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=3, hour=15, minute=17),
        ...     ],
        ...     "is_A": [1, 4, 1, 3, 3,  3],
        ...     "is_B": [0, 2, 0, 2, 10, 2],
        ...     "is_C": [1, 1, 1, 0, 1,  1],
        ... })
        >>> check_constraints({"is_A": (None, None), "is_B": (2, 6), "is_C": (1, 1)}, df)
        Traceback (most recent call last):
            ...
        ValueError: Invalid constraint for 'is_A': None - None
        >>> check_constraints({"is_A": (2, 1), "is_B": (2, 6), "is_C": (1, 1)}, df)
        Traceback (most recent call last):
            ...
        ValueError: Invalid constraint for 'is_A': 2 - 1
        >>> check_constraints({"is_A": (3, 4), "is_B": (2, 6), "is_C": (1, 1)}, df)
        shape: (2, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> check_constraints({"is_A": (3, 4), "is_B": (2, None), "is_C": (None, 1)}, df)
        shape: (4, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 3    ┆ 2    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 3    ┆ 10   ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 3],
        ...     "timestamp": [datetime(1980, 12, 28), datetime(2010, 6, 20), datetime(2010, 5, 11)],
        ...     "A": [False, False, False],
        ...     "_ANY_EVENT": [True, True, True],
        ... })
        >>> check_constraints({"_ANY_EVENT": (1, None)}, predicates_df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬───────┬────────────┐
        │ subject_id ┆ timestamp           ┆ A     ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ ---   ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ bool  ┆ bool       │
        ╞════════════╪═════════════════════╪═══════╪════════════╡
        │ 1          ┆ 1980-12-28 00:00:00 ┆ false ┆ true       │
        │ 1          ┆ 2010-06-20 00:00:00 ┆ false ┆ true       │
        │ 3          ┆ 2010-05-11 00:00:00 ┆ false ┆ true       │
        └────────────┴─────────────────────┴───────┴────────────┘
    """

    should_drop = pl.lit(False)

    for col, (valid_min_inc, valid_max_inc) in window_constraints.items():
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

        logger.info(
            f"Excluding {summary_df.select(drop_expr.sum()).item():,} rows "
            f"as they failed to satisfy '{valid_min_inc} <= {col} <= {valid_max_inc}'."
        )

        should_drop = should_drop | drop_expr

    return summary_df.filter(~should_drop)


def _block_satisfied_expr(window_constraints: dict[str, tuple[int | None, int | None]]) -> pl.Expr:
    """Return a boolean expression that is ``True`` for rows satisfying every constraint in the block.

    Each constraint maps a predicate column to an inclusive ``(min, max)`` count range, with ``None`` leaving
    an endpoint unconstrained. This is the conjunction (an AND of leaf constraints) shared by a single ``has``
    block and each block of a ``has_any`` disjunction.

    Raises:
        ValueError: If a constraint is fully unconstrained (both endpoints ``None``) or has ``max < min``.
    """
    satisfied = pl.lit(True)
    for col, (valid_min_inc, valid_max_inc) in window_constraints.items():
        if (valid_min_inc is None and valid_max_inc is None) or (
            valid_min_inc is not None and valid_max_inc is not None and valid_max_inc < valid_min_inc
        ):
            raise ValueError(f"Invalid constraint for '{col}': {valid_min_inc} - {valid_max_inc}")

        if col == "*":
            col = ANY_EVENT_COLUMN

        if valid_min_inc is not None:
            satisfied = satisfied & (pl.col(col) >= valid_min_inc)
        if valid_max_inc is not None:
            satisfied = satisfied & (pl.col(col) <= valid_max_inc)

    return satisfied


def check_any_constraints(
    constraint_blocks: list[dict[str, tuple[int | None, int | None]]], summary_df: pl.DataFrame
) -> pl.DataFrame:
    """Filters to rows satisfying at least one of the disjunctive constraint blocks (a ``has_any`` field).

    Each block is itself a conjunction of per-predicate ``(min, max)`` count constraints -- exactly the form
    consumed by :func:`check_constraints`. A row is retained iff *at least one* block is fully satisfied. This
    is the evaluation layer for a window's ``has_any`` field.

    Args:
        constraint_blocks: A non-empty list of constraint blocks. Each block is a dictionary from predicate
            column name to an inclusive ``(min, max)`` count range (with ``None`` leaving an endpoint open).
        summary_df: A dataframe with a row per prospective window and the referenced predicate count columns.

    Returns: A filtered dataframe containing only the rows that satisfy at least one block.

    Raises:
        ValueError: If ``constraint_blocks`` is empty, a block is empty, or a constraint is malformed.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2],
        ...     "timestamp": [
        ...         datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=2, hour=5,  minute=17),
        ...         datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=6, hour=11, minute=0),
        ...         datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=3, hour=15, minute=17),
        ...     ],
        ...     "is_A": [1, 4, 1, 3, 3,  3],
        ...     "is_B": [0, 2, 0, 2, 10, 2],
        ...     "is_C": [1, 1, 1, 0, 1,  1],
        ... })
        >>> check_any_constraints([{"is_A": (4, None)}, {"is_B": (10, None)}], df)
        shape: (2, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 3    ┆ 10   ┆ 1    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> check_any_constraints([], df)
        Traceback (most recent call last):
            ...
        ValueError: 'has_any' requires at least one constraint block.
        >>> check_any_constraints([{}], df)
        Traceback (most recent call last):
            ...
        ValueError: Each 'has_any' block must contain at least one constraint.
    """
    if not constraint_blocks:
        raise ValueError("'has_any' requires at least one constraint block.")

    block_exprs = []
    for block in constraint_blocks:
        if not block:
            raise ValueError("Each 'has_any' block must contain at least one constraint.")
        block_exprs.append(_block_satisfied_expr(block))

    satisfies_any = pl.any_horizontal(block_exprs)

    logger.info(
        f"Excluding {summary_df.select((~satisfies_any).sum()).item():,} rows "
        f"as they satisfied none of the {len(constraint_blocks)} 'has_any' constraint blocks."
    )

    return summary_df.filter(satisfies_any)


def check_static_variables(patient_demographics: list[str], predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Checks the constraints on the counts of predicates in the summary dataframe.

    Args:
        patient_demographics: List of columns representing static patient demographics.
        predicates_df: Dataframe containing a row for each event with patient demographics and timestamps.

    Returns: A filtered dataframe containing only the rows that satisfy the patient demographics.

    Raises:
        ValueError: If the static predicate used by constraint is not in the predicates dataframe.

    Examples:
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 1, 2, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         None,
        ...         datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=2, hour=5,  minute=17),
        ...         datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=6, hour=11, minute=0),
        ...         # Subject 2
        ...         None,
        ...         datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=3, hour=15, minute=17),
        ...     ],
        ...     "is_A": [0, 1, 4, 1, 0, 3, 3,  3],
        ...     "is_B": [0, 0, 2, 0, 0, 2, 10, 2],
        ...     "is_C": [0, 1, 1, 1, 0, 0, 1,  1],
        ...     "male": [1, 0, 0, 0, 0, 0, 0,  0]
        ... })

        >>> check_static_variables(['male'], predicates_df)
        shape: (4, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 0    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 1    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> check_static_variables(['female'], predicates_df)
        Traceback (most recent call last):
            ...
        ValueError: Static predicate 'female' not found in the predicates dataframe.
    """

    predicates_constraints = []

    for demographic in patient_demographics:
        if demographic not in predicates_df.columns:
            raise ValueError(f"Static predicate '{demographic}' not found in the predicates dataframe.")

        predicates_constraints.append(
            (pl.col("timestamp").is_null() & (pl.col(demographic) > 0)).any().over("subject_id")
        )

    predicate_filter = pl.all_horizontal(predicates_constraints)

    return predicates_df.filter(predicate_filter).drop_nulls(subset=["timestamp"]).drop(patient_demographics)
