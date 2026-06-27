"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""

import logging

import polars as pl
from bigtree import preorder_iter

from .config import TaskExtractorConfig
from .constraints import check_constraints, check_static_variables
from .extract_subtree import extract_subtree
from .types import PRED_CNT_TYPE
from .utils import log_tree

logger = logging.getLogger(__name__)

LABEL_JOIN_KEYS = ["subject_id", "trigger"]


def query(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: TaskExtractorConfig object of the configuration file.
        predicates_df: Polars predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query, containing subjects who satisfy the conditions
            defined in cfg. Timestamps for the start/end boundaries of each window specified in the task
            configuration, as well as predicate counts for each window, are provided.

    Raises:
        TypeError: If predicates_df is not a polars.DataFrame.
        ValueError: If the (subject_id, timestamp) columns are not unique.

    Examples:
        >>> from .config import PlainPredicateConfig, WindowConfig, EventConfig

        >>> cfg = None # This is obviously invalid, but we're just testing the error case.
        >>> predicates_df = {"subject_id": [1, 1], "timestamp": [1, 1]}
        >>> query(cfg, predicates_df)
        Traceback (most recent call last):
            ...
        TypeError: Predicates dataframe type must be a polars.DataFrame. Got: <class 'dict'>.
        >>> query(cfg, pl.DataFrame(predicates_df))
        Traceback (most recent call last):
            ...
        ValueError: The (subject_id, timestamp) columns must be unique.
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A")},
        ...     trigger=EventConfig("_ANY_EVENT"),
        ...     windows={
        ...         "pre": WindowConfig(None, "trigger", True, False, index_timestamp="start"),
        ...         "post": WindowConfig("pre.end", None, True, True, label="A"),
        ...     },
        ...     index_timestamp_window="pre",
        ...     label_window="post",
        ... )
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 3],
        ...     "timestamp": [datetime(1980, 12, 28), datetime(2010, 6, 20), datetime(2010, 5, 11)],
        ...     "A": [False, False, False],
        ...     "_ANY_EVENT": [True, True, True],
        ... })
        >>> with caplog.at_level(logging.INFO):
        ...     result = query(cfg, predicates_df)
        >>> result.select("subject_id", "trigger")
        shape: (3, 2)
        ┌────────────┬─────────────────────┐
        │ subject_id ┆ trigger             │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 1          ┆ 1980-12-28 00:00:00 │
        │ 1          ┆ 2010-06-20 00:00:00 │
        │ 3          ┆ 2010-05-11 00:00:00 │
        └────────────┴─────────────────────┘
        >>> "index_timestamp" in result.columns
        True
        >>> "label" in result.columns
        True
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A", static=True)},
        ...     trigger=EventConfig("_ANY_EVENT"),
        ...     windows={},
        ... )
        >>> with caplog.at_level(logging.INFO):
        ...     query(cfg, predicates_df)
        shape: (0, 0)
        ┌┐
        ╞╡
        └┘
        >>> "Static variable criteria specified, filtering patient demographics..." in caplog.text
        True
        >>> "No static variable criteria specified, removing all rows with null timestamps..." in caplog.text
        True
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 3],
        ...     "timestamp": [None, datetime(2010, 6, 20), datetime(2010, 5, 11)],
        ...     "A": [True, False, False],
        ...     "_ANY_EVENT": [False, False, False],
        ... })
        >>> with caplog.at_level(logging.INFO):
        ...     result = query(cfg, predicates_df)
        >>> "No valid rows found for the trigger event" in caplog.text
        True
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got: {type(predicates_df)}.")

    logger.info("Checking if '(subject_id, timestamp)' columns are unique...")

    is_unique = predicates_df.n_unique(subset=["subject_id", "timestamp"]) == predicates_df.shape[0]

    if not is_unique:
        raise ValueError("The (subject_id, timestamp) columns must be unique.")

    if cfg.windows_pos is not None:
        return _query_label_windows(cfg, predicates_df)

    return _extract_cohort(cfg, predicates_df)


def _extract_cohort(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Extract the cohort defined by ``cfg``'s base windows, returning one row per valid trigger.

    This is the single-pass ACES extraction: it realizes the window tree, attaches the ``label`` and
    ``index_timestamp`` columns when those fields are present, and is the building block the label-window
    multi-pass driver (:func:`_query_label_windows`) runs for the eligible, positive, and negative passes.
    """
    log_tree(cfg.window_tree)

    logger.info("Beginning query...")

    static_variables = [pred for pred in cfg.predicates if cfg.predicates[pred].static]
    if static_variables:
        logger.info("Static variable criteria specified, filtering patient demographics...")
        predicates_df = check_static_variables(static_variables, predicates_df)
    else:
        logger.info("No static variable criteria specified, removing all rows with null timestamps...")
        predicates_df = predicates_df.drop_nulls(subset=["subject_id", "timestamp"])

    if predicates_df.is_empty():
        logger.warning("No valid rows found after filtering patient demographics. Exiting.")
        return pl.DataFrame()

    logger.info("Identifying possible trigger nodes based on the specified trigger event...")
    prospective_root_anchors = check_constraints({cfg.trigger.predicate: (1, None)}, predicates_df).select(
        "subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp")
    )

    if prospective_root_anchors.is_empty():
        logger.warning(f"No valid rows found for the trigger event '{cfg.trigger.predicate}'. Exiting.")
        return pl.DataFrame()

    result = extract_subtree(cfg.window_tree, prospective_root_anchors, predicates_df)
    if result.is_empty():  # pragma: no cover
        logger.warning("No valid rows found.")
        return pl.DataFrame()
    else:
        # number of patients
        logger.info(
            f"Done. {result.shape[0]:,} valid rows returned corresponding to "
            f"{result['subject_id'].n_unique():,} subjects."
        )

    result = result.rename({"subtree_anchor_timestamp": "trigger"})

    to_return_cols = [
        "subject_id",
        "trigger",
        *[f"{node.node_name}_summary" for node in preorder_iter(cfg.window_tree)][1:],
    ]

    # add label column if specified
    if cfg.label_window:
        logger.info(  # pragma: no cover
            f"Extracting label '{cfg.windows[cfg.label_window].label}' from window '{cfg.label_window}'..."
        )
        label_col = "end" if cfg.windows[cfg.label_window].root_node == "start" else "start"
        result = result.with_columns(
            pl.col(f"{cfg.label_window}.{label_col}_summary")
            .struct.field(cfg.windows[cfg.label_window].label)
            .alias("label")
        )
        to_return_cols.insert(1, "label")

        if result["label"].n_unique() == 1:  # pragma: no cover
            logger.warning(
                f"All labels in the extracted cohort are the same: '{result['label'][0]}'. "
                "This may indicate an issue with the task logic. "
                "Please double-check your configuration file if this is not expected."
            )

    # add index_timestamp column if specified
    if cfg.index_timestamp_window:
        logger.info(  # pragma: no cover
            f"Setting index timestamp as '{cfg.windows[cfg.index_timestamp_window].index_timestamp}' "
            f"of window '{cfg.index_timestamp_window}'..."
        )
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


def _label_keys(cohort: pl.DataFrame, eligible: pl.DataFrame) -> pl.DataFrame:
    """Return the unique ``(subject_id, trigger)`` keys identifying the rows in a label-pass ``cohort``.

    ``cohort`` may be an empty :class:`polars.DataFrame` (no columns) when a pass yields no rows; in that case
    an empty key frame with the schema of ``eligible`` is returned so downstream joins remain well-typed.
    """
    if cohort.is_empty() or "subject_id" not in cohort.columns:
        return eligible.select(LABEL_JOIN_KEYS).clear()
    return cohort.select(LABEL_JOIN_KEYS).unique()


def _order_label_columns(result: pl.DataFrame) -> pl.DataFrame:
    """Order columns as ``subject_id, [index_timestamp], label, trigger, <window summaries...>``."""
    leading = ["subject_id"]
    if "index_timestamp" in result.columns:
        leading.append("index_timestamp")
    leading.append("label")
    ordered = leading + [c for c in result.columns if c not in leading]
    return result.select(ordered)


def _query_label_windows(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Assign binary labels via the ``windows_pos`` / ``windows_neg`` label-defining window groups.

    Implements the multi-pass reference design: the base ``windows`` define the eligible cohort ``E`` (one row
    per valid trigger); a second pass over base + ``windows_pos`` yields the positive subset ``P ⊆ E``; and,
    when ``windows_neg`` is given, a third pass over base + ``windows_neg`` yields the negative set ``N``. The
    passes share the same trigger/index, so they are joined on ``(subject_id, trigger)``.

    - ``windows_neg`` omitted: every eligible trigger is emitted; ``label = 1`` on ``P``, else ``0``.
    - ``windows_neg`` present: only ``P ∪ N`` are emitted (``label = 1`` on ``P``, ``0`` on ``N``); eligible
      triggers matching neither are dropped (the ambiguous middle). A trigger matching *both* groups is a
      config defect (``Wp``/``Wn`` are not mutually exclusive) and raises a ``ValueError``.

    Raises:
        ValueError: If any eligible trigger satisfies both ``windows_pos`` and ``windows_neg``.
    """
    logger.info("Labeling via 'windows_pos'%s...", " / 'windows_neg'" if cfg.windows_neg else "")

    eligible = _extract_cohort(cfg, predicates_df)
    if eligible.is_empty():
        logger.warning("No eligible triggers found for the base windows. Exiting.")
        return eligible

    positive = _extract_cohort(cfg.positive_config, predicates_df)
    pos_keys = _label_keys(positive, eligible).with_columns(pl.lit(True).alias("_is_pos"))

    if cfg.windows_neg is None:
        result = eligible.join(pos_keys, on=LABEL_JOIN_KEYS, how="left").with_columns(
            pl.col("_is_pos").fill_null(False).cast(PRED_CNT_TYPE).alias("label")
        )
        result = result.drop("_is_pos")
    else:
        negative = _extract_cohort(cfg.negative_config, predicates_df)
        neg_keys = _label_keys(negative, eligible).with_columns(pl.lit(True).alias("_is_neg"))

        marked = (
            eligible.join(pos_keys, on=LABEL_JOIN_KEYS, how="left")
            .join(neg_keys, on=LABEL_JOIN_KEYS, how="left")
            .with_columns(
                pl.col("_is_pos").fill_null(False),
                pl.col("_is_neg").fill_null(False),
            )
        )

        n_conflict = marked.filter(pl.col("_is_pos") & pl.col("_is_neg")).height
        if n_conflict:
            raise ValueError(
                f"{n_conflict:,} eligible trigger(s) matched both 'windows_pos' and 'windows_neg', which is "
                "ambiguous. Make the positive and negative window groups mutually exclusive."
            )

        # Keep the unambiguous positives and negatives; drop the ambiguous middle (matching neither group).
        # Conflicts (matching both) have already errored out above, so '|' here cannot mislabel a trigger.
        result = (
            marked.filter(pl.col("_is_pos") | pl.col("_is_neg"))
            .with_columns(pl.col("_is_pos").cast(PRED_CNT_TYPE).alias("label"))
            .drop("_is_pos", "_is_neg")
        )

    if not result.is_empty() and result["label"].n_unique() == 1:
        logger.warning(
            f"All labels in the extracted cohort are the same: '{result['label'][0]}'. "
            "This may indicate an issue with the task logic. "
            "Please double-check your configuration file if this is not expected."
        )

    logger.info(
        f"Done. {result.shape[0]:,} labeled rows returned corresponding to "
        f"{result['subject_id'].n_unique():,} subjects."
    )

    return _order_label_columns(result)
