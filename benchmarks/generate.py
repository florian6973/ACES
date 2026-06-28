"""Deterministic synthetic event-stream generator for benchmarking ACES engines.

The generator emits a *direct*-standard table of **plain** predicate columns (one 0/1
column per plain predicate in a task config) and then runs it through ACES's own
:func:`aces.predicates.get_predicates_df`. That way the derived predicates,
``_ANY_EVENT``, and any special ``_RECORD_START`` / ``_RECORD_END`` columns are
materialized by the exact same code path the real pipeline uses -- so the legacy and
compiled engines are always fed identical, realistic, engine-ready input.

Everything is seeded: the same ``(cfg, n_subjects, events_per_subject, seed)`` produces
byte-identical output, which makes parity tests and benchmark runs reproducible.

Example:
    >>> from aces.config import TaskExtractorConfig
    >>> cfg = TaskExtractorConfig.load("sample_configs/inhospital_mortality.yaml")
    >>> df = generate_predicates_df(cfg, n_subjects=5, events_per_subject=8, seed=0)
    >>> df.columns
    ['subject_id', 'timestamp', 'admission', 'discharge', 'death', 'male', 'discharge_or_death', '_ANY_EVENT']
    >>> df["subject_id"].n_unique()
    5
    >>> # (subject_id, timestamp) is unique among non-static rows, as query() requires.
    >>> non_static = df.drop_nulls("timestamp")
    >>> non_static.n_unique(["subject_id", "timestamp"]) == non_static.height
    True
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df

# Plain-predicate firing probability per event, keyed by role. The trigger fires often
# enough to yield many prospective anchors; "boundary" predicates (those used as
# event-window endpoints, e.g. discharge/death) fire often enough that windows realize.
_DEFAULT_TRIGGER_P = 0.25
_DEFAULT_BOUNDARY_P = 0.20
_DEFAULT_OTHER_P = 0.10
_BASE_TIME = datetime(2010, 1, 1)


def _boundary_predicates(cfg: TaskExtractorConfig) -> set[str]:
    """Plain predicates referenced as event-window endpoints in the config tree.

    Derived endpoints (e.g. ``discharge_or_death``) are decomposed into the plain
    predicates that feed them so the underlying plain columns fire often enough for the
    event-bound windows to actually close.
    """
    from bigtree import preorder_iter

    from aces.types import ToEventWindowBounds

    plain = set(cfg.plain_predicates)
    endpoints: set[str] = set()
    for node in preorder_iter(cfg.window_tree):
        expr = getattr(node, "endpoint_expr", None)
        if isinstance(expr, ToEventWindowBounds):
            name = expr.end_event.lstrip("-")
            if name in plain:
                endpoints.add(name)
            elif name in cfg.derived_predicates:
                # Pull the plain predicates out of the derived predicate's expression.
                endpoints.update(p for p in cfg.derived_predicates[name].input_predicates if p in plain)
    return endpoints


def _firing_prob(cfg: TaskExtractorConfig, predicate: str, boundaries: set[str]) -> float:
    if predicate == cfg.trigger.predicate:
        return _DEFAULT_TRIGGER_P
    if predicate in boundaries:
        return _DEFAULT_BOUNDARY_P
    return _DEFAULT_OTHER_P


def generate_raw_events(
    cfg: TaskExtractorConfig,
    n_subjects: int,
    events_per_subject: int,
    seed: int = 0,
    mean_gap_hours: float = 12.0,
) -> pl.DataFrame:
    """Generate a direct-standard table of plain predicate columns.

    One row per (subject, event) with strictly increasing per-subject timestamps (so the
    ``(subject_id, timestamp)`` uniqueness invariant holds), plus one leading
    null-timestamp static row per subject carrying the static predicate values.

    Args:
        cfg: The task config (defines which plain predicates exist and which are static).
        n_subjects: Number of subjects to generate.
        events_per_subject: Number of timestamped events per subject.
        seed: RNG seed; output is a deterministic function of it.
        mean_gap_hours: Mean inter-event gap (exponential) in hours.

    Returns:
        A Polars DataFrame with ``subject_id``, ``timestamp`` and one 0/1 column per
        plain predicate, suitable for writing as a ``standard="direct"`` source.
    """
    rng = np.random.default_rng(seed)
    plain = list(cfg.plain_predicates)
    static = {p for p, c in cfg.predicates.items() if getattr(c, "static", False)}
    dynamic = [p for p in plain if p not in static]
    boundaries = _boundary_predicates(cfg)
    probs = {p: _firing_prob(cfg, p, boundaries) for p in dynamic}

    n_events = n_subjects * events_per_subject
    subject_ids = np.repeat(np.arange(1, n_subjects + 1), events_per_subject)

    # Per-subject strictly increasing timestamps from a cumulative exponential gap.
    gaps = rng.exponential(mean_gap_hours, size=n_events) + 1e-3
    gaps = gaps.reshape(n_subjects, events_per_subject)
    gaps[:, 0] = rng.uniform(0, 24, size=n_subjects)  # per-subject start jitter
    offsets_hours = np.cumsum(gaps, axis=1).reshape(-1)
    timestamps = [_BASE_TIME + timedelta(hours=float(h)) for h in offsets_hours]

    data: dict[str, list] = {"subject_id": list(subject_ids), "timestamp": timestamps}
    for p in dynamic:
        data[p] = list((rng.random(n_events) < probs[p]).astype(np.int64))
    for p in static:
        data[p] = [0] * n_events  # static predicates only carry signal on the null row

    events_df = pl.DataFrame(data)

    if static:
        static_rows = {
            "subject_id": list(range(1, n_subjects + 1)),
            "timestamp": [None] * n_subjects,
        }
        for p in dynamic:
            static_rows[p] = [0] * n_subjects
        for p in static:
            static_rows[p] = list((rng.random(n_subjects) < 0.5).astype(np.int64))
        static_df = pl.DataFrame(static_rows, schema=events_df.schema)
        events_df = pl.concat([static_df, events_df])

    return events_df.sort("subject_id", "timestamp", nulls_last=False)


def generate_predicates_df(
    cfg: TaskExtractorConfig,
    n_subjects: int,
    events_per_subject: int,
    seed: int = 0,
    mean_gap_hours: float = 12.0,
    tmp_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Generate an engine-ready predicates dataframe for ``cfg``.

    Builds raw plain-predicate events (:func:`generate_raw_events`), writes them to a
    temporary parquet, and runs ACES's :func:`get_predicates_df` to materialize derived
    predicates and special columns -- the same path the real pipeline uses.
    """
    import tempfile

    events_df = generate_raw_events(cfg, n_subjects, events_per_subject, seed, mean_gap_hours)

    tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.mkdtemp(prefix="aces_bench_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    data_path = tmp_dir / f"events_seed{seed}_n{n_subjects}_e{events_per_subject}.parquet"
    events_df.write_parquet(data_path)

    data_config = DictConfig({"path": str(data_path), "standard": "direct", "ts_format": None})
    return get_predicates_df(cfg, data_config)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic ACES predicates parquet.")
    parser.add_argument("--config", required=True, help="Path to a task config YAML.")
    parser.add_argument("--n-subjects", type=int, default=1000)
    parser.add_argument("--events-per-subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output parquet path for the predicates df.")
    args = parser.parse_args()

    cfg = TaskExtractorConfig.load(args.config)
    df = generate_predicates_df(cfg, args.n_subjects, args.events_per_subject, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"Wrote {df.height:,} rows ({df['subject_id'].n_unique():,} subjects) to {args.out}")
