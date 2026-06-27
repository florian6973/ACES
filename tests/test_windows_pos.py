"""Tests for the ``windows_pos`` / ``windows_neg`` label-defining window groups.

These exercise the multi-pass labeling driver directly through :func:`aces.query.query` (no subprocess /
tempfiles) so the semantics are checked in isolation:

- ``windows_pos`` is *non-gating*: an eligible trigger that fails it is still emitted with ``label = 0``.
- ``windows_neg`` (optional) states the negative class explicitly, dropping the ambiguous middle and any
  trigger that matches both groups.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from aces.query import query

T0 = datetime(2000, 1, 1)


def _row(sid: int, ts: datetime, adm: int = 0, ami: int = 0) -> dict:
    return {"subject_id": sid, "timestamp": ts, "adm": adm, "ami": ami}


def _predicates() -> dict:
    return {"adm": PlainPredicateConfig("adm"), "ami": PlainPredicateConfig("ami")}


# Every subject triggers once (an ``adm``). The base window is unconstrained, so all triggers are eligible;
# the in-year (<= 365d) presence of an ``ami`` decides the positive class.
_BASE = {"input": WindowConfig(None, "trigger", True, True, index_timestamp="end")}
_POS = {"inyear": WindowConfig("trigger", "start + 365d", True, True, has={"ami": "(1, None)"})}


def _labels(result: pl.DataFrame) -> dict[int, int]:
    return {r["subject_id"]: r["label"] for r in result.select("subject_id", "label").to_dicts()}


def test_windows_pos_implicit_complement_keeps_negatives():
    """Without ``windows_neg``, every eligible trigger is emitted; failing ``windows_pos`` -> label 0."""
    df = pl.DataFrame(
        [
            _row(1, T0, adm=1),
            _row(1, T0 + timedelta(days=10), ami=1),  # in-year ami -> positive
            _row(2, T0, adm=1),
            _row(2, T0 + timedelta(days=5)),  # no ami -> negative, still emitted
            _row(3, T0, adm=1),
            _row(3, T0 + timedelta(days=400), ami=1),  # ami outside the year -> negative, still emitted
        ]
    ).with_columns(pl.col("adm").cast(pl.Int64), pl.col("ami").cast(pl.Int64))

    cfg = TaskExtractorConfig(
        predicates=_predicates(), trigger=EventConfig("adm"), windows=_BASE, windows_pos=_POS
    )
    result = query(cfg, df)

    assert _labels(result) == {1: 1, 2: 0, 3: 0}
    # The cohort (positives + negatives) matches the plain eligible cohort: nothing is pruned by the label.
    assert result.height == 3
    assert result.columns[:3] == ["subject_id", "index_timestamp", "label"]
    # index_timestamp is sourced from the base window and identical for positives and negatives.
    assert set(result["index_timestamp"].to_list()) == {T0}


# Negatives are an explicit "only a late ami (after the prediction year)" group, mutually exclusive with the
# in-year positive group.
_NEG_LATE = {"late": WindowConfig("input.end + 366d", "start + 3284d", True, True, has={"ami": "(1, None)"})}


def test_windows_neg_emits_union_and_drops_ambiguous_middle():
    """With ``windows_neg``, only ``P ∪ N`` are emitted; triggers matching neither group are dropped."""
    df = pl.DataFrame(
        [
            _row(1, T0, adm=1),
            _row(1, T0 + timedelta(days=10), ami=1),  # in-year ami -> positive
            _row(2, T0, adm=1),
            _row(2, T0 + timedelta(days=5)),  # no ami at all -> ambiguous middle -> dropped
            _row(3, T0, adm=1),
            _row(3, T0 + timedelta(days=400), ami=1),  # only a late ami -> negative
        ]
    ).with_columns(pl.col("adm").cast(pl.Int64), pl.col("ami").cast(pl.Int64))

    cfg = TaskExtractorConfig(
        predicates=_predicates(),
        trigger=EventConfig("adm"),
        windows=_BASE,
        windows_pos=_POS,
        windows_neg=_NEG_LATE,
    )
    result = query(cfg, df)

    assert _labels(result) == {1: 1, 3: 0}  # subject 2 (matching neither group) is dropped


def test_windows_neg_conflict_raises():
    """A trigger satisfying both ``windows_pos`` and ``windows_neg`` is a config defect and raises."""
    df = pl.DataFrame(
        [
            _row(4, T0, adm=1),
            _row(4, T0 + timedelta(days=10), ami=1),  # in-year ami  -> matches windows_pos
            _row(4, T0 + timedelta(days=400), ami=1),  # late ami     -> also matches windows_neg
        ]
    ).with_columns(pl.col("adm").cast(pl.Int64), pl.col("ami").cast(pl.Int64))

    cfg = TaskExtractorConfig(
        predicates=_predicates(),
        trigger=EventConfig("adm"),
        windows=_BASE,
        windows_pos=_POS,
        windows_neg=_NEG_LATE,
    )
    with pytest.raises(ValueError, match="matched both 'windows_pos' and 'windows_neg'"):
        query(cfg, df)


def test_label_and_windows_pos_are_mutually_exclusive():
    labeled_base = {"input": WindowConfig(None, "trigger", True, False, label="ami")}
    with pytest.raises(ValueError, match="mutually exclusive"):
        TaskExtractorConfig(
            predicates=_predicates(),
            trigger=EventConfig("adm"),
            windows=labeled_base,
            windows_pos=_POS,
        )


def test_windows_neg_requires_windows_pos():
    with pytest.raises(ValueError, match="'windows_neg' may only be specified alongside 'windows_pos'"):
        TaskExtractorConfig(
            predicates=_predicates(), trigger=EventConfig("adm"), windows=_BASE, windows_neg=_POS
        )


def test_label_window_may_not_set_index_timestamp():
    bad = {"inyear": WindowConfig("trigger", "start + 365d", True, True, index_timestamp="end")}
    with pytest.raises(ValueError, match="may not set 'index_timestamp'"):
        TaskExtractorConfig(
            predicates=_predicates(), trigger=EventConfig("adm"), windows=_BASE, windows_pos=bad
        )


_CONFIG_YAML = """
predicates:
  adm:
    code: ADMISSION
  ami:
    code: MI

trigger: adm

windows:
  input:
    start: NULL
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end

windows_pos:
  inyear:
    start: trigger
    end: start + 365d
    start_inclusive: True
    end_inclusive: True
    has:
      ami: (1, None)

windows_neg:
  noami:
    start: trigger
    end: start + 365d
    start_inclusive: True
    end_inclusive: True
    has:
      ami: (0, 0)
"""


def test_load_parses_windows_pos_and_neg(tmp_path):
    """``windows_pos`` / ``windows_neg`` are parsed from YAML and drive the derived label configs."""
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_CONFIG_YAML)

    cfg = TaskExtractorConfig.load(cfg_path)

    assert set(cfg.windows) == {"input"}
    assert set(cfg.windows_pos) == {"inyear"}
    assert set(cfg.windows_neg) == {"noami"}
    # The label-defining windows referenced 'ami', so it must have been pulled into the predicate set.
    assert "ami" in cfg.predicates
    # Derived configs combine base + each label group and are validated on construction.
    assert set(cfg.positive_config.windows) == {"input", "inyear"}
    assert set(cfg.negative_config.windows) == {"input", "noami"}
    assert cfg.index_timestamp_window == "input"
