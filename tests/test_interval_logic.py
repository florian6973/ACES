"""Tests for the relational predicates (``during`` / ``within``) and the disjunctive ``has_any`` window
constraint.

These exercise the three layers the feature touches:

- **config** parsing/validation of the new predicate kinds and the ``has_any`` window field,
- **predicate materialization** (the 0/1 relational predicate columns) through :func:`aces.predicates`,
- **constraint evaluation** of ``has_any`` end-to-end through :func:`aces.query.query`.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from omegaconf import DictConfig

from aces.config import (
    DuringPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
    WithinPredicateConfig,
)
from aces.predicates import add_within_predicate, get_predicates_df
from aces.query import query

TS_FORMAT = "%m/%d/%Y %H:%M"


# --------------------------------------------------------------------------------------------------------- #
# ``during`` -- point inside a reconstructed open/close pair-interval
# --------------------------------------------------------------------------------------------------------- #


def _during_column(df: pl.DataFrame, **cfg_kwargs: str) -> list[int]:
    """Evaluate a :class:`DuringPredicateConfig` over a sorted single-subject frame, returning the 0/1 col."""
    cfg = DuringPredicateConfig(**cfg_kwargs)
    return df.with_columns(cfg.eval_expr().cast(pl.Int64).alias("out"))["out"].to_list()


# One open/close interval with an AMI on the open boundary, strictly inside, on the close boundary, and after.
_DURING_DF = pl.DataFrame(
    {
        "subject_id": [1, 1, 1, 1],
        "timestamp": [
            datetime(2020, 1, 1),  # ip_start + ami  (open boundary)
            datetime(2020, 1, 2),  # ami             (strictly inside)
            datetime(2020, 1, 3),  # ip_end + ami    (close boundary)
            datetime(2020, 1, 4),  # ami             (after the interval)
        ],
        "ami": [1, 1, 1, 1],
        "ip_start": [1, 0, 0, 0],
        "ip_end": [0, 0, 1, 0],
    }
)


def test_during_closed_both():
    # Both boundaries inclusive: the open- and close-boundary AMIs are inside; the post-discharge one is not.
    assert _during_column(_DURING_DF, event="ami", opens="ip_start", closes="ip_end", closed="both") == [
        1,
        1,
        1,
        0,
    ]


def test_during_closed_left():
    # Left-closed: the close boundary is excluded -> the AMI on ip_end is no longer "during".
    assert _during_column(_DURING_DF, event="ami", opens="ip_start", closes="ip_end", closed="left") == [
        1,
        1,
        0,
        0,
    ]


def test_during_closed_right():
    # Right-closed: the open boundary is excluded -> the AMI on ip_start is no longer "during".
    assert _during_column(_DURING_DF, event="ami", opens="ip_start", closes="ip_end", closed="right") == [
        0,
        1,
        1,
        0,
    ]


def test_during_closed_none():
    # Neither boundary -> only the strictly interior AMI counts.
    assert _during_column(_DURING_DF, event="ami", opens="ip_start", closes="ip_end", closed="none") == [
        0,
        1,
        0,
        0,
    ]


def test_during_requires_event_to_hold():
    # ``during`` is monadic: it is 0 wherever ``event`` itself is absent, even if the time is "admitted".
    df = _DURING_DF.with_columns(pl.Series("ami", [0, 1, 0, 0]))
    assert _during_column(df, event="ami", opens="ip_start", closes="ip_end") == [0, 1, 0, 0]


def test_during_non_overlapping_two_intervals():
    # Net-open count is exact for separated, non-overlapping stays.
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 1],
            "timestamp": [datetime(2020, 1, d) for d in (1, 2, 3, 10, 11, 12)],
            "ami": [0, 1, 0, 0, 1, 0],  # one AMI inside stay 1, one inside stay 2
            "ip_start": [1, 0, 0, 1, 0, 0],
            "ip_end": [0, 0, 1, 0, 0, 1],
        }
    )
    assert _during_column(df, event="ami", opens="ip_start", closes="ip_end") == [0, 1, 0, 0, 1, 0]


def test_during_multi_subject_scoped_by_subject():
    # The cumulative net-open count must not leak across subjects.
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "timestamp": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            ],
            "ami": [0, 1, 0, 1],
            "ip_start": [1, 0, 0, 0],  # subject 2 has no open event, so its AMI is never "during"
            "ip_end": [0, 1, 0, 0],
        }
    )
    assert _during_column(df, event="ami", opens="ip_start", closes="ip_end") == [0, 1, 0, 0]


# --------------------------------------------------------------------------------------------------------- #
# ``within`` -- point inside a fixed offset window of another event
# --------------------------------------------------------------------------------------------------------- #


def test_within_symmetric_window_multi_subject():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "timestamp": [
                datetime(2020, 1, 1),  # smoking
                datetime(2020, 1, 6),  # cs4, smoking within 10d -> 1
                datetime(2020, 6, 1),  # cs4, no smoking nearby   -> 0
                datetime(2020, 1, 1),  # cs4, no smoking for subj 2 -> 0
                datetime(2020, 1, 5),  # smoking
            ],
            "cs4": [0, 1, 1, 1, 0],
            "smoking": [1, 0, 0, 0, 1],
        }
    )
    cfg = WithinPredicateConfig(event="cs4", of="smoking", before="10d", after="10d")
    out = add_within_predicate(df, "cs4_smk", cfg)
    assert out["cs4_smk"].to_list() == [0, 1, 0, 1, 0]  # subj 2's cs4 is within 10d (after) of its smoking


def test_within_one_sided_before_only():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 6), datetime(2020, 1, 20)],
            "cs4": [0, 1, 1],
            "smoking": [1, 0, 0],
        }
    )
    # before=10d, after=0: only smoking in the *preceding* 10 days corroborates.
    cfg = WithinPredicateConfig(event="cs4", of="smoking", before="10d", after="0d")
    out = add_within_predicate(df, "cs4_smk", cfg)
    assert out["cs4_smk"].to_list() == [0, 1, 0]  # Jan 6 is within 10d after Jan 1; Jan 20 is not


def test_within_zero_window_is_same_timestamp():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "cs4": [1, 1],
            "smoking": [1, 0],
        }
    )
    cfg = WithinPredicateConfig(event="cs4", of="smoking")  # before/after default to 0
    out = add_within_predicate(df, "cs4_smk", cfg)
    assert out["cs4_smk"].to_list() == [1, 0]


# --------------------------------------------------------------------------------------------------------- #
# Materialization through ``get_predicates_df`` (the real predicates pipeline)
# --------------------------------------------------------------------------------------------------------- #


def _write_csv(tmp_path, df: pl.DataFrame) -> DictConfig:
    data_path = tmp_path / "data.csv"
    df.write_csv(data_path)
    return DictConfig({"path": str(data_path), "standard": "direct", "ts_format": TS_FORMAT})


def test_during_materialized_in_predicates_df(tmp_path):
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "timestamp": ["01/01/2020 00:00", "01/02/2020 00:00", "01/03/2020 00:00", "01/04/2020 00:00"],
            "ami": [1, 1, 1, 1],
            "ip_start": [1, 0, 0, 0],
            "ip_end": [0, 0, 1, 0],
        }
    )
    cfg = TaskExtractorConfig(
        predicates={
            "ami": PlainPredicateConfig("ami"),
            "ip_start": PlainPredicateConfig("ip_start"),
            "ip_end": PlainPredicateConfig("ip_end"),
            "ami_enc": DuringPredicateConfig(event="ami", opens="ip_start", closes="ip_end"),
        },
        trigger=EventConfig("ami"),
        windows={"w": WindowConfig(None, "trigger", True, True, has={"ami_enc": "(0, None)"})},
    )
    predicates_df = get_predicates_df(cfg, _write_csv(tmp_path, df))
    assert "ami_enc" in predicates_df.columns
    got = predicates_df.sort("timestamp")["ami_enc"].to_list()
    assert got == [1, 1, 1, 0]


def test_within_materialized_in_predicates_df(tmp_path):
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "timestamp": ["01/01/2020 00:00", "01/06/2020 00:00", "06/01/2020 00:00"],
            "cs4": [0, 1, 1],
            "smoking": [1, 0, 0],
        }
    )
    cfg = TaskExtractorConfig(
        predicates={
            "cs4": PlainPredicateConfig("cs4"),
            "smoking": PlainPredicateConfig("smoking"),
            "cs4_smk": WithinPredicateConfig(event="cs4", of="smoking", before="10d", after="10d"),
        },
        trigger=EventConfig("cs4"),
        windows={"w": WindowConfig(None, "trigger", True, True, has={"cs4_smk": "(0, None)"})},
    )
    predicates_df = get_predicates_df(cfg, _write_csv(tmp_path, df))
    got = predicates_df.sort("timestamp")["cs4_smk"].to_list()
    assert got == [0, 1, 0]


def test_relational_predicate_can_feed_derived_expr(tmp_path):
    """A relational predicate column is ordinary and may be referenced by a later derived ``expr``."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "timestamp": ["01/01/2020 00:00", "01/02/2020 00:00", "01/04/2020 00:00"],
            "ami": [1, 1, 1],
            "ip_start": [1, 0, 0],
            "ip_end": [0, 1, 0],
        }
    )
    from aces.config import DerivedPredicateConfig

    cfg = TaskExtractorConfig(
        predicates={
            "ami": PlainPredicateConfig("ami"),
            "ip_start": PlainPredicateConfig("ip_start"),
            "ip_end": PlainPredicateConfig("ip_end"),
            "ami_enc": DuringPredicateConfig(event="ami", opens="ip_start", closes="ip_end"),
            "ami_and_enc": DerivedPredicateConfig("and(ami, ami_enc)"),
        },
        trigger=EventConfig("ami"),
        windows={"w": WindowConfig(None, "trigger", True, True, has={"ami_and_enc": "(0, None)"})},
    )
    predicates_df = get_predicates_df(cfg, _write_csv(tmp_path, df))
    assert predicates_df.sort("timestamp")["ami_and_enc"].to_list() == [1, 1, 0]


# --------------------------------------------------------------------------------------------------------- #
# ``has_any`` -- disjunctive window constraint, end-to-end through ``query``
# --------------------------------------------------------------------------------------------------------- #

T0 = datetime(2000, 1, 1)


def test_has_any_window_is_disjunctive():
    """A window with ``has_any`` keeps a trigger iff at least one block is satisfied over the window."""
    df = pl.DataFrame(
        [
            # subject 1: one `a` before trigger -> block {a:(1,None)} holds
            {"subject_id": 1, "timestamp": T0, "a": 1, "b": 0, "trig": 0},
            {"subject_id": 1, "timestamp": T0 + timedelta(days=1), "a": 0, "b": 0, "trig": 1},
            # subject 2: two `b` before trigger -> block {b:(2,None)} holds
            {"subject_id": 2, "timestamp": T0, "a": 0, "b": 1, "trig": 0},
            {"subject_id": 2, "timestamp": T0 + timedelta(hours=1), "a": 0, "b": 1, "trig": 0},
            {"subject_id": 2, "timestamp": T0 + timedelta(days=1), "a": 0, "b": 0, "trig": 1},
            # subject 3: neither block holds (one `b`, no `a`) -> dropped
            {"subject_id": 3, "timestamp": T0, "a": 0, "b": 1, "trig": 0},
            {"subject_id": 3, "timestamp": T0 + timedelta(days=1), "a": 0, "b": 0, "trig": 1},
        ]
    ).with_columns(pl.col(c).cast(pl.Int64) for c in ("a", "b", "trig"))

    cfg = TaskExtractorConfig(
        predicates={
            "a": PlainPredicateConfig("a"),
            "b": PlainPredicateConfig("b"),
            "trig": PlainPredicateConfig("trig"),
        },
        trigger=EventConfig("trig"),
        windows={
            "w": WindowConfig(
                start=None,
                end="trigger",
                start_inclusive=True,
                end_inclusive=True,
                has_any=[{"a": "(1, None)"}, {"b": "(2, None)"}],
            )
        },
    )
    result = query(cfg, df)
    assert sorted(result["subject_id"].to_list()) == [1, 2]  # subject 3 dropped


def test_has_any_and_has_mutually_exclusive():
    with pytest.raises(ValueError, match="'has' or 'has_any', but not both"):
        WindowConfig(
            start=None,
            end="trigger",
            start_inclusive=True,
            end_inclusive=True,
            has={"a": "(1, None)"},
            has_any=[{"b": "(1, None)"}],
        )


# --------------------------------------------------------------------------------------------------------- #
# YAML loading of the worked examples (#2 encounter-AMI via ``during``; #6 correlated entry via ``has_any``)
# --------------------------------------------------------------------------------------------------------- #

_DURING_YAML = """
predicates:
  ami:
    code: MI
  ip_start:
    code: ADMISSION
  ip_end:
    code: DISCHARGE
  ami_enc:
    during:
      event: ami
      opens: ip_start
      closes: ip_end
      closed: both

trigger: ip_start

windows:
  target:
    start: trigger
    end: start + 365d
    start_inclusive: true
    end_inclusive: true
    has:
      ami_enc: (1, None)
"""


def test_load_during_predicate_from_yaml(tmp_path):
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_DURING_YAML)
    cfg = TaskExtractorConfig.load(cfg_path)

    assert isinstance(cfg.predicates["ami_enc"], DuringPredicateConfig)
    # The relational predicate's input predicates were transitively pulled into the predicate set.
    for dep in ("ami", "ip_start", "ip_end"):
        assert dep in cfg.predicates
    # It is a derived (non-plain) predicate and appears after its inputs in topological order.
    assert "ami_enc" in cfg.derived_predicates


_HAS_ANY_YAML = """
predicates:
  cs1:
    code: CS1
  cs3:
    code: CS3
  cs4:
    code: CS4
  smoking:
    code: SMOKING
  adm:
    code: ADMISSION
  cs13:
    expr: or(cs1, cs3)
  cs4_smk:
    within:
      event: cs4
      of: smoking
      before: 365d
      after: 365d

trigger: adm

windows:
  at_risk_entry:
    start: null
    end: trigger
    start_inclusive: true
    end_inclusive: true
    has_any:
      - cs13: (1, None)
      - cs4: (2, None)
      - cs4_smk: (1, None)
"""


def test_load_within_and_has_any_from_yaml(tmp_path):
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_HAS_ANY_YAML)
    cfg = TaskExtractorConfig.load(cfg_path)

    assert isinstance(cfg.predicates["cs4_smk"], WithinPredicateConfig)
    # has_any predicates (including the relational + derived ones) are all pulled into the predicate set.
    for dep in ("cs1", "cs3", "cs4", "smoking", "cs13", "cs4_smk"):
        assert dep in cfg.predicates
    assert cfg.windows["at_risk_entry"].has_any == [
        {"cs13": (1, None)},
        {"cs4": (2, None)},
        {"cs4_smk": (1, None)},
    ]


def test_load_relational_undefined_input_raises(tmp_path):
    bad_yaml = """
predicates:
  ami:
    code: MI
  ami_enc:
    during:
      event: ami
      opens: ip_start
      closes: ip_end

trigger: ami

windows:
  w:
    start: null
    end: trigger
    start_inclusive: true
    end_inclusive: true
    has:
      ami_enc: (0, None)
"""
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(bad_yaml)
    with pytest.raises(KeyError, match="referenced in 'ami_enc'"):
        TaskExtractorConfig.load(cfg_path)


def test_during_end_to_end_query(tmp_path):
    """Full pipeline: a ``during`` predicate gates a target window via ``query`` (proposal example #2)."""
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2],
            "timestamp": [
                "01/01/2020 00:00",  # subj 1 admission (trigger)
                "01/02/2020 00:00",  # ami during the stay
                "01/03/2020 00:00",  # discharge
                "01/01/2020 00:00",  # subj 2 admission (trigger)
                "01/02/2020 00:00",  # discharge -- the stay is over
                "02/01/2020 00:00",  # ami AFTER discharge -> not an encounter-AMI
            ],
            "adm": [1, 0, 0, 1, 0, 0],
            "ami": [0, 1, 0, 0, 0, 1],
            "dis": [0, 0, 1, 0, 1, 0],
        }
    )
    cfg = TaskExtractorConfig(
        predicates={
            "adm": PlainPredicateConfig("adm"),
            "ami": PlainPredicateConfig("ami"),
            "dis": PlainPredicateConfig("dis"),
            "ami_enc": DuringPredicateConfig(event="ami", opens="adm", closes="dis"),
        },
        trigger=EventConfig("adm"),
        windows={
            "target": WindowConfig(
                start="trigger",
                end="start + 365d",
                start_inclusive=True,
                end_inclusive=True,
                has={"ami_enc": "(1, None)"},
            )
        },
    )
    predicates_df = get_predicates_df(cfg, _write_csv(tmp_path, df))
    result = query(cfg, predicates_df)
    # Only subject 1's admission has an encounter-AMI (an AMI during the adm->dis interval).
    assert result["subject_id"].to_list() == [1]
