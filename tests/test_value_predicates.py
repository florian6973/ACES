"""Value predicates: lookback aggregates, arithmetic between measurements, and thresholds.

Each test states a clinical criterion that plain and derived predicates cannot express, and checks
it against a hand-built cohort whose correct labels are known by construction.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from omegaconf import DictConfig

from aces import query
from aces.config import TaskExtractorConfig

A = datetime(2100, 1, 1)  # ICU admission
H = timedelta(hours=1)


def build(tmp_path, events: dict[int, list[tuple[timedelta, str, float | None]]]) -> DictConfig:
    rows = []
    for sid, evs in events.items():
        rows.append({"subject_id": sid, "time": A, "code": "ICU_ADMISSION", "numeric_value": None})
        for off, code, val in evs:
            rows.append({"subject_id": sid, "time": A + off, "code": code, "numeric_value": val})
    df = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    ).sort(["subject_id", "time"])
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    return DictConfig({"path": str(path), "standard": "meds"})


def labels(cfg_yaml: str, data_config: DictConfig, tmp_path) -> dict[int, bool]:
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(cfg_yaml)
    cfg = TaskExtractorConfig.load(cfg_path)
    from aces.predicates import get_predicates_df

    result = query.query(cfg, get_predicates_df(cfg, data_config))
    col = "label" if "label" in result.columns else "boolean_value"
    return dict(zip(result["subject_id"].to_list(), (result[col] > 0).to_list(), strict=True))


WINDOWS = """
trigger: icu_admission
windows:
  input:
    start: null
    end: trigger + 24h
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
  target:
    start: input.end
    end: start + 24h
    start_inclusive: False
    end_inclusive: True
    label: {label}
"""


def test_ratio_between_two_measurements(tmp_path):
    """PaO2/FiO2: a ratio of two separately-recorded measurements, with a staleness bound.

    Plain predicates threshold one event's value against a constant; they cannot divide one
    measurement by another, nor pair a numerator with the most recent denominator.
    """
    cfg = """
predicates:
  icu_admission: {code: ICU_ADMISSION}
  pao2: {code: PAO2}
  fio2: {code: FIO2}
  pao2_now:    {value_agg: {of: pao2, fn: last}}
  fio2_recent: {value_agg: {of: fio2, fn: last, lookback: 6h}}
  fio2_frac:   {value_expr: {left: fio2_recent, op: "/", right: 100}}
  respiratory_failure:
    value_expr: {left: pao2_now, op: "/", right: fio2_frac}
    value_max: 300
    value_max_inclusive: True
""" + WINDOWS.format(label="respiratory_failure")

    data = build(tmp_path, {
        1: [(26 * H, "FIO2", 40.0), (27 * H, "PAO2", 200.0)],   # 500 -> no
        2: [(26 * H, "FIO2", 80.0), (27 * H, "PAO2", 120.0)],   # 150 -> yes
        3: [(10 * H, "FIO2", 80.0), (30 * H, "PAO2", 120.0)],   # FiO2 20h stale -> no
        4: [(27 * H, "PAO2", 90.0)],                             # no FiO2 at all -> no
        5: [(26 * H, "FIO2", 50.0), (27 * H, "PAO2", 150.0)],   # exactly 300, inclusive -> yes
    })
    assert labels(cfg, data, tmp_path) == {1: False, 2: True, 3: False, 4: False, 5: True}


def test_change_relative_to_patient_baseline(tmp_path):
    """KDIGO creatinine: a rise relative to the patient's OWN prior minimum, not a fixed cutoff.

    Subject 4 is the case that matters: a flat, high creatinine of 3.0 mg/dL is chronic kidney
    disease, not acute injury. Any fixed ">1.3 mg/dL" predicate calls it positive; the baseline-
    relative criterion must not.
    """
    cfg = """
predicates:
  icu_admission: {code: ICU_ADMISSION}
  creatinine: {code: CREAT}
  creat_now:      {value_agg: {of: creatinine, fn: last}}
  creat_baseline: {value_agg: {of: creatinine, fn: min, lookback: 7d}}
  aki:
    value_expr: {left: creat_now, op: "/", right: creat_baseline}
    value_min: 1.5
    value_min_inclusive: True
""" + WINDOWS.format(label="aki")

    flat = lambda v: [(h * H, "CREAT", v) for h in (2, 20, 30)]  # noqa: E731
    data = build(tmp_path, {
        1: flat(1.0),                                                        # flat normal -> no
        2: [(2 * H, "CREAT", 1.0), (20 * H, "CREAT", 1.0), (30 * H, "CREAT", 1.6)],  # 1.6x -> yes
        3: [(2 * H, "CREAT", 1.0), (20 * H, "CREAT", 1.0), (30 * H, "CREAT", 1.4)],  # 1.4x -> no
        4: flat(3.0),                                        # chronically high but FLAT -> no
    })
    assert labels(cfg, data, tmp_path) == {1: False, 2: True, 3: False, 4: False}


def test_conjunction_across_time(tmp_path):
    """Circulatory failure: two criteria that must co-occur within a window, not at one instant.

    `and(...)` is row-wise on the per-timestamp frame, so it only fires when both components were
    recorded at the same instant. Lifting each component to "occurred within the last 8h" first
    turns the same `and(...)` into a conjunction across time.
    """
    cfg = """
predicates:
  icu_admission: {code: ICU_ADMISSION}
  high_lactate: {code: LACTATE, value_min: 2.0, value_min_inclusive: False}
  low_map: {code: MAP, value_max: 65, value_max_inclusive: False}
  vasopressor: {code: NOREPI}
  hypotension_or_pressor: {expr: "or(low_map, vasopressor)"}
  high_lactate_8h:
    value_agg: {of: high_lactate, fn: count, lookback: 8h}
    value_min: 1
    value_min_inclusive: True
  hypo_or_pressor_8h:
    value_agg: {of: hypotension_or_pressor, fn: count, lookback: 8h}
    value_min: 1
    value_min_inclusive: True
  circulatory_failure: {expr: "and(high_lactate_8h, hypo_or_pressor_8h)"}
""" + WINDOWS.format(label="circulatory_failure")

    data = build(tmp_path, {
        1: [(26 * H, "LACTATE", 4.0), (28 * H, "MAP", 55.0)],   # 2h apart -> yes
        2: [(26 * H, "LACTATE", 4.0), (40 * H, "MAP", 55.0)],   # 14h apart -> no
        3: [(26 * H, "LACTATE", 4.0)],                           # lactate alone -> no
        4: [(26 * H, "MAP", 55.0)],                              # hypotension alone -> no
        5: [(26 * H, "LACTATE", 5.0), (29 * H, "NOREPI", None)],  # lactate + pressor -> yes
        6: [(26 * H, "LACTATE", 1.5), (27 * H, "MAP", 55.0)],   # lactate not elevated -> no
        7: [(26 * H, "LACTATE", 4.0), (26 * H, "MAP", 55.0)],   # same instant -> yes
    })
    assert labels(cfg, data, tmp_path) == {
        1: True, 2: False, 3: False, 4: False, 5: True, 6: False, 7: True
    }


def test_rate_needs_an_observation_guard(tmp_path):
    """A rate computed over an unobserved window reads as zero, which must not count as oliguria.

    This is the trap in the KDIGO urine-output criterion: 0 mL charted over 6h looks maximally
    oliguric. Counting the observations in the same window and requiring the window to be covered
    is expressible with the same primitives.
    """
    cfg = """
predicates:
  icu_admission: {code: ICU_ADMISSION}
  urine_output: {code: URINE}
  weight: {code: WEIGHT}
  uo_6h:     {value_agg: {of: urine_output, fn: sum, lookback: 6h}}
  weight_kg: {value_agg: {of: weight, fn: last, lookback: 30d}}
  uo_rate_low:
    value_expr: {left: uo_6h, op: "/", right: weight_kg}
    value_max: 3.0
    value_max_inclusive: False
  uo_observed_6h:
    value_agg: {of: urine_output, fn: count, lookback: 6h}
    value_min: 6
    value_min_inclusive: True
  oliguria: {expr: "and(uo_rate_low, uo_observed_6h)"}
""" + WINDOWS.format(label="oliguria")

    hourly = lambda first, ml: [((first + i) * H, "URINE", ml) for i in range(6)]  # noqa: E731
    data = build(tmp_path, {
        1: [(0 * H, "WEIGHT", 70.0), *hourly(26, 10.0)],   # 60mL/70kg over 6h -> oliguric
        2: [(0 * H, "WEIGHT", 70.0), *hourly(26, 90.0)],   # 540mL/70kg -> not oliguric
        3: [(0 * H, "WEIGHT", 70.0)],                       # nothing charted -> NOT oliguric
    })
    assert labels(cfg, data, tmp_path) == {1: True, 2: False, 3: False}


def test_non_count_aggregate_of_a_boolean_predicate_is_rejected(tmp_path):
    """A derived and(...)/or(...) predicate has no numeric value, so only `count` is meaningful."""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("""
predicates:
  icu_admission: {code: ICU_ADMISSION}
  a: {code: A}
  b: {code: B}
  a_or_b: {expr: "or(a, b)"}
  bad: {value_agg: {of: a_or_b, fn: mean, lookback: 6h}, value_min: 1}
""" + WINDOWS.format(label="bad"))
    with pytest.raises(ValueError, match="carries no numeric value"):
        TaskExtractorConfig.load(cfg_path)
