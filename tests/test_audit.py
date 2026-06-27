"""Tests for the static audit tool (``aces.audit``).

Each test builds a small, self-contained MEDS dataset (data shards + ``metadata/codes.parquet`` +
``metadata/dataset.json``) in a temporary directory and audits hand-written task configs against it,
covering the acceptance criteria from the audit spec.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from aces.audit import DatasetProfile, Severity, run_audit
from aces.audit.cli import main as audit_main

if TYPE_CHECKING:
    from pathlib import Path

MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float64,
}

# Codes present in the synthetic dataset, with rough prevalence by design:
#   ICD10CM//E11 (diabetes) is present in 8/10 subjects but is matched by NO predicate in the
#   "known-good" config -> it is the canary for the inverse-coverage (orphan) check.
INVENTORY = [
    "SEX//MALE",
    "SEX//FEMALE",
    "ADMISSION",
    "DISCHARGE",
    "SNOMED//8867-4",  # heart rate, numeric
    "SNOMED//CATEGORICAL",  # value-less event
    "ICD10CM//I21",  # MI, 4 subjects
    "ICD10CM//E11",  # diabetes, 8 subjects (high-prevalence orphan)
]


def _build_meds(root: Path) -> Path:
    """Write a small MEDS dataset (2 data shards + code/dataset metadata) under ``root``."""
    (root / "data" / "train").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)

    rows: list[dict] = []

    def ev(sid: int, t: datetime | None, code: str, val: float | None = None) -> None:
        rows.append({"subject_id": sid, "time": t, "code": code, "numeric_value": val})

    base = datetime(2020, 1, 1, 12, 30)
    for sid in range(1, 11):
        ev(sid, None, "SEX//MALE" if sid % 2 else "SEX//FEMALE")  # static row (null time)
        ev(sid, base, "ADMISSION")
        for h in range(3):
            ev(sid, base.replace(hour=12 + h), "SNOMED//8867-4", 70.0 + sid + h)
        ev(sid, base, "SNOMED//CATEGORICAL")
        if sid <= 4:
            ev(sid, base, "ICD10CM//I21")
        if sid <= 8:
            ev(sid, base, "ICD10CM//E11")
        ev(sid, base.replace(hour=20), "DISCHARGE")

    df = pl.DataFrame(rows, schema=MEDS_SCHEMA)
    df.filter(pl.col("subject_id") <= 5).write_parquet(root / "data" / "train" / "0.parquet")
    df.filter(pl.col("subject_id") > 5).write_parquet(root / "data" / "train" / "1.parquet")

    pl.DataFrame(
        {
            "code": INVENTORY,
            "description": INVENTORY,
            "parent_codes": [[] for _ in INVENTORY],
        },
        schema={"code": pl.String, "description": pl.String, "parent_codes": pl.List(pl.String)},
    ).write_parquet(root / "metadata" / "codes.parquet")

    (root / "metadata" / "dataset.json").write_text(
        json.dumps({"dataset_name": "test_meds", "dataset_version": "1.0"})
    )
    return root


@pytest.fixture
def meds(tmp_path: Path) -> Path:
    return _build_meds(tmp_path / "dataset")


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


_WINDOWS = """
windows:
  input:
    start: NULL
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
  target:
    start: input.end
    end: start + 1 days
    start_inclusive: False
    end_inclusive: True
    has: {{{has}}}
    {label}
"""


def _cfg(predicates: str, trigger: str, has: str, label: str = "") -> str:
    return f"predicates:\n{predicates}\ntrigger: {trigger}\n" + _WINDOWS.format(has=has, label=label)


KNOWN_GOOD = _cfg(
    predicates="""\
  admission: {code: ADMISSION}
  discharge: {code: DISCHARGE}
  hr: {code: SNOMED//8867-4}
  hr_high:
    code: SNOMED//8867-4
    value_min: 70
    value_max: 300
    value_min_inclusive: true
    value_max_inclusive: true
  mi: {code: ICD10CM//I21}
""",
    trigger="admission",
    has="hr: '(1, None)', mi: '(None, None)', hr_high: '(None, None)', discharge: '(None, None)'",
    label="label: mi",
)


# --------------------------------------------------------------------------- #
# Acceptance tests
# --------------------------------------------------------------------------- #
def test_profile_metadata(meds: Path):
    profile = DatasetProfile.from_meds(meds, scan_data=True)
    assert profile.codes_metadata_complete is True
    assert profile.name == "test_meds"
    assert profile.version == "1.0"
    assert profile.n_subjects == 10
    assert set(profile.vocabularies) == {"<local>", "SEX", "SNOMED", "ICD10CM"}
    assert profile.has_static_rows is True
    # Heart-rate code carries numeric values; categorical code does not.
    assert profile.codes["SNOMED//8867-4"].has_numeric is True
    assert profile.codes["SNOMED//CATEGORICAL"].has_numeric is False


def test_misspelled_code_is_single_p1(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "typo.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  typo: {code: SNOMED//8867-9}\n",
            trigger="admission",
            has="typo: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    p1 = [f for f in report.findings if f.id == "P1" and f.severity is Severity.ERROR]
    assert len(p1) == 1
    assert p1[0].predicate == "typo"
    assert "SNOMED//8867-4" in p1[0].detail["nearest_present"]
    assert report.exit_code() == 1


def test_absent_vocabulary_is_p2(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "vocab.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  old_dx: {code: ICD9CM//250.00}\n",
            trigger="admission",
            has="old_dx: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    p2 = [f for f in report.findings if f.id == "P2"]
    assert len(p2) == 1
    assert p2[0].predicate == "old_dx"
    assert "ICD9CM" in p2[0].detail["absent_vocabularies"]
    # P2 explains the absence, so P1 should not also fire for the same predicate.
    assert not any(f.id == "P1" and f.predicate == "old_dx" for f in report.findings)
    assert report.exit_code() == 1


def test_value_constraint_on_categorical_is_p3(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "p3.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  bad_val: {code: SNOMED//CATEGORICAL, value_min: 1}\n",
            trigger="admission",
            has="bad_val: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    p3 = [f for f in report.findings if f.id == "P3"]
    assert len(p3) == 1
    assert p3[0].predicate == "bad_val"
    assert report.exit_code() == 1


def test_known_good_has_no_errors(meds: Path, tmp_path: Path):
    cfg = _write(tmp_path, "good.yaml", KNOWN_GOOD)
    report = run_audit(cfg, meds)
    errors = [f for f in report.findings if f.severity is Severity.ERROR]
    assert errors == [], f"unexpected errors: {[(f.id, f.predicate) for f in errors]}"
    assert report.exit_code() == 0


def test_derived_undefined_reference_is_d1(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "d1.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  combo: {expr: 'or(admission, ghost)'}\n",
            trigger="admission",
            has="admission: '(None, None)'",
            label="label: combo",
        ),
    )
    report = run_audit(cfg, meds)
    d1 = [f for f in report.findings if f.id == "D1"]
    assert len(d1) == 1
    assert d1[0].detail["undefined_reference"] == "ghost"
    assert report.exit_code() == 1


def test_or_with_healthy_branch_still_fires(meds: Path, tmp_path: Path):
    # admission is healthy, ghost is broken -> the or() can still fire, so it must NOT be reported as
    # "cannot fire" and must not trigger a C1 (empty-cohort) finding.
    cfg = _write(
        tmp_path,
        "or_partial.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n"
            "  ghost: {code: NONEXISTENT}\n"
            "  combo: {expr: 'or(admission, ghost)'}\n",
            trigger="combo",
            has="admission: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    d2 = [f for f in report.findings if f.id == "D2" and f.predicate == "combo"]
    assert len(d2) == 1
    assert "can still fire" in d2[0].message
    assert d2[0].detail["healthy_inputs"] == ["admission"]
    assert d2[0].detail["broken_inputs"] == ["ghost"]
    # combo can still fire, so the trigger is not dead.
    assert not any(f.id == "C1" for f in report.findings)


def test_and_with_broken_input_cannot_fire(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "and_broken.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n"
            "  ghost: {code: NONEXISTENT}\n"
            "  combo: {expr: 'and(admission, ghost)'}\n",
            trigger="combo",
            has="admission: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    d2 = [f for f in report.findings if f.id == "D2" and f.predicate == "combo"]
    assert len(d2) == 1
    assert "cannot fire" in d2[0].message
    # the and() cannot fire, so the trigger cannot fire either -> C1.
    assert any(f.id == "C1" for f in report.findings)


def test_inverse_coverage_surfaces_high_prevalence_orphan(meds: Path, tmp_path: Path):
    cfg = _write(tmp_path, "good.yaml", KNOWN_GOOD)
    report = run_audit(cfg, meds)
    orphans = [f for f in report.findings if f.id == "ORPHAN" and f.detail["vocabulary"] == "ICD10CM"]
    assert len(orphans) == 1
    codes = [c["code"] for c in orphans[0].detail["top_unmatched"]]
    assert "ICD10CM//E11" in codes
    assert orphans[0].severity is Severity.INFO


def test_determinism_byte_identical_json(meds: Path, tmp_path: Path):
    cfg = _write(tmp_path, "good.yaml", KNOWN_GOOD)
    a = run_audit(cfg, meds).to_json()
    b = run_audit(cfg, meds).to_json()
    assert a == b


def test_no_data_scan_matches_static_findings(meds: Path, tmp_path: Path):
    broken = _cfg(
        predicates="""\
  admission: {code: GHOST_ADMIT}
  bad_vocab: {code: ICD9CM//1}
  bad_col: {code: ADMISSION, weird_col: "x"}
""",
        trigger="admission",
        has="bad_vocab: '(None, None)', bad_col: '(None, None)'",
    )
    cfg = _write(tmp_path, "broken.yaml", broken)

    static_ids = {"P1", "P2", "P5", "D1", "C1"}
    scanned = run_audit(cfg, meds, scan_data=True)
    not_scanned = run_audit(cfg, meds, scan_data=False)

    def static(report):
        return sorted((f.id, f.predicate) for f in report.findings if f.id in static_ids)

    assert static(scanned) == static(not_scanned)
    # The expected static findings are all present.
    got = set(static(scanned))
    assert ("P1", "admission") in got
    assert ("C1", "admission") in got
    assert ("P2", "bad_vocab") in got
    assert ("P5", "bad_col") in got


def test_matched_codes_equals_extraction(meds: Path):
    """The audit's notion of which codes a predicate matches must equal extraction's."""
    from aces.config import PlainPredicateConfig

    profile = DatasetProfile.from_meds(meds, scan_data=False)
    pred = PlainPredicateConfig(code={"regex": "ICD10CM//.*"})

    audit_codes = set(profile.matched_codes(pred))

    # Reference: run real extraction over one shard and read back the matched code set.
    shard = meds / "data" / "train" / "0.parquet"
    df = pl.read_parquet(shard).rename({"time": "timestamp"})
    matched = df.filter(pred.MEDS_eval_expr())["code"].unique().to_list()
    # The audit inventory spans all codes; intersect with what this shard could exercise.
    shard_codes = set(df["code"].to_list())
    assert {c for c in audit_codes if c in shard_codes} == set(matched)


def test_partial_any_list_warns(meds: Path, tmp_path: Path):
    cfg = _write(
        tmp_path,
        "any.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  dx: {code: {any: [ICD10CM//I21, ICD10CM//BOGUS]}}\n",
            trigger="admission",
            has="dx: '(None, None)'",
        ),
    )
    report = run_audit(cfg, meds)
    p1a = [f for f in report.findings if f.id == "P1A"]
    assert len(p1a) == 1
    assert p1a[0].detail["unmatched_members"] == ["ICD10CM//BOGUS"]
    assert p1a[0].severity is Severity.WARNING


def test_cli_exit_codes_and_json(meds: Path, tmp_path: Path, capsys):
    good = _write(tmp_path, "good.yaml", KNOWN_GOOD)
    out_json = tmp_path / "out.json"
    rc = audit_main(["--config", str(good), "--meds", str(meds), "--json", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["aces_audit_version"]
    assert payload["dataset"]["n_subjects"] == 10

    typo = _write(
        tmp_path,
        "typo.yaml",
        _cfg(
            "  admission: {code: ADMISSION}\n  typo: {code: SNOMED//8867-9}\n",
            trigger="admission",
            has="typo: '(None, None)'",
        ),
    )
    rc = audit_main(["--config", str(typo), "--meds", str(meds)])
    assert rc == 1

    rc = audit_main(["--config", str(tmp_path / "missing.yaml"), "--meds", str(meds)])
    assert rc == 2


def test_cli_profile_only(meds: Path, capsys):
    rc = audit_main(["--meds", str(meds), "--profile-only", "--config", "ignored"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "subjects: 10" in out
