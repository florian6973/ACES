# Auditing Task Configurations (`aces-audit`)

ACES deliberately offloads dataset-specific concepts to user-authored predicates. A predicate that
references an absent code, the wrong vocabulary, an impossible value constraint, or a non-existent
column silently produces an all-zero predicate column and an empty or wrong cohort — a class of error
that is otherwise only discovered after a full extraction.

`aces-audit` is a fast, **static** pre-flight tool that profiles a MEDS dataset's code/vocabulary
space and lints a task config's predicates against it *before* any extraction is run. It reuses ACES's
own predicate code-matching logic, so its notion of "which codes a predicate matches" is identical to
extraction's by construction.

## Quick start

```bash
aces-audit --config path/to/task.yaml --meds path/to/meds_dataset
```

```text
ACES static audit
  dataset: MIMIC-IV v2.2
  subjects: 100 | codes: 32 | time granularity: second | data scanned: True
  vocabularies: <local>(2), ADMISSION(3), DISCHARGE(3), LAB(22), SEX(2)
  config: task.yaml (5 predicates, trigger='admission')

ERROR (1)
  x P1 [lab_spo2]: predicate 'lab_spo2' matched 0 codes in the dataset
      code_spec=LAB//SpO2_typo; matched_codes=[]; nearest_present=[LAB//SpO2]

Summary: 1 error, 0 warning, 2 info
```

## Exit codes

`aces-audit` is designed to gate CI on a task-config repository:

| Exit code | Meaning |
| --- | --- |
| `0` | No ERROR findings (WARN/INFO allowed). |
| `1` | At least one ERROR finding. |
| `2` | Tool/input failure (config unparseable, dataset path invalid, etc.). |

## Options

| Flag | Purpose |
| --- | --- |
| `--config PATH` | The ACES task config YAML to audit (required). |
| `--meds PATH` | MEDS dataset root (with `data/` and optionally `metadata/`) or a single `.parquet` shard (required). |
| `--predicates PATH` | Optional predicates-override YAML (mirrors ACES's `predicates_path`). |
| `--no-data-scan` | Lint against `metadata/codes.parquet` only; skips the prevalence/value scan (&lt; 5 s on any dataset). |
| `--min-prevalence N` | Subject-count floor below which a matched predicate is flagged (default: 1). |
| `--top-orphans N` | High-prevalence unmatched codes to surface per targeted vocabulary (default: 25). |
| `--json PATH` | Write the deterministic machine-readable JSON report. |
| `--profile-only` | Print just the dataset profile; do not lint a config. |
| `--quiet` | Suppress the human-readable report on stdout. |

## What it checks

Each finding is phrased descriptively — *"predicate X cannot fire on this dataset because …"* — because
a genuinely absent concept (e.g. a lab predicate audited against a claims dataset) is a flag for human
review, not necessarily a defect.

### Plain predicates

| ID | Check | Severity |
| --- | --- | --- |
| P1 | `code` (exact/regex/any) matches **zero** codes in the dataset. | ERROR |
| P1A | An `any`-list matches some, but not all, of its members. | WARNING |
| P2 | The predicate targets a vocabulary namespace **entirely absent** from the dataset (strong "wrong CDM / unmapped" signal). | ERROR |
| P3 | A value constraint is set, but all matched codes have **no numeric value** (constraint can never be satisfied). | ERROR |
| P4 | The value constraint window lies **entirely outside** the observed value range (possible unit mismatch, e.g. mmol/L vs mg/dL). | WARNING |
| P5 | `other_cols` references a column **absent** from the data schema. | ERROR |
| P6 | Matched codes are present but their prevalence is **below the floor** (`--min-prevalence`). | WARNING |
| P7 | The targeted vocabulary is absent, but a **different version** of it is present (e.g. `ICD9CM` vs `ICD10CM`). | INFO |

### Derived predicates

| ID | Check | Severity |
| --- | --- | --- |
| D1 | The `expr` references a predicate name that is **not defined**. | ERROR |
| D2 | A base predicate is ERROR-flagged (propagated). Operator-aware: an `and(...)` cannot fire if **any** input is broken; an `or(...)` cannot fire only if **all** inputs are broken — a healthy `or` branch keeps it alive (the dead branch is still reported, e.g. a typo). | WARNING |

### Config level

| ID | Check | Severity |
| --- | --- | --- |
| C1 | The `trigger` predicate cannot fire — the cohort cannot be triggered at all. | ERROR |
| C2 | A defined predicate is referenced by no window and is not the trigger (dead predicate). | INFO |
| ORPHAN | High-prevalence codes in a targeted vocabulary that **no** predicate matches (inverse coverage). | INFO |
| T1 | Dataset timestamps are day-level but the config defines sub-day windows. | INFO |

`aces-audit` lints **every defined predicate**, including ones not yet wired into a window (which are
additionally reported as dead via C2), so problems are caught before a predicate is put to use.

## Edge cases & graceful degradation

- **Un-namespaced codes** (no `//`): vocabulary checks (P2, P7, ORPHAN) degrade to code-level checks;
  the vocabulary is reported as `<local>`.
- **Missing `metadata/codes.parquet`**: the code inventory is derived from a data scan instead, and a
  warning notes that the inventory may be incomplete after pre-processing.
- **`--no-data-scan`**: skips the *prevalence/value* scan. Only the scan-independent checks (P1, P1A,
  P2, P5, D1, D2, C1, C2) run; the value/prevalence checks (P3, P4, P6) and the inverse-coverage panel
  are omitted, and the header's subject count / time granularity show as unknown. Note that the tool
  still needs a code *inventory*: with `metadata/codes.parquet` present this costs no data read, but
  **without** it the `code` column of the shards is still read once (a minimal projection, not the full
  scan) to enumerate the codes to check against.

## Python API

`DatasetProfile` is a standalone, reusable artifact (code inventory, vocabularies, value typing,
prevalence, descriptions, `parent_codes`); the audit is layered on top of it.

```python
from aces.audit import DatasetProfile, audit_config, run_audit
from aces.config import TaskExtractorConfig

# One-shot orchestration (profile + load + lint):
report = run_audit("task.yaml", "meds_dataset", scan_data=True, min_prevalence=1)
report.has_errors       # bool
report.exit_code()      # 0 / 1
print(report.to_text())
report.to_json()        # deterministic JSON string

# Or compose the pieces yourself:
profile = DatasetProfile.from_meds("meds_dataset", scan_data=True)
cfg = TaskExtractorConfig.load(config_path="task.yaml")
report = audit_config(cfg, profile, min_prevalence=1)
```
