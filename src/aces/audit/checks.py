"""Static lint checks for an ACES task config against a :class:`DatasetProfile`.

The checks implement the predicate / derived / config-level rules and the inverse-coverage ("orphan")
analysis described in the audit spec. Findings are phrased descriptively -- "predicate X cannot fire on
this dataset because …" -- because an absent concept is a flag for human review, not necessarily a
defect in the config.
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import ruamel.yaml

from aces.config import DerivedPredicateConfig, PlainPredicateConfig, TaskExtractorConfig

from .profile import LOCAL_VOCAB, DatasetProfile, vocabulary_of
from .report import AuditReport, Finding, Severity

logger = logging.getLogger(__name__)

ERROR, WARNING, INFO = Severity.ERROR, Severity.WARNING, Severity.INFO


class AuditInputError(Exception):
    """Raised when the config or dataset cannot be loaded (maps to CLI exit code 2)."""


# ---------------------------------------------------------------------------- #
# Public entrypoints
# ---------------------------------------------------------------------------- #
def audit_config(
    task_cfg: TaskExtractorConfig,
    profile: DatasetProfile,
    *,
    min_prevalence: float = 1,
    top_orphans: int = 25,
    all_predicate_objs: dict[str, PlainPredicateConfig | DerivedPredicateConfig] | None = None,
    config_path: str | None = None,
    extra_findings: list[Finding] | None = None,
) -> AuditReport:
    """Lint a config's predicates against ``profile`` and return a structured report.

    Args:
        task_cfg: A loaded ACES task configuration (used for the trigger, windows, and the set of
            *referenced* predicates).
        profile: A dataset profile (see :class:`DatasetProfile`).
        min_prevalence: Subject-count floor below which a matched predicate is flagged (P6).
        top_orphans: Number of high-prevalence unmatched codes to surface per targeted vocabulary.
        all_predicate_objs: Every predicate defined in the raw config (before the loader prunes
            unreferenced ones), so the lint covers predicates that are defined-but-unused too. When
            ``None``, only the referenced predicates retained on ``task_cfg`` are linted.
        config_path: Path string for the report header.
        extra_findings: Findings produced before the audit (e.g. D1) to fold into the report.

    Returns:
        The :class:`AuditReport`.
    """
    findings: list[Finding] = list(extra_findings or [])

    source = all_predicate_objs if all_predicate_objs is not None else dict(task_cfg.predicates)
    plain = {n: c for n, c in source.items() if c.is_plain}
    derived = {n: c for n, c in source.items() if not c.is_plain}

    matched: dict[str, list[str]] = {}
    error_preds: set[str] = set()
    targeted_present_vocabs: set[str] = set()

    for name, cfg in plain.items():
        try:
            codes = profile.matched_codes(cfg)
        except ValueError as e:
            error_preds.add(name)
            matched[name] = []
            findings.append(
                Finding(
                    "P0",
                    ERROR,
                    f"predicate '{name}' has an invalid code specification: {e}",
                    predicate=name,
                    detail={"code_spec": _code_spec(cfg)},
                )
            )
            continue
        matched[name] = codes
        pred_findings, is_error, present = _check_plain(name, cfg, profile, codes, min_prevalence)
        findings.extend(pred_findings)
        if is_error:
            error_preds.add(name)
        targeted_present_vocabs |= present

    # D2: derived predicate affected by broken inputs (propagated). Whether it can still fire depends
    # on the operator: an `and(...)` cannot fire if ANY input is broken, whereas an `or(...)` cannot
    # fire only if ALL inputs are broken -- a healthy branch keeps it alive. A fixpoint loop handles
    # chains of derived-on-derived dependencies without relying on a topological ordering.
    broken = set(error_preds)
    flagged: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, dcfg in derived.items():
            if name in flagged:
                continue
            broken_inputs = sorted(p for p in dcfg.input_predicates if p in broken)
            if not broken_inputs:
                continue

            healthy_inputs = [p for p in dcfg.input_predicates if p not in broken]
            cannot_fire = dcfg.is_and or not healthy_inputs

            if cannot_fire:
                findings.append(
                    Finding(
                        "D2",
                        WARNING,
                        f"derived predicate '{name}' cannot fire because input predicate(s) "
                        f"{broken_inputs} cannot match this dataset",
                        predicate=name,
                        detail={"expr": dcfg.expr, "broken_inputs": broken_inputs},
                    )
                )
                broken.add(name)
                changed = True  # newly broken -> dependents may need re-evaluation
            else:
                # or(...) with a healthy branch: still fires, but a branch is dead weight (likely a typo).
                findings.append(
                    Finding(
                        "D2",
                        WARNING,
                        f"derived predicate '{name}' can still fire via {sorted(healthy_inputs)}, but "
                        f"or-branch(es) {broken_inputs} cannot match this dataset",
                        predicate=name,
                        detail={
                            "expr": dcfg.expr,
                            "broken_inputs": broken_inputs,
                            "healthy_inputs": sorted(healthy_inputs),
                        },
                    )
                )
            flagged.add(name)

    # C1: the trigger cannot fire -> the cohort is empty.
    trigger = task_cfg.trigger.predicate
    if trigger in broken:
        findings.append(
            Finding(
                "C1",
                ERROR,
                f"trigger predicate '{trigger}' cannot match this dataset, so no events can be "
                "triggered and the extracted cohort will be empty",
                predicate=trigger,
            )
        )

    # C2: dead predicates (defined but referenced by no window and not the trigger). The loader keeps
    # only referenced predicates on task_cfg, so anything in the full set but not there is dead.
    if all_predicate_objs is not None:
        referenced = set(task_cfg.predicates.keys())
        for dead in sorted(set(all_predicate_objs) - referenced - {trigger}):
            findings.append(
                Finding(
                    "C2",
                    INFO,
                    f"predicate '{dead}' is defined but referenced by no window and is not the "
                    "trigger (dead predicate)",
                    predicate=dead,
                )
            )

    # 4.3 inverse coverage (requires prevalence ranking).
    if profile.data_scanned:
        findings.extend(_inverse_coverage(profile, matched, targeted_present_vocabs, top_orphans))

    # Coarse-timestamp note (best effort).
    findings.extend(_coarse_timestamp_check(task_cfg, profile))

    return AuditReport(
        dataset=_dataset_header(profile),
        config={
            "path": config_path,
            "n_predicates": len(task_cfg.predicates),
            "trigger_predicate": trigger,
        },
        findings=findings,
    )


def run_audit(
    config_path: str | Path,
    meds_path: str | Path,
    *,
    scan_data: bool = True,
    min_prevalence: float = 1,
    top_orphans: int = 25,
    predicates_path: str | Path | None = None,
) -> AuditReport:
    """Profile ``meds_path``, load ``config_path``, and return the full audit report.

    This is the convenience orchestrator behind the CLI. It detects undefined derived references (D1)
    from the raw YAML *before* loading, because ACES's loader raises on such references; when present,
    the report surfaces them as D1 findings instead of crashing.

    Raises:
        AuditInputError: If the dataset or config cannot be read/parsed for reasons other than an
            undefined predicate reference.
    """
    config_path = Path(config_path)
    try:
        predicates_raw, demographics_raw, trigger_raw, _windows = _read_raw_predicates(
            config_path, predicates_path
        )
    except Exception as e:
        raise AuditInputError(f"Could not parse config '{config_path}': {e}") from e

    all_predicates = {**predicates_raw, **demographics_raw}
    d1 = lint_derived_references(all_predicates)
    predicate_objs = _build_predicate_objs(predicates_raw, demographics_raw)

    try:
        profile = DatasetProfile.from_meds(meds_path, scan_data=scan_data)
    except FileNotFoundError as e:
        raise AuditInputError(str(e)) from e

    try:
        task_cfg = TaskExtractorConfig.load(config_path=config_path, predicates_path=predicates_path)
    except (KeyError, ValueError) as e:
        if d1:
            # The load failure is the undefined reference we already diagnosed as D1.
            return AuditReport(
                dataset=_dataset_header(profile),
                config={
                    "path": str(config_path),
                    "n_predicates": len(all_predicates),
                    "trigger_predicate": _trigger_name(trigger_raw),
                },
                findings=d1,
            )
        raise AuditInputError(f"Could not load config '{config_path}': {e}") from e

    return audit_config(
        task_cfg,
        profile,
        min_prevalence=min_prevalence,
        top_orphans=top_orphans,
        all_predicate_objs=predicate_objs,
        config_path=str(config_path),
        extra_findings=d1,
    )


def lint_derived_references(raw_predicates: dict[str, Any]) -> list[Finding]:
    """Return D1 findings for derived predicates that reference an undefined predicate.

    This works on the raw predicate dictionary (before :class:`TaskExtractorConfig` parsing), so it can
    diagnose configs that ACES's stricter loader would reject outright.
    """
    defined = set(raw_predicates)
    findings: list[Finding] = []
    for name, cfg in raw_predicates.items():
        if not (isinstance(cfg, dict) and "expr" in cfg):
            continue
        try:
            dcfg = DerivedPredicateConfig(expr=cfg["expr"])
        except ValueError:
            continue  # malformed expr is a separate, loader-surfaced concern
        for ref in dcfg.input_predicates:
            if ref not in defined:
                findings.append(
                    Finding(
                        "D1",
                        ERROR,
                        f"derived predicate '{name}' references '{ref}', which is not defined in "
                        "the configuration",
                        predicate=name,
                        detail={"expr": cfg["expr"], "undefined_reference": ref},
                    )
                )
    return findings


# ---------------------------------------------------------------------------- #
# Plain-predicate checks (P1-P7)
# ---------------------------------------------------------------------------- #
def _check_plain(
    name: str,
    cfg: PlainPredicateConfig,
    profile: DatasetProfile,
    codes: list[str],
    min_prevalence: float,
) -> tuple[list[Finding], bool, set[str]]:
    findings: list[Finding] = []
    is_error = False

    target_vocabs = _target_vocabularies(cfg)
    present_vocabs: set[str] = set()
    absent_vocabs: list[str] = []
    if target_vocabs is not None:
        for v in target_vocabs:
            if v == LOCAL_VOCAB or v in profile.vocabularies:
                present_vocabs.add(v)
            else:
                absent_vocabs.append(v)
    all_vocabs_absent = bool(target_vocabs) and len(present_vocabs) == 0

    # P2: a targeted vocabulary namespace is entirely absent.
    if absent_vocabs:
        is_error = True
        absent_sorted = sorted(absent_vocabs)
        findings.append(
            Finding(
                "P2",
                ERROR,
                f"predicate '{name}' targets vocabulary namespace(s) {absent_sorted} that are "
                "entirely absent from the dataset (possible wrong CDM or unmapped vocabulary)",
                predicate=name,
                detail={
                    "absent_vocabularies": absent_sorted,
                    "present_vocabularies": sorted(profile.vocabularies)[:10],
                },
            )
        )
        # P7: version mismatch hint.
        for v in absent_sorted:
            sib = _version_sibling(v, profile.vocabularies)
            if sib:
                findings.append(
                    Finding(
                        "P7",
                        INFO,
                        f"predicate '{name}' targets vocabulary '{v}', absent here, but the dataset "
                        f"contains '{sib}' (possible vocabulary version mismatch)",
                        predicate=name,
                        detail={"targeted": v, "present_sibling": sib},
                    )
                )

    # P1: the code spec matches zero codes (skip if P2 already explains the absence).
    if not codes and not all_vocabs_absent:
        is_error = True
        detail: dict[str, Any] = {
            "code_spec": _code_spec(cfg),
            "matched_codes": [],
            "nearest_present": _nearest_present(cfg, profile),
        }
        if isinstance(cfg.code, dict) and "any" in cfg.code:
            detail["unmatched_members"] = [m for m in cfg.code["any"] if m not in profile.codes]
        findings.append(
            Finding(
                "P1",
                ERROR,
                f"predicate '{name}' matched 0 codes in the dataset",
                predicate=name,
                detail=detail,
            )
        )

    if codes:
        present_vocabs |= {vocabulary_of(c) for c in codes}

        # Partial any-list: some members match nothing (likely typos / wrong vocabulary).
        if isinstance(cfg.code, dict) and "any" in cfg.code and isinstance(cfg.code["any"], list):
            unmatched = [m for m in cfg.code["any"] if m not in profile.codes]
            if unmatched:
                findings.append(
                    Finding(
                        "P1A",
                        WARNING,
                        f"predicate '{name}' uses an any-list whose member(s) {sorted(unmatched)} "
                        "match no code in the dataset",
                        predicate=name,
                        detail={"unmatched_members": sorted(unmatched)},
                    )
                )

        value_findings, value_error = _value_checks(name, cfg, profile, codes)
        findings.extend(value_findings)
        is_error = is_error or value_error
        findings.extend(_prevalence_check(name, profile, codes, min_prevalence))

    # P5: other_cols references a column absent from the data schema.
    if cfg.other_cols and profile.data_columns:
        for col in sorted(cfg.other_cols):
            if col not in profile.data_columns:
                is_error = True
                findings.append(
                    Finding(
                        "P5",
                        ERROR,
                        f"predicate '{name}' constrains column '{col}', which is not present in the "
                        "dataset schema",
                        predicate=name,
                        detail={"missing_column": col, "available_columns": sorted(profile.data_columns)},
                    )
                )

    return findings, is_error, present_vocabs


def _value_checks(
    name: str, cfg: PlainPredicateConfig, profile: DatasetProfile, codes: list[str]
) -> tuple[list[Finding], bool]:
    if (cfg.value_min is None and cfg.value_max is None) or not profile.data_scanned:
        return [], False

    cps = [profile.codes[c] for c in codes]
    total_numeric = sum((cp.n_numeric or 0) for cp in cps)

    # P3: value constraint set but no matched code carries a numeric value -> never satisfiable.
    if total_numeric == 0:
        return (
            [
                Finding(
                    "P3",
                    ERROR,
                    f"predicate '{name}' has a value constraint but none of its matched codes carry "
                    "a numeric_value, so the constraint can never be satisfied",
                    predicate=name,
                    detail={"matched_codes": codes, "value_min": cfg.value_min, "value_max": cfg.value_max},
                )
            ],
            True,
        )

    # P4: the constraint window lies entirely outside the observed value range (unit mismatch?).
    obs_mins = [cp.value_min for cp in cps if cp.value_min is not None]
    obs_maxs = [cp.value_max for cp in cps if cp.value_max is not None]
    if obs_mins and obs_maxs:
        obs_min, obs_max = min(obs_mins), max(obs_maxs)
        if _window_outside(cfg, obs_min, obs_max):
            return (
                [
                    Finding(
                        "P4",
                        WARNING,
                        f"predicate '{name}'s value constraint lies entirely outside the observed "
                        f"value range [{obs_min:g}, {obs_max:g}] (possible unit mismatch)",
                        predicate=name,
                        detail={
                            "value_min": cfg.value_min,
                            "value_max": cfg.value_max,
                            "observed_min": obs_min,
                            "observed_max": obs_max,
                        },
                    )
                ],
                False,
            )
    return [], False


def _prevalence_check(
    name: str, profile: DatasetProfile, codes: list[str], min_prevalence: float
) -> list[Finding]:
    if not profile.data_scanned:
        return []
    cps = [profile.codes[c] for c in codes]
    subjects = max((cp.n_subjects or 0) for cp in cps)
    events = sum((cp.n_events or 0) for cp in cps)
    if subjects < min_prevalence:
        note = " (subject count is for the most prevalent matched code)" if len(codes) > 1 else ""
        return [
            Finding(
                "P6",
                WARNING,
                f"predicate '{name}' matches codes that are present but rare: at most "
                f"{subjects} subject(s) and {events} event(s), below the floor of {min_prevalence}" + note,
                predicate=name,
                detail={"subjects": subjects, "events": events, "matched_codes": codes},
            )
        ]
    return []


# ---------------------------------------------------------------------------- #
# Inverse coverage (4.3)
# ---------------------------------------------------------------------------- #
def _inverse_coverage(
    profile: DatasetProfile,
    matched: dict[str, list[str]],
    targeted_present_vocabs: set[str],
    top_orphans: int,
) -> list[Finding]:
    findings: list[Finding] = []
    matched_all: set[str] = set()
    for codes in matched.values():
        matched_all.update(codes)

    for vocab in sorted(v for v in targeted_present_vocabs if v != LOCAL_VOCAB):
        candidates = [
            cp
            for cp in profile.codes.values()
            if cp.vocabulary == vocab and cp.code not in matched_all and (cp.n_subjects or 0) > 0
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda cp: (-(cp.n_subjects or 0), cp.code))
        top = candidates[:top_orphans]
        findings.append(
            Finding(
                "ORPHAN",
                INFO,
                f"vocabulary '{vocab}' has {len(candidates)} present code(s) that no predicate "
                f"matches; the {len(top)} most prevalent are listed for review",
                detail={
                    "vocabulary": vocab,
                    "n_unmatched": len(candidates),
                    "top_unmatched": [
                        {"code": cp.code, "n_subjects": cp.n_subjects, "description": cp.description}
                        for cp in top
                    ],
                },
            )
        )

    # Best-effort ancestry extension: present, unmatched descendants of matched codes.
    descendants = [
        cp
        for cp in profile.codes.values()
        if cp.code not in matched_all
        and cp.vocabulary in targeted_present_vocabs
        and any(parent in matched_all for parent in cp.parent_codes)
    ]
    if descendants:
        descendants.sort(key=lambda cp: (-(cp.n_subjects or 0), cp.code))
        findings.append(
            Finding(
                "ORPHAN_ANCESTRY",
                INFO,
                f"{len(descendants)} present code(s) are ontological descendants of a matched code "
                "but are themselves unmatched",
                detail={
                    "codes": [cp.code for cp in descendants[:top_orphans]],
                },
            )
        )
    return findings


def _coarse_timestamp_check(task_cfg: TaskExtractorConfig, profile: DatasetProfile) -> list[Finding]:
    if profile.time_granularity != "day":
        return []
    has_subday = False
    for w in (task_cfg.windows or {}).values():
        for expr in (
            getattr(w, "start_endpoint_expr", None),
            getattr(w, "end_endpoint_expr", None),
        ):
            size = getattr(expr, "window_size", None)
            if isinstance(size, timedelta) and timedelta(0) < abs(size) < timedelta(days=1):
                has_subday = True
    if not has_subday:
        return []
    return [
        Finding(
            "T1",
            INFO,
            "dataset timestamps are day-level, but the config defines sub-day windows whose temporal "
            "logic may be unsatisfiable on this data",
        )
    ]


# ---------------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------------- #
def _target_vocabularies(cfg: PlainPredicateConfig) -> set[str] | None:
    """Vocabularies a predicate's code spec targets; ``None`` for regex (cannot be inferred)."""
    if isinstance(cfg.code, dict):
        if "any" in cfg.code and isinstance(cfg.code["any"], list):
            return {vocabulary_of(c) for c in cfg.code["any"]}
        return None
    return {vocabulary_of(cfg.code)}


def _code_spec(cfg: PlainPredicateConfig) -> Any:
    return cfg.code


def _nearest_present(cfg: PlainPredicateConfig, profile: DatasetProfile) -> list[str]:
    codes = list(profile.codes)
    if isinstance(cfg.code, dict):
        if "any" in cfg.code and isinstance(cfg.code["any"], list):
            out: list[str] = []
            for member in cfg.code["any"]:
                out.extend(difflib.get_close_matches(member, codes, n=2, cutoff=0.6))
            return sorted(set(out))[:5]
        return []
    return difflib.get_close_matches(cfg.code, codes, n=5, cutoff=0.5)


def _version_sibling(vocab: str, dataset_vocabs: dict[str, int]) -> str | None:
    stem = re.sub(r"\d+", "", vocab)
    if not stem:
        return None
    for v in sorted(dataset_vocabs):
        if v != vocab and re.sub(r"\d+", "", v) == stem:
            return v
    return None


def _window_outside(cfg: PlainPredicateConfig, obs_min: float, obs_max: float) -> bool:
    """Whether the predicate's [value_min, value_max] window lies entirely outside [obs_min, obs_max]."""
    below = cfg.value_max is not None and (
        cfg.value_max < obs_min or (cfg.value_max == obs_min and not cfg.value_max_inclusive)
    )
    above = cfg.value_min is not None and (
        cfg.value_min > obs_max or (cfg.value_min == obs_max and not cfg.value_min_inclusive)
    )
    return bool(below or above)


def _dataset_header(profile: DatasetProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "version": profile.version,
        "n_subjects": profile.n_subjects,
        "n_codes": len(profile.codes),
        "vocabularies": [{"name": v, "n_codes": n} for v, n in sorted(profile.vocabularies.items())],
        "time_granularity": profile.time_granularity,
        "data_scanned": profile.data_scanned,
        "codes_metadata_complete": profile.codes_metadata_complete,
        "warnings": profile.warnings,
    }


def _read_raw_predicates(
    config_path: Path, predicates_path: str | Path | None
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """Read raw (predicates, demographics, trigger, windows) from the config YAML (mirrors loader)."""
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    loaded = yaml.load(Path(config_path).read_text())
    predicates = dict(loaded.get("predicates", {}) or {})
    demographics = dict(loaded.get("patient_demographics", {}) or {})
    trigger = loaded.get("trigger")
    windows = dict(loaded.get("windows", {}) or {})

    if predicates_path is not None:
        override = yaml.load(Path(predicates_path).read_text())
        predicates.update(override.get("predicates", {}) or {})
        demographics.update(override.get("patient_demographics", {}) or {})

    return predicates, demographics, trigger, windows


def _build_predicate_objs(
    predicates_raw: dict[str, Any], demographics_raw: dict[str, Any]
) -> dict[str, PlainPredicateConfig | DerivedPredicateConfig]:
    """Construct predicate objects for *all* defined predicates (mirrors the loader's parsing).

    Unlike :meth:`TaskExtractorConfig.load`, this does not prune unreferenced predicates, so the audit
    can lint predicates that are defined but not yet wired into any window. Predicates that fail to
    construct are skipped (the stricter loader surfaces those as input errors).
    """
    objs: dict[str, PlainPredicateConfig | DerivedPredicateConfig] = {}

    for name, p in predicates_raw.items():
        if not isinstance(p, dict):
            continue  # malformed (e.g. bare string) -- loader-surfaced concern
        try:
            if "expr" in p:
                fields = {k: v for k, v in p.items() if k in DerivedPredicateConfig.__dataclass_fields__}
                objs[name] = DerivedPredicateConfig(**fields)
            else:
                config_data = {
                    k: v
                    for k, v in p.items()
                    if k in PlainPredicateConfig.__dataclass_fields__ and k != "other_cols"
                }
                other_cols = {k: v for k, v in p.items() if k not in config_data}
                objs[name] = PlainPredicateConfig(**config_data, other_cols=other_cols)
        except (TypeError, ValueError):
            continue

    for name, p in demographics_raw.items():
        if not isinstance(p, dict):
            continue
        try:
            config_data = {
                k: v
                for k, v in p.items()
                if k in PlainPredicateConfig.__dataclass_fields__ and k not in ("other_cols", "static")
            }
            other_cols = {k: v for k, v in p.items() if k not in config_data and k != "static"}
            objs[name] = PlainPredicateConfig(**config_data, static=True, other_cols=other_cols)
        except (TypeError, ValueError):
            continue

    return objs


def _trigger_name(trigger_raw: Any) -> str | None:
    if isinstance(trigger_raw, dict):
        return trigger_raw.get("predicate")
    return trigger_raw
