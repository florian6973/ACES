"""Static profiling of a MEDS dataset's code / vocabulary space.

This module builds a :class:`DatasetProfile` -- a reusable, config-agnostic inventory of the codes,
vocabularies, value typing, and (optionally) prevalence statistics present in a MEDS dataset. The
profile is computed once and then consumed by the audit linter (:mod:`aces.audit.checks`); it is
deliberately decoupled from the audit's reporting so that other tooling (e.g. assisted-predicate
authoring) can import and reuse it independently.

The code inventory and vocabulary set come from ``metadata/codes.parquet`` with no data scan. Only the
prevalence / value statistics require scanning the data shards, which is gated behind ``scan_data``.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import meds
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime

    from aces.config import PlainPredicateConfig

logger = logging.getLogger(__name__)

#: Vocabulary namespace reported for codes that do not follow the ``VOCABULARY//CODE`` convention.
LOCAL_VOCAB = "<local>"

#: MEDS data columns the profiler reasons about (string literals mirror ``meds.DataSchema``).
_SUBJECT_ID = meds.DataSchema.subject_id_name
_TIME = meds.DataSchema.time_name
_CODE = meds.DataSchema.code_name
_NUMERIC_VALUE = meds.DataSchema.numeric_value_name

#: MEDS code-metadata columns.
_DESCRIPTION = meds.CodeMetadataSchema.description_name
_PARENT_CODES = meds.CodeMetadataSchema.parent_codes_name


def vocabulary_of(code: str) -> str:
    """Return the vocabulary namespace of a code: the prefix before the first ``//``.

    Codes without a ``//`` are treated as local / un-namespaced.

    Examples:
        >>> vocabulary_of("ICD10CM//I21.4")
        'ICD10CM'
        >>> vocabulary_of("LAB//HR")
        'LAB'
        >>> vocabulary_of("ADMISSION")
        '<local>'
        >>> vocabulary_of("diagnosis//ICD9CM_41071")
        'diagnosis'
    """
    return code.split("//", 1)[0] if "//" in code else LOCAL_VOCAB


@dataclasses.dataclass
class CodeProfile:
    """Per-code inventory and (optional) data-scan statistics.

    The ``n_*`` / ``value_*`` / ``prevalence`` fields are ``None`` when the data scan was skipped.
    """

    code: str
    vocabulary: str
    description: str | None = None
    parent_codes: list[str] = dataclasses.field(default_factory=list)

    # Populated only when the data is scanned.
    n_events: int | None = None
    n_subjects: int | None = None
    prevalence: float | None = None
    n_numeric: int | None = None
    frac_numeric: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    value_p1: float | None = None
    value_p50: float | None = None
    value_p99: float | None = None

    @property
    def has_numeric(self) -> bool | None:
        """Whether any event for this code carried a ``numeric_value`` (``None`` if not scanned)."""
        if self.n_numeric is None:
            return None
        return self.n_numeric > 0


@dataclasses.dataclass
class DatasetProfile:
    """A static profile of a MEDS dataset's code / vocabulary space.

    Attributes:
        codes: Mapping of code string -> :class:`CodeProfile`.
        vocabularies: Mapping of vocabulary namespace -> number of codes in it.
        data_columns: The set of column names present in the data shards (used to validate
            ``other_cols`` references even when the full data scan is skipped).
        data_scanned: Whether prevalence / value statistics were computed.
        codes_metadata_complete: ``False`` when the inventory was derived from a data scan because no
            ``metadata/codes.parquet`` was available (the all-codes guarantee may not hold).
    """

    codes: dict[str, CodeProfile]
    vocabularies: dict[str, int]
    data_columns: set[str]
    data_scanned: bool
    codes_metadata_complete: bool
    name: str | None = None
    version: str | None = None
    n_subjects: int | None = None
    time_granularity: str | None = None
    has_static_rows: bool | None = None
    time_span: tuple[datetime, datetime] | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_meds(cls, path: str | Path, *, scan_data: bool = True) -> DatasetProfile:
        """Build a :class:`DatasetProfile` from a MEDS dataset.

        Args:
            path: Path to a MEDS dataset root (containing ``data/`` and optionally ``metadata/``),
                or to a single ``.parquet`` data shard.
            scan_data: When ``True``, scan the data shards for prevalence / value / timestamp
                statistics. When ``False``, only the code inventory and data schema are read (fast).

        Returns:
            The constructed profile.

        Raises:
            FileNotFoundError: If no data shards or code metadata can be located at ``path``.
        """
        path = Path(path)
        data_files, codes_path, dataset_json = _resolve_meds_paths(path)
        warnings: list[str] = []

        name, version = _read_dataset_metadata(dataset_json)

        # Data column inventory (cheap: schema only, no scan) -- needed for other_cols checks.
        data_columns: set[str] = set()
        if data_files:
            try:
                data_columns = set(pl.scan_parquet(data_files[0]).collect_schema().names())
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not read data schema from {data_files[0]}: {e}")

        # Code inventory: metadata-first, fall back to a (minimal) data scan.
        if codes_path is not None:
            inventory = _read_codes_metadata(codes_path)
            codes_metadata_complete = True
        elif data_files:
            inventory = _codes_from_data(data_files)
            codes_metadata_complete = False
            warnings.append(
                "No metadata/codes.parquet found; the code inventory was derived from the data "
                "shards and may be incomplete if the dataset was transformed after extraction."
            )
        else:
            raise FileNotFoundError(f"No MEDS data shards or code metadata found at: {path}")

        profile = cls(
            codes=inventory,
            vocabularies=_count_vocabularies(inventory),
            data_columns=data_columns,
            data_scanned=False,
            codes_metadata_complete=codes_metadata_complete,
            name=name,
            version=version,
            warnings=warnings,
        )

        if scan_data and data_files:
            profile._scan_data(data_files)

        return profile

    def _scan_data(self, data_files: list[Path]) -> None:
        """Populate prevalence / value / timestamp statistics from a lazy scan of the shards."""
        lf = pl.scan_parquet(data_files).with_columns(pl.col(_CODE).cast(pl.String))

        n_subjects = lf.select(pl.col(_SUBJECT_ID).n_unique()).collect().item()
        self.n_subjects = int(n_subjects) if n_subjects is not None else None

        per_code = (
            lf.group_by(_CODE)
            .agg(
                pl.len().alias("n_events"),
                pl.col(_SUBJECT_ID).n_unique().alias("n_subjects"),
                pl.col(_NUMERIC_VALUE).is_not_null().sum().alias("n_numeric"),
                pl.col(_NUMERIC_VALUE).min().alias("value_min"),
                pl.col(_NUMERIC_VALUE).max().alias("value_max"),
                pl.col(_NUMERIC_VALUE).quantile(0.01).alias("value_p1"),
                pl.col(_NUMERIC_VALUE).quantile(0.50).alias("value_p50"),
                pl.col(_NUMERIC_VALUE).quantile(0.99).alias("value_p99"),
            )
            .collect()
        )

        for row in per_code.iter_rows(named=True):
            code = row[_CODE]
            cp = self.codes.get(code)
            if cp is None:
                # Code present in data but not in the metadata inventory: add it.
                cp = CodeProfile(code=code, vocabulary=vocabulary_of(code))
                self.codes[code] = cp
            cp.n_events = int(row["n_events"])
            cp.n_subjects = int(row["n_subjects"])
            cp.n_numeric = int(row["n_numeric"])
            cp.frac_numeric = (cp.n_numeric / cp.n_events) if cp.n_events else 0.0
            cp.prevalence = (cp.n_subjects / self.n_subjects) if self.n_subjects else None
            for f in ("value_min", "value_max", "value_p1", "value_p50", "value_p99"):
                v = row[f]
                setattr(cp, f, float(v) if v is not None else None)

        # Codes in the inventory but never observed in the data get zeroed stats.
        observed = set(per_code[_CODE].to_list())
        for code, cp in self.codes.items():
            if code not in observed:
                cp.n_events = 0
                cp.n_subjects = 0
                cp.n_numeric = 0
                cp.frac_numeric = 0.0
                cp.prevalence = 0.0

        self.vocabularies = _count_vocabularies(self.codes)
        self._scan_timestamps(lf)
        self.data_scanned = True

    def _scan_timestamps(self, lf: pl.LazyFrame) -> None:
        """Determine timestamp granularity, presence of static rows, and the observed time span."""
        t = pl.col(_TIME)
        stats = lf.select(
            t.min().alias("tmin"),
            t.max().alias("tmax"),
            t.is_null().any().alias("has_static"),
            (t.dt.truncate("1d") != t).any().alias("sub_day"),
            (t.dt.truncate("1m") != t).any().alias("sub_minute"),
            t.is_not_null().any().alias("any_time"),
        ).collect()
        row = stats.row(0, named=True)

        self.has_static_rows = bool(row["has_static"]) if row["has_static"] is not None else None
        if row["sub_minute"]:
            self.time_granularity = "second"
        elif row["sub_day"]:
            self.time_granularity = "minute"
        elif row["any_time"]:
            self.time_granularity = "day"
        else:
            self.time_granularity = None
        if row["tmin"] is not None and row["tmax"] is not None:
            self.time_span = (row["tmin"], row["tmax"])

    # ------------------------------------------------------------------ #
    # Query helpers (reused by the linter)
    # ------------------------------------------------------------------ #
    def matched_codes(self, predicate: PlainPredicateConfig) -> list[str]:
        """Return the sorted codes in the inventory whose membership matches ``predicate``'s code spec.

        This reuses :meth:`aces.config.PlainPredicateConfig.code_matching_expr`, so the audit's notion
        of "which codes a predicate matches" is identical to extraction's by construction.
        """
        codes = list(self.codes.keys())
        if not codes:
            return []
        frame = pl.DataFrame({_CODE: codes})
        matched = frame.filter(predicate.code_matching_expr())[_CODE].to_list()
        return sorted(matched)


def _resolve_meds_paths(path: Path) -> tuple[list[Path], Path | None, Path | None]:
    """Resolve a MEDS path into (data shard files, codes.parquet | None, dataset.json | None)."""
    if path.is_file():
        return [path], None, None

    if not path.is_dir():
        raise FileNotFoundError(f"MEDS path does not exist: {path}")

    # Data shards: prefer the canonical data/ subdirectory, else any parquet outside metadata/.
    data_dir = path / "data"
    if data_dir.is_dir():
        data_files = sorted(data_dir.glob("**/*.parquet"))
    else:
        metadata_dir = path / "metadata"
        data_files = sorted(p for p in path.glob("**/*.parquet") if metadata_dir not in p.parents)

    codes_path = path / meds.code_metadata_filepath
    dataset_json = path / meds.dataset_metadata_filepath
    return (
        data_files,
        codes_path if codes_path.is_file() else None,
        dataset_json if dataset_json.is_file() else None,
    )


def _read_dataset_metadata(dataset_json: Path | None) -> tuple[str | None, str | None]:
    """Read the dataset name / version from ``metadata/dataset.json`` if present."""
    if dataset_json is None:
        return None, None
    import json

    try:
        meta = json.loads(dataset_json.read_text())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not parse {dataset_json}: {e}")
        return None, None
    return (
        meta.get(meds.DatasetMetadataSchema.dataset_name_name),
        meta.get(meds.DatasetMetadataSchema.dataset_version_name),
    )


def _read_codes_metadata(codes_path: Path) -> dict[str, CodeProfile]:
    """Read the code inventory (code, description, parent_codes) from ``metadata/codes.parquet``."""
    df = pl.read_parquet(codes_path)
    have_desc = _DESCRIPTION in df.columns
    have_parents = _PARENT_CODES in df.columns

    inventory: dict[str, CodeProfile] = {}
    for row in df.iter_rows(named=True):
        code = row[_CODE]
        if code is None:
            continue
        code = str(code)
        parents = row[_PARENT_CODES] if have_parents else None
        inventory[code] = CodeProfile(
            code=code,
            vocabulary=vocabulary_of(code),
            description=row[_DESCRIPTION] if have_desc else None,
            parent_codes=list(parents) if parents else [],
        )
    return inventory


def _codes_from_data(data_files: list[Path]) -> dict[str, CodeProfile]:
    """Derive the code inventory from a minimal scan of the data shards (fallback path)."""
    codes = (
        pl.scan_parquet(data_files).select(pl.col(_CODE).cast(pl.String)).unique().collect()[_CODE].to_list()
    )
    return {c: CodeProfile(code=c, vocabulary=vocabulary_of(c)) for c in codes if c is not None}


def _count_vocabularies(inventory: dict[str, CodeProfile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cp in inventory.values():
        counts[cp.vocabulary] = counts.get(cp.vocabulary, 0) + 1
    return counts
