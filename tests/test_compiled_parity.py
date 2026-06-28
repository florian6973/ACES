"""Differential parity tests: the legacy engine is the spec for the compiled engine.

For every bundled sample config (and a few targeted inline configs), we generate a small
deterministic synthetic cohort, run both ``query`` (legacy) and ``lazy_query`` (compiled),
and assert the results are equal after a canonical sort.

While the compiled engine is unimplemented these xfail on ``NotImplementedError``; they
flip to passing (XPASS) as phases 2-3 land, at which point the markers are removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from aces.config import TaskExtractorConfig
from aces.query import query

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CONFIGS = sorted((REPO_ROOT / "sample_configs").glob("*.yaml"))

# Make ``benchmarks/generate.py`` importable without packaging it.
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.generate import generate_predicates_df  # noqa: E402


def _canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Sort rows/cols into a canonical order so row-order differences don't matter."""
    if df.is_empty():
        return df
    sort_keys = [c for c in ("subject_id", "trigger", "index_timestamp", "label") if c in df.columns]
    return df.sort(sort_keys).select(sorted(df.columns))


def assert_engines_match(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> None:
    from aces.lazy_query import lazy_query

    legacy = query(cfg, predicates_df)
    compiled = lazy_query(cfg, predicates_df)

    # Both-empty is a valid agreement (e.g. no valid cohort rows).
    if legacy.is_empty() and compiled.is_empty():
        return

    assert_frame_equal(_canonical(legacy), _canonical(compiled), check_dtypes=True)


@pytest.mark.parametrize("config_path", SAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_sample_config_parity(config_path: Path) -> None:
    cfg = TaskExtractorConfig.load(str(config_path))
    predicates_df = generate_predicates_df(cfg, n_subjects=60, events_per_subject=40, seed=7)
    assert_engines_match(cfg, predicates_df)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inhospital_mortality_multiseed(seed: int) -> None:
    cfg = TaskExtractorConfig.load(str(REPO_ROOT / "sample_configs" / "inhospital_mortality.yaml"))
    predicates_df = generate_predicates_df(cfg, n_subjects=40, events_per_subject=50, seed=seed)
    assert_engines_match(cfg, predicates_df)


def test_generator_is_deterministic() -> None:
    """A guard that the oracle itself is reproducible (independent of the compiled engine)."""
    cfg = TaskExtractorConfig.load(str(REPO_ROOT / "sample_configs" / "inhospital_mortality.yaml"))
    a = generate_predicates_df(cfg, n_subjects=20, events_per_subject=30, seed=3)
    b = generate_predicates_df(cfg, n_subjects=20, events_per_subject=30, seed=3)
    assert_frame_equal(a, b)
