"""Run a single (config, engine) extraction in an isolated process and report metrics.

Isolation is the only reliable way to attribute *peak* process memory to one engine:
an in-process sampler is fooled because earlier allocations (data generation) leave the
working set already grown, so later work reuses committed pages and shows ~0 delta.

Reads a pre-generated predicates parquet and prints a JSON line with wall time, peak
working-set MB, and result rows. The compiled engine scans the parquet lazily (so the
streaming engine can show its memory profile); the legacy engine must materialize it.

Usage::

    python benchmarks/_run_one.py --config <stem> --parquet <path> --engine legacy|compiled|compiled_mem
"""

from __future__ import annotations

import argparse
import json
import time

import polars as pl

from aces.compile import compile_query
from aces.config import TaskExtractorConfig
from aces.query import query


def _peak_wset_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", getattr(info, "rss", 0)) / 1e6
    except Exception:
        return float("nan")


def _load_cfg(kind: str, config: str, n_windows: int) -> TaskExtractorConfig:
    if kind == "sample":
        return TaskExtractorConfig.load(f"sample_configs/{config}.yaml")
    from complex_configs import BUILDERS

    return BUILDERS[kind](n_windows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="sample", choices=["sample", "chain", "wide"])
    parser.add_argument("--config", default="", help="sample config stem (kind=sample)")
    parser.add_argument("--n-windows", type=int, default=1, help="window count (kind=chain|wide)")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--engine", required=True, choices=["legacy", "compiled", "compiled_mem"])
    args = parser.parse_args()

    cfg = _load_cfg(args.kind, args.config, args.n_windows)

    t0 = time.perf_counter()
    if args.engine == "legacy":
        predicates_df = pl.read_parquet(args.parquet)
        result = query(cfg, predicates_df)
    else:
        plan = compile_query(cfg)
        engine = "streaming" if args.engine == "compiled" else "in-memory"
        result = plan(pl.scan_parquet(args.parquet)).collect(engine=engine)
    elapsed = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "engine": args.engine,
                "seconds": elapsed,
                "peak_wset_mb": _peak_wset_mb(),
                "result_rows": result.height,
            }
        )
    )


if __name__ == "__main__":
    main()
