"""Benchmark legacy vs compiled ACES engines on synthetic cohorts (time + peak memory).

Sweeps subject counts for one or more sample configs, runs each engine, and records wall
time and peak memory. Peak memory uses ``tracemalloc`` (Python allocations) and, when
available, ``psutil`` peak working set (whole-process RSS, which captures Polars' native
allocations -- the part that actually blows up on real data).

Until the compiled engine lands, ``--engines legacy`` runs the baseline alone. Results
are written to ``benchmarks/results/<timestamp>.csv``; pass ``--plot`` to also emit PNGs.

Example (baseline only, small sweep)::

    uv run python benchmarks/bench.py --configs inhospital_mortality \\
        --sizes 1000,5000 --engines legacy --out benchmarks/results
"""

from __future__ import annotations

import argparse
import gc
import threading
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import polars as pl

from aces.config import TaskExtractorConfig
from aces.query import query

try:  # works whether benchmarks/ is on sys.path (script run) or imported as a package (pytest)
    from generate import generate_predicates_df
except ImportError:  # pragma: no cover
    from benchmarks.generate import generate_predicates_df

try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover - psutil is an optional extra
    psutil = None
    _PROC = None

REPO_ROOT = Path(__file__).resolve().parent.parent


def _engine_fn(name: str) -> Callable[[TaskExtractorConfig, pl.DataFrame], pl.DataFrame]:
    if name == "legacy":
        return query
    if name == "compiled":
        from aces.lazy_query import lazy_query

        return lazy_query
    raise ValueError(f"Unknown engine '{name}'.")


def _rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6 if _PROC is not None else float("nan")


class _RSSSampler:
    """Background thread sampling process RSS to capture the peak delta over a run.

    tracemalloc only sees Python allocations; the memory that actually explodes in ACES
    is Polars' native arena. Sampling whole-process RSS during the call and reporting the
    peak rise above the pre-run baseline isolates each run's native footprint.
    """

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.baseline = _rss_mb()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _RSSSampler:
        if _PROC is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self.peak = max(self.peak, _rss_mb())

    @property
    def peak_delta_mb(self) -> float:
        return self.peak - self.baseline if _PROC is not None else float("nan")


def time_engine(
    engine: str,
    cfg: TaskExtractorConfig,
    predicates_df: pl.DataFrame,
    repeats: int = 1,
) -> dict[str, float]:
    """Run ``engine`` on ``predicates_df`` and return timing + memory metrics.

    ``rss_delta_mb`` is the peak whole-process RSS rise during the call (native + Python);
    ``py_peak_mb`` is the tracemalloc Python-only peak. We report the *min* time and *max*
    memory across repeats (best-case speed, worst-case footprint).
    """
    fn = _engine_fn(engine)
    best_time = float("inf")
    n_rows = 0
    py_peak_mb = 0.0
    rss_delta_mb = 0.0
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        with _RSSSampler() as sampler:
            t0 = time.perf_counter()
            result = fn(cfg, predicates_df)
            elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        best_time = min(best_time, elapsed)
        py_peak_mb = max(py_peak_mb, peak / 1e6)
        rss_delta_mb = max(rss_delta_mb, sampler.peak_delta_mb)
        n_rows = result.height
    return {
        "seconds": best_time,
        "py_peak_mb": py_peak_mb,
        "rss_delta_mb": rss_delta_mb,
        "result_rows": float(n_rows),
    }


def run_sweep(
    configs: list[str],
    sizes: list[int],
    engines: list[str],
    events_per_subject: int = 50,
    seed: int = 0,
    repeats: int = 1,
) -> pl.DataFrame:
    rows = []
    for config_name in configs:
        cfg = TaskExtractorConfig.load(str(REPO_ROOT / "sample_configs" / f"{config_name}.yaml"))
        for n_subjects in sizes:
            predicates_df = generate_predicates_df(cfg, n_subjects, events_per_subject, seed)
            for engine in engines:
                metrics = time_engine(engine, cfg, predicates_df, repeats=repeats)
                row = {
                    "config": config_name,
                    "engine": engine,
                    "n_subjects": n_subjects,
                    "n_pred_rows": predicates_df.height,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{config_name:22s} {engine:8s} N={n_subjects:>8,} "
                    f"{metrics['seconds']:8.3f}s  rss_delta={metrics['rss_delta_mb']:8.1f}MB  "
                    f"py_peak={metrics['py_peak_mb']:7.1f}MB  rows={int(metrics['result_rows'])}"
                )
    return pl.DataFrame(rows)


def _plot(df: pl.DataFrame, out_dir: Path) -> None:  # pragma: no cover - plotting is side-effecting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel in [("seconds", "wall time (s)"), ("rss_delta_mb", "peak RSS delta (MB)")]:
        for config_name, sub in df.group_by("config"):
            fig, ax = plt.subplots()
            for engine, ser in sub.group_by("engine"):
                ser = ser.sort("n_subjects")
                ax.plot(ser["n_subjects"], ser[metric], marker="o", label=engine[0])
            ax.set_xlabel("subjects")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{config_name[0]} — {ylabel}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{config_name[0]}_{metric}.png", dpi=120)
            plt.close(fig)


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="inhospital_mortality", help="Comma-separated config stems.")
    parser.add_argument("--sizes", default="1000,5000,10000", help="Comma-separated subject counts.")
    parser.add_argument("--engines", default="legacy", help="Comma-separated: legacy,compiled.")
    parser.add_argument("--events-per-subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "results"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--tag", default="", help="Optional filename tag for the CSV/plots.")
    args = parser.parse_args()

    df = run_sweep(
        configs=args.configs.split(","),
        sizes=[int(s) for s in args.sizes.split(",")],
        engines=args.engines.split(","),
        events_per_subject=args.events_per_subject,
        seed=args.seed,
        repeats=args.repeats,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"bench{tag}.csv"
    df.write_csv(csv_path)
    print(f"\nWrote {csv_path}")
    if args.plot:
        _plot(df, out_dir)
        print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
