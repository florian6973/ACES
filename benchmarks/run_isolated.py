"""Orchestrate isolated-process benchmarks across configs/sizes/engines.

For each (config, size) this generates a predicates parquet once, then runs every engine
in a *fresh subprocess* (via ``_run_one.py``) so peak working-set memory is attributable
to that engine alone. Results are written to ``benchmarks/results/isolated.csv`` and, with
``--plot``, to per-config time/memory PNGs.

Example::

    uv run python benchmarks/run_isolated.py --configs inhospital_mortality \\
        --sizes 20000,80000,200000 --plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

from aces.config import TaskExtractorConfig
from complex_configs import BUILDERS  # benchmarks/ on sys.path when run as a script
from generate import generate_predicates_df

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["legacy", "compiled", "compiled_mem"]


def _parse_spec(spec: str) -> tuple[str, str, int, str]:
    """'inhospital_mortality' -> sample; 'chain:16' / 'wide:8' -> complex builder."""
    if ":" in spec:
        kind, n = spec.split(":")
        return kind, "", int(n), f"{kind}{n}"
    return "sample", spec, 1, spec


def _load_cfg(kind: str, config: str, n_windows: int) -> TaskExtractorConfig:
    if kind == "sample":
        return TaskExtractorConfig.load(str(REPO_ROOT / "sample_configs" / f"{config}.yaml"))
    return BUILDERS[kind](n_windows)


def _run_one(kind: str, config: str, n_windows: int, parquet: Path, engine: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_run_one.py"),
         "--kind", kind, "--config", config, "--n-windows", str(n_windows),
         "--parquet", str(parquet), "--engine", engine],
        capture_output=True, text=True, cwd=str(REPO_ROOT), check=True,
    )
    # _run_one prints a single JSON line (last non-empty stdout line).
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def run(specs: list[str], sizes: list[int], events_per_subject: int, seed: int) -> pl.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory(prefix="aces_isolated_") as tmp:
        for spec in specs:
            kind, config, n_windows, label = _parse_spec(spec)
            cfg = _load_cfg(kind, config, n_windows)
            for n in sizes:
                df = generate_predicates_df(cfg, n, events_per_subject, seed)
                parquet = Path(tmp) / f"{label}_{n}.parquet"
                df.write_parquet(parquet)
                for engine in ENGINES:
                    m = _run_one(kind, config, n_windows, parquet, engine)
                    rows.append({"config": label, "n_subjects": n, "n_pred_rows": df.height, **m})
                    print(
                        f"{label:22s} {engine:13s} N={n:>8,}  "
                        f"{m['seconds']:8.3f}s  peak={m['peak_wset_mb']:9.1f}MB  rows={m['result_rows']}"
                    )
    return pl.DataFrame(rows)


def _plot(df: pl.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel in [("seconds", "wall time (s)"), ("peak_wset_mb", "peak memory (MB)")]:
        for (config,), sub in df.group_by("config"):
            fig, ax = plt.subplots()
            for (engine,), ser in sub.group_by("engine"):
                ser = ser.sort("n_subjects")
                ax.plot(ser["n_subjects"], ser[metric], marker="o", label=engine)
            ax.set_xlabel("subjects")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{config} — {ylabel} (legacy vs compiled)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{config}_{metric}.png", dpi=120)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="inhospital_mortality")
    parser.add_argument("--sizes", default="20000,80000,200000")
    parser.add_argument("--events-per-subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "results"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = run(
        args.configs.split(","),
        [int(s) for s in args.sizes.split(",")],
        args.events_per_subject,
        args.seed,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_dir / "isolated.csv")
    print(f"\nWrote {out_dir / 'isolated.csv'}")
    if args.plot:
        _plot(df, out_dir)
        print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
