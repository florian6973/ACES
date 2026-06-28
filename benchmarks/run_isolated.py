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
from generate import generate_predicates_df  # benchmarks/ is on sys.path when run as a script

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["legacy", "compiled", "compiled_mem"]


def _run_one(config: str, parquet: Path, engine: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_run_one.py"),
         "--config", config, "--parquet", str(parquet), "--engine", engine],
        capture_output=True, text=True, cwd=str(REPO_ROOT), check=True,
    )
    # _run_one prints a single JSON line (last non-empty stdout line).
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def run(configs: list[str], sizes: list[int], events_per_subject: int, seed: int) -> pl.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory(prefix="aces_isolated_") as tmp:
        for config in configs:
            cfg = TaskExtractorConfig.load(str(REPO_ROOT / "sample_configs" / f"{config}.yaml"))
            for n in sizes:
                df = generate_predicates_df(cfg, n, events_per_subject, seed)
                parquet = Path(tmp) / f"{config}_{n}.parquet"
                df.write_parquet(parquet)
                for engine in ENGINES:
                    m = _run_one(config, parquet, engine)
                    rows.append({"config": config, "n_subjects": n, "n_pred_rows": df.height, **m})
                    print(
                        f"{config:22s} {engine:13s} N={n:>8,}  "
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
