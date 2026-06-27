"""Command-line entrypoint for the ACES static audit (``aces-audit``).

Lints an ACES task config against a MEDS dataset before any extraction is run. Exit codes:

* ``0`` -- no ERROR findings (WARN/INFO allowed);
* ``1`` -- at least one ERROR finding;
* ``2`` -- tool/input failure (config unparseable, dataset path invalid, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .checks import AuditInputError, _dataset_header, run_audit
from .profile import DatasetProfile
from .report import AuditReport


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aces-audit",
        description="Statically audit an ACES task config against a MEDS dataset before extraction.",
    )
    p.add_argument("--config", required=True, help="Path to the ACES task config YAML.")
    p.add_argument("--meds", required=True, help="Path to the MEDS dataset (root dir or .parquet).")
    p.add_argument(
        "--predicates",
        default=None,
        help="Optional predicates-override YAML (mirrors ACES's predicates_path).",
    )
    p.add_argument(
        "--no-data-scan",
        action="store_true",
        help="Lint against the code inventory only; skip the prevalence/value data scan.",
    )
    p.add_argument(
        "--min-prevalence",
        type=float,
        default=1,
        help="Subject-count floor below which a matched predicate is flagged (default: 1).",
    )
    p.add_argument(
        "--top-orphans",
        type=int,
        default=25,
        help="Number of high-prevalence unmatched codes to surface per vocabulary (default: 25).",
    )
    p.add_argument("--json", default=None, help="Write the machine-readable report to this path.")
    p.add_argument(
        "--profile-only",
        action="store_true",
        help="Only build and print the dataset profile; do not lint a config.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress the human-readable report on stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the audit CLI. Returns the process exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = _build_parser().parse_args(argv)

    try:
        if args.profile_only:
            report = _profile_only_report(args)
        else:
            report = run_audit(
                config_path=args.config,
                meds_path=args.meds,
                scan_data=not args.no_data_scan,
                min_prevalence=args.min_prevalence,
                top_orphans=args.top_orphans,
                predicates_path=args.predicates,
            )
    except AuditInputError as e:
        print(f"aces-audit: input error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"aces-audit: {e}", file=sys.stderr)
        return 2

    if args.json:
        Path(args.json).write_text(report.to_json())
    if not args.quiet:
        print(report.to_text())

    return report.exit_code()


def _profile_only_report(args: argparse.Namespace) -> AuditReport:
    profile = DatasetProfile.from_meds(args.meds, scan_data=not args.no_data_scan)
    return AuditReport(
        dataset=_dataset_header(profile),
        config={"path": None, "n_predicates": 0, "trigger_predicate": None},
        findings=[],
    )


def cli() -> None:  # pragma: no cover - thin console-script wrapper
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    cli()
