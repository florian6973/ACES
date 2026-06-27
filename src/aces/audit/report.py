"""Result objects for the ACES static audit: severities, findings, and the report.

A single internal :class:`AuditReport` renders to both a machine-readable JSON document and a concise
human-readable terminal summary. JSON output is deterministic (findings sorted, keys sorted, no
timestamps) so that identical inputs produce byte-identical output, making the tool usable as a CI
gate.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
from typing import Any

AUDIT_VERSION = "0.1.0"


class Severity(enum.Enum):
    """Finding severity tiers, ordered most- to least-severe."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        return {"ERROR": 0, "WARNING": 1, "INFO": 2}[self.value]


@dataclasses.dataclass(frozen=True)
class Finding:
    """A single audit finding.

    Attributes:
        id: The check identifier (e.g. ``"P1"``, ``"D1"``, ``"C2"``).
        severity: One of :class:`Severity`.
        message: A human-readable, non-judgemental description of what cannot happen on this dataset.
        predicate: The predicate the finding pertains to, if any.
        detail: Structured, JSON-serialisable supporting data.
    """

    id: str
    severity: Severity
    message: str
    predicate: str | None = None
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def _sort_key(self) -> tuple:
        return (self.severity.rank, self.predicate or "", self.id, self.message)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "severity": self.severity.value}
        if self.predicate is not None:
            d["predicate"] = self.predicate
        d["message"] = self.message
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclasses.dataclass
class AuditReport:
    """The complete result of auditing one config against one dataset profile."""

    dataset: dict[str, Any]
    config: dict[str, Any]
    findings: list[Finding] = dataclasses.field(default_factory=list)

    @property
    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: f._sort_key())

    @property
    def summary(self) -> dict[str, int]:
        out = {"error": 0, "warning": 0, "info": 0}
        for f in self.findings:
            out[f.severity.value.lower()] += 1
        return out

    @property
    def has_errors(self) -> bool:
        return any(f.severity is Severity.ERROR for f in self.findings)

    def exit_code(self) -> int:
        """0 if no ERROR findings, 1 if at least one ERROR finding."""
        return 1 if self.has_errors else 0

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "aces_audit_version": AUDIT_VERSION,
            "dataset": self.dataset,
            "config": self.config,
            "findings": [f.to_dict() for f in self.sorted_findings],
            "summary": self.summary,
        }

    def to_json(self) -> str:
        """Render the report as deterministic, pretty-printed JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def to_text(self, *, color: bool | None = None) -> str:
        """Render a concise human-readable report, grouped by severity then predicate."""
        if color is None:
            color = sys.stdout.isatty()

        # ASCII-only markers: the report must print on consoles that cannot encode Unicode (e.g.
        # Windows cp1252), since it is intended to run as a CI gate.
        markers = {
            Severity.ERROR: ("x", "31"),  # red
            Severity.WARNING: ("!", "33"),  # yellow
            Severity.INFO: ("-", "36"),  # cyan
        }

        def paint(text: str, code: str) -> str:
            return f"\033[{code}m{text}\033[0m" if color else text

        lines: list[str] = []
        ds = self.dataset
        lines.append(paint("ACES static audit", "1"))
        vocabs = ", ".join(f"{v['name']}({v['n_codes']})" for v in ds.get("vocabularies", [])[:8])
        lines.append(
            f"  dataset: {ds.get('name') or '<unnamed>'}"
            + (f" v{ds['version']}" if ds.get("version") else "")
        )
        lines.append(
            f"  subjects: {ds.get('n_subjects', 'n/a')}"
            f" | codes: {ds.get('n_codes', 'n/a')}"
            f" | time granularity: {ds.get('time_granularity', 'n/a')}"
            f" | data scanned: {ds.get('data_scanned')}"
        )
        if vocabs:
            lines.append(f"  vocabularies: {vocabs}")
        lines.append(
            f"  config: {self.config.get('path')}"
            f" ({self.config.get('n_predicates', 0)} predicates,"
            f" trigger='{self.config.get('trigger_predicate')}')"
        )
        for w in ds.get("warnings", []):
            lines.append(paint(f"  ! {w}", "33"))
        lines.append("")

        if not self.findings:
            lines.append(paint("  No findings. OK", "32"))
        else:
            for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
                group = [f for f in self.sorted_findings if f.severity is severity]
                if not group:
                    continue
                mark, col = markers[severity]
                lines.append(paint(f"{severity.value} ({len(group)})", col))
                for f in group:
                    where = f" [{f.predicate}]" if f.predicate else ""
                    lines.append(paint(f"  {mark} {f.id}{where}: {f.message}", col))
                    detail = self._format_detail(f.detail)
                    if detail:
                        lines.append(f"      {detail}")
                lines.append("")

        s = self.summary
        lines.append(
            paint(
                f"Summary: {s['error']} error, {s['warning']} warning, {s['info']} info",
                "1",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _format_detail(detail: dict[str, Any]) -> str:
        if not detail:
            return ""
        parts = []
        for k, v in detail.items():
            if isinstance(v, list):
                shown = ", ".join(str(x) for x in v[:5])
                if len(v) > 5:
                    shown += f", ...(+{len(v) - 5})"
                parts.append(f"{k}=[{shown}]")
            else:
                parts.append(f"{k}={v}")
        return "; ".join(parts)
