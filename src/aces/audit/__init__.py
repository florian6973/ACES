"""Static pre-flight audit for ACES task configs over MEDS datasets.

Profiles a MEDS dataset's code / vocabulary space and lints an ACES config's predicates against it,
flagging predicates that reference codes, vocabularies, value constraints, or columns that do not
exist in the dataset -- before any extraction is run.

Public API:
    >>> from aces.audit import DatasetProfile, audit_config, run_audit, AuditReport
"""

from __future__ import annotations

from .checks import AuditInputError, audit_config, lint_derived_references, run_audit
from .profile import CodeProfile, DatasetProfile, vocabulary_of
from .report import AUDIT_VERSION, AuditReport, Finding, Severity

__all__ = [
    "AUDIT_VERSION",
    "AuditInputError",
    "AuditReport",
    "CodeProfile",
    "DatasetProfile",
    "Finding",
    "Severity",
    "audit_config",
    "lint_derived_references",
    "run_audit",
    "vocabulary_of",
]
