"""Pooling policy: strict YAML loading, threshold resolution, and the decision engine.

Decision states (spec §9): ``COMPATIBLE``, ``REVIEW``, ``INCOMPATIBLE``,
``INSUFFICIENT_EVIDENCE``, plus ``NOT_ASSESSED`` for optional checks that were not
requested. Aggregation precedence, applied to the four assessed components:

1. Any hard incompatibility (any component ``FAIL``) -> ``INCOMPATIBLE``.
2. Else any missing required evidence (any component ``MISSING_EVIDENCE`` or
   ``INSUFFICIENT_OBSERVATIONS``) -> ``INSUFFICIENT_EVIDENCE``.
3. Else any actionable warning (any component ``REVIEW``) -> ``REVIEW``.
4. Else ``COMPATIBLE``.

Optional, unrequested checks never block compatibility. A structural validation
failure never reaches this engine — it aborts the run (exit code 2) beforehand.

Shipped example thresholds are ILLUSTRATIVE AND UNCALIBRATED. They are never
described as validated; calibration guidance is in the README.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import yaml

from seqconsistency.comparison import (
    STATUS_EXCEED,
    STATUS_INSUFFICIENT_OBSERVATIONS,
    STATUS_MISSING_EVIDENCE,
    STATUS_NO_THRESHOLD,
    STATUS_NOT_ASSESSED,
    STATUS_PASS,
    ThresholdResolver,
    ThresholdRule,
)
from seqconsistency.errors import ConfigError
from seqconsistency.models import (
    CheckStatus,
    DatasetOutcome,
    DecisionState,
    IdentityEvidenceLevel,
    InputFormat,
    Severity,
)

POLICY_SCHEMA_VERSION = "1"

_ACTION_VALUES = ("review", "incompatible")
_HETERO_VALUES = ("allowed", "review", "incompatible")

#: The exact sentence the spec requires whenever identity rests on metadata only.
IDENTITY_DECLARED_SENTENCE = (
    "Sample identity accepted from supplied metadata; not independently verified."
)


@dataclass(frozen=True)
class ThresholdConfig:
    metric: str
    method: str
    max: float
    action: str
    calibrated: bool
    profiles: tuple[str, ...] | None = None
    strata: tuple[str, ...] | None = None
    assays: tuple[str, ...] | None = None


@dataclass
class Policy:
    policy_id: str
    policy_version: str
    description: str
    accept_declared_sample_id: bool
    require_independent_verification: bool
    same_sample_id_required: bool
    same_assay_required: bool
    distinct_library_ids: str
    distinct_processing_ids: str
    reference_mismatch_action: str
    require_fastp_evidence: bool
    require_thresholds: bool
    min_observations: int
    thresholds: list[ThresholdConfig]
    raw_text: str
    sha256: str

    def resolver_for(self, profile: str, assay: str) -> ThresholdResolver:
        """Build the (metric, stratum) -> ThresholdRule resolver for one stratum scope."""

        def resolve(metric: str, stratum: str) -> ThresholdRule | None:
            for t in self.thresholds:
                if t.metric != metric:
                    continue
                if t.profiles is not None and profile not in t.profiles:
                    continue
                if t.strata is not None and stratum not in t.strata:
                    continue
                if t.assays is not None and assay not in t.assays:
                    continue
                return ThresholdRule(t.max, t.action, t.calibrated)
            return None

        return resolve


def _require_keys(mapping: dict, required: set[str], allowed: set[str], where: str) -> None:
    missing = required - set(mapping)
    if missing:
        raise ConfigError(f"policy {where}: missing required key(s) {sorted(missing)}")
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(
            f"policy {where}: unknown key(s) {sorted(unknown)} — unknown keys are "
            "rejected, never silently ignored"
        )


def _enum(value: Any, allowed: tuple[str, ...], where: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(f"policy {where}: expected one of {list(allowed)}, got {value!r}")
    return value


def _strict_bool(value: Any, where: str) -> bool:
    """Accept only real YAML booleans. ``bool("false")`` would silently be True."""
    if not isinstance(value, bool):
        raise ConfigError(f"policy {where}: expected a boolean (true/false), got {value!r}")
    return value


def _str_list(value: Any, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"policy {where}: expected a list of strings, got {value!r}")
    return tuple(value)


def _selectors_intersect(a: tuple[str, ...] | None, b: tuple[str, ...] | None) -> bool:
    """None selects everything; otherwise the two tuples must share a value."""
    if a is None or b is None:
        return True
    return bool(set(a) & set(b))


def _rules_conflict(existing: ThresholdConfig, new: ThresholdConfig) -> bool:
    """Same metric+method whose profile/stratum/assay scopes can both match at once.

    Repeating one metric across disjoint profile scopes (e.g. per-profile
    read_length thresholds) is legitimate; an overlap would make rule order
    silently decide which threshold applies.
    """
    return (
        _selectors_intersect(existing.profiles, new.profiles)
        and _selectors_intersect(existing.strata, new.strata)
        and _selectors_intersect(existing.assays, new.assays)
    )


def _looks_like_yaml_text(text: str) -> bool:
    return "\n" in text or text.lstrip().startswith("{") or ":" in text.split("\n", 1)[0]


def load_policy(path_or_text: Any) -> Policy:
    """Load and strictly validate a policy from a path (str/Path) or YAML text."""
    if isinstance(path_or_text, str) and _looks_like_yaml_text(path_or_text):
        raw_text = path_or_text
    elif isinstance(path_or_text, str):
        raw_text = _read_file(path_or_text)
    elif hasattr(path_or_text, "read_text"):
        raw_text = path_or_text.read_text(encoding="utf-8")
    else:
        raw_text = path_or_text
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"policy is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("policy root must be a mapping")

    _require_keys(
        data,
        {"policy_id", "policy_version", "identity", "metadata", "technical"},
        {"policy_id", "policy_version", "description", "identity", "metadata", "technical"},
        "(root)",
    )
    identity = data["identity"]
    metadata = data["metadata"]
    technical = data["technical"]
    for section, name in ((identity, "identity"), (metadata, "metadata"), (technical, "technical")):
        if not isinstance(section, dict):
            raise ConfigError(f"policy section {name!r} must be a mapping")

    _require_keys(
        identity,
        set(),
        {"accept_declared_sample_id", "require_independent_verification"},
        "identity",
    )
    _require_keys(
        metadata,
        set(),
        {
            "same_sample_id_required",
            "same_assay_required",
            "distinct_library_ids",
            "distinct_processing_ids",
            "reference_mismatch",
            "require_fastp_evidence",
        },
        "metadata",
    )
    _require_keys(
        technical,
        {"thresholds", "min_observations"},
        {"thresholds", "min_observations", "require_thresholds"},
        "technical",
    )

    thresholds_raw = technical["thresholds"]
    if not isinstance(thresholds_raw, list):
        raise ConfigError("policy technical.thresholds must be a list")
    from seqconsistency.comparison import KNOWN_METRIC_METHODS

    thresholds: list[ThresholdConfig] = []
    for i, t in enumerate(thresholds_raw):
        where = f"technical.thresholds[{i}]"
        if not isinstance(t, dict):
            raise ConfigError(f"policy {where}: each threshold must be a mapping")
        _require_keys(
            t,
            {"metric", "method", "max", "action", "calibrated"},
            {"metric", "method", "max", "action", "calibrated", "profiles", "strata", "assays"},
            where,
        )
        metric = t["metric"]
        method = t["method"]
        if not isinstance(metric, str) or not isinstance(method, str):
            raise ConfigError(f"policy {where}: 'metric' and 'method' must be strings")
        if (metric, method) not in KNOWN_METRIC_METHODS:
            known = ", ".join(sorted(f"{m}/{meth}" for m, meth in KNOWN_METRIC_METHODS))
            raise ConfigError(
                f"policy {where}: unknown metric/method pair {metric!r}/{method!r}; "
                f"this version implements: {known}"
            )
        max_val = t["max"]
        if (
            not isinstance(max_val, (int, float))
            or isinstance(max_val, bool)
            or not math.isfinite(float(max_val))
            or max_val < 0
        ):
            raise ConfigError(f"policy {where}: 'max' must be a finite non-negative number")
        new_rule = ThresholdConfig(
            metric=metric,
            method=method,
            max=float(max_val),
            action=_enum(t["action"], _ACTION_VALUES, where),
            calibrated=_strict_bool(t["calibrated"], f"{where}.calibrated"),
            profiles=_str_list(t.get("profiles"), f"{where}.profiles"),
            strata=_str_list(t.get("strata"), f"{where}.strata"),
            assays=_str_list(t.get("assays"), f"{where}.assays"),
        )
        for existing in thresholds:
            if (
                existing.metric == new_rule.metric
                and existing.method == new_rule.method
                and _rules_conflict(existing, new_rule)
            ):
                raise ConfigError(
                    f"policy {where}: duplicate threshold rule for metric "
                    f"{metric!r} (method {method!r}) with overlapping "
                    "profile/stratum/assay scopes — merge the rules; repeating a "
                    "metric across disjoint scopes is fine"
                )
        thresholds.append(new_rule)

    min_obs = technical["min_observations"]
    if not isinstance(min_obs, int) or isinstance(min_obs, bool) or min_obs < 0:
        raise ConfigError("policy technical.min_observations must be a non-negative int")

    return Policy(
        policy_id=str(data["policy_id"]),
        policy_version=str(data["policy_version"]),
        description=str(data.get("description", "")),
        accept_declared_sample_id=_strict_bool(
            identity.get("accept_declared_sample_id", True), "identity.accept_declared_sample_id"
        ),
        require_independent_verification=_strict_bool(
            identity.get("require_independent_verification", False),
            "identity.require_independent_verification",
        ),
        same_sample_id_required=_strict_bool(
            metadata.get("same_sample_id_required", True),
            "metadata.same_sample_id_required",
        ),
        same_assay_required=_strict_bool(
            metadata.get("same_assay_required", True), "metadata.same_assay_required"
        ),
        distinct_library_ids=_enum(
            metadata.get("distinct_library_ids", "review"),
            _HETERO_VALUES,
            "metadata.distinct_library_ids",
        ),
        distinct_processing_ids=_enum(
            metadata.get("distinct_processing_ids", "review"),
            _HETERO_VALUES,
            "metadata.distinct_processing_ids",
        ),
        reference_mismatch_action=_enum(
            metadata.get("reference_mismatch", "incompatible"),
            ("review", "incompatible"),
            "metadata.reference_mismatch",
        ),
        require_fastp_evidence=_strict_bool(
            metadata.get("require_fastp_evidence", False),
            "metadata.require_fastp_evidence",
        ),
        require_thresholds=_strict_bool(
            technical.get("require_thresholds", True), "technical.require_thresholds"
        ),
        min_observations=min_obs,
        thresholds=thresholds,
        raw_text=raw_text if isinstance(raw_text, str) else str(raw_text),
        sha256=hashlib.sha256(
            (raw_text if isinstance(raw_text, str) else "").encode("utf-8")
        ).hexdigest(),
    )


def _read_file(path: str) -> str:
    from pathlib import Path

    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read policy file: {exc}") from exc


# --------------------------------------------------------------------------
# Findings and decisions
# --------------------------------------------------------------------------


@dataclass
class Finding:
    finding_id: str
    pool_group: str
    dataset_id: str | None
    affected_datasets: list[str]
    component: str
    severity: Severity
    reason_code: str
    message: str
    evidence_metric: str | None
    evidence_value: float | None
    recommended_action: str

    def to_row(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "pool_group": self.pool_group,
            "dataset_id": self.dataset_id,
            "affected_datasets": self.affected_datasets,
            "component": self.component,
            "severity": self.severity.value,
            "reason_code": self.reason_code,
            "message": self.message,
            "evidence_metric": self.evidence_metric,
            "evidence_value": self.evidence_value,
            "recommended_action": self.recommended_action,
        }


@dataclass
class GroupDecision:
    pool_group: str
    structural_status: CheckStatus
    technical_status: CheckStatus
    metadata_status: CheckStatus
    identity_status: CheckStatus
    overall_decision: DecisionState
    identity_evidence_level: IdentityEvidenceLevel
    assumptions: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_row(self, policy: Policy) -> dict[str, Any]:
        return {
            "pool_group": self.pool_group,
            "structural_status": self.structural_status.value,
            "technical_status": self.technical_status.value,
            "metadata_status": self.metadata_status.value,
            "identity_status": self.identity_status.value,
            "overall_decision": self.overall_decision.value,
            "identity_evidence_level": self.identity_evidence_level.value,
            "assumptions": self.assumptions,
            "reason_codes": self.reason_codes,
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
        }


def _metadata_findings(
    outcomes: list[DatasetOutcome], policy: Policy
) -> tuple[CheckStatus, list[Finding]]:
    """Metadata compatibility across one pooling group."""
    specs = [o.spec for o in outcomes]
    ids = [s.dataset_id for s in specs]
    group = specs[0].pool_group
    status = CheckStatus.PASS
    findings: list[Finding] = []

    def finding(
        reason: str,
        message: str,
        severity: Severity,
        action: str,
        metric: str | None = None,
        value: float | None = None,
    ) -> None:
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=ids,
                component="metadata",
                severity=severity,
                reason_code=reason,
                message=message,
                evidence_metric=metric,
                evidence_value=value,
                recommended_action=action,
            )
        )

    def distinct(values: Iterable[str]) -> list[str]:
        return sorted(set(values))

    def single_finding(
        reason: str,
        message: str,
        severity: Severity,
        action: str,
        affected: list[str],
    ) -> None:
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=affected,
                component="metadata",
                severity=severity,
                reason_code=reason,
                message=message,
                evidence_metric=None,
                evidence_value=None,
                recommended_action=action,
            )
        )

    # Ambiguous quality encoding (review round 4): the classic Phred+64
    # signature is valid under BOTH encodings, so the scan flagged it instead
    # of guessing a rejection; ambiguity withholds automatic approval (REVIEW)
    # but never blocks the report.
    for o in outcomes:
        if o.fastq is None:
            continue
        strands = [o.fastq.r1] + ([o.fastq.r2] if o.fastq.r2 is not None else [])
        if not any(s.encoding_ambiguous for s in strands):
            continue
        if status == CheckStatus.PASS:
            status = CheckStatus.REVIEW
        single_finding(
            "QUALITY_ENCODING_AMBIGUOUS",
            f"Dataset {o.spec.dataset_id!r} FASTQ carries the classic Phred+64 "
            "signature (every quality byte >= 64, some >= 90), which is valid "
            "under BOTH encodings (Sanger FASTQ spans Q0-93, ASCII 33-126). "
            "Metrics are computed under Phred+33, v1's declared reading; ASCII "
            "ranges cannot distinguish the encodings, so this group cannot be "
            "automatically approved.",
            Severity.WARNING,
            "Verify the file's true encoding at its source: a Phred+64 file "
            "must be re-exported as Phred+33 (its qualities would otherwise "
            "read shifted by +31). Never modify the data to pass validation.",
            [o.spec.dataset_id],
        )

    samples = distinct(s.sample_id for s in specs)
    if len(samples) > 1:
        status = CheckStatus.FAIL
        finding(
            "GROUP_DIFFERENT_SAMPLE_IDS",
            f"Pooling group {group!r} declares different biological samples: "
            f"{samples}. Pooling across biological samples is outside the supported "
            "technical-pooling use case.",
            Severity.INCOMPATIBLE,
            "Split the datasets into per-sample pooling groups, or correct the "
            "manifest sample_id values.",
        )
    assays = distinct(s.assay for s in specs)
    if len(assays) > 1:
        status = CheckStatus.FAIL
        finding(
            "GROUP_DIFFERENT_ASSAYS",
            f"Pooling group {group!r} mixes assays {assays}; different assays have "
            "different technical distributions and pooling semantics.",
            Severity.INCOMPATIBLE,
            "Pool only datasets sharing one assay.",
        )
    libraries = distinct(s.library_id for s in specs)
    if len(libraries) > 1:
        action = policy.distinct_library_ids
        msg = (
            f"Pooling group {group!r} combines different libraries: {libraries}. "
            "Different libraries from one sample can be poolable; this is a "
            "policy judgement, not automatic compatibility."
        )
        if action == "incompatible":
            status = CheckStatus.FAIL
            finding(
                "GROUP_DIFFERENT_LIBRARY_IDS",
                msg,
                Severity.INCOMPATIBLE,
                "Set metadata.distinct_library_ids if pooling across libraries is "
                "intended, or split the group.",
            )
        elif action == "review":
            if status == CheckStatus.PASS:
                status = CheckStatus.REVIEW
            finding(
                "GROUP_DIFFERENT_LIBRARY_IDS",
                msg,
                Severity.WARNING,
                "Confirm the libraries share the biological sample and intended "
                "processing, then accept the REVIEW or adjust the policy.",
            )
        else:
            finding(
                "GROUP_DIFFERENT_LIBRARY_IDS",
                msg,
                Severity.INFO,
                "No action required; policy marks distinct libraries allowed.",
            )
    platforms = distinct(s.platform for s in specs)
    if len(platforms) > 1:
        if status == CheckStatus.PASS:
            status = CheckStatus.REVIEW
        finding(
            "GROUP_DIFFERENT_PLATFORMS",
            f"Pooling group {group!r} mixes platforms {platforms}. Even when "
            "syntactically concatenable, cross-platform data is not homogeneous.",
            Severity.WARNING,
            "Keep platforms in separate pooling groups unless the downstream "
            "analysis explicitly models platform effects.",
        )
    processings = distinct(s.processing_id or "" for s in specs)
    if len(processings) > 1 or "" in processings:
        action = policy.distinct_processing_ids
        null_processing = any(s.processing_id is None for s in specs)
        if null_processing:
            if status == CheckStatus.PASS:
                status = CheckStatus.REVIEW
            finding(
                "METADATA_PROCESSING_UNKNOWN",
                f"Pooling group {group!r} has datasets without processing_id; "
                "preprocessing history cannot be compared for them.",
                Severity.WARNING,
                "Supply processing_id for every dataset or confirm identical preprocessing.",
            )
        elif len(processings) > 1:
            if action == "incompatible":
                status = CheckStatus.FAIL
                finding(
                    "GROUP_DIFFERENT_PROCESSING_IDS",
                    f"Pooling group {group!r} mixes preprocessing settings "
                    f"{processings}; policy marks this incompatible.",
                    Severity.INCOMPATIBLE,
                    "Split the group by processing_id.",
                )
            elif action == "review":
                if status == CheckStatus.PASS:
                    status = CheckStatus.REVIEW
                finding(
                    "GROUP_DIFFERENT_PROCESSING_IDS",
                    f"Pooling group {group!r} mixes preprocessing settings {processings}.",
                    Severity.WARNING,
                    "Confirm the preprocessing differences are acceptable for the "
                    "downstream analysis.",
                )
            else:
                finding(
                    "GROUP_DIFFERENT_PROCESSING_IDS",
                    f"Pooling group {group!r} mixes preprocessing settings "
                    f"{processings}; policy marks this allowed.",
                    Severity.INFO,
                    "No action required.",
                )
    reference_ids = distinct(s.reference_id or "" for s in specs if s.reference_id is not None)
    if len(reference_ids) > 1 and policy.reference_mismatch_action == "incompatible":
        status = CheckStatus.FAIL
        finding(
            "GROUP_DIFFERENT_REFERENCES",
            f"Pooling group {group!r} declares different reference_id values "
            f"{sorted(set(reference_ids))}.",
            Severity.INCOMPATIBLE,
            "Pool only datasets aligned to the same reference.",
        )
    return status, findings


def _reference_dictionary_findings(
    outcomes: list[DatasetOutcome], policy: Policy
) -> tuple[CheckStatus, list[Finding]]:
    """Compare BAM reference dictionaries and read groups within one group."""
    bams = [o for o in outcomes if o.bam is not None and o.spec.input_format is InputFormat.BAM]
    group = outcomes[0].spec.pool_group
    ids = [o.spec.dataset_id for o in outcomes]
    findings: list[Finding] = []
    status = CheckStatus.PASS

    def finding(
        reason: str,
        message: str,
        severity: Severity,
        action: str,
        metric: str | None = None,
        value: float | None = None,
    ) -> None:
        nonlocal status
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=ids,
                component="metadata",
                severity=severity,
                reason_code=reason,
                message=message,
                evidence_metric=metric,
                evidence_value=value,
                recommended_action=action,
            )
        )

    if bams:
        first_bam = bams[0].bam
        assert first_bam is not None
        base = first_bam.header.references
        for other in bams[1:]:
            other_bam = other.bam
            assert other_bam is not None
            refs = other_bam.header.references
            names_a = [r[0] for r in base]
            names_b = [r[0] for r in refs]
            if names_a != names_b or [r[1] for r in base] != [r[1] for r in refs]:
                status = CheckStatus.FAIL
                finding(
                    "BAM_REFERENCE_DICTIONARY_MISMATCH",
                    f"Reference dictionaries differ between datasets "
                    f"{bams[0].spec.dataset_id!r} and {other.spec.dataset_id!r} "
                    "(names, lengths, or order). Coordinate-sorted BAMs must share "
                    "one reference dictionary to merge.",
                    Severity.INCOMPATIBLE,
                    "Re-align all datasets to one reference, or split the group.",
                )
        # Checksum conflicts must not depend on manifest order: collect every
        # available M5 per contig across the whole group first, then compare.
        # Comparing each BAM only against the first one hides a conflict between
        # later BAMs whenever the first carries no M5 at all.
        md5_values: dict[str, dict[str, list[str]]] = {}
        missing_md5: dict[str, list[str]] = {}
        for o in bams:
            o_bam = o.bam
            assert o_bam is not None
            for name, _length, md5 in o_bam.header.references:
                if md5:
                    md5_values.setdefault(name, {}).setdefault(md5, []).append(
                        o.spec.dataset_id
                    )
                else:
                    missing_md5.setdefault(name, []).append(o.spec.dataset_id)
        conflicting = [name for name in sorted(md5_values) if len(md5_values[name]) > 1]
        if conflicting:
            status = CheckStatus.FAIL
            detail = "; ".join(
                f"{name}: "
                + " vs ".join(
                    f"{md5} ({', '.join(sorted(ids))})"
                    for md5, ids in sorted(md5_values[name].items())
                )
                for name in conflicting
            )
            finding(
                "BAM_REFERENCE_CHECKSUM_CONFLICT",
                f"Reference checksums (M5) conflict for contigs {conflicting} "
                f"across datasets in group {group!r} ({detail}).",
                Severity.INCOMPATIBLE,
                "Use identical reference FASTA files for all datasets.",
            )
        elif len(bams) > 1 and missing_md5:
            if status == CheckStatus.PASS:
                status = CheckStatus.REVIEW
            absent = ", ".join(
                f"{name} (missing in {', '.join(sorted(set(ids)))})"
                for name, ids in sorted(missing_md5.items())
            )
            finding(
                "BAM_REFERENCE_IDENTITY_UNVERIFIED",
                "Reference names and lengths match but some M5 checksums are "
                f"absent across the group ({absent}); sequence identity is not "
                "fully verified (matching names and lengths do not prove identity).",
                Severity.WARNING,
                "Align from one reference FASTA or verify checksums if strict "
                "identity is required.",
            )
        # Read-group collisions: same RG id with different definitions.
        seen: dict[str, tuple] = {}
        for o in bams:
            o_bam = o.bam
            assert o_bam is not None
            for rg in o_bam.header.read_groups:
                rid = rg.get("ID", "")
                sig = (rg.get("SM"), rg.get("LB"), rg.get("PL"), rg.get("PU"))
                if rid in seen and seen[rid] != sig:
                    status = CheckStatus.FAIL
                    finding(
                        "BAM_READ_GROUP_COLLISION",
                        f"Read group ID {rid!r} is defined differently across "
                        f"datasets in group {group!r} "
                        f"({seen[rid]} vs {sig}).",
                        Severity.INCOMPATIBLE,
                        "Make read-group IDs unique per run/library or align their "
                        "definitions before merging.",
                    )
                else:
                    seen[rid] = sig
    return status, findings


def _bam_sample_metadata_findings(
    outcomes: list[DatasetOutcome],
) -> tuple[CheckStatus, CheckStatus, list[Finding]]:
    """Reconcile BAM header sample metadata with the manifest, per group.

    Returns (identity_status, metadata_status, findings). Header ``SM`` values are
    declared metadata: several distinct values across a group is a declared-identity
    conflict; one value contradicting the manifest ``sample_id`` is a metadata
    discrepancy a human must resolve. Records referencing undefined read-group IDs
    are counted during the scan and surfaced here.
    """
    group = outcomes[0].spec.pool_group
    bams = [o for o in outcomes if o.bam is not None and o.spec.input_format is InputFormat.BAM]
    findings: list[Finding] = []
    identity_status = CheckStatus.PASS
    metadata_status = CheckStatus.PASS
    if not bams:
        return identity_status, metadata_status, findings

    sms: dict[str, list[str]] = {}
    for o in bams:
        bam = o.bam
        assert bam is not None
        for rg in bam.header.read_groups:
            sm = rg.get("SM")
            if sm:
                sms.setdefault(str(sm), []).append(o.spec.dataset_id)
    if len(sms) > 1:
        identity_status = CheckStatus.FAIL
        detail = "; ".join(f"{sm!r} ({', '.join(sorted(ids))})" for sm, ids in sorted(sms.items()))
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=sorted({d for ids in sms.values() for d in ids}),
                component="identity",
                severity=Severity.INCOMPATIBLE,
                reason_code="BAM_SAMPLE_METADATA_CONFLICT",
                message=(
                    f"BAM headers in group {group!r} declare different samples (SM): {detail}."
                ),
                evidence_metric=None,
                evidence_value=None,
                recommended_action="Split the group by sample, or correct the SM tags.",
            )
        )
    elif len(sms) == 1:
        sm = next(iter(sms))
        declared = {o.spec.sample_id for o in outcomes}
        if len(declared) == 1 and sm not in declared:
            metadata_status = CheckStatus.REVIEW
            findings.append(
                Finding(
                    finding_id="",
                    pool_group=group,
                    dataset_id=None,
                    affected_datasets=sorted(sms[sm]),
                    component="metadata",
                    severity=Severity.WARNING,
                    reason_code="BAM_SM_MISMATCHES_MANIFEST",
                    message=(
                        f"BAM read groups declare sample SM={sm!r}, but the manifest "
                        f"declares sample_id {next(iter(declared))!r} for group "
                        f"{group!r}."
                    ),
                    evidence_metric=None,
                    evidence_value=None,
                    recommended_action="Decide which declaration is correct (manifest "
                    "or BAM header) and align them before pooling.",
                )
            )
    undefined_rg = [
        (o.spec.dataset_id, o_bam.flags.records_with_undefined_rg)
        for o in bams
        if (o_bam := o.bam) is not None and o_bam.flags.records_with_undefined_rg
    ]
    if undefined_rg:
        metadata_status = _worse(metadata_status, CheckStatus.REVIEW)
        detail = "; ".join(f"{d}: {n} record(s)" for d, n in undefined_rg)
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=[d for d, _ in undefined_rg],
                component="metadata",
                severity=Severity.WARNING,
                reason_code="BAM_UNDEFINED_RG_RECORDS",
                message=(
                    f"Records reference read-group IDs missing from the BAM header: {detail}."
                ),
                evidence_metric="records.undefined_rg",
                evidence_value=None,
                recommended_action="Fix the RG tags or header definitions before "
                "pooling; merge tools will drop or misattribute these records.",
            )
        )
    return identity_status, metadata_status, findings


def _identity_assessment(
    outcomes: list[DatasetOutcome], policy: Policy
) -> tuple[CheckStatus, IdentityEvidenceLevel, list[Finding]]:
    group = outcomes[0].spec.pool_group
    ids = [o.spec.dataset_id for o in outcomes]
    samples = sorted({o.spec.sample_id for o in outcomes})
    findings: list[Finding] = []

    def finding(reason: str, message: str, severity: Severity, action: str) -> None:
        findings.append(
            Finding(
                finding_id="",
                pool_group=group,
                dataset_id=None,
                affected_datasets=ids,
                component="identity",
                severity=severity,
                reason_code=reason,
                message=message,
                evidence_metric=None,
                evidence_value=None,
                recommended_action=action,
            )
        )

    if len(samples) > 1:
        return (
            CheckStatus.FAIL,
            IdentityEvidenceLevel.CONFLICTING,
            findings,
        )
    if policy.require_independent_verification:
        finding(
            "IDENTITY_VERIFICATION_NOT_IMPLEMENTED",
            "Policy requires independent sample-identity verification, but this "
            "seqconsistency version implements no genotype-based verification "
            "method (planned Phase 3). Identity was NOT verified.",
            Severity.WARNING,
            "Run an external verification tool (e.g. NGSCheckMate) or relax the "
            "policy to accept declared identity.",
        )
        return (
            CheckStatus.MISSING_EVIDENCE,
            IdentityEvidenceLevel.DECLARED_ONLY,
            findings,
        )
    if policy.accept_declared_sample_id:
        finding(
            "IDENTITY_DECLARED_NOT_VERIFIED",
            IDENTITY_DECLARED_SENTENCE + f" All datasets in group {group!r} declare "
            f"sample_id {samples[0]!r}.",
            Severity.INFO,
            "No action required; the declaration is trusted per policy.",
        )
        return CheckStatus.PASS, IdentityEvidenceLevel.DECLARED_ONLY, findings
    finding(
        "IDENTITY_EVIDENCE_POLICY_REFUSED",
        "All datasets declare the same sample_id, but the policy does not accept "
        "declared identity and no independent verification ran.",
        Severity.WARNING,
        "Set identity.accept_declared_sample_id true, or enable independent "
        "verification once implemented.",
    )
    return CheckStatus.MISSING_EVIDENCE, IdentityEvidenceLevel.DECLARED_ONLY, findings


class TechnicalAssessment:
    """Incremental technical assessment over streaming comparison rows.

    ``feed`` processes one batch of rows (typically one dataset pair) and returns
    that batch's findings, so the pipeline can write them out immediately — memory
    stays bounded by one batch instead of one group's full comparison table.
    ``finalize`` applies the group-level rules and returns the component status
    with any tail findings.
    """

    def __init__(self, policy: Policy, profile: str, assay: str) -> None:
        self._policy = policy
        self._profile = profile
        self._assay = assay
        self._resolver = policy.resolver_for(profile, assay)
        self.status = CheckStatus.PASS
        self._assessed = False  # any row compared against a configured threshold
        self._rows_seen = 0
        self._pool_group: str | None = None
        self._datasets: list[str] = []

    def feed(self, rows: Iterable[dict[str, Any]]) -> list[Finding]:
        """Assess one batch of comparison rows; returns this batch's findings."""
        batch: list[Finding] = []

        def finding(
            reason: str, message: str, severity: Severity, action: str, row: dict[str, Any]
        ) -> None:
            batch.append(
                Finding(
                    finding_id="",
                    pool_group=row["pool_group"],
                    dataset_id=None,
                    affected_datasets=[row["dataset_a"], row["dataset_b"]],
                    component="technical",
                    severity=severity,
                    reason_code=reason,
                    message=message,
                    evidence_metric=row["metric"],
                    evidence_value=row["value"],
                    recommended_action=action,
                )
            )

        for row in rows:
            self._rows_seen += 1
            if self._pool_group is None:
                self._pool_group = row["pool_group"]
            for ds in (row.get("dataset_a"), row.get("dataset_b")):
                if ds and ds not in self._datasets:
                    self._datasets.append(ds)
            if row.get("threshold") is not None:
                self._assessed = True
            if row["status"] == STATUS_NOT_ASSESSED:
                # Optional evidence the policy did not request: blocks nothing.
                continue
            if (
                row["status"] == STATUS_INSUFFICIENT_OBSERVATIONS
                or row["status"] == STATUS_MISSING_EVIDENCE
            ):
                self.status = _worse(self.status, CheckStatus.MISSING_EVIDENCE)
                # The decision table must carry WHY approval was blocked, not
                # only that it was (review round 3, point 5): emit a finding
                # with the comparison reason, observed counts, and the minimum.
                reason = row.get("reason_code") or "TECH_BELOW_MIN_OBSERVATIONS"
                na, nb = row.get("observations_a"), row.get("observations_b")
                if row["status"] == STATUS_INSUFFICIENT_OBSERVATIONS:
                    message = (
                        f"Metric {row.get('metric')!r} (stratum "
                        f"{row.get('stratum')}) between "
                        f"{row.get('dataset_a')!r} and {row.get('dataset_b')!r} "
                        f"has only {na}/{nb} observations; the policy minimum is "
                        f"{self._policy.min_observations}, so the distance was "
                        "not computed."
                    )
                    action = (
                        "Raise --sample-size, lower technical.min_observations, "
                        "or pool more reads; the metric cannot be assessed "
                        "without enough observations."
                    )
                else:
                    message = (
                        f"Comparison for metric {row.get('metric')!r} (stratum "
                        f"{row.get('stratum')}) between {row.get('dataset_a')!r} "
                        f"and {row.get('dataset_b')!r} lacks required evidence "
                        f"({reason})."
                    )
                    action = (
                        "Supply the missing evidence or disable its policy "
                        "requirement knowingly."
                    )
                finding(
                    reason,
                    message,
                    Severity.WARNING,
                    action,
                    row,
                )
            elif row["status"] == STATUS_EXCEED:
                rule = self._resolver(row["metric"], row["stratum"])
                action = rule.action if rule else "review"
                severity = Severity.INCOMPATIBLE if action == "incompatible" else Severity.WARNING
                candidate = CheckStatus.FAIL if action == "incompatible" else CheckStatus.REVIEW
                self.status = _worse(self.status, candidate)
                finding(
                    "TECH_THRESHOLD_EXCEEDED",
                    f"{row.get('metric')} distance {row['value']:.6g} between "
                    f"{row.get('dataset_a')!r} and {row.get('dataset_b')!r} exceeds "
                    f"the policy maximum {row['threshold']:.6g} "
                    f"({row.get('method')}, stratum {row.get('stratum')}).",
                    severity,
                    "Inspect the distributions; if the shift is expected, recalibrate "
                    "the threshold with genuine between-run data; otherwise do not pool.",
                    row,
                )
            elif row["status"] == STATUS_NO_THRESHOLD and self._policy.require_thresholds:
                self.status = _worse(self.status, CheckStatus.MISSING_EVIDENCE)
                finding(
                    "TECH_NO_THRESHOLD_CONFIGURED",
                    f"No policy threshold for metric {row.get('metric')!r} "
                    f"(stratum {row.get('stratum')}); with require_thresholds the "
                    "group cannot be COMPATIBLE on uncalibrated distances.",
                    Severity.WARNING,
                    "Add a calibrated threshold for this metric, or set "
                    "technical.require_thresholds false knowingly.",
                    row,
                )
            elif row["status"] == STATUS_NO_THRESHOLD and not self._policy.require_thresholds:
                finding(
                    "TECH_NO_THRESHOLD_CONFIGURED",
                    f"No policy threshold for metric {row.get('metric')!r} "
                    f"(stratum {row.get('stratum')}); the distance "
                    f"{row['value']:.6g} is reported but not assessed.",
                    Severity.INFO,
                    "Add a calibrated threshold before relying on this metric.",
                    row,
                )
            elif row["status"] == STATUS_PASS and row.get("threshold_calibrated") is False:
                # Spec §7: "If no threshold is configured or calibrated, report the
                # metric but do not use it to automatically approve pooling." A PASS
                # against an uncalibrated threshold is reported, never approving.
                # This is a REVIEW (assessment ran, verdict untrusted), distinct from
                # MISSING_EVIDENCE (no threshold at all).
                self.status = _worse(self.status, CheckStatus.REVIEW)
                finding(
                    "TECH_THRESHOLD_UNCALIBRATED",
                    f"Metric {row.get('metric')!r} (stratum {row.get('stratum')}) is "
                    f"within its maximum for {row.get('dataset_a')!r} and "
                    f"{row.get('dataset_b')!r}, but the threshold is marked "
                    "uncalibrated, so it cannot automatically approve pooling.",
                    Severity.WARNING,
                    "Calibrate thresholds with genuine between-run data, then set "
                    "calibrated: true; a human must accept this REVIEW otherwise.",
                    row,
                )
        return batch

    def finalize(self) -> tuple[CheckStatus, list[Finding]]:
        """Apply group-level rules; returns the component status and tail findings."""
        tail: list[Finding] = []
        if self._rows_seen and not self._assessed:
            # Spec §7: "If no threshold is configured or calibrated, report the
            # metric but do not use it to automatically approve pooling." When no
            # row was assessed against a threshold, nothing supports approval —
            # with or without require_thresholds. Distances are still reported.
            self.status = _worse(self.status, CheckStatus.MISSING_EVIDENCE)
            if not self._policy.require_thresholds:
                # The per-row branch above already warns when require_thresholds is
                # true; this finding covers the explicit opt-out mode, where the
                # distances remain informational and the decision stays
                # INSUFFICIENT_EVIDENCE.
                tail.append(
                    Finding(
                        finding_id="",
                        pool_group=self._pool_group or "",
                        dataset_id=None,
                        affected_datasets=self._datasets,
                        component="technical",
                        severity=Severity.WARNING,
                        reason_code="TECH_NO_THRESHOLD_CONFIGURED",
                        message=(
                            f"No comparison in this group was assessed against a "
                            f"policy threshold (profile {self._profile!r}, assay "
                            f"{self._assay!r}); distances are reported but none can "
                            "approve pooling (spec §7)."
                        ),
                        evidence_metric=None,
                        evidence_value=None,
                        recommended_action=(
                            "Add calibrated thresholds for automatic approval; with "
                            "require_thresholds: false the output stays a report and "
                            "the decision stays INSUFFICIENT_EVIDENCE."
                        ),
                    )
                )
        return self.status, tail


def _technical_assessment(
    comparison_rows: list[dict[str, Any]], policy: Policy, profile: str, assay: str
) -> tuple[CheckStatus, list[Finding]]:
    """Group-level technical status from comparison rows (batch convenience)."""
    assessment = TechnicalAssessment(policy, profile, assay)
    findings = assessment.feed(comparison_rows)
    status, tail = assessment.finalize()
    return status, findings + tail


_WORSE_ORDER = [
    CheckStatus.PASS,
    CheckStatus.REVIEW,
    CheckStatus.INSUFFICIENT_OBSERVATIONS,
    CheckStatus.MISSING_EVIDENCE,
    CheckStatus.FAIL,
]


def _worse(current: CheckStatus, candidate: CheckStatus) -> CheckStatus:
    if _WORSE_ORDER.index(candidate) > _WORSE_ORDER.index(current):
        return candidate
    return current


def group_components(
    outcomes: list[DatasetOutcome], policy: Policy
) -> tuple[CheckStatus, CheckStatus, IdentityEvidenceLevel, list[Finding]]:
    """Metadata and identity component statuses plus their findings (bounded by
    dataset count, independent of comparison volume)."""
    metadata_status, meta_findings = _metadata_findings(outcomes, policy)
    ref_status, ref_findings = _reference_dictionary_findings(outcomes, policy)
    metadata_status = _worse(metadata_status, ref_status)
    bam_id_status, bam_md_status, bam_findings = _bam_sample_metadata_findings(outcomes)
    metadata_status = _worse(metadata_status, bam_md_status)
    identity_status, identity_level, identity_findings = _identity_assessment(outcomes, policy)
    identity_status = _worse(identity_status, bam_id_status)
    if bam_id_status is CheckStatus.FAIL:
        identity_level = IdentityEvidenceLevel.CONFLICTING
    findings = meta_findings + ref_findings + bam_findings + identity_findings
    return metadata_status, identity_status, identity_level, findings


def finalize_group_decision(
    outcomes: list[DatasetOutcome],
    policy: Policy,
    metadata_status: CheckStatus,
    identity_status: CheckStatus,
    identity_level: IdentityEvidenceLevel,
    component_findings: list[Finding],
    technical_status: CheckStatus,
    tech_reason_codes: list[str],
) -> tuple[GroupDecision, list[Finding]]:
    """Assemble one group's decision from precomputed component results.

    The pipeline streams technical findings out as ``TechnicalAssessment.feed``
    produces them (memory stays bounded by one batch); it passes only the
    unique reason codes here. This function owns the fastp-evidence rule, the
    aggregation precedence, and the GroupDecision; it returns the decision and
    only the findings IT created (the fastp finding) for the caller to emit.
    """
    group = outcomes[0].spec.pool_group
    tail_findings: list[Finding] = []

    # fastp evidence requirement (merged_short).
    if outcomes[0].spec.read_type.value == "merged_short" and policy.require_fastp_evidence:
        missing = [o.spec.dataset_id for o in outcomes if o.fastp is None]
        if missing:
            technical_status = _worse(technical_status, CheckStatus.REVIEW)
            tail_findings.append(
                Finding(
                    finding_id="",
                    pool_group=group,
                    dataset_id=None,
                    affected_datasets=missing,
                    component="technical",
                    severity=Severity.WARNING,
                    reason_code="MERGED_MISSING_FASTP_EVIDENCE",
                    message=(
                        "merged_short datasets without fastp evidence: "
                        f"{missing}. Merge rates cannot be compared; merged-read "
                        "distributions carry strong selection bias."
                    ),
                    evidence_metric="merge_retained_fraction",
                    evidence_value=None,
                    recommended_action="Supply fastp_json paths for every merged_short "
                    "dataset, or disable require_fastp_evidence knowingly.",
                )
            )

    # Aggregation precedence (spec §9): hard fail > missing evidence > review > pass.
    components = [technical_status, metadata_status, identity_status]
    if CheckStatus.FAIL in components:
        overall = DecisionState.INCOMPATIBLE
    elif (
        CheckStatus.MISSING_EVIDENCE in components
        or CheckStatus.INSUFFICIENT_OBSERVATIONS in components
    ):
        overall = DecisionState.INSUFFICIENT_EVIDENCE
    elif CheckStatus.REVIEW in components:
        overall = DecisionState.REVIEW
    else:
        overall = DecisionState.COMPATIBLE

    all_findings = component_findings + tail_findings
    assumptions: list[str] = []
    if identity_level is IdentityEvidenceLevel.DECLARED_ONLY:
        assumptions.append("identity_declared_not_verified")
    if any(not t.calibrated for t in policy.thresholds):
        assumptions.append("thresholds_illustrative_uncalibrated")
    assumptions.append("technical_pooling_not_statistical_merge_of_biological_replicates")

    reason_codes: list[str] = []
    for f in all_findings:
        if f.reason_code not in reason_codes:
            reason_codes.append(f.reason_code)
    for code in tech_reason_codes:
        if code not in reason_codes:
            reason_codes.append(code)

    decision = GroupDecision(
        pool_group=group,
        structural_status=CheckStatus.PASS,
        technical_status=technical_status,
        metadata_status=metadata_status,
        identity_status=identity_status,
        overall_decision=overall,
        identity_evidence_level=identity_level,
        assumptions=assumptions,
        reason_codes=reason_codes,
    )
    return decision, tail_findings


def evaluate_group(
    outcomes: list[DatasetOutcome],
    comparison_rows: list[dict[str, Any]],
    policy: Policy,
) -> tuple[GroupDecision, list[Finding]]:
    """Evaluate one pooling group from its full comparison-row list.

    Convenience wrapper over group_components + TechnicalAssessment +
    finalize_group_decision for callers that hold the rows in memory (tests);
    the pipeline streams instead.
    """
    profile = outcomes[0].spec.read_type.value
    assay = outcomes[0].spec.assay
    metadata_status, identity_status, identity_level, component_findings = group_components(
        outcomes, policy
    )
    technical_status, tech_findings = _technical_assessment(
        comparison_rows, policy, profile, assay
    )
    tech_reason_codes: list[str] = []
    for f in tech_findings:
        if f.reason_code not in tech_reason_codes:
            tech_reason_codes.append(f.reason_code)
    decision, tail = finalize_group_decision(
        outcomes,
        policy,
        metadata_status,
        identity_status,
        identity_level,
        component_findings,
        technical_status,
        tech_reason_codes,
    )
    return decision, component_findings + tech_findings + tail


def assign_finding_ids(findings: list[Finding]) -> list[Finding]:
    """Deterministic finding ids: groups sorted, then component, reason, datasets."""
    findings_sorted = sorted(
        findings,
        key=lambda f: (
            f.pool_group,
            f.component,
            f.reason_code,
            tuple(f.affected_datasets),
            f.message,
        ),
    )
    for i, f in enumerate(findings_sorted, start=1):
        f.finding_id = f"F{i:04d}"
    return findings_sorted
