"""Policy loading, decision engine, aggregation precedence, exit-code semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import MINIMAL_POLICY
from seqconsistency.comparison import (
    STATUS_EXCEED,
    STATUS_INSUFFICIENT_OBSERVATIONS,
    STATUS_MISSING_EVIDENCE,
    STATUS_NO_THRESHOLD,
    STATUS_PASS,
)
from seqconsistency.errors import ConfigError
from seqconsistency.models import CheckStatus, DecisionState, IdentityEvidenceLevel
from seqconsistency.policy import (
    IDENTITY_DECLARED_SENTENCE,
    Policy,
    evaluate_group,
    load_policy,
)


def _policy(text: str = MINIMAL_POLICY) -> Policy:
    return load_policy(text)


def _row(status: str, **kw) -> dict:
    base = {
        "pool_group": "g",
        "dataset_a": "a",
        "dataset_b": "b",
        "metric": "gc_content",
        "stratum": "R1",
        "method": "jensenshannon_base2",
        "value": 0.1,
        "threshold": 0.05,
        "status": status,
    }
    base.update(kw)
    return base


def _outcome(
    dataset_id: str = "a",
    sample: str = "s1",
    library: str = "l1",
    assay: str = "WGS",
    platform: str = "ILLUMINA",
    processing: str | None = "p",
    read_type: str = "merged_short",
    input_format: str = "fastq",
):
    from seqconsistency.models import DatasetOutcome, DatasetSpec, InputFormat, ReadType

    spec = DatasetSpec(
        dataset_id=dataset_id,
        pool_group="g",
        sample_id=sample,
        library_id=library,
        assay=assay,
        platform=platform,
        read_type=ReadType(read_type),
        input_format=InputFormat(input_format),
        path1=Path("/dev/null"),
        path2=None,
        processing_id=processing,
        reference_id=None,
        fastp_json=None,
    )
    return DatasetOutcome(spec=spec)


def test_policy_load_and_strictness():
    p = _policy()
    assert p.policy_id == "test-policy"
    assert p.min_observations == 10
    assert p.thresholds[0].calibrated  # test fixture marks thresholds calibrated
    with pytest.raises(ConfigError, match="unknown key"):
        _policy(MINIMAL_POLICY.replace("identity:", "identity:\n  bogus_key: 1\n"))
    with pytest.raises(ConfigError, match="missing required key"):
        _policy(MINIMAL_POLICY.replace('policy_version: "1"\n', ""))
    with pytest.raises(ConfigError, match="must be a finite non-negative"):
        _policy(MINIMAL_POLICY.replace("max: 10.0", "max: -1.0"))
    with pytest.raises(ConfigError, match="must be a finite non-negative"):
        _policy(MINIMAL_POLICY.replace("max: 10.0", "max: .inf"))
    with pytest.raises(ConfigError, match="expected a boolean"):
        _policy(MINIMAL_POLICY.replace("calibrated: true", 'calibrated: "false"'))
    with pytest.raises(ConfigError, match="unknown metric/method pair"):
        _policy(MINIMAL_POLICY.replace("metric: read_length", "metric: read_len"))
    with pytest.raises(ConfigError, match="unknown metric/method pair"):
        _policy(MINIMAL_POLICY.replace("method: wasserstein1", "method: nonsense"))
    with pytest.raises(ConfigError, match="expected a list of strings"):
        _policy(MINIMAL_POLICY.replace("calibrated: true", "calibrated: true\n      profiles: ont"))
    with pytest.raises(ConfigError, match="overlapping"):
        _policy(
            MINIMAL_POLICY.replace(
                "    - metric: gc_content",
                "    - metric: read_length\n      method: wasserstein1\n      max: 5.0\n"
                "      action: review\n      calibrated: true\n    - metric: gc_content",
            )
        )
    with pytest.raises(ConfigError, match="expected one of"):
        _policy(MINIMAL_POLICY.replace("action: review", "action: explode"))


def test_policy_threshold_scoping():
    text = MINIMAL_POLICY.replace(
        "    - metric: read_length\n      method: wasserstein1\n      max: 10.0",
        "    - metric: read_length\n      method: wasserstein1\n      max: 10.0"
        "\n      profiles: [ont]\n      strata: [SE]",
    )
    p = load_policy(text)
    assert p.resolver_for("ont", "WGS")("read_length", "SE") is not None
    assert p.resolver_for("ont", "WGS")("read_length", "R1") is None
    assert p.resolver_for("paired_short", "WGS")("read_length", "SE") is None


def test_all_pass_is_compatible():
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.COMPATIBLE
    assert decision.identity_status is CheckStatus.PASS
    assert decision.identity_evidence_level is IdentityEvidenceLevel.DECLARED_ONLY
    assert any(IDENTITY_DECLARED_SENTENCE in f.message for f in findings)
    assert "identity_declared_not_verified" in decision.assumptions
    # The test fixture marks thresholds calibrated, so no uncalibrated assumption.
    assert "thresholds_illustrative_uncalibrated" not in decision.assumptions


def test_exceed_threshold_yields_review():
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_EXCEED)], p)
    assert decision.overall_decision is DecisionState.REVIEW
    assert decision.technical_status is CheckStatus.REVIEW
    assert any(f.reason_code == "TECH_THRESHOLD_EXCEEDED" for f in findings)


def test_exceed_action_incompatible_yields_incompatible():
    p = _policy(MINIMAL_POLICY.replace("action: review", "action: incompatible"))
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_EXCEED)], p)
    assert decision.overall_decision is DecisionState.INCOMPATIBLE
    assert decision.technical_status is CheckStatus.FAIL


def test_hard_incompatibility_beats_missing_evidence():
    # Different sample ids (FAIL) + insufficient observations: INCOMPATIBLE wins.
    p = _policy()
    outcomes = [_outcome("a", sample="s1"), _outcome("b", sample="s2")]
    decision, _ = evaluate_group(outcomes, [_row(STATUS_INSUFFICIENT_OBSERVATIONS)], p)
    assert decision.overall_decision is DecisionState.INCOMPATIBLE


def test_missing_evidence_beats_review():
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    decision, _ = evaluate_group(outcomes, [_row(STATUS_EXCEED), _row(STATUS_MISSING_EVIDENCE)], p)
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE
    assert decision.technical_status is CheckStatus.MISSING_EVIDENCE


def test_no_threshold_blocks_compatible_when_required():
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_NO_THRESHOLD)], p)
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE
    assert any(f.reason_code == "TECH_NO_THRESHOLD_CONFIGURED" for f in findings)


def test_no_thresholds_cannot_approve_even_when_not_required():
    # Spec §7 is unconditional: "If no threshold is configured or calibrated,
    # report the metric but do not use it to automatically approve pooling."
    # With require_thresholds: false AND zero assessed rows, technical status
    # must be MISSING_EVIDENCE, never PASS (review round 2, point 4).
    p = _policy(MINIMAL_POLICY.replace("require_thresholds: true", "require_thresholds: false"))
    outcomes = [_outcome("a"), _outcome("b")]
    rows = [_row(STATUS_NO_THRESHOLD, threshold=None)]
    decision, findings = evaluate_group(outcomes, rows, p)
    assert decision.technical_status is CheckStatus.MISSING_EVIDENCE
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE
    assert any(
        f.reason_code == "TECH_NO_THRESHOLD_CONFIGURED" and f.severity.value == "warning"
        for f in findings
    )
    # Per-row INFO findings still report every unthresholded distance.
    assert any(
        f.reason_code == "TECH_NO_THRESHOLD_CONFIGURED" and f.severity.value == "info"
        for f in findings
    )


def test_assessed_rows_decide_when_others_unthresholded():
    # require_thresholds: false with SOME thresholds: assessed rows drive the
    # decision; unthresholded metrics stay informational (neither approve nor
    # block) — the natural reading of the spec sentence.
    p = _policy(MINIMAL_POLICY.replace("require_thresholds: true", "require_thresholds: false"))
    outcomes = [_outcome("a"), _outcome("b")]
    rows = [
        _row(STATUS_NO_THRESHOLD, threshold=None, metric="read_length"),
        _row(STATUS_PASS, metric="gc_content", threshold=0.10),
    ]
    decision, findings = evaluate_group(outcomes, rows, p)
    assert decision.technical_status is CheckStatus.PASS
    assert decision.overall_decision is DecisionState.COMPATIBLE
    assert any(
        f.reason_code == "TECH_NO_THRESHOLD_CONFIGURED" and f.severity.value == "info"
        for f in findings
    )


def test_insufficient_observations_produce_findings_and_reason_codes():
    # Review round 3, point 5: INSUFFICIENT_EVIDENCE decisions must explain
    # what blocked approval — comparison reason, datasets, observed counts,
    # and the required minimum reach the findings table and the decision row.
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    row = _row(
        STATUS_INSUFFICIENT_OBSERVATIONS,
        threshold=None,
        reason_code="TECH_BELOW_MIN_OBSERVATIONS",
        observations_a=1,
        observations_b=2,
    )
    decision, findings = evaluate_group(outcomes, [row], p)
    assert decision.technical_status is CheckStatus.MISSING_EVIDENCE
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE
    assert "TECH_BELOW_MIN_OBSERVATIONS" in decision.reason_codes
    tech = [f for f in findings if f.component == "technical"]
    assert len(tech) == 1
    msg = tech[0].message
    assert "1/2 observations" in msg and "minimum is 10" in msg
    assert tech[0].severity.value == "warning"


def test_single_dataset_group_without_rows_stays_compatible():
    # One dataset means zero pairs and zero comparison rows; the no-assessed-
    # rows guard must not turn a trivially poolable group into a failure.
    p = _policy(MINIMAL_POLICY.replace("require_thresholds: true", "require_thresholds: false"))
    decision, findings = evaluate_group([_outcome("a")], [], p)
    assert decision.technical_status is CheckStatus.PASS
    assert decision.overall_decision is DecisionState.COMPATIBLE
    assert not [f for f in findings if f.component == "technical"]


def test_identity_verification_requested_yields_insufficient():
    p = _policy(
        MINIMAL_POLICY.replace(
            "require_independent_verification: false", "require_independent_verification: true"
        )
    )
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.identity_status is CheckStatus.MISSING_EVIDENCE
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE
    assert any(f.reason_code == "IDENTITY_VERIFICATION_NOT_IMPLEMENTED" for f in findings)


def test_identity_declared_refused_yields_insufficient():
    p = _policy(
        MINIMAL_POLICY.replace(
            "accept_declared_sample_id: true", "accept_declared_sample_id: false"
        )
    )
    outcomes = [_outcome("a"), _outcome("b")]
    decision, _ = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.identity_status is CheckStatus.MISSING_EVIDENCE
    assert decision.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE


def test_different_sample_ids_are_incompatible():
    p = _policy()
    outcomes = [_outcome("a", sample="s1"), _outcome("b", sample="s2")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.metadata_status is CheckStatus.FAIL
    assert decision.identity_status is CheckStatus.FAIL
    assert decision.identity_evidence_level is IdentityEvidenceLevel.CONFLICTING
    assert decision.overall_decision is DecisionState.INCOMPATIBLE
    assert any(f.reason_code == "GROUP_DIFFERENT_SAMPLE_IDS" for f in findings)


def test_different_libraries_review_by_default():
    p = _policy()
    outcomes = [_outcome("a", library="l1"), _outcome("b", library="l2")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.REVIEW
    assert any(f.reason_code == "GROUP_DIFFERENT_LIBRARY_IDS" for f in findings)


def test_same_sample_different_library_allowed_when_policy_says_so():
    p = _policy(
        MINIMAL_POLICY.replace("distinct_library_ids: review", "distinct_library_ids: allowed")
    )
    outcomes = [_outcome("a", library="l1"), _outcome("b", library="l2")]
    decision, _ = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.metadata_status is CheckStatus.PASS
    assert decision.overall_decision is DecisionState.COMPATIBLE


def test_different_assays_are_incompatible():
    p = _policy()
    outcomes = [_outcome("a", assay="WGS"), _outcome("b", assay="RNA-seq")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.INCOMPATIBLE
    assert any(f.reason_code == "GROUP_DIFFERENT_ASSAYS" for f in findings)


def test_different_platforms_review():
    p = _policy()
    outcomes = [_outcome("a", platform="ILLUMINA"), _outcome("b", platform="ONT")]
    decision, _ = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.REVIEW


def test_null_processing_id_warns():
    p = _policy()
    outcomes = [_outcome("a", processing=None), _outcome("b", processing="p1")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert any(f.reason_code == "METADATA_PROCESSING_UNKNOWN" for f in findings)
    assert decision.overall_decision is DecisionState.REVIEW


def test_bam_reference_mismatch_incompatible():
    from seqconsistency.models import BamFlagCounts, BamHeaderInfo, BamScanResult

    outcomes = [
        _outcome("a", read_type="paired_short", input_format="bam"),
        _outcome("b", read_type="paired_short", input_format="bam"),
    ]
    for o, refs in zip(
        outcomes,
        (
            [("chr1", 1000, None), ("chr2", 500, None)],
            [("chr1", 1000, None)],
        ),
        strict=True,
    ):
        o.bam = BamScanResult(
            dataset_id=o.spec.dataset_id,
            flags=BamFlagCounts(),
            header=BamHeaderInfo("coordinate", refs, []),
            eligible_observations=10,
            total_bases=1000,
            mapq_hist=None,
            insert_hist=None,
            length_hist=None,
            sample_reads={},
            sampled_count=0,
            scan_seconds=0.0,
            records_read=10,
        )
    p = _policy()
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.INCOMPATIBLE
    assert any(f.reason_code == "BAM_REFERENCE_DICTIONARY_MISMATCH" for f in findings)


def _bam_outcome(dataset_id: str, refs):
    from seqconsistency.models import BamFlagCounts, BamHeaderInfo, BamScanResult

    o = _outcome(dataset_id, read_type="paired_short", input_format="bam")
    o.bam = BamScanResult(
        dataset_id=dataset_id,
        flags=BamFlagCounts(),
        header=BamHeaderInfo("coordinate", refs, []),
        eligible_observations=10,
        total_bases=1000,
        mapq_hist=None,
        insert_hist=None,
        length_hist=None,
        sample_reads={},
        sampled_count=0,
        scan_seconds=0.0,
        records_read=10,
    )
    return o


@pytest.mark.parametrize("order", [tuple("abc"), tuple("bac"), tuple("cab")])
def test_bam_checksum_conflict_found_regardless_of_order(order):
    # A conflict between the second and third BAM must surface even when the
    # first BAM carries no M5 at all (order-independent aggregation).
    refs = {
        "a": [("chr1", 1000, None)],
        "b": [("chr1", 1000, "1" * 32)],
        "c": [("chr1", 1000, "2" * 32)],
    }
    outcomes = [_bam_outcome(n, refs[n]) for n in order]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], _policy())
    assert any(f.reason_code == "BAM_REFERENCE_CHECKSUM_CONFLICT" for f in findings)
    assert decision.metadata_status is CheckStatus.FAIL
    assert decision.overall_decision is DecisionState.INCOMPATIBLE


def test_bam_missing_md5_is_review_not_conflict():
    # Absent checksums keep the identity-unverified REVIEW; they must never be
    # reported as a conflict (matching names and lengths are not evidence of
    # either identity or conflict).
    outcomes = [
        _bam_outcome("a", [("chr1", 1000, "1" * 32)]),
        _bam_outcome("b", [("chr1", 1000, None)]),
    ]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], _policy())
    assert any(f.reason_code == "BAM_REFERENCE_IDENTITY_UNVERIFIED" for f in findings)
    assert not any(f.reason_code == "BAM_REFERENCE_CHECKSUM_CONFLICT" for f in findings)
    assert decision.metadata_status is CheckStatus.REVIEW


def test_merged_short_missing_fastp_review_when_required():
    p = _policy(
        MINIMAL_POLICY.replace(
            "reference_mismatch: incompatible",
            "reference_mismatch: incompatible\n  require_fastp_evidence: true",
        )
    )
    outcomes = [_outcome("a"), _outcome("b")]
    decision, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    assert any(f.reason_code == "MERGED_MISSING_FASTP_EVIDENCE" for f in findings)
    assert decision.overall_decision is DecisionState.REVIEW


def test_finding_ids_deterministic():
    from seqconsistency.policy import assign_finding_ids

    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    _, findings = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    ids1 = [f.finding_id for f in assign_finding_ids(list(findings))]
    _, findings2 = evaluate_group(outcomes, [_row(STATUS_PASS)], p)
    ids2 = [f.finding_id for f in assign_finding_ids(list(findings2))]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


def test_uncalibrated_threshold_prevents_automatic_approval():
    # Spec §7: a threshold that is not calibrated must not automatically approve
    # pooling. PASS rows against uncalibrated thresholds -> REVIEW + finding.
    p = _policy(MINIMAL_POLICY.replace("calibrated: true", "calibrated: false"))
    outcomes = [_outcome("a"), _outcome("b")]
    row = dict(_row(STATUS_PASS), threshold_calibrated=False)
    decision, findings = evaluate_group(outcomes, [row], p)
    assert decision.overall_decision is DecisionState.REVIEW
    assert any(f.reason_code == "TECH_THRESHOLD_UNCALIBRATED" for f in findings)
    assert "thresholds_illustrative_uncalibrated" in decision.assumptions


def test_calibrated_threshold_allows_approval():
    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    row = dict(_row(STATUS_PASS), threshold_calibrated=True)
    decision, findings = evaluate_group(outcomes, [row], p)
    assert decision.overall_decision is DecisionState.COMPATIBLE
    assert not any(f.reason_code == "TECH_THRESHOLD_UNCALIBRATED" for f in findings)


def test_not_assessed_rows_block_nothing():
    # Optional unrequested evidence (e.g. fastp NOT_ASSESSED) neither blocks nor
    # downgrades a decision that assessed rows support. Alone it provides no
    # support for approval, so a NOT_ASSESSED-only row set (which the pipeline
    # cannot produce — every pair yields metric rows) stays INSUFFICIENT_EVIDENCE.
    from seqconsistency.comparison import STATUS_NOT_ASSESSED

    p = _policy()
    outcomes = [_outcome("a"), _outcome("b")]
    optional = dict(
        _row(STATUS_NOT_ASSESSED), value=None, threshold=None, reason_code="FASTP_EVIDENCE_OPTIONAL"
    )
    decision, _ = evaluate_group(outcomes, [optional, _row(STATUS_PASS)], p)
    assert decision.overall_decision is DecisionState.COMPATIBLE
    decision_alone, _ = evaluate_group(outcomes, [optional], p)
    assert decision_alone.overall_decision is DecisionState.INSUFFICIENT_EVIDENCE


def test_bam_conflicting_sm_values_are_identity_conflict():
    from seqconsistency.policy import _bam_sample_metadata_findings

    a = _outcome("a", read_type="paired_short", input_format="bam")
    b = _outcome("b", read_type="paired_short", input_format="bam")
    from seqconsistency.models import BamFlagCounts, BamHeaderInfo, BamScanResult

    for o, sm in zip((a, b), ("sampleA", "sampleB"), strict=True):
        o.bam = BamScanResult(
            dataset_id=o.spec.dataset_id,
            flags=BamFlagCounts(),
            header=BamHeaderInfo("coordinate", [("chr1", 1000, None)], [{"ID": "rg1", "SM": sm}]),
            eligible_observations=10,
            total_bases=100,
            mapq_hist=None,
            insert_hist=None,
            length_hist=None,
            sample_reads={},
            sampled_count=0,
            scan_seconds=0.0,
            records_read=10,
        )
    id_status, md_status, findings = _bam_sample_metadata_findings([a, b])
    assert id_status is CheckStatus.FAIL
    assert any(f.reason_code == "BAM_SAMPLE_METADATA_CONFLICT" for f in findings)


def test_bam_sm_mismatching_manifest_is_review():
    from seqconsistency.models import BamFlagCounts, BamHeaderInfo, BamScanResult
    from seqconsistency.policy import _bam_sample_metadata_findings

    a = _outcome("a", read_type="paired_short", input_format="bam", sample="declared_sample")
    b = _outcome("b", read_type="paired_short", input_format="bam", sample="declared_sample")
    for o in (a, b):
        o.bam = BamScanResult(
            dataset_id=o.spec.dataset_id,
            flags=BamFlagCounts(),
            header=BamHeaderInfo(
                "coordinate", [("chr1", 1000, None)], [{"ID": "rg1", "SM": "header_sample"}]
            ),
            eligible_observations=10,
            total_bases=100,
            mapq_hist=None,
            insert_hist=None,
            length_hist=None,
            sample_reads={},
            sampled_count=0,
            scan_seconds=0.0,
            records_read=10,
        )
    id_status, md_status, findings = _bam_sample_metadata_findings([a, b])
    assert id_status is CheckStatus.PASS
    assert md_status is CheckStatus.REVIEW
    assert any(f.reason_code == "BAM_SM_MISMATCHES_MANIFEST" for f in findings)


def test_bam_undefined_rg_records_reported():
    from seqconsistency.models import BamFlagCounts, BamHeaderInfo, BamScanResult
    from seqconsistency.policy import _bam_sample_metadata_findings

    a = _outcome("a", read_type="paired_short", input_format="bam")
    flags = BamFlagCounts()
    flags.records_with_undefined_rg = 5
    a.bam = BamScanResult(
        dataset_id="a",
        flags=flags,
        header=BamHeaderInfo("coordinate", [("chr1", 1000, None)], []),
        eligible_observations=10,
        total_bases=100,
        mapq_hist=None,
        insert_hist=None,
        length_hist=None,
        sample_reads={},
        sampled_count=0,
        scan_seconds=0.0,
        records_read=10,
    )
    id_status, md_status, findings = _bam_sample_metadata_findings([a])
    assert md_status is CheckStatus.REVIEW
    assert any(f.reason_code == "BAM_UNDEFINED_RG_RECORDS" for f in findings)
