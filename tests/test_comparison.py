"""Pairwise comparison engine: thresholds, degeneracy, ordering, nontransitivity."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import MINIMAL_POLICY, make_records, write_fastq
from seqconsistency.comparison import (
    STATUS_EXCEED,
    STATUS_INSUFFICIENT_OBSERVATIONS,
    STATUS_NO_THRESHOLD,
    STATUS_PASS,
    compare_group,
    compare_pair,
    count_total_pairs,
    generate_pairs,
)
from seqconsistency.fastq import scan_fastq_single
from seqconsistency.models import (
    DatasetOutcome,
    DatasetSpec,
    FastqScanResult,
    InputFormat,
    ReadType,
    SamplingConfig,
)
from seqconsistency.policy import load_policy


def _outcome(
    dataset_id: str,
    path: Path,
    read_type: ReadType = ReadType.MERGED_SHORT,
    sample_size: int = 5000,
    seed: int = 42,
) -> DatasetOutcome:
    sampling = SamplingConfig(sample_size=sample_size, master_seed=seed)
    stats, sample, eligible, sampled = scan_fastq_single(
        dataset_id, path, read_type, sampling, sample_size
    )
    spec = DatasetSpec(
        dataset_id=dataset_id,
        pool_group="g",
        sample_id="s1",
        library_id="l1",
        assay="WGS",
        platform="X",
        read_type=read_type,
        input_format=InputFormat.FASTQ,
        path1=path,
        path2=None,
        processing_id="p",
        reference_id=None,
        fastp_json=None,
    )
    return DatasetOutcome(
        spec=spec,
        fastq=FastqScanResult(
            dataset_id=dataset_id,
            r1=stats,
            r2=None,
            pairing=None,
            sample_r1=sample,
            sample_r2=None,
            eligible_observations=eligible,
            sampled_count=sampled,
            scan_seconds=0.0,
            bytes_read=0,
        ),
    )


def _resolver(max_map: dict | None = None):
    policy = load_policy(MINIMAL_POLICY)

    def resolve(metric: str, stratum: str):
        if max_map is not None:
            if metric in max_map:
                from seqconsistency.comparison import ThresholdRule

                return ThresholdRule(max_map[metric], "review", False)
            return None
        return policy.resolver_for("merged_short", "WGS")(metric, stratum)

    return resolve


def test_identical_distributions_pass(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(2000, seed=1))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(2000, seed=1))
    rows = compare_group("g", [_outcome("a", p1), _outcome("b", p2)], 10, _resolver())
    passes = [r for r in rows if r["status"] == STATUS_PASS]
    assert len(passes) >= 5
    gc = next(r for r in rows if r["metric"] == "gc_content")
    assert gc["value"] < 0.05
    ln = next(r for r in rows if r["metric"] == "read_length")
    assert ln["value"] == pytest.approx(0.0, abs=0.01)


def test_shifted_length_exceeds_threshold(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(2000, length=100, seed=2))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(2000, length=140, seed=3))
    rows = compare_group("g", [_outcome("a", p1), _outcome("b", p2)], 10, _resolver())
    ln = next(r for r in rows if r["metric"] == "read_length")
    assert ln["value"] == pytest.approx(40.0, abs=1.0)
    assert ln["status"] == STATUS_EXCEED
    assert ln["threshold"] == 10.0
    assert ln["threshold_direction"] == "max"


def test_shifted_gc_exceeds_threshold(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(3000, gc=0.4, seed=4))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(3000, gc=0.6, seed=5))
    rows = compare_group("g", [_outcome("a", p1), _outcome("b", p2)], 10, _resolver())
    gc = next(r for r in rows if r["metric"] == "gc_content")
    assert gc["value"] > 0.1
    assert gc["status"] == STATUS_EXCEED


def test_missing_threshold_reports_no_threshold(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(500, seed=6))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(500, seed=7))
    rows = compare_group(
        "g", [_outcome("a", p1), _outcome("b", p2)], 10, _resolver(max_map={"gc_content": 0.05})
    )
    assert any(r["status"] == STATUS_NO_THRESHOLD for r in rows)
    # Values are still computed and reported.
    nt = next(r for r in rows if r["status"] == STATUS_NO_THRESHOLD)
    assert nt["value"] is not None and nt["threshold"] is None


def test_insufficient_observations(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(50, seed=8))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(2000, seed=9))
    rows = compare_group(
        "g", [_outcome("a", p1, sample_size=2000), _outcome("b", p2)], 100, _resolver()
    )
    assert rows
    for r in rows:
        if r["metric"] in ("read_length", "mean_read_quality"):
            assert r["status"] == STATUS_INSUFFICIENT_OBSERVATIONS
            assert r["observations_a"] == 50
            assert r["reason_code"] == "TECH_BELOW_MIN_OBSERVATIONS"


def test_pair_ordering_is_deterministic(tmp_path: Path):
    ids = ["d3", "d1", "d2"]
    pairs = list(generate_pairs(ids))
    assert pairs == [("d1", "d2"), ("d1", "d3"), ("d2", "d3")]
    files = [
        write_fastq(tmp_path / f"{i}.fastq", make_records(100, seed=hash(i) % 1000)) for i in ids
    ]
    outs = {i: _outcome(i, f) for i, f in zip(ids, files, strict=True)}
    rows = compare_group("g", list(outs.values()), 10, _resolver())
    pair_keys = {(r["dataset_a"], r["dataset_b"]) for r in rows}
    assert pair_keys == {("d1", "d2"), ("d1", "d3"), ("d2", "d3")}


def test_count_total_pairs_guard():
    assert count_total_pairs([["a", "b", "c"]]) == 3
    assert count_total_pairs([["a", "b"], ["c", "d"]]) == 2


def test_nontransitive_pairwise_compatibility(tmp_path: Path):
    # a: length 100, b: length 110, c: length 130.
    # With threshold 12: a~b (W=10) passes, b~c (W=20) fails, a~c (W=30) fails.
    pa = write_fastq(tmp_path / "a.fastq", make_records(2000, length=100, seed=10))
    pb = write_fastq(tmp_path / "b.fastq", make_records(2000, length=110, seed=11))
    pc = write_fastq(tmp_path / "c.fastq", make_records(2000, length=130, seed=12))
    rows = compare_group(
        "g",
        [_outcome("a", pa), _outcome("b", pb), _outcome("c", pc)],
        10,
        _resolver(max_map={"read_length": 12.0}),
    )
    by_pair = {}
    for r in rows:
        if r["metric"] == "read_length":
            by_pair[(r["dataset_a"], r["dataset_b"])] = r
    assert by_pair[("a", "b")]["status"] == STATUS_PASS
    assert by_pair[("b", "c")]["status"] == STATUS_EXCEED
    assert by_pair[("a", "c")]["status"] == STATUS_EXCEED


def test_zero_variance_degenerate_distributions(tmp_path: Path):
    # All reads same length and same quality: distributions are point masses.
    def fixed(n: int, seq: bytes, qual: bytes):
        return [(f"@x{i}".encode(), seq, qual) for i in range(n)]

    p1 = write_fastq(tmp_path / "a.fastq", fixed(500, b"AC" * 25, bytes([40 + 33]) * 50))
    p2 = write_fastq(tmp_path / "b.fastq", fixed(500, b"AC" * 25, bytes([40 + 33]) * 50))
    rows = compare_group("g", [_outcome("a", p1), _outcome("b", p2)], 10, _resolver())
    for r in rows:
        if r["metric"] == "read_length":
            assert r["value"] == pytest.approx(0.0)
        if r["metric"] == "gc_content":
            assert r["value"] == pytest.approx(0.0)


def test_all_n_reads_degenerate_gc(tmp_path: Path):
    # GC undefined on both sides -> finite values excluded; JSD on the n-histogram
    # still works and the gc comparison must not silently pass as zero distance.
    def all_n(n: int):
        return [(f"@n{i}".encode(), b"N" * 50, bytes([30 + 33]) * 50) for i in range(n)]

    p1 = write_fastq(tmp_path / "a.fastq", all_n(500))
    p2 = write_fastq(tmp_path / "b.fastq", all_n(500))
    outs = [_outcome("a", p1), _outcome("b", p2)]
    rows = compare_group("g", outs, 10, _resolver())
    gc_row = next(r for r in rows if r["metric"] == "gc_content")
    # Both histograms are all-zero -> JSD cannot be computed -> not a silent pass.
    assert gc_row["status"] in (STATUS_INSUFFICIENT_OBSERVATIONS,) or gc_row["value"] is None


def test_unequal_dataset_sizes(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(1000, seed=13))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(50000, seed=14))
    rows = compare_group(
        "g",
        [_outcome("a", p1, sample_size=1000), _outcome("b", p2, sample_size=1000)],
        10,
        _resolver(),
    )
    gc = next(r for r in rows if r["metric"] == "gc_content")
    assert gc["observations_a"] == 1000
    assert gc["observations_b"] == 50000
    assert gc["status"] in (STATUS_PASS, STATUS_EXCEED)  # computed, not skipped


def test_merged_short_fastp_missing_evidence_row(tmp_path: Path):
    p1 = write_fastq(tmp_path / "a.fastq", make_records(500, seed=15))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(500, seed=16))
    outs = [_outcome("a", p1), _outcome("b", p2)]
    rows = compare_pair("g", outs[0], outs[1], 10, _resolver())
    merge = [r for r in rows if r["metric"] == "merge_rate"]
    assert len(merge) == 1
    # Not required by default: assessed as nothing, blocks nothing.
    assert merge[0]["status"] == "NOT_ASSESSED"
    assert merge[0]["reason_code"] == "FASTP_EVIDENCE_OPTIONAL"
    required = compare_pair("g", outs[0], outs[1], 10, _resolver(), require_fastp=True)
    req = [r for r in required if r["metric"] == "merge_rate"]
    assert req[0]["status"] == "MISSING_EVIDENCE"
    assert req[0]["reason_code"] == "FASTP_EVIDENCE_MISSING"


def test_merged_short_fastp_retained_comparison(tmp_path: Path):
    import json

    p1 = write_fastq(tmp_path / "a.fastq", make_records(500, seed=17))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(500, seed=18))
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(
        json.dumps(
            {
                "summary": {
                    "before_filtering": {"total_reads": 1000},
                    "after_filtering": {"total_reads": 900},
                }
            }
        )
    )
    f2.write_text(
        json.dumps(
            {
                "summary": {
                    "before_filtering": {"total_reads": 1000},
                    "after_filtering": {"total_reads": 400},
                }
            }
        )
    )
    from seqconsistency.fastq import parse_fastp_json

    o1, o2 = _outcome("a", p1), _outcome("b", p2)
    o1.fastp = parse_fastp_json(f1)
    o2.fastp = parse_fastp_json(f2)
    rows = compare_pair("g", o1, o2, 10, _resolver())
    merge = next(r for r in rows if r["metric"] == "merge_retained_fraction")
    assert merge["value"] == pytest.approx(0.5)
    assert merge["method"] == "absolute_difference"
    assert merge["status"] == STATUS_EXCEED  # |0.9-0.4| = 0.5 > 0.1
    assert merge["threshold_calibrated"] is True


def test_merge_rate_preferred_over_retained_fraction(tmp_path: Path):
    import json

    from seqconsistency.fastq import parse_fastp_json

    p1 = write_fastq(tmp_path / "a.fastq", make_records(500, seed=17))
    p2 = write_fastq(tmp_path / "b.fastq", make_records(500, seed=18))
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    # Explicit merged-read counts present -> merge_rate compared, retained ignored.
    f1.write_text(
        json.dumps(
            {
                "summary": {
                    "before_filtering": {"total_reads": 1000},
                    "after_filtering": {"total_reads": 900},
                },
                "merging": {"merged_reads": 600},
            }
        )
    )
    f2.write_text(
        json.dumps(
            {
                "summary": {
                    "before_filtering": {"total_reads": 1000},
                    "after_filtering": {"total_reads": 400},
                },
                "merging": {"merged_reads": 610},
            }
        )
    )
    o1, o2 = _outcome("a", p1), _outcome("b", p2)
    o1.fastp = parse_fastp_json(f1)
    o2.fastp = parse_fastp_json(f2)
    rows = compare_pair("g", o1, o2, 10, _resolver())
    metrics = [r["metric"] for r in rows if r["component"] == "processing"]
    assert metrics == ["merge_rate"]  # retained fraction not compared
    row = next(r for r in rows if r["metric"] == "merge_rate")
    assert row["value"] == pytest.approx(0.01)


def test_bam_metrics_selected_by_profile():
    from seqconsistency.comparison import bam_metrics
    from seqconsistency.models import ReadType

    paired = bam_metrics(ReadType.PAIRED_SHORT)
    ont = bam_metrics(ReadType.ONT)
    assert any(d.metric == "insert_size" for d in paired)
    assert not any(d.metric == "insert_size" for d in ont)
    assert {d.metric for d in ont} == {
        "read_length",
        "mean_read_quality",
        "gc_content",
        "n_content",
        "mapq",
    }
