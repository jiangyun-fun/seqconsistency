"""End-to-end CLI runs: exit codes, reproducibility, schemas, multiple groups."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from helpers import (
    MINIMAL_POLICY,
    make_bam,
    make_records,
    write_fastq,
    write_parquet_manifest,
    write_pe,
    write_policy,
    write_tsv_manifest,
)


def _pe_group(tmp_path: Path, stem: str, n: int = 2000, gc: float = 0.5, seed: int = 1):
    return write_pe(tmp_path, stem, n, gc=gc, seed=seed, length=100)


def _paired_manifest(tmp_path: Path, groups: list[tuple[str, list]]) -> Path:
    rows = []
    for group, files in groups:
        for i, (p1, p2) in enumerate(files):
            rows.append(
                {
                    "dataset_id": f"{group}_{i}",
                    "pool_group": group,
                    "sample_id": "s1",
                    "library_id": "l1",
                    "assay": "WGS",
                    "platform": "ILLUMINA",
                    "read_type": "paired_short",
                    "input_format": "fastq",
                    "path1": str(p1),
                    "path2": str(p2),
                    "processing_id": "p1",
                }
            )
    return write_tsv_manifest(tmp_path / "m.tsv", rows)


def test_cli_end_to_end_compatible(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=10)
    b = _pe_group(tmp_path, "b", seed=11)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    result, out, prov = run_cli(tmp_path / "run", manifest, policy, workers=2, expect_code=0)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert len(decisions) == 1
    assert decisions[0]["overall_decision"] == "COMPATIBLE"
    provenance = json.loads(prov.read_text())
    assert provenance["policy"]["policy_id"] == "test-policy"
    assert len(provenance["inputs"]) == 2
    # All five parquet outputs exist and are non-empty.
    for name, path in out.items():
        assert pq.read_table(path).num_rows > 0, name
    # Comparisons carry the expected metric set for paired_short.
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    metrics = {r["metric"] for r in comparisons}
    assert {
        "read_length",
        "gc_content",
        "base_quality",
        "nucleotide_composition",
        "mean_read_quality",
    } <= metrics
    assert all(r["dataset_a"] < r["dataset_b"] for r in comparisons)


def test_cli_shifted_gc_produces_review_exit_1(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", gc=0.40, seed=20)
    b = _pe_group(tmp_path, "b", gc=0.62, seed=21)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] == "REVIEW"
    findings = pq.read_table(out["findings"]).to_pylist()
    assert any(
        f["reason_code"] == "TECH_THRESHOLD_EXCEEDED" and f["evidence_metric"] == "gc_content"
        for f in findings
    )


def test_cli_genuine_incompatibility_exit_1(tmp_path: Path):
    # Different declared samples -> INCOMPATIBLE (exit 1, a scientific result).
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=30)
    b = _pe_group(tmp_path, "b", seed=31)
    manifest = write_tsv_manifest(
        tmp_path / "m.tsv",
        [
            {
                "dataset_id": "a",
                "pool_group": "g",
                "sample_id": "s1",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "paired_short",
                "input_format": "fastq",
                "path1": str(a[0]),
                "path2": str(a[1]),
            },
            {
                "dataset_id": "b",
                "pool_group": "g",
                "sample_id": "DONOR_X",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "paired_short",
                "input_format": "fastq",
                "path1": str(b[0]),
                "path2": str(b[1]),
            },
        ],
    )
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] == "INCOMPATIBLE"
    assert decisions[0]["identity_evidence_level"] == "conflicting_declarations"


def test_cli_insufficient_evidence_exit_1(tmp_path: Path):
    # Policy demands independent identity verification, which v1 cannot do.
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=40)
    b = _pe_group(tmp_path, "b", seed=41)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy_text = Path(__file__).parent / ".." / "policies" / "example-illustrative.yaml"
    policy = write_policy(
        tmp_path / "p.yaml",
        policy_text.read_text().replace(
            "require_independent_verification: false",
            "require_independent_verification: true",
        ),
    )
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] == "INSUFFICIENT_EVIDENCE"


def test_two_groups_in_one_invocation(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=50)
    b = _pe_group(tmp_path, "b", seed=51)
    c = _pe_group(tmp_path, "c", seed=52)
    d = _pe_group(tmp_path, "d", gc=0.61, seed=53)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b]), ("g2", [c, d])])
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(
        tmp_path / "run", manifest, policy, workers=4, threads=4, expect_code=1
    )
    decisions = pq.read_table(out["decisions"]).to_pylist()
    by_group = {d["pool_group"]: d for d in decisions}
    assert by_group["g1"]["overall_decision"] == "COMPATIBLE"
    assert by_group["g2"]["overall_decision"] == "REVIEW"


def test_merged_short_group_end_to_end(tmp_path: Path):
    import numpy as np

    from helpers import run_cli

    rng = np.random.default_rng(60)
    files = []
    for i, base_len in enumerate((220, 225)):
        lengths = rng.integers(base_len - 20, base_len + 20, size=400)
        files.append(
            write_fastq(
                tmp_path / f"m{i}.fastq",
                make_records(400, lengths=lengths, seed=60 + i),
            )
        )
    rows = [
        {
            "dataset_id": f"m{i}",
            "pool_group": "g",
            "sample_id": "s",
            "library_id": "l",
            "assay": "WGS",
            "platform": "I",
            "read_type": "merged_short",
            "input_format": "fastq",
            "path1": str(f),
        }
        for i, f in enumerate(files)
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] in ("REVIEW", "INSUFFICIENT_EVIDENCE")
    # fastp evidence absent and not required: NOT_ASSESSED, never blocking.
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    assert any(r["metric"] == "merge_rate" and r["status"] == "NOT_ASSESSED" for r in comparisons)


def test_long_read_group_end_to_end(tmp_path: Path):
    import numpy as np

    from helpers import run_cli

    files = []
    for i in range(2):
        lengths = np.random.default_rng(70 + i).lognormal(9.0, 0.6, 500).astype(int)
        lengths = np.clip(lengths, 200, 100000)
        files.append(
            write_fastq(
                tmp_path / f"ont{i}.fastq",
                make_records(500, lengths=lengths, seed=70 + i),
            )
        )
    rows = [
        {
            "dataset_id": f"o{i}",
            "pool_group": "g",
            "sample_id": "s",
            "library_id": "l",
            "assay": "WGS",
            "platform": "ONT",
            "read_type": "ont",
            "input_format": "fastq",
            "path1": str(f),
        }
        for i, f in enumerate(files)
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    assert {r["metric"] for r in comparisons} == {
        "read_length",
        "mean_read_quality",
        "gc_content",
        "n_content",
    }


def test_bam_group_end_to_end(tmp_path: Path):
    from helpers import run_cli

    b1 = make_bam(tmp_path / "b1.bam", n=400, seed=80)
    b2 = make_bam(tmp_path / "b2.bam", n=400, seed=81)
    rows = [
        {
            "dataset_id": f"b{i}",
            "pool_group": "g",
            "sample_id": "s",
            "library_id": "l",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "paired_short",
            "input_format": "bam",
            "path1": str(p),
            "reference_id": "hg38",
        }
        for i, p in enumerate((b1, b2))
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    metrics = {r["metric"] for r in comparisons}
    assert {"read_length", "gc_content", "mapq", "insert_size"} <= metrics
    decisions = pq.read_table(out["decisions"]).to_pylist()
    # Identity accepted from metadata alone (declared, not verified).
    assert decisions[0]["identity_evidence_level"] == "declared_not_verified"


def _mate_swapped_bam(path: Path, n: int, q1: int, q2: int, seed: int) -> Path:
    """Paired BAM: R1 at uniform q1, R2 at uniform q2 (R1 leftmost, FR)."""
    import numpy as np
    import pysam

    header = {
        "HD": {"SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1_000_000, "M5": "6a9a0c1f2b3d4e5f60718293a4b5c6d7"}],
        "RG": [{"ID": "rg1", "SM": "s", "LB": "l", "PL": "ILLUMINA"}],
    }
    rng = np.random.default_rng(seed)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(n):
            p = 1000 + i * 400
            seq = "".join(rng.choice(list("ACGT"), size=100))
            for pos, flag, tlen, q in (
                (p, 0x1 | 0x2 | 0x40, 300, q1),
                (p + 200, 0x1 | 0x2 | 0x80 | 0x10, -300, q2),
            ):
                seg = pysam.AlignedSegment(out.header)
                seg.query_name = f"r{i}"
                seg.query_sequence = seq
                seg.query_qualities = np.full(100, q + 33, dtype=np.uint8)
                seg.reference_id = 0
                seg.reference_start = pos
                seg.mapping_quality = 60
                seg.cigarstring = "100M"
                seg.flag = flag
                seg.template_length = tlen
                out.write(seg)
    return path


def test_bam_mate_swapped_qualities_detected_per_mate(tmp_path: Path):
    # Review round 2, point 1: pooling mates into one READS distribution made
    # anti-correlated mate shifts (A: R1 Q10/R2 Q40; B: swapped) compare as
    # identical and approve COMPATIBLE. Per-mate comparisons must catch it.
    from helpers import run_cli

    a = _mate_swapped_bam(tmp_path / "a.bam", 500, q1=10, q2=40, seed=1)
    b = _mate_swapped_bam(tmp_path / "b.bam", 500, q1=40, q2=10, seed=1)
    rows = [
        {
            "dataset_id": ds,
            "pool_group": "g",
            "sample_id": "s",
            "library_id": "l",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "paired_short",
            "input_format": "bam",
            "path1": str(p),
            "processing_id": "p1",
            "reference_id": "hg38",
        }
        for ds, p in (("a", a), ("b", b))
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    qual = {
        (r["stratum"]): r
        for r in comparisons
        if r["metric"] == "mean_read_quality" and r["dataset_a"] == "a"
    }
    assert set(qual) == {"R1", "R2"}, "per-mate strata expected for paired BAM"
    for stratum in ("R1", "R2"):
        assert qual[stratum]["status"] == "EXCEED"
        assert qual[stratum]["value"] == 30.0  # Q10 vs Q40 on both mates
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] == "REVIEW"


def test_reproducible_across_worker_counts(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=90)
    b = _pe_group(tmp_path, "b", seed=91)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    r1, out1, _ = run_cli(tmp_path / "run1", manifest, policy, workers=1, threads=1, expect_code=0)
    r2, out2, _ = run_cli(tmp_path / "run2", manifest, policy, workers=4, threads=4, expect_code=0)
    import polars as pl
    from polars.testing import assert_frame_equal

    for name in out1:
        df1 = pl.from_arrow(pq.read_table(out1[name]))
        df2 = pl.from_arrow(pq.read_table(out2[name]))
        # Exact on purpose: the outputs must be identical. Unlike plain `==`,
        # polars treats null==null and NaN==NaN as equal (verified polars 2.1).
        assert_frame_equal(df1, df2, check_exact=True)


def test_fixed_seed_reproducibility(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=95)
    b = _pe_group(tmp_path, "b", seed=96)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    _, out1, _ = run_cli(tmp_path / "r1", manifest, policy, seed=123, expect_code=0)
    _, out2, _ = run_cli(tmp_path / "r2", manifest, policy, seed=123, expect_code=0)
    m1 = pq.read_table(out1["metrics"]).to_pylist()
    m2 = pq.read_table(out2["metrics"]).to_pylist()
    assert m1 == m2


def test_max_pairs_guard_aborts_before_processing(tmp_path: Path):

    from helpers import run_cli

    groups = [
        (f"g{i}", [_pe_group(tmp_path, f"a{i}_{j}", 20, seed=100 + i * 10 + j) for j in range(6)])
        for i in range(3)
    ]
    manifest = _paired_manifest(tmp_path, groups)  # 3 groups x C(6,2)=15 pairs = 45
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, max_pairs=10, expect_code=2)
    assert "exceeding --max-pairs" in result.stderr
    for path in out.values():
        assert not path.exists()


def test_report_html_written_when_requested(tmp_path: Path):
    import subprocess
    import sys

    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=110)
    b = _pe_group(tmp_path, "b", seed=111)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    outdir = tmp_path / "run"
    result, out, _ = run_cli(outdir, manifest, policy, expect_code=0)
    report = outdir / "report.html"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "seqconsistency.cli",
            "check",
            "--manifest",
            str(manifest),
            "--policy",
            str(policy),
            "--metrics-out",
            str(out["metrics"]),
            "--distributions-out",
            str(out["distributions"]),
            "--comparisons-out",
            str(out["comparisons"]),
            "--findings-out",
            str(out["findings"]),
            "--decisions-out",
            str(out["decisions"]),
            "--provenance-out",
            str(outdir / "provenance2.json"),
            "--report-out",
            str(report),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    html_text = report.read_text()
    assert "seqconsistency pooling-consistency report" in html_text
    assert "COMPATIBLE" in html_text
    assert "not independently verified" in html_text
    assert "http" not in html_text.split("<style>")[0]  # self-contained head


def test_parquet_manifest_end_to_end(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=120)
    b = _pe_group(tmp_path, "b", seed=121)
    rows = [
        {
            "dataset_id": i,
            "pool_group": "g",
            "sample_id": "s",
            "library_id": "l",
            "assay": "WGS",
            "platform": "I",
            "read_type": "paired_short",
            "input_format": "fastq",
            "path1": str(p1),
            "path2": str(p2),
            "processing_id": "p",
        }
        for i, (p1, p2) in zip(("a", "b"), (a, b), strict=True)
    ]
    manifest = write_parquet_manifest(tmp_path / "m.parquet", rows)
    policy = write_policy(tmp_path / "p.yaml")
    run_cli(tmp_path / "run", manifest, policy, expect_code=0)


def test_overwrite_flag_allows_rerun(tmp_path: Path):
    from helpers import run_cli

    a = _pe_group(tmp_path, "a", seed=130)
    b = _pe_group(tmp_path, "b", seed=131)
    manifest = _paired_manifest(tmp_path, [("g1", [a, b])])
    policy = write_policy(tmp_path / "p.yaml")
    run_cli(tmp_path / "run", manifest, policy, expect_code=0)
    import subprocess
    import sys

    outdir = tmp_path / "run"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "seqconsistency.cli",
            "check",
            "--manifest",
            str(manifest),
            "--policy",
            str(policy),
            "--metrics-out",
            str(outdir / "metrics.parquet"),
            "--distributions-out",
            str(outdir / "distributions.parquet"),
            "--comparisons-out",
            str(outdir / "comparisons.parquet"),
            "--findings-out",
            str(outdir / "findings.parquet"),
            "--decisions-out",
            str(outdir / "decisions.parquet"),
            "--provenance-out",
            str(outdir / "provenance.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "exists" in r.stderr
    r2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "seqconsistency.cli",
            "check",
            "--manifest",
            str(manifest),
            "--policy",
            str(policy),
            "--metrics-out",
            str(outdir / "metrics.parquet"),
            "--distributions-out",
            str(outdir / "distributions.parquet"),
            "--comparisons-out",
            str(outdir / "comparisons.parquet"),
            "--findings-out",
            str(outdir / "findings.parquet"),
            "--decisions-out",
            str(outdir / "decisions.parquet"),
            "--provenance-out",
            str(outdir / "provenance.json"),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr


def test_ont_bam_group_has_no_insert_size_requirement(tmp_path: Path):
    from helpers import run_cli

    # make_bam declares RG SM=sample1; the manifest agrees, so no SM finding.
    b1 = make_bam(tmp_path / "o1.bam", n=3000, seed=140, paired=False)
    b2 = make_bam(tmp_path / "o2.bam", n=3000, seed=141, paired=False)
    rows = [
        {
            "dataset_id": f"o{i}",
            "pool_group": "g",
            "sample_id": "sample1",
            "library_id": "l",
            "assay": "WGS",
            "platform": "ONT",
            "read_type": "ont",
            "input_format": "bam",
            "path1": str(p),
            "processing_id": "p",
        }
        for i, p in enumerate((b1, b2))
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=0)
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    metrics = {r["metric"] for r in comparisons}
    assert "insert_size" not in metrics  # unpaired layout: not applicable
    statuses = {r["status"] for r in comparisons}
    assert "INSUFFICIENT_OBSERVATIONS" not in statuses
    # Joint length/quality summary rows exist for the BAM sample.
    dists = pq.read_table(out["distributions"]).to_pylist()
    assert any(r["metric"] == "joint_length_quality" for r in dists)
    # Clipping summary rows exist.
    mets = pq.read_table(out["metrics"]).to_pylist()
    assert any(m["metric"] == "clipping" for m in mets)


def test_large_group_streams_comparisons_and_report(tmp_path):
    # Review round 2, point 6: 30 datasets -> 435 pairs x 12 rows = 5220
    # comparison rows. The streaming stage must write every row to the
    # comparisons Parquet while the HTML report keeps only a bounded preview;
    # retained memory scales with output batches, not with pair counts.
    import subprocess
    import sys

    from helpers import run_cli

    files = [write_pe(tmp_path, f"d{i}", 60, seed=100 + i) for i in range(30)]
    rows = [
        {
            "dataset_id": f"d{i}",
            "pool_group": "g",
            "sample_id": "s1",
            "library_id": "l1",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "paired_short",
            "input_format": "fastq",
            "path1": str(p1),
            "path2": str(p2),
            "processing_id": "p1",
        }
        for i, (p1, p2) in enumerate(files)
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    outdir = tmp_path / "run"
    # Distinct seeds give some pairs a mean-quality distance past the 1.5 Phred
    # test-policy maximum, so the deterministic outcome is REVIEW (exit 1).
    result, out, _ = run_cli(outdir, manifest, policy, workers=2, expect_code=1)
    comparisons = pq.read_table(out["comparisons"]).to_pylist()
    assert len(comparisons) == 435 * 12
    findings_rows = pq.read_table(out["findings"]).to_pylist()
    assert findings_rows  # identity/assumption findings streamed out
    ids = [f["finding_id"] for f in findings_rows]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)  # sequential, unique
    report = outdir / "report.html"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "seqconsistency.cli",
            "check",
            "--manifest",
            str(manifest),
            "--policy",
            str(policy),
            "--metrics-out",
            str(out["metrics"]),
            "--distributions-out",
            str(out["distributions"]),
            "--comparisons-out",
            str(out["comparisons"]),
            "--findings-out",
            str(out["findings"]),
            "--decisions-out",
            str(out["decisions"]),
            "--provenance-out",
            str(outdir / "provenance2.json"),
            "--report-out",
            str(report),
            "--overwrite",
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, r.stderr  # same REVIEW outcome as the first run
    html_text = report.read_text()
    assert f"Showing up to 500 rows of {435 * 12}" in html_text


def test_uncalibrated_findings_stream_completely(tmp_path):
    # Review round 3, point 3: findings emitted per pair as feed() returns
    # them (never accumulated per group). With an uncalibrated policy every
    # PASSING row emits a finding: 12 datasets -> 66 pairs x 12 rows = 792
    # technical findings + 1 identity finding, all written exactly once with
    # sequential unique IDs.
    from helpers import run_cli

    files = [write_pe(tmp_path, f"d{i}", 60, seed=200 + i) for i in range(12)]
    rows = [
        {
            "dataset_id": f"d{i}",
            "pool_group": "g",
            "sample_id": "s1",
            "library_id": "l1",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "paired_short",
            "input_format": "fastq",
            "path1": str(p1),
            "path2": str(p2),
            "processing_id": "p1",
        }
        for i, (p1, p2) in enumerate(files)
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(
        tmp_path / "p.yaml", MINIMAL_POLICY.replace("calibrated: true", "calibrated: false")
    )
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, workers=2, expect_code=1)
    findings = pq.read_table(out["findings"]).to_pylist()
    tech = [f for f in findings if f["component"] == "technical"]
    assert len(tech) == 66 * 12
    assert len(findings) == 66 * 12 + 1  # + identity declaration
    ids = [f["finding_id"] for f in findings]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["overall_decision"] == "REVIEW"
    assert "TECH_THRESHOLD_UNCALIBRATED" in decisions[0]["reason_codes"]


def test_ambiguous_encoding_warns_and_withholds_approval(tmp_path: Path):
    # Review round 4: a dataset whose FASTQ carries the classic Phred+64
    # signature (valid under both encodings) no longer aborts the run — the
    # scan completes, the group gets a QUALITY_ENCODING_AMBIGUOUS warning, and
    # the decision is REVIEW (ambiguity never auto-approves, never blocks).
    from helpers import run_cli

    write_pe(tmp_path, "ok", 200, seed=300)
    amb = tmp_path / "amb.fastq"
    amb.write_bytes(
        b"".join(
            b"@r%d\nACGT\n+\n" % i + bytes([93] * 4) + b"\n" for i in range(200)
        )
    )
    rows = [
        {
            "dataset_id": "ok",
            "pool_group": "g",
            "sample_id": "s1",
            "library_id": "l1",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "merged_short",
            "input_format": "fastq",
            "path1": str(tmp_path / "ok_R1.fastq"),
            "processing_id": "p1",
        },
        {
            "dataset_id": "amb",
            "pool_group": "g",
            "sample_id": "s1",
            "library_id": "l1",
            "assay": "WGS",
            "platform": "ILLUMINA",
            "read_type": "merged_short",
            "input_format": "fastq",
            "path1": str(amb),
            "processing_id": "p1",
        },
    ]
    manifest = write_tsv_manifest(tmp_path / "m.tsv", rows)
    policy = write_policy(tmp_path / "p.yaml")
    result, out, _ = run_cli(tmp_path / "run", manifest, policy, expect_code=1)
    findings = pq.read_table(out["findings"]).to_pylist()
    enc = [f for f in findings if f["reason_code"] == "QUALITY_ENCODING_AMBIGUOUS"]
    assert len(enc) == 1
    assert enc[0]["affected_datasets"] == ["amb"]
    decisions = pq.read_table(out["decisions"]).to_pylist()
    assert decisions[0]["metadata_status"] == "REVIEW"
    assert decisions[0]["overall_decision"] == "REVIEW"
    assert "QUALITY_ENCODING_AMBIGUOUS" in decisions[0]["reason_codes"]
