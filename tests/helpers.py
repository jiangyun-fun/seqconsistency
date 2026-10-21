"""Shared synthetic-data helpers for the seqconsistency test suite."""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np

RNG_SEED = 1234


def make_records(
    n: int,
    length: int = 100,
    gc: float = 0.5,
    qmean: int = 30,
    qstd: int = 5,
    seed: int = RNG_SEED,
    name_prefix: str = "read",
    n_rate: float = 0.0,
    lengths: np.ndarray | None = None,
) -> list[tuple[bytes, bytes, bytes]]:
    """Deterministic FASTQ records: (header, seq, qual) with Phred+33 qualities."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        ln = int(lengths[i]) if lengths is not None else length
        u = rng.random(ln)
        seq = np.where(
            u < (1 - n_rate) * 0.5, ord("A"), np.where(u < (1 - n_rate), ord("T"), ord("N"))
        ).astype(np.uint8)
        # Mix in GC according to target fraction: rewrite half the positions.
        gc_mask = rng.random(ln) < gc
        seq[gc_mask] = rng.choice(np.frombuffer(b"GC", dtype=np.uint8), size=int(gc_mask.sum()))
        q = np.clip(rng.normal(qmean, qstd, ln), 2, 41).astype(np.uint8) + 33
        records.append((f"@{name_prefix}{i}".encode(), seq.tobytes(), q.tobytes()))
    return records


def write_fastq(
    path: Path, records: list[tuple[bytes, bytes, bytes]], gzip_output: bool = False
) -> Path:
    lines = []
    for header, seq, qual in records:
        lines.extend((header, seq, b"+", qual))
    data = b"\n".join(lines) + b"\n"
    if gzip_output:
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


def write_pe(
    dir_path: Path,
    stem: str,
    n: int,
    length: int = 100,
    gc: float = 0.5,
    qmean: int = 30,
    seed: int = RNG_SEED,
    suffix: tuple[str, str] = ("/1", "/2"),
) -> tuple[Path, Path]:
    r1 = make_records(n, length, gc, qmean, seed=seed, name_prefix="r")
    r2 = make_records(n, length, gc, qmean, seed=seed + 1, name_prefix="r")
    p1 = dir_path / f"{stem}_R1.fastq"
    p2 = dir_path / f"{stem}_R2.fastq"
    write_fastq(
        p1, [(h + s1.encode(), s, q) for (h, s, q), s1 in zip(r1, [suffix[0]] * n, strict=True)]
    )
    write_fastq(
        p2, [(h + s2.encode(), s, q) for (h, s, q), s2 in zip(r2, [suffix[1]] * n, strict=True)]
    )
    return p1, p2


def make_bam(
    path: Path,
    n: int,
    read_length: int = 100,
    gc: float = 0.5,
    qmean: int = 30,
    seed: int = RNG_SEED,
    paired: bool = True,
    sort_header: str = "coordinate",
    break_order: bool = False,
    include_secondary: bool = False,
    include_supplementary: bool = False,
    unsorted_records: bool = False,
    no_qualities: bool = False,
    contig_len: int = 1_000_000,
    extra_header: dict | None = None,
    duplicate_fraction: float = 0.0,
    qcfail_fraction: float = 0.0,
) -> Path:
    """Write a synthetic coordinate-sorted BAM (or intentionally broken variants)."""
    import pysam

    header = {
        "HD": {"SO": sort_header},
        "SQ": [{"SN": "chr1", "LN": contig_len, "M5": "6a9a0c1f2b3d4e5f60718293a4b5c6d7"}],
        "RG": [{"ID": "rg1", "SM": "sample1", "LB": "lib1", "PL": "ILLUMINA"}],
    }
    if extra_header:
        for key, value in extra_header.items():
            header[key] = value
    rng = np.random.default_rng(seed)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        positions = sorted(rng.integers(0, contig_len - read_length - 1, size=n))
        if unsorted_records:
            positions = list(rng.permutation(positions))
        if break_order and len(positions) >= 2:
            positions[0], positions[-1] = positions[-1], positions[0]
        for i, pos in enumerate(positions):
            u = rng.random(read_length)
            seq_arr = np.where(u < 0.5, ord("A"), ord("T")).astype(np.uint8)
            gc_mask = rng.random(read_length) < gc
            seq_arr[gc_mask] = rng.choice(
                np.frombuffer(b"GC", dtype=np.uint8), size=int(gc_mask.sum())
            )
            seq = seq_arr.tobytes().decode()
            q = np.clip(rng.normal(qmean, 3, read_length), 2, 41).astype(np.uint8) + 33
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"read{i}"
            a.query_sequence = seq
            a.query_qualities = q.astype(np.uint8) if not no_qualities else None
            a.reference_id = 0
            a.reference_start = int(pos)
            a.mapping_quality = 60
            a.cigarstring = f"{read_length}M"
            flag = 0
            if paired:
                flag |= 0x1 | 0x2 | 0x40  # paired, proper, first-in-pair
                a.template_length = 300
            if duplicate_fraction and rng.random() < duplicate_fraction:
                flag |= 0x400
            if qcfail_fraction and rng.random() < qcfail_fraction:
                flag |= 0x200
            a.flag = flag
            out.write(a)
            if include_secondary:
                b = pysam.AlignedSegment(out.header)
                b.query_name = f"read{i}"
                b.query_sequence = seq
                b.reference_id = 0
                b.reference_start = int(pos)
                b.mapping_quality = 0
                b.cigarstring = f"{read_length}M"
                b.flag = 0x100
                out.write(b)
            if include_supplementary:
                c = pysam.AlignedSegment(out.header)
                c.query_name = f"read{i}"
                c.query_sequence = seq
                c.reference_id = 0
                c.reference_start = int(pos)
                c.mapping_quality = 60
                c.cigarstring = f"{read_length}M"
                c.flag = 0x800
                out.write(c)
    return path


def write_tsv_manifest(path: Path, rows: list[dict]) -> Path:
    columns = [
        "dataset_id",
        "pool_group",
        "sample_id",
        "library_id",
        "assay",
        "platform",
        "read_type",
        "input_format",
        "path1",
        "path2",
        "processing_id",
        "reference_id",
        "fastp_json",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(c) or "") for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_parquet_manifest(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = [
        "dataset_id",
        "pool_group",
        "sample_id",
        "library_id",
        "assay",
        "platform",
        "read_type",
        "input_format",
        "path1",
        "path2",
        "processing_id",
        "reference_id",
        "fastp_json",
    ]
    table = pa.table({c: pa.array([row.get(c) for row in rows], type=pa.string()) for c in columns})
    pq.write_table(table, path)
    return path


MINIMAL_POLICY = """
policy_id: test-policy
policy_version: "1"
identity:
  accept_declared_sample_id: true
  require_independent_verification: false
metadata:
  same_sample_id_required: true
  same_assay_required: true
  distinct_library_ids: review
  distinct_processing_ids: review
  reference_mismatch: incompatible
technical:
  min_observations: 10
  require_thresholds: true
  thresholds:
    - metric: read_length
      method: wasserstein1
      max: 10.0
      action: review
      calibrated: true
    - metric: mean_read_quality
      method: wasserstein1
      max: 1.5
      action: review
      calibrated: true
    - metric: gc_content
      method: jensenshannon_base2
      max: 0.10
      action: review
      calibrated: true
    - metric: n_content
      method: jensenshannon_base2
      max: 0.10
      action: review
      calibrated: true
    - metric: base_quality
      method: jensenshannon_base2
      max: 0.10
      action: review
      calibrated: true
    - metric: nucleotide_composition
      method: jensenshannon_base2
      max: 0.10
      action: review
      calibrated: true
    - metric: mapq
      method: jensenshannon_base2
      max: 0.10
      action: review
      calibrated: true
    - metric: insert_size
      method: wasserstein1
      max: 25.0
      action: review
      calibrated: true
    - metric: merge_retained_fraction
      method: absolute_difference
      max: 0.1
      action: review
      calibrated: true
    - metric: merge_rate
      method: absolute_difference
      max: 0.1
      action: review
      calibrated: true
"""


def write_policy(path: Path, text: str = MINIMAL_POLICY) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(
    tmp_path: Path,
    manifest: Path,
    policy: Path,
    *,
    workers: int = 2,
    threads: int = 2,
    sample_size: int = 5000,
    seed: int = 42,
    max_pairs: int = 10000,
    overwrite: bool = False,
    expect_code: int | None = None,
    extra_env: dict | None = None,
):
    """Invoke the installed CLI in a subprocess (real end-to-end coverage)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = {
        name: tmp_path / f"{name}.parquet"
        for name in ("metrics", "distributions", "comparisons", "findings", "decisions")
    }
    provenance = tmp_path / "provenance.json"
    cmd = [
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
        str(provenance),
        "--workers",
        str(workers),
        "--threads",
        str(threads),
        "--sample-size",
        str(sample_size),
        "--seed",
        str(seed),
        "--max-pairs",
        str(max_pairs),
    ]
    if overwrite:
        cmd.append("--overwrite")
    import os

    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if expect_code is not None:
        assert result.returncode == expect_code, (
            f"expected exit {expect_code}, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result, out, provenance
