"""Generate runnable seqconsistency examples (README § Examples).

Writes synthetic inputs, manifests, and a policy per scenario into --out-dir:

01_paired_short          two Illumina PE runs of one library        -> REVIEW
02_merged_short          two merged-read FASTQ datasets (+fastp)   -> REVIEW
03_ont                   two ONT run datasets                       -> REVIEW
04_bam                   two coordinate-sorted BAM datasets         -> REVIEW
05_shifted_gc            shifted GC content                         -> REVIEW
06_incompatible          different declared samples                 -> INCOMPATIBLE
07_insufficient_identity policy demands independent verification   -> INSUFFICIENT_EVIDENCE
08_multi_group           two pooling groups in one invocation       -> REVIEW+REVIEW

All data is synthetic. Nothing here validates biology; the scenarios demonstrate
mechanics only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parent
POLICY = EXAMPLES_ROOT.parent / "policies" / "example-illustrative.yaml"


def write_fastq(path: Path, records: list[tuple[bytes, bytes, bytes]]) -> None:
    lines: list[bytes] = []
    for h, s, q in records:
        lines.extend((h, s, b"+", q))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(lines) + b"\n")


def make_reads(
    n: int,
    length: int,
    gc: float,
    qmean: float,
    qstd: float,
    seed: int,
    prefix: str = "r",
    lengths: np.ndarray | None = None,
) -> list[tuple[bytes, bytes, bytes]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ln = int(lengths[i]) if lengths is not None else length
        u = rng.random(ln)
        seq = np.where(u < 0.5, ord("A"), ord("T")).astype(np.uint8)
        gc_mask = rng.random(ln) < gc
        seq[gc_mask] = rng.choice(np.frombuffer(b"GC", dtype=np.uint8), size=int(gc_mask.sum()))
        q = np.clip(rng.normal(qmean, qstd, ln), 2, 41).astype(np.uint8) + 33
        out.append((f"@{prefix}{i}".encode(), seq.tobytes(), q.tobytes()))
    return out


def make_pe(
    dir_path: Path, stem: str, n: int, length: int, gc: float, seed: int
) -> tuple[Path, Path]:
    r1 = make_reads(n, length, gc, 32, 5, seed, prefix="inst")
    r2 = make_reads(n, length, gc, 30, 5, seed + 1000, prefix="inst")
    p1 = dir_path / f"{stem}_R1.fastq"
    p2 = dir_path / f"{stem}_R2.fastq"
    write_fastq(p1, [(h + b"/1", s, q) for h, s, q in r1])
    write_fastq(p2, [(h + b"/2", s, q) for h, s, q in r2])
    return p1, p2


def make_bam(path: Path, n: int, seed: int, gc: float = 0.5) -> Path:
    import pysam

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "HD": {"SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 5_000_000, "M5": "0f0b9a2d7d97e7f4a1c1b2c3d4e5f607"}],
        "RG": [{"ID": "rg1", "SM": "sample1", "LB": "lib1"}],
    }
    rng = np.random.default_rng(seed)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i, pos in enumerate(np.sort(rng.integers(0, 5_000_000 - 200, size=n))):
            u = rng.random(100)
            seq = np.where(u < 0.5, ord("A"), ord("T")).astype(np.uint8)
            gc_mask = rng.random(100) < gc
            seq[gc_mask] = rng.choice(np.frombuffer(b"GC", dtype=np.uint8), size=int(gc_mask.sum()))
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"ex{i}"
            a.query_sequence = seq.tobytes().decode()
            a.query_qualities = np.clip(rng.normal(33, 4, 100), 2, 41).astype(np.uint8)
            a.flag = 99
            a.reference_id = 0
            a.reference_start = int(pos)
            a.cigarstring = "100M"
            a.mapping_quality = 60
            a.template_length = 300
            out.write(a)
    return path


def fastp_report(path: Path, before: int, after: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "before_filtering": {"total_reads": before, "q30_rate": 0.85},
                    "after_filtering": {"total_reads": after, "q30_rate": 0.9},
                },
            }
        )
    )
    return path


def write_manifest(path: Path, rows: list[dict]) -> Path:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def pe_row(ds: str, group: str, sample: str, p1: Path, p2: Path, **kw) -> dict:
    row = {
        "dataset_id": ds,
        "pool_group": group,
        "sample_id": sample,
        "library_id": "libA",
        "assay": "WGS",
        "platform": "ILLUMINA",
        "read_type": "paired_short",
        "input_format": "fastq",
        "path1": str(p1),
        "path2": str(p2),
        "processing_id": "trim-v1",
    }
    row.update(kw)
    return row


RUN_COMMANDS = """
Run each example (from the repository root):

  seqconsistency check \\
      --manifest {manifest} --policy {policy} \\
      --metrics-out {out}/metrics.parquet \\
      --distributions-out {out}/distributions.parquet \\
      --comparisons-out {out}/comparisons.parquet \\
      --findings-out {out}/findings.parquet \\
      --decisions-out {out}/decisions.parquet \\
      --provenance-out {out}/provenance.json \\
      --workers 2 --threads 2 --sample-size 20000 --seed 42
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--reads", type=int, default=20000, help="reads per dataset (default 20000)"
    )
    args = parser.parse_args(argv)
    out = args.outdir if hasattr(args, "outdir") else args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    n = args.reads
    notes: list[str] = []

    # 01 — two PE runs of one library.
    d = out / "01_paired_short"
    a = make_pe(d, "run1", n, 100, 0.5, seed=1)
    b = make_pe(d, "run2", n, 100, 0.5, seed=2)
    m = write_manifest(
        d / "manifest.tsv",
        [
            pe_row("run1", "poolA", "sample1", *a),
            pe_row("run2", "poolA", "sample1", *b),
        ],
    )
    notes.append(f"01 paired_short (expect REVIEW — example thresholds uncalibrated): {m}")

    # 02 — merged reads with fastp evidence.
    d = out / "02_merged_short"
    rng = np.random.default_rng(3)
    rows = []
    for i in range(2):
        lengths = rng.integers(180, 260, size=n)
        f = d / f"merged{i}.fastq"
        write_fastq(f, make_reads(n, 0, 0.5, 32, 4, seed=10 + i, lengths=lengths))
        fp = fastp_report(d / f"merged{i}.fastp.json", 2 * n, int(2 * n * (0.86 - 0.02 * i)))
        rows.append(
            {
                "dataset_id": f"merged{i}",
                "pool_group": "poolM",
                "sample_id": "sample1",
                "library_id": "libM",
                "assay": "amplicon",
                "platform": "ILLUMINA",
                "read_type": "merged_short",
                "input_format": "fastq",
                "path1": str(f),
                "processing_id": "vsearch-v1",
                "fastp_json": str(fp),
            }
        )
    m = write_manifest(d / "manifest.tsv", rows)
    notes.append(f"02 merged_short (expect REVIEW — uncalibrated thresholds): {m}")

    # 03 — ONT.
    d = out / "03_ont"
    rows = []
    for i in range(2):
        lengths = np.clip(np.random.default_rng(30 + i).lognormal(9.0, 0.6, n), 200, 150000).astype(
            int
        )
        f = d / f"ont{i}.fastq"
        write_fastq(f, make_reads(n, 0, 0.5, 12, 3, seed=30 + i, lengths=lengths))
        rows.append(
            {
                "dataset_id": f"ont{i}",
                "pool_group": "poolO",
                "sample_id": "sample1",
                "library_id": "libO",
                "assay": "WGS",
                "platform": "ONT",
                "read_type": "ont",
                "input_format": "fastq",
                "path1": str(f),
                "processing_id": "dorado-v1",
            }
        )
    m = write_manifest(d / "manifest.tsv", rows)
    notes.append(f"03 ont (expect REVIEW — uncalibrated thresholds): {m}")

    # 04 — BAM.
    d = out / "04_bam"
    b1 = make_bam(d / "run1.bam", n, seed=40)
    b2 = make_bam(d / "run2.bam", n, seed=41)
    m = write_manifest(
        d / "manifest.tsv",
        [
            {
                "dataset_id": "bam1",
                "pool_group": "poolB",
                "sample_id": "sample1",
                "library_id": "libB",
                "assay": "WGS",
                "platform": "ILLUMINA",
                "read_type": "paired_short",
                "input_format": "bam",
                "path1": str(b1),
                "processing_id": "bwa-v1",
                "reference_id": "chr1-demo",
            },
            {
                "dataset_id": "bam2",
                "pool_group": "poolB",
                "sample_id": "sample1",
                "library_id": "libB",
                "assay": "WGS",
                "platform": "ILLUMINA",
                "read_type": "paired_short",
                "input_format": "bam",
                "path1": str(b2),
                "processing_id": "bwa-v1",
                "reference_id": "chr1-demo",
            },
        ],
    )
    notes.append(f"04 bam (expect REVIEW — uncalibrated thresholds): {m}")

    # 05 — shifted GC.
    d = out / "05_shifted_gc"
    a = make_pe(d, "normal", n, 100, 0.42, seed=50)
    b = make_pe(d, "shifted", n, 100, 0.60, seed=51)
    m = write_manifest(
        d / "manifest.tsv",
        [
            pe_row("normal", "poolG", "sample1", *a),
            pe_row("shifted", "poolG", "sample1", *b),
        ],
    )
    notes.append(f"05 shifted GC (expect REVIEW with gc_content exceed): {m}")

    # 06 — different declared samples.
    d = out / "06_incompatible"
    a = make_pe(d, "s1", n, 100, 0.5, seed=60)
    b = make_pe(d, "s2", n, 100, 0.5, seed=61)
    m = write_manifest(
        d / "manifest.tsv",
        [
            pe_row("donorA", "poolX", "donorA", *a),
            pe_row("donorB", "poolX", "donorB", *b),
        ],
    )
    notes.append(f"06 incompatible samples (expect INCOMPATIBLE): {m}")

    # 07 — policy demands independent verification.
    d = out / "07_insufficient_identity"
    a = make_pe(d, "run1", n, 100, 0.5, seed=70)
    b = make_pe(d, "run2", n, 100, 0.5, seed=71)
    write_manifest(
        d / "manifest.tsv",
        [
            pe_row("run1", "poolI", "sample1", *a),
            pe_row("run2", "poolI", "sample1", *b),
        ],
    )
    strict = d / "policy-strict.yaml"
    strict.write_text(
        POLICY.read_text().replace(
            "require_independent_verification: false",
            "require_independent_verification: true",
        )
    )
    notes.append(
        f"07 insufficient identity (expect INSUFFICIENT_EVIDENCE): "
        f"{d / 'manifest.tsv'} with {strict}"
    )

    # 08 — two groups in one invocation.
    d = out / "08_multi_group"
    a = make_pe(d, "a1", n, 100, 0.5, seed=80)
    b = make_pe(d, "a2", n, 100, 0.5, seed=81)
    c = make_pe(d, "b1", n, 100, 0.45, seed=82)
    e = make_pe(d, "b2", n, 100, 0.62, seed=83)
    m = write_manifest(
        d / "manifest.tsv",
        [
            pe_row("g1_a", "group1", "sample1", *a),
            pe_row("g1_b", "group1", "sample1", *b),
            pe_row("g2_a", "group2", "sample2", *c),
            pe_row("g2_b", "group2", "sample2", *e),
        ],
    )
    notes.append(f"08 multi-group (expect both REVIEW under uncalibrated thresholds): {m}")

    readme = out / "README.txt"
    readme.write_text(
        "seqconsistency synthetic examples\n=================================\n\n"
        + "\n".join(notes)
        + "\n\nExample thresholds are ILLUSTRATIVE AND UNCALIBRATED; expected decisions\n"
        "demonstrate mechanics only.\n"
        + "\nPer-example run command:\n"
        + RUN_COMMANDS.format(
            manifest="<example-dir>/manifest.tsv",
            policy=str(POLICY),
            out="<example-dir>/out",
        )
    )
    print(f"wrote examples under {out}")
    for note in notes:
        print(" ", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
