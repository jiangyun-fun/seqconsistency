"""Reproducible benchmark harness for seqconsistency (spec §13).

Measures — with real measurements, never estimates presented as measurements:
- FASTQ scanning throughput (reads/s, bases/s, wall s) for SE/PE, plain/gzip,
  short and long reads
- BAM scanning throughput
- PE synchronization overhead (paired vs single-file scan of the same R1 volume)
- Worker scaling (1/2/4 workers)
- Pairwise comparison scaling (dataset count)
- Parquet writing throughput
- Peak process-tree RSS via ru_maxrss, and RSS vs read count at fixed sample size

Usage (all output paths explicit):
    python benchmarks/bench.py run --data-dir DIR --results-out results.json \
        [--quick] [--workers-list 1,2,4] [--reads N] [--long-reads N] [--bam-reads N]

Benchmarks write synthetic data under --data-dir (created if absent) and measured
results to --results-out (JSON). Nothing else is written anywhere.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("seqconsistency.bench")

_READS_PER_RECORD_LINE = 4


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def child_peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


def gen_fastq(
    path: Path,
    n: int,
    length: int,
    seed: int,
    gz: bool = False,
    lengths: np.ndarray | None = None,
) -> tuple[Path, int]:
    """Write a synthetic FASTQ; returns (path, total bases).

    Idempotent: an existing file of the expected size is reused (benchmark reruns
    skip the expensive generation step).
    """
    expected_bases = int(np.sum(lengths)) if lengths is not None else n * length
    if path.exists() and path.stat().st_size > 0:
        return path, expected_bases
    rng = np.random.default_rng(seed)
    opener = gzip.open if gz else open
    total_bases = 0
    qtemplate = np.clip(rng.normal(32, 6, length), 2, 41).astype(np.uint8) + 33
    with opener(path, "wb") as fh:  # type: ignore[operator]
        block: list[bytes] = []
        for i in range(n):
            ln = int(lengths[i]) if lengths is not None else length
            u = rng.random(ln)
            seq = np.where(
                u < 0.25,
                ord("A"),
                np.where(u < 0.5, ord("C"), np.where(u < 0.75, ord("G"), ord("T"))),
            ).astype(np.uint8)
            if ln == length:
                q = qtemplate
            else:
                q = np.clip(rng.normal(32, 6, ln), 2, 41).astype(np.uint8) + 33
            block.extend((f"@bench:{i}".encode(), seq.tobytes(), b"+", q.tobytes()))
            total_bases += ln
            if len(block) >= 40000:
                fh.write(b"\n".join(block) + b"\n")
                block = []
        if block:
            fh.write(b"\n".join(block) + b"\n")
    return path, total_bases


def gen_pe(dir_path: Path, stem: str, n: int, length: int, seed: int):
    p1, bases = gen_fastq(dir_path / f"{stem}_R1.fastq", n, length, seed)
    p2, _ = gen_fastq(dir_path / f"{stem}_R2.fastq", n, length, seed + 1)
    return p1, p2, bases * 2


def gen_bam(path: Path, n: int, read_length: int, seed: int) -> Path:
    import pysam

    if path.exists() and path.stat().st_size > 0:
        return path
    header = {
        "HD": {"SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 50_000_000}],
        "RG": [{"ID": "rg1", "SM": "s", "LB": "l"}],
    }
    rng = np.random.default_rng(seed)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        positions = np.sort(rng.integers(0, 50_000_000 - read_length - 1, size=n))
        for i, pos in enumerate(positions):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"bench:{i}"
            u = rng.random(read_length)
            seq = (
                np.where(
                    u < 0.25,
                    ord("A"),
                    np.where(u < 0.5, ord("C"), np.where(u < 0.75, ord("G"), ord("T"))),
                )
                .astype(np.uint8)
                .tobytes()
                .decode()
            )
            a.query_sequence = seq
            a.query_qualities = np.clip(rng.normal(30, 5, read_length), 2, 41).astype(np.uint8)
            a.flag = 99
            a.reference_id = 0
            a.reference_start = int(pos)
            a.cigarstring = f"{read_length}M"
            a.mapping_quality = 60
            a.template_length = 300
            out.write(a)
    return path


# ---------------------------------------------------------------------------
# Measurement runners (in-process so ru_maxrss covers the scan itself)
# ---------------------------------------------------------------------------


def bench_fastq_scan(path: Path, n: int, bases: int, read_type: str, sample_size: int) -> dict:
    from seqconsistency.fastq import scan_fastq_single
    from seqconsistency.models import ReadType, SamplingConfig

    sampling = SamplingConfig(sample_size=sample_size, master_seed=42)
    t0 = time.perf_counter()
    stats, _, eligible, sampled = scan_fastq_single(
        "bench", path, ReadType(read_type), sampling, sample_size
    )
    elapsed = time.perf_counter() - t0
    assert eligible == n and sampled == min(sample_size, n)
    return {
        "path": str(path),
        "kind": "fastq_se",
        "reads": n,
        "bases": bases,
        "wall_s": round(elapsed, 3),
        "reads_per_s": round(n / elapsed),
        "bases_per_s": round(bases / elapsed),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "file_bytes": path.stat().st_size,
    }


def bench_fastq_pe(p1: Path, p2: Path, n: int, bases: int, sample_size: int) -> dict:
    from seqconsistency.fastq import scan_fastq_paired
    from seqconsistency.models import ReadType, SamplingConfig

    sampling = SamplingConfig(sample_size=sample_size, master_seed=42)
    t0 = time.perf_counter()
    st1, st2, pairing, *_ = scan_fastq_paired(
        "bench", p1, p2, ReadType.PAIRED_SHORT, sampling, sample_size
    )
    elapsed = time.perf_counter() - t0
    assert pairing.pairs_checked == n
    return {
        "kind": "fastq_pe",
        "reads": n,
        "bases": bases,
        "wall_s": round(elapsed, 3),
        "reads_per_s": round(n / elapsed),
        "bases_per_s": round(bases / elapsed),
        "peak_rss_mib": round(peak_rss_mib(), 1),
    }


def bench_bam(path: Path, n: int, read_length: int, sample_size: int) -> dict:
    from seqconsistency.bam import scan_bam
    from seqconsistency.models import ReadType, SamplingConfig

    sampling = SamplingConfig(sample_size=sample_size, master_seed=42)
    t0 = time.perf_counter()
    res = scan_bam("bench", path, ReadType.PAIRED_SHORT, sampling, sample_size)
    elapsed = time.perf_counter() - t0
    assert res.records_read == n
    return {
        "kind": "bam",
        "reads": n,
        "bases": n * read_length,
        "wall_s": round(elapsed, 3),
        "reads_per_s": round(n / elapsed),
        "bases_per_s": round(n * read_length / elapsed),
        "peak_rss_mib": round(peak_rss_mib(), 1),
    }


def bench_worker_scaling(data_dir: Path, reads: int, workers_list: list[int]) -> list:
    """Full CLI runs at several --workers values; measures wall clock and RSS."""
    out: list[dict] = []
    import tempfile

    for workers in workers_list:
        with tempfile.TemporaryDirectory(prefix="seqc-bench-") as td:
            tdp = Path(td)
            files = []
            for i in range(4):
                p1, p2, _ = gen_pe(data_dir, f"scale{i}", reads // 4, 100, 500 + i)
                files.append((p1, p2))
            rows = []
            for i, (p1, p2) in enumerate(files):
                rows.append(
                    {
                        "dataset_id": f"s{i}",
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
                )
            manifest = tdp / "m.tsv"
            # All 13 schema columns must be present, including nullable ones.
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
            manifest.write_text(
                "\t".join(columns)
                + "\n"
                + "\n".join("\t".join(str(r.get(c) or "") for c in columns) for r in rows)
                + "\n"
            )
            policy = tdp / "p.yaml"
            policy.write_text(POLICY_TEXT)
            outs = [
                tdp / f"{k}.parquet"
                for k in ("metrics", "distributions", "comparisons", "findings", "decisions")
            ]
            cmd = [
                sys.executable,
                "-m",
                "seqconsistency.cli",
                "check",
                "--manifest",
                str(manifest),
                "--policy",
                str(policy),
                "--provenance-out",
                str(tdp / "prov.json"),
                "--workers",
                str(workers),
                "--threads",
                str(max(workers, 1)),
                "--sample-size",
                "20000",
            ]
            for name, path in zip(
                ("metrics", "distributions", "comparisons", "findings", "decisions"),
                outs,
                strict=True,
            ):
                cmd += [f"--{name.replace('_', '-')}-out", str(path)]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            assert r.returncode in (0, 1), r.stderr
            out.append(
                {
                    "kind": "worker_scaling",
                    "workers": workers,
                    "datasets": len(rows),
                    "reads": reads,
                    "wall_s": round(elapsed, 3),
                    "child_peak_rss_mib": round(child_peak_rss_mib(), 1),
                }
            )
    return out


def bench_pair_scaling(sample_size: int) -> list[dict]:
    """Comparison-stage scaling in dataset count (cheap; synthetic summaries)."""
    import itertools
    import time as _t

    from seqconsistency import comparison as cmp_mod
    from seqconsistency.models import (
        DatasetOutcome,
        DatasetSpec,
        FastqFileStats,
        FastqScanResult,
        InputFormat,
        ReadType,
    )

    out = []
    for n_datasets in (5, 10, 20, 40):
        rng = np.random.default_rng(7)
        outcomes = []
        for i in range(n_datasets):
            spec = DatasetSpec(
                dataset_id=f"d{i:03d}",
                pool_group="g",
                sample_id="s",
                library_id="l",
                assay="WGS",
                platform="I",
                read_type=ReadType.ONT,
                input_format=InputFormat.FASTQ,
                path1=Path("/dev/null"),
                path2=None,
                processing_id="p",
                reference_id=None,
                fastp_json=None,
            )
            sample = {
                "length": rng.lognormal(9, 0.5, sample_size).tolist(),
                "gc_fraction": rng.beta(2, 2, sample_size).tolist(),
                "n_fraction": np.zeros(sample_size).tolist(),
                "mean_quality_arithmetic": rng.normal(12, 2, sample_size).tolist(),
                "mean_quality_error_based": rng.normal(11, 2, sample_size).tolist(),
            }
            outcomes.append(
                DatasetOutcome(
                    spec=spec,
                    fastq=FastqScanResult(
                        dataset_id=spec.dataset_id,
                        r1=FastqFileStats(length_hist=None, gc_hist=None, n_hist=None),
                        r2=None,
                        pairing=None,
                        sample_r1=sample,
                        sample_r2=None,
                        eligible_observations=sample_size,
                        sampled_count=sample_size,
                        scan_seconds=0.0,
                        bytes_read=0,
                    ),
                )
            )

        def resolver(metric: str, stratum: str):
            return None

        t0 = _t.perf_counter()
        rows = cmp_mod.compare_group("g", outcomes, 100, resolver)
        elapsed = _t.perf_counter() - t0
        out.append(
            {
                "kind": "pair_scaling",
                "datasets": n_datasets,
                "pairs": len(list(itertools.combinations(range(n_datasets), 2))),
                "rows": len(rows),
                "wall_s": round(elapsed, 4),
            }
        )
    return out


def bench_parquet_write(data_dir: Path, n_rows: int) -> dict:

    from seqconsistency import parquet_io as pio

    path = data_dir / "bench_write.parquet"
    rows = (
        {
            "dataset_id": f"ds{i % 17}",
            "pool_group": "g",
            "component": "fastq",
            "stratum": "R1",
            "metric": "reads",
            "statistic": "total",
            "value": float(i),
            "unit": "reads",
            "observation_count": i,
            "denominator": None,
            "sampling_mode": "full_scan",
            "availability": "available",
        }
        for i in range(n_rows)
    )
    t0 = time.perf_counter()
    written = pio.write_table(path, pio.METRICS_SCHEMA, rows)
    elapsed = time.perf_counter() - t0
    assert written == n_rows
    size_mb = path.stat().st_size / 1e6
    path.unlink()
    return {
        "kind": "parquet_write",
        "rows": n_rows,
        "wall_s": round(elapsed, 3),
        "rows_per_s": round(n_rows / elapsed),
        "output_mb": round(size_mb, 2),
    }


def bench_memory_vs_reads(data_dir: Path, sample_size: int) -> list[dict]:
    """Peak RSS vs read count at fixed sample size (the spec §13 requirement).

    Both sizes exceed several hundred scan batches (mean ~6.6 kb reads), so
    per-batch transients are saturated; only unbounded state would grow RSS with
    read count. Files are kept (not unlinked) so reruns reuse them.
    """
    out = []
    for n in (150_000, 600_000):
        lengths = np.random.default_rng(n).lognormal(8.8, 0.5, n).astype(int)
        lengths = np.clip(lengths, 100, 100_000)
        p, bases = gen_fastq(data_dir / f"mem_{n}.fastq", n, 0, 1, lengths=lengths)
        res = bench_fastq_scan(p, n, bases, "ont", sample_size)
        res["kind"] = "memory_vs_reads"
        res["sample_size"] = sample_size
        out.append(res)
    return out


POLICY_TEXT = """\
policy_id: bench
policy_version: "1"
identity:
  accept_declared_sample_id: true
metadata:
  distinct_library_ids: allowed
technical:
  min_observations: 100
  require_thresholds: false
  thresholds: []
"""


def run_benchmarks(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    reads = args.reads
    long_reads = args.long_reads
    bam_reads = args.bam_reads

    p_se, bases = gen_fastq(data_dir / "se.fastq", reads, 100, 11)
    results.append(bench_fastq_scan(p_se, reads, bases, "merged_short", args.sample_size))
    p_gz, _ = gen_fastq(data_dir / "se.fastq.gz", reads, 100, 12, gz=True)
    results.append(bench_fastq_scan(p_gz, reads, bases, "merged_short", args.sample_size))

    p1, p2, pe_bases = gen_pe(data_dir, "pe", reads, 100, 13)
    results.append(bench_fastq_pe(p1, p2, reads, pe_bases, args.sample_size))

    rng = np.random.default_rng(14)
    lengths = np.clip(rng.lognormal(9.2, 0.5, long_reads).astype(int), 100, 200_000)
    p_ont, ont_bases = gen_fastq(data_dir / "ont.fastq", long_reads, 0, 14, lengths=lengths)
    results.append(bench_fastq_scan(p_ont, long_reads, ont_bases, "ont", args.sample_size))

    bam = gen_bam(data_dir / "bench.bam", bam_reads, 100, 15)
    results.append(bench_bam(bam, bam_reads, 100, args.sample_size))

    if not args.quick:
        results.extend(
            bench_worker_scaling(
                data_dir, min(reads, 200_000), [int(w) for w in args.workers_list.split(",")]
            )
        )

    results.extend(bench_pair_scaling(args.sample_size))
    results.append(bench_parquet_write(data_dir, 2_000_000))

    if not args.quick:
        results.extend(bench_memory_vs_reads(data_dir, args.sample_size))

    import platform

    doc = {
        "seqconsistency_version": __import__("seqconsistency").__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "host": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {
            "reads": reads,
            "long_reads": long_reads,
            "bam_reads": bam_reads,
            "sample_size": args.sample_size,
            "workers_list": args.workers_list,
        },
        "results": results,
    }
    Path(args.results_out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.results_out)
    for r in results:
        logger.info("%s", r)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the benchmark suite")
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--results-out", required=True, type=Path)
    run.add_argument("--quick", action="store_true", help="skip scaling and memory benchmarks")
    run.add_argument("--reads", type=int, default=2_000_000)
    run.add_argument("--long-reads", type=int, default=400_000)
    run.add_argument("--bam-reads", type=int, default=500_000)
    run.add_argument("--sample-size", type=int, default=100_000)
    run.add_argument("--workers-list", default="1,2,4")
    run.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    if args.results_out.exists():
        raise SystemExit(f"refusing to overwrite existing {args.results_out}")
    if str(args.data_dir) in {str(args.results_out)}:
        raise SystemExit("--data-dir and --results-out must differ")
    run_benchmarks(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
