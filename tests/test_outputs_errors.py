"""Output schemas, staged writes, path validation, and worker failure handling."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from helpers import make_records, write_fastq, write_pe, write_policy, write_tsv_manifest
from seqconsistency import parquet_io as pio
from seqconsistency.errors import PathError
from seqconsistency.pipeline import CheckConfig, validate_output_paths


def _config(tmp_path: Path, **kw) -> CheckConfig:
    defaults = dict(
        manifest=tmp_path / "m.tsv",
        policy=tmp_path / "p.yaml",
        metrics_out=tmp_path / "metrics.parquet",
        distributions_out=tmp_path / "dist.parquet",
        comparisons_out=tmp_path / "cmp.parquet",
        findings_out=tmp_path / "find.parquet",
        decisions_out=tmp_path / "dec.parquet",
        provenance_out=tmp_path / "prov.json",
        report_out=None,
        workers=1,
        threads=1,
        sample_size=100,
        seed=42,
        max_pairs=100,
        overwrite=False,
        hash_inputs=False,
    )
    defaults.update(kw)
    return CheckConfig(**defaults)


def _inputs(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "a", 60, seed=1)
    q1, q2 = write_pe(tmp_path, "b", 60, seed=2)
    manifest = write_tsv_manifest(
        tmp_path / "m.tsv",
        [
            {
                "dataset_id": "a",
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
            },
            {
                "dataset_id": "b",
                "pool_group": "g",
                "sample_id": "s",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "paired_short",
                "input_format": "fastq",
                "path1": str(q1),
                "path2": str(q2),
                "processing_id": "p",
            },
        ],
    )
    policy = write_policy(tmp_path / "p.yaml")
    return manifest, policy


def test_existing_output_rejected_without_overwrite(tmp_path: Path):
    manifest, policy = _inputs(tmp_path)
    config = _config(tmp_path, manifest=manifest, policy=policy)
    config.metrics_out.write_bytes(b"x")
    with pytest.raises(PathError, match="output file exists"):
        validate_output_paths(config, [])


def test_duplicate_output_paths_rejected(tmp_path: Path):
    config = _config(
        tmp_path,
        metrics_out=tmp_path / "same.parquet",
        distributions_out=tmp_path / "other.parquet",
        comparisons_out=tmp_path / "same.parquet",
    )
    with pytest.raises(PathError, match="duplicate output path"):
        validate_output_paths(config, [])


def test_duplicate_output_paths_via_alias_rejected(tmp_path: Path):
    target = tmp_path / "x.parquet"
    alias = tmp_path / "link.parquet"
    config = _config(tmp_path, metrics_out=target, distributions_out=alias)
    import os

    os.symlink(target, alias)
    with pytest.raises(PathError, match="duplicate output path"):
        validate_output_paths(config, [])


def test_missing_parent_dir_rejected(tmp_path: Path):
    config = _config(tmp_path, metrics_out=tmp_path / "no" / "such" / "dir" / "m.parquet")
    with pytest.raises(PathError, match="parent directory does not exist"):
        validate_output_paths(config, [])


def test_output_colliding_with_input_rejected(tmp_path: Path):
    manifest, policy = _inputs(tmp_path)
    config = _config(tmp_path, manifest=manifest, policy=policy)
    config.metrics_out = Path(manifest_row_path(manifest))
    with pytest.raises(PathError, match="collides with an input file"):
        validate_output_paths(config, [config.metrics_out])


def manifest_row_path(manifest: Path) -> str:
    import csv

    with open(manifest) as fh:
        row = next(csv.DictReader(fh, delimiter="\t"))
    return row["path1"]


def test_staged_writer_publishes_and_aborts(tmp_path: Path):
    out = tmp_path / "t.parquet"
    w = pio.StagedWriter(out, pio.METRICS_SCHEMA)
    w.write_rows(
        [
            {
                "dataset_id": "a",
                "pool_group": "g",
                "component": "fastq",
                "stratum": "R1",
                "metric": "reads",
                "statistic": "total",
                "value": 1.0,
                "unit": "reads",
                "observation_count": 1,
                "denominator": None,
                "sampling_mode": "full_scan",
                "availability": "available",
            }
        ]
    )
    assert w.close_publish() == 1
    assert out.exists()
    assert not list(tmp_path.glob(".*tmp*"))

    w2 = pio.StagedWriter(out, pio.METRICS_SCHEMA)
    staged = w2.staged_path
    w2.abort()
    assert not staged.exists()


def test_write_table_empty_schema_valid(tmp_path: Path):
    out = tmp_path / "empty.parquet"
    n = pio.write_table(out, pio.DECISIONS_SCHEMA, [])
    assert n == 0
    table = pq.read_table(out)
    assert table.num_rows == 0
    assert table.schema.names == pio.DECISIONS_SCHEMA.names


def test_schema_metadata_versioned():
    for schema in (
        pio.METRICS_SCHEMA,
        pio.DISTRIBUTIONS_SCHEMA,
        pio.COMPARISONS_SCHEMA,
        pio.FINDINGS_SCHEMA,
        pio.DECISIONS_SCHEMA,
    ):
        md = dict(schema.metadata)
        assert md[b"schema"] == b"seqconsistency"
        assert md[b"schema_version"] == b"1"


def test_write_table_read_back_pandas_polars_duckdb(tmp_path: Path):
    out = tmp_path / "t.parquet"
    pio.write_table(
        out,
        pio.METRICS_SCHEMA,
        [
            {
                "dataset_id": "a",
                "pool_group": "g",
                "component": "fastq",
                "stratum": "R1",
                "metric": "reads",
                "statistic": "total",
                "value": 5.0,
                "unit": "reads",
                "observation_count": 5,
                "denominator": None,
                "sampling_mode": "full_scan",
                "availability": "available",
            }
        ],
    )
    import duckdb

    rel = duckdb.sql(f"select count(*) c from '{out}'")
    assert rel.fetchone()[0] == 1


def test_worker_exception_propagates_and_cleans_up(tmp_path):
    # Manifest pointing at a malformed FASTQ: run must fail with exit 2 and leave
    # no partial outputs.
    import subprocess
    import sys

    tmp = Path(tmp_path)
    bad = tmp / "bad.fastq"
    bad.write_bytes(b"@r1\nACGT\n+\nIII\n@r2\nACG\n+\nIII\n")  # len mismatch
    good = write_fastq(tmp / "good.fastq", make_records(30, seed=3))
    manifest = write_tsv_manifest(
        tmp / "m.tsv",
        [
            {
                "dataset_id": "a",
                "pool_group": "g",
                "sample_id": "s",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "merged_short",
                "input_format": "fastq",
                "path1": str(good),
            },
            {
                "dataset_id": "b",
                "pool_group": "g",
                "sample_id": "s",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "merged_short",
                "input_format": "fastq",
                "path1": str(bad),
            },
        ],
    )
    policy = write_policy(tmp / "p.yaml")
    outs = {
        k: tmp / f"{k}.parquet"
        for k in ("metrics", "distributions", "comparisons", "findings", "decisions")
    }
    cmd = [
        sys.executable,
        "-m",
        "seqconsistency.cli",
        "check",
        "--manifest",
        str(manifest),
        "--policy",
        str(policy),
    ]
    for name, path in outs.items():
        cmd += [f"--{name.replace('_', '-')}-out", str(path)]
    cmd += ["--provenance-out", str(tmp / "prov.json")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2
    assert "sequence length" in result.stderr
    for path in outs.values():
        assert not path.exists(), f"partial output left behind: {path}"
    assert not (tmp / "prov.json").exists()
    assert not list(tmp.glob(".*tmp*")), "staged temporaries leaked"


def test_output_colliding_with_manifest_rejected_even_with_overwrite(tmp_path: Path):
    manifest, policy = _inputs(tmp_path)
    config = _config(
        tmp_path,
        manifest=manifest,
        policy=policy,
        metrics_out=manifest,
        overwrite=True,
    )
    with pytest.raises(PathError, match="collides with an input file"):
        validate_output_paths(config, [manifest, policy])


def test_output_colliding_with_policy_rejected(tmp_path: Path):
    manifest, policy = _inputs(tmp_path)
    config = _config(tmp_path, manifest=manifest, policy=policy, findings_out=policy)
    with pytest.raises(PathError, match="collides with an input file"):
        validate_output_paths(config, [manifest, policy])


def test_output_colliding_with_fastp_report_rejected(tmp_path: Path):
    import json

    fastp = tmp_path / "report.json"
    fastp.write_text(json.dumps({"summary": {}}))
    p1, p2 = write_pe(tmp_path, "x", 10, seed=9)
    manifest = write_tsv_manifest(
        tmp_path / "m.tsv",
        [
            {
                "dataset_id": "a",
                "pool_group": "g",
                "sample_id": "s",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "merged_short",
                "input_format": "fastq",
                "path1": str(p1),
                "fastp_json": str(fastp),
            },
            {
                "dataset_id": "b",
                "pool_group": "g",
                "sample_id": "s",
                "library_id": "l",
                "assay": "WGS",
                "platform": "I",
                "read_type": "merged_short",
                "input_format": "fastq",
                "path1": str(p2),
                "fastp_json": str(fastp),
            },
        ],
    )
    policy = write_policy(tmp_path / "p.yaml")
    config = _config(tmp_path, manifest=manifest, policy=policy, metrics_out=fastp)
    with pytest.raises(PathError, match="collides with an input file"):
        validate_output_paths(config, [manifest, policy, fastp])
