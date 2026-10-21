"""Manifest loading and strict validation tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import (
    make_bam,
    make_records,
    write_fastq,
    write_parquet_manifest,
    write_pe,
    write_tsv_manifest,
)
from seqconsistency.errors import ManifestError
from seqconsistency.manifest import load_manifest


def _base_row(dataset_id: str, group: str = "g1", **overrides) -> dict:
    row = {
        "dataset_id": dataset_id,
        "pool_group": group,
        "sample_id": "s1",
        "library_id": "l1",
        "assay": "WGS",
        "platform": "ILLUMINA",
        "read_type": "merged_short",
        "input_format": "fastq",
        "path1": None,
        "path2": None,
        "processing_id": "p1",
        "reference_id": None,
        "fastp_json": None,
    }
    row.update(overrides)
    return row


def _two_se_files(tmp_path: Path) -> list[Path]:
    return [write_fastq(tmp_path / f"ds{i}.fastq", make_records(20, seed=i)) for i in range(2)]


def test_valid_tsv_manifest(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsB", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    result = load_manifest(m)
    assert [s.dataset_id for s in result.specs] == ["dsA", "dsB"]
    assert result.row_count == 2
    assert len(result.sha256) == 64


def test_valid_parquet_manifest(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsB", path1=str(files[1])),
    ]
    m = write_parquet_manifest(tmp_path / "m.parquet", rows)
    result = load_manifest(m)
    assert len(result.specs) == 2


def test_relative_paths_resolve_against_manifest_parent(tmp_path: Path):
    (tmp_path / "data").mkdir()
    files = [
        write_fastq(tmp_path / "data" / f"ds{i}.fastq", make_records(10, seed=i)) for i in range(2)
    ]
    rows = [
        _base_row("dsA", path1="data/ds0.fastq"),
        _base_row("dsB", path1="data/ds1.fastq"),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    result = load_manifest(m)
    assert result.specs[0].path1 == files[0].resolve()


def test_duplicate_dataset_ids_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsA", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="duplicate dataset_id"):
        load_manifest(m)


def test_missing_required_column_rejected(tmp_path: Path):
    p = tmp_path / "m.tsv"
    p.write_text("dataset_id\tpool_group\na\tg\n")
    with pytest.raises(ManifestError, match="missing required column"):
        load_manifest(p)


def test_unknown_column_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0]), extra_col="x"),
        _base_row("dsB", path1=str(files[1]), extra_col="y"),
    ]
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
        "extra_col",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(c) or "") for c in columns))
    p = tmp_path / "m.tsv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ManifestError, match="unknown column"):
        load_manifest(p)


def test_bad_read_type_enum(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0]), read_type="nanopore"),
        _base_row("dsB", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="read_type"):
        load_manifest(m)


def test_pe_requires_path2(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", read_type="paired_short", path1=str(files[0])),
        _base_row("dsB", read_type="paired_short", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="path2 is required"):
        load_manifest(m)


def test_se_with_path2_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0]), path2=str(files[1])),
        _base_row("dsB", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="path2 must be empty"):
        load_manifest(m)


def test_missing_file_rejected(tmp_path: Path):
    rows = [
        _base_row("dsA", path1=str(tmp_path / "nope.fastq")),
        _base_row("dsB", path1=str(tmp_path / "nope2.fastq")),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(m)


def test_group_needs_two_datasets(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsB", group="g2", path1=str(files[1])),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="at least"):
        load_manifest(m)


def test_mixed_profiles_in_group_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsB", path1=str(files[1]), read_type="ont"),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="mixes comparison profiles"):
        load_manifest(m)


def test_mixed_formats_in_group_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    bam = make_bam(tmp_path / "x.bam", n=5, seed=1)
    rows = [
        _base_row("dsA", path1=str(files[0])),
        _base_row("dsB", path1=str(bam), read_type="paired_short", input_format="bam"),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="mixes input formats"):
        load_manifest(m)


def test_same_file_twice_in_group_rejected(tmp_path: Path):
    f = write_fastq(tmp_path / "same.fastq", make_records(10, seed=0))
    rows = [
        _base_row("dsA", path1=str(f)),
        _base_row("dsB", path1=str(f)),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="same underlying file"):
        load_manifest(m)


def test_same_file_via_alias_rejected(tmp_path: Path):
    f = write_fastq(tmp_path / "same.fastq", make_records(10, seed=0))
    alias = tmp_path / "alias.fastq"
    alias.hardlink_to(f)
    rows = [
        _base_row("dsA", path1=str(f)),
        _base_row("dsB", path1=str(alias)),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="same underlying file"):
        load_manifest(m)


def test_pe_r1_reused_as_other_dataset_r1_in_group_rejected(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "pe", 10, seed=5)
    q1, q2 = write_pe(tmp_path, "qe", 10, seed=6)
    rows = [
        _base_row("dsA", read_type="paired_short", path1=str(p1), path2=str(p2)),
        _base_row("dsB", read_type="paired_short", path1=str(p1), path2=str(q2)),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    with pytest.raises(ManifestError, match="same underlying file"):
        load_manifest(m)


def test_cross_group_reuse_reported_not_rejected(tmp_path: Path):
    f = write_fastq(tmp_path / "shared.fastq", make_records(10, seed=0))
    g = write_fastq(tmp_path / "other.fastq", make_records(10, seed=1))
    h = write_fastq(tmp_path / "third.fastq", make_records(10, seed=2))
    rows = [
        _base_row("dsA", path1=str(f)),
        _base_row("dsB", path1=str(g)),
        _base_row("dsC", group="g2", path1=str(f)),
        _base_row("dsD", group="g2", path1=str(h)),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    result = load_manifest(m)
    assert len(result.cross_group_file_reuse) == 1
    path, ga, gb = result.cross_group_file_reuse[0]
    assert Path(path) == f.resolve()
    assert {ga, gb} == {"g1", "g2"}


def test_bam_manifest(tmp_path: Path):
    b1 = make_bam(tmp_path / "b1.bam", n=10, seed=1)
    b2 = make_bam(tmp_path / "b2.bam", n=10, seed=2)
    rows = [
        _base_row(
            "dsA", read_type="paired_short", input_format="bam", path1=str(b1), reference_id="hg38"
        ),
        _base_row(
            "dsB", read_type="paired_short", input_format="bam", path1=str(b2), reference_id="hg38"
        ),
    ]
    m = write_tsv_manifest(tmp_path / "m.tsv", rows)
    result = load_manifest(m)
    assert result.specs[0].reference_id == "hg38"


def test_parquet_manifest_wrong_types_rejected(tmp_path: Path):
    files = _two_se_files(tmp_path)
    table = pa.table(
        {
            "dataset_id": pa.array(["dsA", "dsB"]),
            "pool_group": pa.array(["g1", "g1"]),
            "sample_id": pa.array([123, 456]),  # ints, not strings
            "library_id": pa.array(["l", "l"]),
            "assay": pa.array(["WGS", "WGS"]),
            "platform": pa.array(["I", "I"]),
            "read_type": pa.array(["merged_short", "merged_short"]),
            "input_format": pa.array(["fastq", "fastq"]),
            "path1": pa.array([str(files[0]), str(files[1])]),
            "path2": pa.array([None, None]),
            "processing_id": pa.array(["p", "p"]),
            "reference_id": pa.array([None, None]),
            "fastp_json": pa.array([None, None]),
        }
    )
    m = tmp_path / "m.parquet"
    pq.write_table(table, m)
    with pytest.raises(ManifestError, match="must be a string"):
        load_manifest(m)


def test_manifest_empty_rows_rejected(tmp_path: Path):
    m = write_tsv_manifest(tmp_path / "m.tsv", [])
    with pytest.raises(ManifestError, match="no rows"):
        load_manifest(m)


def test_unsupported_extension_rejected(tmp_path: Path):
    p = tmp_path / "m.xlsx"
    p.write_text("x")
    with pytest.raises(ManifestError, match="unsupported manifest extension"):
        load_manifest(p)
