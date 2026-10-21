"""Strict input manifest loading (Parquet primary, TSV alternative).

Schema (v1). Required, non-nullable: ``dataset_id``, ``pool_group``, ``sample_id``,
``library_id``, ``assay``, ``platform``, ``read_type``, ``input_format``, ``path1``.
Nullable: ``path2`` (required exactly for ``fastq`` + ``paired_short``, forbidden
otherwise), ``processing_id``, ``reference_id``, ``fastp_json``. Unknown columns are
rejected. Relative input paths resolve against the manifest's parent directory.

Validation failures are ManifestError (exit 2): duplicate dataset ids, groups with
fewer than two datasets, heterogeneous format/profile within a group, wrong path
arity, missing/unreadable files, and reuse of the same underlying file within one
group. Reuse across different groups is returned separately and becomes a warning
finding, not an error.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seqconsistency.errors import ManifestError
from seqconsistency.models import DatasetSpec, InputFormat, ReadType

MANIFEST_SCHEMA_VERSION = "1"

REQUIRED_COLUMNS = (
    "dataset_id",
    "pool_group",
    "sample_id",
    "library_id",
    "assay",
    "platform",
    "read_type",
    "input_format",
    "path1",
)
NULLABLE_COLUMNS = ("path2", "processing_id", "reference_id", "fastp_json")
ALL_COLUMNS = REQUIRED_COLUMNS + NULLABLE_COLUMNS


@dataclass
class ManifestLoadResult:
    specs: list[DatasetSpec]
    manifest_path: Path
    sha256: str
    row_count: int
    #: (resolved_path, group_a, group_b) — same file reused across different groups.
    cross_group_file_reuse: list[tuple[str, str, str]]


def _read_manifest_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except Exception as exc:  # pyarrow raises assorted error types
        raise ManifestError(f"cannot read Parquet manifest: {exc}") from exc
    if table.num_rows == 0:
        raise ManifestError("manifest contains no rows")
    columns = set(table.column_names)
    missing = [c for c in ALL_COLUMNS if c not in columns]
    if missing:
        raise ManifestError(f"manifest is missing required column(s): {missing}")
    unknown = sorted(columns - set(ALL_COLUMNS))
    if unknown:
        raise ManifestError(
            f"manifest has unknown column(s) {unknown}; v1 accepts exactly {list(ALL_COLUMNS)}"
        )
    rows: list[dict[str, Any]] = []
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    for i in range(table.num_rows):
        rows.append({name: cols[name][i] for name in table.column_names})
    return rows


def _read_manifest_tsv(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None:
                raise ManifestError("TSV manifest is empty")
            fieldnames = [f.strip() for f in reader.fieldnames]
            missing = [c for c in ALL_COLUMNS if c not in fieldnames]
            if missing:
                raise ManifestError(f"manifest is missing required column(s): {missing}")
            unknown = sorted(set(fieldnames) - set(ALL_COLUMNS))
            if unknown:
                raise ManifestError(
                    f"manifest has unknown column(s) {unknown}; v1 accepts exactly "
                    f"{list(ALL_COLUMNS)}"
                )
            rows = []
            for line_no, row in enumerate(reader, start=2):
                clean: dict[str, Any] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    v = (value or "").strip()
                    clean[key.strip()] = v if v != "" else None
                clean["_line"] = line_no
                rows.append(clean)
    except OSError as exc:
        raise ManifestError(f"cannot read TSV manifest: {exc}") from exc
    if not rows:
        raise ManifestError("manifest contains no rows")
    return rows


def _string_field(
    row: dict[str, Any], column: str, where: str, *, required: bool = True
) -> str | None:
    value = row.get(column)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        if required:
            raise ManifestError(f"{where}: column {column!r} is required and empty")
        return None
    if not isinstance(value, str):
        raise ManifestError(
            f"{where}: column {column!r} must be a string, got {type(value).__name__}"
        )
    return value.strip()


def _same_file(a: Path, b: Path) -> bool:
    """True when both paths denote the same underlying file (alias-safe)."""
    try:
        sa, sb = os.stat(a), os.stat(b)
        return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)
    except OSError:
        return a.resolve() == b.resolve()


def load_manifest(path: Path) -> ManifestLoadResult:
    """Read and fully validate the manifest; returns DatasetSpec list."""
    if not path.is_file():
        raise ManifestError(f"manifest file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".tab", ".txt"):
        rows = _read_manifest_tsv(path)
    elif suffix in (".parquet", ".pq"):
        rows = _read_manifest_parquet(path)
    else:
        raise ManifestError(f"unsupported manifest extension {suffix!r} (use .parquet or .tsv)")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    base = path.resolve().parent

    specs: list[DatasetSpec] = []
    seen_ids: dict[str, int] = {}
    for i, row in enumerate(rows, start=1):
        where = f"manifest row {row.get('_line', i)}"
        dataset_id = _string_field(row, "dataset_id", where)
        assert dataset_id is not None
        if dataset_id in seen_ids:
            raise ManifestError(
                f"{where}: duplicate dataset_id {dataset_id!r} "
                f"(first seen in row {seen_ids[dataset_id]})"
            )
        seen_ids[dataset_id] = i
        pool_group = _string_field(row, "pool_group", where)
        sample_id = _string_field(row, "sample_id", where)
        library_id = _string_field(row, "library_id", where)
        assay = _string_field(row, "assay", where)
        platform = _string_field(row, "platform", where)
        read_type_raw = _string_field(row, "read_type", where)
        input_format_raw = _string_field(row, "input_format", where)
        try:
            read_type = ReadType(read_type_raw)
        except ValueError:
            raise ManifestError(
                f"{where}: read_type {read_type_raw!r} not in {[t.value for t in ReadType]}"
            ) from None
        try:
            input_format = InputFormat(input_format_raw)
        except ValueError:
            raise ManifestError(
                f"{where}: input_format {input_format_raw!r} not in "
                f"{[f.value for f in InputFormat]}"
            ) from None

        path1_raw = _string_field(row, "path1", where)
        path2_raw = _string_field(row, "path2", where, required=False)
        paired_fastq = input_format is InputFormat.FASTQ and read_type is ReadType.PAIRED_SHORT
        if paired_fastq and path2_raw is None:
            raise ManifestError(f"{where}: path2 is required for fastq + paired_short")
        if not paired_fastq and path2_raw is not None:
            raise ManifestError(
                f"{where}: path2 must be empty for this layout "
                f"(input_format={input_format_raw}, read_type={read_type_raw}); "
                "path2 exists only for PE FASTQ"
            )

        def resolve_input(raw: str | None, column: str, *, where: str) -> Path | None:
            if raw is None:
                return None
            p = Path(raw)
            if not p.is_absolute():
                p = base / p
            p = p.resolve()
            if not p.is_file():
                raise ManifestError(f"{where}: {column} file not found: {p}")
            if not os.access(p, os.R_OK):
                raise ManifestError(f"{where}: {column} file not readable: {p}")
            return p

        path1 = resolve_input(path1_raw, "path1", where=where)
        path2 = resolve_input(path2_raw, "path2", where=where)
        assert path1 is not None

        fastp_raw = _string_field(row, "fastp_json", where, required=False)
        fastp_path = resolve_input(fastp_raw, "fastp_json", where=where)

        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                pool_group=pool_group,
                sample_id=sample_id,
                library_id=library_id,
                assay=assay,
                platform=platform,
                read_type=read_type,
                input_format=input_format,
                path1=path1,
                path2=path2,
                processing_id=_string_field(row, "processing_id", where, required=False),
                reference_id=_string_field(row, "reference_id", where, required=False),
                fastp_json=fastp_path,
            )
        )

    # Group-level validation.
    groups: dict[str, list[DatasetSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.pool_group, []).append(spec)
    for group, members in sorted(groups.items()):
        if len(members) < 2:
            raise ManifestError(
                f"pooling group {group!r} has {len(members)} dataset(s); at least "
                "two are required for a pooling assessment"
            )
        formats = {m.input_format for m in members}
        if len(formats) > 1:
            raise ManifestError(
                f"pooling group {group!r} mixes input formats {sorted(f.value for f in formats)}; "
                "v1 pooling groups must be format-homogeneous (this is a v1 "
                "restriction, not a claim that mixed formats are impossible)"
            )
        profiles = {m.read_type for m in members}
        if len(profiles) > 1:
            raise ManifestError(
                f"pooling group {group!r} mixes comparison profiles "
                f"{sorted(p.value for p in profiles)}; a v1 group must use one "
                "profile (mixed-profile comparison is unsupported in v1)"
            )
        # Duplicate underlying file inside one group (alias-aware).
        flat: list[tuple[Path, DatasetSpec, str]] = []
        for m in members:
            flat.append((m.path1, m, "path1"))
            if m.path2 is not None:
                flat.append((m.path2, m, "path2"))
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                pa, ma, ca = flat[i]
                pb, mb, cb = flat[j]
                if ma is mb and ca == cb:
                    continue
                if _same_file(pa, pb):
                    raise ManifestError(
                        f"pooling group {group!r} reuses the same underlying file "
                        f"for {ma.dataset_id}.{ca} and {mb.dataset_id}.{cb} "
                        f"({pa}); pooling a file with itself is not a comparison"
                    )

    # Cross-group reuse: returned for a warning finding, not an error.
    cross: dict[str, list[str]] = {}
    for spec in specs:
        for p in (spec.path1, spec.path2):
            if p is not None:
                cross.setdefault(str(p), []).append(spec.pool_group)
    reuse = [
        (path, sorted(set(groups_))[0], sorted(set(groups_))[1])
        for path, groups_ in cross.items()
        if len(set(groups_)) > 1
    ]

    return ManifestLoadResult(
        specs=specs,
        manifest_path=path.resolve(),
        sha256=sha256,
        row_count=len(specs),
        cross_group_file_reuse=reuse,
    )
