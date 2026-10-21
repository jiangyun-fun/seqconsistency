"""Typed Parquet output with explicit versioned Arrow schemas and staged writes.

Every persistent output is written to a temporary sibling file first and published
with ``os.replace`` only after the run's computations succeeded. Atomic replacement
is per-file: publication across multiple independent filesystem paths cannot always
be transactional, and this package does not claim otherwise (README § Outputs). On
failure, staged temporaries are removed and already-published outputs from this run
are removed best-effort.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from seqconsistency.errors import OutputError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
COMPRESSION = "zstd"
WRITE_BATCH_ROWS = 65536


def _md(description: str) -> dict[bytes, bytes]:
    return {
        b"schema": b"seqconsistency",
        b"schema_version": SCHEMA_VERSION.encode(),
        b"description": description.encode(),
    }


METRICS_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("pool_group", pa.string()),
        ("component", pa.string()),
        ("stratum", pa.string()),
        ("metric", pa.string()),
        ("statistic", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("observation_count", pa.int64()),
        ("denominator", pa.int64()),
        ("sampling_mode", pa.string()),
        ("availability", pa.string()),
    ],
    metadata=_md("metrics: one row per dataset/stratum/metric/statistic"),
)

DISTRIBUTIONS_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("pool_group", pa.string()),
        ("component", pa.string()),
        ("stratum", pa.string()),
        ("metric", pa.string()),
        ("bin_index", pa.int32()),
        ("bin_start", pa.float64()),
        ("bin_end", pa.float64()),
        ("category", pa.string()),
        ("count", pa.int64()),
        ("frequency", pa.float64()),
        ("value", pa.float64()),
        ("observation_count", pa.int64()),
        ("sampling_mode", pa.string()),
    ],
    metadata=_md(
        "distributions: normalized long-format histograms and per-cycle profiles; "
        "bin i covers [bin_start, bin_end), final bin closed right; per-cycle rows "
        "use absolute cycles with 'value' carrying the per-cycle statistic"
    ),
)

COMPARISONS_SCHEMA = pa.schema(
    [
        ("pool_group", pa.string()),
        ("dataset_a", pa.string()),
        ("dataset_b", pa.string()),
        ("component", pa.string()),
        ("stratum", pa.string()),
        ("metric", pa.string()),
        ("method", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("threshold", pa.float64()),
        ("threshold_direction", pa.string()),
        ("threshold_calibrated", pa.bool_()),
        ("status", pa.string()),
        ("observations_a", pa.int64()),
        ("observations_b", pa.int64()),
        ("reason_code", pa.string()),
        ("method_version", pa.string()),
    ],
    metadata=_md(
        "comparisons: one row per dataset pair (lexicographic order), stratum, and "
        "metric; dataset_a < dataset_b always; threshold_calibrated is null when no "
        "threshold applied — a PASS on an uncalibrated threshold never approves"
    ),
)

FINDINGS_SCHEMA = pa.schema(
    [
        ("finding_id", pa.string()),
        ("pool_group", pa.string()),
        ("dataset_id", pa.string()),
        ("affected_datasets", pa.list_(pa.string())),
        ("component", pa.string()),
        ("severity", pa.string()),
        ("reason_code", pa.string()),
        ("message", pa.string()),
        ("evidence_metric", pa.string()),
        ("evidence_value", pa.float64()),
        ("recommended_action", pa.string()),
    ],
    metadata=_md("findings: one row per finding with stable reason codes"),
)

DECISIONS_SCHEMA = pa.schema(
    [
        ("pool_group", pa.string()),
        ("structural_status", pa.string()),
        ("technical_status", pa.string()),
        ("metadata_status", pa.string()),
        ("identity_status", pa.string()),
        ("overall_decision", pa.string()),
        ("identity_evidence_level", pa.string()),
        ("assumptions", pa.list_(pa.string())),
        ("reason_codes", pa.list_(pa.string())),
        ("policy_id", pa.string()),
        ("policy_version", pa.string()),
    ],
    metadata=_md("decisions: one row per pooling group"),
)


class StagedWriter:
    """Writes one Parquet file incrementally from row-dict iterables.

    Rows are converted and flushed in bounded batches; memory scales with the batch
    size, not the total row count. The staged temporary is published (atomically
    renamed) only on successful close.
    """

    def __init__(self, final_path: Path, schema: pa.Schema) -> None:
        self.final_path = final_path
        self.schema = schema
        self.staged_path = final_path.with_name(f".{final_path.name}.tmp-{os.getpid()}")
        self._writer = pq.ParquetWriter(str(self.staged_path), schema, compression=COMPRESSION)
        self._rows_written = 0

    def write_rows(self, rows: Iterable[dict]) -> None:
        batch: list[dict] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= WRITE_BATCH_ROWS:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict]) -> None:
        arrays = []
        for field in self.schema:
            arrays.append(pa.array([row.get(field.name) for row in batch], type=field.type))
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        self._writer.write_table(table)
        self._rows_written += len(batch)

    def close_publish(self) -> int:
        self._writer.close()
        os.replace(self.staged_path, self.final_path)
        return self._rows_written

    def abort(self) -> None:
        """Close and delete the staged file without publishing.

        Idempotent and never raises: after ``close_publish`` the staged path no
        longer exists, and closing an already-closed writer is harmless to
        ignore. Cleanup failures are logged (never raised) so the error that
        triggered the abort stays the error the caller sees.
        """
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001 — cleanup must not mask the real error
            logger.warning(
                "failed to close staged writer for %s during abort",
                self.final_path,
                exc_info=True,
            )
        try:
            self.staged_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "failed to delete staged file %s during abort",
                self.staged_path,
                exc_info=True,
            )

    @property
    def rows_written(self) -> int:
        return self._rows_written


def write_table(path: Path, schema: pa.Schema, rows: list[dict] | Iterator[dict]) -> int:
    """Convenience staged write for tables that fit in memory as a list."""
    staged = StagedWriter(path, schema)
    try:
        staged.write_rows(rows)
        return staged.close_publish()
    except Exception:
        staged.abort()
        raise


def read_back(path: Path) -> pa.Table:
    """Read a written table back (verification helper used by tests and smoke runs)."""
    try:
        return pq.read_table(path)
    except Exception as exc:
        raise OutputError(f"cannot read back Parquet output {path}: {exc}") from exc


@contextmanager
def output_transaction(paths: list[Path]) -> Iterator[list[Path]]:
    """Collect published outputs; on failure remove everything this run created.

    Best-effort rollback only — multi-path publication cannot be fully transactional
    (documented). Never touches files the run did not create.
    """
    published: list[Path] = []
    try:
        yield published
    except BaseException:
        for p in published:
            try:
                p.unlink(missing_ok=True)
                logger.warning("rolled back published output %s after failure", p)
            except OSError:
                logger.error("could not remove published output %s during rollback", p)
        raise


def chunked(iterable: Iterable[dict], size: int = WRITE_BATCH_ROWS) -> Iterator[list]:
    it = iter(iterable)
    while True:
        chunk = [row for _, row in zip(range(size), it, strict=False)]
        if not chunk:
            return
        yield chunk
