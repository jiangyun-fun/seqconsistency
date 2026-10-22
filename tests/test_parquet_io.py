"""StagedWriter behavior: cross-call batching, abort logging, atomic publish."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seqconsistency import parquet_io as pio


def _default_for(t: pa.DataType):
    if pa.types.is_string(t):
        return "x"
    if pa.types.is_boolean(t):
        return True
    if pa.types.is_integer(t):
        return 1
    if pa.types.is_floating(t):
        return 1.0
    if pa.types.is_list(t):
        return ["a"]
    raise ValueError(str(t))


def _finding_row() -> dict:
    return {f.name: _default_for(f.type) for f in pio.FINDINGS_SCHEMA}


def test_write_rows_batches_across_calls(tmp_path: Path):
    # Review round 3, point 2: the pipeline writes per finding and per pair;
    # per-call flushing created one row group per call (2000 groups / 3.68 MB
    # vs 1 group / 3.6 KB measured). The buffer must span calls.
    n = 2000
    p = tmp_path / "per_call.parquet"
    w = pio.StagedWriter(p, pio.FINDINGS_SCHEMA)
    for _ in range(n):
        w.write_rows([_finding_row()])
    assert w.close_publish() == n
    meta = pq.ParquetFile(p).metadata
    assert meta.num_row_groups == 1
    assert meta.num_rows == n
    assert p.stat().st_size < 100_000  # batched; per-call flushing was ~3.7 MB


def test_write_rows_flushes_at_batch_size(tmp_path: Path):
    p = tmp_path / "big.parquet"
    w = pio.StagedWriter(p, pio.METRICS_SCHEMA)
    row = {f.name: _default_for(f.type) for f in pio.METRICS_SCHEMA}
    for _ in range(pio.WRITE_BATCH_ROWS + 5):
        w.write_rows([row])
    w.close_publish()
    meta = pq.ParquetFile(p).metadata
    assert meta.num_row_groups == 2  # one full batch flushed + remainder on close
    assert meta.num_rows == pio.WRITE_BATCH_ROWS + 5


def test_abort_logs_cleanup_failures_and_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    # Review round 2 follow-up: cleanup failures during abort must be logged,
    # never raised — the error that triggered the abort stays the caller's error.
    p = tmp_path / "doomed.parquet"
    w = pio.StagedWriter(p, pio.FINDINGS_SCHEMA)
    w.write_rows([_finding_row()])

    class Boom(Exception):
        pass

    def exploding_close() -> None:
        raise Boom("writer exploded")

    original_close = w._writer.close
    w._writer.close = exploding_close  # type: ignore[method-assign]
    try:
        with caplog.at_level(logging.WARNING, logger="seqconsistency.parquet_io"):
            w.abort()  # must not raise Boom
        assert any("failed to close staged writer" in r.message for r in caplog.records)
        assert not p.exists()  # staged temp still removed
    finally:
        w._writer.close = original_close  # type: ignore[method-assign] — keep __del__ quiet
