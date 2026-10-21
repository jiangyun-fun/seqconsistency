"""FASTQ streaming, structural validation, and vectorized metric extraction.

Design (README § FASTQ processing):
- Strict four-line records (header/seq/plus/qual). Wrapped (multi-line) records are
  rejected with a clear error — a documented v1 restriction.
- One pass per file combines structural validation, full-file counters, full-file
  histograms, positional accumulation, and reservoir sampling. There is no mode that
  skips full-file validation.
- Per-base work is vectorized with numpy over concatenated record buffers; there are
  no per-base Python loops. Memory is bounded by the batch byte budget, never by the
  file size or read count.
- Quality encoding: Phred+33 only. Bytes outside ASCII [33, 126] abort the run. As a
  documented heuristic (ASCII ranges alone cannot always distinguish encodings), a
  file whose full-file minimum quality byte is >= 64 ('@') and maximum >= 75 ('K') is
  rejected as likely Phred+64: under Phred+33 that would mean every base >= Q31 with
  some > Q41, which Illumina instruments do not emit.
- Alphabet: A/C/G/T/N upper or lower case. Other IUPAC codes abort the run in v1.
"""

from __future__ import annotations

import gzip
import json
import logging
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from seqconsistency import distributions as dist
from seqconsistency.errors import InputDataError
from seqconsistency.models import (
    SHORT_PROFILES,
    FastpEvidence,
    FastqFileStats,
    PairingStats,
    PositionalProfile,
    ReadType,
    SamplingConfig,
)
from seqconsistency.sampling import FeatureReservoir, dataset_seed

_ZLIB_ERROR = zlib.error

logger = logging.getLogger(__name__)

#: Batch is flushed at whichever limit comes first: record count or sequence bytes.
#: 16 MiB of bases keeps every transient numpy buffer far below 2**31 elements.
BATCH_RECORDS = 16384
BATCH_BYTES = 1 << 24
#: Read chunk for line splitting.
READ_CHUNK_BYTES = 1 << 23  # 8 MiB

#: Byte -> category LUT: A/a->0, C/c->1, G/g->2, T/t->3, N/n->4, everything else -> 255.
CAT_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _letters in enumerate((b"Aa", b"Cc", b"Gg", b"Tt", b"Nn")):
    CAT_LUT[_letters[0]] = _i
    CAT_LUT[_letters[1]] = _i

#: Byte -> Phred+33 error probability LUT (float32), for error-based mean quality.
ERR_LUT = np.zeros(256, dtype=np.float32)
for _q in range(33, 127):
    ERR_LUT[_q] = np.float32(10.0 ** (-(_q - 33) / 10.0))

MIN_QUAL_BYTE = 33
MAX_QUAL_BYTE = 126
#: Heuristic Phred+64 rejection thresholds (see module docstring).
LIKELY_P64_MIN_BYTE = 64
LIKELY_P64_MAX_BYTE = 75

POSITIONAL_MAX_CYCLES: dict[ReadType, int] = {
    ReadType.PAIRED_SHORT: 512,
    ReadType.MERGED_SHORT: 1024,
}


@dataclass
class RawBatch:
    """One structurally validated batch of four-line FASTQ records."""

    headers: list[bytes]
    seqs: list[bytes]
    quals: list[bytes]
    first_record_number: int  # 1-based global number of headers[0]


def _open_fastq(path: Path) -> IO[bytes]:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _iter_lines(fh: IO[bytes], path: Path) -> Iterator[bytes]:
    """Yield newline-split lines; a trailing ``\\r`` is stripped (CRLF tolerance)."""
    carry = b""
    while True:
        try:
            chunk = fh.read(READ_CHUNK_BYTES)
        except (OSError, EOFError, _ZLIB_ERROR) as exc:
            raise InputDataError(
                f"failed reading FASTQ stream (truncated or corrupt gzip: {exc})",
                source=str(path),
            ) from exc
        if not chunk:
            break
        carry += chunk
        lines = carry.split(b"\n")
        carry = lines.pop()
        yield from lines
    if carry:
        yield carry


def iter_fastq_batches(
    path: Path,
    batch_records: int = BATCH_RECORDS,
    batch_bytes: int = BATCH_BYTES,
) -> Iterator[RawBatch]:
    """Stream strict four-line FASTQ batches with full structural validation.

    Raises InputDataError on the first structural defect, naming the record number
    and read header. Records are never silently skipped. Batches are flushed on a
    record-count or sequence-byte budget, whichever comes first.
    """
    record_no = 0
    headers: list[bytes] = []
    seqs: list[bytes] = []
    quals: list[bytes] = []
    pending: list[bytes] = []
    seq_bytes = 0
    with _open_fastq(path) as fh:
        for raw in _iter_lines(fh, path):
            line = raw[:-1] if raw.endswith(b"\r") else raw
            pending.append(line)
            if len(pending) == 4:
                record_no += 1
                header, seq, plus, qual = pending
                if not header.startswith(b"@"):
                    raise InputDataError(
                        f"record does not start with '@' (got {header[:20]!r})",
                        source=str(path),
                        location=f"record {record_no}",
                    )
                if not plus.startswith(b"+"):
                    raise InputDataError(
                        f"separator line does not start with '+' (got {plus[:20]!r})",
                        source=str(path),
                        location=f"record {record_no}",
                    )
                if len(seq) != len(qual):
                    raise InputDataError(
                        f"sequence length {len(seq)} != quality length {len(qual)} "
                        f"for read {header[1:21]!r}",
                        source=str(path),
                        location=f"record {record_no}",
                    )
                headers.append(header)
                seqs.append(seq)
                quals.append(qual)
                seq_bytes += len(seq)
                pending = []
                if len(headers) >= batch_records or seq_bytes >= batch_bytes:
                    yield RawBatch(headers, seqs, quals, record_no - len(headers) + 1)
                    headers, seqs, quals, seq_bytes = [], [], [], 0
        if pending:
            raise InputDataError(
                f"truncated FASTQ: {len(pending)} trailing line(s) after the last "
                "complete record (file ends mid-record)",
                source=str(path),
                location=f"record {record_no + 1}",
            )
    if headers:
        yield RawBatch(headers, seqs, quals, record_no - len(headers) + 1)


def normalize_read_name(header: bytes) -> bytes:
    """Read name up to the first whitespace, with a trailing /1 or /2 removed.

    This covers Illumina/CASAVA ``name 1:N:0:ACGT`` comments and legacy ``/1``/``/2``
    suffixes. Arbitrary substrings are never stripped to force synchronization.
    """
    name = header[1:].split(None, 1)[0] if len(header) > 1 else b""
    if name.endswith(b"/1") or name.endswith(b"/2"):
        name = name[:-2]
    return name


def mate_annotation(header: bytes) -> bytes | None:
    """Explicit mate annotation: the /1 or /2 suffix, else None."""
    name = header[1:].split(None, 1)[0] if len(header) > 1 else b""
    if name.endswith(b"/1"):
        return b"/1"
    if name.endswith(b"/2"):
        return b"/2"
    return None


class StrandStats:
    """Accumulates one strand's (R1/R2/SE) full-file counters, histograms, and
    positional profile across batches in bounded memory.

    Per-read scalar values are histogrammed per batch and folded into fixed-size
    accumulator arrays; values themselves are never retained beyond the batch.
    """

    def __init__(self, read_type: ReadType, stratum: str, source: str) -> None:
        self.read_type = read_type
        self.stratum = stratum
        self.source = source
        self.stats = FastqFileStats()
        self.is_short = read_type in SHORT_PROFILES
        self.max_cycles = POSITIONAL_MAX_CYCLES.get(read_type) if self.is_short else None
        n_gc = len(dist.GC_EDGES) - 1
        n_n = len(dist.N_EDGES) - 1
        n_len = len(self._length_edges()) - 1
        self._gc = (np.zeros(n_gc, dtype=np.int64), 0, 0, 0)
        self._nfr = (np.zeros(n_n, dtype=np.int64), 0, 0, 0)
        self._len = (np.zeros(n_len, dtype=np.int64), 0, 0, 0)
        if self.max_cycles is not None:
            self._cov = np.zeros(self.max_cycles, dtype=np.int64)
            self._qsum = np.zeros(self.max_cycles, dtype=np.float64)
            self._bases = np.zeros((self.max_cycles, 5), dtype=np.int64)
            self._positional_overflow = 0

    def _length_edges(self) -> list[float]:
        return dist.SHORT_LENGTH_EDGES if self.is_short else dist.LONG_LENGTH_EDGES

    def add_batch(self, batch: RawBatch) -> np.ndarray:
        """Validate and accumulate one batch.

        Returns the per-read feature matrix (n, 5) in FEATURE_NAMES order:
        [length, gc_fraction, n_fraction, mean_quality_arithmetic,
        mean_quality_error_based]. Rows with undefined values carry NaN.
        """
        lengths = np.fromiter((len(s) for s in batch.seqs), dtype=np.int64, count=len(batch.seqs))
        seq_arr = np.frombuffer(b"".join(batch.seqs), dtype=np.uint8)
        qual_arr = np.frombuffer(b"".join(batch.quals), dtype=np.uint8)
        st = self.stats

        if qual_arr.size and (
            int(qual_arr.min()) < MIN_QUAL_BYTE or int(qual_arr.max()) > MAX_QUAL_BYTE
        ):
            self._raise_bad_quality_byte(qual_arr, batch, lengths)

        ends = np.cumsum(lengths)
        starts = ends - lengths
        total_bases = int(ends[-1]) if lengths.size else 0

        cat = CAT_LUT[seq_arr]
        if cat.size and int(cat.max()) == 255:
            self._raise_bad_alphabet(seq_arr, batch, ends)

        per_read_counts = np.empty((lengths.size, 5), dtype=np.int64)
        for k in range(5):
            c = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(cat == k, dtype=np.int32)))
            per_read_counts[:, k] = c[ends] - c[starts]

        acgt = per_read_counts[:, :4].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            gc_frac = (per_read_counts[:, 1] + per_read_counts[:, 2]) / acgt
            n_frac = per_read_counts[:, 4] / lengths

        q = (qual_arr.astype(np.int16) - 33).astype(np.float64)
        cq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(q)))
        read_qsum = cq[ends] - cq[starts]
        q20 = int(np.count_nonzero(q >= 20))
        q30 = int(np.count_nonzero(q >= 30))
        err = ERR_LUT[qual_arr].astype(np.float64)
        ce = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(err)))
        read_errsum = ce[ends] - ce[starts]
        with np.errstate(invalid="ignore", divide="ignore"):
            meanq_arith = read_qsum / lengths
            mean_err = read_errsum / lengths
        meanq_prob = np.full(lengths.shape[0], np.nan)
        has_len = lengths > 0
        pos_err = has_len & (mean_err > 0)
        meanq_prob[pos_err] = -10.0 * np.log10(mean_err[pos_err])
        # length>0 but zero summed error probability (impossible under Phred+33 with
        # Q<=93, guarded anyway): cap at the maximum representable Phred.
        meanq_prob[has_len & ~pos_err] = 93.0

        if self.max_cycles is not None and total_bases:
            assert self._cov is not None
            self._accumulate_positional(cat, q, starts, lengths, self.max_cycles)

        st.total_reads += int(lengths.size)
        st.total_bases += total_bases
        batch_base_counts = per_read_counts.sum(axis=0)
        for i in range(5):
            st.base_counts[i] += int(batch_base_counts[i])
        st.total_n_bases += int(batch_base_counts[4])
        st.q20_bases += q20
        st.q30_bases += q30
        st.sum_lengths += int(lengths.sum())
        st.sum_sq_lengths += int((lengths * lengths).sum())
        if qual_arr.size:
            st.min_qual_byte = min(st.min_qual_byte, int(qual_arr.min()))
            st.max_qual_byte = max(st.max_qual_byte, int(qual_arr.max()))
        hist = np.bincount(qual_arr, minlength=127)[33:127]
        st.base_qual_counts = (np.asarray(st.base_qual_counts) + hist).tolist()
        valid_q = np.isfinite(meanq_arith)
        st.reads_with_mean_quality += int(valid_q.sum())
        st.sum_read_mean_quality += float(meanq_arith[valid_q].sum())
        st.sum_read_quality_error_based += float(meanq_prob[valid_q].sum())
        valid_gc = np.isfinite(gc_frac)
        st.reads_with_gc += int(valid_gc.sum())
        st.sum_gc_fraction += float(gc_frac[valid_gc].sum())
        self._fold_gc(gc_frac)
        self._fold_n(n_frac)
        self._fold_len(lengths)

        return np.column_stack(
            (
                lengths.astype(np.float64),
                gc_frac,
                n_frac,
                meanq_arith,
                meanq_prob,
            )
        )

    def _fold_gc(self, values: np.ndarray) -> None:
        counts, overflow, underflow = dist.bin_values(values, dist.GC_EDGES)
        arr, ov, un, n = self._gc
        arr += counts
        self._gc = (arr, ov + overflow, un + underflow, n + values.size)

    def _fold_n(self, values: np.ndarray) -> None:
        counts, overflow, underflow = dist.bin_values(values, dist.N_EDGES)
        arr, ov, un, n = self._nfr
        arr += counts
        self._nfr = (arr, ov + overflow, un + underflow, n + values.size)

    def _fold_len(self, values: np.ndarray) -> None:
        counts, overflow, underflow = dist.bin_values(values, self._length_edges())
        arr, ov, un, n = self._len
        arr += counts
        self._len = (arr, ov + overflow, un + underflow, n + values.size)

    def _accumulate_positional(
        self,
        cat: np.ndarray,
        q: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        max_cycles: int,
    ) -> None:
        starts_per_base = np.repeat(starts.astype(np.int32), lengths.astype(np.int32))
        pos = np.arange(cat.size, dtype=np.int32) - starts_per_base
        in_cap = pos < max_cycles
        pos_in = pos[in_cap]
        self._cov += np.bincount(pos_in, minlength=max_cycles).astype(np.int64)
        self._qsum += np.bincount(pos_in, weights=q[in_cap], minlength=max_cycles)
        self._bases += (
            np.bincount(pos_in * 5 + cat[in_cap], minlength=max_cycles * 5)
            .reshape(max_cycles, 5)
            .astype(np.int64)
        )
        self._positional_overflow += int((~in_cap).sum())

    def _raise_bad_quality_byte(
        self, qual_arr: np.ndarray, batch: RawBatch, lengths: np.ndarray
    ) -> None:
        bad = int(np.where((qual_arr < MIN_QUAL_BYTE) | (qual_arr > MAX_QUAL_BYTE))[0][0])
        record_idx = int(np.searchsorted(np.cumsum(lengths), bad, side="right"))
        raise InputDataError(
            "quality byte outside ASCII [33, 126] is not valid Phred+33 "
            f"(byte {int(qual_arr[bad])} in read {batch.headers[record_idx][1:41]!r})",
            source=self.source,
            location=f"record {batch.first_record_number + record_idx}",
        )

    def _raise_bad_alphabet(self, seq_arr: np.ndarray, batch: RawBatch, ends: np.ndarray) -> None:
        bad = int(np.where(CAT_LUT[seq_arr] == 255)[0][0])
        record_idx = int(np.searchsorted(ends, bad, side="right"))
        raise InputDataError(
            f"unsupported nucleotide character {chr(seq_arr[bad])!r} in read "
            f"{batch.headers[record_idx][1:41]!r} (v1 supports A/C/G/T/N, upper or "
            "lower case)",
            source=self.source,
            location=f"record {batch.first_record_number + record_idx}",
        )

    def finalize(self) -> FastqFileStats:
        st = self.stats
        if st.total_reads == 0:
            raise InputDataError(
                f"FASTQ contains no reads ({self.stratum} strand of {self.source})",
                source=self.source,
            )
        if st.min_qual_byte >= LIKELY_P64_MIN_BYTE and st.max_qual_byte >= LIKELY_P64_MAX_BYTE:
            raise InputDataError(
                "quality encoding rejected: every quality byte >= ASCII 64 with "
                f"maximum {st.max_qual_byte} is consistent with Phred+64 and not with "
                "Phred+33 instrument output; v1 supports Phred+33 only "
                "(ASCII ranges alone cannot always distinguish encodings — this "
                "heuristic is documented in the README)",
                source=self.source,
            )
        arr, ov, un, n = self._gc
        st.gc_hist = dist.Histogram("per_read_gc", dist.GC_EDGES, arr.tolist(), ov, un, n)
        arr, ov, un, n = self._nfr
        st.n_hist = dist.Histogram("per_read_n", dist.N_EDGES, arr.tolist(), ov, un, n)
        arr, ov, un, n = self._len
        edges = self._length_edges()
        st.length_hist = dist.Histogram("read_length", edges, arr.tolist(), ov, un, n)
        if self.max_cycles is not None and self._cov is not None:
            st.positional = PositionalProfile(
                name="per_cycle",
                max_cycles=self.max_cycles,
                coverage=self._cov.tolist(),
                qual_sum=[int(round(v)) for v in self._qsum.tolist()],
                base_counts=self._bases.tolist(),
                overflow_bases=int(self._positional_overflow),
            )
        return st


def scan_fastq_single(
    dataset_id: str,
    path: Path,
    read_type: ReadType,
    sampling: SamplingConfig,
    sample_size: int,
) -> tuple[FastqFileStats, dict[str, list[float]], int, int]:
    """Scan one single-end FASTQ: validate fully, compute stats, sample reads.

    Returns (stats, per-read sample features keyed by FEATURE_NAMES, eligible
    observations, sampled count). Sampling unit: read.
    """
    strand = StrandStats(read_type, "SE", str(path))
    rng = np.random.default_rng(dataset_seed(sampling.master_seed, dataset_id))
    reservoir = FeatureReservoir(sample_size)
    for batch in iter_fastq_batches(path):
        features = strand.add_batch(batch)
        reservoir.add_block(features, keys=rng.random(features.shape[0]))
    stats = strand.finalize()
    return stats, reservoir.sample(), stats.total_reads, reservoir.sampled_count


def scan_fastq_paired(
    dataset_id: str,
    path1: Path,
    path2: Path,
    read_type: ReadType,
    sampling: SamplingConfig,
    sample_size: int,
) -> tuple[FastqFileStats, FastqFileStats, PairingStats, dict, dict, int, int]:
    """Scan one PE FASTQ pair with per-pair identifier synchronization checks.

    Sampling unit: mate pair — R1 and R2 features of one pair share one reservoir
    key. The first identifier mismatch or contradictory mate annotation aborts the
    run (broken synchronization is an execution error, not a finding).

    Returns (r1_stats, r2_stats, pairing_stats, sample_r1, sample_r2, pairs, sampled).
    """
    s1 = StrandStats(read_type, "R1", str(path1))
    s2 = StrandStats(read_type, "R2", str(path2))
    pairing = PairingStats()
    rng = np.random.default_rng(dataset_seed(sampling.master_seed, dataset_id))
    res1 = FeatureReservoir(sample_size)
    res2 = FeatureReservoir(sample_size)

    gen1 = iter_fastq_batches(path1)
    gen2 = iter_fastq_batches(path2)
    pend1: RawBatch | None = None
    pend2: RawBatch | None = None
    i1 = i2 = 0  # consumed positions inside pending batches

    while True:
        if pend1 is not None and i1 >= len(pend1.headers):
            pend1, i1 = None, 0
        if pend2 is not None and i2 >= len(pend2.headers):
            pend2, i2 = None, 0
        if pend1 is None:
            pend1 = next(gen1, None)
            i1 = 0
        if pend2 is None:
            pend2 = next(gen2, None)
            i2 = 0
        if pend1 is None and pend2 is None:
            break
        if pend1 is None or pend2 is None:
            missing, have = (str(path2), str(path1)) if pend2 is None else (str(path1), str(path2))
            raise InputDataError(
                f"unequal mate counts: {missing} ended before {have} "
                f"({pairing.pairs_checked} complete pairs processed)",
                source=missing,
                location=f"pair {pairing.pairs_checked + 1}",
            )
        k = min(len(pend1.headers) - i1, len(pend2.headers) - i2)
        h1 = pend1.headers[i1 : i1 + k]
        h2 = pend2.headers[i2 : i2 + k]
        num1 = pend1.first_record_number + i1
        for idx in range(k):
            name1 = normalize_read_name(h1[idx])
            name2 = normalize_read_name(h2[idx])
            pair_no = num1 + idx
            if name1 != name2:
                raise InputDataError(
                    f"paired-read identifiers out of sync: R1 {name1!r} vs R2 {name2!r}",
                    source=f"{path1}|{path2}",
                    location=f"pair {pair_no}",
                )
            ann1 = mate_annotation(h1[idx])
            ann2 = mate_annotation(h2[idx])
            if ann1 == b"/2" or ann2 == b"/1":
                raise InputDataError(
                    f"contradictory mate annotations: R1 carries {ann1!r}, R2 carries "
                    f"{ann2!r} for read {name1!r}",
                    source=f"{path1}|{path2}",
                    location=f"pair {pair_no}",
                )
        pairing.pairs_checked += k
        keys = rng.random(k)
        b1 = RawBatch(h1, pend1.seqs[i1 : i1 + k], pend1.quals[i1 : i1 + k], num1)
        b2 = RawBatch(h2, pend2.seqs[i2 : i2 + k], pend2.quals[i2 : i2 + k], num1)
        res1.add_block(s1.add_batch(b1), keys=keys)
        res2.add_block(s2.add_batch(b2), keys=keys)
        i1 += k
        i2 += k

    st1 = s1.finalize()
    st2 = s2.finalize()
    return (
        st1,
        st2,
        pairing,
        res1.sample(),
        res2.sample(),
        st1.total_reads,
        res1.sampled_count,
    )


def parse_fastp_json(path: Path) -> FastpEvidence:
    """Extract the fastp fields seqconsistency understands.

    fastp report schemas vary across versions. We read the stable ``summary``
    subsection and report a merge rate only when the file carries an explicit
    merged-read count under a recognized key; a merged FASTQ alone never implies a
    merge rate. Absent fields stay ``None`` (unavailable), never zero.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputDataError(f"cannot read fastp JSON report: {exc}", source=str(path)) from exc
    if not isinstance(data, dict):
        raise InputDataError("fastp report root is not an object", source=str(path))

    def section_total(section: dict) -> int | None:
        v = section.get("total_reads")
        return int(v) if isinstance(v, (int, float)) else None

    def q30(section: dict) -> float | None:
        v = section.get("q30_rate")
        return float(v) if isinstance(v, (int, float)) else None

    summary = data.get("summary", {})
    before = summary.get("before_filtering", {}) if isinstance(summary, dict) else {}
    after = summary.get("after_filtering", {}) if isinstance(summary, dict) else {}
    before_total = section_total(before) if isinstance(before, dict) else None
    after_total = section_total(after) if isinstance(after, dict) else None

    merged: int | None = None
    merged_keys = ("merged_reads", "reads_merged", "merged_read_pairs")

    def find_merged(obj: object) -> None:
        nonlocal merged
        if merged is not None or not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if merged is not None:
                return
            if isinstance(value, dict):
                find_merged(value)
            elif isinstance(value, int) and key in merged_keys:
                merged = int(value)

    find_merged(data)

    retained: float | None = None
    if before_total and after_total is not None:
        retained = after_total / before_total
    return FastpEvidence(
        source=path,
        before_total_reads=before_total,
        after_total_reads=after_total,
        before_q30_rate=q30(before) if isinstance(before, dict) else None,
        after_q30_rate=q30(after) if isinstance(after, dict) else None,
        merged_reads=merged,
        retained_fraction=retained,
    )
