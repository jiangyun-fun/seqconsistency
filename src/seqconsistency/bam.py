"""BAM streaming, structural validation, and metric extraction via pysam/HTSlib.

Design (README § BAM processing):
- Sequential scan only; no index required and none created.
- Coordinate sort is verified two ways: the header must declare ``SO:coordinate`` and
  the observed (tid, pos) sequence must be non-decreasing, with unplaced records
  (tid == -1) allowed only at the end. A header declaration alone never proves order.
- Eligibility for read-level metrics: primary alignments (not secondary, not
  supplementary). Supplementary and secondary records are counted as categories but
  never as independent sequencing reads.
- Insert size: one observation per pair — primary, paired, properly-paired,
  first-in-pair, TLEN > 0. This counts each properly-paired template once and never
  builds a mate dictionary. Pairs with TLEN <= 0 or without the proper-pair flag are
  excluded (undercount reported through the eligible counter — a documented
  limitation).
- GC, base-quality, and per-read mean-quality distributions come from the sampled
  eligible primary records (``sampling_mode=sample``): pysam per-record sequence
  access dominates scan cost, so full-file sequence extraction is deliberately not
  performed. Category counters and length/MAPQ/insert-size histograms are full-scan.
- Missing base qualities or sequence count as unavailable (never zero) and are
  reported through dedicated full-scan counters.
- Duplicate-marked fractions depend on upstream processing; they are never
  interpreted as independently measured PCR duplication rates.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from seqconsistency import distributions as dist
from seqconsistency.errors import InputDataError
from seqconsistency.models import (
    SHORT_PROFILES,
    BamFlagCounts,
    BamHeaderInfo,
    BamScanResult,
    ReadType,
    SamplingConfig,
)
from seqconsistency.sampling import FeatureReservoir, dataset_seed

logger = logging.getLogger(__name__)

FLAG_PAIRED = 0x1
FLAG_PROPER = 0x2
FLAG_UNMAPPED = 0x4
FLAG_FIRST = 0x40
FLAG_SECONDARY = 0x100
FLAG_QCFAIL = 0x200
FLAG_DUP = 0x400
FLAG_SUPPL = 0x800

#: Flush buffered per-record values into histograms at this chunk size.
HIST_FLUSH_RECORDS = 65536


def _record_features(record) -> tuple[float, float, float, float, float]:
    """Per-read features for one eligible BAM record.

    Returns (length, gc_fraction, n_fraction, mean_quality_arithmetic,
    mean_quality_error_based); unavailable components are NaN, never zero.
    """
    seq = record.query_sequence
    quals = record.query_qualities
    length = float(len(seq)) if seq else float("nan")
    if seq:
        upper = seq.upper()
        a = upper.count("A")
        c = upper.count("C")
        g = upper.count("G")
        t = upper.count("T")
        acgt = a + c + g + t
        gc = (g + c) / acgt if acgt else float("nan")
        n_frac = upper.count("N") / len(upper)
    else:
        gc = float("nan")
        n_frac = float("nan")
    if quals is not None and len(quals):
        q = np.frombuffer(quals, dtype=np.uint8).astype(np.float64)
        meanq_arith = float(q.mean())
        mean_err = float(np.mean(np.power(10.0, -q / 10.0)))
        meanq_prob = -10.0 * np.log10(mean_err) if mean_err > 0 else 93.0
    else:
        meanq_arith = float("nan")
        meanq_prob = float("nan")
    return (length, gc, n_frac, meanq_arith, meanq_prob)


def scan_bam(
    dataset_id: str,
    path: Path,
    read_type: ReadType,
    sampling: SamplingConfig,
    sample_size: int,
) -> BamScanResult:
    """Fully scan one BAM: validate structure/order, count categories, sample reads.

    Sampling unit: eligible primary record with a query sequence. The reservoir key
    is drawn from the dataset's deterministic stream in record order, so the sample
    never depends on worker scheduling and never reduces structural-validation
    coverage (every record is visited).
    """
    import pysam

    started = time.perf_counter()
    try:
        af = pysam.AlignmentFile(str(path), "rb", check_sq=False)
    except (OSError, ValueError) as exc:
        raise InputDataError(f"cannot open BAM via HTSlib: {exc}", source=str(path)) from exc
    with af:
        try:
            header = af.header.to_dict()
        except (OSError, ValueError) as exc:
            raise InputDataError(f"cannot read BAM header: {exc}", source=str(path)) from exc
        sort_order = str(header.get("HD", {}).get("SO", "unknown"))
        if sort_order != "coordinate":
            raise InputDataError(
                f"BAM is not declared coordinate-sorted (HD SO:{sort_order!r}); v1 "
                "requires coordinate-sorted BAM",
                source=str(path),
            )
        references = [
            (str(sq.get("SN", "")), int(sq.get("LN", 0)), sq.get("M5"))
            for sq in header.get("SQ", []) or []
        ]
        read_groups = [dict(rg) for rg in header.get("RG", []) or []]

        flags = BamFlagCounts()
        length_edges = (
            dist.SHORT_LENGTH_EDGES if read_type in SHORT_PROFILES else dist.LONG_LENGTH_EDGES
        )
        mapq_acc = dist.HistogramAccumulator("mapq", dist.MAPQ_EDGES)
        insert_acc = dist.HistogramAccumulator("insert_size", dist.INSERT_EDGES)
        length_acc = dist.HistogramAccumulator("read_length", length_edges)
        buf_mapq: list[int] = []
        buf_insert: list[int] = []
        buf_len: list[int] = []

        rng = np.random.default_rng(dataset_seed(sampling.master_seed, dataset_id))
        reservoir = FeatureReservoir(sample_size)
        insert_reservoir = FeatureReservoir(sample_size)
        prev_tid = -2
        prev_pos = -1
        seen_unplaced = False
        record_no = 0
        total_bases = 0
        eligible = 0
        length_sum = length_sqsum = length_n = 0
        insert_sum = insert_sqsum = insert_n = 0

        try:
            for record in af.fetch(until_eof=True):
                record_no += 1
                flag = record.flag
                tid = record.reference_id
                pos = record.reference_start
                if tid == -1:
                    seen_unplaced = True
                else:
                    if seen_unplaced:
                        raise InputDataError(
                            "records are not coordinate-sorted: a placed record "
                            "follows unplaced (tid -1) records",
                            source=str(path),
                            location=f"record {record_no}",
                        )
                    if tid < prev_tid or (tid == prev_tid and pos < prev_pos):
                        raise InputDataError(
                            "records are not coordinate-sorted: position moved "
                            f"backwards ({prev_tid}:{prev_pos} -> {tid}:{pos}) "
                            "despite header SO:coordinate",
                            source=str(path),
                            location=f"record {record_no}",
                        )
                    prev_tid, prev_pos = tid, pos

                is_secondary = bool(flag & FLAG_SECONDARY)
                is_supplementary = bool(flag & FLAG_SUPPL)
                flags.total_records += 1
                if is_secondary:
                    flags.secondary += 1
                    continue
                if is_supplementary:
                    flags.supplementary += 1
                    continue
                flags.primary += 1
                unmapped = bool(flag & FLAG_UNMAPPED)
                if unmapped:
                    flags.unmapped_primary += 1
                else:
                    flags.mapped_primary += 1
                    buf_mapq.append(record.mapping_quality)
                if flag & FLAG_DUP:
                    flags.duplicate_marked_primary += 1
                if flag & FLAG_QCFAIL:
                    flags.qcfail_primary += 1
                if flag & FLAG_PAIRED:
                    flags.paired_primary += 1
                    if flag & FLAG_PROPER:
                        flags.properly_paired_primary += 1
                    if flag & FLAG_FIRST:
                        flags.first_in_pair_primary += 1
                    if flag & FLAG_FIRST and not unmapped:
                        flags.read1_mapped += 1
                    if flag & FLAG_FIRST and flag & FLAG_PROPER and not unmapped:
                        tlen = record.template_length
                        if tlen > 0:
                            buf_insert.append(tlen)
                            insert_sum += tlen
                            insert_sqsum += tlen * tlen
                            insert_n += 1
                            ikey = float(rng.random())
                            if ikey < insert_reservoir.threshold or not insert_reservoir.is_full:
                                insert_reservoir.add_block(
                                    np.array([[float(tlen)]], dtype=np.float64),
                                    keys=np.array([ikey]),
                                )
                qlen = record.query_length or 0
                if qlen > 0:
                    buf_len.append(qlen)
                    total_bases += qlen
                    length_sum += qlen
                    length_sqsum += qlen * qlen
                    length_n += 1
                seq = record.query_sequence
                if seq is None:
                    flags.records_missing_sequence += 1
                else:
                    eligible += 1
                    if record.query_qualities is None:
                        flags.records_missing_qualities += 1
                    key = float(rng.random())
                    if key < reservoir.threshold or not reservoir.is_full:
                        feats = _record_features(record)
                        reservoir.add_block(
                            np.array([feats], dtype=np.float64),
                            keys=np.array([key]),
                        )
                if len(buf_mapq) >= HIST_FLUSH_RECORDS:
                    mapq_acc.add_values(buf_mapq)
                    buf_mapq = []
                if len(buf_insert) >= HIST_FLUSH_RECORDS:
                    insert_acc.add_values(buf_insert)
                    buf_insert = []
                if len(buf_len) >= HIST_FLUSH_RECORDS:
                    length_acc.add_values(buf_len)
                    buf_len = []
        except InputDataError:
            raise
        except (OSError, ValueError) as exc:
            raise InputDataError(
                f"truncated or corrupt BAM stream: {exc}",
                source=str(path),
                location=f"record {record_no + 1}",
            ) from exc
        mapq_acc.add_values(buf_mapq)
        insert_acc.add_values(buf_insert)
        length_acc.add_values(buf_len)

    return BamScanResult(
        dataset_id=dataset_id,
        flags=flags,
        header=BamHeaderInfo(sort_order=sort_order, references=references, read_groups=read_groups),
        eligible_observations=eligible,
        total_bases=total_bases,
        mapq_hist=mapq_acc.result(),
        insert_hist=insert_acc.result(),
        length_hist=length_acc.result(),
        sample_reads=reservoir.sample(),
        sampled_count=reservoir.sampled_count,
        scan_seconds=time.perf_counter() - started,
        records_read=record_no,
        length_sum=length_sum,
        length_sqsum=length_sqsum,
        length_n=length_n,
        insert_sum=insert_sum,
        insert_sqsum=insert_sqsum,
        insert_n=insert_n,
        sample_inserts=insert_reservoir.sample()["length"],
    )
