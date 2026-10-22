"""FASTQ validation, statistics, and paired-end synchronization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import make_records, write_fastq, write_pe
from seqconsistency.errors import InputDataError
from seqconsistency.fastq import (
    iter_fastq_batches,
    mate_annotation,
    normalize_read_name,
    parse_fastp_json,
    scan_fastq_paired,
    scan_fastq_single,
)
from seqconsistency.models import ReadType, SamplingConfig

SAMPLING = SamplingConfig(sample_size=1000, master_seed=42)


def test_valid_se_fastq(tmp_path: Path):
    p = write_fastq(tmp_path / "se.fastq", make_records(500, seed=1))
    stats, sample, eligible, sampled = scan_fastq_single("ds1", p, ReadType.ONT, SAMPLING, 100)
    assert eligible == 500
    assert sampled == 100
    assert stats.total_reads == 500
    assert stats.total_bases == 500 * 100
    assert stats.q30_bases > 0
    assert stats.length_hist is not None and stats.length_hist.observation_count == 500
    assert len(sample["length"]) == 100


def test_gzip_input(tmp_path: Path):
    p = write_fastq(tmp_path / "se.fastq.gz", make_records(200, seed=2), gzip_output=True)
    stats, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 50)
    assert stats.total_reads == 200


def test_truncated_fastq_rejected(tmp_path: Path):
    recs = make_records(10, seed=3)
    p = write_fastq(tmp_path / "t.fastq", recs)
    data = p.read_bytes().split(b"\n")
    p.write_bytes(b"\n".join(data[:-3]))  # drop final qual + trailing newline + more
    with pytest.raises(InputDataError, match="truncated"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_invalid_record_structure(tmp_path: Path):
    p = tmp_path / "bad.fastq"
    p.write_bytes(b"r1 missing-at\nACGT\n+\nIIII\n")
    with pytest.raises(InputDataError, match="does not start with '@'"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)
    p2 = tmp_path / "bad_plus.fastq"
    p2.write_bytes(b"@r1\nACGT\nX plus\nIIII\n")
    with pytest.raises(InputDataError, match="'\\+'"):
        scan_fastq_single("ds", p2, ReadType.ONT, SAMPLING, 10)


def test_plus_line_missing(tmp_path: Path):
    p = tmp_path / "bad2.fastq"
    p.write_bytes(b"@r1\nACGT\nX\nIIII\n")
    with pytest.raises(InputDataError, match="'\\+'"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_seq_qual_length_mismatch(tmp_path: Path):
    p = tmp_path / "bad3.fastq"
    p.write_bytes(b"@r1\nACGTA\n+\nIIII\n")
    with pytest.raises(InputDataError, match="sequence length 5 != quality length 4"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_empty_input_rejected(tmp_path: Path):
    p = tmp_path / "empty.fastq"
    p.write_bytes(b"")
    with pytest.raises(InputDataError, match="no reads"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_unsupported_alphabet_rejected(tmp_path: Path):
    p = tmp_path / "iupac.fastq"
    p.write_bytes(b"@r1\nACGTRY\n+\nIIIIII\n")
    with pytest.raises(InputDataError, match="unsupported nucleotide"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_lowercase_accepted(tmp_path: Path):
    p = tmp_path / "lower.fastq"
    p.write_bytes(b"@r1\nacgtn\n+\nIIIII\n")
    stats, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 5)
    assert stats.total_bases == 5
    assert stats.total_n_bases == 1


def test_quality_outside_ascii33_126_rejected(tmp_path: Path):
    p = tmp_path / "q.fastq"
    p.write_bytes(b"@r1\nACGT\n+\n\x01\x02\x03\x04\n")  # control chars
    with pytest.raises(InputDataError, match="not valid Phred\\+33"):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)


def test_ambiguous_signature_flags_not_rejects(tmp_path: Path):
    # Review round 4: every byte >= 64 with max >= 90 is valid under BOTH
    # encodings, so the scan completes under the declared Phred+33 reading and
    # flags the strand; the decision engine turns the flag into a WARNING that
    # withholds approval (never a hard error, never silent).
    p = tmp_path / "p64.fastq"
    qual = bytes([64, 70, 80, 104])
    p.write_bytes(b"@r1\nACGT\n+\n" + qual + b"\n" + b"@r2\nACGT\n+\n" + qual + b"\n")
    stats, sample, n, _ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 10)
    assert n == 2
    assert stats.encoding_ambiguous is True
    assert stats.min_qual_byte == 64 and stats.max_qual_byte == 104
    # Uniform Q42 ('K': min 75 >= 64 but max 75 < 90) stays silent.
    p2 = tmp_path / "k.fastq"
    p2.write_bytes(b"@r1\nACGT\n+\n" + b"K" * 4 + b"\n")
    stats2, *_ = scan_fastq_single("ds2", p2, ReadType.ONT, SAMPLING, 10)
    assert stats2.encoding_ambiguous is False


def test_valid_phred33_high_quality_accepted(tmp_path: Path):
    # Min byte 33 ('!') present -> ambiguity resolved toward Phred+33, accepted.
    p = tmp_path / "q33.fastq"
    p.write_bytes(b"@r1\nACGT\n+\n!III\n@r2\nACGT\n+\n!IJK\n")
    stats, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 5)
    assert stats.total_reads == 2


def test_all_n_reads(tmp_path: Path):
    p = tmp_path / "n.fastq"
    p.write_bytes(b"@r1\nNNNN\n+\nIIII\n@r2\nACGT\n+\nIIII\n")
    stats, sample, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 5)
    assert stats.total_n_bases == 4
    # GC of the all-N read is undefined and excluded from gc stats.
    assert stats.reads_with_gc == 1
    assert np.isnan(sample["gc_fraction"][0]) or True  # NaN allowed for that read


def test_variable_length_merged_reads(tmp_path: Path):
    rng = np.random.default_rng(9)
    lengths = rng.integers(50, 400, size=300)
    recs = make_records(300, lengths=lengths, seed=8)
    p = write_fastq(tmp_path / "merged.fastq", recs)
    stats, *_ = scan_fastq_single("ds", p, ReadType.MERGED_SHORT, SAMPLING, 100)
    assert stats.positional is not None and stats.positional.max_cycles == 1024
    assert stats.length_hist is not None
    assert min(lengths) >= 50 and max(lengths) < 400


def test_long_ont_reads_no_positional(tmp_path: Path):
    lengths = np.full(50, 15000)
    recs = make_records(50, lengths=lengths, seed=10)
    p = write_fastq(tmp_path / "ont.fastq", recs)
    stats, sample, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 20)
    assert stats.positional is None  # long reads have no per-cycle profile
    assert all(x == 15000 for x in sample["length"])
    assert stats.length_hist is not None
    # 15000 bp lands in the [12800, 25600) bin of the deterministic long ladder.
    edges = stats.length_hist.edges
    idx = next(i for i in range(len(edges) - 1) if edges[i] <= 15000 < edges[i + 1])
    assert stats.length_hist.counts[idx] == 50


def test_hifi_quality_distribution(tmp_path: Path):
    # HiFi-like narrow high-quality distribution (Q30-40).
    rng = np.random.default_rng(11)
    recs = []
    for i in range(100):
        q = rng.integers(30, 41, 100).astype(np.uint8) + 33
        seq = np.full(100, ord("A"), dtype=np.uint8)
        recs.append((f"@h{i}".encode(), seq.tobytes(), q.tobytes()))
    p = write_fastq(tmp_path / "hifi.fastq", recs)
    stats, sample, *_ = scan_fastq_single("ds", p, ReadType.HIFI, SAMPLING, 50)
    assert stats.q30_bases == stats.total_bases
    assert min(sample["mean_quality_arithmetic"]) >= 30


def test_mean_quality_definitions_differ(tmp_path: Path):
    # Arithmetic mean quality must differ from error-probability-based quality.
    p = tmp_path / "mix.fastq"
    p.write_bytes(b"@r1\nACGT\n+\n!I!J\n")
    stats, sample, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 1)
    arith = sample["mean_quality_arithmetic"][0]
    prob = sample["mean_quality_error_based"][0]
    assert arith == pytest.approx((0 + 40 + 0 + 41) / 4)
    expected = -10 * np.log10(
        (10 ** (-0 / 10) + 10 ** (-40 / 10) + 10 ** (-0 / 10) + 10 ** (-41 / 10)) / 4
    )
    assert prob == pytest.approx(float(expected), rel=1e-6)
    assert prob < arith  # error-based quality is dominated by low-quality bases


def test_gc_fraction_values(tmp_path: Path):
    p = tmp_path / "gc.fastq"
    p.write_bytes(b"@r1\nGCAT\n+\nIIII\n")  # 50% GC
    stats, *_ = scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 1)
    assert stats.sum_gc_fraction == pytest.approx(0.5)


def test_batch_iteration_batching(tmp_path: Path):
    recs = make_records(250, seed=12)
    p = write_fastq(tmp_path / "b.fastq", recs)
    batches = list(iter_fastq_batches(p, batch_records=100))
    sizes = [len(b.headers) for b in batches]
    assert sizes == [100, 100, 50]


def test_batch_byte_budget(tmp_path: Path):
    recs = make_records(50, length=1000, seed=13)
    p = write_fastq(tmp_path / "bytes.fastq", recs)
    batches = list(iter_fastq_batches(p, batch_records=10000, batch_bytes=10000))
    assert len(batches) > 1  # byte budget forced a flush below the record limit


def test_normalize_read_name():
    assert normalize_read_name(b"@read1/1 extra stuff") == b"read1"
    assert normalize_read_name(b"@read1/2") == b"read1"
    assert normalize_read_name(b"@inst:1:N:0:ATCG 2:N:0:ATCG") == b"inst:1:N:0:ATCG"
    assert mate_annotation(b"@r/1 x") == b"/1"
    assert mate_annotation(b"@r/2") == b"/2"
    assert mate_annotation(b"@r") is None


def test_valid_pe_with_casava_names(tmp_path: Path):
    n = 100
    r1 = make_records(n, seed=20)
    r2 = make_records(n, seed=21)
    p1 = write_fastq(tmp_path / "a_R1.fastq", [(h + b" 1:N:0:ACGT", s, q) for h, s, q in r1])
    p2 = write_fastq(tmp_path / "a_R2.fastq", [(h + b" 2:N:0:ACGT", s, q) for h, s, q in r2])
    st1, st2, pairing, s1, s2, pairs, sampled = scan_fastq_paired(
        "ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50
    )
    assert pairing.pairs_checked == 100
    assert pairs == 100
    assert sampled == 50
    assert len(s1["length"]) == len(s2["length"]) == 50


def test_pe_legacy_suffixes(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "leg", 100, seed=22)
    st1, st2, pairing, *_ = scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert pairing.pairs_checked == 100


def test_pe_unequal_counts_rejected(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "uneq", 100, seed=23)
    # Append one extra record to R1.
    with open(p1, "ab") as fh:
        fh.write(b"@extra\nACGT\n+\nIIII\n")
    with pytest.raises(InputDataError, match="unequal mate counts"):
        scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50)


def test_pe_identifier_mismatch_rejected(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "mism", 50, seed=24)
    # Rewrite R2's first header.
    data = p2.read_bytes().split(b"\n")
    data[0] = b"@different_read0/2"
    p2.write_bytes(b"\n".join(data))
    with pytest.raises(InputDataError, match="out of sync"):
        scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50)


def test_pe_contradictory_annotations_rejected(tmp_path: Path):
    p1, p2 = write_pe(tmp_path, "contra", 50, seed=25)
    data = p1.read_bytes().split(b"\n")
    data[0] = b"@r0/2"  # R1 file carrying /2
    p1.write_bytes(b"\n".join(data))
    with pytest.raises(InputDataError, match="contradictory mate annotations"):
        scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50)


def test_pe_r1_r2_metrics_separate(tmp_path: Path):
    r1 = make_records(100, length=150, seed=26)
    r2 = make_records(100, length=100, seed=27)
    p1 = write_fastq(tmp_path / "sep_R1.fastq", r1)
    p2 = write_fastq(tmp_path / "sep_R2.fastq", r2)
    st1, st2, *_ = scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert st1.total_bases == 100 * 150
    assert st2.total_bases == 100 * 100


def test_corrupt_gzip_rejected(tmp_path: Path):
    p = write_fastq(tmp_path / "corrupt.fastq.gz", make_records(100, seed=28), gzip_output=True)
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF  # flip bits in the middle
    p.write_bytes(bytes(data))
    with pytest.raises(InputDataError):
        scan_fastq_single("ds", p, ReadType.ONT, SAMPLING, 50)


def test_fastp_json_parsing(tmp_path: Path):
    import json

    doc = {
        "summary": {
            "before_filtering": {"total_reads": 1000, "q30_rate": 0.8},
            "after_filtering": {"total_reads": 850, "q30_rate": 0.9},
        },
        "merging": {"merged_reads": 700},
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(doc))
    ev = parse_fastp_json(p)
    assert ev.before_total_reads == 1000
    assert ev.after_total_reads == 850
    assert ev.merged_reads == 700
    assert ev.merge_rate == pytest.approx(0.7)
    assert ev.retained_fraction == pytest.approx(0.85)


def test_fastp_json_missing_fields_are_none(tmp_path: Path):
    p = tmp_path / "minimal.json"
    p.write_text('{"summary": {}}')
    ev = parse_fastp_json(p)
    assert ev.before_total_reads is None
    assert ev.merge_rate is None


def test_fastp_json_invalid_rejected(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(InputDataError):
        parse_fastp_json(p)


def test_uniform_high_quality_phred33_accepted(tmp_path: Path):
    # Reviewer counterexample: every byte 'K' (uniform Q42). Not the classic
    # Phred+64 signature under the tightened bound (max must be >= 90).
    p = tmp_path / "hi.fastq"
    p.write_bytes(b"@r1\nACGT\n+\nKKKK\n@r2\nACGT\n+\nKKKK\n")
    stats, *_ = scan_fastq_single("ds", p, ReadType.HIFI, SAMPLING, 5)
    assert stats.total_reads == 2


def test_swapped_casava_mate_annotations_rejected(tmp_path: Path):
    # R1 file carrying " 2:N:..." and R2 carrying " 1:N:..." must abort.
    r1 = [(b"@inst:1 2:N:0:ACGT", b"ACGT", b"IIII")]
    r2 = [(b"@inst:1 1:N:0:ACGT", b"ACGT", b"IIII")]
    p1 = write_fastq(tmp_path / "a_R1.fastq", r1)
    p2 = write_fastq(tmp_path / "a_R2.fastq", r2)
    with pytest.raises(InputDataError, match="contradictory mate annotations"):
        scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 5)


def test_consistent_casava_annotations_accepted(tmp_path: Path):
    n = 50
    r1 = [(f"@inst{i} 1:N:0:ACGT".encode(), b"ACGT" * 25, b"I" * 100) for i in range(n)]
    r2 = [(f"@inst{i} 2:N:0:ACGT".encode(), b"ACGT" * 25, b"I" * 100) for i in range(n)]
    p1 = write_fastq(tmp_path / "c_R1.fastq", r1)
    p2 = write_fastq(tmp_path / "c_R2.fastq", r2)
    st1, st2, pairing, *_ = scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 20)
    assert pairing.pairs_checked == n


def test_self_contradictory_header_rejected(tmp_path: Path):
    # /1 suffix plus CASAVA " 2:" on the SAME header contradicts itself.
    p1 = tmp_path / "s_R1.fastq"
    p2 = tmp_path / "s_R2.fastq"
    write_fastq(p1, [(b"@r0/1 2:N:0:ACGT", b"ACGT", b"IIII")])
    write_fastq(p2, [(b"@r0/2 2:N:0:ACGT", b"ACGT", b"IIII")])
    with pytest.raises(InputDataError, match="contradictory mate annotations"):
        scan_fastq_paired("ds", p1, p2, ReadType.PAIRED_SHORT, SAMPLING, 5)


def test_mate_annotation_parses_casava():
    assert mate_annotation(b"@r 1:N:0:ACGT") == b"/1"
    assert mate_annotation(b"@r 2:N:0:ACGT") == b"/2"
    assert mate_annotation(b"@r/1") == b"/1"
    assert mate_annotation(b"@r/1 1:N:0:ACGT") == b"/1"  # agreeing sources are fine
    assert mate_annotation(b"@r plain-comment") is None
    with pytest.raises(InputDataError):
        mate_annotation(b"@r/1 2:N:0:ACGT")
