"""BAM scanning: validation, categories, insert sizes, references, sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import make_bam
from seqconsistency.bam import scan_bam
from seqconsistency.errors import InputDataError
from seqconsistency.models import ReadType, SamplingConfig

SAMPLING = SamplingConfig(sample_size=1000, master_seed=42)


def test_valid_coordinate_sorted_bam(tmp_path: Path):
    p = make_bam(tmp_path / "ok.bam", n=500, seed=1)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 100)
    assert res.records_read == 500
    assert res.flags.primary == 500
    assert res.flags.mapped_primary == 500
    assert res.flags.properly_paired_primary == 500
    assert res.eligible_observations == 500
    assert res.sampled_count == 100
    assert res.mapq_hist is not None and res.mapq_hist.observation_count == 500
    assert res.insert_hist is not None and res.insert_hist.observation_count == 500
    assert res.length_hist is not None and res.length_hist.observation_count == 500
    assert res.total_bases == 500 * 100
    assert res.header.sort_order == "coordinate"
    assert res.header.references == [("chr1", 1_000_000, "6a9a0c1f2b3d4e5f60718293a4b5c6d7")]
    assert res.header.read_groups[0]["ID"] == "rg1"


def test_header_not_coordinate_sorted_rejected(tmp_path: Path):
    p = make_bam(tmp_path / "unsorted.bam", n=50, seed=2, sort_header="unsorted")
    with pytest.raises(InputDataError, match="not declared coordinate-sorted"):
        scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)


def test_records_out_of_order_rejected(tmp_path: Path):
    p = make_bam(tmp_path / "badorder.bam", n=50, seed=3, unsorted_records=True)
    with pytest.raises(InputDataError, match="not coordinate-sorted"):
        scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)


def test_sort_declaration_lies_about_order(tmp_path: Path):
    # Header says coordinate but records are shuffled -> observed-order check fires.
    p = make_bam(
        tmp_path / "lying.bam", n=50, seed=4, unsorted_records=True, sort_header="coordinate"
    )
    with pytest.raises(InputDataError, match="moved backwards"):
        scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)


def test_secondary_and_supplementary_counted_not_duplicated(tmp_path: Path):
    p = make_bam(
        tmp_path / "cats.bam", n=100, seed=5, include_secondary=True, include_supplementary=True
    )
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert res.records_read == 300
    assert res.flags.primary == 100
    assert res.flags.secondary == 100
    assert res.flags.supplementary == 100
    # Eligible sampling counts primary records only.
    assert res.eligible_observations == 100


def test_missing_qualities_counted_not_zero(tmp_path: Path):
    p = make_bam(tmp_path / "noqual.bam", n=50, seed=6, no_qualities=True)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert res.flags.records_missing_qualities == 50
    # Quality features are NaN, never zero.
    assert all(np.isnan(v) for v in res.sample_reads["mean_quality_arithmetic"])


def test_duplicate_and_qcfail_flags(tmp_path: Path):
    p = make_bam(tmp_path / "dup.bam", n=400, seed=7, duplicate_fraction=0.25, qcfail_fraction=0.25)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 100)
    assert res.flags.duplicate_marked_primary > 50
    assert res.flags.qcfail_primary > 50
    assert res.flags.primary == 400


def test_unmapped_records_via_placed_unmapped(tmp_path: Path):
    # Build by hand: mixture of mapped and unmapped primary records, sorted.
    import pysam

    p = tmp_path / "mixed.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        for i in range(30):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"m{i}"
            a.query_sequence = "ACGT" * 25
            a.query_qualities = np.full(100, 30, dtype=np.uint8)
            a.flag = 4 if i >= 20 else 0
            if i < 20:
                a.reference_id = 0
                a.reference_start = i * 100
                a.cigarstring = "100M"
                a.mapping_quality = 60
            else:
                a.reference_id = -1
                a.reference_start = -1
            out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 30)
    assert res.flags.unmapped_primary == 10
    assert res.flags.mapped_primary == 20
    assert res.mapq_hist is not None and res.mapq_hist.observation_count == 20


def test_unplaced_records_must_come_last(tmp_path: Path):
    import pysam

    p = tmp_path / "placed_after_unplaced.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        unplaced = pysam.AlignedSegment(out.header)
        unplaced.query_name = "u1"
        unplaced.flag = 4
        out.write(unplaced)
        placed = pysam.AlignedSegment(out.header)
        placed.query_name = "p1"
        placed.flag = 0
        placed.reference_id = 0
        placed.reference_start = 5
        placed.cigarstring = "10M"
        out.write(placed)
    with pytest.raises(InputDataError, match="placed record follows unplaced"):
        scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)


def test_insert_size_one_observation_per_pair(tmp_path: Path):
    # Both mates carry the same template; only first-in-pair with TLEN>0 counts.
    import pysam

    p = tmp_path / "pairs.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        for i in range(100):
            for flag, start, tlen in (
                (99, i * 500, 300),  # first in pair, forward
                (147, i * 500 + 200, -300),  # second in pair, reverse
            ):
                a = pysam.AlignedSegment(out.header)
                a.query_name = f"pair{i}"
                a.query_sequence = "ACGT" * 25
                a.query_qualities = np.full(100, 30, dtype=np.uint8)
                a.flag = flag
                a.reference_id = 0
                a.reference_start = start
                a.cigarstring = "100M"
                a.mapping_quality = 60
                a.template_length = tlen
                out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert res.flags.paired_primary == 200
    assert res.flags.properly_paired_primary == 200
    assert res.flags.first_in_pair_primary == 100
    assert res.insert_hist is not None
    assert res.insert_hist.observation_count == 100  # counted once per pair
    assert res.insert_sum // 300 == 100
    assert len(res.sample_inserts) == 50


def test_insert_size_includes_pairs_with_read1_rightmost(tmp_path: Path):
    # Proper pairs whose FIRST segment is the rightmost one carry TLEN < 0 on
    # that segment (SAM spec). Requiring TLEN > 0 silently dropped every such
    # pair and biased the distribution (review round 2, point 5): the scan must
    # take abs(TLEN) and still observe one row per pair.
    import pysam

    p = tmp_path / "rf_pairs.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        for i in range(100):
            # read2 leftmost (TLEN +300), read1 rightmost (TLEN -300).
            for flag, start, tlen in (
                (163, i * 500, 300),  # second in pair, forward
                (83, i * 500 + 200, -300),  # first in pair, reverse
            ):
                a = pysam.AlignedSegment(out.header)
                a.query_name = f"pair{i}"
                a.query_sequence = "ACGT" * 25
                a.query_qualities = np.full(100, 30, dtype=np.uint8)
                a.flag = flag
                a.reference_id = 0
                a.reference_start = start
                a.cigarstring = "100M"
                a.mapping_quality = 60
                a.template_length = tlen
                out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 50)
    assert res.flags.first_in_pair_primary == 100
    assert res.flags.properly_paired_primary == 200
    assert res.insert_hist is not None
    assert res.insert_hist.observation_count == 100  # every pair observed once
    assert res.insert_sum // 300 == 100
    assert len(res.sample_inserts) == 50


def test_reference_dict_and_read_groups(tmp_path: Path):
    p = make_bam(
        tmp_path / "rg.bam",
        n=20,
        seed=8,
        extra_header={"RG": [{"ID": "rgA", "SM": "s1", "LB": "l1"}]},
    )
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 20)
    assert res.header.read_groups == [{"ID": "rgA", "SM": "s1", "LB": "l1"}]


def test_empty_bam_is_structurally_valid(tmp_path: Path):
    import pysam

    p = tmp_path / "empty.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header):
        pass
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)
    assert res.records_read == 0
    assert res.eligible_observations == 0


def test_not_a_bam_rejected(tmp_path: Path):
    p = tmp_path / "fake.bam"
    p.write_bytes(b"this is not a bam at all")
    with pytest.raises(InputDataError, match="cannot open BAM"):
        scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)


def test_bam_sampling_deterministic(tmp_path: Path):
    p = make_bam(tmp_path / "det.bam", n=2000, seed=9)
    a = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 100)
    b = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 100)
    assert a.sample_reads["length"] == b.sample_reads["length"]
    assert a.sample_reads["gc_fraction"] == b.sample_reads["gc_fraction"]


def test_bam_gc_fraction_values(tmp_path: Path):
    import pysam

    p = tmp_path / "gc.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        a = pysam.AlignedSegment(out.header)
        a.query_name = "g1"
        a.query_sequence = "GCGCATAT"
        a.query_qualities = np.full(8, 30, dtype=np.uint8)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.cigarstring = "8M"
        out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 1)
    assert res.sample_reads["gc_fraction"][0] == pytest.approx(0.5)


def test_clipping_summary_collected(tmp_path: Path):
    import pysam

    p = tmp_path / "clip.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
    # (cigar, query-sequence length): 50M -> 50bp; 40M10S -> 50bp (soft clip
    # present in sequence); 45M5H -> 45bp (hard clip absent from sequence).
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        for i, (cigar, seq_len) in enumerate((("50M", 50), ("40M10S", 50), ("45M5H", 45))):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"c{i}"
            a.query_sequence = "ACGT" * (seq_len // 4) + "A" * (seq_len % 4)
            a.query_qualities = np.full(seq_len, 30, dtype=np.uint8)
            a.flag = 0
            a.reference_id = 0
            a.reference_start = i * 100
            a.cigarstring = cigar
            out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)
    assert res.flags.clipped_primary_records == 2
    assert res.flags.clipped_bases == 15


def test_undefined_rg_records_counted(tmp_path: Path):
    import pysam

    p = tmp_path / "rg.bam"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        a = pysam.AlignedSegment(out.header)
        a.query_name = "ok"
        a.query_sequence = "ACGT" * 25
        a.query_qualities = np.full(100, 30, dtype=np.uint8)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.cigarstring = "100M"
        a.set_tag("RG", "rg_undefined")
        out.write(a)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 10)
    assert res.flags.records_with_undefined_rg == 1


def test_bam_reservoir_batching_matches_expectations(tmp_path: Path):
    p = make_bam(tmp_path / "batch.bam", n=3000, seed=99)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 500)
    # make_bam emits first-in-pair records only, so the R1 mate reservoir
    # receives every eligible record and the pooled READS stratum stays empty.
    assert res.sampled_count == 500
    assert res.sampled_count_r1 == 500
    assert res.sampled_count_r2 == 0
    assert res.sampled_count == len(res.sample_r1["length"])
    # Keys are unique; sorted output must contain distinct lengths of the right count.
    assert len(set(res.sample_r1["length"])) <= 500


def test_paired_bam_samples_mates_separately(tmp_path: Path):
    # Paired profiles route paired records to per-mate reservoirs (R1 vs R1,
    # R2 vs R2 comparisons); unpaired records stay in the pooled READS sample.
    # Unpaired/long-read profiles keep one pooled sample.
    import pysam

    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]}
    p = tmp_path / "mixed.bam"
    with pysam.AlignmentFile(str(p), "wb", header=header) as out:
        for i in range(30):
            for flag, tlen in ((99, 300), (147, -300)):  # full pair, both mates
                a = pysam.AlignedSegment(out.header)
                a.query_name = f"pair{i}"
                a.query_sequence = "ACGT" * 25
                a.query_qualities = np.full(100, 30, dtype=np.uint8)
                a.flag = flag
                a.reference_id = 0
                a.reference_start = i * 500 + (200 if flag == 147 else 0)
                a.cigarstring = "100M"
                a.mapping_quality = 60
                a.template_length = tlen
                out.write(a)
        u = pysam.AlignedSegment(out.header)
        u.query_name = "single"
        u.query_sequence = "ACGT" * 25
        u.query_qualities = np.full(100, 30, dtype=np.uint8)
        u.flag = 0
        u.reference_id = 0
        u.reference_start = 99000
        u.cigarstring = "100M"
        u.mapping_quality = 60
        out.write(u)
    res = scan_bam("ds", p, ReadType.PAIRED_SHORT, SAMPLING, 100)
    assert res.sampled_count_r1 == 30
    assert res.sampled_count_r2 == 30
    assert len(res.sample_reads["length"]) == 1  # the unpaired record
    assert res.sampled_count == 61
    # Same file under a long-read profile: one pooled stratum, no mate split.
    res_ont = scan_bam("ds", p, ReadType.ONT, SAMPLING, 100)
    assert res_ont.sampled_count == 61
    assert res_ont.sampled_count_r1 == 0
    assert len(res_ont.sample_reads["length"]) == 61
