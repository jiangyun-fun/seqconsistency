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
    assert res.header.references == [("chr1", 1_000_000, None)]
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
