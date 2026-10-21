"""Metric assembly: turn scan results into long-format metric and distribution rows.

Row keys match the Parquet schemas in :mod:`seqconsistency.parquet_io`. Every metric
row states its unit, denominator, observation count, sampling mode, and availability —
a value is never emitted with a hidden default of zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from seqconsistency import distributions as dist
from seqconsistency.models import (
    STRATUM_PAIR,
    STRATUM_R1,
    STRATUM_R2,
    STRATUM_READS,
    STRATUM_SE,
    Availability,
    DatasetOutcome,
    FastqFileStats,
    Histogram,
    SamplingMode,
)

METRIC_ALGORITHM_VERSION = "metrics-v1"


def _row(
    dataset_id: str,
    pool_group: str,
    component: str,
    stratum: str,
    metric: str,
    statistic: str,
    value: float | None,
    unit: str,
    observation_count: int,
    denominator: int | None,
    sampling_mode: SamplingMode | None,
    availability: Availability = Availability.AVAILABLE,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "pool_group": pool_group,
        "component": component,
        "stratum": stratum,
        "metric": metric,
        "statistic": statistic,
        "value": value,
        "unit": unit,
        "observation_count": observation_count,
        "denominator": denominator,
        "sampling_mode": sampling_mode,
        "availability": availability,
    }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _sample_stats(
    values: list[float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(mean, std, median, p10, p90) of finite sample values; None when empty."""
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return (None,) * 5
    std = float(arr.std()) if arr.size > 1 else 0.0
    return (
        float(arr.mean()),
        std,
        float(np.median(arr)),
        float(np.quantile(arr, 0.10)),
        float(np.quantile(arr, 0.90)),
    )


def _fastq_stratum_rows(
    dataset_id: str, pool_group: str, stratum: str, st: FastqFileStats
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comp = "fastq"

    def add(
        metric: str,
        statistic: str,
        value: float | None,
        unit: str,
        n: int,
        den: int | None,
        mode: SamplingMode | None = None,
        avail: Availability = Availability.AVAILABLE,
    ) -> None:
        rows.append(
            _row(
                dataset_id,
                pool_group,
                comp,
                stratum,
                metric,
                statistic,
                value,
                unit,
                n,
                den,
                mode,
                avail,
            )
        )

    full = SamplingMode.FULL_SCAN
    add("reads", "total", float(st.total_reads), "reads", st.total_reads, None, full)
    add("bases", "total", float(st.total_bases), "bases", st.total_bases, None, full)
    if st.total_reads:
        mean_len = st.sum_lengths / st.total_reads
        var = st.sum_sq_lengths / st.total_reads - mean_len * mean_len
        add("read_length", "mean", mean_len, "bp", st.total_reads, st.total_reads, full)
        add(
            "read_length",
            "std",
            math.sqrt(max(var, 0.0)),
            "bp",
            st.total_reads,
            st.total_reads,
            full,
        )
    else:
        add("read_length", "mean", None, "bp", 0, 0, None, Availability.INSUFFICIENT_OBSERVATIONS)
        add("read_length", "std", None, "bp", 0, 0, None, Availability.INSUFFICIENT_OBSERVATIONS)
    if st.base_qual_counts and sum(st.base_qual_counts):
        add(
            "base_quality",
            "q20_fraction",
            st.q20_bases / st.total_bases,
            "fraction",
            st.total_bases,
            st.total_bases,
            full,
        )
        add(
            "base_quality",
            "q30_fraction",
            st.q30_bases / st.total_bases,
            "fraction",
            st.total_bases,
            st.total_bases,
            full,
        )
    else:
        add(
            "base_quality",
            "q20_fraction",
            None,
            "fraction",
            0,
            st.total_bases,
            None,
            Availability.INSUFFICIENT_OBSERVATIONS,
        )
        add(
            "base_quality",
            "q30_fraction",
            None,
            "fraction",
            0,
            st.total_bases,
            None,
            Availability.INSUFFICIENT_OBSERVATIONS,
        )
    add(
        "n_content",
        "base_fraction",
        _safe_div(st.total_n_bases, st.total_bases),
        "fraction",
        st.total_bases,
        st.total_bases,
        full,
        Availability.AVAILABLE if st.total_bases else Availability.INSUFFICIENT_OBSERVATIONS,
    )
    add(
        "gc_content",
        "per_read_mean",
        _safe_div(st.sum_gc_fraction, st.reads_with_gc),
        "fraction",
        st.reads_with_gc,
        st.reads_with_gc,
        full,
        Availability.AVAILABLE if st.reads_with_gc else Availability.INSUFFICIENT_OBSERVATIONS,
    )
    add(
        "read_quality",
        "mean_of_arithmetic_mean",
        _safe_div(st.sum_read_mean_quality, st.reads_with_mean_quality),
        "phred",
        st.reads_with_mean_quality,
        st.reads_with_mean_quality,
        full,
        Availability.AVAILABLE
        if st.reads_with_mean_quality
        else Availability.INSUFFICIENT_OBSERVATIONS,
    )
    add(
        "read_quality",
        "mean_of_error_based_mean",
        _safe_div(st.sum_read_quality_error_based, st.reads_with_mean_quality),
        "phred",
        st.reads_with_mean_quality,
        st.reads_with_mean_quality,
        full,
        Availability.AVAILABLE
        if st.reads_with_mean_quality
        else Availability.INSUFFICIENT_OBSERVATIONS,
    )
    return rows


def _sample_quantile_rows(
    dataset_id: str,
    pool_group: str,
    component: str,
    stratum: str,
    sample: dict[str, list[float]],
    sampled_count: int,
) -> list[dict[str, Any]]:
    """Sample-based length and mean-quality statistics for one stratum."""
    rows: list[dict[str, Any]] = []
    for metric, feature, unit in (
        ("read_length", "length", "bp"),
        ("read_quality", "mean_quality_arithmetic", "phred"),
    ):
        values = sample.get(feature, [])
        if sampled_count == 0:
            for statistic in ("mean", "std", "median", "p10", "p90"):
                rows.append(
                    _row(
                        dataset_id,
                        pool_group,
                        component,
                        stratum,
                        metric,
                        statistic,
                        None,
                        unit,
                        0,
                        0,
                        SamplingMode.SAMPLE,
                        Availability.INSUFFICIENT_OBSERVATIONS,
                    )
                )
            continue
        mean, std, median, p10, p90 = _sample_stats(values)
        for statistic, value in (
            ("mean", mean),
            ("std", std),
            ("median", median),
            ("p10", p10),
            ("p90", p90),
        ):
            n = len([v for v in values if math.isfinite(v)])
            avail = Availability.AVAILABLE if n else Availability.INSUFFICIENT_OBSERVATIONS
            rows.append(
                _row(
                    dataset_id,
                    pool_group,
                    component,
                    stratum,
                    metric,
                    statistic,
                    value,
                    unit,
                    n,
                    n,
                    SamplingMode.SAMPLE,
                    avail,
                )
            )
    return rows


def _bam_rows(dataset_id: str, pool_group: str, outcome: DatasetOutcome) -> list[dict[str, Any]]:
    bam = outcome.bam
    assert bam is not None
    rows: list[dict[str, Any]] = []
    f = bam.flags
    comp = "alignment"

    def add(
        metric: str,
        statistic: str,
        value: float | None,
        unit: str,
        n: int,
        den: int | None,
        mode: SamplingMode = SamplingMode.FULL_SCAN,
    ) -> None:
        rows.append(
            _row(
                dataset_id,
                pool_group,
                comp,
                STRATUM_READS,
                metric,
                statistic,
                value,
                unit,
                n,
                den,
                mode,
            )
        )

    add("records", "total", float(f.total_records), "records", f.total_records, None)
    add("records", "primary", float(f.primary), "records", f.primary, f.total_records)
    add("records", "secondary", float(f.secondary), "records", f.secondary, f.total_records)
    add(
        "records",
        "supplementary",
        float(f.supplementary),
        "records",
        f.supplementary,
        f.total_records,
    )
    add(
        "records",
        "primary_mapped_fraction",
        _safe_div(f.mapped_primary, f.primary),
        "fraction",
        f.primary,
        f.primary,
    )
    add(
        "records",
        "primary_unmapped_fraction",
        _safe_div(f.unmapped_primary, f.primary),
        "fraction",
        f.primary,
        f.primary,
    )
    add(
        "records",
        "duplicate_marked_fraction",
        _safe_div(f.duplicate_marked_primary, f.primary),
        "fraction",
        f.primary,
        f.primary,
    )
    add(
        "records",
        "qcfail_fraction",
        _safe_div(f.qcfail_primary, f.primary),
        "fraction",
        f.primary,
        f.primary,
    )
    add(
        "records",
        "paired_fraction",
        _safe_div(f.paired_primary, f.primary),
        "fraction",
        f.primary,
        f.primary,
    )
    add(
        "records",
        "properly_paired_fraction",
        _safe_div(f.properly_paired_primary, f.paired_primary),
        "fraction",
        f.paired_primary,
        f.paired_primary,
    )
    add(
        "records",
        "missing_qualities",
        float(f.records_missing_qualities),
        "records",
        f.records_missing_qualities,
        f.eligible_observations,
    )
    add(
        "records",
        "missing_sequence",
        float(f.records_missing_sequence),
        "records",
        f.records_missing_sequence,
        f.total_records,
    )
    add("bases", "total", float(bam.total_bases), "bases", bam.total_bases, None)
    add(
        "reads",
        "eligible_primary",
        float(bam.eligible_observations),
        "records",
        bam.eligible_observations,
        None,
    )
    if bam.length_n:
        mean_len = bam.length_sum / bam.length_n
        var = bam.length_sqsum / bam.length_n - mean_len * mean_len
        add("read_length", "mean", mean_len, "bp", bam.length_n, bam.length_n)
        add("read_length", "std", math.sqrt(max(var, 0.0)), "bp", bam.length_n, bam.length_n)
    else:
        for statistic in ("mean", "std"):
            rows.append(
                _row(
                    dataset_id,
                    pool_group,
                    comp,
                    STRATUM_READS,
                    "read_length",
                    statistic,
                    None,
                    "bp",
                    0,
                    0,
                    None,
                    Availability.INSUFFICIENT_OBSERVATIONS,
                )
            )
    if bam.insert_n:
        mean_ins = bam.insert_sum / bam.insert_n
        var = bam.insert_sqsum / bam.insert_n - mean_ins * mean_ins
        rows.append(
            _row(
                dataset_id,
                pool_group,
                comp,
                STRATUM_PAIR,
                "insert_size",
                "mean",
                mean_ins,
                "bp",
                bam.insert_n,
                bam.insert_n,
                SamplingMode.FULL_SCAN,
            )
        )
        rows.append(
            _row(
                dataset_id,
                pool_group,
                comp,
                STRATUM_PAIR,
                "insert_size",
                "std",
                math.sqrt(max(var, 0.0)),
                "bp",
                bam.insert_n,
                bam.insert_n,
                SamplingMode.FULL_SCAN,
            )
        )
    if bam.mapq_hist and bam.mapq_hist.observation_count:
        counts = np.asarray(bam.mapq_hist.counts, dtype=np.float64)
        n = counts.sum()
        mean_mapq = float((counts * np.arange(counts.size)).sum() / n)
        var = float((counts * np.arange(counts.size) ** 2).sum() / n - mean_mapq**2)
        add("mapq", "mean", mean_mapq, "mapq", int(n), int(n))
        add("mapq", "std", math.sqrt(max(var, 0.0)), "mapq", int(n), int(n))
    rows.extend(
        _sample_quantile_rows(
            dataset_id, pool_group, comp, STRATUM_READS, bam.sample_reads, bam.sampled_count
        )
    )
    if bam.sample_inserts:
        mean, std, median, p10, p90 = _sample_stats(bam.sample_inserts)
        for statistic, value in (
            ("mean", mean),
            ("std", std),
            ("median", median),
            ("p10", p10),
            ("p90", p90),
        ):
            rows.append(
                _row(
                    dataset_id,
                    pool_group,
                    comp,
                    STRATUM_PAIR,
                    "insert_size",
                    statistic,
                    value,
                    "bp",
                    len(bam.sample_inserts),
                    len(bam.sample_inserts),
                    SamplingMode.SAMPLE,
                )
            )
    return rows


def metric_rows(outcome: DatasetOutcome) -> list[dict[str, Any]]:
    """All metric rows for one dataset (long format, one row per statistic)."""
    rows: list[dict[str, Any]] = []
    spec = outcome.spec
    if outcome.fastq is not None:
        rows.extend(
            _fastq_stratum_rows(spec.dataset_id, spec.pool_group, STRATUM_R1, outcome.fastq.r1)
        )
        rows.extend(
            _sample_quantile_rows(
                spec.dataset_id,
                spec.pool_group,
                "fastq",
                STRATUM_R1,
                outcome.fastq.sample_r1,
                outcome.fastq.sampled_count,
            )
        )
        if outcome.fastq.r2 is not None:
            rows.extend(
                _fastq_stratum_rows(spec.dataset_id, spec.pool_group, STRATUM_R2, outcome.fastq.r2)
            )
            rows.extend(
                _sample_quantile_rows(
                    spec.dataset_id,
                    spec.pool_group,
                    "fastq",
                    STRATUM_R2,
                    outcome.fastq.sample_r2,
                    outcome.fastq.sampled_count,
                )
            )
        if outcome.fastp is not None:
            fp = outcome.fastp
            comp = "processing"
            stratum = STRATUM_SE
            for statistic, value in (
                ("before_total_reads", fp.before_total_reads),
                ("after_total_reads", fp.after_total_reads),
                ("merged_reads", fp.merged_reads),
                ("merge_rate", fp.merge_rate),
                ("retained_fraction", fp.retained_fraction),
            ):
                n = fp.before_total_reads or 0
                avail = Availability.AVAILABLE if value is not None else Availability.MISSING_DATA
                rows.append(
                    _row(
                        spec.dataset_id,
                        spec.pool_group,
                        comp,
                        stratum,
                        "fastp",
                        statistic,
                        float(value) if value is not None else None,
                        "reads" if "reads" in statistic else "fraction",
                        n,
                        n,
                        SamplingMode.FULL_SCAN,
                        avail,
                    )
                )
    if outcome.bam is not None:
        rows.extend(_bam_rows(spec.dataset_id, spec.pool_group, outcome))
    return rows


def _hist_rows(
    dataset_id: str,
    pool_group: str,
    component: str,
    stratum: str,
    hist: Histogram,
    sampling_mode: SamplingMode,
    category: str = "",
) -> list[dict[str, Any]]:
    total = hist.observation_count
    rows = []
    for i, count in enumerate(hist.counts):
        rows.append(
            {
                "dataset_id": dataset_id,
                "pool_group": pool_group,
                "component": component,
                "stratum": stratum,
                "metric": hist.name,
                "bin_index": i,
                "bin_start": hist.edges[i],
                "bin_end": hist.edges[i + 1],
                "category": category,
                "count": int(count),
                "frequency": (count / total) if total else 0.0,
                "value": None,
                "observation_count": total,
                "sampling_mode": sampling_mode,
            }
        )
    if hist.overflow or hist.underflow:
        rows.append(
            {
                "dataset_id": dataset_id,
                "pool_group": pool_group,
                "component": component,
                "stratum": stratum,
                "metric": hist.name,
                "bin_index": len(hist.counts),
                "bin_start": hist.edges[-1],
                "bin_end": float("inf"),
                "category": category,
                "count": hist.overflow,
                "frequency": (hist.overflow / total) if total else 0.0,
                "value": None,
                "observation_count": total,
                "sampling_mode": sampling_mode,
            }
        )
    return rows


def distribution_rows(outcome: DatasetOutcome) -> list[dict[str, Any]]:
    """All distribution rows for one dataset (long format, one row per bin)."""
    rows: list[dict[str, Any]] = []
    spec = outcome.spec
    if outcome.fastq is not None:
        for stratum, st in (
            (STRATUM_R1, outcome.fastq.r1),
            (STRATUM_R2, outcome.fastq.r2),
        ):
            if st is None:
                continue
            for hist, mode in (
                (st.length_hist, SamplingMode.FULL_SCAN),
                (st.gc_hist, SamplingMode.FULL_SCAN),
                (st.n_hist, SamplingMode.FULL_SCAN),
            ):
                if hist is not None:
                    rows.extend(
                        _hist_rows(spec.dataset_id, spec.pool_group, "fastq", stratum, hist, mode)
                    )
            # Base-quality histogram: 94 Phred bins, category empty.
            bq = st.base_qual_counts
            if sum(bq):
                hist = Histogram(
                    "base_quality", dist.PHRED_EDGES, list(bq), observation_count=int(sum(bq))
                )
                rows.extend(
                    _hist_rows(
                        spec.dataset_id,
                        spec.pool_group,
                        "fastq",
                        stratum,
                        hist,
                        SamplingMode.FULL_SCAN,
                    )
                )
            # Nucleotide composition: categorical.
            total = sum(st.base_counts)
            if total:
                for i, cat in enumerate(dist.NUC_CATEGORIES):
                    rows.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "pool_group": spec.pool_group,
                            "component": "fastq",
                            "stratum": stratum,
                            "metric": "nucleotide_composition",
                            "bin_index": i,
                            "bin_start": float("nan"),
                            "bin_end": float("nan"),
                            "category": cat,
                            "count": st.base_counts[i],
                            "frequency": st.base_counts[i] / total,
                            "value": None,
                            "observation_count": total,
                            "sampling_mode": SamplingMode.FULL_SCAN,
                        }
                    )
            if st.positional is not None:
                p = st.positional
                for i in range(p.max_cycles):
                    if not p.coverage[i]:
                        continue
                    rows.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "pool_group": spec.pool_group,
                            "component": "fastq",
                            "stratum": stratum,
                            "metric": "per_cycle_coverage",
                            "bin_index": i,
                            "bin_start": float(i),
                            "bin_end": float(i + 1),
                            "category": "",
                            "count": int(p.coverage[i]),
                            "frequency": p.coverage[i] / st.total_reads,
                            "value": None,
                            "observation_count": st.total_reads,
                            "sampling_mode": SamplingMode.FULL_SCAN,
                        }
                    )
                    rows.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "pool_group": spec.pool_group,
                            "component": "fastq",
                            "stratum": stratum,
                            "metric": "per_cycle_mean_quality",
                            "bin_index": i,
                            "bin_start": float(i),
                            "bin_end": float(i + 1),
                            "category": "",
                            "count": int(p.coverage[i]),
                            "frequency": p.coverage[i] / st.total_reads,
                            "value": p.qual_sum[i] / p.coverage[i],
                            "observation_count": int(p.coverage[i]),
                            "sampling_mode": SamplingMode.FULL_SCAN,
                        }
                    )
    if outcome.bam is not None:
        bam = outcome.bam
        for hist in (bam.mapq_hist, bam.insert_hist, bam.length_hist):
            if hist is not None and hist.observation_count:
                stratum = STRATUM_PAIR if hist.name == "insert_size" else STRATUM_READS
                rows.extend(
                    _hist_rows(
                        spec.dataset_id,
                        spec.pool_group,
                        "alignment",
                        stratum,
                        hist,
                        SamplingMode.FULL_SCAN,
                    )
                )
        # Sample-based per-read GC / N / mean-quality histograms for BAM.
        if bam.sampled_count:
            gc_hist = dist.make_histogram(
                "per_read_gc", bam.sample_reads["gc_fraction"], dist.GC_EDGES
            )
            rows.extend(
                _hist_rows(
                    spec.dataset_id,
                    spec.pool_group,
                    "alignment",
                    STRATUM_READS,
                    gc_hist,
                    SamplingMode.SAMPLE,
                )
            )
            n_hist = dist.make_histogram("per_read_n", bam.sample_reads["n_fraction"], dist.N_EDGES)
            rows.extend(
                _hist_rows(
                    spec.dataset_id,
                    spec.pool_group,
                    "alignment",
                    STRATUM_READS,
                    n_hist,
                    SamplingMode.SAMPLE,
                )
            )
            mq_hist = dist.make_histogram(
                "per_read_mean_quality",
                bam.sample_reads["mean_quality_arithmetic"],
                dist.MEANQ_EDGES,
            )
            rows.extend(
                _hist_rows(
                    spec.dataset_id,
                    spec.pool_group,
                    "alignment",
                    STRATUM_READS,
                    mq_hist,
                    SamplingMode.SAMPLE,
                )
            )
    return rows


def sampling_summary(outcome: DatasetOutcome) -> dict[str, Any]:
    """Sampling report per dataset (spec §5): counts, fraction, unit, algorithm, seed."""
    if outcome.fastq is not None:
        eligible = outcome.fastq.eligible_observations
        sampled = outcome.fastq.sampled_count
        unit = "mate_pair" if outcome.spec.is_paired_fastq else "read"
    elif outcome.bam is not None:
        eligible = outcome.bam.eligible_observations
        sampled = outcome.bam.sampled_count
        unit = "primary_record"
    else:
        raise AssertionError("outcome carries neither fastq nor bam result")
    return {
        "dataset_id": outcome.spec.dataset_id,
        "eligible_observations": eligible,
        "sampled_count": sampled,
        "sampling_fraction": (sampled / eligible) if eligible else 0.0,
        "sampling_unit": unit,
        "algorithm": "uniform_key_reservoir",
    }
