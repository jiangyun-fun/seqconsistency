"""Pairwise dataset comparisons within one pooling group.

Every applicable (metric, stratum) is compared for every dataset pair — pairs are
never silently omitted. Dataset ordering inside a pair is lexicographic by
``dataset_id`` so no duplicate mirrored rows can appear.

Distance conventions come from :mod:`seqconsistency.statistics` (README § Statistics):
Wasserstein-1 on empirical samples, Jensen–Shannon distance (base 2) on histograms
over identical deterministic bins. Pairwise consistency is NOT transitive; the
decision engine never assumes it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from seqconsistency import distributions as dist
from seqconsistency.models import (
    STRATUM_PAIR,
    STRATUM_R1,
    STRATUM_R2,
    STRATUM_READS,
    STRATUM_SE,
    DatasetOutcome,
    FastqFileStats,
    InputFormat,
    ReadType,
)
from seqconsistency.statistics import (
    STATS_VERSION,
    jensen_shannon_distance,
    wasserstein_distance,
)

#: Comparison status values (stable strings, part of the output schema).
STATUS_PASS = "PASS"
STATUS_EXCEED = "EXCEED"
STATUS_NO_THRESHOLD = "NO_THRESHOLD"
STATUS_INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
STATUS_MISSING_EVIDENCE = "MISSING_EVIDENCE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

METHOD_WASSERSTEIN = "wasserstein1"
METHOD_JSD = "jensenshannon_base2"
METHOD_ABS_DIFF = "absolute_difference"

#: reason codes emitted by this module (stable API).
REASON_BELOW_THRESHOLD = "TECH_BELOW_THRESHOLD"
REASON_EXCEEDED = "TECH_THRESHOLD_EXCEEDED"
REASON_NO_THRESHOLD = "TECH_NO_THRESHOLD_CONFIGURED"
REASON_MIN_OBS = "TECH_BELOW_MIN_OBSERVATIONS"
REASON_METRIC_UNAVAILABLE = "TECH_METRIC_UNAVAILABLE"
REASON_FASTP_MISSING = "FASTP_EVIDENCE_MISSING"


@dataclass(frozen=True)
class MetricDefinition:
    metric: str
    method: str
    strata: tuple[str, ...]
    unit: str


#: Applicable comparisons per profile. Only these are emitted; anything else is
#: structurally not applicable to the profile (documented, never silently skipped —
#: the HTML/README state which metrics each profile lacks).
PROFILE_METRICS: dict[ReadType, tuple[MetricDefinition, ...]] = {
    ReadType.PAIRED_SHORT: (
        MetricDefinition("read_length", METHOD_WASSERSTEIN, (STRATUM_R1, STRATUM_R2), "bp"),
        MetricDefinition(
            "mean_read_quality", METHOD_WASSERSTEIN, (STRATUM_R1, STRATUM_R2), "phred"
        ),
        MetricDefinition("gc_content", METHOD_JSD, (STRATUM_R1, STRATUM_R2), "jsd_base2"),
        MetricDefinition("n_content", METHOD_JSD, (STRATUM_R1, STRATUM_R2), "jsd_base2"),
        MetricDefinition("base_quality", METHOD_JSD, (STRATUM_R1, STRATUM_R2), "jsd_base2"),
        MetricDefinition(
            "nucleotide_composition", METHOD_JSD, (STRATUM_R1, STRATUM_R2), "jsd_base2"
        ),
    ),
    ReadType.MERGED_SHORT: (
        MetricDefinition("read_length", METHOD_WASSERSTEIN, (STRATUM_SE,), "bp"),
        MetricDefinition("mean_read_quality", METHOD_WASSERSTEIN, (STRATUM_SE,), "phred"),
        MetricDefinition("gc_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
        MetricDefinition("n_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
        MetricDefinition("base_quality", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
        MetricDefinition("nucleotide_composition", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
    ),
    ReadType.ONT: (
        MetricDefinition("read_length", METHOD_WASSERSTEIN, (STRATUM_SE,), "bp"),
        MetricDefinition("mean_read_quality", METHOD_WASSERSTEIN, (STRATUM_SE,), "phred"),
        MetricDefinition("gc_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
        MetricDefinition("n_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
    ),
    ReadType.HIFI: (
        MetricDefinition("read_length", METHOD_WASSERSTEIN, (STRATUM_SE,), "bp"),
        MetricDefinition("mean_read_quality", METHOD_WASSERSTEIN, (STRATUM_SE,), "phred"),
        MetricDefinition("gc_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
        MetricDefinition("n_content", METHOD_JSD, (STRATUM_SE,), "jsd_base2"),
    ),
}

#: Extra comparisons for BAM inputs regardless of read profile.
BAM_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("read_length", METHOD_WASSERSTEIN, (STRATUM_READS,), "bp"),
    MetricDefinition("mean_read_quality", METHOD_WASSERSTEIN, (STRATUM_READS,), "phred"),
    MetricDefinition("gc_content", METHOD_JSD, (STRATUM_READS,), "jsd_base2"),
    MetricDefinition("mapq", METHOD_JSD, (STRATUM_READS,), "jsd_base2"),
    MetricDefinition("insert_size", METHOD_WASSERSTEIN, (STRATUM_PAIR,), "bp"),
)


@dataclass
class ThresholdRule:
    value: float
    action: str  # "review" | "incompatible"
    calibrated: bool


ThresholdResolver = Callable[[str, str], ThresholdRule | None]


def _fastq_sample(outcome: DatasetOutcome, stratum: str) -> dict[str, list[float]] | None:
    if outcome.fastq is None:
        return None
    if stratum == STRATUM_R1:
        return outcome.fastq.sample_r1
    if stratum == STRATUM_R2:
        return outcome.fastq.sample_r2
    return outcome.fastq.sample_r1  # SE


def _fastq_stats(outcome: DatasetOutcome, stratum: str) -> FastqFileStats:
    assert outcome.fastq is not None
    if stratum == STRATUM_R2:
        assert outcome.fastq.r2 is not None
        return outcome.fastq.r2
    return outcome.fastq.r1


def _wasserstein_inputs(
    a: DatasetOutcome, b: DatasetOutcome, metric: str, stratum: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    def sample_values(o: DatasetOutcome) -> np.ndarray:
        if o.fastq is not None:
            sample = _fastq_sample(o, stratum)
            key = "length" if metric == "read_length" else "mean_quality_arithmetic"
            return np.asarray(sample.get(key, []), dtype=np.float64) if sample else np.zeros(0)
        assert o.bam is not None
        if metric == "insert_size":
            return np.asarray(o.bam.sample_inserts, dtype=np.float64)
        key = "length" if metric == "read_length" else "mean_quality_arithmetic"
        return np.asarray(o.bam.sample_reads.get(key, []), dtype=np.float64)

    va, vb = sample_values(a), sample_values(b)
    return (
        va[np.isfinite(va)],
        vb[np.isfinite(vb)],
        int(va[np.isfinite(va)].size),
        int(vb[np.isfinite(vb)].size),
    )


def _jsd_inputs(
    a: DatasetOutcome, b: DatasetOutcome, metric: str, stratum: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    def hist_for(o: DatasetOutcome) -> np.ndarray:
        if o.fastq is not None:
            st = _fastq_stats(o, stratum)
            if metric == "gc_content":
                h = st.gc_hist
            elif metric == "n_content":
                h = st.n_hist
            elif metric == "base_quality":
                h = dist.Histogram(
                    "base_quality",
                    dist.PHRED_EDGES,
                    list(st.base_qual_counts),
                    observation_count=int(sum(st.base_qual_counts)),
                )
                return np.asarray(h.counts, dtype=np.float64)
            elif metric == "nucleotide_composition":
                return np.asarray(st.base_counts, dtype=np.float64)
            else:
                raise AssertionError(f"unknown JSD metric {metric}")
            return np.asarray(h.counts if h else [], dtype=np.float64)
        assert o.bam is not None
        if metric == "gc_content":
            h = dist.make_histogram("per_read_gc", o.bam.sample_reads["gc_fraction"], dist.GC_EDGES)
        elif metric == "n_content":
            h = dist.make_histogram("per_read_n", o.bam.sample_reads["n_fraction"], dist.N_EDGES)
        elif metric == "mapq":
            h = o.bam.mapq_hist
        else:
            raise AssertionError(f"unknown BAM JSD metric {metric}")
        return np.asarray(h.counts if h else [], dtype=np.float64)

    ha, hb = hist_for(a), hist_for(b)
    return ha, hb, int(ha.sum()), int(hb.sum())


def _comparison_row(
    pool_group: str,
    a: str,
    b: str,
    component: str,
    stratum: str,
    metric: str,
    method: str,
    value: float | None,
    unit: str,
    threshold: float | None,
    direction: str | None,
    status: str,
    obs_a: int,
    obs_b: int,
    reason_code: str | None,
) -> dict[str, Any]:
    return {
        "pool_group": pool_group,
        "dataset_a": a,
        "dataset_b": b,
        "component": component,
        "stratum": stratum,
        "metric": metric,
        "method": method,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "threshold_direction": direction,
        "status": status,
        "observations_a": obs_a,
        "observations_b": obs_b,
        "reason_code": reason_code,
        "method_version": STATS_VERSION,
    }


def _compare_one(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    definition: MetricDefinition,
    stratum: str,
    min_observations: int,
    resolver: ThresholdResolver,
) -> dict[str, Any]:
    component = "fastq" if a.spec.input_format is InputFormat.FASTQ else "alignment"

    # At least one observation per side is always required: distances on empty
    # samples are undefined, so min_observations=0 still means >= 1.
    effective_min = max(min_observations, 1)
    if definition.method == METHOD_WASSERSTEIN:
        va, vb, na, nb = _wasserstein_inputs(a, b, definition.metric, stratum)
        if na < effective_min or nb < effective_min:
            return _comparison_row(
                pool_group,
                a.spec.dataset_id,
                b.spec.dataset_id,
                component,
                stratum,
                definition.metric,
                definition.method,
                None,
                definition.unit,
                None,
                None,
                STATUS_INSUFFICIENT_OBSERVATIONS,
                na,
                nb,
                REASON_MIN_OBS,
            )
        value = wasserstein_distance(va, vb, definition.metric)
    elif definition.method == METHOD_JSD:
        ha, hb, na, nb = _jsd_inputs(a, b, definition.metric, stratum)
        if na < effective_min or nb < effective_min:
            return _comparison_row(
                pool_group,
                a.spec.dataset_id,
                b.spec.dataset_id,
                component,
                stratum,
                definition.metric,
                definition.method,
                None,
                definition.unit,
                None,
                None,
                STATUS_INSUFFICIENT_OBSERVATIONS,
                na,
                nb,
                REASON_MIN_OBS,
            )
        value = jensen_shannon_distance(ha, hb, definition.metric)
    else:
        raise AssertionError(f"unknown method {definition.method}")

    rule = resolver(definition.metric, stratum)
    if rule is None:
        return _comparison_row(
            pool_group,
            a.spec.dataset_id,
            b.spec.dataset_id,
            component,
            stratum,
            definition.metric,
            definition.method,
            value,
            definition.unit,
            None,
            None,
            STATUS_NO_THRESHOLD,
            na,
            nb,
            REASON_NO_THRESHOLD,
        )
    exceeds = not (value <= rule.value)
    if not exceeds:
        return _comparison_row(
            pool_group,
            a.spec.dataset_id,
            b.spec.dataset_id,
            component,
            stratum,
            definition.metric,
            definition.method,
            value,
            definition.unit,
            rule.value,
            "max",
            STATUS_PASS,
            na,
            nb,
            REASON_BELOW_THRESHOLD,
        )
    return _comparison_row(
        pool_group,
        a.spec.dataset_id,
        b.spec.dataset_id,
        component,
        stratum,
        definition.metric,
        definition.method,
        value,
        definition.unit,
        rule.value,
        "max",
        STATUS_EXCEED,
        na,
        nb,
        REASON_EXCEEDED,
    )


def compare_pair(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    min_observations: int,
    resolver: ThresholdResolver,
) -> list[dict[str, Any]]:
    """All applicable comparison rows for one dataset pair (a < b by dataset_id)."""
    assert a.spec.dataset_id < b.spec.dataset_id, "pair must be in deterministic order"
    definitions = PROFILE_METRICS[a.spec.read_type]
    if a.spec.input_format is InputFormat.BAM:
        definitions = BAM_METRICS
    rows = [
        _compare_one(pool_group, a, b, d, stratum, min_observations, resolver)
        for d in definitions
        for stratum in d.strata
    ]
    # merged_short processing evidence: compare retained fractions when both sides
    # carry fastp reports; report missing evidence otherwise (never silently pass).
    if a.spec.read_type is ReadType.MERGED_SHORT:
        fa = a.fastp.retained_fraction if a.fastp else None
        fb = b.fastp.retained_fraction if b.fastp else None
        if fa is not None and fb is not None:
            value = abs(fa - fb)
            rule = resolver("merge_retained_fraction", STRATUM_SE)
            if rule is None:
                rows.append(
                    _comparison_row(
                        pool_group,
                        a.spec.dataset_id,
                        b.spec.dataset_id,
                        "processing",
                        STRATUM_SE,
                        "merge_retained_fraction",
                        METHOD_ABS_DIFF,
                        value,
                        "fraction",
                        None,
                        None,
                        STATUS_NO_THRESHOLD,
                        1,
                        1,
                        REASON_NO_THRESHOLD,
                    )
                )
            else:
                rows.append(
                    _comparison_row(
                        pool_group,
                        a.spec.dataset_id,
                        b.spec.dataset_id,
                        "processing",
                        STRATUM_SE,
                        "merge_retained_fraction",
                        METHOD_ABS_DIFF,
                        value,
                        "fraction",
                        rule.value,
                        "max",
                        STATUS_PASS if value <= rule.value else STATUS_EXCEED,
                        1,
                        1,
                        REASON_BELOW_THRESHOLD if value <= rule.value else REASON_EXCEEDED,
                    )
                )
        else:
            rows.append(
                _comparison_row(
                    pool_group,
                    a.spec.dataset_id,
                    b.spec.dataset_id,
                    "processing",
                    STRATUM_SE,
                    "merge_retained_fraction",
                    METHOD_ABS_DIFF,
                    None,
                    "fraction",
                    None,
                    None,
                    STATUS_MISSING_EVIDENCE,
                    0,
                    0,
                    REASON_FASTP_MISSING,
                )
            )
    return rows


def generate_pairs(dataset_ids: list[str]) -> list[tuple[str, str]]:
    """Deterministic all-pairs list: ``itertools.combinations`` over sorted ids."""
    return list(itertools.combinations(sorted(dataset_ids), 2))


def count_total_pairs(groups: list[Sequence[Any]]) -> int:
    """Total unordered pairs across groups (each entry is one group's members)."""
    return sum(math.comb(len(members), 2) for members in groups)


def compare_group(
    pool_group: str,
    outcomes: list[DatasetOutcome],
    min_observations: int,
    resolver: ThresholdResolver,
) -> list[dict[str, Any]]:
    """Compare every dataset pair in one pooling group (spec §7)."""
    by_id = {o.spec.dataset_id: o for o in outcomes}
    rows: list[dict[str, Any]] = []
    for id_a, id_b in generate_pairs(list(by_id)):
        rows.extend(compare_pair(pool_group, by_id[id_a], by_id[id_b], min_observations, resolver))
    return rows
