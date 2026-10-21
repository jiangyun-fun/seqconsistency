"""Pairwise dataset comparisons within one pooling group.

Every applicable (metric, stratum) is compared for every dataset pair — pairs are
never silently omitted. Dataset ordering inside a pair is lexicographic by
``dataset_id`` so no duplicate mirrored rows can appear.

Distance conventions come from :mod:`seqconsistency.statistics` (README § Statistics):
Wasserstein-1 on empirical samples, Jensen–Shannon distance (base 2) on histograms
over identical deterministic bins. Pairwise consistency is NOT transitive; the
decision engine never assumes it.

Comparison rows record whether the applied threshold was calibrated
(``threshold_calibrated``): a PASS against an uncalibrated threshold is reported but
does not contribute to automatic approval (spec §7 — "report the metric but do not
use it to automatically approve pooling"); the policy engine implements that rule.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
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
#: Optional evidence that the policy did not request: assessed as nothing, blocks
#: nothing, and is reported for transparency.
STATUS_NOT_ASSESSED = "NOT_ASSESSED"

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
REASON_FASTP_OPTIONAL = "FASTP_EVIDENCE_OPTIONAL"


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


def bam_metrics(read_type: ReadType) -> tuple[MetricDefinition, ...]:
    """Applicable comparisons for a BAM input, selected by the declared profile.

    Insert size exists only for paired-end data (`paired_short`); applying it to
    unpaired or long-read BAMs would demand evidence those layouts cannot produce.
    For `paired_short`, read-level sample metrics compare per mate (R1 vs R1,
    R2 vs R2), mirroring PE FASTQ (spec §6.2): pooling both mates into one READS
    distribution hides anti-correlated mate shifts (one mate better, the other
    worse) as a false match. `mapq` stays on the pooled `READS` stratum — it is a
    full-scan histogram over every mapped primary record, mate-independent.
    """
    read_strata: tuple[str, ...] = (
        (STRATUM_R1, STRATUM_R2) if read_type is ReadType.PAIRED_SHORT else (STRATUM_READS,)
    )
    base = (
        MetricDefinition("read_length", METHOD_WASSERSTEIN, read_strata, "bp"),
        MetricDefinition("mean_read_quality", METHOD_WASSERSTEIN, read_strata, "phred"),
        MetricDefinition("gc_content", METHOD_JSD, read_strata, "jsd_base2"),
        MetricDefinition("n_content", METHOD_JSD, read_strata, "jsd_base2"),
        MetricDefinition("mapq", METHOD_JSD, (STRATUM_READS,), "jsd_base2"),
    )
    if read_type is ReadType.PAIRED_SHORT:
        return (*base, MetricDefinition("insert_size", METHOD_WASSERSTEIN, (STRATUM_PAIR,), "bp"))
    return base


def applicable_definitions(outcome: DatasetOutcome) -> tuple[MetricDefinition, ...]:
    if outcome.spec.input_format is InputFormat.BAM:
        return bam_metrics(outcome.spec.read_type)
    return PROFILE_METRICS[outcome.spec.read_type]


#: Every (metric, method) pair this version can compare. Policy thresholds are
#: validated against this set — a typo must fail at load time, not silently never
#: match (policy.py).
KNOWN_METRIC_METHODS: frozenset[tuple[str, str]] = (
    frozenset((d.metric, d.method) for defs in PROFILE_METRICS.values() for d in defs)
    | {(d.metric, d.method) for rt in ReadType for d in bam_metrics(rt)}
    | {
        ("merge_rate", METHOD_ABS_DIFF),
        ("merge_retained_fraction", METHOD_ABS_DIFF),
    }
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


@dataclass
class _Prepared:
    """Per-dataset reusable comparison inputs, computed once per group.

    Wasserstein features are finite-filtered once; sample-based JSD histograms are
    built once — pairs then reuse them instead of re-converting per partner.
    """

    w: dict[tuple[str, str], np.ndarray]  # (metric, stratum) -> finite sample
    jsd: dict[tuple[str, str], np.ndarray]  # (metric, stratum) -> counts


def _prepare_dataset(
    outcome: DatasetOutcome, definitions: tuple[MetricDefinition, ...]
) -> _Prepared:
    w: dict[tuple[str, str], np.ndarray] = {}
    jsd: dict[tuple[str, str], np.ndarray] = {}
    for d in definitions:
        for stratum in d.strata:
            key = (d.metric, stratum)
            if d.method == METHOD_WASSERSTEIN:
                arr = _raw_wasserstein(outcome, d.metric, stratum)
                w[key] = arr[np.isfinite(arr)]
            elif d.method == METHOD_JSD:
                h = _hist_for(outcome, d.metric, stratum)
                jsd[key] = np.asarray(h.counts if h is not None else [], dtype=np.float64)
    return _Prepared(w=w, jsd=jsd)


def _bam_sample(outcome: DatasetOutcome, stratum: str) -> dict[str, list[float]]:
    """Per-stratum BAM read sample: R1/R2 for paired profiles, pooled READS else
    (and for unpaired records inside paired BAMs)."""
    assert outcome.bam is not None
    if stratum == STRATUM_R1:
        return outcome.bam.sample_r1
    if stratum == STRATUM_R2:
        return outcome.bam.sample_r2
    return outcome.bam.sample_reads


def _raw_wasserstein(outcome: DatasetOutcome, metric: str, stratum: str) -> np.ndarray:
    key = "length" if metric == "read_length" else "mean_quality_arithmetic"
    if outcome.fastq is not None:
        sample = _fastq_sample(outcome, stratum)
        if sample is None:
            return np.zeros(0)
        return np.asarray(sample.get(key, []), dtype=np.float64)
    assert outcome.bam is not None
    if metric == "insert_size":
        return np.asarray(outcome.bam.sample_inserts, dtype=np.float64)
    return np.asarray(_bam_sample(outcome, stratum).get(key, []), dtype=np.float64)


def _hist_for(outcome: DatasetOutcome, metric: str, stratum: str) -> dist.Histogram | None:
    if outcome.fastq is not None:
        st = _fastq_stats(outcome, stratum)
        if metric == "gc_content":
            return st.gc_hist
        if metric == "n_content":
            return st.n_hist
        if metric == "base_quality":
            return dist.Histogram(
                "base_quality",
                dist.PHRED_EDGES,
                list(st.base_qual_counts),
                observation_count=int(sum(st.base_qual_counts)),
            )
        if metric == "nucleotide_composition":
            total = sum(st.base_counts)
            if not total:
                return None
            return dist.Histogram(
                "nucleotide_composition",
                [float(i) for i in range(len(dist.NUC_CATEGORIES) + 1)],
                list(st.base_counts),
                observation_count=total,
            )
        raise AssertionError(f"unknown FASTQ JSD metric {metric}")
    assert outcome.bam is not None
    if metric == "gc_content":
        return dist.make_histogram(
            "per_read_gc", _bam_sample(outcome, stratum)["gc_fraction"], dist.GC_EDGES
        )
    if metric == "n_content":
        return dist.make_histogram(
            "per_read_n", _bam_sample(outcome, stratum)["n_fraction"], dist.N_EDGES
        )
    if metric == "mapq":
        return outcome.bam.mapq_hist
    raise AssertionError(f"unknown BAM JSD metric {metric}")


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
    threshold_calibrated: bool | None = None,
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
        "threshold_calibrated": threshold_calibrated,
        "status": status,
        "observations_a": obs_a,
        "observations_b": obs_b,
        "reason_code": reason_code,
        "method_version": STATS_VERSION,
    }


def _insufficient_row(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    component: str,
    stratum: str,
    definition: MetricDefinition,
    na: int,
    nb: int,
) -> dict[str, Any]:
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


def _compare_one(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    definition: MetricDefinition,
    stratum: str,
    min_observations: int,
    resolver: ThresholdResolver,
    prepared: dict[str, _Prepared],
) -> dict[str, Any]:
    component = "fastq" if a.spec.input_format is InputFormat.FASTQ else "alignment"

    # At least one observation per side is always required: distances on empty
    # samples are undefined, so min_observations=0 still means >= 1.
    effective_min = max(min_observations, 1)
    if definition.method == METHOD_WASSERSTEIN:
        va = prepared[a.spec.dataset_id].w[(definition.metric, stratum)]
        vb = prepared[b.spec.dataset_id].w[(definition.metric, stratum)]
        na, nb = int(va.size), int(vb.size)
        if na < effective_min or nb < effective_min:
            return _insufficient_row(pool_group, a, b, component, stratum, definition, na, nb)
        value = wasserstein_distance(va, vb, definition.metric)
    elif definition.method == METHOD_JSD:
        ha = prepared[a.spec.dataset_id].jsd[(definition.metric, stratum)]
        hb = prepared[b.spec.dataset_id].jsd[(definition.metric, stratum)]
        na, nb = int(ha.sum()), int(hb.sum())
        if na < effective_min or nb < effective_min:
            return _insufficient_row(pool_group, a, b, component, stratum, definition, na, nb)
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
    status = STATUS_PASS if not exceeds else STATUS_EXCEED
    reason = REASON_BELOW_THRESHOLD if not exceeds else REASON_EXCEEDED
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
        status,
        na,
        nb,
        reason,
        threshold_calibrated=rule.calibrated,
    )


def _scalar_processing_row(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    metric: str,
    value: float,
    resolver: ThresholdResolver,
) -> dict[str, Any]:
    rule = resolver(metric, STRATUM_SE)
    if rule is None:
        return _comparison_row(
            pool_group,
            a.spec.dataset_id,
            b.spec.dataset_id,
            "processing",
            STRATUM_SE,
            metric,
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
    exceeds = not (value <= rule.value)
    return _comparison_row(
        pool_group,
        a.spec.dataset_id,
        b.spec.dataset_id,
        "processing",
        STRATUM_SE,
        metric,
        METHOD_ABS_DIFF,
        value,
        "fraction",
        rule.value,
        "max",
        STATUS_PASS if not exceeds else STATUS_EXCEED,
        1,
        1,
        REASON_BELOW_THRESHOLD if not exceeds else REASON_EXCEEDED,
        threshold_calibrated=rule.calibrated,
    )


def _fastp_pair_rows(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    resolver: ThresholdResolver,
    require_fastp: bool,
) -> list[dict[str, Any]]:
    """Processing-evidence comparisons for merged_short pairs.

    ``merge_rate`` (merged reads / before-filtering reads) is compared when both
    reports carry an explicit merged-read count — the quantity that isolates
    merging. Otherwise retained fraction (after/before) is compared when both
    reports carry it; retained fraction conflates filtering and merging, and the
    metric name says so. Missing evidence blocks nothing unless the policy requires
    it (``require_fastp_evidence``).
    """
    fa, fb = a.fastp, b.fastp
    if fa is not None and fb is not None:
        if fa.merge_rate is not None and fb.merge_rate is not None:
            return [
                _scalar_processing_row(
                    pool_group,
                    a,
                    b,
                    "merge_rate",
                    abs(fa.merge_rate - fb.merge_rate),
                    resolver,
                )
            ]
        if fa.retained_fraction is not None and fb.retained_fraction is not None:
            return [
                _scalar_processing_row(
                    pool_group,
                    a,
                    b,
                    "merge_retained_fraction",
                    abs(fa.retained_fraction - fb.retained_fraction),
                    resolver,
                )
            ]
    if require_fastp:
        return [
            _comparison_row(
                pool_group,
                a.spec.dataset_id,
                b.spec.dataset_id,
                "processing",
                STRATUM_SE,
                "merge_rate",
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
        ]
    return [
        _comparison_row(
            pool_group,
            a.spec.dataset_id,
            b.spec.dataset_id,
            "processing",
            STRATUM_SE,
            "merge_rate",
            METHOD_ABS_DIFF,
            None,
            "fraction",
            None,
            None,
            STATUS_NOT_ASSESSED,
            0,
            0,
            REASON_FASTP_OPTIONAL,
        )
    ]


def compare_pair(
    pool_group: str,
    a: DatasetOutcome,
    b: DatasetOutcome,
    min_observations: int,
    resolver: ThresholdResolver,
    require_fastp: bool = False,
    prepared: dict[str, _Prepared] | None = None,
) -> list[dict[str, Any]]:
    """All applicable comparison rows for one dataset pair (a < b by dataset_id)."""
    assert a.spec.dataset_id < b.spec.dataset_id, "pair must be in deterministic order"
    definitions = applicable_definitions(a)
    if prepared is None:
        prepared = {o.spec.dataset_id: _prepare_dataset(o, definitions) for o in (a, b)}
    rows = [
        _compare_one(pool_group, a, b, d, stratum, min_observations, resolver, prepared)
        for d in definitions
        for stratum in d.strata
    ]
    if a.spec.read_type is ReadType.MERGED_SHORT:
        rows.extend(_fastp_pair_rows(pool_group, a, b, resolver, require_fastp))
    return rows


def generate_pairs(dataset_ids: list[str]) -> Iterator[tuple[str, str]]:
    """Lazily yield deterministic all-pairs: ``itertools.combinations`` over sorted ids."""
    yield from itertools.combinations(sorted(dataset_ids), 2)


def count_total_pairs(groups: list[Sequence[Any]]) -> int:
    """Total unordered pairs across groups (each entry is one group's members)."""
    return sum(math.comb(len(members), 2) for members in groups)


def compare_group(
    pool_group: str,
    outcomes: list[DatasetOutcome],
    min_observations: int,
    resolver: ThresholdResolver,
    require_fastp: bool = False,
) -> list[dict[str, Any]]:
    """Compare every dataset pair in one pooling group (spec §7).

    Per-dataset comparison inputs are prepared once and reused across the group's
    pairs. Pairs are generated lazily; row output order is deterministic.
    """
    by_id = {o.spec.dataset_id: o for o in outcomes}
    definitions = applicable_definitions(outcomes[0])
    prepared = {ds_id: _prepare_dataset(o, definitions) for ds_id, o in by_id.items()}
    rows: list[dict[str, Any]] = []
    for id_a, id_b in generate_pairs(list(by_id)):
        rows.extend(
            compare_pair(
                pool_group,
                by_id[id_a],
                by_id[id_b],
                min_observations,
                resolver,
                require_fastp=require_fastp,
                prepared=prepared,
            )
        )
    return rows
