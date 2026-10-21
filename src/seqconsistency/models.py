"""Typed domain models and configuration for seqconsistency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ReadType(StrEnum):
    """Comparison profile declared per dataset in the manifest."""

    PAIRED_SHORT = "paired_short"
    MERGED_SHORT = "merged_short"
    ONT = "ont"
    HIFI = "hifi"


class InputFormat(StrEnum):
    FASTQ = "fastq"
    BAM = "bam"


class SamplingMode(StrEnum):
    """How a metric value was computed.

    ``FULL_SCAN``: every eligible observation in the file contributed.
    ``SAMPLE``: a deterministic reservoir sample contributed; results are estimates.
    """

    FULL_SCAN = "full_scan"
    SAMPLE = "sample"


class Availability(StrEnum):
    """Why a metric value is present or absent."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    MISSING_DATA = "missing_data"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    INCOMPATIBLE = "incompatible"


class CheckStatus(StrEnum):
    """Status of one decision component or check.

    Distinct meanings, never collapsed:
    - PASS: check ran to completion, no issue detected.
    - REVIEW: check ran, an actionable warning was raised.
    - FAIL: check ran, a hard incompatibility was detected.
    - MISSING_EVIDENCE: required evidence is absent (unrunnable check).
    - INSUFFICIENT_OBSERVATIONS: check ran but with too few observations.
    - NOT_REQUESTED: optional check the policy did not request.
    - UNSUPPORTED: check this version cannot perform.
    """

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    NOT_REQUESTED = "NOT_REQUESTED"
    UNSUPPORTED = "UNSUPPORTED"


class DecisionState(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    REVIEW = "REVIEW"
    INCOMPATIBLE = "INCOMPATIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_ASSESSED = "NOT_ASSESSED"


class IdentityEvidenceLevel(StrEnum):
    """What the identity conclusion rests on. Never described as *verified* unless an
    independent method (Phase 3) produced it."""

    DECLARED_ONLY = "declared_not_verified"
    CONFLICTING = "conflicting_declarations"
    INDEPENDENTLY_VERIFIED = "independently_verified"


SHORT_PROFILES = (ReadType.PAIRED_SHORT, ReadType.MERGED_SHORT)
LONG_PROFILES = (ReadType.ONT, ReadType.HIFI)

#: Strata used in metrics/distributions/comparisons tables. R1/R2 exist only for
#: paired_short FASTQ; SE covers single-end FASTQ (merged_short, ont, hifi); READS
#: covers BAM eligible primary reads; PAIR covers BAM insert-size observations.
STRATUM_R1 = "R1"
STRATUM_R2 = "R2"
STRATUM_SE = "SE"
STRATUM_READS = "READS"
STRATUM_PAIR = "PAIR"

#: Per-read scalar features kept in the reservoir sample (columns, in order).
FEATURE_LENGTH = "length"
FEATURE_GC = "gc_fraction"
FEATURE_N = "n_fraction"
FEATURE_MEANQ_ARITH = "mean_quality_arithmetic"
FEATURE_MEANQ_PROB = "mean_quality_error_based"
FEATURE_NAMES = (
    FEATURE_LENGTH,
    FEATURE_GC,
    FEATURE_N,
    FEATURE_MEANQ_ARITH,
    FEATURE_MEANQ_PROB,
)

#: Maximum Phred quality under Phred+33 with ASCII 33..126.
MAX_PHRED = 93


@dataclass(frozen=True)
class DatasetSpec:
    """One logical dataset row from the manifest, after validation."""

    dataset_id: str
    pool_group: str
    sample_id: str
    library_id: str
    assay: str
    platform: str
    read_type: ReadType
    input_format: InputFormat
    path1: Path
    path2: Path | None
    processing_id: str | None
    reference_id: str | None
    fastp_json: Path | None

    @property
    def is_paired_fastq(self) -> bool:
        return self.input_format is InputFormat.FASTQ and self.read_type is ReadType.PAIRED_SHORT


@dataclass(frozen=True)
class SamplingConfig:
    """Deterministic sampling controls."""

    sample_size: int
    master_seed: int

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")
        if self.master_seed < 0:
            raise ValueError(f"master_seed must be >= 0, got {self.master_seed}")


@dataclass(frozen=True)
class ResourceConfig:
    """Resource controls, already resolved to a consistent state.

    Resolution rule (documented in README): effective workers = min(--workers,
    --threads); threads_per_worker = max(1, threads // effective_workers); the main
    process keeps the remainder for comparisons and Parquet writing.
    """

    workers: int
    threads: int
    threads_per_worker: int
    main_threads: int

    @classmethod
    def resolve(cls, workers: int, threads: int) -> ResourceConfig:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        eff_workers = min(workers, threads)
        per = max(1, threads // eff_workers)
        main = max(1, threads - per * (eff_workers - 1))
        return cls(
            workers=eff_workers,
            threads=threads,
            threads_per_worker=per,
            main_threads=main,
        )


@dataclass
class Histogram:
    """Counts over deterministic, shared bins.

    Bin convention (README): bin ``i`` covers ``[bin_start, bin_end)``; the final bin
    is closed on the right (``[bin_start, bin_end]``). ``overflow`` counts observations
    at or above the last edge; ``underflow`` below the first edge.
    """

    name: str
    edges: list[float]
    counts: list[int]
    overflow: int = 0
    underflow: int = 0
    observation_count: int = 0

    def frequencies(self) -> list[float]:
        total = self.observation_count
        if total == 0:
            return [0.0] * len(self.counts)
        return [c / total for c in self.counts]


@dataclass
class PositionalProfile:
    """Per-cycle profile for short reads.

    Positions are absolute cycles 0..max_cycles-1. ``coverage[i]`` counts reads
    covering cycle ``i``; ``qual_sum[i]`` sums their base qualities at that cycle;
    ``base_counts[i]`` is a 5-vector (A, C, G, T, N). Bases beyond ``max_cycles``
    land in ``overflow_bases`` and are documented as unprofiled.
    """

    name: str
    max_cycles: int
    coverage: list[int]
    qual_sum: list[int]
    base_counts: list[list[int]]
    overflow_bases: int = 0


@dataclass
class FastqFileStats:
    """Full-file counters and histograms for one FASTQ strand (R1, R2, or SE)."""

    total_reads: int = 0
    total_bases: int = 0
    total_n_bases: int = 0
    base_counts: list[int] = field(default_factory=lambda: [0] * 5)  # A,C,G,T,N
    base_qual_counts: list[int] = field(default_factory=lambda: [0] * (MAX_PHRED + 1))
    q20_bases: int = 0
    q30_bases: int = 0
    sum_read_mean_quality: float = 0.0
    sum_read_quality_error_based: float = 0.0
    reads_with_mean_quality: int = 0
    sum_gc_fraction: float = 0.0
    reads_with_gc: int = 0
    min_qual_byte: int = 127
    max_qual_byte: int = 0
    sum_lengths: int = 0
    sum_sq_lengths: int = 0
    gc_hist: Histogram | None = None
    n_hist: Histogram | None = None
    length_hist: Histogram | None = None
    positional: PositionalProfile | None = None


@dataclass
class PairingStats:
    """Mate-pair synchronization results for PE FASTQ."""

    pairs_checked: int = 0
    mismatched_names: int = 0
    contradictory_annotations: int = 0


@dataclass
class FastqScanResult:
    """Everything one dataset scan produced (FASTQ input)."""

    dataset_id: str
    r1: FastqFileStats
    r2: FastqFileStats | None
    pairing: PairingStats | None
    sample_r1: dict[str, list[float]]
    sample_r2: dict[str, list[float]] | None
    eligible_observations: int
    sampled_count: int
    scan_seconds: float
    bytes_read: int


@dataclass
class BamFlagCounts:
    total_records: int = 0
    primary: int = 0
    secondary: int = 0
    supplementary: int = 0
    unmapped_primary: int = 0
    mapped_primary: int = 0
    duplicate_marked_primary: int = 0
    qcfail_primary: int = 0
    paired_primary: int = 0
    properly_paired_primary: int = 0
    first_in_pair_primary: int = 0
    read1_mapped: int = 0
    records_missing_qualities: int = 0
    records_missing_sequence: int = 0
    clipped_primary_records: int = 0
    clipped_bases: int = 0
    records_with_undefined_rg: int = 0


@dataclass
class BamHeaderInfo:
    sort_order: str
    references: list[tuple[str, int, str | None]]  # (name, length, md5-or-None)
    read_groups: list[dict[str, str]]


@dataclass
class BamScanResult:
    dataset_id: str
    flags: BamFlagCounts
    header: BamHeaderInfo
    eligible_observations: int  # eligible primary records offered to the sampler
    total_bases: int  # sum of query lengths over eligible primary records
    mapq_hist: Histogram | None
    insert_hist: Histogram | None
    length_hist: Histogram | None
    sample_reads: dict[str, list[float]]
    sampled_count: int
    scan_seconds: float
    records_read: int
    length_sum: int = 0
    length_sqsum: int = 0
    length_n: int = 0
    insert_sum: int = 0
    insert_sqsum: int = 0
    insert_n: int = 0
    sample_inserts: list[float] = field(default_factory=list)


@dataclass
class DatasetOutcome:
    """One dataset's scan outcome, unified across input formats (pipeline-level)."""

    spec: DatasetSpec
    fastq: FastqScanResult | None = None
    bam: BamScanResult | None = None
    fastp: FastpEvidence | None = None


@dataclass
class FastpEvidence:
    """Fields extracted from an optional fastp JSON report."""

    source: Path
    before_total_reads: int | None
    after_total_reads: int | None
    before_q30_rate: float | None
    after_q30_rate: float | None
    merged_reads: int | None
    retained_fraction: float | None

    @property
    def merge_rate(self) -> float | None:
        if self.merged_reads is not None and self.before_total_reads:
            return self.merged_reads / self.before_total_reads
        return None
