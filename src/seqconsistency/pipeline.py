"""End-to-end pipeline: scan datasets, compare, decide, write outputs.

The CLI is a thin wrapper around :func:`run_check`. Computation (scans,
comparisons), decisions (policy), and serialization (parquet_io) are separated.
Worker scheduling never affects results: each dataset is scanned by exactly one
worker with its own deterministic RNG stream, and comparisons run sequentially in
the main process over precomputed summaries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seqconsistency import metrics as metrics_mod
from seqconsistency import parquet_io as pio
from seqconsistency.bam import scan_bam
from seqconsistency.comparison import compare_group, count_total_pairs
from seqconsistency.errors import ConfigError, PathError, SeqConsistencyError
from seqconsistency.fastq import parse_fastp_json, scan_fastq_paired, scan_fastq_single
from seqconsistency.manifest import ManifestLoadResult, load_manifest
from seqconsistency.models import (
    DatasetOutcome,
    DatasetSpec,
    FastqScanResult,
    InputFormat,
    ResourceConfig,
    SamplingConfig,
    Severity,
)
from seqconsistency.policy import (
    Finding,
    Policy,
    assign_finding_ids,
    evaluate_group,
    load_policy,
)
from seqconsistency.reporting import build_provenance

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_NOT_COMPATIBLE = 1
EXIT_FAILURE = 2


@dataclass
class CheckConfig:
    manifest: Path
    policy: Path
    metrics_out: Path
    distributions_out: Path
    comparisons_out: Path
    findings_out: Path
    decisions_out: Path
    provenance_out: Path
    report_out: Path | None
    workers: int
    threads: int
    sample_size: int
    seed: int
    max_pairs: int
    overwrite: bool
    hash_inputs: bool
    debug: bool = False
    cli_args: dict[str, Any] = field(default_factory=dict)


def validate_output_paths(config: CheckConfig, input_paths: list[Path]) -> None:
    """Reject duplicate outputs, input collisions, existing files, bad parents."""
    outputs = [
        config.metrics_out,
        config.distributions_out,
        config.comparisons_out,
        config.findings_out,
        config.decisions_out,
        config.provenance_out,
    ]
    if config.report_out is not None:
        outputs.append(config.report_out)
    seen: dict[str, str] = {}
    for out in outputs:
        rp = os.path.realpath(out)
        if rp in seen:
            raise PathError(f"duplicate output path: {out} and {seen[rp]} resolve to the same file")
        seen[rp] = str(out)
        if out.exists() and not config.overwrite:
            raise PathError(f"output file exists: {out} (pass --overwrite to replace it)")
        if not out.parent.is_dir():
            raise PathError(
                f"output parent directory does not exist: {out.parent} "
                "(seqconsistency does not create directories implicitly)"
            )
        if not os.access(out.parent, os.W_OK):
            raise PathError(f"output parent directory is not writable: {out.parent}")
    input_real = {os.path.realpath(p) for p in input_paths}
    for out in outputs:
        if os.path.realpath(out) in input_real:
            raise PathError(
                f"output path {out} collides with an input file — seqconsistency "
                "never overwrites input data"
            )


def _scan_one(spec: DatasetSpec, sampling: SamplingConfig, sample_size: int) -> DatasetOutcome:
    """Worker entry point: scan one dataset. Any failure propagates."""
    fastq: FastqScanResult | None = None
    if spec.input_format is InputFormat.BAM:
        bam = scan_bam(spec.dataset_id, spec.path1, spec.read_type, sampling, sample_size)
    else:
        bam = None
        if spec.is_paired_fastq:
            r1, r2, pairing, s1, s2, pairs, sampled = scan_fastq_paired(
                spec.dataset_id,
                spec.path1,
                spec.path2,
                spec.read_type,
                sampling,
                sample_size,
            )
            fastq = FastqScanResult(
                dataset_id=spec.dataset_id,
                r1=r1,
                r2=r2,
                pairing=pairing,
                sample_r1=s1,
                sample_r2=s2,
                eligible_observations=pairs,
                sampled_count=sampled,
                scan_seconds=0.0,
                bytes_read=0,
            )
        else:
            stats, sample, eligible, sampled = scan_fastq_single(
                spec.dataset_id, spec.path1, spec.read_type, sampling, sample_size
            )
            fastq = FastqScanResult(
                dataset_id=spec.dataset_id,
                r1=stats,
                r2=None,
                pairing=None,
                sample_r1=sample,
                sample_r2=None,
                eligible_observations=eligible,
                sampled_count=sampled,
                scan_seconds=0.0,
                bytes_read=spec.path1.stat().st_size,
            )
    fastp = parse_fastp_json(spec.fastp_json) if spec.fastp_json else None
    return DatasetOutcome(spec=spec, fastq=fastq, bam=bam, fastp=fastp)


def _scan_one_timed(
    spec: DatasetSpec, sampling: SamplingConfig, sample_size: int
) -> DatasetOutcome:
    started = time.perf_counter()
    outcome = _scan_one(spec, sampling, sample_size)
    elapsed = time.perf_counter() - started
    if outcome.fastq is not None:
        outcome.fastq.scan_seconds = elapsed
        if outcome.fastq.bytes_read == 0 and spec.is_paired_fastq and spec.path2:
            outcome.fastq.bytes_read = spec.path1.stat().st_size + spec.path2.stat().st_size
    elif outcome.bam is not None:
        outcome.bam.scan_seconds = elapsed
    return outcome


def _worker_init(threads_per_worker: int) -> None:
    # Cap native math-library threads per worker to avoid CPU oversubscription
    # (documented resource-sharing model in the README).
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads_per_worker))
    import pyarrow

    pyarrow.set_cpu_count(threads_per_worker)


def scan_datasets(
    specs: list[DatasetSpec],
    sampling: SamplingConfig,
    sample_size: int,
    resources: ResourceConfig,
) -> list[DatasetOutcome]:
    """Scan all datasets; worker exceptions propagate, pending futures cancel."""
    if resources.workers <= 1 or len(specs) == 1:
        _worker_init(resources.threads_per_worker)
        return [_scan_one_timed(s, sampling, sample_size) for s in specs]
    outcomes: list[DatasetOutcome] = []
    queue = list(specs)
    pending: dict[Any, DatasetSpec] = {}
    with ProcessPoolExecutor(
        max_workers=resources.workers,
        initializer=_worker_init,
        initargs=(resources.threads_per_worker,),
    ) as executor:
        while queue or pending:
            while queue and len(pending) < resources.workers * 2:
                spec = queue.pop(0)
                pending[executor.submit(_scan_one_timed, spec, sampling, sample_size)] = spec
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                spec = pending.pop(fut)
                try:
                    outcomes.append(fut.result())
                except Exception as exc:
                    for f in pending:
                        f.cancel()
                    if not isinstance(exc, SeqConsistencyError):
                        raise SeqConsistencyError(
                            f"unexpected failure while scanning dataset "
                            f"{spec.dataset_id!r} ({spec.path1}): "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    raise
    order = {s.dataset_id: i for i, s in enumerate(specs)}
    outcomes.sort(key=lambda o: order[o.spec.dataset_id])
    return outcomes


def run_check(config: CheckConfig) -> int:
    """Execute the full check; returns the process exit code (0/1)."""
    started = time.perf_counter()
    manifest: ManifestLoadResult = load_manifest(config.manifest)
    policy: Policy = load_policy(config.policy)
    input_paths = [p for spec in manifest.specs for p in (spec.path1, spec.path2) if p is not None]
    validate_output_paths(config, input_paths)

    groups: dict[str, list[DatasetSpec]] = {}
    for spec in manifest.specs:
        groups.setdefault(spec.pool_group, []).append(spec)
    total_pairs = count_total_pairs([list(g) for g in groups.values()])
    if total_pairs > config.max_pairs:
        raise ConfigError(
            f"all-pairs comparison would produce {total_pairs} dataset pairs, "
            f"exceeding --max-pairs {config.max_pairs}; refusing before expensive "
            "processing (raise --max-pairs or split the groups)"
        )

    sampling = SamplingConfig(sample_size=config.sample_size, master_seed=config.seed)
    resources = ResourceConfig.resolve(config.workers, config.threads)
    if resources.workers != config.workers:
        logger.warning(
            "--workers %d exceeds --threads %d; using %d workers (documented resolution rule)",
            config.workers,
            config.threads,
            resources.workers,
        )

    outcomes = scan_datasets(manifest.specs, sampling, config.sample_size, resources)
    scan_seconds = time.perf_counter() - started

    comparison_started = time.perf_counter()
    all_comparison_rows: list[dict[str, Any]] = []
    decisions_rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    for group in sorted(groups):
        members = [o for o in outcomes if o.spec.pool_group == group]
        resolver = policy.resolver_for(members[0].spec.read_type.value, members[0].spec.assay)
        rows = compare_group(group, members, policy.min_observations, resolver)
        all_comparison_rows.extend(rows)
        decision, group_findings = evaluate_group(members, rows, policy)
        decisions_rows.append(decision.to_row(policy))
        findings.extend(group_findings)
    for path, ga, gb in manifest.cross_group_file_reuse:
        findings.append(
            Finding(
                finding_id="",
                pool_group=ga,
                dataset_id=None,
                affected_datasets=[ga, gb],
                component="metadata",
                severity=Severity.INFO,
                reason_code="FILE_REUSED_ACROSS_GROUPS",
                message=(f"Input file {path} is used in pooling groups {ga!r} and {gb!r}."),
                evidence_metric=None,
                evidence_value=None,
                recommended_action=("Confirm this reuse is intentional (e.g. a common control)."),
            )
        )
    findings = assign_finding_ids(findings)
    comparison_seconds = time.perf_counter() - comparison_started

    # Serialize outputs: staged writes, published only after all computation
    # succeeded. Multi-path publication is not transactional (README § Outputs).
    with pio.output_transaction(
        [
            config.metrics_out,
            config.distributions_out,
            config.comparisons_out,
            config.findings_out,
            config.decisions_out,
            config.provenance_out,
            *([config.report_out] if config.report_out else []),
        ]
    ) as published:
        metrics_rows: list[dict[str, Any]] = []
        for outcome in outcomes:
            metrics_rows.extend(metrics_mod.metric_rows(outcome))
        pio.write_table(config.metrics_out, pio.METRICS_SCHEMA, metrics_rows)
        published.append(config.metrics_out)

        dist_writer = pio.StagedWriter(config.distributions_out, pio.DISTRIBUTIONS_SCHEMA)
        try:
            dist_writer.write_rows(
                row for o in outcomes for row in metrics_mod.distribution_rows(o)
            )
            dist_writer.close_publish()
        except BaseException:
            dist_writer.abort()
            raise
        published.append(config.distributions_out)

        pio.write_table(config.comparisons_out, pio.COMPARISONS_SCHEMA, all_comparison_rows)
        published.append(config.comparisons_out)
        pio.write_table(config.findings_out, pio.FINDINGS_SCHEMA, [f.to_row() for f in findings])
        published.append(config.findings_out)
        pio.write_table(config.decisions_out, pio.DECISIONS_SCHEMA, decisions_rows)
        published.append(config.decisions_out)

        provenance = build_provenance(
            cli_args=config.cli_args,
            manifest=manifest,
            policy=policy,
            outcomes=outcomes,
            sampling=sampling,
            resources=resources,
            scan_seconds_total=scan_seconds,
            comparison_seconds=comparison_seconds,
            hash_inputs=config.hash_inputs,
        )
        _write_text_atomic(
            config.provenance_out,
            json.dumps(provenance, indent=2, sort_keys=True),
        )
        published.append(config.provenance_out)

        if config.report_out is not None:
            from seqconsistency.html_report import render_html_report

            html = render_html_report(
                decisions_rows,
                findings,
                all_comparison_rows,
                outcomes,
                policy,
                provenance,
            )
            _write_text_atomic(config.report_out, html)
            published.append(config.report_out)

    any_not_compatible = any(d["overall_decision"] != "COMPATIBLE" for d in decisions_rows)
    return EXIT_NOT_COMPATIBLE if any_not_compatible else EXIT_OK


def _write_text_atomic(path: Path, text: str) -> None:
    staged = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(staged, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(staged, path)
    except OSError:
        staged.unlink(missing_ok=True)
        raise
