"""Provenance JSON assembly (spec §10.6).

Avoids expensive full-file cryptographic hashes of sequencing inputs by default —
``--hash-inputs`` opts in deliberately (README § Provenance).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from seqconsistency import __version__
from seqconsistency import distributions as dist
from seqconsistency.manifest import MANIFEST_SCHEMA_VERSION
from seqconsistency.models import DatasetOutcome, ResourceConfig, SamplingConfig
from seqconsistency.parquet_io import SCHEMA_VERSION
from seqconsistency.policy import POLICY_SCHEMA_VERSION, Policy
from seqconsistency.sampling import dataset_seed


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _file_info(path: Path, hash_it: bool) -> dict[str, Any]:
    stat = path.stat()
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": _dt.datetime.fromtimestamp(stat.st_mtime, _dt.UTC).isoformat(timespec="seconds"),
    }
    if hash_it:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def build_provenance(
    cli_args: dict[str, Any],
    manifest: Any,
    policy: Policy,
    outcomes: list[DatasetOutcome],
    sampling: SamplingConfig,
    resources: ResourceConfig,
    scan_seconds_total: float,
    comparison_seconds: float,
    hash_inputs: bool,
) -> dict[str, Any]:
    """Assemble the provenance document written to --provenance-out."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    import numpy
    import pyarrow
    import scipy

    versions.update(
        {
            "seqconsistency": __version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
        }
    )
    try:
        import pysam

        versions["pysam"] = pysam.__version__
        versions["htslib"] = getattr(getattr(pysam, "version", None), "htslib", "unknown")
    except Exception:  # pragma: no cover - only when pysam metadata misbehaves
        versions["pysam"] = "unavailable"

    inputs: list[dict[str, Any]] = []
    for o in outcomes:
        entry: dict[str, Any] = {
            "dataset_id": o.spec.dataset_id,
            "pool_group": o.spec.pool_group,
            "files": [_file_info(o.spec.path1, hash_inputs)],
        }
        if o.spec.path2 is not None:
            entry["files"].append(_file_info(o.spec.path2, hash_inputs))
        if o.spec.fastp_json is not None:
            entry["fastp_json"] = _file_info(o.spec.fastp_json, hash_inputs)
        entry["sampling"] = {
            "seed": dataset_seed(sampling.master_seed, o.spec.dataset_id),
        }
        if o.fastq is not None:
            entry["scan_seconds"] = o.fastq.scan_seconds
            entry["eligible_observations"] = o.fastq.eligible_observations
            entry["sampled_count"] = o.fastq.sampled_count
        if o.bam is not None:
            entry["scan_seconds"] = o.bam.scan_seconds
            entry["eligible_observations"] = o.bam.eligible_observations
            entry["sampled_count"] = o.bam.sampled_count
            entry["references"] = [
                {"name": n, "length": ln, "md5": md} for n, ln, md in o.bam.header.references
            ]
        inputs.append(entry)

    return {
        "seqconsistency_version": __version__,
        "created_utc": _utc_now(),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "dependency_versions": versions,
        "cli_arguments": cli_args,
        "policy": {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "schema_version": POLICY_SCHEMA_VERSION,
            "sha256": policy.sha256,
        },
        "manifest": {
            "path": str(manifest.manifest_path),
            "sha256": manifest.sha256,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "row_count": manifest.row_count,
        },
        "inputs": inputs,
        "sampling": {
            "master_seed": sampling.master_seed,
            "sample_size": sampling.sample_size,
            "algorithm": "uniform_key_reservoir",
            "seed_derivation": "blake2b-128(f'{master_seed}:{dataset_id}')",
            "worker_independent": True,
        },
        "histogram_parameters": {
            "gc_bins": "50 uniform bins over [0,1] (width 0.02)",
            "n_bins": "20 uniform bins over [0,1] (width 0.05)",
            "short_read_length_bins": f"unit bins 0..{int(dist.SHORT_LENGTH_EDGES[-1])}",
            "long_read_length_bins": "log2-style ladder to 1048576 bp",
            "base_quality_bins": "one bin per Phred 0..93",
            "mapq_bins": "one bin per MAPQ 0..255",
            "insert_size_bins": "10 bp bins to 20000 bp",
            "bin_convention": "bin i covers [start, end); final bin closed right; "
            "values beyond the last edge count as overflow",
        },
        "resources": {
            "workers": resources.workers,
            "threads": resources.threads,
            "threads_per_worker": resources.threads_per_worker,
            "main_threads": resources.main_threads,
            "resolution_rule": "workers = min(--workers, --threads); "
            "threads_per_worker = max(1, threads // workers)",
        },
        "timings": {
            "scan_seconds_total": round(scan_seconds_total, 3),
            "comparison_seconds": round(comparison_seconds, 3),
        },
        "output_schema_versions": {
            "metrics": SCHEMA_VERSION,
            "distributions": SCHEMA_VERSION,
            "comparisons": SCHEMA_VERSION,
            "findings": SCHEMA_VERSION,
            "decisions": SCHEMA_VERSION,
        },
    }
