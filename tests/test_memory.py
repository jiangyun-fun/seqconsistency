"""Memory boundedness: peak RSS must not scale with read count at fixed sample size.

Runs the scanner as a subprocess on synthetic FASTQ files of increasing size and
compares peak RSS (VmHWM of the child process). The assertion is a bound, not a
timing measurement — no wall-clock assertions here (spec §12).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import make_records, write_fastq

SCRIPT = """
import sys
from pathlib import Path
from seqconsistency.fastq import scan_fastq_single
from seqconsistency.models import ReadType, SamplingConfig
path = Path(sys.argv[1])
n = int(sys.argv[2])
stats, sample, eligible, sampled = scan_fastq_single(
    "ds", path, ReadType.ONT, SamplingConfig(sample_size=2000, master_seed=1), 2000,
)
assert eligible == n, (eligible, n)
assert sampled == 2000
# VmHWM (KiB) rather than ru_maxrss: on Linux ru_maxrss is inherited across
# fork+exec, so a child forked from the fixture-laden test parent would report
# the parent's high-water mark. VmHWM resets at execve and measures this process.
with open("/proc/self/status") as fh:
    for line in fh:
        if line.startswith("VmHWM"):
            print(int(line.split()[1]))
            break
"""


def _peak_rss_mb(tmp_path: Path, n_reads: int, length: int = 400) -> float:
    rng = np.random.default_rng(n_reads)
    lengths = rng.integers(length // 2, length * 2, size=n_reads)
    p = write_fastq(
        tmp_path / f"ont_{n_reads}.fastq",
        make_records(n_reads, lengths=lengths, seed=n_reads % 1000),
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(p), str(n_reads)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip()) / 1024.0  # KiB -> MiB


@pytest.mark.slow
def test_long_read_memory_bounded_by_sample_size(tmp_path: Path):
    # Both cases exceed several full scan batches, so per-batch transients are
    # saturated; only truly unbounded state (whole-file retention, per-read lists)
    # could make peak RSS grow with read count.
    small = _peak_rss_mb(tmp_path, 60_000)
    large = _peak_rss_mb(tmp_path, 600_000)
    assert abs(large - small) < 80.0, f"peak RSS grew {small:.1f} -> {large:.1f} MiB"
    assert large < 700.0


@pytest.mark.slow
def test_short_read_memory_bounded(tmp_path: Path):
    small = _peak_rss_mb(tmp_path, 200_000, length=150)
    large = _peak_rss_mb(tmp_path, 1_500_000, length=150)
    assert abs(large - small) < 80.0, f"peak RSS grew {small:.1f} -> {large:.1f} MiB"
    assert large < 700.0
