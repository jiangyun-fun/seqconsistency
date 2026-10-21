"""seqconsistency: sequencing dataset consistency and pooling-suitability assessment.

Phase 1 scope: FASTQ (SE/PE, gzip) and coordinate-sorted BAM inputs across four
comparison profiles (``paired_short``, ``merged_short``, ``ont``, ``hifi``), full-file
structural validation, deterministic bounded-memory sampling, Wasserstein and
Jensen–Shannon pairwise comparisons, and a policy-driven decision engine.
"""

from __future__ import annotations

from seqconsistency.errors import (
    ConfigError,
    DependencyError,
    InputDataError,
    ManifestError,
    OutputError,
    PathError,
    SeqConsistencyError,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "DependencyError",
    "InputDataError",
    "ManifestError",
    "OutputError",
    "PathError",
    "SeqConsistencyError",
    "__version__",
]
