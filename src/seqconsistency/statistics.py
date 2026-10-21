"""Statistical distance calculations with explicit, documented conventions.

- Wasserstein-1 (earth mover's) distance on empirical per-observation samples.
  Units: the unit of the compared feature (bp for lengths, Phred for qualities).
  Two implementations, same quantity:
  ``wasserstein_distance`` delegates to SciPy (the reference implementation);
  ``wasserstein_distance_sorted`` computes the exact same value on inputs that
  are already sorted ascending, without re-sorting them. Comparisons sort each
  dataset's sample once and use the sorted variant, because SciPy re-sorts both
  inputs on every call and per-pair re-sorting dominated comparison cost for
  large groups. Equivalence is locked by test (random sizes, heavy ties,
  identical arrays). This is a documented exact reimplementation, not a silent
  approximation.
- Jensen–Shannon *distance* (square root of the divergence) via
  ``scipy.spatial.distance.jensenshannon`` with logarithm base 2, so the distance is
  bounded to [0, 1] for probability distributions. Inputs are normalized histograms
  over identical, deterministic bins; no pseudocount is added — bins with zero count
  in both distributions contribute zero, and the convention is that both histograms
  must come from the same bin definition (enforced by comparison.py).
- p-values are deliberately not used: with sample sizes typical of sequencing data,
  microscopic differences become "significant". Thresholds, where configured, are on
  distance magnitudes and are never described as statistically calibrated.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance as scipy_wasserstein

from seqconsistency.errors import ConfigError

JSD_BASE = 2
STATS_VERSION = "sorted-merge-w1-jsd-base2-v2"


def wasserstein_distance(a: np.ndarray, b: np.ndarray, feature: str) -> float:
    """Empirical Wasserstein-1 distance between two samples of one feature.

    NaN/Inf entries (undefined per-read values) are excluded pairwise here; callers
    report observation counts after exclusion. Reference implementation (SciPy).
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    va = va[np.isfinite(va)]
    vb = vb[np.isfinite(vb)]
    if va.size == 0 or vb.size == 0:
        raise ConfigError(
            f"cannot compute Wasserstein distance for feature {feature!r}: "
            f"one side has no finite observations ({va.size} vs {vb.size})"
        )
    return float(scipy_wasserstein(va, vb))


def wasserstein_distance_sorted(u: np.ndarray, v: np.ndarray, feature: str) -> float:
    """Exact Wasserstein-1 distance between two **presorted** finite samples.

    Inputs must be ascending and finite-filtered (comparison.py guarantees both).
    Same quantity as ``wasserstein_distance``: W1 = ∫|F_u − F_v| over the merged
    grid, where both empirical CDFs are piecewise constant between grid points.
    No sort happens here — the merged grid comes from binary-search positions,
    so per-pair cost is O((n+m)·log) instead of SciPy's three sorts per call.
    """
    va = np.asarray(u, dtype=np.float64)
    vb = np.asarray(v, dtype=np.float64)
    if va.size == 0 or vb.size == 0:
        raise ConfigError(
            f"cannot compute Wasserstein distance for feature {feature!r}: "
            f"one side has no observations ({va.size} vs {vb.size})"
        )
    # Stable merge of two sorted arrays via binary-search positions: element i
    # of u lands at i + (count of v strictly before it); element j of v lands at
    # j + (count of u at or before it). Ties never collide (left vs right side).
    pos_u = np.arange(va.size) + np.searchsorted(vb, va, side="left")
    pos_v = np.arange(vb.size) + np.searchsorted(va, vb, side="right")
    grid = np.empty(va.size + vb.size, dtype=np.float64)
    grid[pos_u] = va
    grid[pos_v] = vb
    # CDFs just after each grid point (piecewise constant on the segment that
    # starts there); zero-width segments from ties contribute zero.
    cdf_u = np.searchsorted(va, grid, side="right") / va.size
    cdf_v = np.searchsorted(vb, grid, side="right") / vb.size
    return float(np.dot(np.abs(cdf_u[:-1] - cdf_v[:-1]), np.diff(grid)))


def jensen_shannon_distance(hist_a: np.ndarray, hist_b: np.ndarray, metric: str) -> float:
    """Jensen–Shannon distance (base-2) between two count histograms.

    Histograms must share the same bin definition (checked by comparison.py).
    Both-empty or all-zero histograms raise: that is missing evidence, not zero
    distance — it must never be treated as a pass.
    """
    ha = np.asarray(hist_a, dtype=np.float64)
    hb = np.asarray(hist_b, dtype=np.float64)
    if ha.shape != hb.shape:
        raise ConfigError(
            f"JSD bin mismatch for {metric!r}: {ha.shape} vs {hb.shape} "
            "(bins must be identical across datasets)"
        )
    if ha.sum() == 0 or hb.sum() == 0:
        raise ConfigError(
            f"cannot compute JSD for {metric!r}: a histogram has zero observations "
            f"({ha.sum()} vs {hb.sum()} counts)"
        )
    value = jensenshannon(ha, hb, base=JSD_BASE)
    if not np.isfinite(value):
        raise ConfigError(f"JSD produced a non-finite value for {metric!r}")
    return float(value)
