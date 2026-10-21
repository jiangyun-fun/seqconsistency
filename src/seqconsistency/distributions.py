"""Deterministic histogram bin definitions shared across datasets.

All histograms compared between datasets must use the same edges, defined here once.
Bin convention: bin ``i`` covers ``[edges[i], edges[i+1])``; the last bin is closed on
the right (``[edges[-2], edges[-1]]``). Values at or above ``edges[-1]`` are overflow;
values below ``edges[0]`` are underflow. Both are counted, never dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from seqconsistency.models import MAX_PHRED, Histogram

#: Per-read GC fraction: 50 bins of width 0.02 over [0, 1].
GC_EDGES = [round(0.02 * i, 4) for i in range(51)]

#: Per-read N fraction: 20 bins of width 0.05 over [0, 1].
N_EDGES = [round(0.05 * i, 4) for i in range(21)]

#: Base-quality histogram: one bin per Phred value 0..93 (Phred+33 bytes 33..126).
PHRED_EDGES = [float(q) for q in range(MAX_PHRED + 2)]

#: Nucleotide composition is categorical (A, C, G, T, N) — no numeric edges.
NUC_CATEGORIES = ["A", "C", "G", "T", "N"]

#: Short-read length histogram: unit bins 0..1024, overflow above.
SHORT_LENGTH_EDGES = [float(i) for i in range(1026)]

#: Long-read length histogram: deterministic log2-ish ladder up to 1 Mb, overflow above.
LONG_LENGTH_EDGES = [
    0.0,
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
    25600.0,
    51200.0,
    102400.0,
    204800.0,
    409600.0,
    819200.0,
    1048576.0,
]

#: Mapping-quality histogram: one bin per MAPQ 0..255.
MAPQ_EDGES = [float(q) for q in range(257)]

#: Insert-size histogram (absolute template length): bins of 10 bp to 20 kb, overflow.
INSERT_EDGES = [float(i) for i in range(0, 20001, 10)]

#: Per-read mean quality histogram: bins of 1 Phred over [0, 94].
MEANQ_EDGES = [float(q) for q in range(95)]

#: Joint read-length x mean-quality summary (spec §6.1), computed on the sample.
#: Length: 16 bins (short profiles: 64 bp bins to 1024; long: log2-style ladder).
#: Quality: 13 bins of 8 Phred over [0, 104]. Same bin conventions as elsewhere:
#: bin i is [start, end), final bin closed right, overflow excluded (rows carry
#: observation counts so exclusions are visible).
JOINT_SHORT_LEN_EDGES = [float(i) for i in range(0, 1025, 64)]
JOINT_LONG_LEN_EDGES = [
    0.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12800.0,
    25600.0,
    51200.0,
    102400.0,
    204800.0,
    409600.0,
    819200.0,
    1048576.0,
    2097152.0,
]
JOINT_QUAL_EDGES = [float(q) for q in range(0, 105, 8)]


def joint_bin_counts(
    lengths: np.ndarray | Sequence[float],
    qualities: np.ndarray | Sequence[float],
    length_edges: list[float],
) -> tuple[np.ndarray, int]:
    """2-D counts (n_len_bins, n_qual_bins) over finite (length, quality) pairs.

    Returns (counts, n_finite_pairs); non-finite values on either axis are
    excluded pairwise.
    """
    la = np.asarray(lengths, dtype=np.float64)
    qa = np.asarray(qualities, dtype=np.float64)
    ok = np.isfinite(la) & np.isfinite(qa)
    n_len = len(length_edges) - 1
    n_qual = len(JOINT_QUAL_EDGES) - 1
    counts = np.zeros((n_len, n_qual), dtype=np.int64)
    if not ok.any():
        return counts, 0
    li = np.searchsorted(length_edges, la[ok], side="right") - 1
    li[la[ok] == length_edges[-1]] = n_len - 1  # final bin closed right
    qi = np.searchsorted(JOINT_QUAL_EDGES, qa[ok], side="right") - 1
    qi[qa[ok] == JOINT_QUAL_EDGES[-1]] = n_qual - 1
    in_range = (li >= 0) & (li < n_len) & (qi >= 0) & (qi < n_qual)
    flat = np.bincount(li[in_range] * n_qual + qi[in_range], minlength=n_len * n_qual)
    return flat.reshape(n_len, n_qual), int(ok.sum())


def bin_values(values: np.ndarray, edges: list[float]) -> tuple[list[int], int, int]:
    """Histogram ``values`` into ``edges``; return (counts, overflow, underflow).

    Convention: last bin is closed on the right, so ``v == edges[-1]`` falls in the
    last bin while ``v > edges[-1]`` is overflow. Non-finite values count as
    underflow, keeping ``sum(counts) + overflow + underflow == len(values)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0] * (len(edges) - 1), 0, 0
    finite = arr[np.isfinite(arr)]
    nan_count = int(arr.size - finite.size)
    overflow_mask = finite > edges[-1]
    idx = np.searchsorted(edges, finite, side="right") - 1
    idx[finite == edges[-1]] = len(edges) - 2  # last bin closed on the right
    in_range = (idx >= 0) & (~overflow_mask)
    counts = np.bincount(idx[in_range], minlength=len(edges) - 1)
    overflow = int(overflow_mask.sum())
    underflow = int((~in_range & ~overflow_mask).sum()) + nan_count
    return counts.tolist(), overflow, underflow


def make_histogram(name: str, values: np.ndarray | list[float], edges: list[float]) -> Histogram:
    counts, overflow, underflow = bin_values(np.asarray(values, dtype=np.float64), edges)
    n = int(np.asarray(values).size)
    return Histogram(
        name=name,
        edges=edges,
        counts=counts,
        overflow=overflow,
        underflow=underflow,
        observation_count=n,
    )


class HistogramAccumulator:
    """Incrementally folds value chunks into one Histogram in bounded memory.

    Callers push bounded chunks (e.g. per-scan-block lists); counts live in the
    fixed-size bin array, never in a per-observation list.
    """

    def __init__(self, name: str, edges: list[float]) -> None:
        self.name = name
        self.edges = edges
        self._counts = np.zeros(len(edges) - 1, dtype=np.int64)
        self._overflow = 0
        self._underflow = 0
        self._n = 0

    def add_values(self, values: np.ndarray | Sequence[float]) -> None:
        if len(values) == 0:
            return
        counts, overflow, underflow = bin_values(np.asarray(values, dtype=np.float64), self.edges)
        self._counts += counts
        self._overflow += overflow
        self._underflow += underflow
        self._n += len(values)

    def result(self) -> Histogram:
        return Histogram(
            name=self.name,
            edges=self.edges,
            counts=self._counts.tolist(),
            overflow=self._overflow,
            underflow=self._underflow,
            observation_count=self._n,
        )
