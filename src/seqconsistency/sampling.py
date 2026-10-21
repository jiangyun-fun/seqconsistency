"""Deterministic, bounded-memory reservoir sampling.

Algorithm: every observation receives an independent uniform key (drawn in
per-observation order from a per-dataset PCG64 stream seeded by a stable hash of the
master seed and dataset id); the retained sample is the set of observations with the
k smallest keys. For i.i.d. keys this is exactly a uniform sample without replacement
(Vitter 1985 key-based reservoir), and each observation's key depends only on its
position in the stream — never on worker scheduling or chunk boundaries beyond the
fixed internal block size.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from seqconsistency.errors import InputDataError
from seqconsistency.models import FEATURE_NAMES


def dataset_seed(master_seed: int, dataset_id: str) -> int:
    """Stable per-dataset seed from the master seed and dataset id.

    Uses blake2b — deterministic across processes and platforms. Python's builtin
    ``hash()`` is randomized per process and is never used.
    """
    digest = hashlib.blake2b(f"{master_seed}:{dataset_id}".encode(), digest_size=16).digest()
    return int.from_bytes(digest, "big")


class FeatureReservoir:
    """Key-based reservoir over per-observation feature rows.

    Memory is bounded by ``capacity * n_features`` floats plus one block of pending
    keys — never by the total observation count. Long-read sequences are never stored;
    callers pass only compact numeric features.
    """

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        self.capacity = capacity
        self._rng = rng
        self._keys: np.ndarray | None = None  # shape (k,)
        self._rows: np.ndarray | None = None  # shape (k, n_features)
        self._threshold: float = math.inf  # current k-th smallest key
        self.offered = 0

    def add_block(self, rows: np.ndarray, keys: np.ndarray | None = None) -> None:
        """Offer one block of feature rows (n, n_features) to the reservoir.

        ``keys`` lets a caller share one key stream across several reservoirs (paired
        R1/R2 must sample the same mate pairs); when omitted, this reservoir's own
        generator draws the keys.
        """
        if rows.ndim != 2:
            raise InputDataError(f"reservoir block must be 2-D, got shape {rows.shape}")
        n = rows.shape[0]
        if n == 0:
            return
        self.offered += n
        key_arr: np.ndarray
        if keys is None:
            if self._rng is None:
                raise ValueError("reservoir has no rng and no keys were supplied")
            key_arr = np.asarray(self._rng.random(n), dtype=np.float64)
        else:
            key_arr = np.asarray(keys, dtype=np.float64)
        if self.capacity == 0:
            return
        row_arr = rows.astype(np.float64, copy=False)
        if self._keys is None:
            self._keys = key_arr
            self._rows = row_arr.copy()
        else:
            cur_keys = self._keys
            if cur_keys.size >= self.capacity:
                keep = key_arr < self._threshold
                if not bool(np.any(keep)):
                    return
                key_arr = key_arr[keep]
                row_arr = row_arr[keep]
            self._keys = np.concatenate([cur_keys, key_arr])
            self._rows = np.concatenate([self._rows, row_arr])
        assert self._keys is not None and self._rows is not None
        if self._keys.size > self.capacity:
            sel = np.argpartition(self._keys, self.capacity - 1)[: self.capacity]
            self._keys = self._keys[sel]
            self._rows = self._rows[sel]
        # Deliberately NO per-update sort: partition selection keeps the k smallest
        # set; sorting is deferred to sample(). Per-update sorting made single-row
        # callers (BAM) quadratic — O(k log k) per accepted record.
        self._threshold = (
            float(np.max(self._keys)) if self._keys.size >= self.capacity else math.inf
        )

    @property
    def threshold(self) -> float:
        """Current k-th smallest key; a new observation enters only below it."""
        return self._threshold

    @property
    def is_full(self) -> bool:
        """True once ``capacity`` observations are retained."""
        return self.sampled_count >= self.capacity

    def sample(self) -> dict[str, list[float]]:
        """Return the retained sample as one list per named feature.

        Rows are key-sorted here (the only sort, done once) so output order is
        deterministic regardless of how blocks were offered. Reservoirs holding
        fewer columns than FEATURE_NAMES (e.g. insert-size-only) return empty
        lists for the unused feature names.
        """
        if self._rows is None or self._rows.shape[0] == 0:
            return {name: [] for name in FEATURE_NAMES}
        assert self._keys is not None
        order = np.argsort(self._keys, kind="stable")
        rows = self._rows[order]
        ncols = rows.shape[1]
        return {
            name: rows[:, i].tolist() if i < ncols else [] for i, name in enumerate(FEATURE_NAMES)
        }

    @property
    def sampled_count(self) -> int:
        return 0 if self._rows is None else int(self._rows.shape[0])
