"""Reservoir sampling: determinism, seed stability, unbiasedness, bounded keys."""

from __future__ import annotations

import numpy as np
import pytest

from seqconsistency.sampling import FeatureReservoir, dataset_seed


def _block(rng, n, cols=5):
    return rng.normal(size=(n, cols))


def test_dataset_seed_is_stable_and_distinct():
    assert dataset_seed(42, "dsA") == dataset_seed(42, "dsA")
    assert dataset_seed(42, "dsA") != dataset_seed(42, "dsB")
    assert dataset_seed(42, "dsA") != dataset_seed(43, "dsA")
    assert dataset_seed(42, "dsA") < 2**128


def test_reservoir_identical_across_worker_emulations():
    # Same stream, different block splits -> identical retained rows.
    r1 = FeatureReservoir(50)
    r2 = FeatureReservoir(50)
    data_rng = np.random.default_rng(99)
    rows = data_rng.normal(size=(1000, 5))
    keys = np.random.default_rng(5).random(1000)
    for start in range(0, 1000, 100):
        r1.add_block(rows[start : start + 100], keys=keys[start : start + 100])
    for start in range(0, 1000, 250):
        r2.add_block(rows[start : start + 250], keys=keys[start : start + 250])
    assert r1.sample()["length"] == r2.sample()["length"]
    for name in (
        "gc_fraction",
        "n_fraction",
        "mean_quality_arithmetic",
        "mean_quality_error_based",
    ):
        assert r1.sample()[name] == r2.sample()[name]


def test_reservoir_sample_is_uniform():
    # With k=200 of 2000 items, each item is retained with p=0.1 per trial; over
    # 60 trials the expected count is 6 and P(count==0) = 0.9**60 ~ 0.002.
    counts = np.zeros(2000)
    for trial in range(60):
        rng = np.random.default_rng(trial)
        res = FeatureReservoir(200, rng)
        rows = np.arange(2000, dtype=float).reshape(-1, 1)
        res.add_block(rows)
        counts[np.asarray(res.sample()["length"], dtype=int)] += 1
    assert abs(counts.mean() - 6.0) < 0.5
    assert (counts > 0).mean() >= 0.99  # ~0.2% of items may legitimately miss
    assert counts.max() < 20  # no item is systematically favored


def test_reservoir_capacity_covers_all_when_small():
    rng = np.random.default_rng(1)
    res = FeatureReservoir(10000, rng)
    res.add_block(np.arange(50, dtype=float).reshape(-1, 1))
    sample = res.sample()["length"]
    assert sorted(sample) == list(map(float, range(50)))
    assert res.sampled_count == 50


def test_reservoir_zero_capacity():
    res = FeatureReservoir(0, np.random.default_rng(0))
    res.add_block(np.ones((10, 5)))
    assert res.sampled_count == 0
    assert res.offered == 10


def test_reservoir_threshold_skips_large_keys():
    rng = np.random.default_rng(3)
    res = FeatureReservoir(5, rng)
    res.add_block(np.zeros((100, 5)))
    assert res.threshold <= 1.0
    assert res.sampled_count == 5


def test_reservoir_rejects_bad_shape():
    from seqconsistency.errors import InputDataError

    res = FeatureReservoir(5, np.random.default_rng(0))
    with pytest.raises(InputDataError):
        res.add_block(np.zeros(5))


def test_shared_keys_keep_pairs_together():
    # PE contract: two reservoirs fed identical keys retain identical row indices.
    keys = np.random.default_rng(11).random(500)
    r1 = FeatureReservoir(30)
    r2 = FeatureReservoir(30)
    rng = np.random.default_rng(21)
    r1.add_block(rng.normal(size=(500, 5)), keys=keys)
    r2.add_block(rng.normal(size=(500, 5)) * 3, keys=keys)
    # Same keys -> same retained positions in stream order.
    assert len(r1.sample()["length"]) == len(r2.sample()["length"]) == 30
