"""Statistical distance correctness against independently computed examples."""

from __future__ import annotations

import numpy as np
import pytest

from seqconsistency.errors import ConfigError
from seqconsistency.statistics import (
    jensen_shannon_distance,
    wasserstein_distance,
)


def test_wasserstein_identical_distributions():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert wasserstein_distance(a, a, "x") == pytest.approx(0.0)


def test_wasserstein_hand_computed_shift():
    # a = [1,2,3], b = [3,4,5]: all mass moves 2 units -> W1 = 2.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    assert wasserstein_distance(a, b, "x") == pytest.approx(2.0)


def test_wasserstein_hand_computed_half_mass():
    # a = [0,0,10,10] vs b = [0,0,0,10]: 1/4 of the 10s (0.25 mass) moves to 0.
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([0.0, 0.0, 0.0, 10.0])
    assert wasserstein_distance(a, b, "x") == pytest.approx(2.5)


def test_wasserstein_unequal_sizes():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0])
    # Empirical CDFs: distance = mean |F_a - F_b| integrated; compute by hand via
    # the definition W1 = integral |F_a(t) - F_b(t)| dt over the union support.
    # Supports 1..5: F_a = [.2,.4,.6,.8,1], F_b = [1/3,2/3,1,1,1] at t=1..3,
    # F_b stays 1 afterwards.
    diffs = [abs(0.2 - 1 / 3), abs(0.4 - 2 / 3), abs(0.6 - 1.0), abs(0.8 - 1.0), abs(1.0 - 1.0)]
    expected = sum(diffs)  # each support point contributes 1 unit interval
    assert wasserstein_distance(a, b, "x") == pytest.approx(expected, rel=1e-9)


def test_wasserstein_excludes_nan():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([2.0, 4.0])
    assert wasserstein_distance(a, b, "x") == pytest.approx(1.0)


def test_wasserstein_empty_side_raises():
    with pytest.raises(ConfigError):
        wasserstein_distance(np.array([]), np.array([1.0]), "x")


def test_jsd_identical_distributions():
    h = np.array([10, 20, 30, 40], dtype=float)
    assert jensen_shannon_distance(h, h, "m") == pytest.approx(0.0)


def test_jsd_hand_computed_disjoint():
    # Disjoint supports -> JSD distance = 1 bit (base 2) = 1.0.
    p = np.array([0.0, 1.0])
    q = np.array([1.0, 0.0])
    assert jensen_shannon_distance(p, q, "m") == pytest.approx(1.0)


def test_jsd_hand_computed_small_case():
    # p = [3/4, 1/4], q = [1/4, 3/4]; m = [1/2, 1/2].
    # JSD divergence (base 2):
    #   0.5 * (0.75*log2(0.75/0.5) + 0.25*log2(0.25/0.5))  [p vs m]
    # + 0.5 * (0.25*log2(0.25/0.5) + 0.75*log2(0.75/0.5))  [q vs m]
    # = 0.5 * (0.75*log2(1.5) + 0.25*log2(0.5)) * 2
    # = 0.75*log2(1.5) - 0.25
    div = 0.75 * np.log2(1.5) - 0.25
    p = np.array([0.75, 0.25])
    q = np.array([0.25, 0.75])
    assert jensen_shannon_distance(p, q, "m") == pytest.approx(float(np.sqrt(div)))


def test_jsd_zero_bins_do_not_break():
    # Zero-count bins on both sides contribute nothing; no pseudocount added.
    # Normalized: p = [0,.5,0,.5], q = [.5,0,0,.5] -> JSD divergence 0.5 bits,
    # distance sqrt(0.5).
    p = np.array([0.0, 10.0, 0.0, 10.0])
    q = np.array([10.0, 0.0, 0.0, 10.0])
    assert jensen_shannon_distance(p, q, "m") == pytest.approx(float(np.sqrt(0.5)))


def test_jsd_is_distance_not_divergence():
    # sqrt-scale check: JSD distance of p vs 2p-everywhere must be <= 1.
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    d = jensen_shannon_distance(p, q, "m")
    assert 0.0 <= d <= 1.0
    assert d > 0.0


def test_jsd_shape_mismatch_raises():
    with pytest.raises(ConfigError, match="bin mismatch"):
        jensen_shannon_distance(np.ones(3), np.ones(4), "m")


def test_jsd_all_zero_histogram_raises_not_zero_distance():
    with pytest.raises(ConfigError, match="zero observations"):
        jensen_shannon_distance(np.zeros(3), np.array([1.0, 1.0, 1.0]), "m")


def test_degenerate_zero_variance_wasserstein():
    a = np.full(100, 5.0)
    b = np.full(100, 5.0)
    assert wasserstein_distance(a, b, "x") == pytest.approx(0.0)
    c = np.full(100, 7.0)
    assert wasserstein_distance(a, c, "x") == pytest.approx(2.0)
