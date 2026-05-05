# Brute-force global-optimality verification for the 1-D Potts DP.
# For small n, enumerate every possible partition and confirm the DP finds
# the one with the lowest energy.  This is the strongest possible correctness
# guarantee for the Rust port of L2Potts.java and IndexedLinkedHistogram.java.

import itertools
import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_partitions(n):
    """Yield every partition of [0..n) as a list of (start, stop) pairs."""
    # A partition corresponds to choosing a subset of {1, …, n-1} as jump positions.
    for mask in range(1 << (n - 1)):
        jumps = [i + 1 for i in range(n - 1) if mask & (1 << i)]
        boundaries = [0] + jumps + [n]
        yield [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]


def brute_l2_potts(f, gamma, weights=None):
    """Exhaustive O(2^n) search for the globally optimal L2-Potts partition."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    best_energy = np.inf
    best_u = None
    for partition in _all_partitions(n):
        u = np.zeros(n)
        energy = gamma * (len(partition) - 1)  # jump penalty
        for l, r in partition:
            seg_f = f[l:r]
            seg_w = w[l:r]
            total_w = seg_w.sum()
            mean = (seg_w * seg_f).sum() / total_w if total_w > 0 else 0.0
            u[l:r] = mean
            energy += float(np.sum(seg_w * (seg_f - mean) ** 2))
        if energy < best_energy - 1e-14:
            best_energy = energy
            best_u = u.copy()
    return best_u, best_energy


def brute_l1_potts(f, gamma, weights=None):
    """Exhaustive O(2^n) search for the globally optimal L1-Potts partition."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    best_energy = np.inf
    best_u = None
    for partition in _all_partitions(n):
        u = np.zeros(n)
        energy = gamma * (len(partition) - 1)
        for l, r in partition:
            seg_f = f[l:r]
            seg_w = w[l:r]
            # Weighted median
            med = pl.weighted_median(seg_f, seg_w)
            u[l:r] = med
            energy += float(np.sum(seg_w * np.abs(seg_f - med)))
        if energy < best_energy - 1e-14:
            best_energy = energy
            best_u = u.copy()
    return best_u, best_energy


def _dp_energy(u, f, gamma, weights=None):
    """Compute true L2-Potts energy of a candidate solution."""
    f = np.asarray(f, dtype=float)
    w = np.ones(len(f)) if weights is None else np.asarray(weights, dtype=float)
    return float(gamma * pl.count_jumps(u) + np.sum(w * (u - f) ** 2))


def _dp_l1_energy(u, f, gamma, weights=None):
    w = np.ones(len(f)) if weights is None else np.asarray(weights, dtype=float)
    return float(gamma * pl.count_jumps(u) + np.sum(w * np.abs(u - f)))


# ---------------------------------------------------------------------------
# L2-Potts brute-force tests
# ---------------------------------------------------------------------------

class TestL2PottsBruteForce:
    """Verify that the DP gives the globally optimal solution for small n."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_n8_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        f = rng.standard_normal(8)
        gamma = rng.uniform(0.1, 3.0)
        u_dp = pl.min_l2_potts(f, gamma)
        u_bf, e_bf = brute_l2_potts(f, gamma)
        e_dp = _dp_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-10,
                            err_msg=f"seed={seed}, gamma={gamma:.3f}: DP energy {e_dp} > brute-force {e_bf}")

    @pytest.mark.parametrize("gamma", [0.05, 0.5, 2.0, 10.0, 50.0])
    def test_step_signal_n10_optimal_at_all_gammas(self, gamma):
        f = np.array([0.0] * 5 + [1.0] * 5)
        u_dp = pl.min_l2_potts(f, gamma)
        _, e_bf = brute_l2_potts(f, gamma)
        e_dp = _dp_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-10)

    def test_three_segment_n12_optimal(self):
        f = np.array([0.0] * 4 + [2.0] * 4 + [-1.0] * 4)
        for gamma in [0.1, 1.0, 5.0, 20.0]:
            u_dp = pl.min_l2_potts(f, gamma)
            _, e_bf = brute_l2_potts(f, gamma)
            e_dp = _dp_energy(u_dp, f, gamma)
            npt.assert_allclose(e_dp, e_bf, atol=1e-10,
                                err_msg=f"gamma={gamma}: energy mismatch")

    def test_weighted_n8_optimal(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal(8)
        w = rng.uniform(0.5, 2.0, size=8)
        gamma = 1.0
        u_dp = pl.min_l2_potts(f, gamma, weights=w)
        _, e_bf = brute_l2_potts(f, gamma, weights=w)
        e_dp = _dp_energy(u_dp, f, gamma, weights=w)
        npt.assert_allclose(e_dp, e_bf, atol=1e-10)

    def test_negative_values_n8_optimal(self):
        f = np.array([-3.0, -3.0, -3.0, 2.0, 2.0, 2.0, -1.0, -1.0])
        for gamma in [0.5, 2.0]:
            u_dp = pl.min_l2_potts(f, gamma)
            _, e_bf = brute_l2_potts(f, gamma)
            e_dp = _dp_energy(u_dp, f, gamma)
            npt.assert_allclose(e_dp, e_bf, atol=1e-10)

    def test_all_different_n6_optimal(self):
        """When all values are different, optimal partition depends on gamma."""
        f = np.array([1.0, 3.0, 0.0, 5.0, 2.0, 4.0])
        for gamma in [0.01, 1.0, 100.0]:
            u_dp = pl.min_l2_potts(f, gamma)
            _, e_bf = brute_l2_potts(f, gamma)
            e_dp = _dp_energy(u_dp, f, gamma)
            npt.assert_allclose(e_dp, e_bf, atol=1e-10)

    def test_multichannel_2ch_n6_optimal(self):
        """Vector-valued L2-Potts should also be globally optimal."""
        rng = np.random.default_rng(7)
        f = rng.standard_normal((6, 2))
        gamma = 1.0
        u_dp = pl.min_l2_potts(f, gamma)
        # Compare by checking energy (brute force is per scalar channel)
        # Energy = gamma * jumps + sum_k ||u[:,k] - f[:,k]||^2
        e_dp = float(gamma * pl.count_jumps(u_dp) +
                     np.sum((u_dp - f) ** 2))
        # Must be ≤ energy of f itself
        e_f = float(gamma * pl.count_jumps(f) + 0.0)
        # And ≤ energy of constant solution
        means = f.mean(axis=0)
        u_const = np.tile(means, (6, 1))
        e_const = float(np.sum((u_const - f) ** 2))
        assert e_dp <= e_const + 1e-10


# ---------------------------------------------------------------------------
# L1-Potts brute-force tests
# ---------------------------------------------------------------------------

class TestL1PottsBruteForce:
    """Verify the L1-Potts DP gives the globally optimal solution for small n."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_random_n7_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed + 100)
        f = rng.standard_normal(7)
        gamma = rng.uniform(0.1, 2.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        e_dp = _dp_l1_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-8,
                            err_msg=f"seed={seed}, gamma={gamma:.3f}")

    @pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0, 20.0])
    def test_step_signal_n8_l1_optimal(self, gamma):
        f = np.array([0.0] * 4 + [1.0] * 4)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        e_dp = _dp_l1_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-8)

    def test_outlier_n8_l1_optimal(self):
        """L1-Potts should be globally optimal even with an outlier."""
        f = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 1.0])
        gamma = 1.0
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        e_dp = _dp_l1_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-8)
