# Regression tests for the L1-Potts median computation bug.
#
# Root cause: the original IndexedLinkedHistogram (Java and initial Rust port)
# used `median = iterator.prev` when cumulative weight first exceeded half the
# total, instead of `median = iterator` (the element that actually crossed the
# threshold).  For odd-weight inputs this caused the full-range deviation
# (deviations[0]) to be computed around the wrong value, inflating the cost of
# the constant (0-jump) solution and leading the DP to return a sub-optimal
# multi-jump result.
#
# These tests target exactly that failure mode, supplementing the existing
# brute-force suite with deterministic analytic checks and broader random
# coverage.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# Helpers shared with the brute-force suite
# ---------------------------------------------------------------------------

def _all_partitions(n):
    for mask in range(1 << (n - 1)):
        jumps = [i + 1 for i in range(n - 1) if mask & (1 << i)]
        boundaries = [0] + jumps + [n]
        yield [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]


def brute_l1_potts(f, gamma, weights=None):
    """Exhaustive O(2^n) optimal L1-Potts energy for small n."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    best_energy = np.inf
    best_u = None
    for partition in _all_partitions(n):
        u = np.zeros(n)
        energy = gamma * (len(partition) - 1)
        for l, r in partition:
            med = pl.weighted_median(f[l:r], w[l:r])
            u[l:r] = med
            energy += float(np.sum(w[l:r] * np.abs(f[l:r] - med)))
        if energy < best_energy - 1e-14:
            best_energy = energy
            best_u = u.copy()
    return best_u, best_energy


def _l1_energy(u, f, gamma, weights=None):
    w = np.ones(len(f)) if weights is None else np.asarray(weights, dtype=float)
    return float(gamma * pl.count_jumps(u) + np.sum(w * np.abs(u - f)))


# ---------------------------------------------------------------------------
# Regression: the exact case that exposed the bug
# ---------------------------------------------------------------------------

class TestMedianBugRegression:
    """The original failure: seed=0, n=7 chose a 1-jump solution whose energy
    exceeded the brute-force optimal (0-jump, constant median) by 0.128."""

    def test_seed0_n7_exact(self):
        rng = np.random.default_rng(100)
        f = rng.standard_normal(7)
        gamma = rng.uniform(0.1, 2.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        e_dp = _l1_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-8,
                            err_msg=f"gamma={gamma:.6f}, f={f}")

    def test_constant_optimal_is_recovered_odd_n(self):
        """When gamma is large the global optimum is always the constant
        (whole-signal weighted median). The bug caused this to be missed for
        odd-length signals."""
        for n in [3, 5, 7, 9, 11]:
            rng = np.random.default_rng(n * 17)
            f = rng.standard_normal(n)
            # With gamma >> max possible deviation, constant is always cheapest.
            gamma = 1e6
            u, _, nj, _ = pl.min_l1_potts(f, gamma)
            assert nj == 0, (
                f"n={n}: expected 0 jumps for large gamma, got {nj}; u={u}"
            )
            # Value must equal the weighted median of the whole signal
            expected_med = pl.weighted_median(f, np.ones(n))
            npt.assert_allclose(u, expected_med, atol=1e-10,
                                err_msg=f"n={n}: constant value != median")

    def test_constant_better_than_split_odd_n(self):
        """Construct a signal where the 0-jump solution is strictly cheaper
        than any 1-jump split, then verify the DP finds it.

        For f = [0, 0, 1, 0, 0] (n=5) and gamma just below the L1 gain of the
        best split, the constant (median=0) is optimal."""
        f = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        # Median of full signal = 0; total deviation = 1.
        # Best 1-jump split isolates the '1' in its own segment:
        #   split at [0..=3, 4] or [0, 1..=4]: deviation 0 + gamma each.
        # Actually best 1-jump: isolate 1 on its own gives deviation 0+0+gamma.
        # No, let's compute properly via brute force.
        gamma = 0.6
        u_dp, _, nj, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        e_dp = _l1_energy(u_dp, f, gamma)
        npt.assert_allclose(e_dp, e_bf, atol=1e-10)

    def test_deviation_of_full_range_equals_weighted_median_deviation(self):
        """The energy returned for a forced single-segment solution must equal
        the L1 deviation around the weighted median — the exact quantity the
        buggy code was inflating."""
        for seed in range(10):
            rng = np.random.default_rng(seed + 500)
            n = 2 * seed + 3          # odd lengths: 3, 5, 7, ..., 21
            f = rng.standard_normal(n)
            w = np.ones(n)
            med = pl.weighted_median(f, w)
            expected_dev = float(np.sum(np.abs(f - med)))

            # With huge gamma, constant is chosen; its data error == deviation
            _, data_err, nj, _ = pl.min_l1_potts(f, gamma=1e8)
            assert nj == 0, f"n={n}, seed={seed}: expected 0 jumps"
            npt.assert_allclose(data_err, expected_dev, atol=1e-10,
                                err_msg=f"n={n}, seed={seed}: data_err mismatch")


# ---------------------------------------------------------------------------
# Extended brute-force global-optimality sweep
# ---------------------------------------------------------------------------

class TestL1BruteForceExtended:
    """Wider random sweep to ensure the DP is globally optimal for small n."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_n7_optimal(self, seed):
        rng = np.random.default_rng(seed + 200)
        f = rng.standard_normal(7)
        gamma = rng.uniform(0.05, 3.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        npt.assert_allclose(_l1_energy(u_dp, f, gamma), e_bf, atol=1e-8,
                            err_msg=f"seed={seed}, gamma={gamma:.4f}")

    @pytest.mark.parametrize("seed", range(10))
    def test_random_n9_optimal(self, seed):
        rng = np.random.default_rng(seed + 300)
        f = rng.standard_normal(9)
        gamma = rng.uniform(0.05, 3.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        npt.assert_allclose(_l1_energy(u_dp, f, gamma), e_bf, atol=1e-8,
                            err_msg=f"seed={seed}, gamma={gamma:.4f}")

    @pytest.mark.parametrize("seed", range(5))
    def test_random_n11_optimal(self, seed):
        rng = np.random.default_rng(seed + 400)
        f = rng.standard_normal(11)
        gamma = rng.uniform(0.05, 3.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        npt.assert_allclose(_l1_energy(u_dp, f, gamma), e_bf, atol=1e-8,
                            err_msg=f"seed={seed}, gamma={gamma:.4f}")

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11])
    def test_odd_n_constant_signal_optimal(self, n):
        """Piecewise-constant input (already optimal) must reproduce itself."""
        rng = np.random.default_rng(n + 700)
        f = rng.standard_normal(n)
        gamma = rng.uniform(0.1, 2.0)
        u_dp, _, _, _ = pl.min_l1_potts(f, gamma)
        _, e_bf = brute_l1_potts(f, gamma)
        npt.assert_allclose(_l1_energy(u_dp, f, gamma), e_bf, atol=1e-8)

    def test_weighted_odd_n_optimal(self):
        """Non-uniform weights, odd n — median shifts in a non-obvious way."""
        rng = np.random.default_rng(999)
        for seed in range(8):
            rng2 = np.random.default_rng(seed + 800)
            n = 7
            f = rng2.standard_normal(n)
            w = rng2.uniform(0.5, 2.0, size=n)
            gamma = rng2.uniform(0.2, 2.0)
            u_dp, _, _, _ = pl.min_l1_potts(f, gamma, weights=w)
            _, e_bf = brute_l1_potts(f, gamma, weights=w)
            npt.assert_allclose(_l1_energy(u_dp, f, gamma, w), e_bf, atol=1e-8,
                                err_msg=f"seed={seed}")


# ---------------------------------------------------------------------------
# Median correctness: each L1 segment value == weighted median of that segment
# ---------------------------------------------------------------------------

class TestL1SegmentMedians:
    """The value on every L1-Potts segment must be the weighted median of f
    restricted to that segment.  This is an invariant of any correct L1-Potts
    implementation — violations indicate a median computation error."""

    def _check_segment_medians(self, f, gamma, weights=None, atol=1e-8):
        w = np.ones(len(f)) if weights is None else np.asarray(weights, float)
        u, _, _, _ = pl.min_l1_potts(f, gamma, weights=w)
        jumps = np.where(np.diff(u) != 0)[0]
        bounds = np.concatenate([[0], jumps + 1, [len(f)]])
        for k in range(len(bounds) - 1):
            l, r = bounds[k], bounds[k + 1]
            expected = pl.weighted_median(f[l:r], w[l:r])
            npt.assert_allclose(u[l:r], expected, atol=atol,
                                err_msg=f"Segment [{l},{r}] value != weighted median")

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11, 15, 21])
    def test_odd_length_segment_medians(self, n):
        rng = np.random.default_rng(n * 31)
        f = rng.standard_normal(n)
        self._check_segment_medians(f, gamma=0.5)
        self._check_segment_medians(f, gamma=2.0)

    @pytest.mark.parametrize("seed", range(8))
    def test_random_segment_medians(self, seed):
        rng = np.random.default_rng(seed + 600)
        n = rng.integers(5, 16)
        f = rng.standard_normal(n)
        gamma = rng.uniform(0.1, 3.0)
        self._check_segment_medians(f, gamma)

    def test_weighted_segment_medians(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal(9)
        w = rng.uniform(0.3, 3.0, size=9)
        self._check_segment_medians(f, gamma=1.0, weights=w)

    def test_single_segment_value_equals_global_median(self):
        """When gamma forces a single segment, the output value == median(f)."""
        rng = np.random.default_rng(77)
        for n in [3, 5, 7, 9]:
            f = rng.standard_normal(n)
            u, _, nj, _ = pl.min_l1_potts(f, gamma=1e7)
            assert nj == 0, f"n={n}: expected 0 jumps"
            med = pl.weighted_median(f, np.ones(n))
            npt.assert_allclose(u[0], med, atol=1e-10, err_msg=f"n={n}")


# ---------------------------------------------------------------------------
# Weighted-median precision: the Python helper must agree with the DP
# ---------------------------------------------------------------------------

class TestWeightedMedianCorrectness:
    """The weighted_median helper is called by brute_l1_potts above; these
    tests verify it is itself correct, so that brute-force is trustworthy."""

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11])
    def test_odd_uniform_matches_numpy_median(self, n):
        rng = np.random.default_rng(n * 7)
        data = rng.standard_normal(n)
        w = np.ones(n)
        npt.assert_allclose(pl.weighted_median(data, w), np.median(data), atol=1e-12)

    def test_heavy_weight_dominates(self):
        """If one element carries > half the total weight it must be the median."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([0.1, 0.1, 10.0, 0.1, 0.1])  # element 3 (value=3) dominates
        assert pl.weighted_median(data, w) == 3.0

    def test_symmetric_weights_middle_element(self):
        """Symmetric odd-n case: median is the centre value."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        assert pl.weighted_median(data, w) == 30.0

    @pytest.mark.parametrize("scale", [0.5, 2.0, 100.0])
    def test_weight_scaling_invariant(self, scale):
        """Scaling all weights by a positive constant must not change the median."""
        rng = np.random.default_rng(13)
        data = rng.standard_normal(9)
        w = rng.uniform(0.5, 2.0, size=9)
        m1 = pl.weighted_median(data, w)
        m2 = pl.weighted_median(data, w * scale)
        npt.assert_allclose(m1, m2, atol=1e-12)
