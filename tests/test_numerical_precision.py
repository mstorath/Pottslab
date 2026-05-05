# Numerical precision and analytical solution tests for pottslab.
# These check that the Rust port matches mathematical closed-form answers
# precisely, and that accumulated floating-point errors are within tight bounds.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# Known closed-form solutions for L2-Potts
# ---------------------------------------------------------------------------

class TestL2PottsAnalytical:
    """For simple cases the optimal solution is known analytically."""

    def test_two_equal_segments_exact_mean(self):
        """Two equal-length constant segments: solution equals input exactly."""
        n = 20
        a, b = 1.3, 7.6
        f = np.array([a] * n + [b] * n)
        # With small gamma, the optimal solution must be the two-segment one
        gamma = 0.1  # << n * (b - a)^2 / 4 = 20 * 40.69 / 4 ≈ 203
        u = pl.min_l2_potts(f, gamma)
        npt.assert_allclose(u[:n], a, atol=1e-12)
        npt.assert_allclose(u[n:], b, atol=1e-12)

    def test_large_gamma_mean_is_weighted_average(self):
        """With gamma >> signal, constant solution = (weighted) mean of f."""
        f = np.array([0.0, 2.0, 4.0, 6.0])
        w = np.array([1.0, 2.0, 1.0, 2.0])
        u = pl.min_l2_potts(f, gamma=1e9, weights=w)
        expected_mean = np.sum(w * f) / np.sum(w)
        npt.assert_allclose(u, expected_mean, atol=1e-8)

    def test_threshold_exactly_at_boundary(self):
        """At the exact threshold gamma = n * delta^2 / 4, both solutions are tied.
        The DP must return energy ≤ brute force energy (ties go to either)."""
        n = 40
        delta = 1.0
        gamma_threshold = n * delta ** 2 / 4  # = 10
        f = np.array([0.0] * (n // 2) + [delta] * (n // 2))

        u = pl.min_l2_potts(f, gamma=gamma_threshold - 1e-6)
        e = pl.energy_l2_potts(u, f, gamma_threshold - 1e-6)

        # Constant solution energy
        u_const = np.full(n, delta / 2)
        e_const = pl.energy_l2_potts(u_const, f, gamma_threshold - 1e-6)

        assert e <= e_const + 1e-10

    def test_l2_solution_segment_means_exact(self):
        """The value on each segment must be the exact weighted mean of f there."""
        rng = np.random.default_rng(50)
        n = 60
        f = rng.standard_normal(n)
        gamma = 2.0
        u = pl.min_l2_potts(f, gamma)
        # Find segments
        jumps = np.where(np.diff(u) != 0)[0]
        bounds = np.concatenate([[0], jumps + 1, [n]])
        for k in range(len(bounds) - 1):
            l, r = bounds[k], bounds[k + 1]
            expected_mean = f[l:r].mean()
            npt.assert_allclose(u[l:r], expected_mean, atol=1e-10,
                                err_msg=f"Segment [{l},{r}] mean mismatch")

    def test_weighted_solution_segment_means_exact(self):
        """With weights, each segment value must be the weighted mean of f."""
        rng = np.random.default_rng(51)
        n = 20
        f = rng.standard_normal(n)
        w = rng.uniform(0.5, 2.0, size=n)
        u = pl.min_l2_potts(f, gamma=1.5, weights=w)
        jumps = np.where(np.diff(u) != 0)[0]
        bounds = np.concatenate([[0], jumps + 1, [n]])
        for k in range(len(bounds) - 1):
            l, r = bounds[k], bounds[k + 1]
            wm = np.sum(w[l:r] * f[l:r]) / np.sum(w[l:r])
            npt.assert_allclose(u[l:r], wm, atol=1e-10)

    def test_multichannel_segment_means_exact(self):
        """For vector-valued signals, each segment value == per-channel mean."""
        rng = np.random.default_rng(52)
        f = rng.standard_normal((30, 3))
        u = pl.min_l2_potts(f, gamma=1.5)
        jumps = np.where(np.any(np.diff(u, axis=0) != 0, axis=1))[0]
        bounds = np.concatenate([[0], jumps + 1, [30]])
        for k in range(len(bounds) - 1):
            l, r = bounds[k], bounds[k + 1]
            expected = f[l:r].mean(axis=0)   # shape (3,)
            # Broadcast: each row of u[l:r] must equal the segment mean
            npt.assert_allclose(u[l:r], np.tile(expected, (r - l, 1)), atol=1e-10)


# ---------------------------------------------------------------------------
# Known closed-form solutions for L1-Potts
# ---------------------------------------------------------------------------

class TestL1PottsAnalytical:
    """For constant-segment inputs the L1-Potts solution is exact."""

    def test_constant_segments_exact(self):
        """Clean piecewise-constant input (< threshold gamma) is unchanged."""
        f = np.array([1.0] * 15 + [5.0] * 15)
        # Cost of preserving: 0 + gamma
        # Cost of merging: gamma * 0 + ||f - med||_1 = sum |f - 3| = 15*2 + 15*2 = 60
        # Preserve if gamma < 60 <=> here gamma=1 works
        u, *_ = pl.min_l1_potts(f, gamma=1.0)
        npt.assert_allclose(u[:15], 1.0, atol=1e-10)
        npt.assert_allclose(u[15:], 5.0, atol=1e-10)

    def test_l1_segment_values_are_weighted_medians(self):
        """The value on each L1 segment must be the weighted median of f there."""
        rng = np.random.default_rng(53)
        n = 20
        f = rng.standard_normal(n)
        w = np.ones(n)
        u, *_ = pl.min_l1_potts(f, gamma=2.0, weights=w)
        jumps = np.where(np.diff(u) != 0)[0]
        bounds = np.concatenate([[0], jumps + 1, [n]])
        for k in range(len(bounds) - 1):
            l, r = bounds[k], bounds[k + 1]
            expected_med = pl.weighted_median(f[l:r], w[l:r])
            npt.assert_allclose(u[l:r], expected_med, atol=1e-8,
                                err_msg=f"Segment [{l},{r}] median mismatch")


# ---------------------------------------------------------------------------
# Weighted median precision tests
# ---------------------------------------------------------------------------

class TestWeightedMedianPrecision:
    def test_equal_weights_odd_n(self):
        """Uniform weights, odd n: must return the middle value."""
        data = np.array([4.0, 2.0, 7.0, 1.0, 9.0])
        w = np.ones(5)
        med = pl.weighted_median(data, w)
        npt.assert_allclose(med, np.median(data), atol=1e-12)

    def test_equal_weights_even_n(self):
        """Uniform weights, even n: result is one of the two middle values."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        med = pl.weighted_median(data, w)
        assert med in (2.0, 3.0)

    @pytest.mark.parametrize("n", [3, 7, 11, 21])
    def test_integer_valued_data(self, n):
        """Integer data, uniform weights: matches numpy median for odd n."""
        rng = np.random.default_rng(n)
        data = rng.integers(0, 100, size=n).astype(float)
        w = np.ones(n)
        med = pl.weighted_median(data, w)
        npt.assert_allclose(med, np.median(data), atol=1e-12)

    def test_weight_sum_invariance(self):
        """Scaling all weights by a constant must not change the median."""
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        med1 = pl.weighted_median(data, w)
        med2 = pl.weighted_median(data, 10 * w)
        npt.assert_allclose(med1, med2, atol=1e-12)


# ---------------------------------------------------------------------------
# Soft threshold precision
# ---------------------------------------------------------------------------

class TestSoftThresholdPrecision:
    @pytest.mark.parametrize("x,tau,expected", [
        (1.5, 0.5, 1.0),
        (-1.5, 0.5, -1.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (100.0, 0.0, 100.0),
        (-0.3, 0.1, -0.2),
    ])
    def test_scalar_cases(self, x, tau, expected):
        npt.assert_allclose(pl.soft_threshold(np.array([x]), tau)[0],
                            expected, atol=1e-12)

    def test_array_symmetry(self):
        """soft_threshold(x) = -soft_threshold(-x)."""
        x = np.linspace(-2, 2, 41)
        tau = 0.7
        npt.assert_allclose(pl.soft_threshold(x, tau),
                            -pl.soft_threshold(-x, tau), atol=1e-12)

    def test_non_expansion(self):
        """||soft_threshold(x)|| ≤ ||x|| for all x."""
        rng = np.random.default_rng(54)
        x = rng.standard_normal(50)
        u = pl.soft_threshold(x, tau=0.5)
        assert np.linalg.norm(u) <= np.linalg.norm(x) + 1e-12


# ---------------------------------------------------------------------------
# samples_to_weights precision
# ---------------------------------------------------------------------------

class TestSamplesToWeightsPrecision:
    def test_two_equidistant_samples(self):
        s = np.array([0.0, 2.0])
        w = pl.samples_to_weights(s)
        npt.assert_allclose(w, [1.0, 1.0], atol=1e-12)

    def test_three_equidistant_samples_weights(self):
        s = np.array([0.0, 1.0, 2.0])
        w = pl.samples_to_weights(s)
        npt.assert_allclose(w, [0.5, 1.0, 0.5], atol=1e-12)

    def test_nonuniform_total_equals_span(self):
        s = np.array([0.0, 0.5, 1.5, 3.0])
        w = pl.samples_to_weights(s)
        npt.assert_allclose(w.sum(), s[-1] - s[0], atol=1e-12)

    def test_monotone_input_required(self):
        """Non-monotone samples are rejected with a clear error."""
        s = np.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            pl.samples_to_weights(s)
