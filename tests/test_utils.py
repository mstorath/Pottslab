# Tests for utility functions.
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestCountJumps:
    def test_no_jumps(self):
        assert pl.count_jumps(np.array([5.0, 5.0, 5.0])) == 0

    def test_one_jump(self):
        assert pl.count_jumps(np.array([0.0, 0.0, 1.0, 1.0])) == 1

    def test_two_jumps(self):
        assert pl.count_jumps(np.array([0.0, 0.0, 1.0, 1.0, 2.0])) == 2

    def test_all_different(self):
        assert pl.count_jumps(np.array([0.0, 1.0, 2.0, 3.0])) == 3

    def test_single_element(self):
        assert pl.count_jumps(np.array([42.0])) == 0

    def test_two_elements_same(self):
        assert pl.count_jumps(np.array([1.0, 1.0])) == 0

    def test_two_elements_different(self):
        assert pl.count_jumps(np.array([1.0, 2.0])) == 1

    def test_multichannel(self):
        """Jump counted when ANY channel changes."""
        u = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert pl.count_jumps(u) == 1


class TestWeightedMedian:
    def test_uniform_weights_matches_numpy(self):
        """Uniform weights: weighted median == numpy median (for odd n)."""
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        w = np.ones(len(data))
        wmed = pl.weighted_median(data, w)
        npt.assert_allclose(wmed, np.median(data), atol=1e-10)

    def test_single_element(self):
        assert pl.weighted_median(np.array([7.0]), np.array([1.0])) == 7.0

    def test_two_equal_weights(self):
        """Two elements equal weights: median is one of them."""
        wmed = pl.weighted_median(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        assert wmed in (1.0, 3.0)

    def test_heavy_weight_dominates(self):
        """Element with heavy weight should dominate the median."""
        data = np.array([0.0, 100.0])
        w = np.array([0.01, 0.99])  # 100 is almost certain median
        wmed = pl.weighted_median(data, w)
        assert wmed == 100.0

    def test_returns_float(self):
        assert isinstance(
            pl.weighted_median(np.array([1.0, 2.0]), np.array([1.0, 1.0])),
            float
        )


class TestSoftThreshold:
    def test_above_threshold(self):
        npt.assert_allclose(pl.soft_threshold(0.8, 0.3), 0.5, atol=1e-12)

    def test_below_threshold_zero(self):
        npt.assert_allclose(pl.soft_threshold(0.2, 0.3), 0.0, atol=1e-12)

    def test_negative_above_threshold(self):
        npt.assert_allclose(pl.soft_threshold(-0.8, 0.3), -0.5, atol=1e-12)

    def test_negative_below_threshold_zero(self):
        npt.assert_allclose(pl.soft_threshold(-0.2, 0.3), 0.0, atol=1e-12)

    def test_exactly_at_threshold(self):
        npt.assert_allclose(pl.soft_threshold(0.3, 0.3), 0.0, atol=1e-12)

    def test_array_input(self):
        x = np.array([0.5, -0.5, 0.1, -0.1])
        u = pl.soft_threshold(x, 0.3)
        expected = np.array([0.2, -0.2, 0.0, 0.0])
        npt.assert_allclose(u, expected, atol=1e-12)


class TestSamplesToWeights:
    def test_equidistant_gives_uniform_weights(self):
        """Equidistant samples → uniform trapezoidal weights."""
        samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        w = pl.samples_to_weights(samples)
        # Interior weights = 1, endpoints = 0.5
        npt.assert_allclose(w[0], 0.5, atol=1e-12)
        npt.assert_allclose(w[-1], 0.5, atol=1e-12)
        npt.assert_allclose(w[1:-1], 1.0, atol=1e-12)

    def test_total_weight_equals_span(self):
        """Sum of weights = total span (trapezoidal rule for constant = 1)."""
        samples = np.array([0.0, 1.0, 2.0, 3.0])
        w = pl.samples_to_weights(samples)
        npt.assert_allclose(np.sum(w), 3.0, atol=1e-12)

    def test_single_sample_weight_one(self):
        """Single sample: weight = 1."""
        w = pl.samples_to_weights(np.array([5.0]))
        npt.assert_allclose(w, [1.0], atol=1e-12)


class TestEnergyL2Potts:
    def test_optimal_signal_energy_zero_data_term(self, step_signal_1d):
        """For u = f (exact data), ||u - f||^2 = 0."""
        energy = pl.energy_l2_potts(step_signal_1d, step_signal_1d, gamma=1.0)
        n_jumps = pl.count_jumps(step_signal_1d)
        npt.assert_allclose(energy, 1.0 * n_jumps, atol=1e-12)

    def test_constant_solution_energy(self):
        """Constant solution: 0 jumps, ||u - f||^2 = variance * n."""
        f = np.array([0.0, 0.0, 1.0, 1.0])
        u = np.array([0.5, 0.5, 0.5, 0.5])
        energy = pl.energy_l2_potts(u, f, gamma=10.0)
        expected = 0 * 10.0 + np.sum((u - f) ** 2)
        npt.assert_allclose(energy, expected, atol=1e-12)
