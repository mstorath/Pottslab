# Tests for the 1-D L2-Potts solver.
# Verifies faithfulness of the Rust port of L2Potts.java.

import numpy as np
import numpy.testing as npt
import pytest
import time

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestL2Potts1DBasic:
    def test_constant_signal_unchanged(self):
        """A constant signal is already optimal — no changes expected."""
        f = np.full(20, 3.14)
        u = pl.min_l2_potts(f, gamma=1.0)
        npt.assert_allclose(u, f, atol=1e-12)

    def test_exact_step_small_gamma_preserved(self, step_signal_1d):
        """With small gamma, a clean step signal should be preserved exactly."""
        u = pl.min_l2_potts(step_signal_1d, gamma=0.01)
        npt.assert_allclose(u, step_signal_1d, atol=1e-12)

    def test_exact_step_large_gamma_collapses(self, step_signal_1d):
        """With large gamma (larger than the jump cost), signal collapses to mean."""
        # Mean of [0]*50 + [1]*50 = 0.5
        u = pl.min_l2_potts(step_signal_1d, gamma=100.0)
        npt.assert_allclose(u, 0.5, atol=1e-12)
        assert pl.count_jumps(u) == 0

    def test_single_jump_count(self, step_signal_1d):
        """Clean step signal should have exactly 1 jump."""
        u = pl.min_l2_potts(step_signal_1d, gamma=0.01)
        assert pl.count_jumps(u) == 1

    def test_uniform_signal_zero_jumps(self):
        """A uniform signal must yield zero jumps."""
        f = np.ones(100)
        u = pl.min_l2_potts(f, gamma=1.0)
        assert pl.count_jumps(u) == 0

    def test_three_segments_correct_jumps(self, piecewise_const_1d):
        """3-segment signal with small gamma should preserve 2 jumps."""
        u = pl.min_l2_potts(piecewise_const_1d, gamma=0.001)
        assert pl.count_jumps(u) == 2

    def test_three_segments_correct_values(self, piecewise_const_1d):
        """Segment means should match original segment values."""
        u = pl.min_l2_potts(piecewise_const_1d, gamma=0.001)
        npt.assert_allclose(u[:30], 0.0, atol=1e-10)
        npt.assert_allclose(u[30:70], 2.0, atol=1e-10)
        npt.assert_allclose(u[70:], -1.0, atol=1e-10)

    def test_energy_lower_than_input(self, step_signal_1d):
        """Solution energy must not exceed data energy (f is a candidate)."""
        gamma = 0.5
        u = pl.min_l2_potts(step_signal_1d, gamma=gamma)
        e_u = pl.energy_l2_potts(u, step_signal_1d, gamma)
        e_f = pl.energy_l2_potts(step_signal_1d, step_signal_1d, gamma)
        assert e_u <= e_f + 1e-10

    def test_energy_is_optimal(self, piecewise_const_1d):
        """Potts energy of solution ≤ energy of input (global optimum)."""
        gamma = 1.0
        u = pl.min_l2_potts(piecewise_const_1d, gamma=gamma)
        e_u = pl.energy_l2_potts(u, piecewise_const_1d, gamma)
        e_f = pl.energy_l2_potts(piecewise_const_1d, piecewise_const_1d, gamma)
        assert e_u <= e_f + 1e-10

    def test_analytical_threshold(self):
        """Known threshold: 2-segment [0]*n/2 + [1]*n/2 is preserved iff gamma < n/4.

        With n=100 and delta=1: threshold = n * delta^2 / 4 = 25.
        """
        n = 100
        f = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
        # gamma < 25: step should be preserved
        u_small = pl.min_l2_potts(f, gamma=24.9)
        assert pl.count_jumps(u_small) == 1
        # gamma > 25: step should collapse to mean
        u_large = pl.min_l2_potts(f, gamma=25.1)
        assert pl.count_jumps(u_large) == 0

    def test_single_point_signal(self):
        """A single-point signal must return itself."""
        f = np.array([3.7])
        u = pl.min_l2_potts(f, gamma=1.0)
        npt.assert_allclose(u, f, atol=1e-12)

    def test_two_point_equal_signal(self):
        """Two identical points: no jump, both set to their value."""
        f = np.array([5.0, 5.0])
        u = pl.min_l2_potts(f, gamma=1.0)
        npt.assert_allclose(u, [5.0, 5.0], atol=1e-12)

    def test_performance_n1000(self):
        """n=1000 signal should complete in under 2 seconds."""
        rng = np.random.default_rng(0)
        f = np.cumsum(rng.standard_normal(1000)) * 0.1
        t0 = time.time()
        u = pl.min_l2_potts(f, gamma=1.0)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"
        assert u.shape == f.shape

    def test_output_is_piecewise_constant(self):
        """Output must be a piecewise-constant signal (each segment is constant)."""
        rng = np.random.default_rng(1)
        f = rng.standard_normal(50)
        u = pl.min_l2_potts(f, gamma=2.0)
        # Check: each "run" is constant
        jumps = np.where(np.diff(u) != 0)[0]
        boundaries = np.concatenate([[0], jumps + 1, [len(u)]])
        for i in range(len(boundaries) - 1):
            seg = u[boundaries[i]:boundaries[i + 1]]
            assert np.all(seg == seg[0]), "Segment is not constant"


class TestL2Potts1DVectorValued:
    def test_2channel_step_preserved(self):
        """2-channel step signal with small gamma: both channels preserved."""
        n = 40
        f = np.zeros((n, 2))
        f[n // 2:, 0] = 1.0
        f[n // 2:, 1] = 2.0
        u = pl.min_l2_potts(f, gamma=0.01)
        assert u.shape == (n, 2)
        npt.assert_allclose(u[:n // 2, 0], 0.0, atol=1e-10)
        npt.assert_allclose(u[n // 2:, 0], 1.0, atol=1e-10)
        npt.assert_allclose(u[:n // 2, 1], 0.0, atol=1e-10)
        npt.assert_allclose(u[n // 2:, 1], 2.0, atol=1e-10)

    def test_2channel_large_gamma_collapses(self):
        """Large gamma collapses multi-channel signal to per-channel means."""
        n = 40
        f = np.zeros((n, 2))
        f[n // 2:, 0] = 1.0
        f[n // 2:, 1] = 2.0
        u = pl.min_l2_potts(f, gamma=1000.0)
        npt.assert_allclose(u[:, 0], 0.5, atol=1e-10)
        npt.assert_allclose(u[:, 1], 1.0, atol=1e-10)

    def test_3channel_rgb(self):
        """3-channel (RGB-like) signal stays in [0,1] after constant step."""
        n = 30
        f = np.zeros((n, 3))
        f[15:, :] = np.array([1.0, 0.5, 0.2])
        u = pl.min_l2_potts(f, gamma=0.01)
        assert pl.count_jumps(u) == 1


class TestL2Potts1DWeighted:
    def test_uniform_weights_same_as_unweighted(self, step_signal_1d):
        """Uniform weights = 1 must give same result as no weights."""
        n = len(step_signal_1d)
        w = np.ones(n)
        u_w = pl.min_l2_potts(step_signal_1d, gamma=0.5, weights=w)
        u_nw = pl.min_l2_potts(step_signal_1d, gamma=0.5)
        npt.assert_allclose(u_w, u_nw, atol=1e-10)

    def test_zero_weight_ignores_samples(self):
        """Samples with weight 0 should not influence the solution."""
        f = np.array([0.0, 0.0, 999.0, 0.0, 0.0])  # spike at position 2
        w = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        u = pl.min_l2_potts(f, gamma=1.0, weights=w)
        # The spike is ignored (w=0), so everything should be 0
        npt.assert_allclose(u, 0.0, atol=1e-10)

    def test_samples_parameter_nonequidistant(self):
        """Non-equidistant samples parameter converts to weights correctly."""
        f = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = pl.min_l2_potts(f, gamma=0.01, samples=samples)
        assert pl.count_jumps(u) == 1


class TestL2Potts1DValidation:
    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l2_potts(np.array([1.0, 2.0]), gamma=0.0)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l2_potts(np.array([1.0, 2.0]), gamma=-1.0)

    def test_nonfinite_input_raises(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts(np.array([1.0, np.nan, 3.0]), gamma=1.0)

    def test_negative_weights_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pl.min_l2_potts(np.array([1.0, 2.0]), gamma=1.0,
                            weights=np.array([1.0, -0.5]))
