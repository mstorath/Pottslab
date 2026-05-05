# Tests for the 1-D L1-Potts solver.
# Verifies faithfulness of the Rust port of IndexedLinkedHistogram.java.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestL1Potts1DBasic:
    def test_constant_signal_unchanged(self):
        """A constant signal is already optimal."""
        f = np.full(20, 2.5)
        u, _, nj, _ = pl.min_l1_potts(f, gamma=1.0)
        npt.assert_allclose(u, f, atol=1e-10)
        assert nj == 0

    def test_step_small_gamma_preserved(self, step_signal_1d):
        """Small gamma: step signal is preserved."""
        u, _, nj, _ = pl.min_l1_potts(step_signal_1d, gamma=0.01)
        assert nj == 1
        npt.assert_allclose(u[:50], 0.0, atol=1e-10)
        npt.assert_allclose(u[50:], 1.0, atol=1e-10)

    def test_step_large_gamma_collapses(self, step_signal_1d):
        """Large gamma: step collapses to median of signal."""
        u, _, nj, _ = pl.min_l1_potts(step_signal_1d, gamma=1000.0)
        # Median of [0]*50 + [1]*50 is any value in [0, 1] (here either 0 or 1 by definition)
        assert nj == 0
        assert u[0] == u[-1]  # all same value

    def test_returns_tuple_of_four(self, step_signal_1d):
        """min_l1_potts returns (u, data_error, n_jumps, energy)."""
        result = pl.min_l1_potts(step_signal_1d, gamma=1.0)
        assert len(result) == 4
        u, data_error, n_jumps, energy = result
        assert u.shape == step_signal_1d.shape
        assert data_error >= 0
        assert n_jumps >= 0
        assert energy >= 0

    def test_energy_formula(self, step_signal_1d):
        """energy == gamma * n_jumps + data_error."""
        gamma = 0.5
        u, data_error, n_jumps, energy = pl.min_l1_potts(step_signal_1d, gamma=gamma)
        expected_energy = gamma * n_jumps + data_error
        npt.assert_allclose(energy, expected_energy, rtol=1e-10)

    def test_energy_optimality(self, step_signal_1d):
        """Solution energy ≤ energy of original signal (global optimum)."""
        gamma = 1.0
        u, data_error_u, n_jumps_u, energy_u = pl.min_l1_potts(step_signal_1d, gamma=gamma)
        # Energy of f itself: gamma * count_jumps(f) + ||f - f||_1 = gamma * 1 + 0
        energy_f = gamma * pl.count_jumps(step_signal_1d)
        assert energy_u <= energy_f + 1e-10

    def test_outlier_robustness(self):
        """L1-Potts is robust to outliers — a single large spike shouldn't affect
        the rest of a constant region."""
        f = np.zeros(20)
        f[10] = 1000.0  # massive outlier
        u, _, nj, _ = pl.min_l1_potts(f, gamma=0.5)
        # The outlier should be isolated (or the rest kept near 0)
        non_outlier = np.concatenate([u[:10], u[11:]])
        # Either the spike is absorbed or suppressed; non-outlier region stays near 0
        assert np.mean(np.abs(non_outlier)) < 50.0  # much less than 1000

    def test_three_segments(self, piecewise_const_1d):
        """3-segment signal with small gamma preserves structure."""
        u, _, nj, _ = pl.min_l1_potts(piecewise_const_1d, gamma=0.01)
        assert nj == 2
        npt.assert_allclose(u[:30], 0.0, atol=1e-10)
        npt.assert_allclose(u[30:70], 2.0, atol=1e-10)
        npt.assert_allclose(u[70:], -1.0, atol=1e-10)

    def test_output_shape(self, step_signal_1d):
        """Output signal has same shape as input."""
        u, _, _, _ = pl.min_l1_potts(step_signal_1d, gamma=1.0)
        assert u.shape == step_signal_1d.shape


class TestL1Potts1DValidation:
    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l1_potts(np.array([1.0, 2.0]), gamma=0.0)

    def test_nonfinite_input_raises(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_potts(np.array([1.0, np.inf]), gamma=1.0)
