# Tests for sparsity regularisation (L0 minimisers).
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestL2Sparsity:
    def test_zero_signal_all_zeros(self):
        """Zero signal has no spikes."""
        f = np.zeros(20)
        u, _, ns, _ = pl.min_l2_spars(f, gamma=1.0)
        npt.assert_allclose(u, 0.0, atol=1e-12)
        assert ns == 0

    def test_small_values_suppressed(self):
        """Values with |f[i]| <= sqrt(gamma) are set to zero."""
        gamma = 1.0  # threshold = sqrt(1) = 1
        f = np.array([0.5, 0.9, 1.5, -2.0])
        u, _, _, _ = pl.min_l2_spars(f, gamma=gamma)
        assert u[0] == 0.0  # |0.5| < 1
        assert u[1] == 0.0  # |0.9| < 1
        npt.assert_allclose(u[2], 1.5, atol=1e-12)
        npt.assert_allclose(u[3], -2.0, atol=1e-12)

    def test_spike_count(self):
        """n_spikes counts non-zero entries in solution."""
        f = np.array([0.1, 3.0, 0.2, -4.0])
        u, _, ns, _ = pl.min_l2_spars(f, gamma=1.0)
        assert ns == int(np.sum(u != 0.0))

    def test_energy_formula(self):
        """energy == gamma * n_spikes + data_error."""
        f = np.array([0.5, 2.0, -0.3, 1.5])
        gamma = 1.0
        u, data_error, ns, energy = pl.min_l2_spars(f, gamma=gamma)
        npt.assert_allclose(energy, gamma * ns + data_error, rtol=1e-10)

    def test_energy_optimality(self):
        """Solution energy ≤ trivial energy (all zeros or all f)."""
        f = np.array([0.5, 2.0, -0.3, 1.5])
        gamma = 1.0
        u, _, ns, energy = pl.min_l2_spars(f, gamma=gamma)
        # Trivial candidate: u = 0  =>  energy = ||f||^2
        trivial_energy = float(np.sum(f ** 2))
        assert energy <= trivial_energy + 1e-10

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l2_spars(np.array([1.0, 2.0]), gamma=0.0)


class TestL1Sparsity:
    def test_zero_signal(self):
        """Zero signal → zero solution."""
        f = np.zeros(10)
        u, _, ns, _ = pl.min_l1_spars(f, gamma=1.0)
        npt.assert_allclose(u, 0.0, atol=1e-12)
        assert ns == 0

    def test_soft_threshold_relation(self):
        """L1-sparsity uses soft thresholding with tau = gamma/2."""
        f = np.array([1.0, 0.2, -0.8, 3.0])
        gamma = 1.0
        u, _, _, _ = pl.min_l1_spars(f, gamma=gamma)
        expected = pl.soft_threshold(f, gamma / 2)
        npt.assert_allclose(u, expected, atol=1e-12)

    def test_energy_formula(self):
        """energy == gamma * n_spikes + ||u - f||_1."""
        f = np.array([0.5, 2.0, -0.3, 1.5])
        gamma = 0.5
        u, data_error, ns, energy = pl.min_l1_spars(f, gamma=gamma)
        npt.assert_allclose(energy, gamma * ns + data_error, rtol=1e-10)
        npt.assert_allclose(data_error, np.sum(np.abs(u - f)), atol=1e-12)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l1_spars(np.array([1.0]), gamma=0.0)
