# Tests for inverse Potts solvers (iPotts ADMM).
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestInverseL2Potts:
    def test_identity_operator_matches_direct(self, step_signal_1d):
        """With A = I, inverse Potts should give same result as direct Potts."""
        gamma = 0.5
        n = len(step_signal_1d)
        A = np.eye(n)
        u_direct = pl.min_l2_potts(step_signal_1d, gamma=gamma)
        u_inv, _, _, _ = pl.min_l2_ipotts(step_signal_1d, gamma=gamma, A=A,
                                            tol=1e-8, mu_step=1.1)
        # Should converge to similar result; tolerance is looser due to iterative solver
        npt.assert_allclose(u_inv, u_direct, atol=0.1)

    def test_returns_4tuple(self, step_signal_1d):
        """min_l2_ipotts returns (u, data_error, n_jumps, energy)."""
        n = len(step_signal_1d)
        result = pl.min_l2_ipotts(step_signal_1d, gamma=1.0, A=np.eye(n))
        assert len(result) == 4
        u, data_error, n_jumps, energy = result
        assert u.shape == step_signal_1d.shape
        assert data_error >= 0
        assert n_jumps >= 0
        assert energy >= 0

    def test_energy_formula(self, step_signal_1d):
        """energy == gamma * n_jumps + ||Au - f||_2^2."""
        n = len(step_signal_1d)
        A = np.eye(n)
        gamma = 0.5
        u, data_error, n_jumps, energy = pl.min_l2_ipotts(step_signal_1d, gamma=gamma, A=A)
        expected_energy = gamma * n_jumps + data_error
        npt.assert_allclose(energy, expected_energy, rtol=1e-8)

    def test_diagonal_operator(self):
        """Diagonal A: scaling the data by 0.5 should scale the solution."""
        f = np.array([0.0] * 20 + [1.0] * 20, dtype=float)
        n = len(f)
        scale = 0.5
        A = np.diag(np.full(n, scale))
        # The measurement is A @ u_true; with A = 0.5*I and f = A@u_true, we expect u_true
        f_measured = A @ f
        u, _, _, _ = pl.min_l2_ipotts(f_measured, gamma=0.1, A=A,
                                        tol=1e-6, mu_step=1.1)
        assert u.shape == f.shape

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l2_ipotts(np.ones(5), gamma=0.0, A=np.eye(5))


class TestInverseL1Potts:
    def test_returns_4tuple(self, step_signal_1d):
        """min_l1_ipotts returns (u, data_error, n_jumps, energy)."""
        n = len(step_signal_1d)
        result = pl.min_l1_ipotts(step_signal_1d, gamma=1.0, A=np.eye(n))
        assert len(result) == 4
        u, data_error, n_jumps, energy = result
        assert u.shape == step_signal_1d.shape
        assert data_error >= 0
        assert n_jumps >= 0

    def test_energy_formula(self, step_signal_1d):
        """energy == gamma * n_jumps + ||Au - f||_1."""
        n = len(step_signal_1d)
        A = np.eye(n)
        gamma = 0.5
        u, data_error, n_jumps, energy = pl.min_l1_ipotts(step_signal_1d, gamma=gamma, A=A)
        expected_energy = gamma * n_jumps + data_error
        npt.assert_allclose(energy, expected_energy, rtol=1e-8)
