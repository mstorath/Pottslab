# Tests for Tikhonov regularisation solvers.
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestL2Tikhonov:
    def test_identity_operator(self):
        """A = I: solution is f / (1 + lam) * lam + f / (1 + lam) = f / (1 + lam)."""
        n = 10
        f = np.arange(1.0, n + 1.0)
        A = np.eye(n)
        lam = 1.0
        u, flag = pl.min_l2_tikhonov(f, lam, A)
        # Closed form: u = f / (1 + lam) for A=I
        expected = f / (1 + lam)
        npt.assert_allclose(u, expected, atol=1e-8)

    def test_diagonal_operator(self):
        """A = diag(d): (d_i^2 + lam) u_i = d_i f_i."""
        n = 5
        d = np.array([1.0, 2.0, 3.0, 0.5, 4.0])
        A = np.diag(d)
        f = np.ones(n)
        lam = 1.0
        u, flag = pl.min_l2_tikhonov(f, lam, A)
        expected = d * f / (d ** 2 + lam)
        npt.assert_allclose(u, expected, atol=1e-8)

    def test_returns_tuple(self):
        """Returns (u, flag) where flag=0 means convergence."""
        A = np.eye(5)
        f = np.ones(5)
        result = pl.min_l2_tikhonov(f, 1.0, A)
        assert len(result) == 2
        u, flag = result
        assert flag == 0


class TestL1Tikhonov:
    def test_identity_operator_smoketest(self):
        """A = I: result should be meaningful (no crash, finite output)."""
        n = 10
        f = np.sin(np.linspace(0, 2 * np.pi, n))
        A = np.eye(n)
        u, _ = pl.min_l1_tikhonov(f, 1.0, A)
        assert u.shape == (n,)
        assert np.all(np.isfinite(u))

    def test_returns_tuple(self):
        """Returns (u, flag)."""
        A = np.eye(5)
        f = np.ones(5)
        result = pl.min_l1_tikhonov(f, 1.0, A)
        assert len(result) == 2
