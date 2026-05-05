# Property-based / invariant tests for pottslab.
# These check mathematical properties that must hold regardless of the input,
# without comparing to any reference output.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# Idempotency: applying the solver twice gives the same result
# ---------------------------------------------------------------------------

class TestIdempotency:
    """u* = argmin E  =>  argmin E(u*) = u*  (u* is already optimal)."""

    @pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
    def test_l2_potts_idempotent(self, gamma):
        rng = np.random.default_rng(0)
        f = rng.standard_normal(30)
        u1 = pl.min_l2_potts(f, gamma)
        u2 = pl.min_l2_potts(u1, gamma)
        # Segment means recomputed from already-piecewise-constant input may
        # differ by machine epsilon due to floating-point accumulation order.
        npt.assert_allclose(u1, u2, atol=1e-13)

    @pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
    def test_l1_potts_idempotent(self, gamma):
        rng = np.random.default_rng(1)
        f = rng.standard_normal(20)
        u1, *_ = pl.min_l1_potts(f, gamma)
        u2, *_ = pl.min_l1_potts(u1, gamma)
        npt.assert_array_equal(u1, u2)

    def test_l2_potts_vv_idempotent(self):
        rng = np.random.default_rng(2)
        f = rng.standard_normal((20, 3))
        u1 = pl.min_l2_potts(f, gamma=1.0)
        u2 = pl.min_l2_potts(u1, gamma=1.0)
        npt.assert_allclose(u1, u2, atol=1e-13)

    def test_l2_spars_idempotent(self):
        rng = np.random.default_rng(3)
        f = rng.standard_normal(25)
        u1, *_ = pl.min_l2_spars(f, gamma=1.0)
        u2, *_ = pl.min_l2_spars(u1, gamma=1.0)
        npt.assert_array_equal(u1, u2)

    def test_l1_spars_zero_is_fixed_point(self):
        """Zero elements stay zero: soft threshold of 0 is always 0."""
        rng = np.random.default_rng(4)
        f = rng.standard_normal(25)
        u1, *_ = pl.min_l1_spars(f, gamma=1.0)
        # Positions set to zero are fixed points: re-applying keeps them zero
        npt.assert_array_equal(u1[u1 == 0.0], 0.0)


# ---------------------------------------------------------------------------
# Monotonicity: fewer jumps as gamma increases
# ---------------------------------------------------------------------------

class TestMonotonicityInGamma:
    """Increasing gamma can only reduce or preserve the number of jumps."""

    def test_l2_potts_jump_count_nonincreasing(self):
        rng = np.random.default_rng(5)
        f = rng.standard_normal(50)
        gammas = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0]
        jump_counts = [pl.count_jumps(pl.min_l2_potts(f, g)) for g in gammas]
        for i in range(len(jump_counts) - 1):
            assert jump_counts[i] >= jump_counts[i + 1], (
                f"Jumps increased: {jump_counts[i]} -> {jump_counts[i+1]} "
                f"at gamma={gammas[i]:.3f} -> {gammas[i+1]:.3f}"
            )

    def test_l1_potts_jump_count_nonincreasing(self):
        rng = np.random.default_rng(6)
        f = rng.standard_normal(30)
        gammas = [0.01, 0.2, 1.0, 5.0, 50.0]
        jump_counts = [pl.min_l1_potts(f, g)[2] for g in gammas]
        for i in range(len(jump_counts) - 1):
            assert jump_counts[i] >= jump_counts[i + 1]

    def test_l2_potts_large_gamma_is_constant(self, step_signal_1d):
        """With very large gamma every signal collapses to a constant."""
        u = pl.min_l2_potts(step_signal_1d, gamma=1e9)
        assert pl.count_jumps(u) == 0

    def test_l2_potts_small_gamma_preserves_all(self):
        """With gamma → 0, every distinct value is its own segment."""
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = pl.min_l2_potts(f, gamma=1e-9)
        npt.assert_allclose(u, f, atol=1e-10)
        assert pl.count_jumps(u) == 4

    def test_l2_spars_spike_count_nonincreasing(self):
        rng = np.random.default_rng(7)
        f = rng.standard_normal(40)
        gammas = [0.01, 0.1, 1.0, 5.0, 20.0]
        spike_counts = [pl.min_l2_spars(f, g)[2] for g in gammas]
        for i in range(len(spike_counts) - 1):
            assert spike_counts[i] >= spike_counts[i + 1]


# ---------------------------------------------------------------------------
# Energy bounds and consistency
# ---------------------------------------------------------------------------

class TestEnergyProperties:
    """The Potts energy must satisfy basic mathematical properties."""

    def test_l2_solution_energy_le_zero_solution(self, step_signal_1d):
        """Energy of u* ≤ energy of the all-zeros candidate."""
        gamma = 1.0
        u = pl.min_l2_potts(step_signal_1d, gamma)
        e_u = pl.energy_l2_potts(u, step_signal_1d, gamma)
        u_zeros = np.zeros_like(step_signal_1d)
        e_zeros = pl.energy_l2_potts(u_zeros, step_signal_1d, gamma)
        assert e_u <= e_zeros + 1e-10

    def test_l2_solution_energy_le_mean_solution(self, step_signal_1d):
        """Energy of u* ≤ energy of the constant-mean solution."""
        gamma = 0.5
        u = pl.min_l2_potts(step_signal_1d, gamma)
        e_u = pl.energy_l2_potts(u, step_signal_1d, gamma)
        u_mean = np.full_like(step_signal_1d, step_signal_1d.mean())
        e_mean = pl.energy_l2_potts(u_mean, step_signal_1d, gamma)
        assert e_u <= e_mean + 1e-10

    def test_energy_nonnegative(self):
        rng = np.random.default_rng(8)
        f = rng.standard_normal(20)
        u = pl.min_l2_potts(f, gamma=1.0)
        assert pl.energy_l2_potts(u, f, gamma=1.0) >= -1e-12

    def test_l1_energy_nonnegative(self):
        rng = np.random.default_rng(9)
        f = rng.standard_normal(20)
        u, data_err, nj, energy = pl.min_l1_potts(f, gamma=1.0)
        assert energy >= -1e-12
        assert data_err >= -1e-12

    def test_njumps_consistent_with_count_jumps(self, step_signal_1d):
        """n_jumps field in L1-Potts tuple == count_jumps(u)."""
        u, _, nj, _ = pl.min_l1_potts(step_signal_1d, gamma=0.5)
        assert nj == pl.count_jumps(u)

    @pytest.mark.parametrize("seed", range(5))
    def test_sparsity_energy_consistent(self, seed):
        rng = np.random.default_rng(seed + 200)
        f = rng.standard_normal(15)
        gamma = rng.uniform(0.2, 2.0)
        u, data_err, ns, energy = pl.min_l2_spars(f, gamma)
        npt.assert_allclose(energy, gamma * ns + data_err, rtol=1e-10)
        npt.assert_allclose(data_err, np.sum((u - f) ** 2), rtol=1e-10)


# ---------------------------------------------------------------------------
# Determinism: same input → same output every time
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_l2_potts_deterministic(self):
        f = np.random.default_rng(0).standard_normal(100)
        u1 = pl.min_l2_potts(f.copy(), gamma=1.0)
        u2 = pl.min_l2_potts(f.copy(), gamma=1.0)
        npt.assert_array_equal(u1, u2)

    def test_l1_potts_deterministic(self):
        f = np.random.default_rng(1).standard_normal(50)
        u1, *_ = pl.min_l1_potts(f.copy(), gamma=1.0)
        u2, *_ = pl.min_l1_potts(f.copy(), gamma=1.0)
        npt.assert_array_equal(u1, u2)

    def test_potts_2d_deterministic(self, step_image_2d):
        u1 = pl.min_l2_potts_2d(step_image_2d.copy(), gamma=0.01, tol=1e-6)
        u2 = pl.min_l2_potts_2d(step_image_2d.copy(), gamma=0.01, tol=1e-6)
        npt.assert_array_equal(u1, u2)


# ---------------------------------------------------------------------------
# Scale and shift equivariance
# ---------------------------------------------------------------------------

class TestEquivariance:
    """The Potts functional is equivariant under affine transforms."""

    def test_l2_potts_shift_equivariant(self):
        """u*(f + c) = u*(f) + c for any constant c."""
        rng = np.random.default_rng(10)
        f = rng.standard_normal(30)
        gamma = 1.0
        c = 5.0
        u = pl.min_l2_potts(f, gamma)
        u_shifted = pl.min_l2_potts(f + c, gamma)
        npt.assert_allclose(u_shifted, u + c, atol=1e-10)

    def test_l2_potts_scale_equivariant(self):
        """u*(alpha * f) = alpha * u*(f) and jumps at same positions."""
        rng = np.random.default_rng(11)
        f = rng.standard_normal(30)
        alpha = 3.0
        # Scale gamma by alpha^2 to keep the same optimal partition
        u = pl.min_l2_potts(f, gamma=1.0)
        u_scaled = pl.min_l2_potts(alpha * f, gamma=alpha ** 2)
        npt.assert_allclose(u_scaled, alpha * u, atol=1e-9)

    def test_l2_potts_negate_equivariant(self):
        """u*(-f) = -u*(f)."""
        rng = np.random.default_rng(12)
        f = rng.standard_normal(25)
        gamma = 1.5
        u = pl.min_l2_potts(f, gamma)
        u_neg = pl.min_l2_potts(-f, gamma)
        npt.assert_allclose(u_neg, -u, atol=1e-10)

    def test_l1_potts_shift_equivariant(self):
        rng = np.random.default_rng(13)
        f = rng.standard_normal(20)
        gamma, c = 1.0, 7.0
        u, *_ = pl.min_l1_potts(f, gamma)
        u_s, *_ = pl.min_l1_potts(f + c, gamma)
        npt.assert_allclose(u_s, u + c, atol=1e-10)


# ---------------------------------------------------------------------------
# Output is piecewise constant (segment structure)
# ---------------------------------------------------------------------------

class TestPiecewiseConstant:
    def _check_piecewise_constant(self, u):
        jumps = np.where(np.diff(u.ravel()) != 0)[0]
        bounds = np.concatenate([[0], jumps + 1, [len(u.ravel())]])
        for k in range(len(bounds) - 1):
            seg = u.ravel()[bounds[k]:bounds[k + 1]]
            assert np.all(seg == seg[0])

    @pytest.mark.parametrize("gamma", [0.05, 0.5, 5.0])
    def test_l2_output_piecewise_constant(self, gamma):
        rng = np.random.default_rng(14)
        f = rng.standard_normal(40)
        u = pl.min_l2_potts(f, gamma)
        self._check_piecewise_constant(u)

    @pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
    def test_l1_output_piecewise_constant(self, gamma):
        rng = np.random.default_rng(15)
        f = rng.standard_normal(30)
        u, *_ = pl.min_l1_potts(f, gamma)
        self._check_piecewise_constant(u)

    def test_l2_multichannel_output_piecewise_constant(self):
        rng = np.random.default_rng(16)
        f = rng.standard_normal((25, 3))
        u = pl.min_l2_potts(f, gamma=1.0)
        # Jump at position i means ALL channels change simultaneously
        row_changed = np.any(np.diff(u, axis=0) != 0, axis=1)
        jumps = np.where(row_changed)[0]
        bounds = np.concatenate([[0], jumps + 1, [len(u)]])
        for k in range(len(bounds) - 1):
            seg = u[bounds[k]:bounds[k + 1], :]
            # Every row in the segment must equal the first row
            npt.assert_array_equal(seg, np.tile(seg[[0], :], (len(seg), 1)))
