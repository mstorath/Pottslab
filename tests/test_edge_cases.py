# Edge-case tests for pottslab.
# Covers unusual inputs: tiny arrays, degenerate signals, extreme parameters,
# type coercion, non-square images, and boundary conditions.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# L2-Potts 1-D edge cases
# ---------------------------------------------------------------------------

class TestL2Potts1DEdgeCases:
    def test_n2_step(self):
        """n=2 with a jump: correct segment means returned."""
        f = np.array([0.0, 1.0])
        # With gamma < 0.25 the step is preserved (cost = 0 + gamma < 0.25)
        u = pl.min_l2_potts(f, gamma=0.1)
        npt.assert_allclose(u, f, atol=1e-10)

    def test_n2_same_collapses(self):
        """n=2 identical: no jump regardless of gamma."""
        f = np.array([3.0, 3.0])
        u = pl.min_l2_potts(f, gamma=0.001)
        npt.assert_allclose(u, [3.0, 3.0], atol=1e-12)
        assert pl.count_jumps(u) == 0

    def test_integer_input_coerced(self):
        """Integer array input is correctly coerced to float64."""
        f = np.array([0, 0, 0, 1, 1, 1])
        u = pl.min_l2_potts(f, gamma=0.1)
        assert u.dtype == np.float64
        assert pl.count_jumps(u) == 1

    def test_float32_input_coerced(self):
        f = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        u = pl.min_l2_potts(f, gamma=0.1)
        assert u.dtype == np.float64

    def test_staircase_signal(self):
        """Monotonically increasing staircase: each step may or may not be preserved."""
        f = np.repeat([0.0, 1.0, 2.0, 3.0], 10)
        u = pl.min_l2_potts(f, gamma=0.5)
        assert pl.count_jumps(u) <= 3  # can't gain jumps vs input
        assert pl.count_jumps(u) >= 0

    def test_all_negative_values(self):
        """Signal entirely in negative range should work correctly."""
        f = np.array([-5.0] * 10 + [-2.0] * 10)
        u = pl.min_l2_potts(f, gamma=0.1)
        npt.assert_allclose(u[:10], -5.0, atol=1e-10)
        npt.assert_allclose(u[10:], -2.0, atol=1e-10)

    def test_very_large_values(self):
        """Large magnitude values should not overflow."""
        f = np.array([1e8, 1e8, 2e8, 2e8])
        u = pl.min_l2_potts(f, gamma=1e10)
        assert np.all(np.isfinite(u))

    def test_single_channel_explicit_2d(self):
        """(n, 1) array is treated as vector-valued with one channel."""
        f_1d = np.array([0.0, 0.0, 1.0, 1.0])
        f_2d = f_1d[:, np.newaxis]
        u_1d = pl.min_l2_potts(f_1d, gamma=0.1)
        u_2d = pl.min_l2_potts(f_2d, gamma=0.1)
        npt.assert_allclose(u_2d[:, 0], u_1d, atol=1e-12)

    def test_output_does_not_mutate_input(self):
        """The solver must not modify the input array."""
        f = np.array([0.0, 0.0, 1.0, 1.0])
        f_copy = f.copy()
        pl.min_l2_potts(f, gamma=0.1)
        npt.assert_array_equal(f, f_copy)

    def test_high_gamma_returns_mean(self):
        """With very high gamma, output is mean of f repeated n times."""
        f = np.arange(10, dtype=float)
        u = pl.min_l2_potts(f, gamma=1e12)
        npt.assert_allclose(u, f.mean(), atol=1e-8)

    def test_two_channels_jumps_at_same_positions(self):
        """Multi-channel output: both channels jump at the same positions."""
        rng = np.random.default_rng(20)
        f = rng.standard_normal((20, 2))
        u = pl.min_l2_potts(f, gamma=1.0)
        # Jumps in ch0 and ch1 must coincide
        jumps_0 = set(np.where(np.diff(u[:, 0]) != 0)[0])
        jumps_1 = set(np.where(np.diff(u[:, 1]) != 0)[0])
        assert jumps_0 == jumps_1


# ---------------------------------------------------------------------------
# L1-Potts 1-D edge cases
# ---------------------------------------------------------------------------

class TestL1Potts1DEdgeCases:
    def test_n2_step_preserved(self):
        f = np.array([0.0, 1.0])
        u, _, nj, _ = pl.min_l1_potts(f, gamma=0.1)
        assert nj == 1

    def test_output_does_not_mutate_input(self):
        f = np.array([0.0, 0.0, 1.0, 1.0])
        f_copy = f.copy()
        pl.min_l1_potts(f, gamma=0.5)
        npt.assert_array_equal(f, f_copy)

    def test_all_same_values_zero_error(self):
        f = np.full(10, 2.5)
        u, data_err, nj, energy = pl.min_l1_potts(f, gamma=1.0)
        npt.assert_allclose(data_err, 0.0, atol=1e-12)
        assert nj == 0

    def test_single_outlier_absorbed_l1(self):
        """L1: outlier is absorbed into its own segment or the region stays near 0."""
        f = np.zeros(10)
        f[5] = 50.0
        u, _, nj, _ = pl.min_l1_potts(f, gamma=0.5)
        # Either the outlier forms its own segment (nj=2) or is part of a region
        # Either way, the non-outlier mean should stay close to 0
        non_outlier = np.concatenate([u[:5], u[6:]])
        assert np.mean(np.abs(non_outlier)) < 5.0

    def test_weighted_step_signal(self):
        """Weighted L1-Potts: double-weighting one half should not move the other."""
        f = np.array([0.0] * 6 + [1.0] * 6)
        # Emphasise the second half with high weight
        w = np.array([1.0] * 6 + [5.0] * 6)
        u, _, nj, _ = pl.min_l1_potts(f, gamma=0.01, weights=w)
        assert nj == 1
        npt.assert_allclose(u[:6], 0.0, atol=1e-8)
        npt.assert_allclose(u[6:], 1.0, atol=1e-8)

    def test_integer_input_coerced(self):
        f = np.array([0, 0, 1, 1])
        u, *_ = pl.min_l1_potts(f, gamma=0.1)
        assert u.dtype == np.float64


# ---------------------------------------------------------------------------
# 2-D Potts edge cases
# ---------------------------------------------------------------------------

class TestPotts2DEdgeCases:
    def test_non_square_image_tall(self):
        """Tall image (more rows than columns) should work."""
        f = np.zeros((64, 8), dtype=np.float64)
        f[:, 4:] = 1.0
        u = pl.min_l2_potts_2d(f, gamma=0.001, isotropic=False,
                                tol=1e-6, quantize=False)
        assert u.shape == (64, 8)
        assert np.mean(u[:, :4]) < 0.2
        assert np.mean(u[:, 4:]) > 0.8

    def test_non_square_image_wide(self):
        """Wide image (more columns than rows) should work."""
        f = np.zeros((8, 64), dtype=np.float64)
        f[4:, :] = 1.0
        u = pl.min_l2_potts_2d(f, gamma=0.001, isotropic=False,
                                tol=1e-6, quantize=False)
        assert u.shape == (8, 64)
        assert np.mean(u[:4, :]) < 0.2
        assert np.mean(u[4:, :]) > 0.8

    def test_small_4x4_image(self):
        """Tiny images should not crash."""
        f = np.array([[0.0, 0.0, 1.0, 1.0]] * 4)
        u = pl.min_l2_potts_2d(f, gamma=0.001, tol=1e-6, quantize=False)
        assert u.shape == (4, 4)

    def test_single_row_image(self):
        """A (1, n) image reduces to a 1-D problem."""
        f = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        u = pl.min_l2_potts_2d(f, gamma=0.001, tol=1e-6, quantize=False)
        assert u.shape == (1, 6)

    def test_single_column_image(self):
        """A (n, 1) image reduces to a 1-D problem."""
        f = np.zeros((8, 1))
        f[4:, :] = 1.0
        u = pl.min_l2_potts_2d(f, gamma=0.001, tol=1e-6, quantize=False)
        assert u.shape == (8, 1)

    def test_large_gamma_constant_output(self):
        """Very large gamma should collapse image to a constant."""
        rng = np.random.default_rng(30)
        f = rng.random((16, 16))
        u = pl.min_l2_potts_2d(f, gamma=1e6, quantize=False)
        # All values should be (nearly) the same
        assert u.max() - u.min() < 1e-4

    def test_4quadrant_image(self):
        """Image with 4 distinct quadrants should segment into ≥ 2 regions."""
        f = np.zeros((32, 32))
        f[:16, 16:] = 0.5
        f[16:, :16] = 0.7
        f[16:, 16:] = 1.0
        u = pl.min_l2_potts_2d(f, gamma=0.0001, tol=1e-8, quantize=False)
        unique_vals = np.unique(np.round(u, 3))
        assert len(unique_vals) >= 2

    def test_custom_mu_init(self, step_image_2d):
        """Custom mu_init should not crash and still converge."""
        u = pl.min_l2_potts_2d(step_image_2d, gamma=0.01, mu_init=0.001,
                                tol=1e-6, quantize=False)
        assert u.shape == step_image_2d.shape
        assert np.all(np.isfinite(u))

    def test_verbose_does_not_crash(self, step_image_2d, capsys):
        """verbose=True should emit output but not raise."""
        pl.min_l2_potts_2d(step_image_2d, gamma=0.01, verbose=True,
                            tol=1e-6, quantize=False)
        # No assertion on output content — just verify no exception

    def test_nonnegative_weights(self, step_image_2d):
        """Custom per-pixel weights emphasise certain regions."""
        h, w = step_image_2d.shape
        weights = np.ones((h, w))
        weights[:, :32] = 2.0  # emphasise left half
        u = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, weights=weights,
                                tol=1e-6, quantize=False)
        assert u.shape == step_image_2d.shape
        assert np.all(np.isfinite(u))

    def test_explicit_3d_input_single_channel(self):
        """Input (H, W, 1) should be accepted and returned as (H, W, 1)."""
        f = np.zeros((16, 16, 1))
        f[:, 8:, :] = 1.0
        u = pl.min_l2_potts_2d(f, gamma=0.001, tol=1e-6, quantize=False)
        assert u.shape == (16, 16, 1)

    def test_quantize_values_are_multiples_of_1_over_255(self):
        """With quantize=True, all values should be k/255 for integer k."""
        f = np.random.default_rng(31).random((16, 16))
        u = pl.min_l2_potts_2d(f, gamma=0.1, quantize=True)
        residuals = (u * 255) - np.round(u * 255)
        npt.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Sparsity edge cases
# ---------------------------------------------------------------------------

class TestSparsityEdgeCases:
    def test_all_below_threshold_l2(self):
        """All values below sqrt(gamma): all zeros."""
        gamma = 4.0  # threshold = sqrt(4) = 2
        f = np.array([0.5, 1.0, -1.5, 0.8])
        u, _, ns, _ = pl.min_l2_spars(f, gamma)
        npt.assert_allclose(u, 0.0, atol=1e-12)
        assert ns == 0

    def test_all_above_threshold_l2(self):
        """All values above sqrt(gamma): all preserved."""
        gamma = 0.01  # threshold = 0.1
        f = np.array([1.0, -2.0, 3.0, -4.0])
        u, _, ns, _ = pl.min_l2_spars(f, gamma)
        npt.assert_allclose(u, f, atol=1e-12)
        assert ns == 4

    def test_negative_spikes_preserved_l2(self):
        gamma = 1.0
        f = np.array([-3.0, 0.0, 0.0, -2.5])
        u, _, _, _ = pl.min_l2_spars(f, gamma)
        npt.assert_allclose(u[0], -3.0, atol=1e-12)
        npt.assert_allclose(u[-1], -2.5, atol=1e-12)

    def test_l1_spars_large_gamma_zeros(self):
        """Very large gamma/2 > all |f[i]|: soft threshold collapses to 0."""
        f = np.array([0.5, -0.3, 0.1])
        u, *_ = pl.min_l1_spars(f, gamma=2.0)  # tau = 1.0 > all |f_i|
        npt.assert_allclose(u, 0.0, atol=1e-12)

    def test_output_shape_preserved(self):
        f = np.arange(10, dtype=float)
        u, *_ = pl.min_l2_spars(f, gamma=1.0)
        assert u.shape == (10,)
        u1, *_ = pl.min_l1_spars(f, gamma=1.0)
        assert u1.shape == (10,)


# ---------------------------------------------------------------------------
# Utility edge cases
# ---------------------------------------------------------------------------

class TestUtilityEdgeCases:
    def test_count_jumps_empty(self):
        """Empty array: no jumps."""
        assert pl.count_jumps(np.array([])) == 0

    def test_soft_threshold_zero_tau(self):
        """tau=0: soft_threshold is identity."""
        x = np.array([1.0, -2.0, 0.5])
        npt.assert_allclose(pl.soft_threshold(x, 0.0), x, atol=1e-12)

    def test_soft_threshold_large_tau(self):
        """tau > |x|: all zeros."""
        x = np.array([0.5, -0.3, 0.1])
        npt.assert_allclose(pl.soft_threshold(x, 1.0), 0.0, atol=1e-12)

    def test_weighted_median_all_weight_on_one(self):
        """If one element has all the weight, it is the median."""
        data = np.array([1.0, 2.0, 3.0])
        w = np.array([0.0, 1.0, 0.0])
        assert pl.weighted_median(data, w) == 2.0

    def test_samples_to_weights_two_samples(self):
        """Two samples: each gets half the distance."""
        s = np.array([0.0, 4.0])
        w = pl.samples_to_weights(s)
        npt.assert_allclose(w, [2.0, 2.0], atol=1e-12)

    def test_samples_to_weights_non_uniform(self):
        """Non-uniform spacing: mid-point weights differ."""
        s = np.array([0.0, 1.0, 3.0])  # gap sizes: 1, 2
        w = pl.samples_to_weights(s)
        assert w[0] < w[1]  # interior weight larger

    def test_energy_l2_potts_with_operator(self):
        """energy_l2_potts with A: computes ||Au - f||^2 + gamma * jumps."""
        n = 6
        A = 2.0 * np.eye(n)  # simple scaling
        f = np.ones(n)
        u = np.ones(n)
        gamma = 1.0
        # A @ u = 2*u = 2, f = 1, so data error = sum((2-1)^2) = n
        e = pl.energy_l2_potts(u, f, gamma, A=A)
        npt.assert_allclose(e, n, atol=1e-10)

    def test_seg_to_label_constant_image(self):
        """Constant image → single label."""
        f = np.ones((8, 8))
        labels = pl.seg_to_label(f)
        assert len(np.unique(labels)) <= 2  # background (0) + one region

    def test_seg_to_label_two_regions(self):
        """Step image → two distinct labelled regions."""
        f = np.zeros((16, 16))
        f[:, 8:] = 1.0
        labels = pl.seg_to_label(f)
        unique = np.unique(labels[labels > 0])  # ignore 0 = background
        assert len(unique) == 2

    def test_count_jumps_result_of_l2_potts(self):
        """count_jumps(min_l2_potts(f)) == count_jumps of the returned array."""
        f = np.array([0.0] * 10 + [1.0] * 10 + [0.5] * 10)
        u = pl.min_l2_potts(f, gamma=0.01)
        assert pl.count_jumps(u) == 2  # three piecewise-constant regions


# ---------------------------------------------------------------------------
# Tikhonov edge cases
# ---------------------------------------------------------------------------

class TestTikhonovEdgeCases:
    def test_underdetermined_system(self):
        """m < n (more unknowns than observations): solver must not crash."""
        m, n = 5, 10
        rng = np.random.default_rng(40)
        A = rng.standard_normal((m, n))
        f = rng.standard_normal(m)
        u, flag = pl.min_l2_tikhonov(f, lam=1.0, A=A)
        assert u.shape == (n,)
        assert np.all(np.isfinite(u))

    def test_overdetermined_system(self):
        """m > n (more observations than unknowns)."""
        m, n = 20, 5
        rng = np.random.default_rng(41)
        A = rng.standard_normal((m, n))
        f = rng.standard_normal(m)
        u, flag = pl.min_l2_tikhonov(f, lam=1.0, A=A)
        assert u.shape == (n,)
        assert np.all(np.isfinite(u))

    def test_initial_guess_accepted(self):
        """init parameter is accepted and result is finite."""
        A = np.eye(5)
        f = np.ones(5)
        init = np.zeros(5)
        u, _ = pl.min_l2_tikhonov(f, lam=1.0, A=A, init=init)
        assert np.all(np.isfinite(u))

    def test_large_lambda_near_zero(self):
        """Very large lambda: solution → 0 (heavily penalised)."""
        A = np.eye(5)
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u, _ = pl.min_l2_tikhonov(f, lam=1e8, A=A)
        npt.assert_allclose(u, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Inverse Potts edge cases
# ---------------------------------------------------------------------------

class TestInversePottsEdgeCases:
    def test_verbose_does_not_crash(self, step_signal_1d):
        """verbose=True should not raise."""
        n = len(step_signal_1d)
        pl.min_l2_ipotts(step_signal_1d, gamma=1.0, A=np.eye(n),
                         tol=1e-4, verbose=True)

    def test_wrong_a_shape_raises(self, step_signal_1d):
        """A with wrong number of columns raises ValueError."""
        with pytest.raises(ValueError):
            pl.min_l2_ipotts(step_signal_1d, gamma=1.0,
                             A=np.eye(len(step_signal_1d) + 1))

    def test_l1_ipotts_identity_finite(self, step_signal_1d):
        """L1-inverse with identity operator returns finite solution."""
        n = len(step_signal_1d)
        u, data_err, nj, energy = pl.min_l1_ipotts(step_signal_1d, gamma=1.0,
                                                     A=np.eye(n), tol=1e-4)
        assert np.all(np.isfinite(u))
        assert data_err >= 0
        assert nj >= 0
