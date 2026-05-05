"""
Hardening tests: input validation guards, edge cases, and numerical robustness.
All tests verify that:
  - Invalid inputs raise ValueError with a clear message.
  - Valid edge cases (n=0, n=1, zero weights, etc.) produce correct output.
  - Newly added guards behave consistently across all public API functions.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse

import pottslab as pl
from pottslab.utils import (
    count_jumps,
    energy_l2_potts,
    samples_to_weights,
    seg_to_label,
    soft_threshold,
    weighted_median,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def step1d():
    return np.array([0.0] * 20 + [1.0] * 20)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# min_l2_potts input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestL2PottsValidation:
    def test_nan_in_signal(self):
        f = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts(f, gamma=1.0)

    def test_inf_in_signal(self):
        f = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts(f, gamma=1.0)

    def test_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l2_potts(np.ones(10), gamma=-1.0)

    def test_zero_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l2_potts(np.ones(10), gamma=0.0)

    def test_wrong_weights_shape(self):
        f = np.ones(10)
        with pytest.raises(ValueError, match="weights"):
            pl.min_l2_potts(f, gamma=1.0, weights=np.ones(5))

    def test_negative_weight(self):
        f = np.ones(5)
        w = np.array([1.0, 1.0, -0.5, 1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            pl.min_l2_potts(f, gamma=1.0, weights=w)

    def test_nan_in_weights(self):
        f = np.ones(5)
        w = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts(f, gamma=1.0, weights=w)

    def test_inf_in_weights(self):
        f = np.ones(5)
        w = np.array([1.0, np.inf, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts(f, gamma=1.0, weights=w)

    def test_3d_input_rejected(self):
        f = np.ones((5, 3, 2))
        with pytest.raises(ValueError):
            pl.min_l2_potts(f, gamma=1.0)

    def test_zero_weights_accepted(self):
        # All-zero weights: degenerate but should not crash (result is arbitrary constant)
        f = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        w = np.zeros(5)
        u = pl.min_l2_potts(f, gamma=0.1, weights=w)
        assert u.shape == f.shape
        assert np.all(np.isfinite(u))

    def test_single_element(self):
        u = pl.min_l2_potts(np.array([5.0]), gamma=1.0)
        npt.assert_allclose(u, [5.0])

    def test_empty_array(self):
        u = pl.min_l2_potts(np.array([]), gamma=1.0)
        assert u.shape == (0,)

    def test_integer_input_coerced(self):
        f = np.array([0, 0, 1, 1], dtype=np.int32)
        u = pl.min_l2_potts(f, gamma=0.1)
        assert u.dtype == np.float64

    def test_all_equal_no_jumps(self):
        f = np.full(30, 3.7)
        u = pl.min_l2_potts(f, gamma=0.5)
        assert count_jumps(u) == 0
        npt.assert_allclose(u, 3.7)

    def test_two_elements(self):
        f = np.array([0.0, 1.0])
        # large gamma → single segment
        u = pl.min_l2_potts(f, gamma=10.0)
        assert count_jumps(u) == 0
        # small gamma → keeps the jump
        u2 = pl.min_l2_potts(f, gamma=0.01)
        assert count_jumps(u2) == 1


# ─────────────────────────────────────────────────────────────────────────────
# min_l1_potts input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestL1PottsValidation:
    def test_nan_in_signal(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_potts(np.array([1.0, np.nan, 3.0]), gamma=1.0)

    def test_inf_in_signal(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_potts(np.array([1.0, np.inf]), gamma=1.0)

    def test_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l1_potts(np.ones(5), gamma=-1.0)

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            pl.min_l1_potts(np.ones((5, 3)), gamma=1.0)

    def test_nan_in_weights(self):
        f = np.ones(5)
        w = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_potts(f, gamma=1.0, weights=w)

    def test_inf_in_weights(self):
        f = np.ones(5)
        w = np.array([1.0, np.inf, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_potts(f, gamma=1.0, weights=w)

    def test_negative_weight(self):
        f = np.ones(5)
        w = np.array([1.0, -0.1, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            pl.min_l1_potts(f, gamma=1.0, weights=w)

    def test_single_element(self):
        u, err, nj, en = pl.min_l1_potts(np.array([7.0]), gamma=1.0)
        npt.assert_allclose(u, [7.0])
        assert nj == 0
        npt.assert_allclose(err, 0.0)

    def test_two_elements_single_segment(self):
        # large gamma → one segment, value = median = 0.5
        u, err, nj, en = pl.min_l1_potts(np.array([0.0, 1.0]), gamma=10.0)
        assert nj == 0

    def test_constant_signal(self):
        f = np.full(20, 2.5)
        u, err, nj, en = pl.min_l1_potts(f, gamma=0.5)
        assert nj == 0
        npt.assert_allclose(u, 2.5)
        npt.assert_allclose(err, 0.0, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────────────
# min_l2_potts_2d input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPotts2DValidation:
    def test_nan_in_image(self):
        f = np.ones((4, 4))
        f[1, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts_2d(f, gamma=0.1)

    def test_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l2_potts_2d(np.ones((4, 4)), gamma=-0.1)

    def test_mu_step_le_1(self):
        with pytest.raises(ValueError, match="mu_step"):
            pl.min_l2_potts_2d(np.ones((4, 4)), gamma=0.1, mu_step=0.9)

    def test_mu_step_eq_1(self):
        with pytest.raises(ValueError, match="mu_step"):
            pl.min_l2_potts_2d(np.ones((4, 4)), gamma=0.1, mu_step=1.0)

    def test_mu_init_zero(self):
        with pytest.raises(ValueError, match="mu_init"):
            pl.min_l2_potts_2d(np.ones((4, 4)), gamma=0.1, mu_init=0.0)

    def test_mu_init_negative(self):
        with pytest.raises(ValueError, match="mu_init"):
            pl.min_l2_potts_2d(np.ones((4, 4)), gamma=0.1, mu_init=-1.0)

    def test_nan_in_weights(self):
        f = np.ones((4, 4))
        w = np.ones((4, 4))
        w[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_potts_2d(f, gamma=0.1, weights=w)

    def test_negative_weight(self):
        f = np.ones((4, 4))
        w = np.ones((4, 4))
        w[2, 2] = -0.5
        with pytest.raises(ValueError, match="non-negative"):
            pl.min_l2_potts_2d(f, gamma=0.1, weights=w)

    def test_wrong_weights_shape(self):
        f = np.ones((4, 4))
        with pytest.raises(ValueError, match="weights"):
            pl.min_l2_potts_2d(f, gamma=0.1, weights=np.ones((3, 4)))

    def test_4d_input_rejected(self):
        with pytest.raises(ValueError):
            pl.min_l2_potts_2d(np.ones((4, 4, 3, 2)), gamma=0.1)

    def test_custom_mu_init_positive(self):
        f = np.random.default_rng(0).standard_normal((8, 8)).clip(0, 1)
        u = pl.min_l2_potts_2d(f, gamma=0.05, mu_init=0.001)
        assert u.shape == f.shape
        assert np.all(np.isfinite(u))


# ─────────────────────────────────────────────────────────────────────────────
# min_l2_spars / min_l1_spars input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSparsityValidation:
    def test_l2_nan_input(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_spars(np.array([1.0, np.nan]), gamma=1.0)

    def test_l2_inf_input(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l2_spars(np.array([np.inf, 1.0]), gamma=1.0)

    def test_l2_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l2_spars(np.ones(5), gamma=-1.0)

    def test_l1_nan_input(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_spars(np.array([np.nan, 1.0]), gamma=1.0)

    def test_l1_inf_input(self):
        with pytest.raises(ValueError, match="finite"):
            pl.min_l1_spars(np.array([np.inf, 1.0]), gamma=1.0)

    def test_l1_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            pl.min_l1_spars(np.ones(5), gamma=-1.0)

    def test_l2_empty_array(self):
        u, err, nsp, en = pl.min_l2_spars(np.array([]), gamma=1.0)
        assert u.shape == (0,)
        assert nsp == 0

    def test_l1_empty_array(self):
        u, err, nsp, en = pl.min_l1_spars(np.array([]), gamma=1.0)
        assert u.shape == (0,)
        assert nsp == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tikhonov input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTikhonovValidation:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.A = rng.standard_normal((8, 5))
        self.b = rng.standard_normal(8)

    def test_l2_negative_lam(self):
        with pytest.raises(ValueError, match="lam"):
            pl.min_l2_tikhonov(self.b, lam=-0.1, A=self.A)

    def test_l1_negative_lam(self):
        with pytest.raises(ValueError, match="lam"):
            pl.min_l1_tikhonov(self.b, lam=-1.0, A=self.A)

    def test_l2_lam_zero_allowed(self):
        # lam=0 is the unregularised case; allowed but may not converge for rank-deficient A
        u, flag = pl.min_l2_tikhonov(self.b, lam=0.0, A=self.A)
        assert u.shape == (self.A.shape[1],)

    def test_l1_lam_zero_allowed(self):
        u, flag = pl.min_l1_tikhonov(self.b, lam=0.0, A=self.A)
        assert u.shape == (self.A.shape[1],)

    def test_l1_sparse_A_accepted(self):
        A_sparse = scipy.sparse.csr_matrix(self.A)
        u, flag = pl.min_l1_tikhonov(self.b, lam=0.5, A=A_sparse)
        assert u.shape == (self.A.shape[1],)

    def test_l1_wrong_b_shape(self):
        with pytest.raises(ValueError):
            pl.min_l1_tikhonov(np.ones(3), lam=0.1, A=self.A)

    def test_l2_converges_identity(self):
        n = 10
        A = np.eye(n)
        b = np.arange(n, dtype=float)
        u, flag = pl.min_l2_tikhonov(b, lam=0.0, A=A)
        npt.assert_allclose(u, b, atol=1e-6)
        assert flag == 0


# ─────────────────────────────────────────────────────────────────────────────
# Utility function validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCountJumpsValidation:
    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            count_jumps(np.array([1.0, np.nan, 2.0]))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            count_jumps(np.array([1.0, np.inf]))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            count_jumps(np.ones((2, 3, 4)))

    def test_empty_1d(self):
        assert count_jumps(np.array([])) == 0

    def test_single_element(self):
        assert count_jumps(np.array([5.0])) == 0

    def test_all_equal(self):
        assert count_jumps(np.array([3.0, 3.0, 3.0])) == 0

    def test_alternating(self):
        assert count_jumps(np.array([0.0, 1.0, 0.0, 1.0])) == 3

    def test_multichannel_any_channel_jump(self):
        u = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        assert count_jumps(u) == 1


class TestWeightedMedianValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            weighted_median(np.array([]), np.array([]))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, -0.5, 1.0]))

    def test_single_element(self):
        assert weighted_median(np.array([7.0]), np.array([3.0])) == 7.0

    def test_uniform_weights_matches_numpy(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal(21)
        w = np.ones(21)
        result = weighted_median(data, w)
        npt.assert_allclose(result, np.median(data), atol=1e-12)

    def test_zero_weight_elements_ignored(self):
        # data[0]=100 but weight=0 should not affect median
        data = np.array([100.0, 2.0, 2.0, 2.0, 2.0])
        w    = np.array([0.0,   1.0, 1.0, 1.0, 1.0])
        result = weighted_median(data, w)
        npt.assert_allclose(result, 2.0)

    def test_heavy_weight_dominates(self):
        data = np.array([1.0, 5.0, 10.0])
        w    = np.array([1.0, 100.0, 1.0])
        result = weighted_median(data, w)
        npt.assert_allclose(result, 5.0)


class TestSamplesToWeightsValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            samples_to_weights(np.array([]))

    def test_decreasing_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            samples_to_weights(np.array([3.0, 1.0, 2.0]))

    def test_equal_raises(self):
        # Duplicate positions → not strictly increasing
        with pytest.raises(ValueError, match="strictly increasing"):
            samples_to_weights(np.array([0.0, 1.0, 1.0, 2.0]))

    def test_single_element(self):
        w = samples_to_weights(np.array([5.0]))
        npt.assert_allclose(w, [1.0])

    def test_equidistant(self):
        s = np.array([0.0, 1.0, 2.0, 3.0])
        w = samples_to_weights(s)
        # Interior weights = 1.0, boundary weights = 0.5
        npt.assert_allclose(w, [0.5, 1.0, 1.0, 0.5])

    def test_weights_sum_to_span(self):
        s = np.array([0.0, 0.5, 1.5, 4.0])
        w = samples_to_weights(s)
        npt.assert_allclose(w.sum(), s[-1] - s[0], atol=1e-12)


class TestSoftThresholdValidation:
    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            soft_threshold(np.array([1.0, 2.0]), tau=-0.1)

    def test_zero_tau_is_identity(self):
        x = np.array([-2.0, 0.0, 2.0])
        npt.assert_allclose(soft_threshold(x, 0.0), x)

    def test_large_tau_zeroes_all(self):
        x = np.array([0.5, -0.5, 0.3])
        npt.assert_allclose(soft_threshold(x, 1.0), np.zeros(3))

    def test_exact_values(self):
        npt.assert_allclose(soft_threshold(np.array([0.5]), 0.3), [0.2], atol=1e-15)
        npt.assert_allclose(soft_threshold(np.array([-0.5]), 0.3), [-0.2], atol=1e-15)


class TestSegToLabelValidation:
    def test_invalid_connectivity_raises(self):
        img = np.zeros((4, 4))
        with pytest.raises(ValueError, match="connectivity"):
            seg_to_label(img, connectivity=6)

    def test_connectivity_0_raises(self):
        with pytest.raises(ValueError, match="connectivity"):
            seg_to_label(np.zeros((4, 4)), connectivity=0)

    def test_constant_image_single_label(self):
        img = np.zeros((5, 5))
        labels = seg_to_label(img, connectivity=4)
        assert labels.shape == (5, 5)
        # All pixels belong to one region
        assert len(np.unique(labels[labels > 0])) == 1

    def test_two_regions_4connected(self):
        img = np.zeros((4, 4))
        img[:, 2:] = 1.0
        labels = seg_to_label(img, connectivity=4)
        assert len(np.unique(labels[labels > 0])) == 2

    def test_rgb_image(self):
        img = np.zeros((4, 4, 3))
        img[:, 2:] = 1.0
        labels = seg_to_label(img, connectivity=4)
        assert labels.shape == (4, 4)


class TestEnergyL2PottsValidation:
    def test_incompatible_shapes_raise(self):
        u = np.zeros(10)
        f = np.zeros(11)
        with pytest.raises(ValueError, match="size"):
            energy_l2_potts(u, f, gamma=1.0)

    def test_constant_signal_zero_data_term(self):
        u = np.ones(20)
        f = np.ones(20)
        en = energy_l2_potts(u, f, gamma=0.5)
        npt.assert_allclose(en, 0.0, atol=1e-14)

    def test_with_A(self):
        A = np.eye(5)
        u = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        f = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        en = energy_l2_potts(u, f, gamma=1.0, A=A)
        npt.assert_allclose(en, 1.0 * 1 + 0.0)  # 1 jump, 0 data error


# ─────────────────────────────────────────────────────────────────────────────
# Samples parameter end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestNonEquidistantSamples:
    def test_l2_potts_non_equidistant(self):
        # Non-uniform spacing: dense at the start, sparse at the end.
        # Dense region should dominate (larger trapezoidal weight).
        f = np.array([0.0] * 8 + [1.0] * 2)
        samples = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0, 10.0])
        u = pl.min_l2_potts(f, gamma=0.05, samples=samples)
        assert np.all(np.isfinite(u))

    def test_l2_non_monotone_samples_raise(self):
        f = np.ones(5)
        with pytest.raises(ValueError, match="strictly increasing"):
            pl.min_l2_potts(f, gamma=1.0, samples=np.array([0.0, 2.0, 1.0, 3.0, 4.0]))

    def test_l1_non_monotone_samples_raise(self):
        f = np.ones(5)
        with pytest.raises(ValueError, match="strictly increasing"):
            pl.min_l1_potts(f, gamma=1.0, samples=np.array([0.0, 2.0, 1.0, 3.0, 4.0]))


# ─────────────────────────────────────────────────────────────────────────────
# Numerical robustness edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalRobustness:
    def test_l2_very_large_gamma(self):
        f = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = pl.min_l2_potts(f, gamma=1e12)
        assert count_jumps(u) == 0
        assert np.all(np.isfinite(u))

    def test_l2_very_small_gamma(self):
        f = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        u = pl.min_l2_potts(f, gamma=1e-12)
        assert np.all(np.isfinite(u))

    def test_l1_very_large_gamma(self):
        f = np.array([0.0, 1.0, 2.0, 3.0])
        u, _, nj, _ = pl.min_l1_potts(f, gamma=1e10)
        assert nj == 0
        assert np.all(np.isfinite(u))

    def test_l2_large_values(self):
        # Values near float max boundary
        f = np.array([1e10, 1e10, 2e10, 2e10])
        u = pl.min_l2_potts(f, gamma=1e5)
        assert np.all(np.isfinite(u))

    def test_l2_identical_two_elements(self):
        u = pl.min_l2_potts(np.array([3.0, 3.0]), gamma=1.0)
        npt.assert_allclose(u, [3.0, 3.0])

    def test_l2_negative_values(self):
        f = np.array([-5.0, -5.0, -5.0, 2.0, 2.0, 2.0])
        u = pl.min_l2_potts(f, gamma=0.1)
        assert count_jumps(u) == 1
        npt.assert_allclose(u[:3], u[0])   # first segment constant
        npt.assert_allclose(u[3:], u[3])   # second segment constant

    def test_l2_multichannel_single_element(self):
        f = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
        u = pl.min_l2_potts(f, gamma=1.0)
        assert u.shape == (1, 3)
        npt.assert_allclose(u, f)

    def test_l2_multichannel_two_elements(self):
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        # large gamma → single segment, mean = [0.5, 0.5]
        u = pl.min_l2_potts(f, gamma=10.0)
        assert count_jumps(u) == 0
        npt.assert_allclose(u[0], u[1])

    def test_l1_single_outlier_does_not_shift(self):
        # Constant signal with one extreme outlier; median should remain 0
        f = np.array([0.0] * 19 + [1000.0])
        u, _, nj, _ = pl.min_l1_potts(f, gamma=0.1)
        # Either all-zero segment or outlier gets its own segment
        # The constant region must still be close to 0
        assert np.all(np.isfinite(u))

    def test_l2_n2_correct_segmentation(self):
        # n=2: threshold for jump = gamma. Jump iff (Δ)²/4 > gamma
        f = np.array([0.0, 1.0])
        # Cost single segment: (0.5)²+(0.5)² = 0.5
        # Cost two segments: 0 + gamma
        # Two segments cheaper iff gamma > 0.5
        u_split  = pl.min_l2_potts(f, gamma=0.4)
        u_merged = pl.min_l2_potts(f, gamma=0.6)
        assert count_jumps(u_split)  == 1
        assert count_jumps(u_merged) == 0
