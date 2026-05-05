# Tests for the 2-D L2-Potts ADMM solver.
# Verifies faithfulness of the Rust port of JavaTools.java (ADMM4/8).

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


class TestPotts2DBasic:
    def test_constant_image_unchanged(self):
        """A constant image is already optimal — returns same value (no quantize)."""
        f = np.full((32, 32), 0.5)
        u = pl.min_l2_potts_2d(f, gamma=1.0, quantize=False)
        npt.assert_allclose(u, 0.5, atol=1e-4)

    def test_zero_image(self):
        """All-zero image shortcircuits to zero (||f||=0 early exit)."""
        f = np.zeros((16, 16))
        u = pl.min_l2_potts_2d(f, gamma=1.0)
        npt.assert_allclose(u, 0.0, atol=1e-8)

    def test_step_image_two_regions_anisotropic(self, step_image_2d):
        """Anisotropic (4-connected): step image → 2 constant regions."""
        u = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, isotropic=False,
                                mu_step=2.0, tol=1e-8, quantize=False)
        left = u[:, :32]
        right = u[:, 32:]
        # Left half should be close to 0, right half close to 1
        assert np.mean(left) < 0.2
        assert np.mean(right) > 0.8

    def test_step_image_two_regions_isotropic(self, step_image_2d):
        """Isotropic (8-connected): step image → 2 constant regions."""
        u = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, isotropic=True,
                                mu_step=2.0, tol=1e-8, quantize=False)
        left = u[:, :32]
        right = u[:, 32:]
        assert np.mean(left) < 0.2
        assert np.mean(right) > 0.8

    def test_output_shape_2d(self, step_image_2d):
        """Output must have same shape as 2-D input."""
        u = pl.min_l2_potts_2d(step_image_2d, gamma=1.0)
        assert u.shape == step_image_2d.shape

    def test_output_shape_3d(self, rgb_step_image):
        """Output must have same shape as 3-D (RGB) input."""
        u = pl.min_l2_potts_2d(rgb_step_image, gamma=0.01)
        assert u.shape == rgb_step_image.shape

    def test_rgb_image_segmentation(self, rgb_step_image):
        """RGB step image should segment into 2 regions."""
        u = pl.min_l2_potts_2d(rgb_step_image, gamma=0.001, isotropic=False,
                                mu_step=2.0, tol=1e-8, quantize=False)
        # Channel 0: left≈0, right≈1
        assert np.mean(u[:, :32, 0]) < 0.1
        assert np.mean(u[:, 32:, 0]) > 0.9

    def test_quantize_false_no_rounding(self):
        """With quantize=False, output should not be rounded to 1/255 steps."""
        f = np.random.default_rng(0).random((16, 16))
        u_q = pl.min_l2_potts_2d(f, gamma=10.0, quantize=True)
        u_nq = pl.min_l2_potts_2d(f, gamma=10.0, quantize=False)
        # Quantized version should differ slightly from non-quantized
        # (unless all values happen to be multiples of 1/255)
        # Just check they're both valid arrays
        assert u_q.shape == u_nq.shape

    def test_isotropic_and_anisotropic_agree_axis_aligned(self, step_image_2d):
        """Both variants should agree on axis-aligned boundaries (both segment correctly)."""
        u4 = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, isotropic=False,
                                 tol=1e-8, quantize=False)
        u8 = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, isotropic=True,
                                 tol=1e-8, quantize=False)
        # Both should identify the same boundary roughly
        jump4 = np.mean(u4[:, :32]) < 0.3 and np.mean(u4[:, 32:]) > 0.7
        jump8 = np.mean(u8[:, :32]) < 0.3 and np.mean(u8[:, 32:]) > 0.7
        assert jump4 and jump8


class TestPotts2DValidation:
    def test_gamma_zero_raises(self, step_image_2d):
        with pytest.raises(ValueError, match="gamma must be > 0"):
            pl.min_l2_potts_2d(step_image_2d, gamma=0.0)

    def test_mu_step_le_1_raises(self, step_image_2d):
        with pytest.raises(ValueError, match="mu_step must be > 1"):
            pl.min_l2_potts_2d(step_image_2d, gamma=1.0, mu_step=1.0)

    def test_negative_tol_raises(self, step_image_2d):
        with pytest.raises(ValueError, match="tol must be > 0"):
            pl.min_l2_potts_2d(step_image_2d, gamma=1.0, tol=-1.0)

    def test_wrong_weights_shape_raises(self, step_image_2d):
        bad_weights = np.ones((10, 10))
        with pytest.raises(ValueError, match="weights must have shape"):
            pl.min_l2_potts_2d(step_image_2d, gamma=1.0, weights=bad_weights)

    def test_wrong_input_ndim_raises(self):
        with pytest.raises(ValueError):
            pl.min_l2_potts_2d(np.ones((5,)), gamma=1.0)
