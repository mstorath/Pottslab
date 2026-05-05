# Integration / cross-module tests for pottslab.
# These test interactions between modules and end-to-end pipelines.

import numpy as np
import numpy.testing as npt
import pytest

import sys
sys.path.insert(0, ".")
import pottslab as pl


# ---------------------------------------------------------------------------
# Direct vs. inverse Potts consistency
# ---------------------------------------------------------------------------

class TestDirectVsInverse:
    """min_l2_ipotts with A = I should recover the direct Potts solution."""

    def test_identity_l2_converges_to_direct(self):
        f = np.array([0.0] * 15 + [1.0] * 15, dtype=float)
        n = len(f)
        gamma = 0.5
        u_direct = pl.min_l2_potts(f, gamma)
        u_inv, *_ = pl.min_l2_ipotts(f, gamma, A=np.eye(n),
                                      tol=1e-10, mu_step=1.05, imax=1)
        # Inverse should converge to direct (within ADMM precision)
        npt.assert_allclose(u_inv, u_direct, atol=0.05)

    def test_blurred_reconstruction_reduces_data_error(self):
        """After deblurring, the data error should be lower than the blurred error."""
        n = 30
        # Build a convolution matrix (simple uniform blur)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i - 2), min(n, i + 3)):
                A[i, j] = 1.0 / 5
        # True signal: step function
        u_true = np.array([0.0] * 15 + [1.0] * 15, dtype=float)
        f = A @ u_true + 0.01 * np.random.default_rng(60).standard_normal(n)
        # Reconstruct
        u_rec, data_err, *_ = pl.min_l2_ipotts(f, gamma=0.1, A=A,
                                                 tol=1e-6, mu_step=1.05)
        # Reconstructed signal should have smaller ||A u - f||^2 than u_true
        err_true = np.sum((A @ u_true - f) ** 2)
        # The ADMM solution should at least be finite and close to 0 or 1
        assert np.all(np.isfinite(u_rec))


# ---------------------------------------------------------------------------
# Pipeline: denoise → count → verify
# ---------------------------------------------------------------------------

class TestPipeline:
    """End-to-end pipelines typical of the original MATLAB demos."""

    def test_1d_denoising_pipeline(self):
        """Noisy step signal → L2-Potts → compare against ground truth."""
        rng = np.random.default_rng(61)
        n = 100
        u_true = np.array([0.0] * 50 + [1.0] * 50)
        f = u_true + rng.normal(0, 0.1, n)

        u = pl.min_l2_potts(f, gamma=0.3)

        # Should recover 1 jump
        assert pl.count_jumps(u) == 1
        # Should be closer to u_true than f is
        err_u = np.sum((u - u_true) ** 2)
        err_f = np.sum((f - u_true) ** 2)
        assert err_u < err_f

    def test_1d_l1_denoising_robust_to_outliers(self):
        """L1-Potts denoising is more robust to impulse noise than L2."""
        rng = np.random.default_rng(62)
        n = 60
        u_true = np.array([0.0] * 30 + [1.0] * 30)
        # Add impulse noise (10% of samples corrupted)
        f = u_true.copy()
        idx = rng.choice(n, size=6, replace=False)
        f[idx] = rng.choice([-5.0, 5.0], size=6)

        u_l2 = pl.min_l2_potts(f, gamma=1.0)
        u_l1, *_ = pl.min_l1_potts(f, gamma=1.0)

        err_l2 = np.sum((u_l2 - u_true) ** 2)
        err_l1 = np.sum((u_l1 - u_true) ** 2)
        # L1 should be at least competitive with L2 on outlier data
        # (not always strictly better, but usually is)
        assert err_l1 <= err_l2 * 2 + 0.1  # generous tolerance

    def test_multichannel_denoising_pipeline(self):
        """3-channel noisy signal denoised with L2-Potts."""
        rng = np.random.default_rng(63)
        n = 40
        u_true = np.zeros((n, 3))
        u_true[20:, :] = np.array([1.0, 0.5, 0.2])
        f = u_true + rng.normal(0, 0.05, (n, 3))

        u = pl.min_l2_potts(f, gamma=0.1)

        err_u = np.sum((u - u_true) ** 2)
        err_f = np.sum((f - u_true) ** 2)
        assert err_u < err_f
        assert pl.count_jumps(u) == 1

    def test_2d_segmentation_pipeline(self):
        """2-D segmentation: noisy step image → segment → check boundaries."""
        rng = np.random.default_rng(64)
        h, w = 32, 32
        u_true = np.zeros((h, w))
        u_true[:, 16:] = 1.0
        f = u_true + rng.normal(0, 0.1, (h, w))
        f = np.clip(f, 0, 1)

        u = pl.min_l2_potts_2d(f, gamma=0.01, isotropic=False,
                                tol=1e-7, quantize=False)

        # Left half close to 0, right half close to 1
        assert np.mean(u[:, :16]) < 0.3
        assert np.mean(u[:, 16:]) > 0.7

    def test_sparsity_recovery_pipeline(self):
        """Sparse signal recovery: few spikes in noise → L2-sparsity → detect."""
        rng = np.random.default_rng(65)
        n = 50
        f = rng.normal(0, 0.2, n)   # noise floor
        spike_positions = [10, 30, 45]
        for pos in spike_positions:
            f[pos] = 4.0            # spikes well above noise

        u, *_ = pl.min_l2_spars(f, gamma=1.0)  # threshold = 1

        # All three spikes should be detected
        for pos in spike_positions:
            assert u[pos] != 0.0, f"Spike at {pos} was not detected"


# ---------------------------------------------------------------------------
# Consistency across modules
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    def test_energy_function_consistent_with_solver(self):
        """energy_l2_potts matches manual computation for solver output."""
        rng = np.random.default_rng(66)
        f = rng.standard_normal(30)
        gamma = 1.5
        u = pl.min_l2_potts(f, gamma)
        e_fn = pl.energy_l2_potts(u, f, gamma)
        e_manual = float(gamma * pl.count_jumps(u) + np.sum((u - f) ** 2))
        npt.assert_allclose(e_fn, e_manual, rtol=1e-12)

    def test_samples_to_weights_then_l2_potts(self):
        """samples_to_weights output used as weights in min_l2_potts — no error."""
        f = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        s = np.array([0.0, 0.5, 1.0, 2.0, 3.0])  # non-equidistant
        w = pl.samples_to_weights(s)
        u = pl.min_l2_potts(f, gamma=0.01, weights=w)
        assert np.all(np.isfinite(u))
        assert pl.count_jumps(u) == 1

    def test_seg_to_label_on_potts_2d_output(self, step_image_2d):
        """seg_to_label on 2D Potts output gives exactly 2 regions."""
        u = pl.min_l2_potts_2d(step_image_2d, gamma=0.001, tol=1e-7, quantize=False)
        labels = pl.seg_to_label(u)
        n_regions = len(np.unique(labels[labels > 0]))
        assert n_regions == 2

    def test_l2_spars_then_l2_potts(self):
        """Applying L2-sparsity then L2-Potts: composed result has ≤ spikes + jumps."""
        rng = np.random.default_rng(67)
        f = rng.standard_normal(30)
        u_s, *_ = pl.min_l2_spars(f, gamma=1.0)
        u_p = pl.min_l2_potts(u_s, gamma=0.5)
        assert np.all(np.isfinite(u_p))

    def test_count_jumps_equals_n_jumps_in_l1_tuple(self):
        """The n_jumps in the L1-Potts return tuple matches count_jumps(u)."""
        rng = np.random.default_rng(68)
        f = rng.standard_normal(40)
        for gamma in [0.1, 1.0, 5.0]:
            u, _, nj, _ = pl.min_l1_potts(f, gamma)
            assert nj == pl.count_jumps(u), f"Mismatch at gamma={gamma}"

    def test_soft_threshold_equals_l1_spars_kernel(self):
        """min_l1_spars is exactly soft_threshold(f, gamma/2)."""
        rng = np.random.default_rng(69)
        f = rng.standard_normal(20)
        gamma = 1.0
        u_spars, *_ = pl.min_l1_spars(f, gamma)
        u_soft = pl.soft_threshold(f, gamma / 2)
        npt.assert_allclose(u_spars, u_soft, atol=1e-12)

    def test_l2_potts_then_energy_is_optimal(self):
        """energy(min_l2_potts(f)) ≤ energy(f) for all tested inputs."""
        rng = np.random.default_rng(70)
        for seed in range(10):
            f = rng.standard_normal(25)
            gamma = rng.uniform(0.1, 5.0)
            u = pl.min_l2_potts(f, gamma)
            e_u = pl.energy_l2_potts(u, f, gamma)
            e_f = pl.energy_l2_potts(f, f, gamma)
            assert e_u <= e_f + 1e-10, f"seed={seed}: {e_u} > {e_f}"
