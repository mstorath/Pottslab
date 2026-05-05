# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
2-D Potts functional minimiser via ADMM splitting.

Ported from:
  Potts2D/minL2Potts2DADMM.m
  Java/src/pottslab/JavaTools.java  (minL2PottsADMM4, minL2PottsADMM8)
  Java/src/pottslab/PLProcessor.java
"""

import math
import numpy as np
from numpy.typing import NDArray
from typing import Optional

try:
    import pottslab._core as _core
except ImportError:
    import pottslab._core_py as _core  # pure-Python fallback (slower)


def min_l2_potts_2d(
    f: NDArray[np.floating],
    gamma: float,
    *,
    weights: Optional[NDArray[np.floating]] = None,
    mu_init: Optional[float] = None,
    mu_step: float = 2.0,
    tol: float = 1e-10,
    isotropic: bool = True,
    verbose: bool = False,
    quantize: bool = True,
) -> NDArray[np.floating]:
    """Minimise the 2-D L2-Potts (image segmentation) functional.

    Solves  min_u  gamma * ||Du||_0  +  <weights, ||u - f||_2^2>
    using an ADMM splitting strategy that reduces the 2-D problem to a
    sequence of 1-D Potts problems solved in parallel.

    Parameters
    ----------
    f : array_like, shape (H, W) or (H, W, C)
        Input image. Values should be in [0, 1] for standard images.
        Float64 is used internally.
    gamma : float
        Jump penalty (must be > 0).
    weights : array_like, shape (H, W), optional
        Per-pixel data weights. Default: all ones.
    mu_init : float, optional
        Initial ADMM coupling parameter. Default: gamma * 1e-2.
    mu_step : float, optional
        ADMM coupling growth factor (must be > 1). Default: 2.0.
    tol : float, optional
        Convergence tolerance. Default: 1e-10.
    isotropic : bool, optional
        If True (default), use near-isotropic 8-connected neighbourhood.
        If False, use anisotropic 4-connected neighbourhood.
    verbose : bool, optional
        Print iteration progress. Default: False.
    quantize : bool, optional
        Round result to 1/255 steps, matching the original MATLAB output
        behaviour. Default: True.

    Returns
    -------
    u : ndarray, same shape as f
        Piecewise-constant minimiser.

    References
    ----------
    M. Storath, A. Weinmann, "Fast partitioning of vector-valued images",
    SIAM Journal on Imaging Sciences, 2014.

    Ported from Potts2D/minL2Potts2DADMM.m and Java/src/pottslab/JavaTools.java.
    """
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("Input f must contain only finite values.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if mu_step <= 1:
        raise ValueError("mu_step must be > 1.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")

    # Normalise to (H, W, C)
    squeezed = f.ndim == 2
    if squeezed:
        f3d = f[:, :, np.newaxis]
    elif f.ndim == 3:
        f3d = f
    else:
        raise ValueError("f must be 2-D (H, W) or 3-D (H, W, C).")

    h, w, c = f3d.shape

    if weights is None:
        weights_2d = np.ones((h, w), dtype=np.float64)
    else:
        weights_2d = np.asarray(weights, dtype=np.float64)
        if weights_2d.shape != (h, w):
            raise ValueError(f"weights must have shape ({h}, {w}).")
        if not np.all(np.isfinite(weights_2d)):
            raise ValueError("weights must contain only finite values.")
        if np.any(weights_2d < 0):
            raise ValueError("weights must be non-negative.")

    if mu_init is None:
        mu_init = gamma * 1e-2
    elif mu_init <= 0:
        raise ValueError("mu_init must be > 0.")

    f_c = np.ascontiguousarray(f3d)
    w_c = np.ascontiguousarray(weights_2d)

    if isotropic:
        # Near-isotropic weights from original MATLAB code
        omega_c = math.sqrt(2.0) - 1.0
        omega_d = 1.0 - math.sqrt(2.0) / 2.0
        result = np.asarray(
            _core.admm_8connected(f_c, gamma, w_c, mu_init, mu_step, tol, verbose, omega_c, omega_d)
        )
    else:
        result = np.asarray(
            _core.admm_4connected(f_c, gamma, w_c, mu_init, mu_step, tol, verbose)
        )

    if quantize:
        result = np.round(result * 255.0) / 255.0

    if squeezed:
        return result[:, :, 0]
    return result
