# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
1-D Potts functional minimisers.

Ported from:
  Potts/minL2Potts.m  (L2 data fidelity)
  Potts/minL1Potts.m  (L1 data fidelity)
  Java/src/pottslab/L2Potts.java
  Java/src/pottslab/IndexedLinkedHistogram.java
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

try:
    import pottslab._core as _core
except ImportError:
    import pottslab._core_py as _core  # pure-Python fallback (slower)
from pottslab.utils import count_jumps, energy_l2_potts, samples_to_weights


def min_l2_potts(
    f: NDArray[np.floating],
    gamma: float,
    weights: Optional[NDArray[np.floating]] = None,
    samples: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Minimise the univariate L2-Potts functional.

    Solves  min_u  gamma * ||Du||_0  +  ||u - f||_{2,w}^2
    via dynamic programming in O(n^2) time.

    Parameters
    ----------
    f : array_like, shape (n,) or (n, channels)
        Input signal. Multi-channel signals are processed jointly.
    gamma : float
        Jump penalty (must be > 0).
    weights : array_like, shape (n,), optional
        Per-sample weights for the data term. Default: all ones.
    samples : array_like, shape (n,), optional
        Sorted sample positions for non-equidistant sampling.
        Overrides `weights` via trapezoidal quadrature.

    Returns
    -------
    u : ndarray, same shape as f
        Piecewise-constant minimiser.

    Ported from Potts/minL2Potts.m and Java/src/pottslab/L2Potts.java.
    """
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("Input f must contain only finite values.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    scalar = f.ndim == 1
    if scalar:
        f2d = f[:, np.newaxis]
    else:
        if f.ndim != 2:
            raise ValueError("f must be 1-D (n,) or 2-D (n, channels).")
        f2d = f

    if samples is not None:
        weights = samples_to_weights(np.asarray(samples, dtype=np.float64))

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (f2d.shape[0],):
            raise ValueError("weights must have shape (n,).")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

    n, ch = f2d.shape
    w_arg = weights if weights is not None else None

    if ch == 1:
        f_c = np.ascontiguousarray(f2d[:, 0])
        u_flat = np.asarray(_core.solve_l2_potts_1d(f_c, gamma, w_arg))
        return u_flat if scalar else u_flat[:, np.newaxis]
    else:
        f_c = np.ascontiguousarray(f2d)
        u2d = np.asarray(_core.solve_l2_potts_vv(f_c, gamma, w_arg))
        return u2d


def min_l1_potts(
    f: NDArray[np.floating],
    gamma: float,
    weights: Optional[NDArray[np.floating]] = None,
    samples: Optional[NDArray[np.floating]] = None,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Minimise the univariate L1-Potts functional.

    Solves  min_u  gamma * ||Du||_0  +  ||u - f||_{1,w}
    via dynamic programming in O(n^2) time using an indexed linked histogram.

    Parameters
    ----------
    f : array_like, shape (n,)
        Scalar input signal.
    gamma : float
        Jump penalty (must be > 0).
    weights : array_like, shape (n,), optional
        Per-sample weights. Default: all ones.
    samples : array_like, shape (n,), optional
        Sorted sample positions (overrides weights).

    Returns
    -------
    u : ndarray, shape (n,)
        Piecewise-constant minimiser.
    data_error : float
        Weighted L1 data fidelity ||u - f||_1.
    n_jumps : int
        Number of jump discontinuities in u.
    energy : float
        Total Potts energy gamma * n_jumps + data_error.

    Ported from Potts/minL1Potts.m and Java/src/pottslab/IndexedLinkedHistogram.java.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError("L1-Potts supports scalar (1-D) signals only.")
    f = f.ravel()
    if not np.all(np.isfinite(f)):
        raise ValueError("Input f must contain only finite values.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    if samples is not None:
        weights = samples_to_weights(np.asarray(samples, dtype=np.float64))

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != f.shape:
            raise ValueError("weights must have shape (n,).")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

    f_c = np.ascontiguousarray(f)
    w_arg = weights if weights is not None else None
    u = np.asarray(_core.solve_l1_potts_1d(f_c, gamma, w_arg))

    n_jumps = count_jumps(u)
    if weights is not None:
        data_error = float(np.sum(weights * np.abs(u - f)))
    else:
        data_error = float(np.sum(np.abs(u - f)))
    energy = gamma * n_jumps + data_error

    return u, data_error, n_jumps, energy
