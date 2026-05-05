# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
Utility functions ported from Auxiliary/ in the original MATLAB pottslab.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def count_jumps(u: NDArray[np.floating]) -> int:
    """Count the number of jump discontinuities in a 1-D signal.

    A jump occurs wherever consecutive values differ. For multi-channel
    signals (shape (n, c)), a jump is counted when any channel differs.

    Ported from Auxiliary/countJumps.m.
    """
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite values (no NaN or Inf).")
    if u.ndim == 1:
        return int(np.sum(u[1:] != u[:-1]))
    elif u.ndim == 2:
        diff = np.any(u[1:] != u[:-1], axis=1)
        return int(np.sum(diff))
    else:
        raise ValueError("u must be 1-D or 2-D (n, channels)")


def energy_l2_potts(
    u: NDArray[np.floating],
    f: NDArray[np.floating],
    gamma: float,
    A: Optional[NDArray[np.floating]] = None,
    isotropic: bool = False,
) -> float:
    """Compute the L2-Potts energy  gamma * ||Du||_0 + ||Au - f||_2^2.

    If A is None the direct case Au = u is assumed.

    Ported from Auxiliary/energyL2Potts.m.
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    if A is None:
        if u.size != f.size:
            raise ValueError(
                f"u and f must have the same total size; got {u.size} and {f.size}."
            )
        data_term = float(np.sum((u.ravel() - f.ravel()) ** 2))
    else:
        residual = A @ u.ravel() - f.ravel()
        data_term = float(np.sum(residual ** 2))
    n_jumps = count_jumps(u.reshape(u.shape[0], -1) if u.ndim > 1 else u)
    return float(gamma * n_jumps + data_term)


def samples_to_weights(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert a sorted array of sample positions to per-sample weights.

    Weights are proportional to the distances between consecutive samples,
    giving a trapezoidal quadrature rule. This supports non-equidistant
    sampling for weighted Potts problems.

    Ported from Auxiliary/samplesToWeights.m.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty.")
    if n == 1:
        return np.array([1.0])
    if not np.all(np.diff(samples) > 0):
        raise ValueError("samples must be strictly increasing.")
    weights = np.zeros(n)
    weights[0] = (samples[1] - samples[0]) / 2.0
    weights[-1] = (samples[-1] - samples[-2]) / 2.0
    for i in range(1, n - 1):
        weights[i] = (samples[i + 1] - samples[i - 1]) / 2.0
    return weights


def seg_to_label(
    u: NDArray[np.floating],
    connectivity: int = 4,
) -> NDArray[np.intp]:
    """Convert a piecewise-constant image to an integer label image.

    Each constant region gets a unique integer label starting from 0.

    Ported from Auxiliary/segToLabel.m.

    Parameters
    ----------
    u : array (H, W) or (H, W, C)
    connectivity : 4 or 8 (pixel connectivity)
    """
    from scipy.ndimage import label as scipy_label

    u = np.asarray(u, dtype=float)
    if u.ndim == 3:
        # Work on L2-norm across channels to identify constant regions
        u_flat = np.linalg.norm(u, axis=2)
    else:
        u_flat = u

    # Mark transitions: label image via flood fill (scipy connected components)
    # Strategy: round to detect boundaries, then use connected components.
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")
    if connectivity == 4:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        struct = np.ones((3, 3), dtype=bool)

    # Build boundary mask: a pixel is on a boundary if it differs from a neighbour
    # Use differences to find constant regions
    rounded = np.round(u_flat * 1e8) / 1e8  # eliminate float noise
    h, w = rounded.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:-1, :] |= rounded[:-1, :] != rounded[1:, :]
    boundary[:, :-1] |= rounded[:, :-1] != rounded[:, 1:]
    if connectivity == 8:
        boundary[:-1, :-1] |= rounded[:-1, :-1] != rounded[1:, 1:]
        boundary[1:, :-1] |= rounded[1:, :-1] != rounded[:-1, 1:]

    # Connected components on non-boundary pixels
    non_boundary = ~boundary
    labeled, _ = scipy_label(non_boundary, structure=struct)
    return labeled.astype(np.intp)


def soft_threshold(x: NDArray[np.floating], tau: float) -> NDArray[np.floating]:
    """Soft thresholding operator: sign(x) * max(|x| - tau, 0).

    Ported from Auxiliary/softThreshold.m.
    """
    x = np.asarray(x, dtype=float)
    if tau < 0:
        raise ValueError("tau must be >= 0.")
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def weighted_median(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    """Compute the weighted median of a 1-D array.

    The weighted median is the value m that minimises sum_i w_i |x_i - m|.
    Uses a vectorised NumPy implementation (O(n log n) sort + O(n) scan).

    Ported from Auxiliary/medianw.m.
    """
    data = np.asarray(data, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if len(data) == 0:
        raise ValueError("data must be non-empty.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if len(data) == 1:
        return float(data[0])
    idx = np.argsort(data)
    sorted_data = data[idx]
    sorted_weights = weights[idx]
    cumsum = np.cumsum(sorted_weights)
    half = cumsum[-1] / 2.0
    # np.searchsorted gives O(log n) lookup after O(n log n) sort
    median_idx = np.searchsorted(cumsum, half)
    return float(sorted_data[min(median_idx, len(sorted_data) - 1)])
