# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
Sparsity regularisation (L0) minimisers.

Ported from:
  Sparsity/minL2Spars.m
  Sparsity/minL1Spars.m
  Sparsity/SparsityCore/minSpars.m
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from pottslab.utils import soft_threshold


def min_l2_spars(
    f: NDArray[np.floating],
    gamma: float,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Minimise the L2-sparsity functional.

    Solves  min_u  gamma * ||u||_0  +  ||u - f||_2^2
    via hard thresholding: u[i] = f[i] if f[i]^2 > gamma, else 0.

    Parameters
    ----------
    f : array_like, shape (n,)
        Input signal.
    gamma : float
        Sparsity penalty (must be > 0).

    Returns
    -------
    u : ndarray, shape (n,)
        Sparse solution.
    data_error : float
        ||u - f||_2^2
    n_spikes : int
        Number of non-zero entries in u.
    energy : float
        gamma * n_spikes + data_error

    Ported from Sparsity/minL2Spars.m and Sparsity/SparsityCore/minSpars.m.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    if not np.all(np.isfinite(f)):
        raise ValueError("Input f must contain only finite values.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    threshold = np.sqrt(gamma)
    u = np.where(np.abs(f) > threshold, f, 0.0)
    n_spikes = int(np.sum(u != 0))
    data_error = float(np.sum((u - f) ** 2))
    energy = gamma * n_spikes + data_error
    return u, data_error, n_spikes, energy


def min_l1_spars(
    f: NDArray[np.floating],
    gamma: float,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Minimise the L1-sparsity functional.

    Solves  min_u  gamma * ||u||_0  +  ||u - f||_1
    via soft thresholding: u = sign(f) * max(|f| - gamma/2, 0).

    Note: the L0 penalty makes this identical to soft thresholding with
    threshold gamma/2 when the L1 data term dominates.

    Parameters
    ----------
    f : array_like, shape (n,)
        Input signal.
    gamma : float
        Sparsity penalty (must be > 0).

    Returns
    -------
    u : ndarray, shape (n,)
        Sparse solution.
    data_error : float
        ||u - f||_1
    n_spikes : int
        Number of non-zero entries in u.
    energy : float
        gamma * n_spikes + data_error

    Ported from Sparsity/minL1Spars.m.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    if not np.all(np.isfinite(f)):
        raise ValueError("Input f must contain only finite values.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    u = soft_threshold(f, gamma / 2.0)
    n_spikes = int(np.sum(u != 0))
    data_error = float(np.sum(np.abs(u - f)))
    energy = gamma * n_spikes + data_error
    return u, data_error, n_spikes, energy
