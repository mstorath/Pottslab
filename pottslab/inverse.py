# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
Inverse Potts solvers via ADMM splitting.

Ported from:
  Potts/minL2iPotts.m
  Potts/minL1iPotts.m
  Potts/PottsCore/iPottsADMM.m

Solves  min_u  gamma * ||Du||_0  +  ||Au - f||_p^p
for indirect measurements f = Au + noise, where A is a linear operator
(e.g. blurring, Radon transform, MRI, etc.).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union

from pottslab.potts1d import min_l2_potts
from pottslab.tikhonov import min_l2_tikhonov, min_l1_tikhonov
from pottslab.utils import count_jumps


def min_l2_ipotts(
    f: NDArray[np.floating],
    gamma: float,
    A: NDArray[np.floating],
    *,
    mu_init: Optional[float] = None,
    mu_step: float = 1.05,
    tol: float = 1e-6,
    imax: int = 1,
    verbose: bool = False,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Minimise the inverse L2-Potts functional via ADMM.

    Solves  min_u  gamma * ||Du||_0  +  ||Au - f||_2^2
    by alternating between a direct L2-Potts step and a Tikhonov step.

    Parameters
    ----------
    f : array_like, shape (m,)
        Observations / indirect measurements.
    gamma : float
        Jump penalty (must be > 0).
    A : array_like, shape (m, n)
        Forward / measurement operator.
    mu_init : float, optional
        Initial ADMM coupling. Default: gamma * 1e-6.
    mu_step : float, optional
        ADMM coupling growth factor (> 1). Default: 1.05.
    tol : float, optional
        Convergence tolerance on ||u - v||_2^2. Default: 1e-6.
    imax : int, optional
        Inner ADMM iterations per coupling level. Default: 1.
    verbose : bool, optional
        Print iteration progress. Default: False.

    Returns
    -------
    u : ndarray, shape (n,)
        Piecewise-constant minimiser.
    data_error : float
        ||Au - f||_2^2
    n_jumps : int
        Number of jump discontinuities in u.
    energy : float
        gamma * n_jumps + data_error

    References
    ----------
    A. Weinmann, M. Storath. "Iterative Potts and Blake-Zisserman minimization..."
    Proc. Royal Society A, 2015.

    Ported from Potts/minL2iPotts.m and Potts/PottsCore/iPottsADMM.m.
    """
    return _ipotts_admm(f, gamma, A, p=2, mu_init=mu_init, mu_step=mu_step,
                        tol=tol, imax=imax, verbose=verbose)


def min_l1_ipotts(
    f: NDArray[np.floating],
    gamma: float,
    A: NDArray[np.floating],
    *,
    mu_init: Optional[float] = None,
    mu_step: float = 1.05,
    tol: float = 1e-6,
    imax: int = 1,
    verbose: bool = False,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Minimise the inverse L1-Potts functional via ADMM.

    Solves  min_u  gamma * ||Du||_0  +  ||Au - f||_1
    by alternating between a direct L2-Potts step and an L1-Tikhonov step.

    Parameters
    ----------
    f : array_like, shape (m,)
        Observations / indirect measurements.
    gamma : float
        Jump penalty (must be > 0).
    A : array_like, shape (m, n)
        Forward / measurement operator.
    mu_init : float, optional
        Initial ADMM coupling. Default: gamma * 1e-6.
    mu_step : float, optional
        ADMM coupling growth factor (> 1). Default: 1.05.
    tol : float, optional
        Convergence tolerance. Default: 1e-6.
    imax : int, optional
        Inner iterations per coupling level. Default: 1.
    verbose : bool, optional
        Print iteration progress. Default: False.

    Returns
    -------
    u : ndarray, shape (n,)
    data_error : float  (||Au - f||_1)
    n_jumps : int
    energy : float

    Ported from Potts/minL1iPotts.m and Potts/PottsCore/iPottsADMM.m.
    """
    return _ipotts_admm(f, gamma, A, p=1, mu_init=mu_init, mu_step=mu_step,
                        tol=tol, imax=imax, verbose=verbose)


def _ipotts_admm(
    f: NDArray[np.floating],
    gamma: float,
    A: NDArray[np.floating],
    p: int,
    mu_init: Optional[float],
    mu_step: float,
    tol: float,
    imax: int,
    verbose: bool,
) -> Tuple[NDArray[np.floating], float, int, float]:
    """Internal ADMM solver for the inverse Potts problem.

    Direct port of Potts/PottsCore/iPottsADMM.m.
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if mu_step <= 1:
        raise ValueError("mu_step must be > 1.")
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2.")

    m, n = A.shape
    if f.shape != (m,):
        raise ValueError(f"f must have shape ({m},) to match A with shape ({m}, {n}).")

    if mu_init is None:
        mu_init = gamma * 1e-6

    # ADMM variables (following iPottsADMM.m naming)
    lam = 0.0
    v = A.T @ f           # initial guess (A^T f)
    w = np.zeros(n)
    u = np.full(n, np.inf)

    mu = mu_init
    iteration = 0

    while np.sum((u - v) ** 2) > tol:
        for _ in range(imax):
            # Step 1: solve direct L2-Potts (substitution: v - lambda/mu)
            u = min_l2_potts(v - lam / mu, 2.0 * gamma / mu)
            n_jumps = count_jumps(u)

            # Step 2: prepare RHS for Tikhonov step
            b_rhs = A @ (u + lam / mu) - f

            if p == 2:
                w, linflag = min_l2_tikhonov(b_rhs, mu / 2.0, A)
            else:
                w, linflag = min_l1_tikhonov(b_rhs, mu / 2.0, A)

            if linflag != 0 and verbose:
                print(f"Warning: linear solver did not converge (flag={linflag}).")

            # Resubstitution: v = u - w + lambda/mu
            v = u - w + lam / mu

            # Update multiplier
            lam = lam + mu * (u - v)

        mu *= mu_step
        iteration += 1

        if verbose and iteration % 5 == 0:
            print("*", end="", flush=True)

    if verbose:
        print(f" Done. ({iteration} outer iterations)")

    err = A @ u - f
    if p == 2:
        data_error = float(np.sum(err ** 2))
    else:
        data_error = float(np.sum(np.abs(err)))
    energy = gamma * n_jumps + data_error

    return u, data_error, n_jumps, energy
