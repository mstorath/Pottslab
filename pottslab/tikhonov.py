# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
Tikhonov regularisation solvers.

Uses scipy.sparse.linalg for all linear algebra — CG for L2, LSQR for L1 —
which handles both dense matrices and LinearOperator objects efficiently.

Ported from:
  Tikhonov/minL2Tikhonov.m
  Tikhonov/minL1Tikhonov.m
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
import scipy.sparse
import scipy.sparse.linalg as spla


def min_l2_tikhonov(
    b: NDArray[np.floating],
    lam: float,
    A: Union[NDArray[np.floating], spla.LinearOperator],
    *,
    init: Optional[NDArray[np.floating]] = None,
    maxit: int = 500,
) -> Tuple[NDArray[np.floating], int]:
    """Minimise the L2-Tikhonov functional.

    Solves  min_u  lam * ||u||_2^2  +  ||Au - b||_2^2
    equivalently:  (A^T A + lam I) u = A^T b
    via CG (scipy.sparse.linalg.cg), accepting dense or sparse A.

    Parameters
    ----------
    b : array_like, shape (m,)
    lam : float  — regularisation weight
    A : (m, n) matrix or LinearOperator
    init : optional initial guess, shape (n,)
    maxit : max CG iterations

    Returns
    -------
    u : ndarray, shape (n,)
    flag : int (0 = converged)

    Ported from Tikhonov/minL2Tikhonov.m.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    if lam < 0:
        raise ValueError("lam must be >= 0.")

    if isinstance(A, np.ndarray):
        n = A.shape[1]
        AtA = A.T @ A
        Atb = A.T @ b
        # Use scipy.sparse for the solve — handles both dense and sparse
        lhs = AtA + lam * np.eye(n)
        u, flag = spla.cg(lhs, Atb, x0=init, maxiter=maxit, atol=1e-10)
    elif scipy.sparse.issparse(A):
        n = A.shape[1]
        AtA = (A.T @ A).toarray()
        Atb = A.T @ b
        lhs = AtA + lam * np.eye(n)
        u, flag = spla.cg(lhs, Atb, x0=init, maxiter=maxit, atol=1e-10)
    else:
        # LinearOperator
        n = A.shape[1]
        def matvec(x: NDArray) -> NDArray:
            return A.rmatvec(A.matvec(x)) + lam * x  # type: ignore[attr-defined]
        op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        Atb = A.rmatvec(b)  # type: ignore[attr-defined]
        u, flag = spla.cg(op, Atb, x0=init, maxiter=maxit, atol=1e-10)

    return u, int(flag)


def min_l1_tikhonov(
    b: NDArray[np.floating],
    lam: float,
    A: Union[NDArray[np.floating], spla.LinearOperator],
    *,
    maxit: int = 100,
) -> Tuple[NDArray[np.floating], int]:
    """Minimise the L1-Tikhonov functional.

    Approximately solves  min_u  lam * ||u||_2^2  +  ||Au - b||_1
    via Iteratively Reweighted Least Squares (IRLS):
    at each step solves  (A^T W_k A + lam I) u = A^T W_k b  with
    diagonal W_k = diag(1 / (|r_k| + eps)).

    Uses scipy.sparse.linalg.lsqr for the inner linear solve.

    Ported from Tikhonov/minL1Tikhonov.m.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    if lam < 0:
        raise ValueError("lam must be >= 0.")
    if scipy.sparse.issparse(A):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"b must have shape ({m},) to match A with shape ({m}, {n}).")
    eps = 1e-8

    u = np.zeros(n)
    flag = 0

    for _ in range(maxit):
        residual = A @ u - b
        w_diag = 1.0 / (np.abs(residual) + eps)
        # Build sqrt-weighted system: (sqrt(W) A) u = sqrt(W) b  + lam regulariser
        sqrt_w = np.sqrt(w_diag)
        AW = sqrt_w[:, np.newaxis] * A
        bW = sqrt_w * b
        # Augment with Tikhonov rows: stack [AW; sqrt(lam)*I] and [bW; 0]
        sqrt_lam = np.sqrt(lam)
        AW_aug = np.vstack([AW, sqrt_lam * np.eye(n)])
        bW_aug = np.concatenate([bW, np.zeros(n)])
        # lsqr minimises ||Ax - b||_2 directly
        result = spla.lsqr(AW_aug, bW_aug, iter_lim=500, atol=1e-10, btol=1e-10)
        u_new = result[0]
        if np.max(np.abs(u_new - u)) < 1e-8:
            break
        u = u_new
        flag = result[1]  # stop condition

    return u, flag
