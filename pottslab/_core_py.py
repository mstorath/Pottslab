"""Pure-Python fallback for pottslab._core (Rust/PyO3 extension).

Loaded automatically when the compiled extension is unavailable — e.g. after
``pip install .`` without a Rust toolchain, or on an unsupported platform.

All functions produce identical results to the Rust extension but without SIMD
or Rayon parallelism.  Expect 10–100× slower execution on large inputs.
A deprecation-style warning is printed once at import time so users know which
backend is active.

Public API mirrors the Rust module exactly so callers need no changes.
"""

import warnings
import numpy as np

warnings.warn(
    "pottslab: Rust extension (_core.so) not found — using pure-Python fallback. "
    "Performance will be significantly lower. "
    "Install a Rust toolchain and run `pip install .` (or `maturin develop --release`) "
    "to build the fast backend.",
    RuntimeWarning,
    stacklevel=2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_median(data, weights):
    """Weighted median via sort + cumulative sum (O(n log n))."""
    if len(data) == 0:
        return 0.0
    idx = np.argsort(data)
    cumw = np.cumsum(weights[idx])
    half = cumw[-1] / 2.0
    mi = np.searchsorted(cumw, half)
    return float(data[idx[min(mi, len(data) - 1)]])


def _l2potts_vv_inplace(data, vec_len, weights, gamma):
    """In-place O(n²) L2-Potts DP for row-major data of shape (n, vec_len).

    data    — flat array, length n * vec_len, modified in-place
    weights — 1-D weight array of length n, or None for uniform weights
    gamma   — jump penalty
    """
    n = len(data) // vec_len
    if n == 0:
        return

    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.ones(n)
    data2d = data.reshape(n, vec_len)

    # Weighted cumulative sums
    wf  = w[:, np.newaxis] * data2d           # (n, vec_len)
    wff = (wf * data2d).sum(axis=1)           # (n,) — weighted ||f||²

    cw  = np.zeros(n + 1);                cw[1:]  = np.cumsum(w)
    cs  = np.zeros((n + 1, vec_len));     cs[1:]  = np.cumsum(wf, axis=0)
    css = np.zeros(n + 1);               css[1:] = np.cumsum(wff)

    arr_p = np.full(n + 1, np.inf)
    arr_p[0] = -gamma
    arr_j = np.zeros(n + 1, dtype=np.intp)

    for r in range(1, n + 1):
        w_seg  = cw[r]  - cw[:r]          # (r,)
        s_seg  = cs[r]  - cs[:r]          # (r, vec_len)
        ss_seg = css[r] - css[:r]         # (r,)
        nz = w_seg > 0
        costs = np.where(
            nz,
            ss_seg - (s_seg ** 2).sum(axis=1) / np.where(nz, w_seg, 1.0),
            0.0,
        )
        cands = arr_p[:r] + gamma + costs
        best  = int(np.argmin(cands))
        arr_p[r] = cands[best]
        arr_j[r] = best

    # Reconstruct piecewise-constant signal
    r = n
    while r > 0:
        l = int(arr_j[r])
        wt = cw[r] - cw[l]
        val = (cs[r] - cs[l]) / wt if wt > 0 else np.zeros(vec_len)
        data2d[l:r] = val
        r = l

    data[:] = data2d.ravel()


# ─── Processor helpers (mirror processor.rs) ─────────────────────────────────

def _apply_horizontally(img, weights_2d, gamma):
    """Apply 1D L2-Potts along each row of img (H, W, C) in-place."""
    H, W, C = img.shape
    for i in range(H):
        row = np.ascontiguousarray(img[i].ravel())
        _l2potts_vv_inplace(row, C, weights_2d[i], gamma)
        img[i] = row.reshape(W, C)


def _apply_vertically(img, weights_2d, gamma):
    """Apply 1D L2-Potts along each column of img (H, W, C) in-place."""
    H, W, C = img.shape
    for j in range(W):
        col = np.ascontiguousarray(img[:, j, :].ravel())
        _l2potts_vv_inplace(col, C, weights_2d[:, j], gamma)
        img[:, j, :] = col.reshape(H, C)


def _diag_indices(rows, cols):
    diags = []
    for k in range(cols):
        length = min(rows, cols - k)
        diags.append([(j, j + k) for j in range(length)])
    for k in range(1, rows):
        length = min(rows - k, cols)
        diags.append([(j + k, j) for j in range(length)])
    return diags


def _anti_diag_indices(rows, cols):
    diags = []
    for k in range(cols):
        length = min(rows, cols - k)
        diags.append([(j, cols - 1 - (j + k)) for j in range(length)])
    for k in range(1, rows):
        length = min(rows - k, cols)
        diags.append([(j + k, cols - 1 - j) for j in range(length)])
    return diags


def _apply_diag(img, weights_2d, gamma):
    H, W, C = img.shape
    for diag in _diag_indices(H, W):
        if len(diag) < 2:
            continue
        ri = [p[0] for p in diag]
        ci = [p[1] for p in diag]
        seg = np.ascontiguousarray(img[ri, ci, :].ravel())
        _l2potts_vv_inplace(seg, C, weights_2d[ri, ci], gamma)
        seg2d = seg.reshape(len(diag), C)
        for s, (i, j) in enumerate(diag):
            img[i, j, :] = seg2d[s]


def _apply_anti_diag(img, weights_2d, gamma):
    H, W, C = img.shape
    for diag in _anti_diag_indices(H, W):
        if len(diag) < 2:
            continue
        ri = [p[0] for p in diag]
        ci = [p[1] for p in diag]
        seg = np.ascontiguousarray(img[ri, ci, :].ravel())
        _l2potts_vv_inplace(seg, C, weights_2d[ri, ci], gamma)
        seg2d = seg.reshape(len(diag), C)
        for s, (i, j) in enumerate(diag):
            img[i, j, :] = seg2d[s]


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (must match the Rust module's function signatures exactly)
# ─────────────────────────────────────────────────────────────────────────────

def solve_l2_potts_1d(f, gamma, weights=None):
    """O(n²) L2-Potts DP for a scalar 1-D signal."""
    data = np.ascontiguousarray(f, dtype=np.float64)
    _l2potts_vv_inplace(data, 1, weights, gamma)
    return data


def solve_l2_potts_vv(f, gamma, weights=None):
    """O(n²) L2-Potts DP for a vector-valued signal (n, channels)."""
    f = np.asarray(f, dtype=np.float64)
    n, ch = f.shape
    data = np.ascontiguousarray(f.ravel())
    _l2potts_vv_inplace(data, ch, weights, gamma)
    return data.reshape(n, ch)


def solve_l1_potts_1d(f, gamma, weights=None):
    """O(n²) L1-Potts DP with per-segment weighted median."""
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    if n == 0:
        return np.array([], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.ones(n)

    b    = np.full(n + 1, np.inf)
    b[0] = -gamma
    part = np.zeros(n + 1, dtype=np.intp)

    for r in range(1, n + 1):
        for l in range(r):
            seg_f = f[l:r]
            seg_w = w[l:r]
            med = _weighted_median(seg_f, seg_w)
            dev = float(np.sum(seg_w * np.abs(seg_f - med)))
            cand = b[l] + gamma + dev
            if cand < b[r]:
                b[r] = cand
                part[r] = l

    u = np.empty(n)
    r = n
    while r > 0:
        l = int(part[r])
        u[l:r] = _weighted_median(f[l:r], w[l:r])
        r = l
    return u


def weighted_median_rs(data, weights):
    """Weighted median of a 1-D array."""
    return _weighted_median(
        np.asarray(data, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def admm_4connected(img, gamma, weights, mu_init, mu_step, stop_tol, verbose):
    """2D L2-Potts via ADMM with 4-connected (anisotropic) neighbourhood."""
    img = np.asarray(img, dtype=np.float64).copy()
    H, W, C = img.shape
    f_norm = float(np.sum(img ** 2))
    if f_norm == 0.0:
        return img

    u   = np.zeros_like(img)
    v   = img.copy()
    lam = np.zeros_like(img)
    mu  = float(mu_init)
    error = np.inf
    n_iter = 0

    while error >= stop_tol * f_norm:
        gamma_prime = 2.0 * gamma
        wp_2d = weights + mu                                    # (H, W)
        wp    = wp_2d[:, :, np.newaxis]                        # (H, W, 1)

        # u-step: weighted average of data and v, then horizontal Potts
        u = (img * weights[:, :, np.newaxis] + v * mu - lam) / wp
        _apply_horizontally(u, wp_2d, gamma_prime)

        # v-step: weighted average of data and u, then vertical Potts
        v = (img * weights[:, :, np.newaxis] + u * mu + lam) / wp
        _apply_vertically(v, wp_2d, gamma_prime)

        # Multiplier update and convergence check
        diff  = u - v
        lam  += diff * mu
        error = float(np.sum(diff ** 2))

        mu *= mu_step
        n_iter += 1
        if verbose:
            print("*", end="", flush=True)

    if verbose:
        print(f"\nTotal iterations: {n_iter}")
    return u


def admm_8connected(img, gamma, weights, mu_init, mu_step, stop_tol, verbose,
                    omega_c, omega_d):
    """2D L2-Potts via ADMM with 8-connected (near-isotropic) neighbourhood."""
    img = np.asarray(img, dtype=np.float64).copy()
    H, W, C = img.shape
    f_norm = float(np.sum(img ** 2))
    if f_norm == 0.0:
        return img

    u     = np.zeros_like(img)
    v     = img.copy()
    w_img = img.copy()
    z     = img.copy()
    lam1  = np.zeros_like(img)   # u - v
    lam2  = np.zeros_like(img)   # u - w
    lam3  = np.zeros_like(img)   # u - z
    lam4  = np.zeros_like(img)   # v - w
    lam5  = np.zeros_like(img)   # v - z
    lam6  = np.zeros_like(img)   # w - z

    mu    = float(mu_init)
    error = np.inf
    n_iter = 0

    while error >= stop_tol * f_norm:
        gpc    = 4.0 * omega_c * gamma
        gpd    = 4.0 * omega_d * gamma
        wp_2d  = weights + 6.0 * mu                            # (H, W)
        wp     = wp_2d[:, :, np.newaxis]                       # (H, W, 1)
        fW     = img * weights[:, :, np.newaxis]

        # u-step (horizontal direction)
        u = (fW + 2.0 * mu * (w_img + v + z)
             + 2.0 * (-lam1 - lam2 - lam3)) / wp
        _apply_horizontally(u, wp_2d, gpc)

        # w-step (diagonal direction)
        w_img = (fW + 2.0 * mu * (u + v + z)
                 + 2.0 * (lam2 + lam4 - lam6)) / wp
        _apply_diag(w_img, wp_2d, gpd)

        # v-step (vertical direction)
        v = (fW + 2.0 * mu * (u + w_img + z)
             + 2.0 * (lam1 - lam4 - lam5)) / wp
        _apply_vertically(v, wp_2d, gpc)

        # z-step (anti-diagonal direction)
        z = (fW + 2.0 * mu * (u + w_img + v)
             + 2.0 * (lam3 + lam5 + lam6)) / wp
        _apply_anti_diag(z, wp_2d, gpd)

        # Multiplier updates
        lam1 += mu * (u - v)
        lam2 += mu * (u - w_img)
        lam3 += mu * (u - z)
        lam4 += mu * (v - w_img)
        lam5 += mu * (v - z)
        lam6 += mu * (w_img - z)

        diff  = u - v
        error = float(np.sum(diff ** 2))

        mu *= mu_step
        n_iter += 1
        if verbose:
            print("*", end="", flush=True)

    if verbose:
        print(f"\nTotal iterations: {n_iter}")
    return u
