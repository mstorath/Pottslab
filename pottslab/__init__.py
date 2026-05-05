# Ported from MATLAB/Java pottslab (Storath & Weinmann) by Claude Sonnet coding agent, Anthropic, 2026.
"""
pottslab — Jump-sparse signal and image reconstruction via the Potts model.

This package is an **automated port** of the original MATLAB/Java pottslab library
(Martin Storath, Andreas Weinmann) to Python, performed by Claude Sonnet coding
agent (Anthropic, 2026). Time-critical algorithms are implemented in Rust (via PyO3)
and compiled as a C extension; all other code is pure Python/NumPy/SciPy.

Core algorithms
---------------
The Potts model (piecewise-constant Mumford-Shah model) solves problems of the form

    min_u  gamma * ||Du||_0  +  data_fidelity(u, f)

where ||Du||_0 counts jump discontinuities and gamma controls the trade-off
between smoothness and data fit.

Public API
----------
1-D solvers (O(n²) dynamic programming):
  min_l2_potts(f, gamma)          — L2 data fidelity, scalar or vector-valued
  min_l1_potts(f, gamma)          — L1 data fidelity, robust to outliers

2-D solver (ADMM splitting):
  min_l2_potts_2d(f, gamma)       — image segmentation, 4- or 8-connected

Inverse problems (ADMM):
  min_l2_ipotts(f, gamma, A)      — indirect measurements, L2 fidelity
  min_l1_ipotts(f, gamma, A)      — indirect measurements, L1 fidelity

Sparsity (thresholding):
  min_l2_spars(f, gamma)          — hard threshold
  min_l1_spars(f, gamma)          — soft threshold

Tikhonov regularisation:
  min_l2_tikhonov(f, lam, A)      — CG solver
  min_l1_tikhonov(f, lam, A)      — IRLS solver

Utilities:
  count_jumps, energy_l2_potts, samples_to_weights,
  seg_to_label, soft_threshold, weighted_median

References
----------
1. M. Storath, A. Weinmann, J. Frikel, M. Unser. "Joint image reconstruction and
   segmentation using the Potts model." Inverse Problems, 2015.
2. A. Weinmann, M. Storath. "Iterative Potts and Blake-Zisserman minimization..."
   Proc. Royal Society A, 2015.
3. A. Weinmann, M. Storath, L. Demaret. "The L1-Potts functional for robust
   jump-sparse reconstruction." SIAM Journal on Numerical Analysis, 2015.
4. M. Storath, A. Weinmann. "Fast partitioning of vector-valued images."
   SIAM Journal on Imaging Sciences, 2014.
5. M. Storath, A. Weinmann, L. Demaret. "Jump-sparse and sparse recovery using
   Potts functionals." IEEE Transactions on Signal Processing, 2014.
"""

__version__ = "1.0.0"
__original_authors__ = "Martin Storath, Andreas Weinmann"
__ported_by__ = (
    "Claude Sonnet coding agent (Anthropic, 2026) — "
    "automated port from MATLAB/Java pottslab"
)

from pottslab.potts1d import min_l2_potts, min_l1_potts
from pottslab.potts2d import min_l2_potts_2d
from pottslab.inverse import min_l2_ipotts, min_l1_ipotts
from pottslab.sparsity import min_l2_spars, min_l1_spars
from pottslab.tikhonov import min_l2_tikhonov, min_l1_tikhonov
from pottslab.utils import (
    count_jumps,
    energy_l2_potts,
    samples_to_weights,
    seg_to_label,
    soft_threshold,
    weighted_median,
)

__all__ = [
    # 1-D Potts
    "min_l2_potts",
    "min_l1_potts",
    # 2-D Potts
    "min_l2_potts_2d",
    # Inverse Potts
    "min_l2_ipotts",
    "min_l1_ipotts",
    # Sparsity
    "min_l2_spars",
    "min_l1_spars",
    # Tikhonov
    "min_l2_tikhonov",
    "min_l1_tikhonov",
    # Utilities
    "count_jumps",
    "energy_l2_potts",
    "samples_to_weights",
    "seg_to_label",
    "soft_threshold",
    "weighted_median",
    # Metadata
    "__version__",
    "__original_authors__",
    "__ported_by__",
]
