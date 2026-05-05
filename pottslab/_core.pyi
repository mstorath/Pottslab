# Type stubs for pottslab._core (Rust/PyO3 extension).
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
from numpy.typing import NDArray
from typing import Optional

def solve_l2_potts_1d(
    f: NDArray[np.float64],
    gamma: float,
    weights: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Solve 1-D scalar L2-Potts in-place and return solution."""
    ...

def solve_l2_potts_vv(
    f: NDArray[np.float64],
    gamma: float,
    weights: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Solve 1-D vector-valued L2-Potts. f shape (n, ch), returns (n, ch)."""
    ...

def solve_l1_potts_1d(
    f: NDArray[np.float64],
    gamma: float,
    weights: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Solve 1-D scalar L1-Potts and return solution."""
    ...

def weighted_median_rs(
    data: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Compute weighted median (Rust implementation)."""
    ...

def admm_4connected(
    img: NDArray[np.float64],
    gamma: float,
    weights: NDArray[np.float64],
    mu_init: float,
    mu_step: float,
    stop_tol: float,
    verbose: bool,
) -> NDArray[np.float64]:
    """2D ADMM (4-connected). img shape (H,W,C), weights (H,W)."""
    ...

def admm_8connected(
    img: NDArray[np.float64],
    gamma: float,
    weights: NDArray[np.float64],
    mu_init: float,
    mu_step: float,
    stop_tol: float,
    verbose: bool,
    omega_c: float,
    omega_d: float,
) -> NDArray[np.float64]:
    """2D ADMM (8-connected). img shape (H,W,C), weights (H,W)."""
    ...
