# Port Attribution

This Python/Rust package (`pottslab`) is an **automated port** of the original
MATLAB/Java *pottslab* library:

| | |
|---|---|
| **Original authors** | Martin Storath, Andreas Weinmann |
| **Original license** | MIT |
| **Original language** | MATLAB (high-level), Java (performance core) |

## Port details

| | |
|---|---|
| **Port author** | Claude Sonnet coding agent (Anthropic, 2026) |
| **Port language** | Python (API layer) + Rust/PyO3 (performance core) |
| **Port license** | MIT (unchanged) |

## What was ported

All algorithms are faithful translations of the original MATLAB/Java implementations:

- **`src/l2potts.rs`** — Direct port of `Java/src/pottslab/L2Potts.java::call()`:
  O(n²) dynamic programming solver for the L2-Potts functional with early-termination
  acceleration and piecewise-constant reconstruction.
- **`src/l1potts.rs`** — Direct port of `Java/src/pottslab/IndexedLinkedHistogram.java`:
  sorted linked-list histogram for efficient L1-Potts DP.
- **`src/processor.rs`** — Direct port of `Java/src/pottslab/PLProcessor.java`:
  parallel (Rayon) application of 1D Potts in horizontal, vertical, diagonal,
  and anti-diagonal directions.
- **`src/admm.rs`** — Direct port of `Java/src/pottslab/JavaTools.java::minL2PottsADMM4/8()`:
  ADMM splitting for the 2D Potts problem with 4- and 8-connected neighborhoods.
- **`pottslab/inverse.py`** — Direct port of `Potts/PottsCore/iPottsADMM.m`:
  inverse Potts solver for `γ‖Du‖₀ + ‖Au-f‖ₚ` with scipy linear sub-solvers.
- **`pottslab/sparsity.py`** — Port of `Sparsity/SparsityCore/minSpars.m`.
- **`pottslab/tikhonov.py`** — Port of `Tikhonov/minL2Tikhonov.m` and `minL1Tikhonov.m`.
- **`pottslab/utils.py`** — Port of helpers in `Auxiliary/`.

## What changed

- API names follow Python conventions (`min_l2_potts` instead of `minL2Potts`).
- Data is passed as NumPy arrays (float64) instead of MATLAB matrices.
- Performance-critical loops are in Rust (replacing the Java JAR).
- The 2D ADMM and directional processor use Rayon for parallelism.
- The Python API layer uses keyword-only arguments with the same defaults as MATLAB.

## Academic references (from original authors)

1. M. Storath, A. Weinmann, J. Frikel, M. Unser. "Joint image reconstruction and
   segmentation using the Potts model." *Inverse Problems*, 2015.
2. A. Weinmann, M. Storath. "Iterative Potts and Blake-Zisserman minimization
   on the recovery of functions with discontinuities from indirect measurements."
   *Proc. Royal Society A*, 2015.
3. A. Weinmann, M. Storath, L. Demaret. "The L1-Potts functional for robust
   jump-sparse reconstruction." *SIAM Journal on Numerical Analysis*, 2015.
4. M. Storath, A. Weinmann. "Fast partitioning of vector-valued images."
   *SIAM Journal on Imaging Sciences*, 2014.
5. M. Storath, A. Weinmann, L. Demaret. "Jump-sparse and sparse recovery using
   Potts functionals." *IEEE Transactions on Signal Processing*, 2014.
