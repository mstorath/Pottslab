# Changelog

## 1.0.0 — 2026-05-05

First stable release of the Python/Rust port.

### What's included

- **L2-Potts 1D** (`min_l2_potts`) — O(n²) dynamic programming solver,
  scalar and vector-valued signals, optional per-sample weights and
  non-equidistant sample positions.
- **L1-Potts 1D** (`min_l1_potts`) — robust variant using weighted-median
  segments; O(n²) via the IndexedLinkedHistogram algorithm.
- **L2-Potts 2D** (`min_l2_potts_2d`) — ADMM splitting with 4-connected
  (anisotropic) and 8-connected (near-isotropic) neighbourhoods; parallel
  1D solves via Rayon.
- **Inverse Potts** (`min_l2_ipotts`, `min_l1_ipotts`) — joint
  deconvolution/deblurring and segmentation; accepts dense matrices,
  sparse matrices, and `scipy.sparse.linalg.LinearOperator`.
- **Tikhonov regularisation** (`min_l2_tikhonov`, `min_l1_tikhonov`).
- **Sparsity** (`min_l2_spars`, `min_l1_spars`) — hard and soft thresholding.
- **Utilities** — `count_jumps`, `energy_l2_potts`, `samples_to_weights`,
  `seg_to_label`, `soft_threshold`, `weighted_median`.
- **Pure-Python fallback** — all algorithms available without a Rust
  toolchain (10–100× slower; activated automatically on import if the
  compiled extension is absent).
- **418 tests** covering correctness, brute-force cross-validation,
  numerical precision, edge cases, and input-validation guards.

### Bug fixed (relative to original Java source)

**Off-by-one in weighted-median computation** inside
`IndexedLinkedHistogram`: the original code used
`cumulative_weight > half_total → median = previous_node`, which is
incorrect for inputs with odd total weight.  The correct convention is
`cumulative_weight >= half_total → median = current_node`.  The bug caused
the L1-Potts dynamic program to occasionally select a suboptimal multi-jump
solution when a single constant segment is provably optimal.

Fixed in both the Rust port (`src/l1potts.rs`) and the original Java source
(`Java/src/pottslab/IndexedLinkedHistogram.java`).

### Performance

All time-critical loops run in compiled Rust (via PyO3) with SIMD-friendly
layout and Rayon parallelism.  Typical speedup over the original Java JAR
(which was itself the performance layer called from MATLAB):

| Function | n / size | Speedup vs Java |
|---|---|---|
| `min_l2_potts` 1D scalar | 1 000 | **4×** |
| `min_l1_potts` 1D | 100 | **3.2×** |
| `min_l2_potts_2d` | 256×256 | **2.1×** |
| `min_l2_potts_2d` | 512×512 | **2.2×** |

Java timings exclude MATLAB script + MEX bridge overhead (typically 2–5× on
top of the raw Java time), so the effective user-facing speedup is larger.
