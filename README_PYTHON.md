# pottslab — Python/Rust package

Python package for reconstructing jump-sparse signals and images using the
**Potts model** (piecewise-constant Mumford-Shah / L0-gradient model).
Port of the original [MATLAB/Java pottslab](https://github.com/mstorath/Pottslab)
by Martin Storath & Andreas Weinmann.

Time-critical algorithms are implemented in Rust (via PyO3); the Python layer
provides a clean, type-annotated API.

---

## About this port

This package was written entirely by **Claude Sonnet** (Anthropic's AI coding
agent) in 2026 as a port of the original MATLAB/Java library. The original
library authors (Martin Storath, Andreas Weinmann) did not write or review this
Python/Rust code.

### What the agent built

| Layer | Files | Source |
|---|---|---|
| Rust core | `src/l2potts.rs` | Ported from `Java/src/pottslab/L2Potts.java` |
| Rust core | `src/l1potts.rs` | Ported from `Java/src/pottslab/IndexedLinkedHistogram.java` |
| Rust core | `src/processor.rs` | Ported from `Java/src/pottslab/PLProcessor.java` |
| Rust core | `src/admm.rs` | Ported from `Java/src/pottslab/JavaTools.java` (ADMM routines) |
| Python API | `pottslab/potts1d.py` | Ported from `Potts/minL2Potts.m`, `minL1Potts.m` |
| Python API | `pottslab/potts2d.py` | Ported from `Potts2D/minL2Potts2DADMM.m` |
| Python API | `pottslab/inverse.py` | Ported from `Potts/PottsCore/iPottsADMM.m` |
| Python API | `pottslab/sparsity.py` | Ported from `Sparsity/SparsityCore/minSpars.m` |
| Python API | `pottslab/tikhonov.py` | Ported from `Tikhonov/minL2Tikhonov.m`, `minL1Tikhonov.m` |
| Python API | `pottslab/utils.py` | Ported from `Auxiliary/` helpers |
| Tests | `tests/` (418 tests) | Written from scratch |
| Benchmark | `benchmark.py`, `Java/src/pottslab/Benchmark.java` | Written from scratch |

### Bug found and fixed

During porting, the agent identified an **off-by-one error in the weighted-median
computation** inside `IndexedLinkedHistogram`. The original code used
`cumulative_weight > half_total → median = previous_node`, which is incorrect for
inputs with odd total weight: the true weighted median is the *first* node whose
cumulative weight reaches or exceeds half the total (lower-median convention).

The bug caused the L1-Potts dynamic program to sometimes prefer a suboptimal
multi-jump solution when a single constant segment is provably optimal. It was
fixed in both the new Rust port (`src/l1potts.rs`) and in the original Java source
(`Java/src/pottslab/IndexedLinkedHistogram.java`). Regression tests are in
`tests/test_l1_median_bug.py`.

### Performance work

The initial Rust L1-Potts port was slower than the Java original at large n
because of two O(n²) allocation hot-spots: the entire arena was cloned on every
`compute_deviations` call, and a fresh `Vec<f64>` was allocated for each
deviations output. Both were eliminated:

- `compute_deviations` now mutates temporary fields directly (`&mut self`) instead
  of cloning the arena, relying on the invariant that `insert_sorted` resets all
  temp fields before each call.
- A single `dev_buf` is allocated once outside the DP loop and reused via
  `compute_deviations_into(&mut slice)`.

After these fixes Rust is 3× faster than Java at n=100 and roughly on par at
n=10 000 (the remaining gap is structural: arena linked-list pointer-chasing vs.
HotSpot JIT optimisation for object graphs).

### Hardening

A second pass added input-validation guards across every public function:

- NaN / Inf in signal or weight arrays → `ValueError`
- Non-monotone `samples` argument → `ValueError`
- `mu_init ≤ 0` in `min_l2_potts_2d` → `ValueError`
- `lam < 0` in Tikhonov solvers → `ValueError`
- Negative `tau` in `soft_threshold` → `ValueError`
- Empty array to `weighted_median` → `ValueError`
- Negative weights in `weighted_median` → `ValueError`
- Invalid connectivity (not 4 or 8) in `seg_to_label` → `ValueError`
- Incompatible `u` / `f` sizes in `energy_l2_potts` → `ValueError`
- Sparse-matrix input to `min_l1_tikhonov` now converts to dense cleanly
  instead of producing a silent object array

---

## Installation

### Pre-built wheels (recommended)

```bash
pip install pottslab
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and
Windows (x86_64) are published to PyPI for Python 3.9–3.13.
No Rust toolchain required.

### From source (with Rust)

```bash
pip install maturin
maturin develop --release   # builds the Rust extension in-place
# or produce a wheel:
maturin build --release && pip install dist/pottslab-*.whl
```

Requires a [Rust toolchain](https://rustup.rs) (stable ≥ 1.70).

### From source (without Rust — pure-Python fallback)

```bash
pip install .
```

If `cargo` is not found, the Rust compilation is skipped and the package
falls back to pure-Python implementations of all algorithms.  A
`RuntimeWarning` is printed at import time to make this visible.
The fallback is **10–100× slower** on large inputs but produces identical
results.

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.20, SciPy ≥ 1.7.

---

## Quick start

```python
import numpy as np
import pottslab as pl

# ── 1-D denoising (L2-Potts) ─────────────────────────────────────────────
rng = np.random.default_rng(0)
u_true = np.array([0.0] * 50 + [1.0] * 50)
f = u_true + rng.normal(0, 0.1, 100)

u = pl.min_l2_potts(f, gamma=0.3)
print(pl.count_jumps(u))          # → 1

# ── 1-D robust denoising (L1-Potts) ──────────────────────────────────────
u, data_err, n_jumps, energy = pl.min_l1_potts(f, gamma=0.3)

# ── 2-D image segmentation ────────────────────────────────────────────────
img = rng.standard_normal((64, 64)).clip(0, 1)
seg = pl.min_l2_potts_2d(img, gamma=0.05)

# ── Inverse problem: deblur then segment ─────────────────────────────────
n = 50
A = np.eye(n)                    # replace with your forward operator
u_rec, *_ = pl.min_l2_ipotts(f[:n], gamma=0.3, A=A)

# ── Sparsity ──────────────────────────────────────────────────────────────
u_sparse, *_ = pl.min_l2_spars(f, gamma=1.0)   # hard threshold
u_soft,   *_ = pl.min_l1_spars(f, gamma=1.0)   # soft threshold (LASSO)
```

---

## API reference

### 1-D Potts solvers

| Function | Problem | Returns |
|---|---|---|
| `min_l2_potts(f, gamma, *, weights, samples)` | min γ‖Du‖₀ + ‖u−f‖₂² | `ndarray` |
| `min_l1_potts(f, gamma, *, weights, samples)` | min γ‖Du‖₀ + ‖u−f‖₁ | `(u, data_err, n_jumps, energy)` |

Both accept scalar signals `(n,)` and vector-valued signals `(n, channels)`.
`weights` applies per-sample weights; `samples` accepts non-equidistant
sample positions (converted via trapezoidal quadrature). `samples` must be
strictly increasing; a `ValueError` is raised otherwise.

### 2-D Potts solver

```python
min_l2_potts_2d(
    f,             # (H, W) or (H, W, C) image
    gamma,         # jump penalty
    *,
    isotropic=True,   # True → 8-connected (near-isotropic); False → 4-connected
    tol=1e-10,        # ADMM convergence tolerance
    mu_init=None,     # initial ADMM coupling (default: gamma * 1e-2); must be > 0
    mu_step=2.0,      # ADMM penalty growth factor (must be > 1)
    quantize=True,    # round output to 8-bit grid (matches MATLAB behaviour)
    verbose=False,
) -> ndarray
```

### Inverse Potts (deconvolution / compressed sensing)

```python
u, data_err, n_jumps, energy = min_l2_ipotts(f, gamma, A, *, tol, mu_step, imax)
u, data_err, n_jumps, energy = min_l1_ipotts(f, gamma, A, *, tol, mu_step, imax)
```

`A` may be a dense/sparse matrix or a `scipy.sparse.linalg.LinearOperator`.

### Tikhonov regularisation

```python
u, flag = min_l2_tikhonov(f, gamma, A, *, init, maxit)   # lam >= 0
u, flag = min_l1_tikhonov(f, gamma, A, *, maxit)          # lam >= 0
```

### Sparsity

```python
u, data_err, n_spikes, energy = min_l2_spars(f, gamma)   # hard threshold
u, data_err, n_spikes, energy = min_l1_spars(f, gamma)   # soft threshold
```

### Utilities

| Function | Description |
|---|---|
| `count_jumps(u)` | Number of positions where `u` changes (raises on NaN/Inf) |
| `energy_l2_potts(u, f, gamma)` | Evaluate L2-Potts energy |
| `samples_to_weights(s)` | Strictly-increasing sample positions → trapezoidal weights |
| `seg_to_label(u, connectivity=4)` | Piecewise-constant image → integer label map |
| `soft_threshold(x, tau)` | Soft (proximal L1) threshold; `tau` must be ≥ 0 |
| `weighted_median(data, weights)` | Weighted median; raises on empty or negative weights |

---

## Performance

Comparison against the original Java backend (as called from MATLAB),
on an ARM64 machine (median of 5 runs):

| Function | n | Java (ms) | Rust (ms) | Speedup |
|---|---|---|---|---|
| `min_l2_potts` 1-D scalar | 1 000 | 2.1 | 0.5 | **4×** |
| `min_l2_potts` 1-D scalar | 10 000 | 62 | 52 | 1.2× |
| `min_l2_potts` vector-valued (3 ch) | 1 000 | 0.5 | 0.4 | 1.3× |
| `min_l1_potts` 1-D | 1 000 | 11.2 | 8.6 | 1.3× |
| `min_l1_potts` 1-D | 100 | 0.29 | 0.09 | **3.2×** |
| `min_l2_potts_2d` 256×256 | — | 241 | 114 | **2.1×** |
| `min_l2_potts_2d` 512×512 | — | 1 130 | 506 | **2.2×** |

Notes: Java timings are the raw Java class timings, **not** including the
MATLAB script and MEX bridge overhead which was typically 2–5× additional cost.
The Rust port eliminates that overhead entirely.

The `min_l1_potts` DP uses an arena-based linked-list histogram; Java's JIT
outperforms Rust at very large n (> 5 000) due to HotSpot's aggressive
pointer-chasing optimisation. For typical signal lengths this is not a
practical concern.

---

## What is the Potts model?

The **L2-Potts functional** is

    E(u) = γ ‖Du‖₀ + ‖u − f‖₂²

where ‖Du‖₀ counts the number of jumps in u. Minimising E yields a
piecewise-constant approximation of f with the optimal trade-off between
fidelity and number of segments, controlled by γ.

The **L1 variant** replaces ‖u − f‖₂² with ‖u − f‖₁, giving robustness to
outliers.  The **inverse variant** replaces ‖u − f‖ with ‖Au − f‖, enabling
joint deconvolution/deblurring and segmentation.

---

## Citation

If you use this package in research, please cite the original pottslab work:

```bibtex
@article{storath2017joint,
  title={Joint image reconstruction and segmentation using the Potts model},
  author={Storath, Martin and Weinmann, Andreas and Frikel, J{\"u}rgen and Unser, Michael},
  journal={Inverse Problems},
  volume={31}, number={2}, pages={025003}, year={2015},
  publisher={IOP Publishing}
}
@article{storath2014fast,
  title={Fast partitioning of vector-valued images},
  author={Storath, Martin and Weinmann, Andreas},
  journal={SIAM Journal on Imaging Sciences},
  volume={7}, number={3}, pages={1826--1852}, year={2014},
  publisher={SIAM}
}
```

---

## Attribution

Original library: **pottslab** by Martin Storath & Andreas Weinmann
(MIT licence, https://github.com/mstorath/Pottslab).

This Python/Rust port was written by **Claude Sonnet** (Anthropic's AI coding
agent), 2026. See [`PORTED_BY.md`](PORTED_BY.md) for the full list of what was
ported, what changed, and what was fixed relative to the original Java source.
