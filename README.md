# Pottslab — Multilabel image segmentation via the Potts model

[![PyPI](https://img.shields.io/pypi/v/pottslab.svg)](https://pypi.org/project/pottslab/)
[![Python](https://img.shields.io/pypi/pyversions/pottslab.svg)](https://pypi.org/project/pottslab/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-supported-orange.svg)](#matlab)
[![View Pottslab on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://de.mathworks.com/matlabcentral/fileexchange/62641-pottslab-multilabel-segmentation-of-vectorial-data)

Unsupervised multilabel image segmentation (colour, grayscale, multichannel) via the Potts model — also known as the piecewise-constant Mumford-Shah model or the *ℓ⁰* gradient model. Solvers for 1-D denoising / step detection, 2-D image segmentation, and joint reconstruction-and-segmentation (deconvolution, Radon / MRI / PET).

![Vector-valued segmentation](/Docs/titleImage.png)

Featured in the MATLAB Central [Pick of the Week](https://blogs.mathworks.com/pick/2017/12/07/minimizing-energy-to-segment-images-or-cluster-data/) (2017).

## Quickstart

### Python (Rust core)

```bash
pip install pottslab
```

```python
import numpy as np
from pottslab import min_l2_potts, min_l2_potts_2d

# 1-D: denoise / step-detect a noisy piecewise-constant signal
y = np.array([0., 0.1, 0.05, 1.05, 0.95, 1.1, 0.0])
u = min_l2_potts(y, gamma=0.5)

# 2-D: segment a vector-valued image (H, W, C)
img = np.random.rand(64, 64, 3)
seg = min_l2_potts_2d(img, gamma=0.3)
```

The Python package wraps a Rust extension built with [PyO3](https://pyo3.rs)
and [maturin](https://maturin.rs); algorithm crates live under `src/`,
demos under [`demos_python/`](demos_python/).
See [`README_PYTHON.md`](README_PYTHON.md) for the full Python API,
inverse-Potts / sparsity / Tikhonov variants, and performance figures.

### MATLAB

The original MATLAB / Java reference implementation is in this same repository:

1. Run `installPottslab.m` to add the necessary folders to the MATLAB path.
2. For best performance, increase Java heap space in the MATLAB preferences (MATLAB → General → Java Heap Memory).
3. Run a demo from the `Demos/` folder.

#### Troubleshooting

- *OutOfMemoryException* — Increase Java heap space in the MATLAB preferences.
- *Undefined variable "pottslab" or class "pottslab.JavaTools.minL2Potts"* — Run `setPLJavaPath.m`. You may also need to install Java 1.7 (see e.g. [undocumentedmatlab.com](http://undocumentedmatlab.com/blog/using-java-7-in-matlab-r2013a-and-earlier)).

### Standalone (command line, plain image segmentation only)

```bash
java -jar pottslab-standalone.jar input output.png gamma
```

where `gamma` is a positive real number, e.g. `0.1` (thanks to fxtentacle).

## Application examples

### Segmentation of vector-valued images

- Supports vector-valued images (multispectral, feature images)
- Linear complexity in the number of channels
- Label-free: no label discretisation required

![Vector-valued segmentation](/Docs/titleImage.png)

*Left:* A natural image. *Right:* Result using the Potts model.

![Texture segmentation](/Docs/texture.png)

*Texture segmentation using high-dimensional curvelet-based feature vectors.*

Used as the segmentation method in:

- A. Breger et al. [*Supervised learning and dimension reduction techniques for quantification of retinal fluid in optical coherence tomography images,*](https://www.nature.com/articles/eye201761) Eye (2017).

### Joint image reconstruction and segmentation

- Applicable to many imaging operators (convolution, Radon, MRI, PET, MPI) — only the proximal mapping is needed
- Supports vector-valued data
- Label-free: labels need not be chosen a priori

![Phantom](/Docs/radon/phantom.png)
![FBP](/Docs/radon/recFBPRamLak.png)
![Potts](/Docs/radon/recPotts.png)

*Left:* Shepp-Logan phantom. *Centre:* Filtered backprojection from 7 angular projections. *Right:* Joint reconstruction and segmentation using the Potts model from 7 angular projections.

### Denoising / step detection of jump-sparse signals

- L1 Potts model is robust to noise and to moderately blurred data
- Fast and exact solver for the L1 Potts model
- Approximative strategies for severely blurred data

![Phantom](/Docs/potts1d.png)

*Top:* Noisy signal. *Bottom:* Minimiser of the Potts functional (ground truth in red).

Used as the step-detection algorithm in:

- A. Nord et al. [*Catch bond drives stator mechanosensitivity in the bacterial flagellar motor,*](http://www.pnas.org/content/early/2017/11/27/1716002114.full) PNAS, 2017.
- A. Szorkovszky et al. [*Assortative interactions revealed by sorting of animal groups,*](https://www.sciencedirect.com/science/article/pii/S0003347218301799) Animal Behaviour, 2018.

## Plugins for image-analysis GUIs

Parts of Pottslab can be used without MATLAB as pure Java plugins:

- [Icy plugin](http://icy.bioimageanalysis.org/plugin/potts-segmentation/) — interactive image segmentation based on Pottslab (Vasileios Angelopoulos).
- [ImageJ plugin](Plugins/PottsSegmentationJ_.jar) — ImageJ frontend for Pottslab (Michael Kaul).

## How to cite

If you use this software, please cite the relevant paper(s) below. GitHub's "Cite this repository" button on the repo page reads the `version` and `date-released` fields from [`CITATION.cff`](CITATION.cff) and renders BibTeX/APA.

## References

- M. Storath, A. Weinmann, J. Frikel, M. Unser. *Joint image reconstruction and segmentation using the Potts model,* Inverse Problems, 2015.
- A. Weinmann, M. Storath. *Iterative Potts and Blake-Zisserman minimization for the recovery of functions with discontinuities from indirect measurements,* Proceedings of the Royal Society A, 471(2176), 2015.
- A. Weinmann, M. Storath, L. Demaret. *The L1-Potts functional for robust jump-sparse reconstruction,* SIAM Journal on Numerical Analysis, 2015.
- M. Storath, A. Weinmann. *Fast partitioning of vector-valued images,* SIAM Journal on Imaging Sciences, 2014.
- M. Storath, A. Weinmann, L. Demaret. *Jump-sparse and sparse recovery using Potts functionals,* IEEE Transactions on Signal Processing, 2014.

## See also

Sibling projects from the same research program on variational methods for signal and image processing:

- [L1TV](https://github.com/mstorath/L1TV) — exact L1-TV regularisation of real- or circle-valued signals
- [CSSD](https://github.com/mstorath/CSSD) — cubic smoothing splines for signals with discontinuities
- [MumfordShah2D](https://github.com/mstorath/MumfordShah2D) — edge-preserving image restoration via the Mumford-Shah model
- [CircleMedianFilter](https://github.com/mstorath/CircleMedianFilter) — fast median filtering for phase or orientation data
- [DCEBE](https://github.com/mstorath/DCEBE) — bolus arrival time estimation for DCE-MRI signals

Related external projects:

- [PALMS Image Segmentation](https://github.com/lu-kie/PALMS_ImagePartitioning) — piecewise affine linear estimation.
- [Higher order Mumford-Shah models](https://github.com/lu-kie/HOMS_SignalProcessing) — piecewise smooth signals.

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

### Project history

The Python / Rust port of this codebase was generated from the original MATLAB / Java reference by a Claude coding agent in 2026. The agent also found and fixed an off-by-one bug in the weighted-median computation inside `IndexedLinkedHistogram` (in both the new Rust port and the original Java).
See [`PORTED_BY.md`](PORTED_BY.md) for full attribution and the porting plan.
