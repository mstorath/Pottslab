# Pottslab 

Pottslab is a Matlab/Java toolbox for the reconstruction of 
jump-sparse and images using Potts functionals.
Applications include denoising of piecewise constant signals and 
image segmentation.

## Application examples

### Segmentation of vector-valued images


   - Supports segmentation of vector-valued images (e.g. multispectral images, feature images)
   - Linear complexity in number of color channels
   - Label-free: No label discretization required


![Vector-valued segmentation](/Docs/titleImage.png)

Left: A natural image; Right: Result using Potts model

![Vector-valued segmentation](/Docs/texture.png)

Texture segmentation using highdimensional curvelet-based feature vectors 

### Joint image reconstruction and segmentation


   - Applicable to many imaging operators, e.g. convolution, Radon transform, MRI, PET, MPI: only implementation of proximal mapping reuqired
    - Supports vector-valued data
    - Label-free: Labels need NOT be chosen a-priori


![Phantom](/Docs/radon/phantom.png)
![Phantom](/Docs/radon/recFBPRamLak.png)
![Phantom](/Docs/radon/recPotts.png)

Left: Shepp-Logan phantom; Center: Filtered backprojection from 7 angular projections; Right: Joint reconstruction and segmentation using the Potts model from 7 angular projections

<!---
![Phantom](/Docs/deconv/noisy.png)
![Phantom](/Docs/deconv/uPottsRho.png)

 Texture segmentation using highdimensional curvelet-based feature vectors --->

### Denoising of jump-sparse/piecewise-constant signals, or step detection/changepoint detection


   - L1 Potts model is robust to noise and to moderately blurred data
   - Fast and exact solver for L1 Potts model
   - Approximative strategies for severely blurred data


![Phantom](/Docs/potts1d.png)

Top: Noisy signal; Bottom: Minimizer of Potts functional (ground truth in red)


## Installation 
### Quickstart:
   - Run the script "installPottslab.m", it should set all necessary paths
   - Run a demo from the Demos folder

### Troubleshooting:
   * Problem: OutOfMemoryException
   * Solution: Increase Java heap space in the Matlab preferences

   * Problem: Undefined variable "pottslab" or class "pottslab.JavaTools.minL2Potts"
   * Solution: 
        - Run setPLJavaPath.m
        - Maybe you need to install Java 1.7 (see e.g. http://undocumentedmatlab.com/blog/using-java-7-in-matlab-r2013a-and-earlier)

## Plugins for Image Analysis GUIs
   - [Icy plugin](http://icy.bioimageanalysis.org/plugin/Potts_Segmentation) - an interactive image segmentation plugin based on Pottslab (written by Vasileios Angelopoulos)
   - [ImageJ plugin](http://www.pottslab.de/download/PottsSegmentationJ_.jar) - an ImageJ frontend for Pottslab (written by Michael Kaul) 

## References
- M. Storath, A. Weinmann, J. Frikel, M. Unser.
    "Joint image reconstruction and segmentation using the Potts model"
    Inverse Problems, 2015
- A. Weinmann, M. Storath. "Iterative Potts and Blake-Zisserman minimization for the recovery of functions with discontinuities from indirect measurements." Proceedings of The Royal Society A, 471(2176), 2015
- A. Weinmann, M. Storath, L. Demaret.
    "The L1-Potts functional for robust jump-sparse reconstruction"
    SIAM Journal on Numerical Analysis, 2015
- M. Storath, A. Weinmann.
    "Fast partitioning of vector-valued images"
    SIAM Journal on Imaging Sciences, 2014
- M. Storath, A. Weinmann, L. Demaret.
    "Jump-sparse and sparse recovery using Potts functionals"
    IEEE Transactions on Signal Processing, 2014


  
