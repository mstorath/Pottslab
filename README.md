# Pottslab 

Pottslab is a Matlab toolbox for the reconstruction of 
jump-sparse and images using Potts functionals.
Applications include denoising of piecewise constant signals and 
image segmentation.

## Application examples

### Segmentation of vector-valued images
![Vector-valued segmentation](/Docs/titleImage.png)

Left: A natural image; Right: Result using Potts model

### Joint image reconstruction and segmentation

![Phantom](/Docs/radon/phantom.png)
![Phantom](/Docs/radon/recFBPRamLak.png)
![Phantom](/Docs/radon/recPotts.png)

Left: Shepp-Logan phantom; Center: Filtered backprojection from 7 angular projections; Right: Joint reconstruction and segmentation using the Potts model from 7 angular projections

<!---
![Phantom](/Docs/deconv/noisy.png)
![Phantom](/Docs/deconv/uPottsRho.png)

 Texture segmentation using highdimensional curvelet-based feature vectors --->

### Denoising of jump-sparse/piecewise-constant signals, or step detection/changepoint detection

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

### References
- M. Storath, A. Weinmann, J. Frikel, M. Unser
    "Joint image reconstruction and segmentation using the Potts model"
    Inverse Problems, 2015
- M. Storath, A. Weinmann
    "Fast partitioning of vector-valued images"
    SIAM Journal on Imaging Sciences, 2014
- M. Storath, A. Weinmann, L. Demaret
    "Jump-sparse and sparse recovery using Potts functionals"
    IEEE Transactions on Signal Processing, 2014
- A. Weinmann, M. Storath, L. Demaret
    "The $L^1$-Potts functional for robust jump-sparse reconstruction"
    SIAM Journal on Numerical Analysis, 2015

  
