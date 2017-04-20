# Pottslab 

Pottslab is a Matlab toolbox for the reconstruction of 
jump-sparse and images using Potts functionals.
Applications include denoising of piecewise constant signals and 
image segmentation.


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
If you use functions of Pottslab please cite the corresponding 
publications:
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

  
