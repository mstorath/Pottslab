Pottslab - A Matlab toolbox for jump-sparse recovery using Potts functionals

Authors:
    Development Team:
        Martin Storath (EPFL)
        Andreas Weinmann (Helmholtz Zentrum Muenchen)
    Contributors:
        Laurent Demaret (Helmholtz Zentrum Muenchen)
        Juergen Frikel (Helmholtz Zentrum Muenchen)
        Kilian Hohm (TU Muenchen)

Description:
    Pottslab is a Matlab toolbox for the reconstruction of 
    jump-sparse and images using Potts functionals.
    Applications include denoising of piecewise constant signals and 
    image segmentation.

System Requirements:
    Some functions require the Image Processing Toolbox
    Some functions may require Matlab 2014a

Quickstart:
    - Run the script "installPottslab.m", it should set all necessary paths
    - Run a demo from the Demos folder

Troubleshooting:
   * Problem: OutOfMemoryException
   * Solution: Increase Java heap space in the Matlab preferences

   * Problem: Undefined variable "pottslab" or class "pottslab.JavaTools.minL2Potts"
   * Solution: 
        - Run setPLJavaPath.m
        - Maybe you need to install Java 1.7 (see e.g. http://undocumentedmatlab.com/blog/using-java-7-in-matlab-r2013a-and-earlier)

Web:
    http://pottslab.de

Terms of use:
  * Pottslab is under the GNU-GPL3 (see file License.txt).
  * Please cite our corresponding publications (see file HowToCite.txt).

Development platform:
    This software was developed and tested under Matlab 2014b and Java 1.7 
    on Mac OS 10.9.

Comments and suggestions:
    martin.storath@epfl.ch

If you use functions of Pottslab please cite the corresponding 
publications:
 
[4] M. Storath, A. Weinmann, J. Frikel, M. Unser
    "Joint image reconstruction and segmentation using the Potts model"
    Inverse Problems, 2015

[3] M. Storath, A. Weinmann
    "Fast partitioning of vector-valued images"
    SIAM Journal on Imaging Sciences, 2014

[2] M. Storath, A. Weinmann, L. Demaret
    "Jump-sparse and sparse recovery using Potts functionals"
    IEEE Transactions on Signal Processing, 2014

[1] A. Weinmann, M. Storath, L. Demaret
    "The $L^1$-Potts functional for robust jump-sparse reconstruction"
    SIAM Journal on Numerical Analysis, 2015




  