# Pottslab 

Pottslab is a Matlab/Java toolbox for the reconstruction of 
jump-sparse signals and images using the Potts model (also known as "piecewise constant Mumford-Shah model" or "l0 gradient model").
Applications include denoising of piecewise constant signals, step detection and 
segmentation of multichannel image.

-- See also the <a href="https://blogs.mathworks.com/pick/2017/12/07/minimizing-energy-to-segment-images-or-cluster-data/">Pick of the Week</a> by Mathworks --

## Application examples

### Segmentation of vector-valued images


   - Supports segmentation of vector-valued images (e.g. multispectral images, feature images)
   - Linear complexity in number of color channels
   - Label-free: No label discretization required

![Vector-valued segmentation](/Docs/titleImage.png)

Left: A natural image; Right: Result using Potts model

![Vector-valued segmentation](/Docs/texture.png)

Texture segmentation using highdimensional curvelet-based feature vectors 

Used as segmentation method in
   - A. Breger et al., [Supervised learning and dimension reduction techniques for quantification of retinal fluid in optical coherence tomography images,](https://www.nature.com/articles/eye201761) Eye (2017).

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

Used as step detection algorithm in 

   * A. Nord et al., [Catch bond drives stator mechanosensitivity in the bacterial flagellar motor](http://www.pnas.org/content/early/2017/11/27/1716002114.full), Proceedings of the National Academy of Sciences, 2017
   * A. Szorkovszky et al., [Assortative interactions revealed by sorting of animal groups](https://www.sciencedirect.com/science/article/pii/S0003347218301799), Animal Behaviour, 2018

## Usage Instructions
### Standalone usage from command line (only image plain image segmentation supported)
   - Call "java -jar pottslab-standalone.jar input output.png gamma" where gamma is a positive real number, e.g. 0.1 (thanks to fxtentacle)

### Installation for Matlab (all features usable)
#### Quickstart:
   - Run the script "installPottslab.m", it should set all necessary paths
   - For best performance, increase Java heap space in the Matlab preferences (MATLAB - General - Java heap memory)
   - Run a demo from the Demos folder

#### Troubleshooting:
   * Problem: OutOfMemoryException
   * Solution: Increase Java heap space in the Matlab preferences (MATLAB - General - Java heap memory)

   * Problem: Undefined variable "pottslab" or class "pottslab.JavaTools.minL2Potts"
   * Solution: 
        - Run setPLJavaPath.m
        - Maybe you need to install Java 1.7 (see e.g. http://undocumentedmatlab.com/blog/using-java-7-in-matlab-r2013a-and-earlier)

## Plugins for Image Analysis GUIs
Parts of Pottslab can be used without Matlab as pure Java plugins
   - [Icy plugin](http://icy.bioimageanalysis.org/plugin/Potts_Segmentation) - an interactive image segmentation plugin based on Pottslab (written by Vasileios Angelopoulos)
   - [ImageJ plugin](Plugins/PottsSegmentationJ_.jar) - an ImageJ frontend for Pottslab (written by Michael Kaul) 

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


  
