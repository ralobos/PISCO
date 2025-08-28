# PISCO Software v2.0

This software corresponds to a newer version of the original PISCO software, 
which is available at: http://mr.usc.edu/download/pisco/

## Overview

This MATLAB software performs subspace-based estimation of sensitivity maps 
for multichannel MRI reconstruction. It follows a nullspace-based approach 
according to the theoretical framework and computational methods originally 
proposed in [1] and [2].

## Key Improvements

The main differences with previous versions are:

**1. Sketched SVD Method:**
   A new computational method based on sketched SVD has been added. This 
   method has been recently proposed in [3].

**2. Expanded 2D/3D Data Support:**
   The software now supports both 2D and 3D data processing.

**3. Page-wise Matrix Operations:**
   Functions based on page-wise matrix operations have been implemented to 
   improve performance.

## Contents

- **Main Function**: `PISCO_sensitivity_maps_estimation.m` - Contains all the PISCO 
  computational methods
- **Example Files**: `example_2D.m` and `example_3D.m` - Demonstration 
  scripts for 2D and 3D data processing
- **Documentation**: `PISCO_software_v2_0_documentation.pdf` - Detailed 
  documentation in PDF format

## Getting Started

Please refer to the included PDF documentation and example files to begin 
using the software.

## References

**[1]** R. A. Lobos, C.-C. Chan, J. P. Haldar. New Theory and Faster
        Computations for Subspace-Based Sensitivity Map Estimation in
        Multichannel MRI. *IEEE Transactions on Medical Imaging* 
        43:286-296, 2024.

**[2]** R. A. Lobos, C.-C. Chan, J. P. Haldar. Extended Version of "New 
        Theory and Faster Computations for Subspace-Based Sensitivity Map 
        Estimation in Multichannel MRI", 2023, arXiv:2302.13431.
        (https://arxiv.org/abs/2302.13431)

**[3]** R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta,
        Z. Liu, J. A. Fessler, D. C. Noll. Spatiotemporal Maps for Dynamic 
        MRI Reconstruction, 2025, arXiv:2507.14429.
        (https://arxiv.org/abs/2507.14429)