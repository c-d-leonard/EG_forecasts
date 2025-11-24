This repository contains the code used in the analysis of the paper 'Towards testing gravity with LSST using $E_G$' (C. D. Leonard, S. Alam, S. Singh, R. Mandelbaum, M. M. Rau, C. Zanoletti for LSST DESC). 

## Scripts and notebooks used to generate results for the paper:

### General:

- specs.py contains functions which specify things about the lens and source samples like theire redshift distributions etc

- utils.py contains utility functions for things like converting between bin centres and bin edges

- fiducial.py computes theoretical fiducial values of the quantities needed for the data vector


### Covariances:

- get_cov_sims_addSNmatrix.ipynb constructs the covariance matrix from simulations, constructs and adds the appropriate shape noise covariance matrix, and outputs the correlation matrix in the required units.

- add_shape_noise.py contains the functions to construct the shape noise covariance matrix

- cov_scipt.py contains functions for loading and computing the covariance matrix from simulation data.

- Eg_cov.py contains functions for calculating the Eg covariance given the joint-probes covariance.

- joint_cov.py contains functions to calculate the analytic covariance for the constituent probes of Eg. Get the Upsilon covariances from DeltaSigma. Contains some code for getting approximate covariances (e.g. shape noise only, Fisher for beta) that were not used in the paper.

- cov_hankel.py contains functions to get the Delta Sigma analytic covariances

- hankel_transform.py contains functions to do the required hankel_transform for the analytic covariance. This script was written by Sukhdeep Singh originally.

- plot_covariances_paper.ipynb plots the covariances for the paper (joint probes and Eg). It also calls the function to calcuate the covariance for Eg from the join probes covariance.

- plot_analytic_covariances_paper.ipynb plots the covariances using the analytic method to display in the appendix.


### Scale cuts and nonlinear bias:

- get_linear_scale_cuts.ipynb finds the linear-only scale cuts in the case of GR with linear galaxy bias.

- get_linear_scale_cuts_MG.ipynb finds the linear-only scale cuts in the case of linear galaxy bias and in f(R) or nDGP gravity.

- get_linear_scale_cuts_nLbias.ipynb finds the linear-only scale cuts in the case of GR with nonlinear bias.

- get_linear_scale_cuts_nLbias_correction.ipynb finds the scale cuts in the case of the correction factor for nonlinear bias, including calculating the covariance matrix for the bias factor and the combined covariance matrix.

### Posterior predictive tests:

- post_pred.py runs the posterior predictive tests and outputs the results.

- analyse_postpred_results_printruns.py analyses the output of post_pred.py to calculate the number of runs in each case where we accept and accept the GR hypothesis.

- 





