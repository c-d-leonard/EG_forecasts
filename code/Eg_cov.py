# This contains functions to take as input the covariance matrix of [Upsilon_{gm}(r_p), Upsilon_{gg}(r_p), beta], and outputs the covariance matrix of E_G via sampling from that covariance matrix.

import numpy as np
import joint_cov as jp
import utils as u
import subprocess as sp
import fiducial as fid

############# FUNCTIONS ###############


def get_Eg_sample(params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename):
	""" Samples from the multiD Gaussian PDF Gaussian PDF for Upsilon_{gm}(r_p), Upsilon_{gg}(r_p), and beta, and combines for a sample of Eg values.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	Nsamps : number of samples at which to sample Eg values in estimating the covarinace.
	endfilename : tag for the files produced to keep track of the run."""
	
	# Get beta
	beta_mean = fid.beta(params, lens)
	rp = np.logspace(np.log10(rp0), np.log10(50.), 500)
	# Get Upsilon_gg
	Ygg_mean = fid.Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename)
	# Get Upsilon_gm
	Ygm_mean = fid.Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename)
	
	# We assume the data vector is listed as Upsilon_{gm}(r_p^1)..(r_p^i) ... Upsilon_{gg}(r_p^1)..(r_p^i)..beta
	Ygm_mean = np.asarray(Ygm_mean); Ygg_mean = np.asarray(Ygg_mean); beta_mean = np.asarray([beta_mean])
	
	# Check the length of Ygm_mean and Ygg_mean are the same
	n_rp = len(Ygm_mean)
	if ( (len(Ygm_mean)!=len(Ygg_mean))):
		raise(ValueError, "Upsilon_gm and Upsilon_{gg} must have the same number of radial bins.")
	
	# Check that beta is a single value:
	if (len(beta_mean)!=1):
		raise(ValueError, "beta should be a single number.")
	
	means_list = np.append(Ygm_mean, np.append(Ygg_mean, beta_mean))
	
	joint_cov = jp.get_joint_covariance(params, lens, src, rp_bin_edges, rp_bin_c, rp0, endfilename)

	# Get Nsamps samples from the data vector
	datavec_samps = np.random.multivariate_normal(means_list, joint_cov, Nsamps)

	Eg_samp = [[datavec_samps[Ns,elem] / (datavec_samps[Ns, len(Ygm_mean)-1+elem] * datavec_samps[Ns, -1]) for Ns in range(Nsamps)] for elem in range(len(Ygm_mean))]
	
	Eg_samp_arr = np.zeros((len(Ygm_mean), Nsamps))
	for ri in range(len(Ygm_mean)):
		Eg_samp_arr[ri, :] = Eg_samp[ri]

	return Eg_samp_arr
	
def get_egcov(params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename):
	""" Get the covariance matrix in rp bins of Eg.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	Nsamps : number of samples at which to sample Eg values in estimating the covarinace.
	endfilename : tag for the files produced to keep track of the run."""
	
	Eg_samp = get_Eg_sample(params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename)
	Eg_cov = np.cov(Eg_samp)
	
	return Eg_cov
	


	

















