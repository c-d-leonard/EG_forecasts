# This contains functions to take as input the covariance matrix of [Upsilon_{gm}(r_p), Upsilon_{gg}(r_p), beta], and outputs the covariance matrix of E_G via sampling from that covariance matrix.

import numpy as np
import joint_cov as jp
import utils as u
import subprocess as sp
import fiducial as fid

############# FUNCTIONS ###############


def get_Eg_sample(joint_cov, params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename):
    """ Samples from the multiD Gaussian PDF Gaussian PDF for Upsilon_{gm}(r_p), Upsilon_{gg}(r_p), and beta, and combines for a sample of Eg values.
    joint_cov : joint-probes covariance matrix to be used, in Ups_gm, Ups_gg, beta order.
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
    #rp = np.logspace(np.log10(rp0), np.log10(50.), 500)
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
	
    #joint_cov = jp.get_joint_covariance(params, lens, src, rp_bin_edges, rp_bin_c, rp0, endfilename)

    # Get Nsamps samples from the data vector
    datavec_samps = np.random.multivariate_normal(means_list, joint_cov, Nsamps)
        
    Eg_samp = [[datavec_samps[Ns,elem] / (datavec_samps[Ns, len(Ygm_mean)+elem] * datavec_samps[Ns, len(Ygm_mean)+len(Ygg_mean)]) for Ns in range(Nsamps)] for elem in range(len(Ygm_mean))]
	
    """Eg_samp = [[0]*Nsamps]*len(Ygm_mean)
    for elem in range(len(Ygm_mean)):
        for Ns in range(Nsamps):
	    print('Ns=', Ns)
	    print('elem=', elem)
	    print('len(Ygm_mean)-1+elem=', len(Ygm_mean)-1+elem)
	    print('len(Ygm_mean)=',len(Ygm_mean))
	    exit()
	    Eg_samp[elem][Ns] = datavec_samps[Ns,elem] / (datavec_samps[Ns, len(Ygm_mean)-1+elem] * datavec_samps[Ns, -1])"""
	
    Eg_samp_arr = np.zeros((len(Ygm_mean), Nsamps))
    for ri in range(len(Ygm_mean)):
        Eg_samp_arr[ri, :] = Eg_samp[ri]

    return Eg_samp_arr
	
def get_egcov(joint_cov, params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename):
	""" Get the covariance matrix in rp bins of Eg.
	joint_cov : joint-probes covariance matrix to be used, in Ups_gm, Ups_gg, beta order.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	Nsamps : number of samples at which to sample Eg values in estimating the covarinace.
	endfilename : tag for the files produced to keep track of the run."""
	
	Eg_samp = get_Eg_sample(joint_cov, params, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename)
	Eg_cov = np.cov(Eg_samp)
	
	return Eg_cov


def cov_bias_corr(params, bias_par_means, bias_par_cov, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename):
    """ Get the covariance term for the nonlinear bias correction factor.
    Do this by drawing samples of the nonlinear bias parameters and using 
    a sample covariance method.
    params: parameters to compute the sample at. The bias paremters passed will be overwritten for each sample.
    bias_par_means: mean of the LAGRANGIAN bias parameters, used to draw samples. (Order: b1, b2)
    bias_par_cov: covariance of the LAGRANGIAN bias parameters, used to draw samples.
    rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	Nsamps : number of samples at which to sample bias values in estimating the covarinace.
	endfilename : tag for the files produced to keep track of the run.
    """

    # Sample from the bias pars:
    bias_pars_samp = np.random.multivariate_normal(bias_par_means, bias_par_cov, Nsamps) # [b1,b2]

    print('shape=',bias_pars_samp.shape)

    # Loop over the number of samples to get the correction factor at each of these.
    Cb_samps = np.zeros((len(rp_bin_c), Nsamps))
    for i in range(0,Nsamps):
        print('sample number=', i)
        b1_LPT = bias_pars_samp[i,0]
        b2_LPT = bias_pars_samp[i,1]
        bs_LPT = 0 # Fixed to 0 in Kitinidis & White.

        # Convert these to Eulerian bias parameters because that's what we use to calculate

        b1 = 1.0 + b1_LPT
        b2 = b2_LPT + 8./21.*(b1_LPT)
        bs = bs_LPT - 2./7*(b1_LPT)

        params['b'] = b1
        params['b_2'] = b2
        params['b_s'] = bs

        # Now get the correction factor at this sample:
        Cb_samps[:,i] = fid.bias_correction(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename)

    # Now compute the sample covariance over these samples:
    Cb_cov = np.cov(Cb_samps)

    return Cb_cov

def corrected_EG_cov(params, covEG, covCb, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, endfilename):
    """ combine the covariance matrix for the raw E_G with that for the bias correction factor
    to get a new version that accounts for both.
    params = parameters. We are assuming the mean bias parameters are the bias parameters in this dict.
    covEG is the raw EG covariance.
    covCb is the covariance of the bias factor.
    rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	Nsamps : number of samples at which to sample bias values in estimating the covarinace.
	endfilename : tag for the files produced to keep track of the run.
    """

    # Check at least that the covariances are all the same size
    if (covEG.shape != covCb.shape):
         print('In corrected_EG_cov: covEG and covCb need to be the same size!')
         exit()

    # First compute the bias correction at the mean parameter values:
    Cb_mean = fid.bias_correction(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename)

    EG_mean = fid.E_G(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin=True, nl_bias=True)
    
    # Construct the combined covariance.
    # We are assuming no correlation between the correction and the measured E_G value.
    Cov_EG_corr = np.zeros((len(rp_bin_c), len(rp_bin_c)))
    #first_term = np.zeros((len(rp_bin_c), len(rp_bin_c)))
    #second_term = np.zeros((len(rp_bin_c), len(rp_bin_c)))
    for ri in range(0,len(rp_bin_c)):
         for rj in range(0,len(rp_bin_c)):
              Cov_EG_corr[ri,rj] = Cb_mean[ri]*Cb_mean[rj]*covEG[ri,rj] + EG_mean[ri]*EG_mean[rj]*covCb[ri,rj]
              #first_term[ri,rj] = Cb_mean[ri]*Cb_mean[rj]*covEG[ri,rj]
              #second_term[ri,rj] = EG_mean[ri]*EG_mean[rj]*covCb[ri,rj]

    return Cov_EG_corr


	

















