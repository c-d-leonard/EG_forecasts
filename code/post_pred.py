import numpy as np
import matplotlib.pyplot as plt
import fiducial as fid
import utils as u
import scipy.stats 
from scipy.stats import chi2
from scipy.stats import rv_histogram
from scipy.stats import binned_statistic
import specs as sp 
import concurrent.futures
import json
from pathlib import Path

###### HERE ARE THE SET-UP THINGS YOU MIGHT NEED TO CHANGE #######
src = 'LSSTY10'

# Define Gaussian uncertainty for prior distribution on OmegaM0
# DES year 3 gives OmegaM0 = 0.339 + 0.032 - 0.031 for LCDM model. 
#OmMerr = 0.03
# Planck 2018 TT EE TE + lowE gives 0.3166 ± 0.0084, try this
OmMerr = 0.0084

grav_theory = 'fR'

# nDGP
Omega_rc = 0.5
# f(R)
fr0 = 10**(-6)

# Covariance matrix:
# Notice this is the bias-correction version.
egcov_raw = np.loadtxt('../txtfiles/cov_EG_nLbiascorrected_Y10_Jul2025.dat')

# Scale cuts
rp_c_scalecuts, scalecuts = np.loadtxt('../txtfiles/scalecuts_nLbias_CORRECTIONFACTOR_KitanidisWhite2022_LSSTY10_Jul2025.dat', unpack=True)

# SET UP A BUNCH OF STUFF.

# Parameters
lens = 'DESI'
Pimax=900.
rp0 = 1.5
endfilename = 'post_pred_Mar25'

# Use the same cosmological parameters as Shadab's simulations:
h=0.69
OmB = 0.022/h**2
OmM = 0.292

# Using now nonliner bias parameters as fit in Kitanis & White 2022. 
# They fit LTP parameters so we convert these to their Eulerian equivalents.
b1_LPT = 1.333
b2_LPT = 0.514
bs_LPT = 0 # They fix this to 0.

# Convert to Eulerian using the conversions in Chen, Vlah & White 2020 (these use the same convention as Kitanis & White 2022)
b1 = 1.0 + b1_LPT
b2 = b2_LPT + 8./21.*(b1_LPT)
bs = bs_LPT - 2./7*(b1_LPT)

# A_s value is designed to match sigma8=0.82 in LCDM for other cosmological parameters. 
# We do this by manually finding the value of A_s that gives the right sigma8 using ccl_sigma8.
h0rc = 1./np.sqrt(4*Omega_rc)
params = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965,'b':b1, 'OmM': OmM, 'b_2':b2, 'b_s': bs, 'fR0' : fr0, 'A_s':2.115 * 10**(-9), 'fR_n': 1, 'H0rc':h0rc} 

# Compute the mean redshift over the lenses, we need this later:
zvec, dNdz = sp.get_dNdzL(params, 'DESI')
zbar = scipy.integrate.simps(zvec*dNdz, zvec)

# Scale cuts. 0 means cut the bin, 1 means keep it.
rp_bin_c_raw = np.loadtxt('../data_for_Danielle/test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon')[:,0]
rp_bin_edges_raw = u.rp_bin_edges_log(rp_bin_c_raw)

#Cut below rp0 making sure rp0 is in the lowest bin. Go one bin lower because this 
ind = next(j[0] for j in enumerate(rp_bin_edges_raw) if j[1]>rp0)
rp_bin_c = rp_bin_c_raw[ind:][4:]
rp_bin_edges = rp_bin_edges_raw[ind:][4:]

covsamps = 100 # Number of simulations used to estimate covariance
egcov = egcov_raw[4:,4:]

hartlap = (covsamps-len(rp_bin_c)-2) / (covsamps-1)
inv_egcov = hartlap*np.linalg.inv(egcov)

# For the fit to constant model of the E_G data realisation:
vals = np.linspace(0.01,0.99,10000) # values at which we grid-sampl E_G constant model

# For getting the posterior on OmegaM0 in the GR model:
OmMvals = np.linspace(0.1,0.5,10000) # values at which grid sample OmegaM0
EGvals = fid.EG_theory(OmMvals, zbar) # Get corrsponding EG values
logP_OmM = -0.5*(OmMvals - params['OmM'])**2/(OmMerr**2) # define DES Y3 LCDM prior

N_OmMfitsamp = 10000

### FUNCTION TO RUN THE SAMPLING ONE TIME ###

def run_simulation():

    #### GENERATE DATA REALISATION ####

    # Draw from OmegaM0 prior.
    OmMsamp = np.random.normal(params['OmM'], OmMerr, 1)

    # Using this value of OmegaM0, compute corresponding E_G 
    # (we assume this has been fully corrected for nonlinear
    # bias as we cannot compute the nonlinear bias correction
    # outside GR.

    # Compute E_G - alter final argument different gravity theory.
    params.update({'OmM':OmMsamp[0]})
    EG_fid = fid.E_G(params, rp_bin_edges, rp0, lens, src, 
                 Pimax, endfilename, nonlin=False, MG=True, MGtheory=grav_theory)

    # Using this as a mean with Eg_cov, draw a data realisation.
    EG_data = np.random.multivariate_normal(EG_fid, egcov, 1)

    #### FIT CONSTANT TO DATA REALISATION DRAW ####

    # Sample the likelihood. It's only a one dimensional parameter space so we can just grid-sample.

    loglike_vals = np.zeros(len(vals))
    like_vals = np.zeros(len(vals))
    like_vals_norm = np.zeros(len(vals))
    for j in range(0,len(vals)):
        loglike_vals[j] = u.logL(vals[j]*np.ones(len(EG_data[0,:])), EG_data[0,:], inv_egcov)
    like_vals = np.exp(loglike_vals)
    like_vals_norm = like_vals/ scipy.integrate.simps(like_vals, vals)

    # Get the maximum of the posterior point of the fit to the constant, 
    max_post_ind = np.argmax(like_vals_norm)
    max_post_val = vals[max_post_ind]

    # Get the chi^2
    chisq = np.dot((max_post_val - EG_data[0,:]), 
                          np.dot(inv_egcov, (max_post_val - EG_data[0,:])))
    
    # We want to look at a chi squared distribution with the correct number 
    # of degrees of freedom

    # Define the number of degrees of freedom:
    len_cut_dvec = len(EG_data[0,:])
    model_par = 1 # 1 parameter for a constant model
    DOF = len_cut_dvec - model_par

    # What we want is the CDF value at the chi square.
    # What this tells us is the probability that, given our constant model 
    # is correct, we would have got the chi square we got or a lower chi squared (fixing our measured data vector and cov).
    # Eg if CDF = 0.95, this means that if our model is correct, 95% of the time the chi squared would have been lower.
    # If we care about the model and the data agreeing within 1 sigma, we look for the case where CDF=0.68 or less
    # i.e. the case where if our model is correct, 68% of the time we would have drawn a lower chi-squared. 
    cdf_samps = chi2.cdf(chisq, DOF)

    # Set to 1 where the cdf>=0.95 (corresponds to p value 0.05)
    if cdf_samps>=0.95:
        const_bad_fit = 1
        # Set outside95 to 0 because we aren't even going to check that:
        outside95 = 0
        return const_bad_fit, outside95 # No point in continuing in this case, we already know we reject GR
    # If we get to this point it means constant was a good fit:
    const_bad_fit=0

    #### GET POSTERIOR IN OMEGA_M_0_FIT IN GR MODEL ####

    loglike_vals_OmM = np.zeros(len(OmMvals))
    like_vals_OmM = np.zeros(len(OmMvals))
    like_vals_norm_OmM = np.zeros(len(OmMvals))
    for j in range(0,len(OmMvals)):
        loglike_vals_OmM[j] = u.logL(EGvals[j], EG_data[0,:], inv_egcov)
    like_vals_OmM = np.exp(loglike_vals_OmM + logP_OmM) # This version adds a prior.
    like_vals_norm_OmM = like_vals_OmM/ scipy.integrate.simps(like_vals_OmM, OmMvals)

    # Define a pdf on OmM from posterior values computed above.
    bin_means, bin_edges, bin_number = binned_statistic(OmMvals, like_vals_norm_OmM, statistic ='mean',bins=200)
    hist_OmM = bin_means, bin_edges
    OmMfit_dist = rv_histogram(hist_OmM, density = True) # can set density=True because normalised values.

    #### GET REPLICATED E_G DATA UNDER GR MODEL ####

    # Sample values from this distribution
    OmMfitsamps = OmMfit_dist.rvs(size=N_OmMfitsamp)

    # Get corresponding GR theory E_G values:
    EG_rep_data = fid.EG_theory(OmMfitsamps, zbar)

    # Get distribution of GR value for E_G.

    EG_rep_hist = np.histogram(EG_rep_data, bins = 50, density=True)
    EG_rep_dist = rv_histogram(EG_rep_hist, density=True)

    # Get the 5% and 95% confidence points of the distribution 

    ninetyfive_intervals_EGrep = EG_rep_dist.interval(0.95)

    # Does the constant best fit value to the original data fall
    # outside the 95%?

    if max_post_val <= ninetyfive_intervals_EGrep[0] or max_post_val>= ninetyfive_intervals_EGrep[1]:
        outside_95 = 1 
    else:
        outside_95 = 0

    return const_bad_fit, outside_95


def run_and_store(index):
    result_1, result_2 = run_simulation()  
    return {'run': index, 'const_bad_fit': result_1, 'outside_95': result_2}

def main():
    N_RUNS = 4
    N_WORKERS = 4
    OUTPUT_FILE = "../txtfiles/post_pred_test.jsonl"

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = executor.map(run_and_store, range(N_RUNS))

    # Write to newline-delimited JSON for easy incremental writing and parsing
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
