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
import gc
from scipy.stats import percentileofscore
import argparse



### FUNCTION TO RUN THE SAMPLING ONE TIME ###

def run_simulation(seed, sim_config):

    # Unpack what we need from sim_config
    params = sim_config["params"]
    OmMerr = sim_config["OmMerr"]
    rp_bin_edges = sim_config["rp_bin_edges"]
    rp0 = sim_config["rp0"]
    lens = sim_config["lens"]
    src = sim_config["src"]
    Pimax = sim_config["Pimax"]
    endfilename = sim_config["endfilename"]
    grav_theory = sim_config["grav_theory"]
    egcov = sim_config["egcov"]
    inv_egcov = sim_config["inv_egcov"]
    vals = sim_config["vals"]
    OmMvals = sim_config["OmMvals"]
    EGvals = sim_config["EGvals"]
    logP_OmM = sim_config["logP_OmM"]
    zbar = sim_config["zbar"]
    N_OmMfitsamp = sim_config["N_OmMfitsamp"]
    rp_bin_c = sim_config["rp_bin_c"]


    #### GENERATE DATA REALISATION ####

    rng = np.random.default_rng(seed)

    # Draw from OmegaM0 prior.
    #OmMsamp = np.random.normal(params['OmM'], OmMerr, 1)
    OmMsamp = rng.normal(params['OmM'], OmMerr, 1)[0]

    # Using this value of OmegaM0, compute corresponding E_G 
    # (we assume this has been fully corrected for nonlinear
    # bias as we cannot compute the nonlinear bias correction
    # outside GR.
 
    # Compute E_G - alter final argument different gravity theory.
    params_new = params.copy()
    params_new['OmM'] = OmMsamp
    EG_fid = fid.E_G(params_new, rp_bin_edges, rp0, lens, src, 
                 Pimax, endfilename, nonlin=False, MG=True, MGtheory=grav_theory)
    
    print('EG_fid MG=', EG_fid)

    EG_GR = fid.E_G(params_new, rp_bin_edges, rp0, lens, src,
                 Pimax, endfilename, nonlin=False, MG=False)

    # Using this as a mean with Eg_cov, draw a data realisation.
    #EG_data = np.random.multivariate_normal(EG_fid, egcov, 1)
    EG_data = rng.multivariate_normal(EG_fid, egcov, 1)
    print('EG_data=', EG_data[0,:])


    # Save E_G realisation and error bars
    EG_draw = np.column_stack((rp_bin_c, EG_data[0,:], np.sqrt(np.diag(egcov))))
    np.savetxt('../txtfiles/EG_data_realisation_Omrc0pt5_CMBprior_LSSTY10.dat', EG_draw)
    #print('got EG draw')

    #### FIT CONSTANT TO DATA REALISATION DRAW ####

    # Sample the likelihood. It's only a one dimensional parameter space so we can just grid-sample.

    loglike_vals = np.zeros(len(vals))
    like_vals = np.zeros(len(vals))
    like_vals_norm = np.zeros(len(vals))
    for j in range(0,len(vals)):
        loglike_vals[j] = u.logL(vals[j]*np.ones(len(EG_data[0,:])), EG_data[0,:], inv_egcov)
    like_vals = np.exp(loglike_vals)
    #print('like_vals=', like_vals)
    like_vals_norm = like_vals/ scipy.integrate.simps(like_vals, vals)
    #print('like_vals_norm=', like_vals_norm)
    # Get the maximum of the posterior point of the fit to the constant, 
    max_post_ind = np.argmax(like_vals_norm)
    max_post_val = vals[max_post_ind]
    #print('max post val=', max_post_val)

    # Save the max likelihood value of E_G for this draw
    np.savetxt('../txtfiles/EG_fit_data_realisation_Omrc0pt5_CMBprior_LSSTY10.dat', [max_post_val])
    
    gc.collect()

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
        print('const bad fit')
        const_bad_fit = 1
        # Set outside95 to 0 because we aren't even going to check that:
        outside95 = 0
        OmM_fit_mean = np.nan
        percentile_rank = np.nan
        return const_bad_fit, outside95, OmMsamp, OmM_fit_mean, percentile_rank # No point in continuing in this case, we already know we reject GR
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

    # Save the OmM likelihood:
    like_OmM = np.column_stack((OmMvals, like_vals_norm_OmM))
    np.savetxt('../txtfiles/OmMlikelihood_Omrc0pt5_CMBprior_LSSTY10.dat', like_OmM)
    #print('got like OmM')

    # Define a pdf on OmM from posterior values computed above.
    bin_means, bin_edges, bin_number = binned_statistic(OmMvals, like_vals_norm_OmM, statistic ='mean',bins=200)
    hist_OmM = bin_means, bin_edges
    OmMfit_dist = rv_histogram(hist_OmM, density = True) # can set density=True because normalised values.

    gc.collect()

    #### GET REPLICATED E_G DATA UNDER GR MODEL ####

    # Sample values from this distribution
    OmMfitsamps = OmMfit_dist.rvs(size=N_OmMfitsamp)

    # Get corresponding GR theory E_G values:
    EG_rep_data = fid.EG_theory(OmMfitsamps, zbar)
    #print('EG_rep_data=', EG_rep_data)
    # Get distribution of GR value for E_G.

    # Save EG_rep:
    np.savetxt('../txtfiles/EG_replicated_Omrc0pt5_CMBprior_LSSTY10.dat', EG_rep_data)

    # Calculate percentile rank of the best-fit constant E_G in the posterior predictive samples:
    percentile_rank = percentileofscore(EG_rep_data, max_post_val, kind='rank')  # percentile in [0, 100]


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

    # Store diagnostics:
    OmM_fit_mean = np.mean(OmMfitsamps)  

    return const_bad_fit, outside_95, OmMsamp, OmM_fit_mean, percentile_rank


def run_and_store(index, sim_config):
    seed = 13+ 3 # 13 is random, just so it isn't 0 for index 0

    try:
        const_bad_fit, outside_95, OmM_true, OmM_fit_mean, percentile_rank = run_simulation(seed=seed, sim_config=sim_config)
        return {
            'run': index,
            'const_bad_fit': const_bad_fit,
            'outside_95': outside_95,
            'OmegaM_true': OmM_true,
            'OmegaM_fit_mean': OmM_fit_mean,
            'percentile_rank': percentile_rank
        }
    except Exception as e:
        return {'run': index, 'error': str(e)}

def main():
    
    parser = argparse.ArgumentParser(description="Read input variables.")
    parser.add_argument("--nruns", type=int, required=True, help="Total number of simulated data realisations to run.")
    parser.add_argument("--nworkers", type=int, required=True, help="Number of workers (i.e. cores here) to use.")
    parser.add_argument("--outfile", type=str, required=True, help="File to output the results.")
    parser.add_argument("--OmMerr", type=float, required=True, help="1 sigma width of the OmMerr Gaussian prior.")
    parser.add_argument("--gravtheory", type=str, required=True, help="The true gravity theory in the simulated Universr. fR or nDGP.")
    parser.add_argument("--gravpar", type=float, required=True, help="The value of the gravity-theory-specific parameter that we are varying. fR0 or Omega_rc as appropriate.")
    parser.add_argument("--srcsamp", type=str, required=True, help="The source galaxy sample we are assuming. LSST or LSSTY10")
    parser.add_argument("--covfile", type=str, required=True, help="The file containing the precomputed Eg covariance matrix.")

    args = parser.parse_args()

    print(args.nruns)

    ###### HERE ARE THE SET-UP THINGS YOU MIGHT NEED TO CHANGE #######
    #src = 'LSST'
    src = args.srcsamp
    print('src=', src)

    ## output file
    OUTPUT_FILE = args.outfile
    print('outfile =', OUTPUT_FILE)

    # Define Gaussian uncertainty for prior distribution on OmegaM0
    # DES year 3 gives OmegaM0 = 0.339 + 0.032 - 0.031 for LCDM model. 
    #OmMerr = 0.03
    # Planck 2018 TT EE TE + lowE gives 0.3166 ± 0.0084, try this
    #OmMerr = 0.0084
    OmMerr = args.OmMerr
    print('OmMerr=', OmMerr)

    #grav_theory = 'fR'
    grav_theory = args.gravtheory
    print('grav_theory=', grav_theory)

    # Covariance matrix:
    # Notice this is the bias-correction version.
    egcovfile = args.covfile
    print('egcovfile=', egcovfile)
    egcov_raw = np.loadtxt(egcovfile)

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

    if args.gravtheory=='nDGP':
        # A_s value is designed to match sigma8=0.82 in LCDM for other cosmological parameters. 
        #  We do this by manually finding the value of A_s that gives the right sigma8 using ccl_sigma8.
        Omega_rc = args.gravpar
        print('Omega_rc=', Omega_rc)
        h0rc = 1./np.sqrt(4*Omega_rc)
        # Have to give some dummy fR values so we don't trip an error
        fR0= 10**(-4)
        params = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965,'b':b1, 'OmM': OmM, 'b_2':b2, 'b_s': bs, 'A_s':2.115 * 10**(-9),'H0rc':h0rc, 'fR0': fR0, 'fR_n': 1} 
    elif args.gravtheory=='fR':
        fR0 = args.gravpar
        print('fR0=', fR0)
        # Have to give some dummy H0rc value so we don't trip an error
        Omega_rc = 0.25
        h0rc = 1./np.sqrt(4*Omega_rc)
        params = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965,'b':b1, 'OmM': OmM, 'b_2':b2, 'b_s': bs, 'fR0' : fR0, 'A_s':2.115 * 10**(-9), 'fR_n': 1, 'H0rc':h0rc} 


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

    sim_config = {
    "params": params,
    "OmMerr": OmMerr,
    "rp_bin_edges": rp_bin_edges,
    "rp0": rp0,
    "lens": lens,
    "src": src,
    "Pimax": Pimax,
    "endfilename": endfilename,
    "grav_theory": grav_theory,
    "egcov": egcov,
    "inv_egcov": inv_egcov,
    "vals": vals,
    "OmMvals": OmMvals,
    "EGvals": EGvals,
    "logP_OmM": logP_OmM,
    "zbar": zbar,
    "N_OmMfitsamp": N_OmMfitsamp,
    "rp_bin_c": rp_bin_c,
}


    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nworkers) as executor:
        #results = executor.map(run_and_store, range(N_RUNS))
        futures = [executor.submit(run_and_store, i, sim_config) for i in range(args.nruns)]

    # Write to newline-delimited JSON for easy incremental writing and parsing
    with open(OUTPUT_FILE, "w") as f:
        #for result in results:
        #    f.write(json.dumps(result) + "\n")
        for future in concurrent.futures.as_completed(futures):
                result = future.result()
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
