## This script acts as a main() for getting forecast parameter constraints.

endfilename='analytic_cov_Oct2024_rp0=1.5'

import numpy as np
import pyccl as ccl
import utils as u
import Eg_cov as egcov
import joint_cov as jp
import fiducial as fid
import fisher as F
#import postprocess as pp
import matplotlib.pyplot as plt
#import tests as t
import time
import specs as sp
import cov_hankel as ch

a = time.time()

# Let's use the same cosmological parameters as Shadab's simulations:
h=0.69
OmB = 0.022/h**2

# But use the bias parameters from Kitanidis & White
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

params_fid_var = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965, 'A_s':2.115 * 10**(-9),'b':b1, 'OmM': 0.292} 
params_fid_fix = {}
#params_Eg_insens = {'b':2.2, 'sigma8:': 0.83} # Parameters to which E_G is by-design totally insensitive.

params_fid = params_fid_var.copy()   
params_fid.update(params_fid_fix)

#scale = 0.15
#hdict = {'OmM': params_fid['OmM']*scale, 'OmB':params_fid['OmB']*scale, 'h':params_fid['h']*scale, 'sigma8':params_fid['sigma8']*scale, 'n_s':params_fid['n_s']*scale, 'b': params_fid['b']*scale, 'mu_0': 0.05*scale, 'sigma_0':0.05*scale}

lens = 'DESI'
src = 'LSSTY10'

Pimax=900. # This is Pimax for Upgg only. 900 is found to achieve a converged value of the integral.

# This is an approximate rp0 value; we will set the rp0 value to be the lower edge of rp_edges in a second
#rp0_approx = 1.5

# Set up the projected radial vectors
# I am currently trying to match the central rp bin values used by Shadab
# So this is a bit of a convoluted method
# Load the rp bin centers from Shadab
rp0 = 1.5
rp_bin_c_raw = np.loadtxt('../data_for_Danielle/test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon')[:,0]
rp_bin_edges_raw = u.rp_bin_edges_log(rp_bin_c_raw)
#print(rp_bin_c_raw)
#Cut below rp0 making sure rp0 is in the lowest bin.
ind = next(j[0] for j in enumerate(rp_bin_edges_raw) if j[1]>rp0) - 1
rp_bin_edges = rp_bin_edges_raw[ind:]
rp_bin_c_rp0 = rp_bin_c_raw[ind:]
print('rp edges=', rp_bin_edges) 
print('rp centres=', rp_bin_c_rp0)
# These are even spaced in log-space. Use this fact to get the edges.
#half_dlogrp = (np.log(rp_bin_c_raw)[1] - np.log(rp_bin_c_raw)[0])/2
#rp_bin_edges_raw = np.exp(np.asarray([np.log(rp_bin_c_raw[0])-half_dlogrp]+[np.log(rp_bin_c_raw[i])+half_dlogrp for i in range(0,len(rp_bin_c_raw))]))
# Now we need to cut both so that an appropriate rp0 value is the first
# entry in rp_bin_edges
#rp0_init = 1.5 # We want rp0 to be at least thing.
#ind_rp0 = next(j[0] for j in enumerate(rp_bin_edges_raw) if j[1]>=rp0_init)
#rp_bin_edges = rp_bin_edges_raw[ind_rp0:]
#rp_bin_c = rp_bin_c_raw[ind_rp0:]
#rp0 = rp_bin_edges[0]

#joint_cov = jp.get_joint_covariance(params_fid, lens, src, rp_bin_edges, rp_bin_c_rp0, rp0, endfilename)

#np.savetxt('../txtfiles/joint_covariance_rp0=1.5_Oct2024.dat', joint_cov)
#exit()

#joint_cov = np.loadtxt('../txtfiles/sims_cov_myunits_wSN_Oct24.dat')
joint_cov = np.loadtxt('../txtfiles/sims_cov_myunits_wSN_Jan25_LSSTY10.dat')

#print(rp_bin_rp0)
#print(rp_bin_edges)

#print("rp bins=", rp_bin_c)
#print("num rp bins=", len(rp_bin_c))

# This is the normal independent way to define rp from scratch.
#rp_bin_edges = np.logspace(np.log10(rp0), np.log10(50.), 11)
#rp_bin_c = u.rp_bins_mid(rp_bin_edges)

print("params_fid=", params_fid)

egcov = egcov.get_egcov(joint_cov, params_fid, rp_bin_edges, rp_bin_c_rp0, rp0, lens, src, Pimax, 100000, 'hybrid_cov_March12_2025')

np.savetxt('../txtfiles/egcov_sims_wSN_Mar2025_LSSTY10.txt', egcov)

exit()
#egcov = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/eg_cov_DESIonly_FTF.txt')

# Error on a constant E_G:

#inv_egcov = np.linalg.inv(egcov)
#deriv = np.ones(len(rp_bin_c))
#Fisher = np.dot(deriv, np.dot(inv_egcov, deriv))
#print "Fisher=", Fisher
#print "var =", 1./Fisher
#print "error=", np.sqrt(1./Fisher)

#np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Eg_err_DESI_4MOST_ELGs.txt', [np.sqrt(1./Fisher)]) 

############ GET FISHER MATRICES ###########

########### Get the Fisher matrices in each case ############
print("Getting Fisher matrix.")
(Fisher_Eg, Fisher_jp, keys_list) = F.get_Fisher_matrices(params_fid, params_fid_var, hdict, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, endfilename, 100000)

# Save Fisher and keys list to file
np.savetxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_Eg_'+endfilename+'.txt', Fisher_Eg)
np.savetxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_jp_'+endfilename+'.txt', Fisher_jp)

f_keys = open("/home/danielle/Research/EG_comparison/txtfiles/keys_"+endfilename+".txt", "w")
for key in keys_list:
	f_keys.write(key+"\n")
f_keys.close() 

Fisher_Eg = np.loadtxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_Eg_'+endfilename+'.txt')
Fisher_jp = np.loadtxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_jp_'+endfilename+'.txt')

f_keys = "/home/danielle/Research/EG_comparison/txtfiles/keys_"+endfilename+".txt"

with open(f_keys) as f:
    keys_list = f.readlines()
    
keys_list = [x.strip() for x in keys_list] 

############# ADD PRIORS ############### 

# We will want to add external priors on certain parameters 
# which are poorly constrained by the late-time observables we consider:
# Omega_b, h, n_s, at least.
# We are using priors obtained by Naren from chains reproducing the resutls of 
# Planck 2015 DE / MG, chains by the UT Dallas group.

priors_file = '/home/danielle/Research/EG_comparison/txtfiles/priors/priors_cov_MG_smallmat.txt'
keys_file_priors = '/home/danielle/Research/EG_comparison/txtfiles/priors/Planck4.paramnames_smallmat.txt'

Fisher_Eg_priors = F.add_priors(Fisher_Eg, keys_list, priors_file, keys_file_priors)
Fisher_jp_priors = F.add_priors(Fisher_jp, keys_list, priors_file, keys_file_priors)

Fisher_Eg_priors = np.savetxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_Eg_priors_'+endfilename+'.txt', Fisher_Eg_priors)
Fisher_jp_priors = np.savetxt('/home/danielle/Research/EG_comparison/txtfiles/Fisher_jp_priors_'+endfilename+'.txt', Fisher_jp_priors)

exit()

######## Output results ########

# Set up sets of parameters we want to fix and vary in plotting:

# Vary mu_0 and sigma_0
params_var_0 = {'mu_0': 0., 'sigma_0':0., 'OmB':0.05, 'h':0.68, 'n_s':0.96}
params_fix_0 = {'sigma8':0.83,'b':2.2,'OmM': 0.3}

# Vary mu_0, sigma_0, and bias
params_var_1 = {'mu_0': 0., 'sigma_0':0.,'b':2.2, 'OmB':0.05, 'h':0.68, 'n_s':0.96}
params_fix_1 = {'sigma8':0.83, 'OmM': 0.3}

# Vary mu_0, sigma_0, and sigma8
params_var_2 = {'mu_0': 0., 'sigma_0':0., 'sigma8':0.83, 'OmB':0.05, 'h':0.68, 'n_s':0.96}
params_fix_2 = {'b':2.2, 'OmM': 0.3}

# Vary mu_0, sigma_0, bias and sigma8
params_var_3 = {'mu_0': 0., 'sigma_0':0., 'sigma8':0.83, 'b':2.2, 'OmB':0.05, 'h':0.68, 'n_s':0.96}
params_fix_3 = { 'OmM': 0.3}

# Vary mu_0, sigma_0, bias, sigma8, and OmegaM
params_var_4 = {'mu_0': 0., 'sigma_0':0., 'sigma8':0.83, 'b':2.2, 'OmM': 0.3, 'OmB':0.05, 'h':0.68, 'n_s':0.96}
params_fix_4 = {}

# Vary mu_0, sigma_0, bias, sigma8, OmegaM, OmegaB
#params_var_5 = {'mu_0': 0., 'sigma_0':0., 'sigma8':0.83, 'b':2.2, 'OmM': 0.3, 'OmB':0.05}
#params_fix_5 = {'h':0.68, 'n_s':0.96}

# Get all the stuff for individual ellipses and store them in these lists
Evals = [0]*3; Evecs = [0]*3; rotate=[0]*3; label=[0]*3; color=[0]*3; linestyle=[0]*3;

# Joint probes, fix everything but mu0 and sigma0
Evals[0], Evecs[0], rotate[0] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_0, params_fix_0)
label[0] = 'Multiprobe: Fix $\sigma_8$, b'; color[0] = 'k'; linestyle[0]='--'

# E_G, fix everything but mu0 and sigma0
Evals[1], Evecs[1], rotate[1] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_Eg, keys_list, params_var_0, params_fix_0)
label[1] = '$E_G$ (insensitive to $\sigma_8$, b)'; color[1] = 'b'; linestyle[1]=':'

# Joint probes, vary bias
#Evals[2], Evecs[2], rotate[2] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_1, params_fix_1)
#label[2] = 'JP: Vary b'; color[2] = 'm'; linestyle[2]='--'

# E_G, vary bias: exactly the same as E_G fixing everything else because E_G is totally insensitive to linear bias

# Joint probes, vary sigma8
#Evals[3], Evecs[3], rotate[3] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_2, params_fix_2)
#label[3] = 'JP: Vary $\sigma_8$'; color[3] = 'g'; linestyle[3]='--'

# E_G, vary sigma8 : exactly the same as fixing everything else because E_G is totally insensitive to sigma8

# Joint probes, vary sigma8 & bias
Evals[2], Evecs[2], rotate[2] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_3, params_fix_3)
label[2] = 'Multiprobe: Vary $\sigma_8$, b'; color[2] = '#FF9966'; linestyle[2]='--'

# E_G, vary sigma8 and bias: exactly the same as fixing everything else because E_G is totally insensitive to sigma 8 and bias

# Joint probes, vary sigma8, bias, and OmegaM
#Evals[5], Evecs[5], rotate[5] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_4, params_fix_4)
#label[5] = 'JP: Vary $\sigma_8$, b, $\Omega_M$'; color[5] = 'b'; linestyle[5]='--'

# EG, vary sigma8, bias, and OmegaM
#Evals[6], Evecs[6], rotate[6] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_Eg, keys_list, params_var_4, params_fix_4)
#label[6] = '$E_G$: Vary $\sigma_8$, b, $\Omega_M$'; color[6] = 'b'; linestyle[6]=':'

# Joint probes, vary sigma8, bias, OmegaM, and OmegaB
#Evals[7], Evecs[7], rotate[7] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_jp, keys_list, params_var_5, params_fix_5)
#label[7] = 'JP: Vary $\sigma_8$, b, $\Omega_M$, $\Omega_B$'; color[7] = 'r'; linestyle[7]='--'

## EG, vary sigma8, bias, OmegaM, OmegaB
#Evals[8], Evecs[8], rotate[8] = pp.single_ell_data(['mu_0', 'sigma_0'], Fisher_Eg, keys_list, params_var_5, params_fix_5)
#label[8] = '$E_G$: Vary $\sigma_8$, b, $\Omega_M$, $\Omega_B$'; color[8] = 'r'; linestyle[8]=':'


#print Evals[6], Evecs[6], rotate[6]

pp.ellipse_plots([0, 0], Evals, Evecs, rotate, label,color, linestyle, endfilename+'_plotLSSTasia')

print('\nTime for completion:', '%.1f' % (time.time() - a), 'seconds')


