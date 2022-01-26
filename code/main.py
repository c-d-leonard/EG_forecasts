## This script acts as a main() for getting forecast parameter constraints.

endfilename='cov_comb_with_Shadab'

import numpy as np
import pyccl as ccl
import utils as u
import Eg_cov as egcov
import joint_cov as jp
import fiducial as fid
import fisher as F
import postprocess as pp
import matplotlib.pyplot as plt
#import tests as t
import time
import specs as sp
import cov_hankel as ch

a = time.time()

params_fid_var = {'mu_0': 0., 'sigma_0':0.} # Intrinsic alignment params? Photo-z params? tau?
params_fid_fix = {'sigma8':0.83,'b':2.2,'OmB':0.05, 'h':0.68, 'n_s':0.96, 'OmM': 0.3}
params_Eg_insens = {'b':2.2, 'sigma8:': 0.83} # Parameters to which E_G is by-design totally insensitive.

params_fid = params_fid_var.copy()   
params_fid.update(params_fid_fix)

scale = 0.15
hdict = {'OmM': params_fid['OmM']*0.1, 'OmB':params_fid['OmB']*scale, 'h':params_fid['h']*scale, 'sigma8':params_fid['sigma8']*scale, 'n_s':params_fid['n_s']*scale, 'b': params_fid['b']*scale, 'mu_0': 0.05*scale, 'sigma_0':0.05*scale}

lens = 'DESI'
src = 'LSST'

Pimax=100.

# Set up the projected radial vectors
# I am currently trying to match the central rp bin values used by Shadab
# So this is a bit of a convoluted method

# Load the rp bin centers from Shadab
rp_bin_c_raw = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/data_for_Danielle/test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon')[:,0]
# These are even spaced in log-space. Use this fact to get the edges.
half_dlogrp = (np.log(rp_bin_c_raw)[1] - np.log(rp_bin_c_raw)[0])/2
rp_bin_edges_raw = np.exp(np.asarray([np.log(rp_bin_c_raw[0])-half_dlogrp]+[np.log(rp_bin_c_raw[i])+half_dlogrp for i in range(0,len(rp_bin_c_raw))]))
# Now we need to cut both so that an appropriate rp0 value is the first
# entry in rp_bin_edges
rp0_init = 1.5 # We want rp0 to be at least thing.
ind_rp0 = next(j[0] for j in enumerate(rp_bin_edges_raw) if j[1]>=rp0_init)
rp_bin_edges = rp_bin_edges_raw[ind_rp0:]
rp_bin_c = rp_bin_c_raw[ind_rp0:]
rp0 = rp_bin_edges[0]

print "rp bins=", rp_bin_c
print "num rp bins=", len(rp_bin_c)

# This is the normal independent way to define rp from scratch.
#rp_bin_edges = np.logspace(np.log10(rp0), np.log10(50.), 11)
#rp_bin_c = u.rp_bins_mid(rp_bin_edges)

print "params_fid=", params_fid 

#egcov = egcov.get_egcov(params_fid, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, 100000, 'DESI_4MOST_ELGs')

#np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/eg_cov_DESI.txt', egcov)
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
print "Getting Fisher matrix."
(Fisher_Eg, Fisher_jp, keys_list) = F.get_Fisher_matrices(params_fid, params_fid_var, hdict, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, endfilename, 100000)

exit()

# Save Fisher and keys list to file
np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Fisher_Eg_'+endfilename+'.txt', Fisher_Eg)
np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Fisher_jp_'+endfilename+'.txt', Fisher_jp)

f_keys = open("/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/keys_"+endfilename+".txt", "w")
for key in keys_list:
	f_keys.write(key+"\n")
f_keys.close() 

Fisher_Eg = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Fisher_Eg_'+endfilename+'.txt')
Fisher_jp = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Fisher_jp_'+endfilename+'.txt')

f_keys = "/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/keys_"+endfilename+".txt"

with open(f_keys) as f:
    keys_list = f.readlines()
    
keys_list = [x.strip() for x in keys_list] 

############# ADD PRIORS ############### 

# We will want to add external priors on certain parameters 
# which are poorly constrained by the late-time observables we consider:
# Omega_b, h, n_s, at least.
# We are using priors obtained by Naren from chains reproducing the resutls of 
# Planck 2015 DE / MG, chains by the UT Dallas group.

priors_file = '/home/danielle/Documents/CMU/Research/EG_comparison/storage/Planck_Chains_UT_Dallas/priors_cov_MG_smallmat.txt'
keys_file_priors = '/home/danielle/Documents/CMU/Research/EG_comparison/storage/Planck_Chains_UT_Dallas/Planck4.paramnames_smallmat.txt'

Fisher_Eg = F.add_priors(Fisher_Eg, keys_list, priors_file, keys_file_priors)
Fisher_jp = F.add_priors(Fisher_jp, keys_list, priors_file, keys_file_priors)

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

print '\nTime for completion:', '%.1f' % (time.time() - a), 'seconds'


