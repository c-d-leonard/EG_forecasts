# Script to plot the averaged data vector from Shadab's simulations
# vs my analytic Upsilon_gm and Upsilon_gg data vector to check if units 
# are even remotely similar

import numpy as np
import fiducial as fd
import matplotlib.pyplot as plt

def dvec_avg_sim(r0):
    """ loads the sim data and averages. returns (rp, Ups_gm, Ups_gg) """
	
    datadir = '/home/danielle/Documents/CMU/Research/EG_comparison/data_for_Danielle/'
    datafile_gm = 'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
    datafile_gg = 'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-sel-All-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
	
    data_gm = np.loadtxt(datadir + datafile_gm)
    data_gg = np.loadtxt(datadir + datafile_gg)
	
    # cut below r0
    # first column of upsilon data is rp
    # same for gg and gm so only do it once.
    ind=data_gm[:,0]>r0
    rp_ups=data_gm[ind,0]
    
    ups_gm_all = data_gm[ind, 1:]
    ups_gg_all = data_gg[ind, 1:]
    
    # Average:
    ups_gm = np.zeros(len(rp_ups))
    ups_gg = np.zeros(len(rp_ups))
    for i in range(len(rp_ups)):
        ups_gm[i] = sum(ups_gm_all[i,:]) / len(ups_gm_all[i,:])
        ups_gg[i] = sum(ups_gg_all[i,:]) / len(ups_gg_all[i,:])
		
    return (rp_ups, ups_gm, ups_gg)
	
############

Pimax = 100.

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

endfilename='compare_sims'

params_fid_var = {'mu_0': 0., 'sigma_0':0.} # Intrinsic alignment params? Photo-z params? tau?
params_fid_fix = {'sigma8':0.83,'b':2.2,'OmB':0.05, 'h':0.68, 'n_s':0.96, 'OmM': 0.3}
params_Eg_insens = {'b':2.2, 'sigma8:': 0.83} # Parameters to which E_G is by-design totally insensitive.

params_fid = params_fid_var.copy()   
params_fid.update(params_fid_fix)
lens = 'DESI'
src = 'LSST'

Upgg_me = fd.Upsilon_gg(params_fid, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin=True)
Upgm_me = fd.Upsilon_gm(params_fid, rp_bin_edges, rp0, lens, src, endfilename, nonlin=True)

rp_sim, Upgm_sim, Upgg_sim = dvec_avg_sim(rp0)

# Get in the same units (I hope)
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^2


plt.figure()
plt.loglog(rp_bin_c, Upgg_me, 'o', label='Analytic')	
plt.loglog(rp_sim, rho_crit * Upgg_sim, 'o', label= 'Sim, avg')
plt.legend()
plt.savefig('./Upgg_analytic_v_sim.pdf')
plt.close()

plt.figure()
plt.loglog(rp_bin_c, np.asarray(Upgm_me), 'o', label='Analytic')	
plt.loglog(rp_sim, rho_crit * params_fid['OmM'] * Upgm_sim, 'o', label= 'Sim, avg')
plt.legend()
plt.savefig('./Upgm_analytic_v_sim.pdf')
plt.close()

plt.figure()
plt.loglog(rp_bin_c, np.asarray(Upgm_me) / (rho_crit * params_fid['OmM'] * Upgm_sim ), 'o')	
plt.savefig('./Upgm_analytic_v_sim_ratio.pdf')
plt.close()

plt.figure()
plt.loglog(rp_bin_c, np.asarray(Upgg_me) / (rho_crit * Upgg_sim), 'o')	
plt.savefig('./Upgg_analytic_v_sim_ratio.pdf')
plt.close()

