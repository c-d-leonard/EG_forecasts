## 

endfilename='test'

import numpy as np
import pyccl as ccl
import fiducial as fid
import matplotlib.pyplot as plt
import utils as u
import Eg_cov as egcov
import joint_cov as jp
import fisher as F

rp0 = 1.5 # Mpc/h
bias = 1.
z = 0.77
h = 0.68

# Set up the fiducial cosmology.
cosmo_fid = ccl.Cosmology(Omega_c = 0.25, Omega_b = 0.05, h = h, sigma8=0.83, n_s = 0.96)

############# GET FIDUCIAL QUANTITIES ##############

# Get the growth rate
beta = ccl.growth_rate(cosmo_fid, z) / bias

# Set up the projected radial vectors
rp = np.logspace(np.log10(rp0), np.log10(50.), 500)
rp_bin_edges = np.logspace(np.log10(rp0), np.log10(50.), 11)
rp_bin_c = u.rp_bins_mid(rp_bin_edges)

# Load the 3D correlation function
(r, xigg) = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/xigg_'+endfilename+'.txt', unpack=True)
# Get wgg and Upsilon_gg
print "getting Upsilon gg"
wgg = fid.wgg(rp, 100., r, xigg)
Upgg = fid.Upsilon_gg(rp, wgg, rp0, rp_bin_edges)

print "getting Upsilon gm"
# Get Upsilon_gm
Upgm = fid.Upsilon_gm(rp, rp_bin_edges, rp0)

############ GET DATA COVARIANCE MATRICES ###########

# Get the joint probe covariance matrix
joint_cov = jp.get_joint_covariance(rp_bin_c)
print "getting Eg samp"
# Get the covariance matrix for E_G given the covariance matrix of the joint probes
Eg_samp = egcov.get_Eg_sample(beta, Upgm, Upgg, joint_cov, Nsamps = 100)
Eg_cov = np.cov(Eg_samp)
np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Eg_cov_'+endfilename+'.txt', Eg_cov)

########### Get the Fisher matrices in each case ############

Fisher_Eg = F.get_Fisher_Eg()

Fisher_jp = F.get_Fisher_jp()

