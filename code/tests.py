import numpy as np
import scipy.integrate
import scipy.interpolate
import pyccl as ccl
import specs 
import utils as u
import subprocess as sp

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^2

def Upsilon_gm_from_rho(params, cosmo_fid, rp_bin_edges, rp0, lens, endfilename):
	""" Computes Upsilon_gm as it is theoretically given in GR using
	an integral over the correlation function and direct multiplication 
	by density. 
	params = dictionary of cosmological parameters.
	rp_bin_edges : edges of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	endfilename : tag for the files produced to keep track of the run."""
	
	# Set up the fiducial cosmology.
	#cosmo_fid = ccl.Cosmology(Omega_c = params['OmC'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
	
	print "get the lens distribution"
	# Get the distribution over the lens redshifts and save
	(zl, dNdzl) = specs.get_dNdzL(lens)
	zlsave = [str('{:1.12f}'.format(zl[zi])) for zi in range(len(zl))]
	zfile = open("/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/zl_"+endfilename+"_upgm.txt", "w")
	for z in zlsave:
		zfile.write("%s\n" % z)
	zfile.close()
	print "get the chil"
	# Get the radial comoving distances (in Mpc/h) at the same zl
	chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
	print "get p(k)"
	# Get the power spectrum at each zl
	k = np.logspace(-4, 5, 40000) # units h / Mpc
	Pkgm_ofzl = [[ params['b']*ccl.nonlin_matter_power(cosmo_fid, k[ki] * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for ki in range(len(k))] for zi in range(len(zl))]
	# Save (to enable FFTing into correlation function)
	for zi in range(0,len(zl)):
		pksave = np.column_stack((k, Pkgm_ofzl[zi]))
		np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/pkgm_z='+zlsave[zi]+'_'+endfilename+'.txt', pksave)
		
	# Call external FFT program
	sp.call("/home/danielle/Documents/CMU/Software/FFTLog-master-slosar/test_gm_multiz.out "+ endfilename, shell=True)

	print "loading"
	# Load the correlation functions, from the files produced from the external FFT
	corr_gm = [np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/xigm_z='+zlsave[zi]+'_'+endfilename+'.txt', unpack=True)[1] for zi in range(len(zlsave))]
	
	# r is the same for each case so just load that once.
	r = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/xigm_z='+zlsave[0]+'_'+endfilename+'.txt', unpack=True)[0]
	
	# Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
	Pipos = scipy.logspace(np.log10(0.0001), np.log10(100),100)
	Pi_rev= list(Pipos)
	Pi_rev.reverse()
	index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
	Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
	Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]
	
	rp = np.logspace(np.log10(rp0), np.log10(50.), 50) 
	
	corr_interp = [scipy.interpolate.interp1d(np.log(r), corr_gm[zi]) for zi in range(len(zl))]
	corr_rp_term = [[ corr_interp[zi](np.log(np.sqrt(rp[rpi]**2 + Pi[zi]**2))) for zi in range(len(zl))] for rpi in range(len(rp))]
	
	Pi_int = [ [ scipy.integrate.simps(np.asarray(corr_rp_term[rpi][zli]), Pi[zli]) for zli in range(len(zl))] for rpi in range(0, len(rp))]
	
	# Do the integral over zl 
	zl_int = [ scipy.integrate.simps(dNdzl * Pi_int[rpi], zl) for rpi in range(0,len(rp))]
	"""zl_int = np.zeros(len(rp))
	for rpi in range(len(rp)):
		zl_int[rpi] = Pi_int[rpi][95] * dNdzl[95]"""
	
	# Get Sigma_gm
	Sig_gm =  rho_crit * (params['OmC'] + params['OmB']) * np.asarray(zl_int)
	
	# Now process slightly to get Upsilon_gm
	
	# We need a more well-sampled rp vector for integration
	rp_finer = np.logspace(np.log10(rp[0]), np.log10(rp[-1]), 30000)
	interp_Sig_gm= scipy.interpolate.interp1d(np.log(rp), np.log(np.asarray(Sig_gm)))
	Sig_gm_finer = np.exp(interp_Sig_gm(np.log(rp_finer)))
	
	index_rp = [next(j[0] for j in enumerate(rp_finer) if j[1]>= rp[rpi]) for rpi in range(len(rp))]
	
	first_term = [ ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(Sig_gm_finer[0:index_rp[rpi]] * rp_finer[0:index_rp[rpi]]**2, np.log(rp_finer[0:index_rp[rpi]])) for rpi in range(1, len(rp))]
	
	Ups_gm = (np.asarray(first_term) - np.asarray(Sig_gm)[1:] + (rp0 / np.asarray(rp[1:]))**2 * Sig_gm[0]) 
	#Ups_gm = (np.asarray(first_term) - np.asarray(Sig_gm_finer)[1:] + (rp0 / np.asarray(rp_finer[1:]))**2 * Sig_gm_finer[0]) 
	
	# Get this in each bin
	Ups_gm_binned = u.average_in_bins(Ups_gm, rp[1:], rp_bin_edges)
	
	return Ups_gm_binned
	
	
	
	
	
	
	
	
