# Functions for calculating fiducial signals that we can't get elsewhere.

import numpy as np
import scipy.integrate
import scipy.interpolate
import pyccl as ccl
import utils as u
import subprocess as sp
import specs
import matplotlib.pyplot as plt
import pyfftlog as fft

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^2

def wgg(params, rp, lens, Pimax, endfilename, nonlin = False):
	""" Projects the 3D gg correlation function to get wgg 
	params : dictionary of parameters at which to evaluate E_G
	rp: a vector of projected radial distances at which to compute wgg
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	nonlin (optional) : use nonlinear halofit correction """
	
	# Set up the fiducial cosmology.
	if nonlin==False:
		matpow_label = 'linear'
	else:
		matpow_label = 'halofit'
	
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum=matpow_label)
	
	# Get the distribution over the lens redshifts and save
	#print "getting dNdzl"
	(zl, dNdzl) = specs.get_dNdzL(params, lens)
	#print "zl=", zl, "dNdzl=", dNdzl
	
	#zlsave = [str('{:1.12f}'.format(zl[zi])) for zi in range(len(zl))]
	#zfile = open("/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/zl_"+endfilename+"_wgg.txt", "w")
	#for z in zlsave:
	#	zfile.write("%s\n" % z)
	#zfile.close()
	# Get the radial comoving distances (in Mpc/h) at the same zl
	#print "getting comoving"
	chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
	# Get the power spectrum at each zl
	k = np.logspace(-4, 5, 40000) # units h / Mpc
	# We use the nonlin matter power function, but if nonlin=False this will still be the linear case.
	Pkgg_ofzl = [ params['b']**2* ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]

	r_corr_gg = [fft.pk2xi(k, Pkgg_ofzl[zli]) for zli in range(len(zl))]
	r = r_corr_gg[0][0]
	corr_gg = [r_corr_gg[zli][1] for zli in range(len(zl))]
		
	# Call external FFT program
	#sp.call("/home/danielle/Documents/CMU/Software/FFTLog-master-slosar/test_gg_multiz.out "+ endfilename, shell=True)
	
	interp_corr = [scipy.interpolate.interp1d(np.log(r), corr_gg[zi]) for zi in range(len(zl))]

	if (min(rp)<(min(r)/np.sqrt(2))):
		raise(ValueError, "You have asked for wgg at a projected radial value too small for the radial vector you have passed.")
		
	# Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
	Pipos = scipy.logspace(np.log10(0.0001), np.log10(Pimax),50)
	Pi_rev= list(Pipos)
	Pi_rev.reverse()
	index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
	Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
	Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]
	
	corr_2d = [[interp_corr[zi](np.log(np.sqrt(rp[rpi]**2 + Pi[zi]**2))) for zi in range(len(zl))] for rpi in range(len(rp)) ]
	
	projected = [[scipy.integrate.simps(corr_2d[rpi][zi], Pi[zi]) for zi in range(len(zl))] for rpi in range(len(rp))] 
	
	wgg = [scipy.integrate.simps(projected[rpi] * dNdzl, zl) for rpi in range(len(rp))]
	
	return wgg

def Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin=False):
    """ Takes wgg in Mpc/h and gets Upsilon_gg in Msol h / pc^2 for a given rp0. 
    params : dictionary of parameters at which to evaluate E_G
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
    lens : label indicating which lens sample we are using
    Pimax : maximum integration for wgg along LOS, Mpc/h
    endfilename : tag for the files produced to keep track of the run.
    nonlin (optional) : set to true if we want to use nonlinear halofit corrections.
    """
    
    rp = np.logspace(np.log10(rp0), np.log10(50.), 50)
    w_gg = wgg(params, rp, lens, Pimax, endfilename, nonlin=nonlin)

    rp_finer = np.logspace(np.log10(rp[0]), np.log10(rp[-1]), 5000)
    wgg_interp = scipy.interpolate.interp1d(np.log(rp), np.log(w_gg))
    w_gg_finer = np.exp(wgg_interp(np.log(rp_finer)))
	
    index_rp = [next(j[0] for j in enumerate(rp_finer) if j[1]>= rp[rpi]) for rpi in range(len(rp))]
    first_term = [ ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(w_gg_finer[0:index_rp[rpi]] * rp_finer[0:index_rp[rpi]]**2, np.log(rp_finer[0:index_rp[rpi]])) for rpi in range(1, len(rp))]

    Ups_gg = rho_crit * (np.asarray(first_term) - np.asarray(w_gg[1:]) + rp0**2 / np.asarray(rp[1:])**2 * w_gg[0]) 
    Ups_gg_binned = u.average_in_bins(Ups_gg, rp[1:], rp_bin_edges)
	
    return Ups_gg_binned
	
def Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin=False):
	""" Gets Upsilon_gm in Msol h / pc^2 for a given rp0.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	endfilename : tag for the files produced to keep track of the run.
	nonlin(optional) : set to true if we want to use halofit nonlinear correction. """
	
	# Set up the fiducial cosmology.
	
	if nonlin==False:
		matpow_label = 'linear'
	else:
		matpow_label = 'halofit'
	
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum=matpow_label)
	
	# Get the distribution over the lens redshifts and save
	(zl, dNdzl) = specs.get_dNdzL(params, lens)
	#zlsave = [str('{:1.12f}'.format(zl[zi])) for zi in range(len(zl))]
	#zfile = open("/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/zl_"+endfilename+"_upgm.txt", "w")
	#for z in zlsave:
	#	zfile.write("%s\n" % z)
	#zfile.close()

	# Get the radial comoving distances (in Mpc/h) at the same zl
	chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
	# Get the power spectrum at each zl
	k = np.logspace(-4, 5, 40000) # units h / Mpc
	Pkgm_ofzl = [ params['b']*ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
	
	r_corr_gm = [fft.pk2xi(k, Pkgm_ofzl[zli]) for zli in range(len(zl))]
	r = r_corr_gm[0][0]
	corr_gm = [r_corr_gm[zli][1] for zli in range(len(zl))]
	
	# Save (to enable FFTing into correlation function)
	#for zi in range(0,len(zl)):
	#	pksave = np.column_stack((k, Pkgm_ofzl[zi]))
	#	np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/pkgm_z='+zlsave[zi]+'_'+endfilename+'.txt', pksave)
		
	# Call external FFT program
	#sp.call("/home/danielle/Documents/CMU/Software/FFTLog-master-slosar/test_gm_multiz.out "+ endfilename, shell=True)
	
	# Load the correlation functions, from the files produced from the external FFT
	#corr_gm = [np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/xigm_z='+zlsave[zi]+'_'+endfilename+'.txt', unpack=True)[1] for zi in range(len(zlsave))]

	# r is the same for each case so just load that once.
	#r = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/xigm_z='+zlsave[0]+'_'+endfilename+'.txt', unpack=True)[0]
	
	# Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
	Pipos = scipy.logspace(np.log10(0.0001), np.log10(100),50)
	Pi_rev= list(Pipos)
	Pi_rev.reverse()
	index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
	Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
	Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]

	# Interpolate the correlation function in 2D (rp & Pi)
	rp = np.logspace(np.log10(rp0), np.log10(50.), 50)  
	
	corr_interp = [scipy.interpolate.interp1d(np.log(r), corr_gm[zi]) for zi in range(len(zl))]
	corr_rp_term = [[ corr_interp[zi](np.log(np.sqrt(rp[rpi]**2 + Pi[zi]**2))) for zi in range(len(zl))] for rpi in range(len(rp))]

	# Get the source redshift distribution
	(zs, dNdzs) = specs.get_dNdzS(src)

	# Equivalent comoving distances
	chis = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zs)) * params['h']
	
	# Get the redshift at zl + Pi (2D list)
	z_ofChi = u.z_ofcom_func(params)
	z_Pi = [[z_ofChi(chil[zli] + Pi[zli][pi]) for pi in range(len(Pi[zli]))] for zli in range(len(zl))] 

	# Do the integral over zs
	wSigC = specs.weights_times_SigC(params, src, zs, zl)
	# Get the index of the zs vector that corresponds to zl + z(Pi) ( = z_Pi)
	index_low = [[next(j[0] for j in enumerate(zs) if j[1]>= z_Pi[zli][pi]) for pi in range(0,len(Pi[zli]))] for zli in range(len(zl))]
	 
	zs_int = [ [ scipy.integrate.simps( dNdzs[index_low[zli][pi]:] * ( chis[index_low[zli][pi]:] - chil[zli] - Pi[zli][pi]) / chis[index_low[zli][pi]:] * wSigC[:, zli][index_low[zli][pi]:], zs[index_low[zli][pi]:]) for pi in range(len(Pi[zli]))] for zli in range(len(zl))]

	# Get the normalization for the weights
	w = specs.weights(params, src,zs,zl)
	zs_int_w = [[ scipy.integrate.simps(dNdzs[index_low[zli][pi]:]  * w[:,zli][index_low[zli][pi]:] , zs[index_low[zli][pi]:] ) for pi in range(len(Pi[zli]))] for zli in range(len(zl))]
	
	# Do the integral over Pi
	Sigma = [ [ ccl.Sig_MG(cosmo_fid, 1. / (1. + z_Pi[zli][pi])) for pi in range(len(Pi[zli]))] for zli in range(len(zl))] 
	
	Pi_int = [ [ scipy.integrate.simps( (1. + np.asarray(Sigma[zli])) * np.asarray(zs_int[zli]) / np.asarray(zs_int_w[zli]) * (chil[zli] + Pi[zli]) * (zl[zli] + 1.) * np.asarray(corr_rp_term[rpi][zli]), Pi[zli]) for zli in range(len(zl))] for rpi in range(0, len(rp))]

	# Do the integral over zl 
	zl_int = [ scipy.integrate.simps(dNdzl * Pi_int[rpi], zl) for rpi in range(0,len(rp))]
	
	# Now do the averaging over rp:
	# We need a more well-sampled rp vector for integration
	rp_finer = np.logspace(np.log10(rp[0]), np.log10(rp[-1]), 5000)
	interp_zl_int = scipy.interpolate.interp1d(np.log(rp), np.log(np.asarray(zl_int)))
	zl_int_finer = np.exp(interp_zl_int(np.log(rp_finer)))
	
	# Get the index of the previous rp vector which corresponds to this one:
	index_rp = [next(j[0] for j in enumerate(rp_finer) if j[1]>= rp[rpi]) for rpi in range(len(rp))]
	first_term = [ ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(zl_int_finer[0:index_rp[rpi]] * rp_finer[0:index_rp[rpi]]**2, np.log(rp_finer[0:index_rp[rpi]])) for rpi in range(1, len(rp))]

	Ups_gm = 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * rho_crit * (params['OmM']) * (np.asarray(first_term) - np.asarray(zl_int)[1:] + (rp0 / np.asarray(rp[1:]))**2 * zl_int[0]) 

	Ups_gm_binned = u.average_in_bins(Ups_gm, rp[1:], rp_bin_edges)
	
	return Ups_gm_binned
	
def beta(params, lens):
	""" Gets beta.
	 params : dictionary of parameters at which to evaluate E_G
	 lens : label indicating which lens sample we are using
	 """
	
	# Set up the cosmology.
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
	# Set up the dNdz for the lens galaxies (the galaxies for which we get wgg)
	(zl, dNdzl) = specs.get_dNdzL(params, lens)
	
	beta_ofzl = [ccl.growth_rate(cosmo, 1./(1. + zl[zi])) / params['b'] for zi in range(len(zl))]
	
	# Integrate over the lens galaxy dNdz
	beta_val = scipy.integrate.simps(beta_ofzl * dNdzl, zl)
	
	return beta_val

def E_G(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin = False):
	""" Returns the value of E_G given it's components.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	nonlin (optional): set to true if we want to use halofit nonlinear corrections."""
	
	# Get beta
	beta_val = beta(params, lens) # beta is definitionally linear so we don't need to pass it nonlin
	# Get wgg and Upsilon_gg
	Upgg = Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin = nonlin)
	# Get Upsilon_gm
	Upgm = Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin = nonlin)
	
	if (len(Upgm)!=len(Upgg)):
		raise(ValueError, "Upsilon_gm and Upsilon_gg must be the same number of rp bins.");
		
	if (hasattr(beta_val, 'len')):
		raise(ValueError, "beta should be a single float.")
		
	Eg = np.asarray(Upgm) / (beta_val * np.asarray(Upgg))
	
	return Eg
	
def jp_datavector(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin=False):
	""" Returns the value of E_G given it's components.
	params : dictionary of parameters at which to evaluate E_G
	rp_bin_edges : edges of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	nonlin (optional): set to true if we want to use halofit nonlinear corrections."""
	
	# Get beta
	beta_val = np.asarray(beta(params, lens))
	# Get wgg and Upsilon_gg
	Upgg = np.asarray(Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin = nonlin))
	# Get Upsilon_gm
	Upgm = np.asarray(Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin = nonlin))
	
	if (len(Upgm)!=len(Upgg)):
		raise(ValueError, "Upsilon_gm and Upsilon_gg must be the same number of rp bins.");
		
	if (hasattr(beta_val, 'len')):
		raise(ValueError, "beta should be a single float.")
		
	data_vec = np.append(Upgm, np.append(Upgg, beta_val))
	
	if(len(data_vec)!= (len(Upgm) + len(Upgg) +1)):
		raise(ValueError, "Something has gone wrong with the length of the data vector.")
	
	return data_vec
