""" This script gets the covariance of Delta Sigma gm with Sukhdeep's 
hankel transform code. """

import hankel_transform as ht
import numpy as np
import specs as sp
import pyccl as ccl
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate
import joint_cov as jp
import utils as u

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^

def get_DeltaSigma_covs(params, rp_bin_edges, rp_bin_c, lens, src, endfilename):
    """ Get the total covariance matrix for joint probes.
    params : dictionary of cosmological parameters. 
    rp_bin_edges : edges of the projected radial bins 
    lens : label for the lens sample 
    src : label for the source sample """
	
    # Define a ccl cosmology using params
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])

    # Set up the hankel transform we need; we will use this for all the Delta Sigma covariances.
    k = np.logspace(np.log10(8.*10**(-4)), np.log10(30.), 5000)
    HT=ht.hankel_transform(rmin=0.6,rmax=110,kmax=k[-1],j_nu=[2],n_zeros=28000,kmin=k[0])
    
    # Get the required integrals over the line-of-sight window
    (DPi1_gm, DPi2_gm) = DeltaPi_gm(params, lens, src)
    (DPi1_gg, DPi2_gg) = DeltaPi_gg(params, lens)
    DPi2_gmgg          = DeltaPi_gmgg(params, lens, src)
	
    # Get quantities we will need for more than one of the component matrices so we don't have to compute them twice
    ShapeNoise = sp.shape_noise(params, src, lens)
    ShotNoise =  sp.shot_noise(lens)
    vol = sp.volume(params, src, lens) # Volume of the lenses
	
    z, dNdzl = sp.get_dNdzL(params, lens)
    pk_ofz = [ccl.nonlin_matter_power(cosmo, k * params['h'], 1./ (1. + z[zi])) * params['h']**3 for zi in range(len(z))]
    pk_ofz_arr = np.zeros((len(k), len(z)))
    for zi in range(len(z)):
        pk_ofz_arr[:, zi] = pk_ofz[zi]
        
    pk = [ scipy.integrate.simps(pk_ofz_arr[ki, :] * dNdzl, z) for ki in range(len(k)) ]
    
    # Get the covariance of (DS_gm, DS_gm)
    (r_bin, ht_gmgm) = get_cov_DeltSig_gm(params, HT, k, pk, DPi2_gm, ShapeNoise, ShotNoise, vol, rp_bin_edges, lens, src)
    
    # Get the covariance of (DS_gg, DS_gg)
    (r_bin, ht_gggg) = get_cov_DeltSig_gg(params, HT, k, pk, DPi2_gg, ShotNoise, vol, rp_bin_edges)
    
    # Get the covariance of (DS_gm, DS_gg)
    (r_bin, ht_gmgg) = get_cov_DeltSig_gmgg(params, HT, k, pk, DPi2_gmgg, ShotNoise, vol, rp_bin_edges)
    
    return (ht_gmgm, ht_gggg, ht_gmgg)
    
def get_cov_DeltSig_gm(params, HT, k, pk, DPi2_l, ShapeNoise, ShotNoise, vol, rp_bin_edges, lens, src):
    """ Get the covarariance of Delta Sigma gm at projected radius rp. 
    params : dictionary of cosmological parameters
    HT : Hankel Transform set up structure from Sukdheep's code
    k : k at which the matter power spectrum is calculated
    pk : matter power spectrum integrated over lens z dist
    DPi2_l : integral over the lensing window function squared
    SigCavg: Sigma crit inv averaged over dists and inverted again
    ShapeNoise : shape noise associated with source sample
    ShotNoise : shot noise associated with lens sample
    vol : survey volume in (Mpc/h)**3 
    rp_bin_edges : bins in projected radius """
    
    p_g = np.asarray(pk) * params['b']**2
    p_kappa = np.asarray(pk) * (rho_crit * (params['OmM']))**2 * DPi2_l
    
    SigC2 = jp.SigCsq_avg(params, lens, src)

    # Taper and get the ggkk covariance term then bin in r
    taper_kw               =     dict({'large_k_lower':10,'large_k_upper':k[-1],'low_k_lower':k[0],'low_k_upper':k[0]*1.2})
    r,cov_ggkk             =     HT.projected_covariance(k_pk = k, pk1 = p_g + ShotNoise, pk2 = p_kappa + (ShapeNoise * SigC2), j_nu=2, taper=True, **taper_kw)
    r_bin, cov_ggkk_bin    =     HT.bin_cov(r = r,cov = cov_ggkk,r_bins = rp_bin_edges)
    
    # Get the gkgk covariance term
    p_gk                   =     np.asarray(pk) * params['b'] * rho_crit * (params['OmM'])
    r,cov_gkgk_tmp         =     HT.projected_covariance(k_pk = k, pk1 = p_gk, pk2 = p_gk, kmax=100, j_nu=2, rmin=.8, rmax=110, n_zeros=3500)
    cov_gkgk               =     cov_gkgk_tmp * DPi2_l
    r_bin,cov_gkgk_bin     =     HT.bin_cov(r=r,cov=cov_gkgk,r_bins=rp_bin_edges)
    
    #plt.figure()
    #plt.loglog(k, 2. * ( p_g + ShotNoise) * (p_g + ShotNoise), 'b', label='2(Pgg+1/ng)')
    #plt.hold(True)
    #plt.loglog(k, (p_kappa / DPi2_l + ShapeNoise) * (p_g + ShotNoise) + p_gk * p_gk , 'm', label='PkkPgg terms')
    #plt.legend()
    #plt.savefig('../plots/powerspectra_terms_'+lens+'_'+src+'.png')
    #plt.close()
    
    #plt.figure()
    #plt.loglog(k, 2. * ( p_g + ShotNoise) / (p_kappa / DPi2_l + ShapeNoise  + p_gk * p_gk / (p_g + ShotNoise)), 'b', label='ratio')
    #plt.legend()
    #plt.savefig('../plots/powerspectra_terms_ratio_'+lens+'_'+src+'.png')
    #plt.close()
    cov = (cov_ggkk_bin + cov_gkgk_bin) / vol

    return (r_bin, cov)
    
def get_cov_DeltSig_gg(params, HT, k, pk, DPi2, ShotNoise, vol, rp_bin_edges):
    """ Get the covarariance of Delta Sigma gg at projected radius rp. 
    params : dictionary of cosmological parameters
    HT : Hankel Transform set up structure from Sukdheep's code
    k : k at which the matter power spectrum is calculated
    pk : matter power spectrum integrated over lens z dist
    DPi2 : integral over the lens galaxy clustering function squared
    ShotNoise : shot noise associated with lens sample
    vol : survey volume in (Mpc/h)**3 
    rp_bin_edges : bins in projected radius """
    
    # galaxy - galaxy power spectrum
    p_g      =     np.asarray(pk) * params['b']**2
    
    # Taper and get the ggkk covariance term then bin in r
    taper_kw               =     dict({'large_k_lower':10,'large_k_upper':k[-1],'low_k_lower':k[0],'low_k_upper':k[0]*1.2})
    r,cov_gggg_tmp         =     HT.projected_covariance(k_pk = k, pk1 = p_g + ShotNoise, pk2 = p_g + ShotNoise, j_nu=2, taper=True, **taper_kw)
    cov_gggg               =     cov_gggg_tmp * 2. * DPi2 * rho_crit * rho_crit
    r_bin, cov_gg_bin      =     HT.bin_cov(r = r,cov = cov_gggg,r_bins = rp_bin_edges)
    
    cov = cov_gg_bin / vol

    return (r_bin, cov)
    
def get_cov_DeltSig_gmgg(params, HT, k, pk, DPi2, ShotNoise, vol, rp_bin_edges):
    """ Get the covarariance of Delta Sigma gm with Delta Sigma gg at 
    projected radius rp. 
    params : dictionary of cosmological parameters
    HT : Hankel Transform set up structure from Sukdheep's code
    k : k at which the matter power spectrum is calculated
    pk : matter power spectrum integrated over lens z dist
    DPi2_g : integral over the lens galaxy clustering function squared
    ShotNoise : shot noise associated with lens sample
    vol : survey volume in (Mpc/h)**3 
    rp_bin_edges : bins in projected radius """
    
    # galaxy - galaxy power spectrum
    p_g = np.asarray(pk) * params['b']**2
    # galaxy - matter power spectrum
    p_gk     =     np.asarray(pk) * params['b'] * rho_crit * (params['OmM'])
    
    # Taper and get the ggkk covariance term then bin in r
    taper_kw               =     dict({'large_k_lower':10,'large_k_upper':k[-1],'low_k_lower':k[0],'low_k_upper':k[0]*1.2})
    r,cov_gmgg_tmp         =     HT.projected_covariance(k_pk = k, pk1 = p_gk, pk2 = p_g + ShotNoise, j_nu=2, taper=True, **taper_kw)
    cov_gmgg               =     cov_gmgg_tmp * 2. * DPi2 * rho_crit
    r_bin, cov_gmgg_bin      =     HT.bin_cov(r = r,cov = cov_gmgg,r_bins = rp_bin_edges)
    
    cov = cov_gmgg_bin / vol

    return (r_bin, cov)
    
def W_ofPi_lensing(params, lens, src):
    """ Computes W(Pi) as in Singh et al. 2016 for Delta Sigma_gm (GGL
    params : dictionary of cosmological parameters
    lens : keyword for lens galaxy distribution
    src : keyword for source galaxy sample """
	
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
    # The numerator is just Sigma_c averaged over lens and src distributions
    SigCavg = np.sqrt(jp.SigCsq_avg(params, lens, src))
	
    (zl, dNdzl) = sp.get_dNdzL(params,lens)
    chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
    (zs, dNdzs) = sp.get_dNdzS(src)
	
    # Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
    Pipos = scipy.logspace(np.log10(0.1), np.log10(3000),4000)
    Pi_rev= list(Pipos)
    Pi_rev.reverse()
    Pi_neg = Pi_rev 
    Pi = np.append(-np.asarray(Pi_neg), Pipos) 

    # Get the redshift at zl + Pi (2D list)
    z_ofChi = u.z_ofcom_func(params)
    z_Pi = [[z_ofChi(chil[zli] + Pi[pi]) if (chil[zli] + Pi[pi]>=0.) else -100. for pi in range(len(Pi))] for zli in range(len(zl))]
	
	# Get the radial comoving distance at the source z's and at z_Pi (lenses)	
    com_s = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zs)) * params['h']
    com_zPi = [[chil[zli] + np.asarray(Pi[pi]) for pi in range(len(Pi))] for zli in range(len(zl))] 

    # Get Sigma_crit(chis, chil + Pi)
    Sig_inv_ofPi = [ [ sp.get_SigmaC_inv_com(params, com_s, com_zPi[zli][pi], z_Pi[zli][pi]) for zli in range(len(zl))]  for pi in range(len(Pi)) ]  

    # Integrate this over source redshift distribution
    zs_int = [[ scipy.integrate.simps(np.asarray(Sig_inv_ofPi[pi][zli]) * dNdzs, zs) for zli in range(len(zl))] for pi in range(len(Pi))] 
    
    # Integrate then over lens distribution
    zl_int = [ scipy.integrate.simps(zs_int[pi] * dNdzl, zl) for pi in range(len(Pi))]
	
	# Mutliply by the fully averaged SigC over the full lens and source
    W_ofPi = np.asarray(zl_int) * SigCavg
	
    return (Pi, W_ofPi)
	
def DeltaPi_gm(params, lens, src):
	""" Computes Delta Pi_1 and Delta Pi 2 as defined in Singh et al. 2016. 
	params: parameters dictionary.
	lens: keyword for lens distribution
	src: keyword for source distribution. """
	
	(Pi, W) = W_ofPi_lensing(params, lens, src)
	
	DeltaPi1 = scipy.integrate.simps(W, Pi)
	
	DeltaPi2 = scipy.integrate.simps(W**2, Pi)
	
	return (DeltaPi1, DeltaPi2)
	
def DeltaPi_gg(params, lens):
    """ Computes Delta Pi_1 and Delta Pi 2 as defined in Singh et al. 2016. 
    for the gg covariance. 
    params : dictionary of cosmological parameters
    lens: keyword for lens distribution """
	
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
    if lens=='LOWZ':
        DeltaPi1 = 200. # From Singh et al. 2017
        DeltaPi2 = 200.
    elif (lens=='DESI' or lens=='DESI_plus_20pc' or lens=='DESI_plus_50pc' or lens=='DESI_4MOST_LRGs' or lens=='DESI150pc_4MOST_LRGs' or lens=='DESI200pc_4MOST_LRGs'):
        DeltaPi1 = (ccl.comoving_radial_distance(cosmo, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo, 1./ (1. +0.6))* params['h'] )
        DeltaPi2 = DeltaPi1
    elif (lens=='DESI_4MOST_ELGs' or lens=='DESI150pc_4MOST_ELGs' or lens=='DESI200pc_4MOST_ELGs' or lens=='DESI_4MOST_18000deg2_ELGs' or lens=='DESI_4MOST_18000deg2_LRGs'):
        DeltaPi1 = (ccl.comoving_radial_distance(cosmo, 1./ (1. +1.5))* params['h'] ) - (ccl.comoving_radial_distance(cosmo, 1./ (1. +0.6))* params['h'] )
        DeltaPi2 = DeltaPi1
    else:
        raise(ValueError, "That lens sample is not supported.")
	
    return (DeltaPi1, DeltaPi2)
	
def DeltaPi_gmgg(params, lens, src):
	""" Computes Delta Pi_2 for Delta Sigma gm x Delta Sigma gg
	params : dictionary of cosmological parameters
	lens: keyword for lens distribution
	src : keyowrd for source galaxy sample """
	
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
	# Get W(Pi) for Delta Sigma
	(Pi, W) = W_ofPi_lensing(params, lens, src)
	
	# The difference between this and the DS_gm x DS_gm calculation is 
	# that we have a top hat window function in Pi for the DS_gg factor:
	
	# Get the indices of Pi we should be using for our integration
	if lens=='LOWZ':
		# In this case we use chi(zl) - 100 Mpc/h and chi(zl) + 100 Mpc/h
		indPilow = next(j[0] for j in enumerate(Pi) if j[1]>-100 )
		indPihigh = next(j[0] for j in enumerate(Pi) if j[1]>100 )
	elif (lens=='DESI' or lens=='DESI_plus_20pc' or lens=='DESI_plus_50pc' or lens=='DESI_4MOST_LRGs' or lens=='DESI150pc_4MOST_LRGs' or lens=='DESI200pc_4MOST_LRGs'):
		# Here we want Pi between z=0.6 and z= 1.0, the redshift extent 
		# of the DESI LRGs
		chi_diff_low = ccl.comoving_radial_distance(cosmo, 1./ (1. +0.6))* params['h']  - ccl.comoving_radial_distance(cosmo, 1./ (1. +0.77))* params['h'] 
		chi_diff_high = ccl.comoving_radial_distance(cosmo, 1./ (1. +1.0))* params['h']  - ccl.comoving_radial_distance(cosmo, 1./ (1. +0.77))* params['h']
		indPilow = next(j[0] for j in enumerate(Pi) if j[1]>chi_diff_low)
		indPihigh = next(j[0] for j in enumerate(Pi) if j[1]>chi_diff_high )
	elif (lens=='DESI_4MOST_ELGs' or lens=='DESI150pc_4MOST_ELGs' or lens=='DESI200pc_4MOST_ELGs' or lens=='DESI_4MOST_18000deg2_ELGs' or lens=='DESI_4MOST_18000deg2_LRGs'):
		# Here we want Pi between z=0.6 and z= 1.5, the redshift extent 
		# of the DESI LRGs
		chi_diff_low = ccl.comoving_radial_distance(cosmo, 1./ (1. +0.6))* params['h']  - ccl.comoving_radial_distance(cosmo, 1./ (1. +1.0))* params['h'] 
		chi_diff_high = ccl.comoving_radial_distance(cosmo, 1./ (1. +1.5))* params['h']  - ccl.comoving_radial_distance(cosmo, 1./ (1. +1.0))* params['h']
		indPilow = next(j[0] for j in enumerate(Pi) if j[1]>chi_diff_low)
		indPihigh = next(j[0] for j in enumerate(Pi) if j[1]>chi_diff_high )	
	else:
		raise(ValueError, "That lens sample is not supported.")
		
	Pi_integ = Pi[indPilow:indPihigh]
	W_integ = W[indPilow:indPihigh]
	
	DeltaPi2 = scipy.integrate.simps(W_integ, Pi_integ)
	
	return DeltaPi2
	
