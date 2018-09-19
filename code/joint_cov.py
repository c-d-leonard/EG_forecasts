import numpy as np
import utils
import specs as sp
import scipy.integrate 
import pyccl as ccl
import utils as u
import matplotlib.pyplot as plt
import scipy.special
import pyfftlog as fft
import fiducial as fid
import cov_hankel as ch

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^2

np.set_printoptions(linewidth=240)

def get_joint_covariance(params, lens, src, rp_bin_edges, rp_bin_c, rp0, endfilename):
    """ Computes the joint covariance matrix of Upsilon_{gm} Upsilon_{gg} and beta.
    params : dictionary of parameters at which to evaluate E_G
    lens : label indicating which lens sample we are using
    src : label indicating which source sample we are using
    rp_bin_edges : edges of projected radial bins
    rp_bin_c : centers of projected radial bins
    rp0 : minimum scale at which to include information for ADSD
    endfilename : tag for the files produced to keep track of the run."""
	
    #### SET UP ####
    nbins = len(rp_bin_c)
    
    # Calculate covariance matrices between Delta Sigma quantities
    cov_DSgm, cov_DSgg, cov_DSgmgg = ch.get_DeltaSigma_covs(params, rp_bin_edges, rp_bin_c, lens, src, endfilename)
    
    # Convert each of these to the covariance in the appropriate Upsilon_xx quantities
    cov_Upgm_avg = cov_Upsilon(cov_DSgm, rp_bin_edges, rp0)
    cov_Upgg_avg = cov_Upsilon(cov_DSgg, rp_bin_edges, rp0)
    cov_Upgmgg_avg = cov_Upsilon(cov_DSgmgg, rp_bin_edges, rp0)
	
    # Get the variance for beta
    beta_var = get_beta_var(lens)
	
    # Load covariance values between Upgg, Upgm and beta - set this to zero for now
    cov_Upgg_beta = np.zeros(len(rp_bin_edges)-1) #np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/covgg_beta_'+lens+'.txt')
    cov_Upgm_beta = np.zeros(len(rp_bin_edges)-1) #np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/covgm_beta_'+lens+'_'+src+'.txt')
	
    # Now stitch it all together:
    joint_cov = np.zeros((2*nbins+1, 2*nbins+1))
    for rpi in range(2*nbins+1):
        for rpj in range(2*nbins+1):
			
            if ((rpi<nbins) and (rpj<nbins)):
                joint_cov[rpi, rpj] = cov_Upgm_avg[rpi, rpj]
            elif ((rpi>=nbins) and (rpi<2*nbins) and (rpj>=nbins) and (rpj<2*nbins)):
                joint_cov[rpi, rpj] = cov_Upgg_avg[rpi-nbins, rpj-nbins]
            elif ((rpi<nbins) and (rpj>=nbins) and (rpj<2*nbins)):
                joint_cov[rpi,rpj] = cov_Upgmgg_avg[rpi, rpj-nbins]
            elif((rpj<nbins) and (rpi>=nbins) and (rpi<2*nbins)):
                joint_cov[rpi, rpj] = cov_Upgmgg_avg[rpj, rpi-nbins]
            elif((rpi==2*nbins) and (rpj<nbins)):
                joint_cov[rpi, rpj] = cov_Upgm_beta[rpj]
            elif((rpj==2*nbins) and (rpi<nbins)):
                joint_cov[rpi, rpj] = cov_Upgm_beta[rpi]
            elif((rpi==2*nbins) and (rpj>=nbins) and (rpj<2*nbins)):
                joint_cov[rpi, rpj] = cov_Upgg_beta[rpj-nbins]
            elif((rpj==2*nbins) and (rpi>=nbins)and (rpi<2*nbins)):
                joint_cov[rpi, rpj] = cov_Upgg_beta[rpi-nbins]
            elif((rpi==2*nbins) and (rpj==2*nbins)):
                joint_cov[rpi, rpj] = beta_var
    
    # Warn if the joint covariance matrix calculated is not symmetric            
    for rpi in range(2*nbins+1):
        for rpj in range(2*nbins+1):
			if (np.abs(joint_cov[rpi, rpj] - joint_cov[rpj, rpi])>10**(-14)):
				print "rpi=",rpi, "rpj=", rpj
				print "not symmetric!"
				print "1=", joint_cov[rpi, rpj], "2=", joint_cov[rpj, rpi]

    # This is just for visualization with the current formatting, do not use this file for inverting the matrix.
    np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/joint_cov_'+endfilename+'.txt', joint_cov, fmt='%1.2e')
	
    return joint_cov
    
def cov_Upsilon(covDS, rp_ed, rp0):
	""" Takes the covariance matrices of a Delta Sigma quantity in bins
	in r_p and outputs the equivalent covariance matrix for Upsilon.
	covDS: covariance matrix in r_p bins of the Delta Sigam
	rp_ed: edges of the projected radial bins
	rp0: projected radius below which to remove information with ADSD
	"""
	
	# Find the bin which contains rp0 
	# We would normally expect this to be the first one but 
	# we leave this general just in case.
	if (rp0< rp_ed[0]):
		#rp0 is lower than the lowest bin so the lowest bin is the closest
		binrp0 = 0
	else:
		binrp0 = (next(j[0] for j in enumerate(rp_ed) if j[1]>rp0)) - 1
	
	cov_Ups_list = [ [covDS[ri, rj] - rp0**2 * (2. / (rp_ed[ri+1]**2 - rp_ed[ri]**2)) * np.log(rp_ed[ri+1] / rp_ed[ri]) * covDS[binrp0, rj] - rp0**2 * (2. / (rp_ed[rj+1]**2 - rp_ed[rj]**2)) * np.log(rp_ed[rj+1] / rp_ed[rj]) * covDS[ri, binrp0] +rp0**4 * (4. / (rp_ed[ri+1]**2 - rp_ed[ri]**2) / (rp_ed[rj+1]**2 - rp_ed[rj]**2)) * np.log(rp_ed[ri+1] / rp_ed[ri]) * np.log(rp_ed[rj+1] / rp_ed[rj]) *  covDS[binrp0, binrp0] for ri in range(len(rp_ed)-1)] for rj in range(len(rp_ed)-1)]
	
	cov_Ups_arr = np.zeros((len(rp_ed)-1, len(rp_ed)-1))
	for ri in range(len(rp_ed)-1):
		cov_Ups_arr[ri, :] = cov_Ups_list[ri]
	
	return cov_Ups_arr
	    
def get_beta_var(lens):
	""" Returns the variance on beta."""
	
	# Load value calculated using White et al. 2008 modified code & square
	beta_var = (np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/beta_err_'+lens+'.txt'))**2
	
	return beta_var
	
def SigCsq_avg(params, lens, src):
	""" Get Sigma C squared, averaged.
	params: dictionary of parameters.
	lens: label for lens sample
	src: label for source sample """
	(zl, dNdzl) = sp.get_dNdzL(params, lens)
	
	(zs, dNdzs) = sp.get_dNdzS(src)
	
	Sigma_inv = sp.get_SigmaC_inv(params, zs, zl)
	
	Sig_inv_sq_zsint = [ scipy.integrate.simps(Sigma_inv[:, zi]**2 * dNdzs, zs) for zi in range(len(zl))]
	Sig_inv_sq_avg = scipy.integrate.simps(Sig_inv_sq_zsint * dNdzl, zl)
	
	Sig_sq_avg = 1. / Sig_inv_sq_avg
	
	# Save this for transfer to cluster if using brute force covariance
	np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Sig_sq_avg_'+lens+'_'+src+'.txt', [Sig_sq_avg])

	return Sig_sq_avg
	   
def shapenoiseonly_cov(params, rp_bin_edges, lens, src):
    """ Returns a diagonal covariance matrix in bins of projected radius 
    for a measurement dominated by shape noise. For debugging only.
    rp_bin_edges : edges of projected radial bins
    lens: lens sample
    src: source sample  """
	
    if ((lens=='DESI') and (src=='LSST')):
        zeff = 0.77
        ns = 26.
        nl = 300.
        Area_l = 3000.
        e_rms = 0.26
        
        # Get the area of each projected radial bin in square arcminutes
        bin_areas       =       get_areas(params, rp_bin_edges, zeff)
        
        SigCavg_sq = SigCsq_avg(params, lens, src)
	
        cov = SigCavg_sq * e_rms**2 / ( nl * Area_l * bin_areas * ns)
        
    elif ((lens=='LOWZ') and (src=='SDSS')):
        zeff = 0.28
        ns = 1.
        nl = 8.7
        Area_l = 7131.
        e_rms = 0.21
        
        # Get the area of each projected radial bin in square arcminutes
        bin_areas       =       get_areas(params, rp_bin_edges, zeff)
        
        SigCavg_sq = SigCsq_avg(params, lens, src)
        
        print "SigCavg=", np.sqrt(SigCavg_sq)
	
        cov = SigCavg_sq * e_rms**2 / ( nl * Area_l * bin_areas * ns)
        
    else:
        raise(ValueError, "That lens / src combination is not yet implemented.")
	
    return cov
    
def shotnoiseonly_cov(params, rp_bin_edges, lens, src, Pimax):
    """ Returns a diagonal covariance matrix in bins of projected radius
    for a measurement of Delta Sigma_gg which is dominated by shot noise.
    For debugging only.
    params :: dictionary of cosmology parameters
    rp_bin_edges : edges of projected radial bins
    lens: lens sample
    src: source sample
    Pimax: projection length of wgg in Mpc/h  """
    
    if ((lens=='DESI') and (src=='LSST')):
        zeff = 0.77
        nl = 300. * 3282.8
        fsky = 0.073
        
        cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
        
        chi_eff = ccl.background.comoving_radial_distance(cosmo, 1./(1.+zeff)) * params['h'] 
	
        cov = [rho_crit**2 * 2. * Pimax * chi_eff**2 / (2. * np.pi**2 * nl**2 * fsky * (rp_bin_edges[i+1]**2 - rp_bin_edges[i]**2)) for i in range(len(rp_bin_edges)-1)]
        
        cov_arr = np.asarray(cov)
        
    else:
        raise(ValueError, "That lens / src combination is not yet implemented.")
    
    return cov_arr
	
def get_areas(params, bins, z_eff):
    """Gets the area of each projected radial bin, in square arcminutes. z_eff = effective lens redshift. """
    
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
    chi_eff = ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_eff)) * params['h']

    # Areas in units (Mpc/h)^2
    areas_mpch = np.asarray([np.pi * (bins[i+1]**2 - bins[i]**2) for i in range(len(bins)-1)])

    # Areas in square arcminutes (466560000 / pi = sqAM in a sphere)
    areas_sqAM = areas_mpch * (466560000. / np.pi) / (4 * np.pi * chi_eff**2)

    return areas_sqAM
    
def plot_covs(params, rp_bin_edges, rp_bin_c, lens, src, Pimax, endfilename):
    """ Plot the imported covariance matrices for inspection.
    For DeltSig_gm and DeltSig_gg plot against shape- and shot-noise
    only calculations for comparison.
    params :: dictionary of cosmology parameters
    rp_bin_edges : edges of projected radial bins
    lens: lens sample
    src: source sample
    Pimax: projection length of wgg in Mpc/h
    endfilename: attach to end of output files to label
    """
	
    cov_DSgm = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/cov_DelSig_gm_'+lens+'_'+src+'_4000rpts.txt')
    cov_DSgg = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/cov_DelSig_gg_'+lens+'_'+src+'_4000rpts.txt')
    cov_DSgmgg = np.loadtxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/cov_DelSig_gmgg_'+lens+'_'+src+'_4000rpts.txt')
    
    cov_gm_shapenoise = shapenoiseonly_cov(params, rp_bin_edges, lens, src)
    cov_gg_shotnoise = shotnoiseonly_cov(params, rp_bin_edges, lens, src, Pimax)

    plt.figure()
    plt.loglog(rp_bin_c, np.sqrt(np.diag(cov_DSgm)), 'mo', label='full')
    plt.hold(True)
    plt.loglog(rp_bin_c, np.sqrt(cov_gm_shapenoise), 'go', label='shape noise')
    plt.ylim(10**(-4), 10**(-1))
    plt.legend()
    plt.savefig('../plots/cov_DSgm_bruteforce_v_shapenoise_'+endfilename+'_4000rpts.png')
    plt.close()
    
    plt.figure()
    plt.loglog(rp_bin_c, np.sqrt(np.diag(cov_DSgg)), 'mo', label='full')
    plt.hold(True)
    plt.loglog(rp_bin_c, np.sqrt(cov_gg_shotnoise), 'go', label='shot noise')
    plt.ylim(10**(-4), 10**(-1))
    plt.legend()
    plt.savefig('../plots/cov_DSgg_bruteforce_v_shotnoise_'+endfilename+'_4000rpts.png')
    plt.close()
    
    plt.figure()
    plt.loglog(rp_bin_c, np.sqrt(np.diag(cov_DSgmgg)), 'mo', label='full')
    plt.legend()
    plt.ylim(10**(-4), 10**(-1))
    plt.savefig('../plots/cov_DSgmgg_bruteforce_'+endfilename+'_4000rpts.png')
    plt.close()

    return
    
def get_Cells(params, ell, lens, src):
    """ Gets the Cell quantities required for the covariance in Delta Sigma """
    
    cosmo= ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
    
    (zL, dNdzL) = sp.get_dNdzL(params, lens)
    (zS, dNdzS) = sp.get_dNdzS(src)
    
    tracer_g = ccl.ClTracerNumberCounts(cosmo, False, False, dNdzL, params['b'] * np.ones(len(zL)), z=zL)
    tracer_k = ccl.ClTracerLensing(cosmo, False, dNdzS, z=zS)
    
    Clgg = ccl.angular_cl(cosmo, tracer_g, tracer_g, ell)
    Clgk = ccl.angular_cl(cosmo, tracer_g, tracer_k, ell)
    Clkk = ccl.angular_cl(cosmo, tracer_k, tracer_k, ell)
    
    # Save so these can be transfered to cluster if using brute-force covariances
    Clgg_save = np.column_stack((ell, Clgg))
    Clgk_save = np.column_stack((ell, Clgk))
    Clkk_save = np.column_stack((ell, Clkk))
    
    np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Clgg_'+lens+'_'+src+'.txt', Clgg_save)
    np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Clgk_'+lens+'_'+src+'.txt', Clgk_save)
    np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/Clkk_'+lens+'_'+src+'.txt', Clkk_save)
    
    return (Clgg, Clgk, Clkk)	

############ CHECK BETA / UPSILON COVARIANCE ###########
	
def cov_beta_Ups(params, lens, src, rp_bin_edges, rp0, Pimax, hder, endfilename):
	""" Get the covariance between Upsilon_{gg} or Upsilon_{gm} due to 
	shot noise. We believe this should be subdominant for LSST + DESI
	but we need to check.
	params: dictionary of cosmological parameters
	lens: label of lens sampel """
	
	# Set up a k and mu vector at which to do the integrals
	# (Result depends on kmax chosen, see White et al. 2008
	
	k = np.logspace(-3, -1, 400)
	mu = np.linspace(-1, 1., 200)
	
	# Get the inverse covariance value at each k and mu
	print "Getting data covariance."
	invcov = Pobs_covinv(params, k, mu, lens)
	
	# Get the derivative of the observed z-space
	# P(k) with respect to beta at each k and mu
	# (linear theory)
	print "Getting beta derivatives."
	dPdbeta = diff_P_beta(params, k, mu, lens)
	
	# Get the derivative of the observed z-space
	# P(k) with respect to Upsilon_gm and Upsilon_gg
	# at each k and mu
	# (linear theory)
	print "Getting Upsilon derivatives."
	(dPdUps_gm, dPdUps_gg) = diff_P_Ups(params, k, mu, lens, src, rp_bin_edges, rp0, Pimax, hder, endfilename)
	
	# Do the integration in k in each case
	print "Doing k integration."
	int_in_k_gm = [[ scipy.integrate.simps(k**2 * dPdbeta[mi] * invcov[mi] * dPdUps_gm[mi][rpi], k) for mi in range(len(mu))] for rpi in range(len(rp_bin_edges)-1)]
	int_in_k_gg = [[ scipy.integrate.simps(k**2 * dPdbeta[mi] * invcov[mi] * dPdUps_gg[mi][rpi], k) for mi in range(len(mu))] for rpi in range(len(rp_bin_edges)-1)]
	
	# And in mu.
	print "Doing mu integration."
	int_in_mu_gm = [ scipy.integrate.simps(np.asarray(int_in_k_gm[rpi]), mu) for rpi in range(len(rp_bin_edges)-1)]
	int_in_mu_gg = [ scipy.integrate.simps(np.asarray(int_in_k_gg[rpi]), mu) for rpi in range(len(rp_bin_edges)-1)]
	
	# Add necessary factors of volume (Mpc/h)^3 and pi etc
	if (lens=='DESI'):
		V = 1.35e10 #(Mpc/h)**3
	else:
		raise(ValueError, "We do not have that lens sample.")
		
	Fish_gm = V * np.asarray(int_in_mu_gm) / (2 * np.pi)**2
	Fish_gg = V * np.asarray(int_in_mu_gg) / (2. * np.pi)**2
	
	# The inverse of this is the covariance element between each Upsilon
	# quantity and beta
	
	cov_beta_gm = 1. / Fish_gm
	cov_beta_gg = 1. / Fish_gg
	
	return	(cov_beta_gm, cov_beta_gg)
	
def diff_P_beta(params, k, mu, lens):
	""" Calculate the derivative of the redshift space power spectrum
	wrt beta at each k and mu
	params: dictionary of cosmological parameters
	k: list or array of wavenumbers
	mu: list of array of angles
	lens: lens sample label
	"""
	
	# Set up a ccl cosmology
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
	if lens=='DESI':
		zeff = 0.77
	else: 
		raise(ValueError, "We don't have support for that lens sample.")
	
	# Get the linear power spectrum at the effective z of the lens sample
	Pklin = ccl.linear_matter_power(cosmo, k * params['h'], 1./ (1. + zeff)) * params['h']**3

	# Get the derivative at each mu / k
	b = params['b']
	f = ccl.growth_rate(cosmo, 1./(1. + zeff))
	beta = f / b
	dPdbeta = [2. * b**2 * (1. + beta*mu[mi]**2)* mu[mi]**2 * Pklin  for mi in range(len(mu))]
	
	return dPdbeta
	
def diff_P_Ups(params, k, mu, lens, src, rp_bin_edges, rp0, Pimax, hder, endfilename):
	""" Calculate the derivative of the redshift power spectrum
	wrt each of Upsilon_gm and Upsilon_gg for each k and mu
	params: dictionary of cosmological parameters
	k: list or array of wavenumbers
	mu: list of array of angles
	lens: lens sample label
	src: source sample label
	rp_bin_edges: edges of bins in projected radisu
	rp0: projected radial value for excluding small scale information
	Pimax: projection length for Upsilon_gg
	hder: numerical differentiation spacing
	endfilename : label to attach to output files
	"""
	# Set up a ccl cosmology at the fiducial parameters
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
	if lens=='DESI':
		zeff = 0.77
	else: 
		raise(ValueError, "We don't have support for that lens sample.s")
	
	# Get the linear matter power spectrum in real space at tthe fiducial parameters
	Pklin = ccl.linear_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zeff)) * params['h']**3
	
	# Get Upsilon_gm and Upsilon_gg at the fiducial parameters
	Upsilon_gm = fid.Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename)
	Upsilon_gg = fid.Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename)
	
	# Two terms to this derivative ( d( b +f mu**2)**2 / dUps * Pk & dPk /dUps * ( b +f mu**2)**2
	
	b = params['b']
	f = ccl.growth_rate(cosmo_fid, 1./(1. + zeff))
	
	dP_1_gm = [ [2. * b * (b + f * mu[mi]**2) / Upsilon_gm[rpi] * Pklin for rpi in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	dP_1_gg = [ [b * (b + f * mu[mi]**2) / Upsilon_gg[rpi] * Pklin for rpi in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	
	# Get second of these terms:
	# We have to do this by numerical differentiation 
	
	# Get parameter dictionaries and cosmologies at sigma8 stepped by hder
	params_up = params.copy()
	params_dn = params.copy()
	params_up['sigma8'] = params['sigma8']+hder
	params_dn['sigma8'] = params['sigma8']-hder
	
	cosmo_up = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=(params['sigma8'] +hder), n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	cosmo_dn = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=(params['sigma8'] -hder), n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
	# Get matter power and Upsilon_gg and gm at each of these cosmologies:
	Pklin_up = ccl.linear_matter_power(cosmo_up, k * params_up['h'], 1./ (1. + zeff)) * params_up['h']**3
	Pklin_dn = ccl.linear_matter_power(cosmo_dn, k * params_dn['h'], 1./ (1. + zeff)) * params_dn['h']**3

	Ups_gm_up = np.asarray(fid.Upsilon_gm(params_up, rp_bin_edges, rp0, lens, src, endfilename))
	Ups_gm_dn = np.asarray(fid.Upsilon_gm(params_dn, rp_bin_edges, rp0, lens, src, endfilename))
	
	Ups_gg_up = np.asarray(fid.Upsilon_gg(params_up, rp_bin_edges, rp0, lens, Pimax, endfilename))
	Ups_gg_dn = np.asarray(fid.Upsilon_gg(params_dn, rp_bin_edges, rp0, lens, Pimax, endfilename))
	
	dPdsig = (Pklin_up - Pklin_dn) /(2. * hder)
	dgmdsig = (Ups_gm_up - Ups_gm_dn) / (2. * hder)
	dggdsig = (Ups_gg_up - Ups_gg_dn) / (2. * hder)
	
	
	dP_2_gm = [ [ (b + f* mu[mi]**2)**2 * dPdsig / dgmdsig[ri] for ri in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	dP_2_gg = [ [ (b + f* mu[mi]**2)**2 * dPdsig / dggdsig[ri] for ri in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	
	dPdgm = [ [ (dP_1_gm[mi][ri] + dP_2_gm[mi][ri]) for ri in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	dPdgg = [ [ (dP_1_gg[mi][ri] + dP_2_gg[mi][ri]) for ri in range(len(rp_bin_edges)-1)] for mi in range(len(mu))]
	
	return (dPdgm, dPdgg)
	
	
def Pobs_covinv(params, k, mu, lens):
	""" Get the inverse covariance of the redshift space observed power 
	spectrum at a list of k and mu (cosine of angle to line of sight) vals.
	params: dictionary of cosmological parameters
	k: list or array of wavenumbers
	mu: list of array of angles
	lens: lens sample label """	
	
	# Set up a ccl cosmology
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
	if lens=='DESI':
		zeff = 0.77
		nbar = 3.2 * 10**(-4)
	else: 
		raise(ValueError, "We don't have support for that lens sample.s")
	
	# Get the linear power spectrum at the effective z of the lens sample
	Pklin = ccl.linear_matter_power(cosmo, k * params['h'], 1./ (1. + zeff)) * params['h']**3
	
	# Get the redshift space galaxy power spectrum in linear theory
	f = ccl.growth_rate(cosmo, 1./(1. + zeff))
	Pgg = [ (params['b'] + f * mu[mi]**2)**2 * Pklin for mi in range(len(mu))]
	
	# Get the covariance matrix at each k and mu
	cov = [ 2. * (Pgg[mi]**2 + 2 * Pgg[mi] / nbar + 1. / nbar**2) for mi in range(len(mu))]
	
	#Pgg_arr = np.zeros((len(k), len(mu)))
	#for mi in range(len(mu)):
	#	Pgg_arr[:, mi] = Pgg[mi]
	
	# Get the inverse at each k and mu
	invcov = [[1./ cov[mi][ki] for ki in range(len(k))] for mi in range(len(mu))]
			
	return invcov
	
	
