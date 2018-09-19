import numpy as np
import scipy.interpolate
import pyccl as ccl
import astropy.convolution

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)

def get_dNdzL(params, lens_sample):
	""" Imports the lens redshift distribution from file, normalizes, interpolates, and outputs at the z vector that's passed."""
	
	if (lens_sample=='DESI'):
		dNdz_file = 'DESI_redshifts_2col.txt'
		
		z, dNdz = np.loadtxt(dNdz_file, unpack=True)
		
		interpolation = scipy.interpolate.interp1d(z, dNdz)
	
		# Create a well-sampled redshift vector to make sure we can get the normalization without numerical problems
		z_highres = np.linspace(z[0], z[-1], 50)
	
		dNdz_getnorm = interpolation(z_highres)
	
		norm = scipy.integrate.simps(dNdz_getnorm, z_highres)
	
		return  (z_highres, dNdz_getnorm / norm)
		
	elif (lens_sample=='LOWZ'):
		
		cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
		
		
		dNdz_file = 'SDSS_LRG_DR7dim_nofz.txt'
		z, dNdz = np.loadtxt(dNdz_file, unpack=True)
		# Filter the curve 
		nofz_filt = astropy.convolution.convolve(dNdz, astropy.convolution.Box1DKernel(10))
		# Convert to dNdz
		OmL = 1. - params['OmM'] 
		c_over_H = 1. / ((10**(5)/c) * ( (params['OmM'])*(1.+z)**3 + OmL )**(0.5))
		fsky = 7131. / 41253.
		chi = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+z)) * params['h']
		dNdz = nofz_filt * 4. * np.pi * fsky * chi**2 * c_over_H # See notes October 12 2017 for this expression.
		
		interpolation = scipy.interpolate.interp1d(z, dNdz)
	
		# Create a well-sampled redshift vector to make sure we can get the normalization without numerical problems
		z_highres = np.linspace(z[0], z[-1], 50)
	
		dNdz_getnorm = interpolation(z_highres)
	
		norm = scipy.integrate.simps(dNdz_getnorm, z_highres)
	
		return (z_highres, dNdz_getnorm / norm)
	
	else:
		raise(ValueError, lens_sample+" is not a supported lens sample tag at this time.")
		
	
	
def get_dNdzS(src_sample):
	""" Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated."""
	
	if (src_sample == 'LSST'):
		z_min = 0.
		z_max = 4.
		alpha = 1.24
		z0 = 0.51
		beta = 1.01
		
		z = scipy.linspace(z_min, z_max, 50)

		# dNdz takes form like in Smail et al. 1994
		nofz_ = z**alpha * np.exp( - (z / z0)**beta)
	
		norm = scipy.integrate.simps(nofz_, z)

		return (z, nofz_ / norm)
	elif (src_sample == 'SDSS'):
		z_min = 0.
		z_max = 2.
		alpha 	= 	2.338
		zs		=	0.303
		
		z = scipy.linspace(z_min, z_max, 50)
		
		# dNdz takes form like in Nakajima et al. 2011
		nofz_ = (z / zs)**(alpha-1.) * np.exp( -0.5 * (z / zs)**2)
		norm = scipy.integrate.simps(nofz_, z)

		return (z, nofz_ / norm)
	
	else:
		raise(ValueError, src_sample+" is not a supported src sample tag at this time.")
		return
	
def shape_noise(params, src, lens):
    """ Calculate the shape noise for the given source sample.
    src: source sample. """
	
    # Set up the fiducial cosmology.
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
    if (src=='LSST' and lens=='DESI'):
        sig_gam = 0.26
        n_s_amin = 26.
        chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+0.77))* params['h'] # FIX ME, USING EFFECTIVE REDSHIFT.
        n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        shape_noise = sig_gam**2 / n_s
        return shape_noise
    elif (src=='SDSS' and lens=='LOWZ'):
        sig_gam = 0.21
        n_s_amin = 1.
        chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1. + 0.28))* params['h'] # FIX ME, USING EFFECTIVE REDSHIFT.
        #n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        n_s = n_s_amin *3600. * (180**2) / np.pi**2 / chieff**2
        #print "n_s=", n_s
        #n_s = 8 # Direct from Sukhdeep's paper
		
        shape_noise = sig_gam**2 / n_s
        return shape_noise	
		
    else:
        raise(ValueError, "Shape noise is not yet implemented for source sample ", src, " and lens sample ", lens, ".")
        return 
	
def shot_noise(lens):
	""" Calculate the shot noise associated with the lens sample.
	lens: lens sample. """
	
	if (lens=='DESI'):
		nbar = 3.2 * 10**(-4) 
		shot_noise = 1./nbar
		return shot_noise
	elif (lens=='LOWZ'):
		nbar = 3.*10**(-4)
		shot_noise = 1./ nbar
		return shot_noise
	else:
		raise(ValueError, "Shot noise is not yet implemented for lens sample ", lens, ".")
		return
		
def volume(params, src, lens):
	""" Volume of the lens survey.
	params : dictionary of fiducial parameters
	src : label for the source samples
	lens : label indicating lens survey
	DPi : integral over single factor of window function, Singh et al. 
	2017 eqn A25 """
	
	# Set up the fiducial cosmology.
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
	if (src=='LSST' and lens=='DESI'):
		area_deg = 3000. # degrees squared; area overlap of DESI with LSST
		area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
		# L_W is the top hat window function over the *lens* galaxies
		L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
		vol=area_com*L_W
	elif (src=='SDSS' and lens=='LOWZ'):
		area_deg = 7131.
		area_com = area_deg* (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.28)) * params['h']) **2 # FIX ME, USIGN EFFECTIVE REDSHIFT.
		L_W = 200. # From Singh et al. 2017
		vol=area_com*L_W
	
	return vol
	
def weights(params, src, z_, z_l_):

	""" Returns the inverse variance weights as a function of redshift. """
    
	if (src == 'LSST'):
		e_rms = 0.26
		sig_e = (2. / 15.6) 
	else:
		print "That source sample is not implemented."
		exit()

	weights = get_SigmaC_inv(params, z_, z_l_)**2/(sig_e**2 + e_rms**2)

	return weights	

def weights_times_SigC(params, src, z_, z_l_):

	""" Returns the inverse variance weights as a function of redshift. """
        
	if (src == 'LSST'):
		e_rms = 0.26
		sig_e = (2. / 15.6) 
	else:
		print "That source sample is not implemented."
		exit()

	weights_SigC = get_SigmaC_inv(params, z_, z_l_)/(sig_e**2 + e_rms**2)

	return weights_SigC
    
def get_SigmaC_inv(params, z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
	z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """
    
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
    
	com_s = ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_s_)) * params['h']
	com_l = ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_l_)) * params['h']

	# The dimensions of D_ls depend on the dimensions of z_s_ and z_l_
	if ((hasattr(z_s_, "__len__")==True) and (hasattr(z_l_, "__len__")==True) and (z_s_.size!=1) and (z_l_.size!=1)):
		Sigma_c_inv_list = [[ (4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * com_l[zli] * (com_s[zsi] - com_l[zli]) * (1 + z_l_[zli]) / com_s[zsi]) if ((com_s[zsi] - com_l[zli])>0) else 0. for zsi in range(len(z_s_))] for zli in range(len(z_l_))]
		
		Sigma_c_inv= np.zeros((len(z_s_), len(z_l_)))
		for zli in range(0,len(z_l_)):
			Sigma_c_inv[:, zli] = Sigma_c_inv_list[zli]
			 
	elif hasattr(z_s_, "__len__"): 
		Sigma_c_inv= [4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l * (com_s[zsi] - com_l) * (1 + z_l_) / com_s[zsi] if ((com_s[zsi] - com_l) >0.) else 0. for zsi in range(len(z_s_))]
             
	elif (hasattr(z_l_, "__len__") and (z_l_.size !=1) ): 
		Sigma_c_inv= [4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l[zli] * (com_s - com_l[zli])* (1 + z_l_[zli]) / com_s if ((com_s - com_l[zli]) > 0.) else 0. for zli in range(len(z_l_))]     
	else:
		if (com_s<com_l):
			Sigma_c_inv=0.
		else:
			Sigma_c_inv= 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l * (com_s - com_l)* (1 + z_l_) / com_s
                    
	return Sigma_c_inv
	
	
def get_SigmaC_inv_com(params, com_s, com_l, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
	z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. """

	# The dimensions of D_ls depend on the dimensions of z_s_ and z_l_
	if ((hasattr(com_s, "__len__")==True) and (hasattr(com_l, "__len__")==True) and (com_s.size!=1) and (com_l.size!=1)):
		Sigma_c_inv_list = [[ (4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * com_l[zli] * (com_s[zsi] - com_l[zli]) * (1 + z_l_[zli]) / com_s[zsi]) if (((com_s[zsi] - com_l[zli])>0) and (com_l[zli]>0.)) else 0. for zsi in range(len(com_s))] for zli in range(len(com_l))]
		
		Sigma_c_inv= np.zeros((len(com_s), len(com_l)))
		for zli in range(0,len(com_l)):
			Sigma_c_inv[:, zli] = Sigma_c_inv_list[zli]
			 
	elif hasattr(com_s, "__len__"): 
		Sigma_c_inv= [4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l * (com_s[zsi] - com_l) * (1 + z_l_) / com_s[zsi] if (((com_s[zsi] - com_l) >0.) and (com_l>=0.)) else 0. for zsi in range(len(com_s))]
		
		Sigma_c_inv = np.asarray(Sigma_c_inv)
             
	elif (hasattr(com_l, "__len__") and (com_l.size !=1) ): 
		Sigma_c_inv= [4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l[zli] * (com_s - com_l[zli])* (1 + z_l_[zli]) / com_s if (((com_s - com_l[zli]) > 0.) and(com_l[zli]>=0.)) else 0. for zli in range(len(com_l))]  
		
		Sigma_c_inv = np.asarray(Sigma_c_inv)
		   
	else:
		if ((com_s<com_l) or (com_l<0.)):
			Sigma_c_inv=0.
		else:
			Sigma_c_inv= 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc *   com_l * (com_s - com_l)* (1 + z_l_) / com_s
                    
	return Sigma_c_inv

