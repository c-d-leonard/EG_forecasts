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
	#if (lens_sample=='DESI' or lens_sample=='DESI_plus_20pc' or lens_sample=='DESI_plus_50pc' or lens_sample=='DESI_4MOST_LRGs' or lens_sample=='DESI150pc_4MOST_LRGs' or lens_sample=='DESI200pc_4MOST_LRGs' or lens_sample=='DESI_4MOST_18000deg2_LRGs'):
		dNdz_file = '../txtfiles/DESI_LRGs_dNdz_2024.dat'
		
		z, dNdz = np.loadtxt(dNdz_file, unpack=True)
		
		interpolation = scipy.interpolate.interp1d(z, dNdz)
	
		# Create a well-sampled redshift vector to make sure we can get the normalization without numerical problems
		z_highres = np.linspace(z[0], z[-1], 50)
	
		dNdz_getnorm = interpolation(z_highres)
	
		norm = scipy.integrate.simps(dNdz_getnorm, z_highres)
	
		return  (z_highres, dNdz_getnorm / norm)
		
	elif (lens_sample=='DESI_4MOST_ELGs' or lens_sample=='DESI150pc_4MOST_ELGs' or lens_sample=='DESI200pc_4MOST_ELGs' or lens_sample=='DESI_4MOST_18000deg2_ELGs'):
		dNdz_file = 'DESI_ELG_redshifts_2col.txt'
		
		z, dNdz = np.loadtxt(dNdz_file, unpack=True)
		
		interpolation = scipy.interpolate.interp1d(z, dNdz)
	
		# Create a well-sampled redshift vector to make sure we can get the normalization without numerical problems
		z_highres = np.linspace(z[0], z[-1], 50)
	
		dNdz_getnorm = interpolation(z_highres)
	
		norm = scipy.integrate.simps(dNdz_getnorm, z_highres)
	
		return  (z_highres, dNdz_getnorm / norm)	
		
	elif (lens_sample=='LOWZ'):
		
		cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'])
		
		
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
    """ Returns the dNdz of the sources as a function of photometric redshift, as well as the z points at which it is evaluated.
    This version convolves with an assumed Gaussian p(z_t, z_p) model with mean shift of zero and variance given by the SRD. """
	
    if (src_sample == 'LSST'):
        z_min = 0.
        z_max = 5.0
		
        # LSST Year 1
        z0 = 0.13
        alpha = 0.78
        sigz=0.05
        deltaz=0.
		
        z = scipy.linspace(z_min, z_max, 100)

        # dNdz takes form like in Smail et al. 1994
        #nofz_ = z**alpha * np.exp( - (z / z0)**beta)
        nofz_ = z**2 * np.exp(- (z/z0)**alpha)
	
        norm = scipy.integrate.simps(nofz_, z)
		
        # Convolve with photo-z uncertainty
        z_p, dNdz_p = dNdz_perturbed(z, nofz_/norm, sigz, deltaz)

        return (z_p, dNdz_p)
	
    elif (src_sample == 'LSSTY10'):
        z_min = 0.
        z_max = 5.0
		
        # LSST Year 10
        z0 = 0.11
        alpha = 0.68
        sigz=0.05
        deltaz=0.
		
        z = scipy.linspace(z_min, z_max, 100)

        # dNdz takes form like in Smail et al. 1994
        #nofz_ = z**alpha * np.exp( - (z / z0)**beta)
        nofz_ = z**2 * np.exp(- (z/z0)**alpha)
	
        norm = scipy.integrate.simps(nofz_, z)
		
        # Convolve with photo-z uncertainty
        z_p, dNdz_p = dNdz_perturbed(z, nofz_/norm, sigz, deltaz)

        return (z_p, dNdz_p)

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

	
def dNdz_perturbed(z, dNdz, sigma, deltaz):
    """ Convolves the source dNdz with a Gaussian of scatter sigma and mean redshift bias deltaz"""

    
    if (np.abs(sigma)>10**(-12)):
        
        # Define a new redshift vector exactly the same as z just to facilitate the convolution
        z_new = z
    
        Gauss = np.zeros((len(z_new), len(z)))
        for zmi in range(0,len(z)):
            Gauss[:,zmi] = scipy.stats.multivariate_normal.pdf(z_new, mean = z[zmi]+deltaz, cov=sigma)
            
        numerator = np.zeros(len(z_new))
        for zni in range(0, len(z_new)):
            numerator[zni] = scipy.integrate.simps(dNdz * Gauss[zni,:], z)
    
        denominator = scipy.integrate.simps(numerator, z_new)
    
        dNdz_new = numerator / denominator
        
    else:
        # In the case where we only have a mean shift, we don't need to do the full integral and it's faster so just do that.
        # Shift all the redshifts 
        z_new_temp = z + deltaz 
        
        # If deltaz is positive, we now have a z that starts above 0. If deltaz is big, this starts to cause a problem for F calculations.
        # Pad out the dndz below this:
        if deltaz>0:
            
            # Check if z vec is linearly spaced, hopefully yes:
            if np.abs((z_new_temp[1]-z_new_temp[0])-(z_new_temp[2]-z_new_temp[1]))>0.000001:
                print(z_new_temp[1]-z_new_temp[0])
                print(z_new_temp[2]-z_new_temp[1])
                print('z is not linearly spaced, not set up for that')
                exit() 
            
            # Get number of z's we want:
            numz_pad = np.int(z_new_temp[0] / (z_new_temp[1]-z_new_temp[0]))
            
            padding_z = np.linspace(0,z_new_temp[0], numz_pad)
            
            z_new_temp = np.append(padding_z, z_new_temp)
            dNdz = np.append(np.zeros(len(padding_z)), dNdz)
        
        if any(z_new_temp<0):
            # Make sure that if this makes the redshifts negative we cut those.
            ind = next(j[0] for j in enumerate(z_new_temp) if j[1]>=0)
            z_new = z_new_temp[ind:]
            dNdz_new_temp = dNdz[ind:]
        else:
            # Nothing to do here but just rename so we have uniformity outside this loop
            z_new = z_new_temp
            dNdz_new_temp = dNdz
          
        # Normalise
        
        norm = scipy.integrate.trapz(dNdz_new_temp, z_new)
        
        dNdz_new = dNdz_new_temp / norm
   
    return z_new, dNdz_new

	
def shape_noise(params, src, lens):
    """ Calculate the shape noise for the given source sample.
    The units are (number of galaxies / (Mpc/h)^2) - this does not include the factor of area here.
    src: source sample. """
	
    # Set up the fiducial cosmology.
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'])
	
    if (src=='LSST' and lens=='DESI'):
        sig_gam = 0.26
        #print('n_s is made very high for debugging!!')
        #n_s_amin = 1000.
        n_s_amin = 10. # LSST Y1 from SRD
        chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+0.72))* params['h'] # z=0.72 is mean redshift of DESI LRG lenses
        # 46656000 is 4 * 180**2 * 3600. 4 pi * (180/pi)**2 is the number of degrees is a sphere.
        n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        shape_noise = sig_gam**2 / n_s
        return shape_noise
        #if (src=='LSST' and (lens=='DESI' or lens=='DESI_plus_20pc' or lens=='DESI_plus_50pc'or lens=='DESI_4MOST_LRGs' or lens=='DESI150pc_4MOST_LRGs' or lens=='DESI200pc_4MOST_LRGs' or lens=='DESI_4MOST_18000deg2_LRGs')):
        #sig_gam = 0.26
        #n_s_amin = 26.
        #chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+0.77))* params['h'] # FIX ME, USING EFFECTIVE REDSHIFT OF LENS SAMPLE. this doesn't make that much sense.
        # 4665600 is 4 * 180**2 * 3600. 4 pi * (180/pi)**2 is the number of degrees is a sphere.
        #n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        #shape_noise = sig_gam**2 / n_s
        #return shape_noise
    elif (src=='LSSTY10' and lens=='DESI'):
        sig_gam = 0.26
        #print('n_s is made very high for debugging!!')
        #n_s_amin = 1000.
        n_s_amin = 27. # LSST Y10 from SRD
        chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+0.72))* params['h'] # z=0.72 is mean redshift of DESI LRG lenses
        # 46656000 is 4 * 180**2 * 3600. 4 pi * (180/pi)**2 is the number of degrees is a sphere.
        n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        shape_noise = sig_gam**2 / n_s
        return shape_noise
        #if (src=='LSST' and (lens=='DESI' or lens=='DESI_plus_20pc' or lens=='DESI_plus_50pc'or lens=='DESI_4MOST_LRGs' or lens=='DESI150pc_4MOST_LRGs' or lens=='DESI200pc_4MOST_LRGs' or lens=='DESI_4MOST_18000deg2_LRGs')):
        #sig_gam = 0.26
        #n_s_amin = 26.
        #chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+0.77))* params['h'] # FIX ME, USING EFFECTIVE REDSHIFT OF LENS SAMPLE. this doesn't make that much sense.
        # 4665600 is 4 * 180**2 * 3600. 4 pi * (180/pi)**2 is the number of degrees is a sphere.
        #n_s = n_s_amin * (466560000. / np.pi) / (4 * np.pi * chieff**2)
        #shape_noise = sig_gam**2 / n_s
        #return shape_noise
    elif (src=='LSST' and (lens=='DESI_4MOST_ELGs' or lens=='DESI150pc_4MOST_ELGs' or lens=='DESI200pc_4MOST_ELGs' or lens=='DESI_4MOST_18000deg2_ELGs')):
        sig_gam = 0.26
        n_s_amin = 26.
        chieff = ccl.comoving_radial_distance(cosmo_fid, 1./ (1.+1.0))* params['h'] # FIX ME, USING EFFECTIVE REDSHIFT.
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
	#if (lens=='DESI' or lens=='DESI_plus_20pc' or lens=='DESI_plus_50pc' or lens=='DESI_4MOST_LRGs' or lens=='DESI150pc_4MOST_LRGs' or lens=='DESI200pc_4MOST_LRGs' or lens=='DESI_4MOST_18000deg2_LRGs'):
		nbar = 5.0 * 10**(-4) # Units (h/MPc)^3 # Zhou++2023
		#print('nbar is arbitrarily changed for debugging!!')
		#nbar = 10**(-3) # Units (h/MPc)^3
		shot_noise = 1./nbar
		return shot_noise
	elif (lens=='DESI_4MOST_ELGs' or 'DESI150pc_4MOST_ELGs' or 'DESI200pc_4MOST_ELGs' or 'DESI_4MOST_18000deg2_ELGs'):
		nbar = 5.0 * 10**(-4) # Units (h/MPc)^3
		shot_noise = 1./nbar
		return shot_noise
	elif (lens=='LOWZ'):
		nbar = 3.*10**(-4) # Units (h/MPc)^3
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
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'])
	
    if ((src=='LSST' or src == 'LSSTY10') and lens=='DESI'):
        area_deg = 5000. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.72))*  params['h'] )**2 # Using effective redshift of DESI LRGs
        # L_W is the top hat window function over the *lens* galaxies 
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.4))* params['h'] ))
        vol=area_com*L_W
    elif (src=='SDSS' and lens=='LOWZ'):
        area_deg = 7131.
        area_com = area_deg* (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.28)) * params['h']) **2 # FIX ME, USIGN EFFECTIVE REDSHIFT.
        L_W = 200. # From Singh et al. 2017
        vol=area_com*L_W
    elif (src=='LSST' and lens=='DESI_plus_20pc'):
        area_deg = 3600. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W	
    elif (src=='LSST' and lens=='DESI_plus_50pc'):
        area_deg = 4500. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W		
    elif (src=='LSST' and lens=='DESI_4MOST_ELGs'):
        area_deg = 4000. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.5))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W	
    elif (src=='LSST' and lens=='DESI_4MOST_LRGs'):
        area_deg = 10500. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W    
    elif (src=='LSST' and lens=='DESI150pc_4MOST_LRGs'):
        area_deg = 12000. 
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W 
    elif (src=='LSST' and lens=='DESI200pc_4MOST_LRGs'):
        area_deg = 13500. 
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W 
    elif (src=='LSST' and lens=='DESI_4MOST_18000deg2_LRGs'):
        area_deg = 18000. 
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.77))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W        
    elif (src=='LSST' and lens=='DESI150pc_4MOST_ELGs'):
        area_deg = 5500. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.5))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W
    elif (src=='LSST' and lens=='DESI200pc_4MOST_ELGs'):
        area_deg = 7000. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.5))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W
    elif(src=='LSST' and lens=='DESI_4MOST_18000deg2_ELGs'):
        area_deg = 18000. # degrees squared; area overlap of DESI with LSST
        area_com = area_deg * (np.pi / 180.)**2 * (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.0))*  params['h'] )**2 # FIX ME, USIGN EFFECTIVE REDSHIFT. 
        # L_W is the top hat window function over the *lens* galaxies
        L_W=  ((ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +1.5))* params['h'] ) - (ccl.comoving_radial_distance(cosmo_fid, 1./ (1. +0.6))* params['h'] ))
        vol=area_com*L_W                                   		
	
    return vol


def weights(params, src, z_, z_l_):

    """ Returns the inverse variance weights as a function of redshift. """
    
    if (src == 'LSST' or src=='LSSTY10'):
        e_rms = 0.26
        sig_e = (2. / 15.6) 
    elif (src == 'SDSS'):
        e_rms = 0.21
        sig_e = 15.
    else:
        print("That source sample is not implemented.")
        exit()

    weights = get_SigmaC_inv(params, z_, z_l_)**2/(sig_e**2 + e_rms**2)

    return weights	

def weights_times_SigC(params, src, z_, z_l_):

    """ Returns the inverse variance weights as a function of redshift. """
        
    if (src == 'LSST' or src=='LSSTY10'):
        e_rms = 0.26
        sig_e = (2. / 15.6) 
    elif (src == 'SDSS'):
        e_rms = 0.21
        sig_e = 15.
    else:
        print("That source sample is not implemented.")
        exit()

    weights_SigC = get_SigmaC_inv(params, z_, z_l_)/(sig_e**2 + e_rms**2)

    return weights_SigC
    
def get_SigmaC_inv(params, z_s_, z_l_):
	""" Returns the theoretical value of 1/Sigma_c, (Sigma_c = the critcial surface mass density).
	z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. 
	Returns in units of pc^2 / (Msol h)"""
    
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'])
    
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
	z_s_ and z_l_ can be 1d arrays, so the returned value will in general be a 2d array. 
	Units of pc^2 / (Msol h)"""

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


def pz_pdf(zp, zs, z_shift, sigz):
    """" Returns the Gaussian PDF p(zs, zp) describing the probability of a source with zp having true zs
    zp = vector of photo-z values at which to return p(zs,zp)
    zs = vector of spec-z / 'true'-z values at which to return p(zs,zp)
    z_shift = shift to the mean of the Gaussian away from zs=zp
    sigz = variance parameter of the Gaussian where variance = sigz^2(1+z_p)^2"""

    pzpdf = np.zeros((len(zp), len(zs)))
    for zpi in range(0,len(zp)):
          for zsj in range(0,len(zs)):
                pzpdf_unnormed = np.exp(-(zp-(zs+z_shift)) / (2*sigz**2*(1+zp)**2))
                # need to figure out how to normalise this and return the normalised quantity
                print('in pz_pdf: this is not normalised, not returning anything yet. Do not use.') 
    

    return