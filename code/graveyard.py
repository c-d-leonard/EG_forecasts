    # Set up the hankel transform we need; we will use this for all  the Delta Sigma covariances.
    #k = np.logspace(np.log10(8.*10**(-4)), np.log10(30.), 5000)
    #HT=ht.hankel_transform(rmin=rp[0],rmax=rp[-1],kmax=k[-1],j_nu=[2],n_zeros=10000,kmin=k[0])
	
    # Get quantities we will need for more than one of the component matrices so we don't have to compute them twice
    #ShapeNoise = sp.shape_noise(params, src, lens)
    #ShotNoise = sp.shot_noise(lens)
    #vol = sp.survey_volume(params, src, lens)
	
    #z, dNdzl = sp.get_dNdzL(params, lens)
    #pk_ofz = [ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + z[zi])) * params['h']**3 for zi in range(len(z))]
    #pk_ofz_arr = np.zeros((len(k), len(z)))
    #for zi in range(len(z)):
    #    pk_ofz_arr[:, zi] = pk_ofz[zi]
        
    #pk = [ scipy.integrate.simps(pk_ofz_arr[ki, :] * dNdzl, z) for ki in range(len(k)) ]
    
    #(DPi1_l, DPi2_l) = DeltaPi_lensing(params, lens, src)
    
        # We will require C^ell_{gg}, C^ell_{gk}, and C^{ell}_{kk} for various 
    # of the covariance so we get them here first from CCL and pass them in
    ell = scipy.logspace(-2, 6, 1000)
    (C_gg, C_gk, C_kk) = get_Cells(params, ell, lens, src)
    
    plt.figure()
    plt.loglog(ell, C_gg, 'm', label='gg')
    plt.hold(True)
    plt.loglog(ell, C_gk, 'g', label='gk')
    plt.hold(True)
    plt.loglog(ell, C_kk, 'b', label='kk')
    plt.legend()
    plt.savefig('../plots/test_Cells.png')
    plt.close()
    
def get_cov_DeltSig_gm(params, rp, HT, k, pk, DPi2_l, ShapeNoise, ShotNoise, vol):
    """ Get the covarariance of Delta Sigma gm at projected radius rp. 
    params : dictionary of cosmological parameters
    rp = vector of projected radial positions.
    HT : Hankel Transform set up structure from Sukdheep's code
    k : k at which the matter power spectrum is calculated
    pk : matter power spectrum integrated over lens z dist
    DPi2_l : integral over the lensing window function squared
    SigCavg: Sigma crit inv averaged over dists and inverted again
    ShapeNoise : shape noise associated with source sample
    ShotNoise : shot noise associated with lens sample
    vol : survey volume in (Mpc/h)**3 """
    
    p_g = np.asarray(pk) * params['b']**2
    p_kappa = np.asarray(pk) * (rho_crit* (params['OmC'] + params['OmB']))**2 * DPi2_l
    
    # Taper and get the ggkk covariance term

    taper_kw=dict({'large_k_lower':10,'large_k_upper':k[-1],'low_k_lower':k[0],'low_k_upper':k[0]*1.2})
    r,cov_ggkk = HT.projected_covariance(k_pk = k, pk1 = p_g + ShotNoise, pk2 = p_kappa + ShapeNoise,j_nu=2, taper=True, **taper_kw)
    
    p_gk=np.asarray(pk) * params['b'] * rho_crit* (params['OmC'] + params['OmB'])

    # Get the gkgk covariance term
    r,cov_gkgk=HT.projected_covariance(k_pk = k, pk1 = p_gk, pk2 = p_gk, kmax=100, j_nu=2)
	
    cov = (cov_ggkk + cov_gkgk) / vol

    return (r, cov)
    
def get_cov_DeltSig_gg(rp):
    """ Get the covarariance of Delta Sigma gg at projected radius rp. 
    rp = vector of projected radial positions. """
	
    cov = np.diag(np.ones(len(rp)))

    return cov
    
def get_cov_DeltSig_gm_gg(rp):
    """ Get the covarariance of Delta Sigma gg at projected radius rp. 
    rp = vector of projected radial positions. """
	
    cov = np.diag(np.ones(len(rp)))

    return cov
    
def W_ofPi_lensing(params, lens, src):
	""" Computes W(Pi) as in Singh et al. 2016 """
	
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmC'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
	
	# The numerator is just Sigma_c averaged over lens and src distributions
	SigCavg = np.sqrt(SigCsq_avg(params, lens, src))
	
	(zl, dNdzl) = sp.get_dNdzL(params,lens)
	chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	(zs, dNdzs) = sp.get_dNdzS(src)
	
	# Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
	Pipos = scipy.logspace(np.log10(0.1), np.log10(4000),4000)
	Pi_rev= list(Pipos)
	Pi_rev.reverse()
	Pi_neg = Pi_rev 
	Pi = np.append(-np.asarray(Pi_neg), Pipos) 

	# Get the redshift at zl + Pi (2D list)
	z_ofChi = u.z_ofcom_func(params)
	z_Pi = [[z_ofChi(chil[zli] + Pi[pi]) if (chil[zli] + Pi[pi]>=0.) else -100. for pi in range(len(Pi))] for zli in range(len(zl))]
	
	com_s = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zs)) * params['h']
	com_zPi = [[chil[zli] + np.asarray(Pi[pi]) for pi in range(len(Pi))] for zli in range(len(zl))]

	Sig_inv_ofPi = [ [ sp.get_SigmaC_inv_com(params, com_s, com_zPi[zli][pi], z_Pi[zli][pi]) for zli in range(len(zl))]  for pi in range(len(Pi)) ] 
	
	Sig_inv = sp.get_SigmaC_inv_com(params, com_s, chil, zl)

	zs_int = [[ scipy.integrate.simps(np.asarray(Sig_inv_ofPi[pi][zli]) * dNdzs, zs) for zli in range(len(zl))] for pi in range(len(Pi))] 
	zl_int = [ scipy.integrate.simps(zs_int[pi] * dNdzl, zl) for pi in range(len(Pi))]
	
	W_ofPi = np.asarray(zl_int) * SigCavg
	
	return (Pi, W_ofPi)
	
def DeltaPi_lensing(params, lens, src):
	""" Computes Delta Pi_1 and Delta Pi 2 as defined in Singh et al. 2016. 
	params: parameters dictionary.
	lens: keyword for lens distribution
	src: keyword for source distribution. """
	
	(Pi, W) = W_ofPi_lensing(params, lens, src)
	
	DeltaPi1 = scipy.integrate.simps(W, Pi)
	
	DeltaPi2 = scipy.integrate.simps(W**2, Pi)
	
	return (DeltaPi1, DeltaPi2)
	
def DeltaSigma_cov_fft(params, rp_bin_edges, lens, src, l, C_gg, C_gk, C_kk):
    """ Gets the covariance matrix of Delta Sigma using FFT methods.
    params :: dictionary of cosmological parametesr.
    rp_bin_edges :: edges of projected radial bins.
    lens :: label for lens sample
    src :: label for source sample 
    l :: vector of ell values
    C_gg :: angular clustering of galaxies
    C_gk :: galaxy galaxy lensing cross spectrum
    C_kk :: cosmic shear cross spectrum """
    
    # Set up a cosmology
    cosmo = ccl.Cosmology(Omega_c = params['OmC'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'])
    
    # Set up an ell vector and get it in log space
    #l = scipy.logspace(-3, 4, 10000)
    nell = len(l)
    dlogl = (np.log10(max(l)) - np.log10(min(l)))/nell
    dlnl = dlogl*np.log(10.0)
    
    # Initialize fourier transform
    (lr, xsave) = fft.fhti(nell, 2., dlnl, 0, 1, 1)
    
    # Get the shape noise term 
    sn_term = shapenoiseterm(lens, src)
    
    # Get the Sigma crit averaged term
    Sigc_sq_avg = SigCsq_avg(params, lens, src)
    
    # Set quantities associated with which src and lens sample we consider
    if (lens=='DESI' and src=='LSST'):
        fsky = 0.073
        z_eff = 0.77
        ns = 26. * 3600.*3282.8 # gals / steradian
        nl = 300. * 3282.8 # gals / steradian
        e_rms = 0.26
    elif (lens=='LOWZ' and src=='SDSS'):
        fsky = 0.173
        z_eff = 0.28
        ns = 1. * 3600.*3282.8
        nl = 8.7 * 3282.8
        e_rms = 0.21
    else:
        raise(ValueError, "That combination of lens and source galaxies is not implemented yet.")
    
    # Get the comoving distance associated with the effective redshift.
    chi_eff = ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_eff)) * params['h']
    print "chi_eff=", chi_eff
    
    # Get the rp vector which corresponds to the ell vector at which we get the fft
    nc = (nell + 1) / 2.0
    loglc = (np.log10(min(l)) + np.log10(max(l)))/2.
    logrc = np.log10(lr) - loglc
    rp = 10**(logrc + (np.arange(1, nell+1) - nc)*dlogl)
    print "rp=", rp
    
    # Set up the thing we are transforming
    
    to_transform = [ scipy.special.jv(2, l / chi_eff * rp[ri]) * (C_gk**2 + C_kk / nl + C_gg * e_rms**2 / ns + C_kk * C_gg) for ri in range(len(rp))]
    #to_transform = [ (C_gk**2 + C_kk / nl + C_gg * e_rms**2 / ns + C_kk * C_gg) for ri in range(len(rp))]
    #to_transform = [ scipy.special.jv(2, rp[ri] * l / chi_eff) * (C_gk**2 + C_kk / nl + C_gg * e_rms**2 / ns + C_kk * C_gg) for ri in range(len(rp))]
    #to_transform = [ np.exp(-0.001* l) for ri in range(len(rp))]
    
    plt.figure()
    plt.loglog(l / chi_eff, scipy.special.jv(2, l / chi_eff * rp[0])*(C_gk**2 + C_kk / nl + C_gg * e_rms**2 / ns + C_kk * C_gg))
    plt.show()
    
    ell_integral = [fft.fht(to_transform[ri], xsave, tdir = -1) for ri in range(len(rp))]
    
    ell_int_arr = np.zeros((len(rp), len(rp)))
    for ri in range(len(rp)):
        ell_int_arr[ri, :] = ell_integral[ri]
    
    #before_avg = Sigc_sq_avg * sn_term * ell_int_arr / (4. * np.pi * fsky)
    before_avg = ell_int_arr 
    
    plt.figure()
    plt.loglog(rp, np.sqrt(np.diag(before_avg)))
    plt.savefig('../plots/before_avg.png')
    
    #avg = utils.average_in_bins_2D(before_avg, rp, rp_bin_edges)
    
    return rp, before_avg #avg
	
def shapenoiseterm(lens, src):
    """ Returns the constant shape noise term to be used with the hankel
    transform in ell space.
    lens: lens sample
    src: source sample  """
	
    if ((lens=='DESI') and (src=='LSST')):
        zeff = 0.77
        ns = 26. * 3600.*3282.8 # gals / steradian
        nl = 300. * 3282.8 # gals / steradian
        e_rms = 0.26
	
	    # Note no Area term here because we are using it with the Hankel transform.
        cov = e_rms**2 / ( nl * ns) 
        
    elif ((lens=='LOWZ') and (src=='SDSS')):
        zeff = 0.28
        ns = 1. * 3600.*3282.8
        nl = 8.7 * 3282.8
        e_rms = 0.21
	
	    # Note no Area term here because we are using it with the Hankel transform.
        cov = e_rms**2 / ( nl * ns)
        
    else:
        raise(ValueError, "That lens / src combination is not yet implemented.")
	
    return cov
