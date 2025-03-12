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
import pyccl.nl_pt as pt
import MG_funcs as mg

# Constants / conversions
mperMpc = 3.0856776*10**22
Msun = 1.989*10**30 # in kg
Gnewt = 6.67408*10**(-11)
c=2.99792458*10**(8)
rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  # Msol h^2 / Mpc / pc^2, to yield Upsilon_gg in Msol h / pc^2


def wgg_zmean(params, rp, lens, Pimax, endfilename, nonlin = False, nl_bias = False):
    """ Projects the 3D gg correlation function to get wgg.
    This version uses the mean redshift of the lens sample. 
    params : dictionary of parameters at which to evaluate E_G
    rp: a vector of projected radial distances at which to compute wgg
    lens : label indicating which lens sample we are using
    Pimax : maximum integration for wgg along LOS, Mpc/h
    endfilename : tag for the files produced to keep track of the run.
    nonlin (optional) : use nonlinear halofit correction 
    nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also."""
		
    if nl_bias ==True and nonlin==False:
        print('Upsilon_gm: Cannot have nonlin=False with nl_bias=True; exiting.')
        exit()
	
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
	
    # Get the distribution over the lens redshifts to get the mean z
    #print "getting dNdzl"
    (zl, dNdzl) = specs.get_dNdzL(params, lens)
    meanz = scipy.integrate.simps(dNdzl*zl,zl) / scipy.integrate.simps(dNdzl,zl)
    print('meanz=', meanz)

    # Get the radial comoving distances (in Mpc/h) at the same zl
    chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+meanz)) * params['h']
	
    # Get the power spectrum at each zl
    k = np.logspace(-4, 5, 40000) # units h / Mpc
    
    if nl_bias ==False:
        if nonlin==False:
            Pkgg_ofzl = params['b']**2* ccl.linear_matter_power(cosmo_fid, k * params['h'], 1./ (1. + meanz)) * params['h']**3
        elif nonlin==True:
            Pkgg_ofzl = params['b']**2* ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + meanz)) * params['h']**3
	
        #save_Pkgg=np.column_stack((k, Pkgg_ofzl[0]))
        #np.savetxt('../txtfiles/Pkgg_nonlin_matter='+str(nonlin)+'.dat', save_Pkgg)
    else:
        # Use nonlinear bias via perturbation theory:
	    
        # Set up galaxy and matter tracers
        ptt_g = pt.PTNumberCountsTracer(b1=params['b'], b2=params['b_2'], bs=params['b_s'])
        #ptt_m = pt.PTMatterTracer()
	    
        # Set up the perturbation theory calculator
        ptc = pt.EulerianPTCalculator(with_NC=True, with_IA=True, log10k_min=-4, log10k_max=5, nk_per_decade=20)
        ptc.update_ingredients(cosmo_fid)
	    
        # Now get the power spectrum
        pk_gg = ptc.get_biased_pk2d(ptt_g)
        Pkgg_ofzl =pk_gg(k* params['h'], 1./(1.0 + meanz), cosmo_fid) * params['h']**3
        
        #pk_mm = ptc.get_biased_pk2d(ptt_m, tracer2=ptt_m)
        #Pkmm_ofzl = [pk_mm(k* params['h'], 1./(1.0+zl[zi]), cosmo_fid) * params['h']**3 for zi in range(len(zl))]
        
        #save_gg = np.column_stack((k, Pkgg_ofzl[0]))
        #np.savetxt('../txtfiles/Pkgg_nonlinbias.txt', save_gg)
        
        #save_mm = np.column_stack((k, Pkmm_ofzl[0]))
        #np.savetxt('../txtfiles/Pkmm_PT.txt', save_mm)
        
    r_corr_gg = fft.pk2xi(k, Pkgg_ofzl)
    r = r_corr_gg[0]
    corr_gg = r_corr_gg[1]
	
    # Why aresn't we doing log corr??
    interp_corr = scipy.interpolate.interp1d(np.log(r), corr_gg)

    if (min(rp)<(min(r)/np.sqrt(2))):
        raise(ValueError, "You have asked for wgg at a projected radial value too small for the radial vector you have passed.")
		
    # Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
    Pipos = scipy.logspace(np.log10(0.0001), np.log10(Pimax),5000)
    Pi_rev= list(Pipos)
    Pi_rev.reverse()
    index_cut = next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil))
    Pi_neg = Pi_rev[index_cut:]
    Pi = np.append(-np.asarray(Pi_neg), Pipos)
    print('Pi vecin wgg=', Pi)
	
    corr_2d = [interp_corr(np.log(np.sqrt(rp[rpi]**2 + Pi**2))) for rpi in range(len(rp))]
	
    wgg = [scipy.integrate.simps(corr_2d[rpi], Pi) for rpi in range(len(rp))] 
	
    return wgg

def wgg(params, rp, lens, Pimax, endfilename, nonlin = False, nl_bias = False, MG = False, MGtheory = None):
    """ Projects the 3D gg correlation function to get wgg 
    params : dictionary of parameters at which to evaluate E_G
    rp: a vector of projected radial distances at which to compute wgg
    lens : label indicating which lens sample we are using
    Pimax : maximum integration for wgg along LOS, Mpc/h
    endfilename : tag for the files produced to keep track of the run.
    nonlin (optional) : use nonlinear halofit correction 
    nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """
	
    # Set up the fiducial cosmology.
    #if nonlin==False:
    #    matpow_label = 'linear'
    #else:
    #    matpow_label = 'halofit'
		
    if nl_bias ==True and nonlin==False:
        print('wgg: Cannot have nonlin=False with nl_bias=True; exiting.')
        exit()

    if nl_bias ==True and MG==True:
        print('wgg: We do not have support for nonlinear bias in modified gravity, exiting.')
        exit()


    if (MG==True and MGtheory==None) or (MG==False and MGtheory!=None):
        print('In wgg: MGtheory should be None if and only if MG=False, exiting.')
        exit()
	
    #cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum=matpow_label)
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
	
    # Get the distribution over the lens redshifts and save
    #print "getting dNdzl"
    (zl, dNdzl) = specs.get_dNdzL(params, lens)

    # Get the radial comoving distances (in Mpc/h) at the same zl
    chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
    # Get the power spectrum at each zl
    k = np.logspace(-4, 5, 40000) # units h / Mpc
    
    if nl_bias ==False:
        if nonlin==False and MG==False:
            # GR linear case
            Pkgg_ofzl = [ params['b']**2* ccl.linear_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
        elif nonlin==True and MG==False:
            # GR nonlinear case
            Pkgg_ofzl = [ params['b']**2* ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
        elif MG==True:
            if MGtheory=='fR':
                #Hu-Sawicki f(R) gravity
                if nonlin==False:
                    #f(R) linear case:
                    #Pk_interp = mg.create_interpolator(params)
                    # This takes k in h/Mpc and returns the power spectrum in (Mpc/h)^3
                    Pkgg_ofzl_temp = params['b']**2*mg.P_k_fR_lin(params, k, zl)
                    #print('Pkgg lin shape =', Pkgg_ofzl_temp.shape)
                    
                    Pkgg_ofzl = [Pkgg_ofzl_temp[i, :] for i in range(0,len(zl))]
                    plt.figure()
                    plt.loglog(k, Pkgg_ofzl[0])
                    plt.title('Linear, f(R)')
                    plt.savefig('../plots/linear_fR_Pk.pdf')
                    plt.close()

                    save_Pklin = np.column_stack((k, Pkgg_ofzl[0]))
                    np.savetxt('./Pklin_fR.dat', save_Pklin)

                    #exit()
                elif nonlin==True:
                    #Pkgg_ofzl = [params['b']**2*mg.P_k_NL_fR(params, k, 1./(1+zl[zi])) for zi in range(len(zl))]
                    Pkgg_ofzl_temp = params['b']**2*mg.P_k_NL_fR(params, k, 1./(1+zl))
                    Pkgg_ofzl = [Pkgg_ofzl_temp[i, :] for i in range(0, len(zl))]
                    plt.figure()
                    plt.loglog(k, Pkgg_ofzl[0])
                    plt.title('Non-linear, f(R)')
                    plt.savefig('../plots/nonlinear_fR_Pk.pdf')
                    plt.close()

                    save_Pknl = np.column_stack((k, Pkgg_ofzl[0]))
                    np.savetxt('./Pknl_fR.dat', save_Pknl)

                    #exit()
            if MGtheory == 'nDGP':
                # Normal branch DGP gravity
                if nonlin==False:
                    # Linear case
                    #Pkgg_ofzl_temp = params['b']**2*P_k_nDGP_lin(params, k, 1./(1+zl))
                    Pkgg_ofzl = [params['b']**2*mg.P_k_nDGP_lin(params, k, 1./(1+zl[i])) for i in range(0,len(zl))]

                    save_Pklin = np.column_stack((k, Pkgg_ofzl[0]))
                    np.savetxt('./Pklin_nDGP.dat', save_Pklin)

                    plt.figure()
                    plt.loglog(k, Pkgg_ofzl[0])
                    plt.title('Linear, nDGP')
                    plt.savefig('../plots/linear_nDGP_Pk.pdf')
                    plt.close()

                if nonlin == True:
                    print('in nonlinear nDGP case')
                    # Nonlinear case:
                    Pkgg_ofzl = [params['b']**2*mg.P_k_NL_nDGP(params, k, 1./(1+zl[i])) for i in range(0, len(zl))]
                    #Pkgg_ofzl = [Pkgg_ofzl_temp[i, :] for i in range(0, len(zl))]
                    plt.figure()
                    plt.loglog(k, Pkgg_ofzl[0])
                    plt.title('Non-linear, nDGP')
                    plt.savefig('../plots/nonlinear_nDGP_Pk.pdf')
                    plt.close()

                    save_Pknl = np.column_stack((k, Pkgg_ofzl[0]))
                    np.savetxt('./Pknl_nDGP.dat', save_Pknl)
	
        #save_Pkgg=np.column_stack((k, Pkgg_ofzl[0]))
        #np.savetxt('../txtfiles/Pkgg_nonlin_matter='+str(nonlin)+'.dat', save_Pkgg)
    else:
        # Use nonlinear bias via perturbation theory:
	    
        # Set up galaxy and matter tracers
        ptt_g = pt.PTNumberCountsTracer(b1=params['b'], b2=params['b_2'], bs=params['b_s'])
        #ptt_m = pt.PTMatterTracer()
	    
        # Set up the perturbation theory calculator
        ptc = pt.EulerianPTCalculator(with_NC=True, with_IA=True, log10k_min=-4, log10k_max=5, nk_per_decade=20)
        ptc.update_ingredients(cosmo_fid)
	    
        # Now get the power spectrum
        pk_gg = ptc.get_biased_pk2d(ptt_g)
        Pkgg_ofzl = [pk_gg(k* params['h'], 1./(1.0 + zl[zi]), cosmo_fid) * params['h']**3 for zi in range(len(zl))]
        
        #pk_mm = ptc.get_biased_pk2d(ptt_m, tracer2=ptt_m)
        #Pkmm_ofzl = [pk_mm(k* params['h'], 1./(1.0+zl[zi]), cosmo_fid) * params['h']**3 for zi in range(len(zl))]
        
        #save_gg = np.column_stack((k, Pkgg_ofzl[0]))
        #np.savetxt('../txtfiles/Pkgg_nonlinbias.txt', save_gg)
        
        #save_mm = np.column_stack((k, Pkmm_ofzl[0]))
        #np.savetxt('../txtfiles/Pkmm_PT.txt', save_mm)
        
    r_corr_gg = [fft.pk2xi(k, Pkgg_ofzl[zli]) for zli in range(len(zl))]
    r = r_corr_gg[0][0]
    corr_gg = [r_corr_gg[zli][1] for zli in range(len(zl))]
	
    interp_corr = [scipy.interpolate.interp1d(np.log(r), corr_gg[zi]) for zi in range(len(zl))]

    if (min(rp)<(min(r)/np.sqrt(2))):
        raise(ValueError, "You have asked for wgg at a projected radial value too small for the radial vector you have passed.")
		
    # Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
    Pipos = scipy.logspace(np.log10(0.0001), np.log10(Pimax),300)
    Pi_rev= list(Pipos)
    Pi_rev.reverse()
    index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
    Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
    Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]
    #print('Pi vec [0] in wgg=', Pi[0])
	
    corr_2d = [[interp_corr[zi](np.log(np.sqrt(rp[rpi]**2 + Pi[zi]**2))) for zi in range(len(zl))] for rpi in range(len(rp)) ]
	
    projected = [[scipy.integrate.simps(corr_2d[rpi][zi], Pi[zi]) for zi in range(len(zl))] for rpi in range(len(rp))] 
	
    wgg = [scipy.integrate.simps(projected[rpi] * dNdzl, zl) for rpi in range(len(rp))]
	
    return wgg

def Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin=False, nl_bias = False, MG = False, MGtheory= None):
    """ Takes wgg in Mpc/h and gets Upsilon_gg in Msol h / pc^2 for a given rp0. 
    params : dictionary of parameters at which to evaluate E_G
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
    lens : label indicating which lens sample we are using
    Pimax : maximum integration for wgg along LOS, Mpc/h
    endfilename : tag for the files produced to keep track of the run.
    nonlin (optional) : set to true if we want to use nonlinear halofit corrections.
    nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' 
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """
    
    rp = np.logspace(np.log10(rp_bin_edges[0]), np.log10(rp_bin_edges[-1]), 100)
    #print('Upsgg, rp=', rp)
    w_gg = wgg(params, rp, lens, Pimax, endfilename, nonlin=nonlin, nl_bias = nl_bias, MG = MG, MGtheory = MGtheory)
    # The above is w_gg at the new rp

    rp_finer = np.logspace(np.log10(rp[0]), np.log10(rp[-1]), 5000)
    wgg_interp = scipy.interpolate.interp1d(np.log(rp), np.log(w_gg))
    w_gg_finer = np.exp(wgg_interp(np.log(rp_finer)))
    # Now it is at the finer version of rp which spans the full rp_bin edge to edge
	
    # The next line get the rp_finer point which is closest a given point in the rp vector (not rp_bin_edges).
    index_rp = [next(j[0] for j in enumerate(rp_finer) if j[1]>= rp[rpi]) for rpi in range(len(rp))]  
    #print('index_rp=', index_rp)
    #print('rp finer at index rp=', rp_finer[index_rp])
    #print('rp=', rp)

    index_rpfiner_rp0 = next(j[0] for j in enumerate(rp_finer) if j[1]>= rp0)
    #print('index_rpfiner_rp0=', index_rpfiner_rp0)

    # The index range of the below was previously starting at 1 but I think this is a bit weird.
    first_term = np.zeros(len(rp))
    for rpi in range(0,len(rp)):
        if index_rpfiner_rp0<=index_rp[rpi]:
            first_term[rpi] = ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(w_gg_finer[index_rpfiner_rp0:index_rp[rpi]] * rp_finer[index_rpfiner_rp0:index_rp[rpi]]**2, np.log(rp_finer[index_rpfiner_rp0:index_rp[rpi]]))
        elif index_rpfiner_rp0>=index_rp[rpi]:
            first_term[rpi] = -( 2. / rp[rpi]**2 ) * scipy.integrate.simps(w_gg_finer[index_rp[rpi]:index_rpfiner_rp0] * rp_finer[index_rp[rpi]:index_rpfiner_rp0]**2, np.log(rp_finer[index_rp[rpi]:index_rpfiner_rp0]))
        elif index_rpfiner_rp0==index_rp[rpi]:
            first_term[rpi] = 0.0

 
    #first_term = [ ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(w_gg_finer[index_rpfiner_rp0:index_rp[rpi]] * rp_finer[index_rpfiner_rp0:index_rp[rpi]]**2, np.log(rp_finer[index_rpfiner_rp0:index_rp[rpi]])) for rpi in range(1, len(rp))]

    # not at all convinced that rp0 is the same as rp[0] which is what the last term of the below implies.
 
    # Find the index of rp closest to rp0:
    index_rp0 = next(j[0] for j in enumerate(rp) if j[1]>= rp0)
    #print('index_rp0=', index_rp0)
    #print('rp[index_rp0]=', rp[index_rp0])
    Ups_gg = rho_crit * (np.asarray(first_term) - np.asarray(w_gg) + rp0**2 / np.asarray(rp)**2 * w_gg[index_rp0]) 
    #Ups_gg = rho_crit * (np.asarray(first_term) - np.asarray(w_gg[1:]) + rp0**2 / np.asarray(rp[1:])**2 * w_gg[index_rp0]) 

    #plt.figure()
    #plt.loglog(rp[1:], first_term, label='first term')
    #plt.loglog(rp[1:], w_gg[1:], label='second term')
    #plt.loglog(rp[1:], rp0**2 / np.asarray(rp[1:])**2 * w_gg[index_rp0], label='third term')
    #plt.legend()
    #plt.savefig('../plots/Upggtest_3terms.pdf')
    #plt.close()  

    # test

    #plt.figure()
    #plt.loglog(rp[1:], Ups_gg)
    #plt.savefig('../plots/Upggtest.pdf')
    #plt.close()

    Ups_gg_binned = u.average_in_bins(Ups_gg, rp, rp_bin_edges)
	
    return Ups_gg_binned

# test
	
def Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin=False, nl_bias = False, MG = False, MGtheory = None):
    """ Gets Upsilon_gm in Msol h / pc^2 for a given rp0.
    params : dictionary of parameters at which to evaluate E_G
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
    lens : label indicating which lens sample we are using
    endfilename : tag for the files produced to keep track of the run.
    nonlin(optional) : set to true if we want to use halofit nonlinear correction. 
    nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """
	
    # Set up the fiducial cosmology.
	
    #if nonlin==False:
    #    matpow_label = 'linear'
    #else:
    #    matpow_label = 'halofit'
		
    if nl_bias ==True and nonlin==False:
        print('Upsilon_gm: Cannot have nonlin=False with nl_bias=True; exiting.')
        exit()

    if nl_bias ==True and MG==True:
        print('Upsilon_gm: We do not have support for nonlinear bias in modified gravity, exiting.')
        exit()

    if (MG==True and MGtheory==None) or (MG==False and MGtheory!=None):
        print('Upsilon_gm: MGtheory should be None if and only if MG=False, exiting.')
        exit()
	
    cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
	
    # Get the distribution over the lens redshifts and save
    (zl, dNdzl) = specs.get_dNdzL(params, lens)

    # Get the radial comoving distances (in Mpc/h) at the same zl
    chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
    # Get the power spectrum at each zl
    k = np.logspace(-4, 5, 40000) # units h / Mpc
	
    if nl_bias ==False:
        if nonlin==False and MG==False:
            # GR linear case
            Pkgm_ofzl = [ params['b']*ccl.linear_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
    
        elif nonlin==True and MG==False:
            # GR nonlinear case
            Pkgm_ofzl = [ params['b']*ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
        elif MG==True:
            if MGtheory=='fR':
                #Hu-Sawicki f(R) gravity
                if nonlin==False:
                    #f(R) linear case:
                    # This takes k in h/Mpc and returns the power spectrum in (Mpc/h)^3
                    Pkgm_ofzl_temp = params['b']*mg.P_k_fR_lin(params, k, zl) 
                    Pkgm_ofzl = [Pkgm_ofzl_temp[i,:] for i in range(0,len(zl))]
                elif nonlin==True:
                    Pkgm_ofzl_temp = params['b']*mg.P_k_NL_fR(params, k, 1./(1+zl))
                    Pkgm_ofzl = [Pkgm_ofzl_temp[i,:] for i in range(0,len(zl))]
            if MGtheory == 'nDGP':
                # Normal branch DGP gravity
                if nonlin==False:
                    # Linear case
                    Pkgm_ofzl = [params['b']*mg.P_k_nDGP_lin(params, k, 1./(1+zl[i])) for i in range(0,len(zl))]
                if nonlin == True:
                    # Nonlinear case:
                    print('in nonlinear nDGP case')
                    # Nonlinear case:
                    Pkgm_ofzl = [params['b']*mg.P_k_NL_nDGP(params, k, 1./(1+zl[i])) for i in range(0, len(zl))]
                    #Pkgg_ofzl = [Pkgg_ofzl_temp[i, :] for i in range(0, len(zl))]
                    #plt.figure()
                    #plt.loglog(k, Pkgg_ofzl[0])
                    #plt.title('Non-linear, nDGP')
                    #plt.savefig('../plots/nonlinear_nDGP_Pk.pdf')
                    #plt.close()

                    #save_Pknl = np.column_stack((k, Pkgg_ofzl[0]))
                    #np.savetxt('./Pknl_nDGP.dat', save_Pknl)

    else:
        # Use nonlinear bias via perturbation theory:
	    
        # Set up galaxy and matter tracers
        ptt_g = pt.PTNumberCountsTracer(b1=params['b'], b2=params['b_2'], bs=params['b_s'])
        ptt_m = pt.PTMatterTracer()
	    
        # Set up the perturbation theory calculator
        ptc = pt.EulerianPTCalculator(with_NC=True, log10k_min=-4, log10k_max=5, nk_per_decade=20)
        ptc.update_ingredients(cosmo_fid)
	    
        # Now get the power spectrum
        pk_gm = ptc.get_biased_pk2d(ptt_g, tracer2=ptt_m)
        Pkgm_ofzl = [pk_gm(k * params['h'], 1./(1.+zl[zi]), cosmo_fid) * params['h']**3 for zi in range(len(zl))]
	
    r_corr_gm = [fft.pk2xi(k, Pkgm_ofzl[zli]) for zli in range(len(zl))]
    r = r_corr_gm[0][0]
    corr_gm = [r_corr_gm[zli][1] for zli in range(len(zl))]
	
    # Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)

    # For Upgm the projection length is defined by the extent of the lens sample.
    # We set the extent to be z=0.4-1.0
    chilow = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+0.4)) * params['h']
    chihigh = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+1.0)) * params['h']
    Pi_extent = chihigh - chilow

    Pipos = scipy.logspace(np.log10(0.0001), np.log10(Pi_extent),300)
    Pi_rev= list(Pipos)
    Pi_rev.reverse()
    index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
    Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
    Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]

    # Interpolate the correlation function in 2D (rp & Pi)
    # What is this random choice? This is very weird. #rp = np.logspace(np.log10(rp0), np.log10(105.), 105)  
    rp = np.logspace(np.log10(rp_bin_edges[0]), np.log10(rp_bin_edges[-1]), 100)  
	
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
	
    # Changing this to define Sigma(z directly because CCL doesn't have this functionality anymore
    #Sigma = [ [ ccl.Sig_MG(cosmo_fid, 1. / (1. + z_Pi[zli][pi])) for pi in range(len(Pi[zli]))] for zli in range(len(zl))] 
    # Slight fudge in letting OmegaL0 = 1 - OmegaM0
    Sigma = [ [Sigma_MG(cosmo_fid, params['sigma_0'], 1.0 - params['OmM'], 1. / (1. + z_Pi[zli][pi])) for pi in range(len(Pi[zli]))] for zli in range(len(zl))] 
	
    # THIS IS WRONG - THE (1+ZL) TERM SHOULD BE (1+Z_PI). FIX ME.
    Pi_int = [ [ scipy.integrate.simps( (1. + np.asarray(Sigma[zli])) * np.asarray(zs_int[zli]) / np.asarray(zs_int_w[zli]) * (chil[zli] + Pi[zli]) * (zl[zli] + 1.) * np.asarray(corr_rp_term[rpi][zli]), Pi[zli]) for zli in range(len(zl))] for rpi in range(0, len(rp))]

    # Do the integral over zl 
    zl_int = [ scipy.integrate.simps(dNdzl * Pi_int[rpi], zl) for rpi in range(0,len(rp))]
	
    # Now do the averaging over rp:
    # We need a more well-sampled rp vector for integration
    rp_finer = np.logspace(np.log10(rp[0]), np.log10(rp[-1]), 5000)
    interp_zl_int = scipy.interpolate.interp1d(np.log(rp), np.log(np.asarray(zl_int)))
    zl_int_finer = np.exp(interp_zl_int(np.log(rp_finer)))
    #print('rp_finer=', rp_finer)
	
    # Get the index of the previous rp vector which corresponds to this one:
    index_rp = [next(j[0] for j in enumerate(rp_finer) if j[1]>= rp[rpi]) for rpi in range(len(rp))]
    #print('index_rp=', index_rp)

    # Get the index of the finer vector at rp0
    index_rpfiner_rp0 = next(j[0] for j in enumerate(rp_finer) if j[1]>= rp0)
    #print('index_rpfiner_rp0=', index_rpfiner_rp0)
    #print('rp0 finer=', rp_finer[index_rpfiner_rp0])

    first_term = np.zeros(len(rp))
    for rpi in range(0,len(rp)):
        if index_rpfiner_rp0<=index_rp[rpi]:
            first_term[rpi] = ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(zl_int_finer[index_rpfiner_rp0:index_rp[rpi]] * rp_finer[index_rpfiner_rp0:index_rp[rpi]]**2, np.log(rp_finer[index_rpfiner_rp0:index_rp[rpi]]))
        elif index_rpfiner_rp0>=index_rp[rpi]:
            first_term[rpi] = -( 2. / rp[rpi]**2 ) * scipy.integrate.simps(zl_int_finer[index_rp[rpi]:index_rpfiner_rp0] * rp_finer[index_rp[rpi]:index_rpfiner_rp0]**2, np.log(rp_finer[index_rp[rpi]:index_rpfiner_rp0]))
        elif index_rpfiner_rp0==index_rp[rpi]:
            first_term[rpi] = 0.0

    #first_term = [ ( 2. / rp[rpi]**2 ) * scipy.integrate.simps(zl_int_finer[0:index_rp[rpi]] * rp_finer[0:index_rp[rpi]]**2, np.log(rp_finer[0:index_rp[rpi]])) for rpi in range(1, len(rp))]

    index_rp0 = next(j[0] for j in enumerate(rp) if j[1]>= rp0)
    #print('rp0=', rp0)
    #print('rp[index_rp0]=', rp[index_rp0])

    #Ups_gm = 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * rho_crit * (params['OmM']) * (np.asarray(first_term) - np.asarray(zl_int)[1:] + (rp0 / np.asarray(rp[1:]))**2 * zl_int[index_rp0]) 
    Ups_gm = 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * rho_crit * (params['OmM']) * (np.asarray(first_term) - np.asarray(zl_int) + (rp0 / np.asarray(rp))**2 * zl_int[index_rp0]) 

    Ups_gm_binned = u.average_in_bins(Ups_gm, rp, rp_bin_edges)
	
    return Ups_gm_binned
	
def Delta_gm(params, rp_bin_edges, lens, src, endfilename, nonlin=False):
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
	cosmo_fid = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum=matpow_label)
	
	# Get the distribution over the lens redshifts and save
	(zl, dNdzl) = specs.get_dNdzL(params, lens)

	# Get the radial comoving distances (in Mpc/h) at the same zl
	chil = ccl.background.comoving_radial_distance(cosmo_fid, 1./(1.+zl)) * params['h']
	
	# Get the power spectrum at each zl
	k = np.logspace(-4, 5, 40000) # units h / Mpc
	Pkgm_ofzl = [ params['b']*ccl.nonlin_matter_power(cosmo_fid, k * params['h'], 1./ (1. + zl[zi])) * params['h']**3 for zi in range(len(zl))]
	
	r_corr_gm = [fft.pk2xi(k, Pkgm_ofzl[zli]) for zli in range(len(zl))]
	r = r_corr_gm[0][0]
	corr_gm = [r_corr_gm[zli][1] for zli in range(len(zl))]
	
	# Define Pi (radial comoving separation in Mpc/h), which depends a bit on zl (don't want to go negative z)
	Pipos = scipy.logspace(np.log10(0.0001), np.log10(100),50)
	Pi_rev= list(Pipos)
	Pi_rev.reverse()
	index_cut = [next(j[0] for j in enumerate(Pi_rev) if j[1]<=(chil[zi])) for zi in range(len(zl))]
	Pi_neg = [Pi_rev[index_cut[zi]:] for zi in range(len(zl))]
	Pi = [np.append(-np.asarray(Pi_neg[zi]), Pipos) for zi in range(len(zl))]

	# Interpolate the correlation function in 2D (rp & Pi)
	rp = np.logspace(np.log10(0.01), np.log10(rp_bin_edges[-1]), 100)  
	
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

	Delta_gm = 4. * np.pi * (Gnewt * Msun) * (10**12 / c**2) / mperMpc * rho_crit * params['OmM'] * (np.asarray(first_term) - np.asarray(zl_int)[1:]) 

	Delta_gm_binned = u.average_in_bins(Delta_gm, rp[1:], rp_bin_edges)
	
	return Delta_gm_binned
	
def beta(params, lens, MG = False, MGtheory = None):
    """ Gets beta.
    params : dictionary of parameters at which to evaluate E_G
    lens : label indicating which lens sample we are using
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' 
    """
	
    # Set up the dNdz for the lens galaxies (the galaxies for which we get wgg)
    (zl, dNdzl) = specs.get_dNdzL(params, lens)

    if MG == False:
        # GR case:

        # Set up the cosmology.
        cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
        beta_ofzl = [ccl.growth_rate(cosmo, 1./(1. + zl[zi])) / params['b'] for zi in range(len(zl))]
         
    elif MG==True:
        if MGtheory=='fR':
            beta_ofzl_temp = mg.growthrate_fR(params, 1./(1.+zl)) / params['b']
            beta_ofzl = [beta_ofzl_temp[i] for i in range(0,len(zl))]
        elif MGtheory == 'nDGP':
            beta_ofzl_temp = mg.growthrate_nDGP(params, 1./(1.+zl)) / params['b']
            beta_ofzl = [beta_ofzl_temp[i] for i in range(0,len(zl))]
        else:
            print('beta:That theory of gravity is not implemented. exiting.')
	
    # Integrate over the lens galaxy dNdz
    beta_val = scipy.integrate.simps(beta_ofzl * dNdzl, zl)
	
    return beta_val

def E_G(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin = False, nl_bias = False, MG = False, MGtheory = None):
    """ Returns the value of E_G given it's components.
    params : dictionary of parameters at which to evaluate E_G
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	nonlin (optional): set to true if we want to use halofit nonlinear corrections.
	nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """
	
	# Get beta
    beta_val = beta(params, lens, MG = MG, MGtheory = MGtheory) # beta is definitionally linear so we don't need to pass it nonlin

    print('getting Upgg')
    # Get wgg and Upsilon_gg
    Upgg = Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin = nonlin, nl_bias = nl_bias, MG = MG, MGtheory = MGtheory)
	
    print('getting Upgm')
    # Get Upsilon_gm
    Upgm = Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin = nonlin, nl_bias = nl_bias, MG = MG, MGtheory = MGtheory)
	
    if (len(Upgm)!=len(Upgg)):
        raise(ValueError, "Upsilon_gm and Upsilon_gg must be the same number of rp bins.");
		
    if (hasattr(beta_val, 'len')):
        raise(ValueError, "beta should be a single float.")
		
    Eg = np.asarray(Upgm) / (beta_val * np.asarray(Upgg))
	
    return Eg
	
def E_G_corrected(params, corr_fact, rp_bin_edges, rp0, lens, src, Pimax, endfilename, 
                  MG = False, MGtheory = None):
    """ This function computes a version of E_G which 
    has a correction factor applied to account for galaxy bias.
    The correction factor may or may not be computed at 
    the same parameters as uncorrected E_G.
    params is the parameter set used to compute the 'as measured' EG (simulating measurement of uncorrected value from data.)
    corr_fact is the correlation factor computed at rp_bin_edges, passed in because we won't resample over parameters of this.
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """

    # Do a minimal check that corr_fact is the right length at least:
    if (len(corr_fact)!=len(rp_bin_edges)-1):
        print("In E_G_corrected: The correction factor needs to have the same number of rp bins as rp_bin_edges.")
        exit()
    
    # Get E_G as though measured from data. Because this function is to compute E_G as corrected for nonliner bias,
    # this should always have nonlinearity and nonlinear bias turned on.

    E_G_raw = E_G(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin = True, nl_bias = True, MG = False, MGtheory = None)

    # Multiple by correction factor
    corrected_EG = corr_fact*E_G_raw

    return corrected_EG






    return

def jp_datavector(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, nonlin=False, nl_bias = False, MG=False, MGtheory=None):
    """ Returns the value of E_G given it's components.
    params : dictionary of parameters at which to evaluate E_G
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
    lens : label indicating which lens sample we are using
    Pimax : maximum integration for wgg along LOS, Mpc/h
    endfilename : tag for the files produced to keep track of the run.
    nonlin (optional): set to true if we want to use halofit nonlinear corrections.
    nl_bias (optional): set to true if we want to use nonlinear bias. This will default to perturbation theory nonlin matter also.
    MG is False for GR, true to use a theory other than GR.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """

	
    # Get beta
    beta_val = np.asarray(beta(params, lens, MG=MG, MGtheory = MGtheory))
    # Get wgg and Upsilon_gg
    Upgg = np.asarray(Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin = nonlin, nl_bias = nl_bias, MG = MG, MGtheory = MGtheory))
    # Get Upsilon_gm
    Upgm = np.asarray(Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin = nonlin, nl_bias = nl_bias, MG=MG, MGtheory = MGtheory))
	
    if (len(Upgm)!=len(Upgg)):
        raise(ValueError, "Upsilon_gm and Upsilon_gg must be the same number of rp bins.");
		
    if (hasattr(beta_val, 'len')):
        raise(ValueError, "beta should be a single float.")
		
    data_vec = np.append(Upgm, np.append(Upgg, beta_val))
	
    if(len(data_vec)!= (len(Upgm) + len(Upgg) +1)):
        raise(ValueError, "Something has gone wrong with the length of the data vector.")

    return data_vec
		
def Sigma_MG(cosmo, sigma0, OmL0, a):
    """ Returns Sigma(z) the modified gravity function """
    
    Sig = sigma0 * ccl.omega_x(cosmo, a, 'dark_energy') / OmL0
    
    return Sig


def bias_correction(params, rp_bin_edges, rp0, lens, src, Pimax, endfilename, 
                    MG = False, MGtheory = None):
    """ This function computes the (optional) correction
    factor for nonlinear galaxy bias to be applied to E_G
    to attempt to correct it.
    params is the parameter values at which to compute
    the correction factor (including nonlinear bias values.)
    rp_bin_edges : edges of projected radial bins
    rp0 : scale at which we below which we cut out information for ADSD
	lens : label indicating which lens sample we are using
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
    MG is False for GR, true to use a theory other than GR. Normally we wouldn't use MG for bias correction but leaving in case useful.
    MGtheory gives the label for the theory / parameterisation to be used. If MG=False, MGtheory must be 'None' """

    print('bias correction: getting Upgg')
    # Get wgg and Upsilon_gg
    # Because this is the bias correction function, we always want nonlin=true and nl_bias =true
    Upgg = Upsilon_gg(params, rp_bin_edges, rp0, lens, Pimax, endfilename, nonlin = True, nl_bias = True, MG = MG, MGtheory = MGtheory)
	
    print('bias correction: getting Upgm')
    # Get Upsilon_gm
    Upgm = Upsilon_gm(params, rp_bin_edges, rp0, lens, src, endfilename, nonlin = True, nl_bias = True, MG = MG, MGtheory = MGtheory)
	
    if (len(Upgm)!=len(Upgg)):
        raise(ValueError, "Upsilon_gm and Upsilon_gg must be the same number of rp bins.");
		
		
    correction = params['OmM']*np.asarray(Upgg) / (params['b']*np.asarray(Upgm))    


    return correction

def EG_theory(OmM0, z):

    EG = OmM0/ (OmM0*(1+z)**3 / (OmM0*(1+z)**3 + (1.0-OmM0)))**0.55

    return EG
