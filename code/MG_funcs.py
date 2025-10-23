# Functions for MG power spectrum calculations
# Some of these are authored by Carola

from emantis import FofrBoost
import sys
import pyccl as ccl
import numpy as np
from scipy.integrate import odeint
import scipy

# nDGP NL P(k) emulator
from nDGPemu import BoostPredictor

#MODULE_PATH = "/home/ncl117/Software/MGCAMB/camb/__init__.py"
MODULE_PATH = "/home/campus.ncl.ac.uk/ncl117/Software/MGCAMB/camb/__init__.py"
MODULE_NAME = "MGCAMB"
import importlib
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)
from MGCAMB import camb


def create_interpolator(cosmo_values):
    """ author: Carola Zanoletti; modified by DL"""

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_values['h'] * 100, 
                       ombh2=cosmo_values['OmB'] * cosmo_values['h']**2, 
                       omch2=cosmo_values['OmM'] * cosmo_values['h']**2, 
                       omk=0, mnu=0.0)
    


    pars.InitPower.set_params(ns=cosmo_values['n_s'], As=cosmo_values['A_s'])
    pars.set_mgparams(MG_flag=3, GRtrans=0.0, QSA_flag=4, F_R0=cosmo_values['fR0'], FRn=cosmo_values['fR_n'])
    pars.NonLinear = camb.model.NonLinear_none
    
    # hubble_units=False and k_hunit=False I think means k units = 1/Mpc, P(k) units = Mpc^3
    # So the below should take k in h/Mpc and output Pk in (Mpc/h)^3
    PK = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=True, k_hunit=False, zmax=100, extrap_kmax=10**5)
    return PK

def P_k_fR_lin(params, k, z):
    """ author: Carola Zanoletti; modified by DL 
    accepts k in units of *1/MPc* 
    outputs linear power spectrum in units of (Mpc/h)^3"""

    # Convert k to 1/Mpc
    k_h = k*params['h']
    
    PK = create_interpolator(params)

    return_Pk = PK.P(z, k_h)

    #print('lin, k=', len(k))
    #print('lin, z=', len(z))
    #print('shape Pk lin=', return_Pk.shape)

    return return_Pk

def P_k_NL_fR(params, k, a):
    """ author: Carola Zanoletti; modified by DL
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL (GR parameters only)
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity with n=1, units (Mpc)^3
    """
    #H0rc, fR0, n, mu, Sigma = MGparams
    
    # The eMANTIS emulator uses the LCDM value for sigma8:
    # `by 𝜎8 we refer to the value obtained assuming a linear ΛCDM evolution, even for 𝑓(𝑅)CDM
    # cosmologies. The parameter 𝜎8 is used as an indirect normalisation of the primordial power spectrum,
    # which is therefore the same in both ΛCDM and 𝑓(𝑅)CDM for a given set of cosmological parameters.

    ''' Initialise EMANTIS emulator'''
    emu_fR = FofrBoost()

    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s =params['A_s'], 
                          n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
	
    sigma8_VAL_lcdm = ccl.sigma8(cosmo)

    # The emulator only works at certain relatively high k: 0.0288 <= k <= 9.7299
    # Below this, we need to stich on the linear P(k)
    index_cut = next(j[0] for j in enumerate(k) if j[1]>=(0.0288))
    index_max = next(j[0] for j in enumerate(k) if j[1]>=(9.7299))
    k_lin = k[0:index_cut]
    k_nl = k[index_cut:index_max]
    #print('k_lin=', k_lin)
    #print('k_nl=', k_nl)
    k_toohigh = k[index_max:]

    Pk_lin = P_k_fR_lin(params, k_lin, 1./a - 1.) # This takes k in h/Mpc and outputs Pk_lin in (Mpc/h)^3
    #print('pk lin shape=', Pk_lin.shape)
    
    #print('OmegaM=', params['OmM'])
    #print('sig8=', sigma8_VAL_lcdm)
    #print('fr0=', -np.log10(params['fR0']))
    #print('a=', a)
    pkratio_fR = emu_fR.predict_boost(params['OmM'], sigma8_VAL_lcdm, -np.log10(params['fR0']), a, k_nl) #This takes k in h/Mpc

    #print('len a=', len(a))

    # Reshape:
    pk_ratio_fR_reshape = np.zeros((len(a),len(k_nl)))
    for i in range(0,len(a)):
        for j in range(0, len(k_nl)):
            pk_ratio_fR_reshape[i,j] = pkratio_fR[i][0][j]


    print('pk_ratio_reshape shape=', pk_ratio_fR_reshape.shape)
    #print('len a=', len(a))
    #print('len knl=', len(k_nl))
    #print('pk ratio shape=', pkratio_fR.shape)
    #print('pk ratio=', pkratio_fR)
    #exit()
    # k is put in with units [h/Mpc]
    #exit()

    if hasattr(a, 'len'):
        Pk_ccl = np.zeros((len(a),len(k_nl)))
        for i in range(0,len(a)):
            Pk_ccl[i,:] = ccl.power.nonlin_power(cosmo, k_nl*params['h'], a=a[i]) * params['h']**3 # units (Mpc/ h)^3
    else:
        Pk_ccl = ccl.power.nonlin_power(cosmo, k_nl*params['h'], a=a)*params['h']**3 # units (Mpc/h)^3
    Pk_nl = pk_ratio_fR_reshape*Pk_ccl

    print('Pk_nl shape=', Pk_nl.shape)

    # Try to use the extrapolated linear above a certain point:
    Pk_extrap = P_k_fR_lin(params, k_toohigh, 1./a - 1.)
    
    ratio = Pk_nl[:,-1] / Pk_extrap[:,0]
    print('shape of ratio=', len(ratio))

    Pk_extrap_rescale = np.zeros((len(a),len(k_toohigh)))
    for i in range(0,len(a)):
        Pk_extrap_rescale[i,:] = ratio[i]*Pk_extrap[i,:]

    #Pk_return = np.append(np.append(Pk_lin, Pk_nl, axis=1), np.zeros((len(a),len(k_toohigh))), axis=1)
    Pk_return = np.append(np.append(Pk_lin, Pk_nl, axis=1), Pk_extrap_rescale, axis=1)

    print('shape Pk_return=', Pk_return.shape)

    return Pk_return

""" Functions to get the growth rate in f(R)"""

# Linear matter power f(R) (function for mu(k,a))
def mu_fR(fR0, cosmo, k, a):
    # k is in units 1/Mpc
    # We want H0 in units 1/Mpc, so H0 = 100h/c
    if fR0 == 0:
        return np.ones(len(k))
    else:
        # from ReACT paper
        f0 = fR0 / (cosmo["h"]*100/3e5)**2
        Zi = (cosmo["Omega_m"] + 4*a**3*(1-cosmo["Omega_m"]))/a**3
        Pi = (k/a)**2 + Zi**3/2/f0/(3*cosmo["Omega_m"] - 4)**2
        return 1 + (k/a)**2/3/Pi
        
def sigma_8_fR(cosmo, MGparams, a_array):
    k_val = np.logspace(-4, 3, 3000)
    sigma_8_vals = []

    for a in a_array:
        P_k_vals = P_k_fR_lin(cosmo, MGparams, k_val, a)
        j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
        integrand = k_val**2 * P_k_vals * j1_vals**2
        integral_val = scipy.integrate.trapz(integrand, x=k_val)
        sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
        sigma_8_vals.append(sigma_8_val)
    
    return np.array(sigma_8_vals)
    
    ## Note: only works if we assume mu is approximately independent of k in f(R) !!
def solverGrowth_fR(y,a,cosmo, MGparams):
    E_val = E(cosmo, a)
    D , a3EdDda = y
    H0rc, fR0, n, mu, Sigma = MGparams
    
    mu = mu_fR(fR0, cosmo, 0.1, a)
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
    return ydot
    
def fsigma8_fR(cosmoMCMCStep, MGparams, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu, Sigma])
    
    output P_k_musigma (array) -> linear matter power spectrum for f(R), units (Mpc)^3
    """
    
    H0rc, fR0, n, mu, Sigma = MGparams
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_fR, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_fR_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
    
    f_fR = np.interp(a, a_solver, f_fR_interp)

    k_val = np.logspace(-4,3,3000)
    return f_fR * sigma_8_fR(cosmoMCMCStep, MGparams, a)

def growthrate_fR(params, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu, Sigma])
    
    output growth rate f in f(R) gravity
    """

    cosmoMCMCStep = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], 
                          n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
    
    MGparams = [params['H0rc'], params['fR0'], params['fR_n'], params['mu_0'], params['sigma_0']]
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_fR, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_fR_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
    
    f_fR = np.interp(a, a_solver, f_fR_interp)

    return f_fR 



"""Functions to get linear matter power spectra nDGP"""

# dimensionless hubble parameter in GR
def E(cosmoMCMCStep, a):
    Omg_r = cosmoMCMCStep["Omega_g"]*(1+ 3.044*7/8 * (4/11)**(7/8))
    return np.sqrt(cosmoMCMCStep["Omega_m"]/a**3 +Omg_r/a**4 + (1 - cosmoMCMCStep["Omega_m"] - Omg_r))

# deriv. of E wrt scale factor, GR
def dEda(cosmo, a):
    Omg_r = cosmo["Omega_g"]*(1+ 3.044*7/8 * (4/11)**(7/8))
    E_val = E(cosmo, a)
    
    return (-3*cosmo["Omega_m"]/a**4 -4*Omg_r/a**5)/2/E_val

# mu(k,a) = mu(a) in nDGP (modified gravity poisson eq. parameter)
def mu_nDGP(MGparams, cosmo, a):
    H0rc, fR0, n, mu, Sigma = MGparams
    if H0rc == 0: # just for convention, we want MGParams = [0,0,0,0] to be gr
        return 1
    elif 1/(4*H0rc**2) == 0:
        return 1
    else:
        Omg_rc = 1/(4*H0rc**2)
        E_val = E(cosmo, a)
        # from ReACT paper
        beta = 1 + E_val/np.sqrt(Omg_rc) * (1+ a*dEda(cosmo, a)/3/E_val)
        return 1 + 1/3/beta
    
def solverGrowth_nDGP(y,a,cosmo, MGparams):
    E_val = E(cosmo, a)
    D , a3EdDda = y
    
    mu = mu_nDGP(MGparams, cosmo, a)
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
    return ydot
    
def P_k_nDGP_lin(params, k, a):
    """
    input k (array) -> wavevector, units h/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n,mu])
    
    output Pk_nDGP (array) -> linear matter power spectrum for nDGP gravity, units (Mpc/h)^3
    """

    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], 
                          n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
    
    MGparams = [params['H0rc'], params['fR0'], params['fR_n'], params['mu_0'], params['sigma_0']]
    
    Omega_rc = 1/(4*params['H0rc']**2)
    
    # Get growth factor in nDGP
    a_solver = np.linspace(1/50.,1,100)
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmo, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmo,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    
    # Get growth factor in GR
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmo, a_solver[0])*a_solver[0]**3)], a_solver,\
                  args=(cosmo,[0,0,0,0,0]), mxstep=int(1e4))
    
    Delta_GR = Soln.T[0]

    # Get Pk linear in GR
    # Adjust units because CCL uses Mpc units instead of Mpc/h
    Pk_GR = ccl.linear_matter_power(cosmo, k=k*params['h'], a=a)*(params['h'])**3

    # find the index for matter domination)
    idx_mdom = 0
    # get normalization at matter domination
    Delta_nDGP_49 = Delta[idx_mdom]
    Delta_GR_49 = Delta_GR[idx_mdom]
    return Pk_GR * np.interp(a, a_solver, (Delta / Delta_nDGP_49) **2 / (Delta_GR / Delta_GR_49)**2)  # units (Mpc)^3

def growthrate_nDGP(params, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output fsigma8 (array) -> fsigma_8 for nDGP
    """

    cosmoMCMCStep = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], 
                          n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
    
    MGparams = [params['H0rc'], params['fR0'], params['fR_n'], params['mu_0'], params['sigma_0']]
    
    #H0rc, fR0, n, mu, Sigma = MGparams
    
    # Get growth factor in nDGP
    Omega_rc = 1/(4*params['H0rc']**2)
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_nDGP_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
    
    f_nDGP = np.interp(a, a_solver, f_nDGP_interp)

    return f_nDGP 


# NL matter power spectra in nDGP
def P_k_NL_nDGP(params, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL (GR parameters only)
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_nDGP (array) -> Nonlinear matter power spectrum for nDGP gravity, units (Mpc)^3
    """

    cosmo= ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], 
                          n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], transfer_function='boltzmann_camb')
    
    MGparams = [params['H0rc'], params['fR0'], params['fR_n'], params['mu_0'], params['sigma_0']]

    # Turn k into units of h/Mpc
    #k = k/cosmo["h"]

    H0rc, fR0, n, mu, Sigma = MGparams

    # nDGP emulator - set parameters
    cosmo_params = {'Om':params['OmM'],
                    'ns':params['n_s'],
                    'As':params["A_s"],
                    'h':params['h'],
                    'Ob':params['OmB']}
    
    # The emulator only works at certain relatively high k: [0.0156606, 4.99465] h/ Mpc
    # Below this, we need to stich on the linear P(k)
    index_cut = next(j[0] for j in enumerate(k) if j[1]>=(0.0156606))
    index_max = next(j[0] for j in enumerate(k) if j[1]>=(4.99465))
    k_lin = k[0:index_cut]
    k_nl = k[index_cut:index_max]
    #print('k_lin=', k_lin)
    #print('k_nl=', k_nl)
    k_toohigh = k[index_max:]

    #print('getting linear')
    # Get the linear power spectrum at k_lin:
    Pk_lin = P_k_nDGP_lin(params, k_lin, a) # k in h/Mpc, Pk_lin in Mpc/h
    #print('got linear')

    #print('getting nonlinear')
    
    model_nDGP = BoostPredictor()
    
    z_in = 1./a -1 
    #print('z=', 1./a -1)
    #print('H0rc=', H0rc)
    #print('cosm params=', cosmo_params)
    #print('k=',k)

    #print('getting boost now')
    # Now get the nonlinear power spectrum in the interpolation range.
    # nDGP emulator - get boost
    # k is taken in h / Mpc
    pkratio_nDGP = model_nDGP.predict(H0rc, z_in , cosmo_params, k_out=k_nl)
    #print('done getting boost')

    # Get GR power spectrum
    Pk_ccl = ccl.power.nonlin_power(cosmo, k_nl*params["h"], a=a) * params['h']**3 # units (Mpc/h)^3

    Pk_return_nl = pkratio_nDGP*Pk_ccl # units (Mpc/h)**3
    #print('got nonlinear')

    # Now get the high-k values. Assume nonlinear behaves like linear at high k but shifted for continuity
    # Try to use the extrapolated linear above a certain point:
    #print('getting extrapolated')
    Pk_extrap = P_k_nDGP_lin(params, k_toohigh, a)
    #print('shape Pk reutnr nl=', Pk_return_nl.shape)
    #print('shape Pk extrap=', Pk_extrap.shape)
    
    ratio = Pk_return_nl[-1] / Pk_extrap[0]
    #print('shape of ratio=', len(ratio))

    #Pk_extrap_rescale = np.zeros((len(a),len(k_toohigh)))
    #for i in range(0,len(a)):
    #    Pk_extrap_rescale[i,:] = ratio[i]*Pk_extrap[i,:]

    Pk_extrap_rescale = ratio*Pk_extrap

    #Pk_return = np.append(np.append(Pk_lin, Pk_nl, axis=1), np.zeros((len(a),len(k_toohigh))), axis=1)
    Pk_return = np.append(np.append(Pk_lin, Pk_return_nl), Pk_extrap_rescale)

    #print('shape Pkreturn=', Pk_return.shape)

    return Pk_return

