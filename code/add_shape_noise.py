# This script contains functionality for adding shape noise to each of the jacknife samples of Upsilon_gm from Shadab's mocks
# This will enable us to use these samples directly in computing a jacknife covariance.

import numpy as np
import specs as sp
import joint_cov as jc
import matplotlib.pyplot as plt
from scipy.linalg import issymmetric
import utils as u

def cov_SN_only(rp_edg, lens, src, params, rp0):
    """ rp_edg = the edges of the projeted radius bins
    lens = name for lens sample
    src = source sample
    params = dictionary of parameters. 
    We are going to construct the covariance matrix in units of Msol^2 h^2 / pc^4
    but convert to (Mpc/h)^3 before returning for compatibility with simulations. """

    if (lens=='DESI'):
        nbar = 5.0 * 10**(-4) # number per (Mpc/h)^3
    else:
        print("In Cov_SN_only: we don't have that lens sample set up, exiting")
        exit()

    # Get the 'shape noise' aka sig_gam^2 / ns. This returns in units of (Mpc/h)^2
    sig_by_ns = sp.shape_noise(params, src, lens)
    #print('sig_by_ns=', sig_by_ns)

    rp_bin_centres= u.rp_bins_mid(rp_edg)

    # Get the volume V
    V = sp.volume(params, src, lens)

    print('nbar*V=', nbar*V)

    #print('nbar*V=', nbar*V)

    # Get SigC^{-2}^-1
    SigC2 = jc.SigCsq_avg(params, lens, src)
    #print('SigC2=', SigC2)

    # Get the constant that multiplies the whole covariance matrix
    const = SigC2 * sig_by_ns / (2*np.pi*V*nbar) # Units: Msol^2 Mpc^2 / pc^4

    # Figure out which bin rp0 is in:
    ind = next(j[0] for j in enumerate(rp_edg) if j[1]>=rp0)
    rp0_low = ind-1 # lower bin edge
    rp0_high = ind # upper bin edge
    #print('rp0_low=', rp0_low)
    #print('rp0_high=', rp0_high)

    Cov_SN = np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    Cov_SN_DS = np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    term1 =  np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    term2 =  np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    term3 =  np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    term4 =  np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    DS_only_term4 = np.zeros((len(rp_edg)-1, len(rp_edg)-1))
    for rpi in range(0,len(rp_edg)-1):
        for rpj in range(0,len(rp_edg)-1):
            
            if rpi==rpj:
                term1[rpi, rpj] = 2. / (rp_edg[rpi+1]**2 - rp_edg[rpi]**2) # units: (h/Mpc)^2
            else:
                term1[rpi, rpj] = 0.

            if rpi==rp0_low: # If bin i is the bin containing rp0
                term2[rpi, rpj] =  - 4.* rp0**2 / (rp_edg[rpi+1]**2 - rp_edg[rpi]**2)/ (rp_edg[rp0_high]**2 - rp_edg[rp0_low]**2)*np.log(rp_edg[rpi+1]/rp_edg[rpi])
                
            else:
                term2[rpi, rpj] = 0.

            if rpj==rp0_low: # If bin j is the bin containing rp0
                term3[rpi, rpj] = - 4.* rp0**2 / (rp_edg[rpj+1]**2 - rp_edg[rpj]**2)/ (rp_edg[rp0_high]**2 - rp_edg[rp0_low]**2)*np.log(rp_edg[rpj+1]/rp_edg[rpj])
                
            else:
                term3[rpi, rpj] = 0.

            term4[rpi, rpj] = (8*rp0**4 / ((rp_edg[rpi+1]**2 - rp_edg[rpi]**2)*(rp_edg[rpj+1]**2 - rp_edg[rpj]**2)*(rp_edg[rp0_high]**2 - rp_edg[rp0_low]**2))*np.log(rp_edg[rpi+1]/rp_edg[rpi])*np.log(rp_edg[rpj+1]/rp_edg[rpj]))
            #print('(rp_edg[rpi+1]**2 - rp_edg[rpi]**2) = ', (rp_edg[rpi+1]**2 - rp_edg[rpi]**2))
            #print('(rp_edg[rpj+1]**2 - rp_edg[rpj]**2)=', (rp_edg[rpj+1]**2 - rp_edg[rpj]**2))
            #print('np.log(rp_edg[rpi+1]/rp_edg[rpi]) =', np.log(rp_edg[rpi+1]/rp_edg[rpi]))
            #print('np.log(rp_edg[rpj+1]/rp_edg[rpj])=', np.log(rp_edg[rpj+1]/rp_edg[rpj]))
            #print('rpi centre=', rp_bin_centres[rpi])
            #print('rpi centre 1/r^2=', 1./ (rp_bin_centres[rpi])**2)
            #print('rpi centre 1/r^2 avg = ', 2 / (rp_edg[rpi+1]**2 - rp_edg[rpi]**2) * np.log(rp_edg[rpi+1]/rp_edg[rpi]))
            # Clearer to write term4 in terms of the rp0 element of term 1:
            # ACTUALLY cannot do this here because for the bins below rp0 this hasn't been filled in yet. 
            #term4[rpi, rpj] = term1[rp0_low, rp0_low]*(4*rp0**4 / ((rp_edg[rpi+1]**2 - rp_edg[rpi]**2)*(rp_edg[rpj+1]**2 - rp_edg[rpj]**2))*np.log(rp_edg[rpi+1]/rp_edg[rpi])*np.log(rp_edg[rpj+1]/rp_edg[rpj]))
            
            DS_only_term4 = 2.0 * const / (rp_edg[rp0_high]**2 - rp_edg[rp0_low]**2)

            Cov_SN[rpi, rpj] = const*(term1[rpi, rpj] + term2[rpi, rpj] + term3[rpi, rpj] + term4[rpi, rpj]) # units Msol^2 h^2 / pc^4
            Cov_SN_DS[rpi, rpj] = const*term1[rpi,rpj]

    # Check which components of the matrix are positive semi-definite. 
    """eig1, eiv1 = np.linalg.eig(term1)
    print('eig term 1=', eig1)

    eig2, eiv2 = np.linalg.eig(-term2)
    print('eig term 2=', eig2)

    eig3, eiv3 = np.linalg.eig(-term3)
    print('eig term 3=', eig3)

    eig4, eiv4 = np.linalg.eig(-term4)
    print('eig term 4=', eig4)"""

    # Check if the Delta Sigma only part of term 4 is the same or very close to the rp0 x rp0 bin value of term 1 (it should be.)
    #print('Delta Sigma only, term 4 = ', DS_only_term4)
    #print('rp0 element of term 1=', Cov_SN_DS[rp0_low, rp0_low])

    #print('term4=', term4)
    #print('is term4 symmetric?=', issymmetric(term4))
    #for rpi in range(0,len(rp_edg)-1):
    #    for rpj in range(0,len(rp_edg)-1):
    #        if term4[rpi,rpj] != term4[rpj,rpi]:
    #            #print('not symmetric at rpi=', rpi, 'rpj=', rpj)

    #np.savetxt('../txtfiles/term4.dat', term4)


    #print('equiv=', np.diag(term1)*SigC2 * sig_by_ns / (2*np.pi*V*nbar))

    # Covert units to (Mpc/h)^2
    mperMpc = 3.0856776*10**22
    Msun = 1.989*10**30 # in kg
    Gnewt = 6.67408*10**(-11) # kg^{-1}m^3 s^{-2}
    c=2.99792458*10**(8) # m / s
    rho_crit = 3. * 10**10 * mperMpc / (8. * np.pi * Gnewt * Msun) / 10**12  
    # rho crit in Msol h^2 / Mpc / pc^2
    rho_m = params['OmM'] * rho_crit

    Cov_SN_simunits = Cov_SN / (rho_m)**2 # units (Mpc/h)^2
    Cov_SN_DS_simunits = Cov_SN_DS / (rho_m)**2 # units (Mpc/h)^2

    #plt.figure()
    #plt.imshow(np.log10(np.abs(const*term1/rho_m**2)))
    #plt.colorbar()
    #plt.savefig('./term1_sn_cov.pdf')
    #plt.close()

    #plt.figure()
    #plt.imshow(np.log10(np.abs(const*term2/rho_m**2)))
    #plt.colorbar()
    #plt.savefig('./term2_sn_cov.pdf')
    #plt.close()

    #plt.figure()
    #plt.imshow(np.log10(np.abs(const*term3/rho_m**2)))
    #plt.colorbar()
    #plt.savefig('./term3_sn_cov.pdf')
    #plt.close()

    """plt.figure()
    plt.imshow(np.log10(const*term4/rho_m**2))
    plt.colorbar()
    plt.savefig('./term4_sn_cov.pdf')
    plt.close()"""

    return Cov_SN_simunits 