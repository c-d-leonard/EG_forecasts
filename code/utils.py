import numpy as np
import scipy.integrate
import pyccl as ccl


def average_in_bins(F_, R_, Rp_):
    """ This function takes a 1D function F_ of projected radius evaluated at projected radial points R_ and outputs the averaged values in bins with edges Rp_
    F_: 1D function of R_
    R_: argument of F_
    Rp_: vector of bin edges in R_""" 
    
    # Interpolate in a more well-sampled vector in R_
    R_f_ = scipy.logspace(np.log10(R_[0]), np.log10(R_[-1]), 10000)
    interp_F = scipy.interpolate.interp1d(np.log(R_), F_)
    F_f_ = interp_F(np.log(R_f_))
    

    indlow= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR])) for iR in range(0, len(Rp_)-1)]
    indhigh= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR+1])) for iR in range(0, len(Rp_)-1)]
		
    F_binned= [2. * scipy.integrate.simps(F_f_[indlow[i]:indhigh[i]]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for i in range(0, len(Rp_) - 1)]
		
    return F_binned
    
def average_in_bins_2D(F_, R_, Rp_):
    """ This function takes a 2D function F_ of projected radius evaluated at projected radial points R_ in both dimensions and outputs the averaged values in bins with edges Rp_
    F_: 1D function of R_
    R_: argument of F_
    Rp_: vector of bin edges in R_""" 
    
    # Interpolate the first dimension in a more well-sampled vector in R_
    R_f_ = scipy.logspace(np.log10(R_[0]), np.log10(R_[-1]), 50000)
    F_f_1 = np.zeros((len(R_), len(R_f_)))
    for ri in range(len(R_)):
        interp_F = scipy.interpolate.interp1d(np.log(R_), F_[ri, :])
        F_f_1[ri, :] = interp_F(np.log(R_f_))
        
    # Get the indices of R_f_ corresponding to the bin edges
    indlow= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR])) for iR in range(0, len(Rp_)-1)]
    indhigh= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR+1])) for iR in range(0, len(Rp_)-1)]
    
    # Integrate in the first dimension	
    F_binned_1 = [[2. * scipy.integrate.simps(F_f_1[rpi, indlow[i]:indhigh[i]]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for rpi in range(len(R_))] for i in range(0, len(Rp_) - 1)] 

    # Interpolate the second dimension 
    F_f_2 = np.zeros((len(R_f_), len(R_f_)))
    for ri in range(len(Rp_)-1):
		interp_F = scipy.interpolate.interp1d(np.log(R_), F_binned_1[ri])
		F_f_2[:, ri] = interp_F(np.log(R_f_))
		
    # Integrate in the first dimension	
    F_binned_2 = [[2. * scipy.integrate.simps(F_f_2[indlow[i]:indhigh[i], rpi]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for i in range(0, len(Rp_) - 1)] for rpi in range(len(Rp_)-1)]
    
    F_binned = np.zeros((len(Rp_)-1, len(Rp_)-1))
    for rpi in range(len(Rp_)-1):
		F_binned[rpi, :] = F_binned_2[rpi]
		
    return F_binned

def rp_bins_mid(rp_edges):
	""" Gets the middle of each projected radius bin."""

	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
	for ri in range(0,len(rp_edges)-1):
		bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	return bin_centers

def z_ofcom_func(params):
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	z_vec = scipy.linspace(0., 10., 4000) # This hardcodes that we don't care about anything over z=2100
	
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	com_vec =  ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_vec)) * params['h']

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com
	
def corr_mat(cov, endfilename):
	""" Takes the covariance matrix and gets the correlation matrix. """
	
	n = (cov.shape)[0] # It will be square
	
	corr = [ [ cov[i, j] / np.sqrt( cov[i,i] * cov[j,j]) for i in range(n)] for j in range(n)]
	
	# transform to array
	corr_arr = np.zeros((n,n))
	for i in range(n):
		corr_arr[i, :] = corr[i]
		
	np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/joint_corrmat_'+endfilename+'.txt', corr_arr, fmt='%1.2f')
	
	return corr_arr
	
