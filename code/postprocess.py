import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import utils as u

def single_ell_data(pars_plot, fish, keys_list, params_var, params_fix, params_Eg_insens=None):
    """ Produces the information needed to plot a single ellipse.
    Returns this to pass to a plotting function which can 
    plot multiple ellipses together. 
    pars_plot :: list of length 2 of the labels of the parameters to be plotted.
    fish :: Fisher matrix 
    keys_list :: ordered list of parameter labels corresponding to order in Fisher matrix.
    params_var :: dictionary of parameter labels and fiducial values to be varied
    params_fix :: dictionary of parameter labels and fiducial values to be fixed
    params_Eg_insense :: dictionary of parameter labels and fiducial values to which Eg is by design insensitive (b and sigma8)
    (optional, only present if Eg case rather than joint probes case)
    """
	
    # Cut the columns and rows of the Fisher matrices for the parameters which are fixed.
    to_fix = [i for i in range(0,len(keys_list)) if (keys_list[i] in params_fix.keys())]
    fish_fix = np.delete(np.delete(fish, to_fix, axis=0), to_fix, axis=1)
    keys_list_fix = np.delete(keys_list, to_fix)
	
    # before inverting, cut from the Fisher matrix any remaining parameters to which 
    #  we are totally insensitive.
    if (params_Eg_insens != None):
        insens = [i for i in range(0, len(keys_list_fix)) if (keys_list_fix[i] in params_Eg_insens.keys())]
        fish_fix = np.delete(np.delete(fish_fix, insens_Eg, axis=0), insens_Eg, axis=1)
        keys_list_fix = np.delete(keys_list_fix, insens_Eg)
	
    # Invert to get the parameter covariance matrices:
    cov = np.linalg.pinv(fish_fix)
	
    # Get the indices of the parameters we want to plot and cut all other rows and columns
    indpar1 = next(j[0] for j in enumerate(keys_list_fix) if j[1]==pars_plot[0]) 
    indpar2 = next(j[0] for j in enumerate(keys_list_fix) if j[1]==pars_plot[1])
	
    to_cut = [i for i in range(0,len(keys_list_fix)) if ((i!=indpar1) and (i!=indpar2))]
	
    cov_cut = np.delete(np.delete(cov, to_cut, axis=0), to_cut, axis=1)
    keys_list_final = np.delete(keys_list_fix, to_cut)
	
    # Get eigenvalues and eigenvectors
    Evals, Evecs=np.linalg.eig(cov_cut)
	
    # Get the angle by which to rotate
    rotate=np.arctan2(Evecs[1,0], Evecs[0,0])*180./np.pi
	
    return Evals, Evecs, rotate
    

def ellipse_plots(pars_plot, Evals, Evecs, rotate, label, color, linestyle, endfilename):
	""" Produces ellipse plots.
	pars_plot :: list of central values of parameters to be plotted
	Evals :: list of pairs of evals for two parameters.
	Evecs :: list of pairs of evecs.
	rotate :: list of angles by which to rotate the ellipses
	label :: list of labels for the ellipses
	color :: list of colours
	linestyle :: list of linestyles
	endfilename :: suffix on output plot """
	
	#The values by which we multiply to get the 1 and 2 sigma contours in the 2 parameter case
	onesigval=2.30
	twosigval=6.17
	
	ellipse = plt.figure()
	plt.rc('font', family='serif', size=14)
	ell=ellipse.add_subplot(111, aspect='equal')
	
	# Add each ellipse to the plot
	for i in range(len(rotate)):
		center = [pars_plot[0], pars_plot[1]]
		ell_add=matplotlib.patches.Ellipse(center, 2*np.sqrt(onesigval*Evals[i][0]), 2*np.sqrt(onesigval*Evals[i][1]), rotate[i], edgecolor=color[i], facecolor='none', label=label[i], linestyle=linestyle[i], linewidth=2)
		ell.add_patch(ell_add)
		
	ell.set_xlim(-0.6,0.6)
	ell.set_ylim(-0.6,0.6)
	ell.set_xlabel('$\Sigma_0$')
	ell.set_ylabel('$\mu_0$')
	ell.tick_params(axis='both', which='major', labelsize=12)
	ell.tick_params(axis='both', which='minor', labelsize=12)
	ell.legend(fontsize=10, loc='upper left')
	plt.title('Forecast constraints: LSST + DESI', fontsize=16)
	plt.tight_layout()
	plt.savefig('/home/danielle/Dropbox/CMU/Research/EG_comparison/plots/ellipse_'+endfilename+'.png')
	plt.close()

	"""#Get the lengthes of the two ellipse axes for the one sigma and two sigma ellipse
	firstaxis_1sig_eg=np.sqrt(onesigval*Evals_eg[0])
	firstaxis_2sig_eg=np.sqrt(twosigval*Evals_eg[0])
	secondaxis_1sig_eg=np.sqrt(onesigval*Evals_eg[1])
	secondaxis_2sig_eg=np.sqrt(twosigval*Evals_eg[1])

	firstaxis_1sig_jp=np.sqrt(onesigval*Evals_jp[0])
	firstaxis_2sig_jp=np.sqrt(twosigval*Evals_jp[0])
	secondaxis_1sig_jp=np.sqrt(onesigval*Evals_jp[1])
	secondaxis_2sig_jp=np.sqrt(twosigval*Evals_jp[1])

	#Get the angle by which the ellipse is rotated in the non-eigenvector basis.
	# The choices of how this angle is calculated forces the first parameter keys_list
	# To be on the horizontal axis in the plot.
	rotate_eg=np.arctan2(Evecs_eg[1,0], Evecs_eg[0,0])*180./np.pi
	rotate_jp=np.arctan2(Evecs_jp[1,0], Evecs_jp[0,0])*180./np.pi

	# Make a tuple of the central plotting parameter values for plotting
	center_eg = [params_var[keyslist_final_eg[0]],params_var[keyslist_final_eg[1]]]
	center_jp = [params_var[keyslist_final_jp[0]],params_var[keyslist_final_jp[1]]]
	
	ellipse = plt.figure()
	plt.rc('font', family='serif', size=14)
	ell=ellipse.add_subplot(111, aspect='equal')
	#Which axis we pick for "width" and "height" reflects the fact that we have chosen to plot mu0 on the y axis
	ell_plot_eg_1sig=matplotlib.patches.Ellipse(center_eg, 2*firstaxis_1sig_eg, 2*secondaxis_1sig_eg, rotate_eg, edgecolor='#FF9966', facecolor='none', label='$E_G$')
	ell_plot_eg_2sig=matplotlib.patches.Ellipse(center_eg, 2*firstaxis_2sig_eg, 2*secondaxis_2sig_eg, rotate_eg, edgecolor='#FF9966', facecolor='none')
	ell_plot_jp_1sig=matplotlib.patches.Ellipse(center_jp, 2*firstaxis_1sig_jp, 2*secondaxis_1sig_jp, rotate_jp, edgecolor='m', facecolor = 'none', label='Joint probes:  $\Upsilon_{gm}$, $\Upsilon_{gg}$, $beta$')
	ell_plot_jp_2sig=matplotlib.patches.Ellipse(center_jp, 2*firstaxis_2sig_jp, 2*secondaxis_2sig_jp, rotate_jp, edgecolor='m', facecolor = 'none')
	ell_plot_eg_1sig.set_label('$E_G$')
	ell.add_patch(ell_plot_eg_1sig)
	ell.add_patch(ell_plot_eg_2sig)
	ell.add_patch(ell_plot_jp_1sig)
	ell.add_patch(ell_plot_jp_2sig)
	ell.set_xlim(-0.4,0.4)
	ell.set_ylim(-0.4,0.4)
	#ell.set_label('$E_G$')
	ell.set_xlabel('$\Sigma_0$')
	ell.set_ylabel('$\mu_0$')
	#ell.set_xlabel(keyslist_final_jp[0])
	#ell.set_ylabel(keyslist_final_jp[1])
	ell.tick_params(axis='both', which='major', labelsize=12)
	ell.tick_params(axis='both', which='minor', labelsize=12)
	ell.legend(fontsize=10, loc='lower right')
	plt.title('Forecast constraints: LSST + DESI', fontsize=16)
	plt.tight_layout()
	plt.savefig('/home/danielle/Dropbox/CMU/Research/EG_comparison/plots/ellipse_'+endfilename+'.jpeg')"""
	
	return
	
def plot_covariance(cov, lens, src, endfilename):
	""" Make a 2D colour plot of the covariance matrix of joint probes
	cov :: covariance matrix of Upsgm, Upsgg, beta in bins
	lens : label for lens galaxies
	src : label for source galaxies
	endfilename : endfilename """
	
	#corr = u.corr_mat(cov, endfilename)
	
	plt.figure(figsize=(10, 10))
	plt.rcParams["font.family"] = "serif"
	plt.imshow(cov, aspect=1, interpolation='None')
	#plt.clim(0,2.52)
	plt.subplots_adjust(top=0.88)
	#plt.suptitle("$\Upsilon_{gm}$, $\Upsilon_{gg}$, $\\beta$ covariance", fontsize='35')
	plt.tight_layout()
	plt.savefig('../plots/corr_jp_'+lens+'_'+src+'_'+endfilename+'.png')
	plt.close()
	
	return
