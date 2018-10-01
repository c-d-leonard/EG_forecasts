# Functions for getting the Fisher matrices
import fiducial as fid
import Eg_cov as egcov
import joint_cov as jp
import numpy as np

def get_Fisher_matrices(params_fid, params_fid_var, h_list, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, endfilename, Nsamps):
	""" Returns the Fisher matrix for E_G as the observables. 
	params_fid : dictionary of fiducial cosmological parameters
	params_fid_var: dictionary of only the cosmological parameters which are varied (rather than fixed)
	h_list : dictionary of numerical derivative spacings for each parameter
	rp_bin_edges : edges of projected radial bins
	rp_bin_c : central values of projected radial bins.
	rp0 : scale at which we below which we cut out information for ADSD
	lens : label for the lens distribution (string)
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	Nsamps : optional parameter for how many samples to draw to estimate the Eg covariance matrix."""
	
	# We ultimately need the Fisher matrix to be ordered. Let's put the keys in a list so we do a list comprehension
	keys_list = list(params_fid_var.keys())
	
	print "keys list=", keys_list
	
	print "Getting eg covaraince matrix."
	Dcov_eg = egcov.get_egcov(params_fid, rp_bin_edges, rp_bin_c, rp0, lens, src, Pimax, Nsamps, endfilename)
	val, vec = np.linalg.eig(Dcov_eg)
	Dcov_eg_inv = np.linalg.inv(Dcov_eg)
	
	print "Getting data derivatives, Eg"
	der_data_eg = {label : deriv(params_fid, label, h_list[label], 'Eg',rp_bin_edges, rp0, lens, src, Pimax, endfilename) for label in params_fid_var.keys()}
	
	print "Getting Fisher list Eg"
	Fisher_list_eg = [[ np.dot(der_data_eg[label_a], np.dot(Dcov_eg_inv, der_data_eg[label_b]))  for label_b in keys_list] for label_a in keys_list]
	
	Fisher_eg = np.zeros((len(params_fid_var), len(params_fid_var)))
	for a in range(0,len(params_fid_var)):
		Fisher_eg[a, :] = Fisher_list_eg[a]
		
	Dcov_jp = jp.get_joint_covariance(params_fid, lens, src, rp_bin_edges, rp_bin_c, rp0, endfilename)
	Dcov_jp_inv = np.linalg.pinv(Dcov_jp)
	
	print "Getting data derivatives, jp"
	der_data_jp = {label : deriv(params_fid, label, h_list[label], 'jp_data',rp_bin_edges, rp0, lens, src, Pimax, endfilename) for label in params_fid_var.keys()}
	print "Getting Fisher list jp"
	Fisher_list_jp = [[ np.dot(der_data_jp[label_a], np.dot(Dcov_jp_inv, der_data_jp[label_b])) for label_b in keys_list] for label_a in keys_list]
	
	Fisher_jp = np.zeros((len(params_fid_var), len(params_fid_var)))
	for a in range(0,len(params_fid_var)):
		Fisher_jp[a, :] = Fisher_list_jp[a]
	
	return (Fisher_eg, Fisher_jp, keys_list)
	
def deriv(params_dict, param_label, hder, func_label, rp_bin_edges, rp0, lens, src, Pimax, endfilename, Nsamps = 10**4):
	""" Returns the numerical derivative of the function
	indicated by func_label. 
	params_dict: dictionary of parameters with fiducial values
	param_label: dictionary label of the parameter wrt which we are taking
	a derivative.
	hder: numerical derivative spacing for that parameter
	func_label: indicates the function of which we are taking a derivative
	rp_bin_edges : edges of projected radial bins
	rp0 : scale at which we below which we cut out information for ADSD
	option are: 'Eg', 'Upgm', 'Upgg', 'beta', 'CovEg', 'CovJp'
	lens : string denoting which lens sample we have
	Pimax : maximum integration for wgg along LOS, Mpc/h
	endfilename : tag for the files produced to keep track of the run.
	Nsamps : optional parameter for how many samples to draw to estimate the Eg covariance matrix. """
	
	print "param =", param_label
	
	params_up = params_dict.copy()
	params_dn = params_dict.copy()
	params_up[param_label] = params_dict[param_label]+hder
	params_dn[param_label] = params_dict[param_label]-hder
	
	if func_label == 'Eg':
		derivative = (fid.E_G(params_up, rp_bin_edges, rp0, lens, src, Pimax, endfilename) - fid.E_G(params_dn, rp_bin_edges, rp0, lens, src, Pimax, endfilename)) / (2. * hder)

	elif func_label == 'jp_data':
		derivative = (fid.jp_datavector(params_up, rp_bin_edges, rp0, lens,src, Pimax, endfilename) - fid.jp_datavector(params_dn, rp_bin_edges, rp0, lens, src, Pimax, endfilename)) / (2. * hder)
		
	elif func_label == 'CovEg':
		derivative = (egcov.get_egcov(params_up, rp_bin_edges, rp0, lens, src, Pimax, 100, endfilename) - egcov.get_egcov(params_dn, rp_bin_edges, rp0, lens, src, Pimax, Nsamps, endfilename) ) / (2. * hder)
		
	elif func_label == 'CovJp':
		derivative = (jp.get_joint_covariance(params_up, lens, src, rp_bin_edges, endfilename) - jp.get_joint_covariance(params_dn, lens, src, rp_bin_edges, endfilename)) / (2. * hder)
		
	else:
		raise(RuntimeError, "The function label %s is not supported for numerical differentiaion." % func_label)
	
	return derivative
	
def get_cov_param(Fisher):
	""" Takes the Fisher matrix and inverts it to get the parameter covariance matrix. """
	
	Cov = np.linalg.inv(Fisher)
	
	return Cov
	
def add_priors(Fisher, keys_list_Fisher, priors_file, keys_file_priors):
    """ Add external priors to the Fisher matrix.
    Fisher :: Fisher matrix from this calculation.
    keys_list_Fisher :: ordering of parameters from Fisher matrix
    priors_file :: file from which to load the priors *covariance* matrix 
    keys_file_priors :: file from which to load the ordering of parameters for the priors matrix. """
	
    # Load the priors cov mat and invert to get the priors "Fisher" matrix
    priors_cov = np.loadtxt(priors_file)
    priors_Fish = np.linalg.pinv(priors_cov)
	
    # Load the parameter keys for the priors file
    with open(keys_file_priors) as f:
        keys_list_priors = f.readlines()
    keys_list_priors = [x.strip() for x in keys_list_priors] 
    
    # Make a list which is the keys list for our Fisher matrix,
    # removing parameters on which we don't set priors. We call this 
    # the 'reordered' keys list for the priors Fisher matrix,
    # such that it's in the same order as our Fisher keys list.
    
    keys_list_priors_reordered = [ keys_list_Fisher[i] for i in range(len(keys_list_Fisher)) if (keys_list_Fisher[i] in keys_list_priors) ]
    
    # Now, we reorder the priors Fisher matrix according to this, so that 
    # the priors Fisher matrix has parameters in the same order as our
    # Fisher matrix, just omitting columns and rows for parameters 
    # for which we don't set priors.
    
    priors_Fisher_reorder = np.zeros((len(keys_list_priors), len(keys_list_priors)))
    for i in range(len(keys_list_priors)):
        for j in range(len(keys_list_priors)):
            priors_Fisher_reorder[i,j] = priors_Fish[keys_list_priors.index(keys_list_priors_reordered[i]), keys_list_priors.index(keys_list_priors_reordered[j])]
            
    # Now get a list of the indices of the parameters which are in keys_list_Fisher
    # but not in keys_list_priors - the parameters for which we do not set priors. 
    no_priors_index = [keys_list_Fisher.index(elem) for elem in keys_list_Fisher if elem not in keys_list_priors]
	
    # Use this list of indices to iteratively add columns and rows of zeros 
    # to the reordered priors Fisher matrix where we impose no prior.
	
    for i in range(len(no_priors_index)):
        add_col = np.zeros(len(priors_Fisher_reorder[:,0]))
        priors_Fisher_reorder = np.insert(priors_Fisher_reorder, no_priors_index[i], add_col, axis=1)
        add_row = np.zeros(len(add_col)+1)
        priors_Fisher_reorder = np.insert(priors_Fisher_reorder, no_priors_index[i], add_row, axis=0)
		
    # priors_Fisher_reorder should now be appropriately padded with zeros
	
    # Now just add the priors Fisher matrix to our Fisher matrix 
    Fish_with_priors = priors_Fisher_reorder + Fisher
            
    return Fish_with_priors
	
	
	
	
	
	
	
