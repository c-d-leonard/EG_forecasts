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
	
	
	
