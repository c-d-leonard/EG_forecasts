#Author: Shadab Alam, May 2019
#This compute the covarainnce and the correlation matrix
from __future__ import print_function
import numpy as np
import pylab as pl
import sys
import matplotlib.pyplot as plt
import add_shape_noise as sn
import utils as u
import specs as sp

# look at the covariance between upsilon and beta
def evaluate_correlation_matrix(stat_list=['gm','beta'],rmin=2.0,plots=0,data_dir='', add_shape_noise=False, lens=None, src=None):
    '''This computes and plots the correlation matrix
    a list of stats can be provided for correlation matrix evluation
    gg: galaxy-galaxy upsilon
    gm: galaxy matter upsilon
    beta, bias and f can also be given '''

    add_shape_noise = bool(add_shape_noise)

    col_dic={'f':1,'bias':2,'beta':3}
    latex_lab={'f':r'$f$','bias':r'$b$','beta':r'$\beta$'}
    #load the f bias bets        
    rsdfile=data_dir+'HOD-model-5-PB00-z0.75-wsFOG.txt'
    rsd_data=np.loadtxt(rsdfile)

    #stat_jn will hold all the measurements
    stat_jn=np.array([])
    xlab=[]

    print('add shape noise=', add_shape_noise)
    
    for pp,par in enumerate(stat_list):
        #load gm if needed
        if(par in ['gm','gg']):
            print('par=', par)
            if(par=='gm'):
                print('add shapenoise =', add_shape_noise)
                if add_shape_noise:
                    gmfile = data_dir + 'ups_gm_with_SN_'+lens+'_'+src+'.dat'
                else:
                    gmfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                print('gmfile=', gmfile)
                ups=np.loadtxt(gmfile)
            elif(par=='gg'):
                ggfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-sel-All-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                ups=np.loadtxt(ggfile)
        

            #remove small scale upsilon not needed
            #ind=ups[:,0]>rmin
            # DL: actually, keep one more bin than the first one that doesn't contain rp0 because you might miss rp0 otherwise.
            print('rp=', ups[:,0])
            ind = next(j[0] for j in enumerate(ups[:,0]) if j[1]>rmin)
            print('ind min=', ind)
            rp_ups=ups[ind:,0]
            xlab.append(rp_ups)
            ups_jn=ups[ind:,4:]
            if(stat_jn.size==0):
                stat_jn=np.copy(ups_jn)
            else:
                stat_jn=np.row_stack([stat_jn,ups_jn])

        else:
            xlab.append(par)
            if(stat_jn.size==0):
                stat_jn=rsd_data[:,col_dic[par]]
            else:
                stat_jn=np.row_stack([stat_jn,rsd_data[:,col_dic[par]]])


    #cov=np.cov(stat_jn)
    # Need to get the covariance from the Jackknife - specific method, not
    # from a sample covariance method.
    N = len(stat_jn[0,:])

    means = np.zeros(len(stat_jn[:,0]))
    for i in range(len(stat_jn[:,0])):
        means[i] = sum(stat_jn[i,:]) / N

    #plt.figure()
    #plt.loglog(rp_ups, means[0:15])
    #plt.loglog(rp_ups, means[15:30])
    #plt.show()
	
    #print("beta=", means[30])
    #print("shape_means=", len(means))
		
    cov = np.zeros((len(stat_jn[:,0]), len(stat_jn[:,0])))
    for i in range(len(stat_jn[:,0])):
        for j in range(len(stat_jn[:,0])):
            cov[i, j] = (N-1.)/N * sum((stat_jn[i,:] - means[i]) * (stat_jn[j,:] - means[j]))
    

    corr=np.copy(cov)
    for ii in range(0,cov.shape[0]):
        for jj in range(0,cov.shape[1]):
            corr[ii,jj]=cov[ii,jj]/np.sqrt(cov[ii,ii]*cov[jj,jj])

    if(plots==1):
        nrow=1;ncol=1
        fig1,axarr=pl.subplots(nrow,ncol,sharex=False,sharey=False,figsize=(10,nrow*7))
        #axarr=axarr.reshape(axarr.size)
        axarr=[axarr]


        pl.sca(axarr[0])
        pl.pcolor(corr,cmap='seismic',vmin=-1,vmax=1)
        pl.colorbar()
        
        #get the axis label
        nskip=3
        xpos=-0.5; 
        xtick_all=[];xtlab_all=[]
        ytick_all=[];ytlab_all=[]
        ncorr=corr.shape[0]
        for pp,par in enumerate(stat_list):
            if(par in ['gg','gm']):
                ytick_all.append(xpos+xlab[pp].size*0.5)
                ytlab_all.append(r'$\Upsilon_{%s}$'%par)
                for vv,val in enumerate(xlab[pp]):
                    xpos=xpos+1
                    if(vv%nskip==0):
                        xtick_all.append(xpos)
                        xtlab_all.append('%4.1f'%xlab[pp][vv])
                
            else:
                xpos=xpos+1
                xtick_all.append(xpos)
                xtlab_all.append('%s'%latex_lab[xlab[pp]])
                ytick_all.append(xpos)
                ytlab_all.append('%s'%latex_lab[xlab[pp]])
            
            
            #draw line for seperation
            pl.plot([xpos+0.5,xpos+0.5],[0,ncorr],'k--',lw=2)
            pl.plot([0,ncorr],[xpos+0.5,xpos+0.5],'k--',lw=2)

        #print(xtick_all,xtlab_all)
               
        pl.xticks(xtick_all, xtlab_all)#, rotation='vertical')
        pl.yticks(ytick_all, ytlab_all)


    return {'corr':corr,'cov':cov}

# Add a function that creates a version of the jacknife samples that also includes a draw from an independent shape noise matrix.

def Ups_gm_samples_with_shapenoise(sim_vol, rp0, params, lens, src, data_dir):
    """ author: Danielle
    data_dir = where we are keeping the jacknife samples from simulation.
    rp0 = the rp0 value for constructing upsilon
    rp = the centres of the projected radial bins at which to compute Ups_gm
    sim_vol is the volume of the simulation
    lens is the lens sample, src is the src sample
    params = parameters
    """

    # Load the jacknife samples and process
    gmfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
    ups=np.loadtxt(gmfile) # This has cols rp, ups mean, ups std, 'All', and the jacknife samples...

    rp = ups[:,0]

    rp_edg = u.rp_bin_edges_log(rp)

    ind = next(j[0] for j in enumerate(rp_edg) if j[1]>rp0)
    rp_cut = rp[ind:]
    rp_edg_cut = rp_edg[ind:]

    ups_jn=ups[ind:,4:]
    
    #print('rp=', rp)

    # Get the bin edges for the rp bin centres:
    #rp_edg = u.rp_bin_edges_log(rp)
    #print('rp_edg=', rp_edg)

    Nsamps = len(ups_jn[0,:])
    #print('Nsamps=', Nsamps)

    # Now call the function to get Cov_SN:
    Cov_SN_raw = sn.cov_SN_only(rp_edg_cut, lens, src, params, rp0)
    print('shape Cov =', Cov_SN_raw.shape)
    eig, eiv = np.linalg.eig(Cov_SN_raw)
    print('eig=', eig)

    # Rescale this to bring this covariance (which uses the LSST_DESI volume) in line with the simulation volume:

    vol_LSST_DESI = sp.volume(params, src, lens)

    #print('sim_vol / vol_LSST_DESI=', sim_vol / vol_LSST_DESI)

    Cov_SN =  vol_LSST_DESI / sim_vol * Cov_SN_raw

    np.savetxt(data_dir+'/Cov_shapenoise_simsvol_MpchUnits_'+lens+'_'+src+'.dat', Cov_SN)

    Corr_SN = np.zeros_like(Cov_SN)
    for i in range(0,len(rp_cut)):
        for j in range(0,len(rp_cut)):
            Corr_SN[i,j]=Cov_SN[i,j]/np.sqrt(Cov_SN[i,i]*Cov_SN[j,j])

    np.savetxt('../txtfiles/Corr_SN_only.dat', Corr_SN)

    #plt.figure()
    #plt.imshow(Corr_SN)
    #plt.colorbar()
    #plt.savefig('../plots/Corr_SN_only.pdf')

    #plt.figure()
    #plt.imshow(np.log10(np.abs(Cov_SN)))
    #plt.colorbar()
    #plt.savefig('./cov_shapenoise_only.pdf')
    #plt.close()

    #print('shape again=', Cov_SN.shape)

    # Set up the shape-noise multivariate Gaussian
    means = np.zeros((len(Cov_SN[0,:])))

    SN_samps = np.random.multivariate_normal(means, Cov_SN, Nsamps) # This should have dimensions Nsamps x rbins

    print('SN_samp_shape=', SN_samps.shape)

    ups_with_SN = np.zeros_like(ups_jn)
    SN_flip = np.zeros_like(ups_jn)
    for i in range(0,Nsamps):
        ups_with_SN[:,i] = ups_jn[:,i] + SN_samps[i, :]
        SN_flip[:,i] = SN_samps[i,:]

    ups_mean = np.zeros(len(rp_cut)) 
    ups_std = np.zeros(len(rp_cut))
    SN_mean = np.zeros(len(rp_cut))
    SN_std = np.zeros(len(rp_cut))
    for i in range(0, len(rp_cut)):
        ups_mean[i] = np.mean(ups_with_SN[i,:])
        ups_std[i] = np.std(ups_with_SN[i,:])
        SN_mean[i] = np.mean(SN_flip[i,:])
        SN_std[i] = np.std(SN_flip[i,:])


    # Save these in the same format as the original file:
    save_stuff = np.column_stack((rp_cut, ups_mean, ups_std, ups_mean, ups_with_SN))
    save_stuff_SN = np.column_stack((rp_cut, SN_mean, SN_std, SN_mean, SN_flip))

    np.savetxt(data_dir+'/ups_gm_with_SN_'+lens+'_'+src+'.dat', save_stuff)
    np.savetxt(data_dir+'/SN_samps_'+lens+'_'+src+'.dat', save_stuff_SN)
    return

if __name__=="__main__":
    data_dir='' #'data_for_Danielle/'
    print('input order:',sys.argv[1:])
    if(len(sys.argv)>1):
        stat_list=sys.argv[1:]
    else:
        stat_list=['gm','beta','bias','f','gg']

    covdic=evaluate_correlation_matrix(stat_list=stat_list,rmin=2.0,plots=1,data_dir=data_dir)
    pl.tight_layout()
    pl.show()
