#Author: Shadab Alam, May 2019
#This compute the covarainnce and the correlation matrix
from __future__ import print_function
import numpy as np
import pylab as pl
import sys
import matplotlib.pyplot as plt

# look at the covariance between upsilon and beta
def evaluate_correlation_matrix(stat_list=['gm','beta'],rmin=2.0,plots=0,data_dir=''):
    '''This computes and plots the correlation matrix
    a list of stats can be provided for correlation matrix evluation
    gg: galaxy-galaxy upsilon
    gm: galaxy matter upsilon
    beta, bias and f can also be given '''
    
    col_dic={'f':1,'bias':2,'beta':3}
    latex_lab={'f':r'$f$','bias':r'$b$','beta':r'$\beta$'}

    

    #load the f bias bets        
    rsdfile=data_dir+'HOD-model-5-PB00-z0.75-wsFOG.txt'
    rsd_data=np.loadtxt(rsdfile)
    
    #stat_jn will hold all the measurements
    stat_jn=np.array([])
    xlab=[]
    
    for pp,par in enumerate(stat_list):
        #load gm if needed
        if(par in ['gm','gg']):
            if(par=='gm'):
                gmfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                ups=np.loadtxt(gmfile)
            elif(par=='gg'):
                ggfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-sel-All-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                ups=np.loadtxt(ggfile)
        

            #remove small scale upsilon not needed
            ind=ups[:,0]>rmin
            rp_ups=ups[ind,0]
            xlab.append(rp_ups)
            ups_jn=ups[ind,4:]
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
