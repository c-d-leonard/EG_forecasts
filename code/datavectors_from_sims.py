import numpy as np

# look at the covariance between upsilon and beta
def get_mean_datavec(stat=['gg'],rmin=1.5,data_dir=''):
    '''This computes and plots the datavector as the mean of the simulation realisations for the stat (gm, gg, beta) provided.
    gg: galaxy-galaxy upsilon
    gm: galaxy matter upsilon
    beta, bias and f can also be given '''
    
    col_dic={'f':1,'bias':2,'beta':3}
    latex_lab={'f':r'$f$','bias':r'$b$','beta':r'$\beta$'}

    #load the f bias bets        
    rsdfile=data_dir+'HOD-model-5-PB00-z0.75-wsFOG.txt'
    rsd_data=np.loadtxt(rsdfile)
    
    #stat_jn will hold the measurements
    stat_jn=np.array([])
    xlab=[]
    
    for pp,par in enumerate(stat):
        #load gm if needed
        if(par in ['gm','gg']):
            if(par=='gm'):
                gmfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-gxm-sel-crossparticles-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                ups=np.loadtxt(gmfile)
            elif(par=='gg'):
                ggfile=data_dir+'test-HOD-PB00-z0.75-w1pz_cat-zRSD-model-5-sel-All-wtag-w1-rfact10-bin1-wp-logrp-pi-NJN-100.txt.upsilon'
                ups=np.loadtxt(ggfile)
        

            #remove small scale upsilon not needed
            #ind=ups[:,0]>rmin
            #rp_ups=ups[ind,0]
            ind = next(j[0] for j in enumerate(ups[:,0]) if j[1]>rmin)
            rp_ups=ups[(ind-1):,0]
            xlab.append(rp_ups)
            ups_jn=ups[(ind-1):,4:]
            if(stat_jn.size==0):
                stat_jn=np.copy(ups_jn)
            else:
                stat_jn=np.row_stack([stat_jn,ups_jn])
                
            N = len(stat_jn[0,:])

            means = np.zeros(len(stat_jn[:,0]))
            for i in range(len(stat_jn[:,0])):
                means[i] = sum(stat_jn[i,:]) / N
        
            print('rp shape=', rp_ups.shape)
            print('means=', means.shape)    
            # Save the rp values and the means at each:
            save_stat = np.column_stack((rp_ups, means))

            np.savetxt('./from_sims_means_'+stat[0]+'.txt', save_stat)

        else:
            xlab.append(par)
            if(stat_jn.size==0):
                stat_jn=rsd_data[:,col_dic[par]]
            else:
                stat_jn=np.row_stack([stat_jn,rsd_data[:,col_dic[par]]])
            
            N = len(stat_jn)
            
            means = sum(stat_jn)/N
            
            save_stat = means


    #cov=np.cov(stat_jn)
    # Need to get the covariance from the Jackknife - specific method, not
    # from a sample covariance method.
   


    return save_stat
