// ***********************************************************
// Code to estimate Fisher matrix constraints on b*sig8 and f*sig8, for 
// multiple populations. If used in publications, please cite
// White, Song & Percival (2008)
//
// Code calls GSL libraries (www.gnu.org/software/gsl/) available under the 
// GNU public license. It should compile using 
// 
// g++ -LGSL_DIR -lm -lgsl -lgslcblas -o fisher_gsl fisher_gsl.c
// 
// where GSL_DIR is the location of the libraries.
//
// Please report any bugs or problems to will.percival@port.ac.uk
//
// This version 8/10/2008, written by Will Percival
// ***********************************************************

// Modified April 2018 by Danielle Leonard 
//  - calls power spectrum from Core Cosmology Library rather than EH
//    (need to link ccl library when compiling)

#include <malloc.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include "ccl.h"

void zspace_mbias_pk(const int,double*,double,double*,double,double,double,double,gsl_matrix*);

int main() {

    const int NSAMP = 1;      // number of samples (with different nbar, bias)
    double vol = 1.51e10;       // volume in h^-3Mpc^3
    double nbar_tot = 3.2e-4; // total galaxy number density, DESI LRGs
    double sig8  = 0.83;       // sigma_8(mass)
    double f = 0.822; // f for our fiducial cosmology
    double klim = 0.1;        // klim / h Mpc^-1
    FILE * output;

    // set up bias and number density
    double *bias = (double*)calloc(NSAMP,sizeof(double));
    double *nbar = (double*)calloc(NSAMP,sizeof(double));
  
    bias[0] = 2.03; // Not correct for DESI LRGs but this is what we ran our covariances for.
    nbar[0] = nbar_tot;
  
    // calculate covariance matrix
    gsl_matrix *cov = gsl_matrix_calloc(NSAMP+1,NSAMP+1);
    zspace_mbias_pk(NSAMP,nbar,sig8,bias,f,0.0,vol,klim,cov);
  
    printf("f*sig8 = %g +/- %g\n",f*sig8,sqrt(cov->data[(NSAMP+1)*NSAMP+NSAMP]));
    printf("(F_(b, b))^(-1)=%f\n", gsl_matrix_get(cov, 0, 0) / sig8 / sig8 );
    printf("(F_(b, f))^(-1)=%f\n", gsl_matrix_get(cov, 0, 1) / sig8 / sig8);
    printf("(F_(f, f))^(-1)=%f\n", gsl_matrix_get(cov, 1, 1) / sig8 / sig8);
    
    // Get error on beta (see equation 5 of White et al. 2008)
    double beta_err;
    double beta = f / bias[0];
    beta_err = (beta) / bias[0] * sqrt( beta * beta * gsl_matrix_get(cov, 0, 0) - 2. * beta * gsl_matrix_get(cov, 0, 1) +  gsl_matrix_get(cov, 1, 1)) / sig8;
    printf("beta=%f, beta err=%f\n", beta, beta_err);
    
    output = fopen("/home/danielle/Research/EG_comparison/txtfiles/beta_err_DESI_4MOST_18000deg2_LRGs.txt", "w");
    fprintf(output, "%1.14f\n", beta_err);
    fclose(output);

    exit(0);
}

void zspace_mbias_pk  ( 
		      const int NSAMP, // number of samples
		      double *nbar,    // The number density in h^3 Mpc^-3
		      double sigma8,   // The real-space, linear clustering amplitude
		      double *bias,    // The real-space, linear bias
		      double f,	       // f ~ Omega_m^(0.6) 
		      double Sigma_z,  // z error translated into comoving distance
		      double vol_mpc,  // The survey volume in h^-3 Gpc^3
		      double kmax,     // Maximum k-value to integrate to
		      gsl_matrix *invfish  // inverse covariance matrix for bs and fs
    ) {

    // Catcher to avoid problems caused by odd inputs
    if(vol_mpc<=0.0) { fprintf(stderr,"volume<0"); exit(0); }
    for(int is=0;is<NSAMP;is++) 
      if(bias[is]<=0.0 || nbar[is]<=0.0) 
        { fprintf(stderr,"bias or nbar <=0"); exit(0); }
  
    double kstep     = 0.001;                     // step in k integration
    double mustep    = 0.001;                     // step in mu intergration
    double Sigma_z2  = Sigma_z*Sigma_z;           // (redshift error)^2

    // if NSAMP = 1 then NPOW = 1
    int NPOW = NSAMP*(NSAMP+1)/2;

    gsl_matrix *bigfish = gsl_matrix_calloc(NSAMP+1,NSAMP+1); // NSAMP = 1 means size is 2x2
    double *Ps      = (double*)calloc(NSAMP,sizeof(double)); 
    double *err     = (double*)calloc(NSAMP,sizeof(double));
    gsl_matrix *cov     = gsl_matrix_calloc(NPOW,NPOW); // NSAMP =1 means size of 1x1
    gsl_matrix *icov    = gsl_matrix_calloc(NPOW,NPOW); // NSAMP =1 means size of 1x1
    gsl_matrix *dPdp    = gsl_matrix_calloc(NPOW,NSAMP+1); //NSAMP = 1 means size of 1x2
  
    // Set up CCL cosmology for getting the power spectrum
    int status = 0;
    ccl_configuration config = default_config;
    config.matter_power_spectrum_method=ccl_halofit;
    ccl_parameters params = ccl_parameters_create_flat_lcdm(0.25, 0.05, 0.68, sigma8, 0.96, &status); // cosmological parameters hardcoded
    ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

    // integral over kmax
    for(double k=0.5*kstep; k<kmax; k+=kstep) {
    
        double pk = ccl_linear_matter_power(cosmo, k, 1. / (1. + 0.72), &status); // at a single k value so gives a single number (redshift currently hardcoded)
    
        // integral over mu
        for(double mu=0.5*mustep; mu<1.; mu+=mustep) {
            double mu2 = mu*mu;
      
            // has been written for simplicity, but could be rewritten for speed
      
            // If Sigz2=0, then zdamp = 1
            double zdamp  = exp(-k*k*Sigma_z2*mu2); // power spectrum damping due to z-error (inc photo-z)

            // calculate power spectra and multiplicative error terms
            // when NSAMPE=1, i=0 only
            for(int i=0;i<NSAMP;i++) {
	            Ps[i]      = (bias[i]+mu2*f)*(bias[i]+mu2*f)*pk*zdamp; // damped redshift-space P(k) // at given k, mu in steps
	            err[i]     = (1.+1./(nbar[i]*Ps[i]));                  // multiplicative shot noise // at given k, mu in steps
            }

            // need covariance < P_ij P_lm >. Loop through these, assuming j>=i, m>=l
            int ip=-1;
            for(int i=0;i<NSAMP;i++) for(int j=i;j<NSAMP;j++) { // if NSAMP=1, i=0 and j=0 only
	            ip++; // ip=0 first loop (if NSAMP=1, only loop)
	            int jp=-1; 
	            for(int l=0;l<NSAMP;l++) for(int m=l;m<NSAMP;m++) { // if NSAMP=1, l=0 and m=0 only
	                jp++; // jp=0 first loop (if NSAMP=1, only loop)
	                int index = ip*NPOW + jp; // index=0 first loop
	                if(ip==jp) {
		                // diagonal elements
		                if(i==j) cov->data[index] = 2.*Ps[i]*Ps[j]*err[i]*err[j]; // NSAMP=1, we only get this part 
		                if(i!=j) cov->data[index] = Ps[i]*Ps[j] + Ps[i]*Ps[j]*err[i]*err[j];
	                } else { // if NSAMP=0, we won't hit this part 
		                // off-diagonal elements
		                cov->data[index] = 2.*sqrt(Ps[i]*Ps[j]*Ps[l]*Ps[m]);
		                if(i==j && (i==l || i==m)) cov->data[index] *= err[i];
		                if(l==m && (l==i || l==j)) cov->data[index] *= err[l];
		                if(i!=j && l!=m && (i==l || i==m)) cov->data[index] = 0.5*cov->data[index]*(1.+err[i]);
		                if(i!=j && l!=m && (j==l || j==m)) cov->data[index] = 0.5*cov->data[index]*(1.+err[j]);
	                }

	            } // end first loop through power spectra P_lm

	            // set up derivatives of power spectra
	            // note, there are derivatives, NOT lnP derivatives.
	            for(int s=0;s<NSAMP;s++) { // if NSAMP=1, s=0 only 
	                int index = ip*(NSAMP+1) + s; // 0 first time through
	                if(i==j && i==s) dPdp->data[index] = Ps[i]*2./(sigma8*(bias[i]+mu2*f)); // if NSAMP=1, we only hit this one
	                if(i!=j && i==s) dPdp->data[index] = Ps[j]   /(sigma8*(bias[j]+mu2*f)); 
	                if(i!=j && j==s) dPdp->data[index] = Ps[i]   /(sigma8*(bias[i]+mu2*f)); 
	            }
	            // when NSAMP=1, the index here is 1
	            dPdp->data[ip*(NSAMP+1)+NSAMP] = (Ps[i]*mu2/(sigma8*(bias[i]+mu2*f))+
					    Ps[j]*mu2/(sigma8*(bias[j]+mu2*f)));
 
	        } // end second loop through power spectra P_ij
      
            // invert covariance matrix
            int signum;
            gsl_permutation *p = gsl_permutation_alloc(NPOW);
            gsl_linalg_LU_decomp(cov,p,&signum);
            gsl_linalg_LU_invert(cov,p,icov);
            gsl_permutation_free(p);

            // now calculate Fisher matrix					
            for(int l=0;l<NSAMP+1;l++) for(int m=0;m<NSAMP+1;m++) { // if NSAMP=1, runs over l=0-1, m=0-1
		        double fish = 0.0;
	            int index_fish = l*(NSAMP+1) + m; // this is because the bigfish->data representation of the Fisher matrix is flattened. Runs over 0,1,2,3.
	            // for NSAMP=1, the below loops run over only i=0, j=0
	            for(int i=0;i<NPOW;i++) for(int j=0;j<NPOW;j++)  fish += icov->data[i*NPOW+j]*dPdp->data[i*(NSAMP+1)+l]*dPdp->data[j*(NSAMP+1)+m];
	            fish *= k*k*vol_mpc;
	            bigfish->data[index_fish] += fish; // This sum is added to over the k and mu loops to perform the integral 
	         }
        }
    }
    // final constant multiplicative terms in double integration
    for(int l=0;l<NSAMP+1;l++) for(int m=0;m<NSAMP+1;m++) { // l=0-1, m=0-1 for NSAMP=1
        int index_fish = l*(NSAMP+1) + m; // again account for flattened storage of Fisher 
        bigfish->data[index_fish] *= 2.0/4.0/M_PI/M_PI*kstep*mustep;
    }

    // invert the Fisher matrix
    int signum;
    gsl_permutation *p = gsl_permutation_alloc(NSAMP+1);
    gsl_linalg_LU_decomp(bigfish,p,&signum);
    gsl_linalg_LU_invert(bigfish,p,invfish);
    gsl_permutation_free(p);

    // free memory
    gsl_matrix_free(bigfish);
    free(Ps);
    free(err);
    gsl_matrix_free(cov);
    gsl_matrix_free(icov);
    gsl_matrix_free(dPdp);

}
