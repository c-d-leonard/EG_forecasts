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

#include <malloc.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include "ccl.h"

void zspace_mbias_pk(const int,double*,double,double*,double,double,double,double,gsl_matrix*);
//double tk_eh98(double);

int main() {

  const int NSAMP = 1;      // number of samples (with different nbar, bias)
  double vol = 1.0e9;       // volume in h^-3Mpc^3
  double nbar_tot = 5.0e-4; // total galaxy number density
  double sig8  = 0.83;       // sigma_8(mass)
  double f = pow(0.25,0.6); // f
  double klim = 0.1;        // klim / h Mpc^-1

  // set up bias and number density
  double *bias = (double*)calloc(NSAMP,sizeof(double));
  double *nbar = (double*)calloc(NSAMP,sizeof(double));
  //for(int is=0;is<NSAMP;is++) {
  //  bias[is] = 1.0 + 1.0*((double)(is+1)-0.5)/(double)NSAMP;
  //  nbar[is] = nbar_tot / (double)NSAMP;
  //}
  
  bias[0] = 3.9; // DESI LRGs
  nbar[0] = nbar_tot;
  
  // calculate covariance matrix
  gsl_matrix *cov = gsl_matrix_calloc(NSAMP+1,NSAMP+1);
  zspace_mbias_pk(NSAMP,nbar,sig8,bias,f,0.0,vol,klim,cov);
  
  printf("\n%d samples:\n",NSAMP);
  for(int i=0;i<NSAMP;i++) 
    printf("sample %d : bias*sig8 = %g +/- %g\n",i,bias[i]*sig8,sqrt(cov->data[i*(NSAMP+1)+i]));
  printf("f*sig8 = %g +/- %g\n",f*sig8,sqrt(cov->data[(NSAMP+1)*NSAMP+NSAMP]));

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

  // catcher to avoid problems caused by odd inputs
  if(vol_mpc<=0.0) { fprintf(stderr,"volume<0"); exit(0); }
  for(int is=0;is<NSAMP;is++) 
    if(bias[is]<=0.0 || nbar[is]<=0.0) 
      { fprintf(stderr,"bias or nbar <=0"); exit(0); }
  
  double kstep     = 0.001;                     // step in k integration
  double mustep    = 0.001;                     // step in mu intergration
  double Sigma_z2  = Sigma_z*Sigma_z;           // (redshift error)^2

  int NPOW = NSAMP*(NSAMP+1)/2;

  gsl_matrix *bigfish = gsl_matrix_calloc(NSAMP+1,NSAMP+1);
  double *Ps      = (double*)calloc(NSAMP,sizeof(double));
  double *err     = (double*)calloc(NSAMP,sizeof(double));
  gsl_matrix *cov     = gsl_matrix_calloc(NPOW,NPOW);
  gsl_matrix *icov    = gsl_matrix_calloc(NPOW,NPOW);
  gsl_matrix *dPdp    = gsl_matrix_calloc(NPOW,NSAMP+1);
  
  // Set up CCL cosmology for getting the power spectrum
  int status = 0;
  ccl_configuration config = default_config;
  config.matter_power_spectrum_method=ccl_halofit;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(0.25, 0.05, 0.68, sigma8, 0.96, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // integral over kmax
  for(double k=0.5*kstep; k<kmax; k+=kstep) {
    
    // Use ccl here instead 
    //double tf   = tk_eh98(k);
    //double pk = 7.03563e+06*k*tf*tf*sigma8*sigma8; 
    
    double pk = ccl_linear_matter_power(cosmo, k, 1.0, &status);
    
    // integral over mu
    for(double mu=0.5*mustep; mu<1.; mu+=mustep) {
      double mu2 = mu*mu;
      
      // has been written for simplicity, but could be rewritten for speed
      
      double zdamp  = exp(-k*k*Sigma_z2*mu2); // power spectrum damping due to z-error (inc photo-z)

      // calculate power spectra and multiplicative error terms
      for(int i=0;i<NSAMP;i++) {
	Ps[i]      = (bias[i]+mu2*f)*(bias[i]+mu2*f)*pk*zdamp; // damped redshift-space P(k)
	err[i]     = (1.+1./(nbar[i]*Ps[i]));                  // multiplicative shot noise
      }

      // need covariance < P_ij P_lm >. Loop through these, assuming j>=i, m>=l
      int ip=-1;
      for(int i=0;i<NSAMP;i++)
	for(int j=i;j<NSAMP;j++) {
	  ip++;

	  int jp=-1;
	  for(int l=0;l<NSAMP;l++) 
	    for(int m=l;m<NSAMP;m++) {
	      jp++;

	      int index = ip*NPOW + jp;

	      if(ip==jp) {
		// diagonal elements
		if(i==j) cov->data[index] = 2.*Ps[i]*Ps[j]*err[i]*err[j];
		if(i!=j) cov->data[index] = Ps[i]*Ps[j] + Ps[i]*Ps[j]*err[i]*err[j];
	      } else {
		// off-diagonal elements
		cov->data[index] = 2.*sqrt(Ps[i]*Ps[j]*Ps[l]*Ps[m]);
		if(i==j && (i==l || i==m)) cov->data[index] *= err[i];
		if(l==m && (l==i || l==j)) cov->data[index] *= err[l];
		if(i!=j && l!=m && (i==l || i==m)) cov->data[index] = 0.5*cov->data[index]*(1.+err[i]);
		if(i!=j && l!=m && (j==l || j==m)) cov->data[index] = 0.5*cov->data[index]*(1.+err[j]);
	      }

	    } // end first loop through power spectra P_lm

	  // set up derivatives of power spectra
	  for(int s=0;s<NSAMP;s++) {
	    int index = ip*(NSAMP+1) + s;
	    if(i==j && i==s) dPdp->data[index] = Ps[i]*2./(sigma8*(bias[i]+mu2*f)); 
	    if(i!=j && i==s) dPdp->data[index] = Ps[j]   /(sigma8*(bias[j]+mu2*f)); 
	    if(i!=j && j==s) dPdp->data[index] = Ps[i]   /(sigma8*(bias[i]+mu2*f)); 
	  }
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
      for(int l=0;l<NSAMP+1;l++) 
	for(int m=0;m<NSAMP+1;m++) {
	  double fish = 0.0;
	  int index_fish = l*(NSAMP+1) + m;
	  for(int i=0;i<NPOW;i++)
	    for(int j=0;j<NPOW;j++) 
	      fish += icov->data[i*NPOW+j]*dPdp->data[i*(NSAMP+1)+l]*dPdp->data[j*(NSAMP+1)+m];
	  fish *= k*k*vol_mpc;
	  bigfish->data[index_fish] += fish;
	}
      
    }
  }
    
  // final constant multiplicative terms in double integration
  for(int l=0;l<NSAMP+1;l++) 
    for(int m=0;m<NSAMP+1;m++) {
      int index_fish = l*(NSAMP+1) + m;
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

// Transfer function of Eisenstein & Hu 1998 
// (Equation numbers refer to this paper)
/*double tk_eh98(double k)
{
  double rk,e,thet,thetsq,thetpf,b1,b2,zd,ze,rd,re,rke,s,rks,q,y,g;
  double ab,a1,a2,ac,bc,f,c1,c2,tc,bb,bn,ss,tb,tk_eh;
  double h,hsq,om_mhsq,om_b,om_m;

  // set up cosmology
  h    = 0.72;
  om_m = 0.25;
  om_b = 0.15*om_m;

  // convert k to Mpc^-1 rather than hMpc^-1
  rk=k*h;
  hsq=h*h;
  om_mhsq=om_m*hsq;

  // constants
  e=exp(1.);      
  thet=2.728/2.7;
  thetsq=thet*thet;
  thetpf=thetsq*thetsq;

  // Equation 4 - redshift of drag epoch
  b1=0.313*pow(om_mhsq,-0.419)*(1.+0.607*pow(om_mhsq,0.674));
  b2=0.238*pow(om_mhsq,0.223);
  zd=1291.*(1.+b1*pow(om_b*hsq,b2))*pow(om_mhsq,0.251)
    /(1.+0.659*pow(om_mhsq,0.828));

  // Equation 2 - redshift of matter-radiation equality
  ze=2.50e4*om_mhsq/thetpf;

  // value of R=(ratio of baryon-photon momentum density) at drag epoch
  rd=31500.*om_b*hsq/(thetpf*zd);

  // value of R=(ratio of baryon-photon momentum density) at epoch of
  // matter-radiation equality
  re=31500.*om_b*hsq/(thetpf*ze);

  // Equation 3 - scale of ptcle horizon at matter-radiation equality
  rke=7.46e-2*om_mhsq/(thetsq);

  // Equation 6 - sound horizon at drag epoch
  s=(2./3./rke)*sqrt(6./re)*log((sqrt(1.+rd)+sqrt(rd+re))/(1.+sqrt(re)));

  // Equation 7 - silk damping scale
  rks=1.6*pow(om_b*hsq,0.52)*pow(om_mhsq,0.73)*(1.+pow(10.4*om_mhsq,-0.95));

  // Equation 10  - define q
  q=rk/13.41/rke;
      
  // Equations 11 - CDM transfer function fits
  a1=pow(46.9*om_mhsq,0.670)*(1.+pow(32.1*om_mhsq,-0.532));
  a2=pow(12.0*om_mhsq,0.424)*(1.+pow(45.0*om_mhsq,-0.582));
  ac=pow(a1,(-om_b/om_m))*pow(a2,pow(-(om_b/om_m),3.));

  // Equations 12 - CDM transfer function fits
  b1=0.944/(1.+pow(458.*om_mhsq,-0.708));
  b2=pow(0.395*om_mhsq,-0.0266);
  bc=1./(1.+b1*(pow(1.-om_b/om_m,b2)-1.));

  // Equation 18
  f=1./(1.+pow(rk*s/5.4,4.));

  // Equation 20
  c1=14.2 + 386./(1.+69.9*pow(q,1.08));
  c2=14.2/ac + 386./(1.+69.9*pow(q,1.08));

  // Equation 17 - CDM transfer function
  tc=f*log(e+1.8*bc*q)/(log(e+1.8*bc*q)+c1*q*q) +
    (1.-f)*log(e+1.8*bc*q)/(log(e+1.8*bc*q)+c2*q*q);

  // Equation 15
  y=(1.+ze)/(1.+zd);
  g=y*(-6.*sqrt(1.+y)+(2.+3.*y)*log((sqrt(1.+y)+1.)/(sqrt(1.+y)-1.)));

  // Equation 14
  ab=g*2.07*rke*s/pow(1.+rd,0.75);

  // Equation 23
  bn=8.41*pow(om_mhsq,0.435);

  // Equation 22
  ss=s/pow(1.+pow(bn/rk/s,3.),1./3.);

  // Equation 24
  bb=0.5+(om_b/om_m) + (3.-2.*om_b/om_m)*sqrt(pow(17.2*om_mhsq,2.)+1.);

  // Equations 19 & 21
  tb=log(e+1.8*q)/(log(e+1.8*q)+c1*q*q)/(1+pow(rk*s/5.2,2.));
  tb=(tb+ab*exp(-pow(rk/rks,1.4))/(1.+pow(bb/rk/s,3.)))*sin(rk*ss)/rk/ss;
    
  // Equation 8
  tk_eh=(om_b/om_m)*tb+(1.-om_b/om_m)*tc;
  
  return tk_eh;
}*/
