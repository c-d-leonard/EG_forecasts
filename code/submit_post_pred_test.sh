#!/bin/bash 

# setup modules
module purge
module load python/3.8.10
module load aocc/4.0.0
module load openmpi
module load fftw/4.0
module load gsl/cray/2.7.1
module load swig/4.0.2
module load cmake/3.21.3

# source the virtual environments
source ../../../Software/egforecasts/bin/activate

# set OMP_NUM_THREADS explicitly to the number of cpus assigned to this task
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#execute the job

python post_pred.py --nruns=3 --nworkers=1 --outfile=../txtfiles/post_pred_test_fR-6_DESY3Prior_LSSTY1_gc_seed_simscov_test.json --OmMerr=0.03 --gravtheory=fR --gravpar=0.000001 --srcsamp=LSST --covfile=../txtfiles/cov_EG_nLbiascorrected_Y1_simscov.dat


#deactivate 

exit 0
