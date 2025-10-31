#!/bin/bash 

### Example OpenMP job ###

#SBATCH --job-name=D50Y10Pl
#SBATCH -o ../txtfiles/output/D25Y10Pl.%j
#SBATCH -e ../txtfiles/errors/D25Y10Pl.%j
#SBATCH -p slurm
#SBATCH -A dp339

# run for one hour, turn off all mail notifications
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE

# 1 task with 128 cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

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

python post_pred.py --nruns=1000 --nworkers=128 --outfile=../txtfiles/post_pred_test_Omrc0pt5_CMBPrior_LSSTY10_gc_seed_simscov_1000runs.json --OmMerr=0.0084 --gravtheory=nDGP --gravpar=0.5 --srcsamp=LSSTY10 --covfile=../txtfiles/cov_EG_nLbiascorrected_Y10_simscov.dat


#deactivate 

exit 0
