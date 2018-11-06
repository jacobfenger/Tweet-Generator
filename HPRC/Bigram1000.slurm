#!/bin/bash
##ENVIRONMENT SETTTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE     #Do not propagate environment
#SBATCH --get-user-env=L  #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Bigram1000      #Set the job name to "Bigram1000"
#SBATCH --time=05:00:00            #Set the wall clock limit to 5hr
#SBATCH --nodes=1                  #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=25G                  #Request 25G per node
#SBATCH --output=BigramOut1000.%j  #Send stdout/err to BigramOut.[jobID]

cd $SCRATCH
cd MP
module purge
module load Python/3.5.2-intel-2017A
module load numpy/1.14.1-intel-2017A-Python-3.5.2
ml
python Viterbi_bigram_mp.py processed1000.txt
