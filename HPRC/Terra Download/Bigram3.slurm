#!/bin/bash
##ENVIRONMENT SETTTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE     #Do not propagate environment
#SBATCH --get-user-env=L  #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Bigram3         #Set the job name to "Bigram3"
#SBATCH --time=05:00:00            #Set the wall clock limit to 5hr
#SBATCH --nodes=1                  #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=10G                  #Request 10G per node
#SBATCH --output=Bigram3.%j        #Send stdout/err to Bigram3.[jobID]

cd $SCRATCH
cd V3
module purge
module load Python/3.5.2-intel-2017A
module load numpy/1.14.1-intel-2017A-Python-3.5.2
#ml
python Viterbi_bigram_mp.py 3 10000
