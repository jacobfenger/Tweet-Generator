#!/bin/bash
##ENVIRONMENT SETTTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE     #Do not propagate environment
#SBATCH --get-user-env=L  #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=General_mp5       #Set the job name to "General_mp5"
#SBATCH --time=05:00:00            #Set the wall clock limit to 5hr
#SBATCH --nodes=1                  #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=10G                  #Request 10G per node
#SBATCH --output=General_mp5.%j      #Send stdout/err to General_mp5.[jobID]

cd $SCRATCH
cd V3
module purge
module load Python/3.5.2-intel-2017A
module load numpy/1.14.1-intel-2017A-Python-3.5.2
#ml
python Viterbi_generalized_mp.py build 5 100
python Viterbi_generalized_mp.py load 5 100
