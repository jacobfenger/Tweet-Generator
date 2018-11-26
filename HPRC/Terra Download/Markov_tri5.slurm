#!/bin/bash
##ENVIRONMENT SETTTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE     #Do not propagate environment
#SBATCH --get-user-env=L  #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Markov_tri5         #Set the job name to "Markov_tri5"
#SBATCH --time=05:00:00            #Set the wall clock limit to 5hr
#SBATCH --nodes=1                  #Request 1 node
#SBATCH --ntasks-per-node=1        #Request 1 tasks/cores per node
#SBATCH --mem=30G                  #Request 30G per node
#SBATCH --output=Markov_tri5.%j        #Send stdout/err to Markov_tri5.[jobID]

cd $SCRATCH
cd V3
module purge
module load Python/3.5.2-intel-2017A
module load numpy/1.14.1-intel-2017A-Python-3.5.2
#ml
python Markov_trigram.py 5 1000
