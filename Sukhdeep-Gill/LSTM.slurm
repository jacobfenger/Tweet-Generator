#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE     #Do not propagate environment
#SBATCH --get-user-env=L  #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=LSTM      #Set the job name to "Filtered1000"
#SBATCH --time=05:00:00            #Set the wall clock limit to 5hr
#SBATCH --ntasks=5                 #Request 1 task
#SBATCH --ntasks-per-node=2        #Request 1 tasks/cores per node
#SBATCH --mem=5G                   #Request 1G per node
#SBATCH --output=NLPLSTM.%j  #Send stdout/err to NLPLSTM.[jobID]

cd $SCRATCH
module purge
module load Anaconda/3-5.0.0.1
source activate keras-gpu-2.0.5
python New_LSTM.py
#python Example.py
#source deactivate
