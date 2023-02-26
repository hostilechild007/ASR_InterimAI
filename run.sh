#!/bin/bash
#SBATCH --job-name=ondemand/sys/myjobs/
#SBATCH --time=20:00:00
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --constraint=12core
#SBATCH --account=PAS2400

qstat -f $SLURM_JOB_ID

#   A Basic Python Serial Job for the OSC Owens cluster
#   https://www.osc.edu/resources/available_software/software_list/python

#
# The following lines set up the Python environment
#
#module load python
module load miniconda3
source activate asr
#
# Move to the directory where the job was submitted from
# You could also 'cd' directly to your working directory
cd $SLURM_SUBMIT_DIR

# Install stuff
#pip install tensorboard==1.14.0
#pip install transformers
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

#
# Run Python
#
python main.py
