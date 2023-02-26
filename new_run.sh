#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH --nodes=1 --ntasks=28 # --gpus-per-node=1
#SBATCH --job-name=5441_LAB_1_TEST 
# account for CSE 5441 Au'21
#SBATCH --account=PAS2400
export PGM=main.py    
export SLURM_SUBMIT_DIR=/fs/ess/PAS2400/Patrick/ASR_Group_7/conformer1

echo job started at `date`
echo on compute node `cat $PBS_NODEFILE`

module load miniconda3
source activate asr

cd ${SLURM_SUBMIT_DIR} # change into directory with program to test on node
# /users/PAS1211/osu1053/CSE_5441/transform_library_mtx
# cp PCS_data_t00100 $TMPDIR #TMP is per node local storage on the cluster for good performance from data files
cd $TMPDIR # be in directory that you are reading data from

# echo job started at `date` >>current.out
# time ${SLURM_SUBMIT_DIR}/${PGM} < PCS_data_t00100  >> current.out 2>&1 # redirect stderr and stdout to the output file
python ${SLURM_SUBMIT_DIR}/${PGM}

# export SAVEDIR=${SLURM_SUBMIT_DIR}/tests/data_test.${SLURM_JOBID}
# mkdir ${SAVEDIR}
# mv current.out ${SAVEDIR}