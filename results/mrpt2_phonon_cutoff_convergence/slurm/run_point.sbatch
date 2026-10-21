#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/DFS-L/DATA/cui/chenxw25/lfmr_runs/mrpt2_mscan/logs/%x-%j.out
#SBATCH --requeue
# Usage: sbatch -J NAME -c CPUS --mem=MEM -p PART run_point.sbatch CODEDIR CASE OMEGA ALPHA [extra driver args]
CODE=$1; CASE=$2; OMEGA=$3; ALPHA=$4; shift 4
R=/DFS-L/DATA/cui/chenxw25/lfmr_runs/mrpt2_mscan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE start=$(date) code=$CODE args=$*"
cd $CODE
/DFS-L/DATA/cui/chenxw25/envs/lfmr/bin/python -m examples.mrpt2_m_convergence \
  --case $CASE --omega $OMEGA --alpha $ALPHA \
  --output $R/$CASE/w${OMEGA}_a${ALPHA}.csv "$@"
rc=$?
echo "end=$(date) rc=$rc"
exit $rc
