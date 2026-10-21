#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/DFS-L/DATA/cui/chenxw25/lfmr_runs/mrpt2_mscan/logs/%x-%j.out
R=/DFS-L/DATA/cui/chenxw25/lfmr_runs/mrpt2_mscan
CODE=$R/code_w3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$CODE PYTHONUNBUFFERED=1
cd $CODE
for a in 2.4 1.6 2.0; do
  /DFS-L/DATA/cui/chenxw25/envs/lfmr/bin/python -m examples.mrpt2_m_convergence --case multi --omega 0.5 --alpha $a \
    --algorithm lean --min-total 5 --max-total 7 --tol 1e-30 --output $R/validation/lean_w0.5_a$a.csv
done
