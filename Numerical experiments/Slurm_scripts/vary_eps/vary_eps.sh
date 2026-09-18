#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=econ
#SBATCH --qos=normal
#SBATCH -t 5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# 5 n-values and 17 replicate-blocks per n => 4*17 = 68 tasks
#SBATCH --array=0-67
#SBATCH --job-name=vary_eps
#SBATCH --output=${SLURM_SUBMIT_DIR}/%x_%A_%a.out
#SBATCH --error=${SLURM_SUBMIT_DIR}/%x_%A_%a.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export R_HOME="$CONDA_PREFIX/lib/R"
export R_LIBS_USER="$CONDA_PREFIX/lib/R/library"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job name        : $SLURM_JOB_NAME"
echo "Job ID          : $SLURM_JOB_ID"
echo "Array task ID   : $SLURM_ARRAY_TASK_ID"
echo "Submit dir      : $SLURM_SUBMIT_DIR"
echo "Python          : $(which python)"
echo "Python version  : $(python -V)"
echo "Conda env       : $CONDA_PREFIX"
echo "CPUs per task   : ${SLURM_CPUS_PER_TASK:-1}"

cd /home/nandy.15/Research/dp_beta_model || { echo "cd failed"; exit 1; }
mkdir -p Results/vary_eps

NS=(50 100 150 200)
N_PER_SIZE=17
REPS_PER_TASK=3

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_INDEX=$(( TASK_ID / N_PER_SIZE ))
REP_BLOCK=$(( TASK_ID % N_PER_SIZE ))

if [ $N_INDEX -ge ${#NS[@]} ]; then
  echo "Nothing to do for task ${TASK_ID} (N_INDEX=${N_INDEX})"
  exit 0
fi

N_VAL=${NS[$N_INDEX]}
REP_OFFSET=$(( REP_BLOCK * REPS_PER_TASK ))

echo "--------------------------------------------"
echo "Running configuration:"
echo "  n            = ${N_VAL}"
echo "  n_index      = ${N_INDEX}"
echo "  rep_block    = ${REP_BLOCK}"
echo "  reps/task    = ${REPS_PER_TASK}"
echo "  rep_offset   = ${REP_OFFSET}"
echo "--------------------------------------------"

python Python_scripts/vary_eps.py \
  --outdir Results/vary_eps \
  --tag n${N_VAL}_r3 \
  --n ${N_VAL} --r 3 --M 1.0 \
  --p0 0.3 --mean 0.5 --sd 0.02 \
  --c_lam 0.0001 \
  --eps_list "1e-4, 1e-3, 1e-2, 1e-1, 1.0" \
  --R 50 \
  --reps_per_task ${REPS_PER_TASK} \
  --rep_offset ${REP_OFFSET} \
  --base_seed $((12345 + 1000 * N_INDEX)) \
  --chunk_size_pairs 250000 \
  --maxiter_mle 10000 \
  --T_cap 10000 \
  --threads 1
