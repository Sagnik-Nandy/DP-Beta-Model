#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=econ
#SBATCH --qos=normal
#SBATCH -t 8:00:00             # Link pred takes longer; increased time limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# 5 n-values and 17 replicate-blocks per n => 4*17 = 68 tasks
#SBATCH --array=0-67
#SBATCH --job-name=linkpred_vary_eps
#SBATCH --output=${SLURM_SUBMIT_DIR}/%x_%A_%a.out
#SBATCH --error=${SLURM_SUBMIT_DIR}/%x_%A_%a.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

# Optimization for Slurm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /home/nandy.15/Research/dp_beta_model || { echo "cd failed"; exit 1; }
mkdir -p Results/link_prediction

# --- Array Mapping Logic ---
NS=(50 100 150 200)           # Your target n-values
N_PER_SIZE=17                  # Number of tasks per n-value
REPS_PER_TASK=3                # 10 tasks * 5 reps = 50 Total Replicates (R=50)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_INDEX=$(( TASK_ID / N_PER_SIZE ))
REP_BLOCK=$(( TASK_ID % N_PER_SIZE ))

if [ $N_INDEX -ge ${#NS[@]} ]; then
  echo "Nothing to do for task ${TASK_ID}"
  exit 0
fi

N_VAL=${NS[$N_INDEX]}
REP_OFFSET=$(( REP_BLOCK * REPS_PER_TASK ))

echo "--------------------------------------------"
echo "Running Link Prediction Configuration:"
echo "  n            = ${N_VAL}"
echo "  rep_offset   = ${REP_OFFSET}"
echo "  reps/task    = ${REPS_PER_TASK}"
echo "--------------------------------------------"

# Note: Added --neg_ratio 1.0 and --holdout_frac 0.2 to match your protocol
python Python_scripts/pred_error.py \
  --outdir Results/link_prediction \
  --tag lp_n${N_VAL}_r3 \
  --n ${N_VAL} --r 3 --M 2.0 \
  --p0 0.3 --mean -1.0 --sd 0.5 \
  --c_lam 0.1 \
  --eps_list "1e-2, 1e-1, 1.0" \
  --holdout_frac 0.20 \
  --neg_ratio 1.0 \
  --R 50 \
  --reps_per_task ${REPS_PER_TASK} \
  --rep_offset ${REP_OFFSET} \
  --base_seed $((54321 + 2000 * N_INDEX)) \
  --chunk_size_pairs 250000 \
  --maxiter_mle 10000 \
  --T_cap 10000 \
  --threads ${SLURM_CPUS_PER_TASK:-1}