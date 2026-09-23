#!/bin/bash
#SBATCH --job-name=AIME-CARL-smoke
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0,20%1
#SBATCH --output=slurm-aime-carl-smoke-%A_%a.out
#SBATCH --error=slurm-aime-carl-smoke-%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logdir/smoke/${SLURM_ARRAY_JOB_ID:-manual}}"
export STEPS="${STEPS:-2000}"

exec bash "${PROJECT_DIR}/run_dreamer_carl_mila.sh" \
  --envs 1 --batch_size 4 --batch_length 32 --train_ratio 32 \
  --prefill 500 --pretrain 1 --time_limit 200 --eval_every 1000 \
  --log_every 1000 --eval_episode_num 2 --video_pred_log False \
  --shs_diag_log False --shs_diag_figures False "$@"
