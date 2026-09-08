#!/bin/bash
# ===========================================================================
# SLURM job array for the full Meta-World sweep.
#
# A job array is the right tool here: one submission, one queue entry per task,
# and the scheduler backfills them. Submitting 100 separate sbatch jobs works
# but clutters the queue and is harder to cancel or resubmit selectively.
#
# USAGE
#   1. Generate the job list (one command per line):
#        python3 benchmarks/metaworld/launch.py --suite iclr --seeds 5 \
#            --arms baseline shs --steps 500000 > jobs.txt
#        wc -l jobs.txt          # sanity check the count before submitting
#
#   2. Submit the array, capped at N concurrent jobs:
#        sbatch --array=1-$(wc -l < jobs.txt)%10 \
#            benchmarks/metaworld/sbatch_array.sh jobs.txt
#
#      The %10 limits concurrency. Raise it if your allocation allows; lower it
#      if you are sharing the account.
#
#   3. Resubmit only the ones that timed out or failed:
#        sacct -j <arrayid> --format=JobID,State | grep -v COMPLETED
#        sbatch --array=3,17,42 benchmarks/metaworld/sbatch_array.sh jobs.txt
#
# Runs auto-resume from <logdir>/latest.pt, so a resubmitted array element
# continues rather than restarting.
#
# MEASURED wall clock on an H100 3g.40gb MIG slice, proprioceptive:
#   baseline 13.3 env steps/s -> 500k in ~10.4h, 1M in ~20.9h
#   SHS       7.7 env steps/s -> 500k in ~18.0h, 1M in ~36.1h
# The 36h figure is why --time is 36:00:00 here. Fir permits up to 168h.
# ===========================================================================

#SBATCH --job-name=mw-sweep
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/gsubbara/AIME/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/gsubbara/AIME/slurm_logs/%x-%A_%a.err

set -euo pipefail

JOBFILE=${1:?usage: sbatch --array=1-N%K sbatch_array.sh <jobs.txt>}
REPO=/home/gsubbara/AIME

cd "$REPO"
module load opencv mujoco/3.3.0
source aime/bin/activate

# GL backend. Proprio configs set mw_render: false and need no GL at all;
# vision configs render every step and require osmesa or egl. Override from the
# submit line for a vision sweep:
#   sbatch --export=ALL,MUJOCO_GL=osmesa --array=... sbatch_array.sh jobs.txt
export MUJOCO_GL=${MUJOCO_GL:-disable}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBFILE")
if [ -z "$CMD" ]; then
    echo "FATAL: no command at line ${SLURM_ARRAY_TASK_ID} of $JOBFILE"
    exit 1
fi

echo "array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | host $(hostname) | $(date)"
echo "cmd: $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail fast on a bad node rather than dying inside model construction.
python3 -c "
import socket, sys, torch
if not torch.cuda.is_available():
    print('FATAL: CUDA unavailable on', socket.gethostname(), flush=True); sys.exit(1)
print('CUDA OK:', torch.cuda.get_device_name(0), flush=True)
" || { echo "GPU check failed on $(hostname)"; exit 1; }

eval "$CMD"

echo "finished $(date)"
