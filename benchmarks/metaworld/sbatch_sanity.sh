#!/bin/bash
#SBATCH --job-name=mw-reach-baseline
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gsubbara/AIME/slurm_logs/%x-%j.out
#SBATCH --error=/home/gsubbara/AIME/slurm_logs/%x-%j.err

# Meta-World proprio sanity run: stock DreamerV3 on reach.
#
# NOTE: mkdir -p /home/gsubbara/AIME/slurm_logs before the first submit --
# SLURM does not create the --output directory and the job dies silently if it
# is missing.
#
# Submit:   sbatch benchmarks/metaworld/sbatch_sanity.sh
# Monitor:  squeue -u $USER ; tail -f slurm_logs/mw-reach-baseline-<jobid>.out

set -euo pipefail

REPO=/home/gsubbara/AIME
LOGDIR=./logdir/sanity/reach_baseline

cd "$REPO"

module load opencv mujoco/3.3.0
source aime/bin/activate

# Proprioceptive run: mw_render is false in metaworld_proprio, so no MuJoCo
# rendering happens and no GL backend is needed. dreamer.py uses setdefault for
# MUJOCO_GL, so this is respected. If the job fails on a mujoco GL import,
# switch to osmesa (or egl) -- 'disable' is only safe while mw_render is false.
export MUJOCO_GL=disable

# 4 CPUs allocated. Env simulation needs roughly 3% of one core at realistic
# training throughput, so the rest goes to torch. Capping avoids oversubscription.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=========================================================="
echo "job          : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "started      : $(date)"
echo "python       : $(which python3)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

# dreamer.py auto-resumes from <logdir>/latest.pt if it exists, and the replay
# episodes live in <logdir>/train_eps, so re-submitting this exact script after
# a 12h timeout continues the run rather than restarting it. Do NOT delete the
# logdir between submissions unless you intend a fresh run.
python3 dreamer.py \
    --configs metaworld_proprio \
    --task metaworld_reach \
    --seed 0 \
    --steps 300000 \
    --logdir "$LOGDIR"

echo "=========================================================="
echo "finished     : $(date)"
echo "=========================================================="
