#!/bin/bash
#SBATCH --job-name=SHS-RSSM
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=2-23:59:59
#SBATCH --account=def-bengioy
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-humanoid_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-humanoid_%N-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ---------------------------------------------------------------- RORQUAL --
# Resource notes (Rorqual GPU nodes are 4x H100-80GB SXM):
#   * --gpus-per-node=h100:1 is the Alliance-documented request form.
#   * Keep --cpus-per-task / --mem at or below the per-GPU share of the node
#     so the scheduler can pack the other three GPUs; 12 cores / 124G is a
#     conservative share for 4 envs + replay.  Check the current per-GPU
#     share on docs.alliancecan.ca (Rorqual) or `sinfo` before raising.
#   * --time: 3M steps ~= 59.5 h at the measured ~14 fps.  2-23:59:59 sits
#     in the SAME <=72 h scheduling bin as 61 h (no queue-priority cost) and
#     leaves ~12 h margin instead of ~2 h.  Dreamer resumes from LOGDIR, so
#     longer targets chain across jobs; TIME_LIMIT mail flags a hit wall.
#   * The #SBATCH --output directory cannot use variables and must EXIST on
#     Rorqual before submitting:  mkdir -p .../AIME/logs
#   * VENVS ARE NOT PORTABLE ACROSS CLUSTERS.  Rebuild on Rorqual once:
#       module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11 \
#                   scipy-stack/2024a opencv/4.10.0 mujoco/3.3.0 arrow
#       virtualenv --no-download "$VENV_DIR" && source "$VENV_DIR/bin/activate"
#       pip install --no-index torch torchvision tensorboard dm_control ruamel.yaml
#     The torch matmul gate below fails loudly on a stale/foreign venv.
# ---------------------------------------------------------------------------

set -euo pipefail

# Runtime overrides, for example:
#   SEED=3 STEPS=2000000 sbatch run_dreamer_shs_rssm_humanoid_rorqual.sh
#   SHS_LEARN_B0=False sbatch ...                 # fixed-b0 ablation
#   SHS_RSTICK_STOPGRAD=False sbatch ...          # joint-objective A/B
#   BATCH_SIZE=128 PRECISION=16 STEPS=30000 sbatch ...   # batch-scaling probe
#   TRAIN_RATIO=768 sbatch ...                  # replay-ratio scaling (see note)
#
# THROUGHPUT (measured, 264k-step humanoid run): ~14 fps
#   2M env steps ~= 40 h        3M env steps ~= 59.5 h
#
# BATCH-SIZE NOTES (this fork trains once every batch_size*batch_length /
# train_ratio env steps, so replayed-data-per-env-step is CONSTANT in batch
# size; bigger batches mean fewer, larger updates):
#   26x64 (stock)   -> 923k updates / 3M steps
#   96x64 (here)    -> 250k updates / 3M steps   (fits H100-80GB at fp32)
#   128x64          -> 187k updates / 3M steps   (marginal at fp32; probe
#                      with PRECISION=16 and a short STEPS run; heartbeat
#                      prints GPU memory so the high-water mark is in .err)
PROJECT_DIR="${PROJECT_DIR:-/home/memole/links/projects/def-irina/memole/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/links/D2E}"
SEED="${SEED:-1}"
STEPS="${STEPS:-3000000}"                   # 2M variant: STEPS=2000000
BATCH_SIZE="${BATCH_SIZE:-96}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
PRECISION="${PRECISION:-32}"                # 16 halves activation memory for
                                            # the CNN/RSSM; A/B before trusting
TRAIN_RATIO="${TRAIN_RATIO:-512}"           # replayed steps per env step; raising
                                            # THIS (not batch) is the preferred
                                            # more-learning-per-env-step lever.
                                            # NB wall-clock scales ~linearly:
                                            # 768 @ 3M steps ~= 89 h > the 72 h
                                            # bin -- chain jobs or raise --time.
SHS_Q_RANK="${SHS_Q_RANK:-4}"
SHS_ACTION_DIM="${SHS_ACTION_DIM:-21}"      # humanoid has 21 actuators (hopper 4, walker 6)
SHS_LEARN_B0="${SHS_LEARN_B0:-True}"        # hierarchical Gamma prior on the noise RATE b0
                                            # (conjugate CAVI; removes the hand-set noise scale)
SHS_B0_STRENGTH="${SHS_B0_STRENGTH:-2.0}"   # prior shape c0 of b0 ~ Gamma(c0, c0/b0): weak
SHS_CONSOLIDATE_EVERY="${SHS_CONSOLIDATE_EVERY:-50}"  # tier-2 frozen-rep Hughes consolidation
                                            # every N completed episodes (0 = curriculum-only)
SHS_RSTICK_STOPGRAD="${SHS_RSTICK_STOPGRAD:-True}"    # False = one joint objective through the
                                            # recurrent gate (paper claim); changes training
                                            # dynamics -- flip deliberately, as an A/B
# Per-seed LOGDIR: two jobs sharing a logdir would silently continue each
# other's checkpoints.
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/dmc_humanoid_shs/seed${SEED}}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-600}"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a
module load opencv/4.10.0
module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow

source "${VENV_DIR}/bin/activate"

unset CUDA_LAUNCH_BLOCKING
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
cd "${PROJECT_DIR}"

# Environment gate: catches a venv built for another cluster (import/ABI
# failure) or a torch without sm_90 kernels (matmul failure) BEFORE queueing
# multi-day compute.
python -X faulthandler - <<'PY'
import pathlib
import torch
import shs_rssm

print("SHS package:", pathlib.Path(shs_rssm.__file__).resolve(), flush=True)
print("SHS version:", getattr(shs_rssm, "__version__", "unset"), flush=True)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available -- rebuild the venv on Rorqual "
                       "(see header) and check the module stack")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
x = torch.randn(1024, 1024, device="cuda")
print("GPU matmul:", float((x @ x).mean()), flush=True)
PY

# q_rank>0 together with action_dim>0 is the combination that crashed before
# the Sfr=self.Lr fix. Verify the deployed tree has it BEFORE spending GPU-days.
python - <<'PY'
import torch
from shs_rssm.regimes_shared import SharedCarryRegimes
r = SharedCarryRegimes(K=16, L=32, G=32 + 2 + 21 + 1, q_rank=2, action_dim=21,
                       device="cpu", dtype=torch.float32)
assert r.Sfr.shape[-1] == r.Lr, (
    f"Sfr width {r.Sfr.shape[-1]} != Lr {r.Lr} -- this checkout PREDATES the Sfr fix; "
    "q_rank>0 with action_dim>0 will crash. Deploy the fixed regimes_shared.py first.")
print(f"Sfr/Lr check OK (Lr={r.Lr})", flush=True)
PY

python - <<'PY'
import pathlib
import torch
from shs_rssm.regime_head import RegimeHead

h = RegimeHead(stoch=8, deter=4, K=4, proj_dim=2, device="cpu",
               learn_b0=True, b0_strength=2.0)
r = h.regimes
assert hasattr(r, "b0_chat") and hasattr(r, "b0_dhat"), \
    "this checkout PREDATES the noise-rate hierarchy (learn_b0)"
assert torch.allclose(r.b0_chat / r.b0_dhat, r.b0), \
    "hierarchy must initialise at the prior mean (== configured fixed b0)"
src = pathlib.Path("dreamer.py").read_text()
assert "_NEW_KEY_SUFFIXES" in src and "b0_chat" in src, \
    "dreamer.py lacks the generalized legacy-checkpoint backfill; resuming an " \
    "old checkpoint with --shs_learn_b0 True would fail its strict load"
print("learn_b0 + checkpoint-backfill gate OK", flush=True)
PY

# A periodic timestamp distinguishes a genuinely running but quiet
# consolidation from a dead scheduler job; the GPU-memory columns double as
# the high-water-mark record for batch-size probes. Does not alter training.
heartbeat() {
  while true; do
    sleep "${HEARTBEAT_SECONDS}"
    echo "[heartbeat] $(date --iso-8601=seconds) logdir=${LOGDIR}" >&2
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits >&2 || true
  done
}
heartbeat &
HEARTBEAT_PID=$!
trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Starting Humanoid: seed=${SEED} steps=${STEPS} batch=${BATCH_SIZE}x${BATCH_LENGTH} precision=${PRECISION} train_ratio=${TRAIN_RATIO} q_rank=${SHS_Q_RANK} action_dim=${SHS_ACTION_DIM} learn_b0=${SHS_LEARN_B0} consolidate_every=${SHS_CONSOLIDATE_EVERY} rstick_stopgrad=${SHS_RSTICK_STOPGRAD} logdir=${LOGDIR}"

python -u dreamer.py \
  --configs dmc_humanoid_shs \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --compile False \
  --precision "${PRECISION}" \
  --train_ratio "${TRAIN_RATIO}" \
  --shs_shared_carry True \
  --shs_q_rank "${SHS_Q_RANK}" \
  --shs_action_dim "${SHS_ACTION_DIM}" \
  --shs_learn_b0 "${SHS_LEARN_B0}" \
  --shs_b0_strength "${SHS_B0_STRENGTH}" \
  --shs_consolidate_every_episodes "${SHS_CONSOLIDATE_EVERY}" \
  --shs_rstick_stopgrad "${SHS_RSTICK_STOPGRAD}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_length "${BATCH_LENGTH}" \
  --logdir "${LOGDIR}"