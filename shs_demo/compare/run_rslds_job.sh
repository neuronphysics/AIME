#!/bin/bash
#SBATCH --job-name=rslds
#SBATCH --account=aip-irina
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-11%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:59:59
#SBATCH -o /home/memole/scratch/AIME/logs/slurm-rslds-%A_%a.out
#SBATCH -e /home/memole/scratch/AIME/logs/slurm-rslds-%A_%a.err
#SBATCH --open-mode=append


set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
COMPARE_DIR="${COMPARE_DIR:-${PROJECT_DIR}/shs_demo/compare}"
RSLDS_VENV="${RSLDS_VENV:-$HOME/scratch/rslds-venv}"

read -r -a DS <<< "${DATASETS:-fhn nascar toyark13 mocap6}"
SEEDS_PER="${SEEDS_PER:-3}"
NJOB=$(( ${#DS[@]} * SEEDS_PER ))

if (( SLURM_ARRAY_TASK_ID >= NJOB )); then
  echo "array index ${SLURM_ARRAY_TASK_ID} >= ${NJOB} (dataset,seed) pairs — nothing to do"
  exit 0
fi

DATASET=${DS[$(( SLURM_ARRAY_TASK_ID / SEEDS_PER ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % SEEDS_PER ))

scontrol update JobId="${SLURM_JOB_ID}" JobName="rslds_${DATASET}_s${SEED}" 2>/dev/null

echo "=== $(date) | rslds | ${DATASET} seed ${SEED} | node $(hostname) ==="
if (( SLURM_ARRAY_TASK_ID == 0 )) && (( NJOB > 12 )); then
  echo "NOTE: ${NJOB} pairs configured but --array covers 12; raise it to 0-$((NJOB - 1))" >&2
fi

module --force purge || true
module load StdEnv/2023 || exit 2
module load gcc/12.3    || exit 2
module load python/3.11 || exit 2
module load scipy-stack/2024a || exit 2

[ -f "${RSLDS_VENV}/bin/activate" ] || {
  echo "no venv at ${RSLDS_VENV} — run install_rslds_login.sh on a login node" >&2
  exit 2
}
source "${RSLDS_VENV}/bin/activate"

python -c "import ssm" 2>/dev/null || {
  echo "ssm not importable in ${RSLDS_VENV} — run install_rslds_venv.sh on a login node" >&2
  exit 2
}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export MPLBACKEND=Agg

cd "${COMPARE_DIR}" || { echo "cannot cd to ${COMPARE_DIR}" >&2; exit 2; }
[ -f run_rslds_ssm.py ] || { echo "run_rslds_ssm.py missing in $(pwd)" >&2; exit 2; }

if [ "${DATASET}" = "fhn" ] && ! ls data_cache/fhn_*.npz >/dev/null 2>&1; then
  echo "data_cache/fhn_*.npz missing — rerun install_rslds_venv.sh (step 4) on a login node" >&2
  exit 2
fi

case "${VARIANT:-rslds}" in
  rslds)            MODEL_TAG=rslds ;;
  rslds-ro)         MODEL_TAG=rslds_ro ;;
  sticky-rslds)     MODEL_TAG=rslds_sticky ;;
  sticky-rslds-ro)  MODEL_TAG=rslds_sticky_ro ;;
  *) echo "unknown VARIANT='${VARIANT}'; expected one of rslds, rslds-ro, sticky-rslds, sticky-rslds-ro" >&2
     exit 2 ;;
esac
TAG="${MODEL_TAG}_seed${SEED}"

ARGS=(--dataset "${DATASET}" --seed "${SEED}" --tag "${TAG}")
[ -n "${VARIANT:-}" ] && ARGS+=(--variant "${VARIANT}")
[ -n "${ITERS:-}" ]   && ARGS+=(--iters "${ITERS}")
[ -n "${DLATENT:-}" ] && ARGS+=(--dlatent "${DLATENT}")
[ -n "${KSTATES:-}" ] && ARGS+=(--K "${KSTATES}")
[ "${QUICK:-0}" = "1" ] && ARGS+=(--quick)

OUT="results/${DATASET}/${TAG}.npz"
if [ -f "${OUT}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "skip ${OUT} (exists; FORCE=1 to redo)"
  exit 0
fi

echo "python run_rslds_ssm.py ${ARGS[*]}"
python run_rslds_ssm.py "${ARGS[@]}"
STATUS=$?

if [ ${STATUS} -eq 0 ] && [ -f "${OUT}" ]; then
  echo "wrote ${COMPARE_DIR}/${OUT}"
else
  echo "no result written for ${DATASET} seed ${SEED} (status ${STATUS})" >&2
fi

echo "=== $(date) | ${DATASET} seed ${SEED} done | status=${STATUS} ==="
echo "when the whole array finishes:  bash refresh_figures.sh   (runs under D2E)"
exit ${STATUS}
