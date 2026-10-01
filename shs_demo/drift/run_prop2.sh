#!/bin/bash
#SBATCH --job-name=AIME-DRIFT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=aip-irina
#SBATCH --output=/home/memole/scratch/AIME/logs/drift_%j.out
#SBATCH --error=/home/memole/scratch/AIME/logs/drift_%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
OUT="${OUT:-${PROJECT_DIR}/shs_demo/drift/results_prop2_final}"
FLAGS="${FLAGS:-}"

module --force purge || true
module load StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2024a
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
echo "Date: $(date)  Host: $(hostname)  Out: ${OUT}"

python -u shs_demo/drift/prop2_experiment.py --out "${OUT}" --seeds 20 --workers "${SLURM_CPUS_PER_TASK:-16}" ${FLAGS}