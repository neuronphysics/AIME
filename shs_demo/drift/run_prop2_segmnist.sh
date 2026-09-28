#!/bin/bash
#SBATCH --job-name=AIME-VISMNIST-P2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=aip-irina
#SBATCH --output=/home/memole/scratch/AIME/logs/vismnist_p2_%j.out
#SBATCH --error=/home/memole/scratch/AIME/logs/vismnist_p2_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
OUT="${OUT:-${PROJECT_DIR}/shs_demo/drift/results_visual_mnist_prop2}"
MNIST_ROOT="${MNIST_ROOT:-${PROJECT_DIR}/data}"
FLAGS="${FLAGS:-}"

module --force purge || true
module load StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2024a
source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUT}"

echo "Date: $(date)  Host: $(hostname)  Out: ${OUT}"

python - <<PY
from torchvision.datasets import MNIST
from pathlib import Path
root = Path(r"${MNIST_ROOT}")
try:
    MNIST(str(root), train=True, download=False)
except Exception as e:
    raise SystemExit(
        f"MNIST is not cached at {root}. Before sbatch, run on a node with internet:\n"
        f"python -c \"from torchvision.datasets import MNIST; MNIST(r'{root}', train=True, download=True)\""
    ) from e
print("MNIST cache OK:", root)
PY

python -u shs_demo/drift/prop2_segmentation_moving_mnist.py \
  --source-root "${PROJECT_DIR}" \
  --mnist-root "${MNIST_ROOT}" \
  --out "${OUT}" \
  --stage run \
  --seeds 20 \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  ${FLAGS}

python -u shs_demo/drift/prop2_segmentation_moving_mnist.py \
  --source-root "${PROJECT_DIR}" \
  --mnist-root "${MNIST_ROOT}" \
  --out "${OUT}" \
  --stage plot \
  --seeds 20 \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  ${FLAGS}

python -u shs_demo/drift/prop2_segmentation_moving_mnist.py \
  --source-root "${PROJECT_DIR}" \
  --mnist-root "${MNIST_ROOT}" \
  --out "${OUT}" \
  --stage protocol \
  --seeds 20 \
  ${FLAGS}
