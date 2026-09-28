#!/bin/bash

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
COMPARE_DIR="${COMPARE_DIR:-${PROJECT_DIR}/shs_demo/compare}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-fhn nascar toyark13 mocap6}"
MODELS="${MODELS:-shs trslds rslds}"

module --force purge || true
module load StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2024a || exit 2
[ -f "${VENV_DIR}/bin/activate" ] || { echo "no venv at ${VENV_DIR}" >&2; exit 2; }
source "${VENV_DIR}/bin/activate"
export MPLBACKEND=Agg

cd "${COMPARE_DIR}" || { echo "cannot cd to ${COMPARE_DIR}" >&2; exit 2; }

echo "[inventory] results on disk"
for d in ${DATASETS}; do
  line="  ${d}:"
  for m in shs trslds rslds; do
    n=0
    for s in ${SEEDS}; do
      [ -f "results/${d}/${m}_seed${s}.npz" ] && n=$((n + 1))
    done
    if [ "${m}" = "shs" ] && [ "${n}" = "0" ]; then
      for s in ${SEEDS}; do
        [ -f "results/${d}/shsPersist_seed${s}.npz" ] && n=$((n + 1))
      done
    fi
    line="${line} ${m}=${n}"
  done
  echo "${line}"
done

echo
echo "[figures] per-dataset tables"
python make_figures.py --dataset all --latex \
  || echo "[figures] make_figures.py failed" >&2

echo "[figures] unified summary"
python summary_fig.py --datasets ${DATASETS} \
  --out results/offline_summary.pdf \
  || { echo "[figures] summary_fig.py failed" >&2; exit 1; }

echo "[figures] cross-dataset comparison table"
python compare_table.py --datasets ${DATASETS} --models ${MODELS} \
  --out results/comparison \
  || { echo "[figures] compare_table.py failed" >&2; exit 1; }

echo
echo "done. Deliverables for the paper:"
for f in results/offline_summary.pdf results/comparison.tex \
         results/comparison.md results/comparison.csv; do
  [ -f "$f" ] && ls -la "$f"
done
