#!/bin/bash
# rSLDS baseline. The vendored 2017 stack needs python<=3.8, so this cannot
# run in the same environment as the rest of the sweep.
#   conda env create -f environment-baselines.yml
#   conda activate shs-baselines
#   bash run_rslds_legacy.sh
# Writes results/<dataset>/rslds_seed<s>.npz and refreshes the unified figure,
# which picks the rSLDS row up automatically once those files exist.
set -uo pipefail
for d in fhn nascar toyark13 mocap6; do
  for s in 0 1 2; do
    if [ -f "results/$d/rslds_seed${s}.npz" ]; then
      echo "[rslds] skip $d/seed$s (exists)"; continue
    fi
    echo "[rslds] $d seed $s"
    python run_rslds.py --dataset "$d" --seed "$s" --tag "rslds_seed${s}" || true
  done
done
python summary_fig.py --datasets fhn nascar toyark13 mocap6 \
  --out results/offline_summary.pdf
echo "[rslds] done -- figure refreshed: results/offline_summary.pdf"
