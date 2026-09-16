#!/bin/bash
# Submit one 4-seed array per task, each with its own job name.
#   bash submit_all.sh
#   ARM=vanilla bash submit_all.sh
#   PILOT=1 bash submit_all.sh
set -euo pipefail

TASKS=(walker hopper cheetah acrobot pendulum finger_turn_hard quadruped_walk
       quadruped_run reacher_hard)
NSEEDS="${NSEEDS:-4}"

for t in "${TASKS[@]}"; do
  sbatch --job-name="DMC-${t}" \
         --array="0-$((NSEEDS-1))" \
         --export=ALL,TASK="${t}",NSEEDS="${NSEEDS}",ARM="${ARM:-shs}",PILOT="${PILOT:-0}",FLAGS="${FLAGS:-}" \
         run_dreamer_shs-rssm_dmc.sh
done