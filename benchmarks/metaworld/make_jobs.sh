#!/usr/bin/env bash
set -euo pipefail

# Generate one command per line for sbatch_array.sh.
#
# Examples:
#   bash benchmarks/metaworld/make_jobs.sh proprio > jobs_proprio.txt
#   bash benchmarks/metaworld/make_jobs.sh vision  > jobs_vision.txt
#   sbatch --array=1-24%8 benchmarks/metaworld/sbatch_array.sh jobs_proprio.txt
#   sbatch --export=ALL,MUJOCO_GL=egl --array=1-24%8 \
#     benchmarks/metaworld/sbatch_array.sh jobs_vision.txt
#
# Override the defaults with environment variables, for example:
#   TASKS="door-open reach hammer" SEEDS="0 1 2 3 4" STEPS=1000000 \
#     bash benchmarks/metaworld/make_jobs.sh vision > jobs_vision.txt
#
# COMMON_FLAGS is appended to every command. SHS_FLAGS is appended only to SHS
# commands. Their values are emitted verbatim into the generated job lines.
# Vision jobs require a working EGL or OSMesa renderer on the compute node.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {proprio|vision}" >&2
  exit 2
fi

MODE=$1
TASKS=${TASKS:-"door-open reach drawer-open button-press-topdown"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-500000}
EXPERIMENT=${EXPERIMENT:-latent_gate_v1}
LOGROOT=${LOGROOT:-"./logdir/${EXPERIMENT}"}
GATE_INPUT=${GATE_INPUT:-latent}
GATE_INNER_ITERS=${GATE_INNER_ITERS:-1}
COMMON_FLAGS=${COMMON_FLAGS:-}
SHS_FLAGS=${SHS_FLAGS:-}

case "$MODE" in
  proprio)
    BASE_CONFIG=metaworld_proprio
    SHS_CONFIG=metaworld_proprio_shs
    LOG_SUFFIX=
    ;;
  vision)
    BASE_CONFIG=metaworld_vision
    SHS_CONFIG=metaworld_vision_shs
    LOG_SUFFIX=_vision
    ;;
  *)
    echo "Unknown input mode '$MODE'. Expected 'proprio' or 'vision'." >&2
    exit 2
    ;;
esac

case "$GATE_INPUT" in
  latent|carry) ;;
  *)
    echo "Unknown GATE_INPUT '$GATE_INPUT'. Expected 'latent' or 'carry'." >&2
    exit 2
    ;;
esac

job_count=0
for task in $TASKS; do
  for seed in $SEEDS; do
    base_cmd="python3 -u dreamer.py --configs ${BASE_CONFIG} --task metaworld_${task} --seed ${seed} --steps ${STEPS} --logdir ${LOGROOT}/baseline${LOG_SUFFIX}/${task}/seed${seed}"
    shs_cmd="python3 -u dreamer.py --configs ${SHS_CONFIG} --task metaworld_${task} --seed ${seed} --steps ${STEPS} --shs_gate_input ${GATE_INPUT} --shs_gate_inner_iters ${GATE_INNER_ITERS} --logdir ${LOGROOT}/shs_${GATE_INPUT}${LOG_SUFFIX}/${task}/seed${seed}"

    if [[ -n "$COMMON_FLAGS" ]]; then
      base_cmd+=" ${COMMON_FLAGS}"
      shs_cmd+=" ${COMMON_FLAGS}"
    fi
    if [[ -n "$SHS_FLAGS" ]]; then
      shs_cmd+=" ${SHS_FLAGS}"
    fi

    printf '%s\n' "$base_cmd"
    printf '%s\n' "$shs_cmd"
    job_count=$((job_count + 2))
  done
done

printf '# Generated %d %s jobs for tasks: %s; seeds: %s\n' \
  "$job_count" "$MODE" "$TASKS" "$SEEDS" >&2
