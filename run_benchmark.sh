#!/bin/bash
#SBATCH --job-name=shs-benchmarks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/slurm-shs-benchmarks-%j.out
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/slurm-shs-benchmarks-%j.err

# Benchmark sweep, three seeds, 21 fits.
#
# nascar and toyark13 use the package defaults with the recurrent gate; that
# configuration already wins on both (Hamming 0.005 and 0.085 against TrSLDS's
# 0.070 and 0.588), so it is left alone.
#
# mocap6 needs its own configuration and gets its own block.  Twelve annotated
# behaviours last tens of frames each; the defaults find 5-6 states switching
# every 3 frames.  Two configurations, chosen from the completed sweeps:
#
#   shsPersist  gate prior 0.99 / 0.10 over a calibrated tight noise prior.
#               Recovers K = 12/12/11 with the best purity (m2o 0.658,
#               NMI 0.542).  Headline row: the count is inferred, not imposed.
#   shsPrune    the same, plus 20 laps of merge/delete with NO births.
#               Best one-to-one Hamming anywhere (0.549, above TrSLDS's 0.532)
#               and by far the longest segments (158 vs ~300, median 6f), but
#               covers the 12 behaviours with 7-8 states.  Ablation row: what
#               pure pruning buys, and what it costs.
#
# Everything else tried is dropped: the 0.98 gate-prior variants (dominated),
# shsK12 (superseded by shsPersist), and the acceptance-margin grid (a
# symmetric margin blocks merges before births, so K rose to 17-21).
#
# Usage:
#   sbatch run_benchmark.sh              # everything
#   MOCAP=0 sbatch run_benchmark.sh      # nascar + toyark13 only
#   STANDARD=0 sbatch run_benchmark.sh   # mocap6 only
#   FRESH=1 sbatch run_benchmark.sh      # archive old outputs first
#   FORCE=1 sbatch run_benchmark.sh      # redo fits that already exist

set -uo pipefail

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
mkdir -p "${PROJECT_DIR}/logs"

module unload python
module load anaconda/3
conda activate trifinger_rl_venv
module load gcc/9.3.0
module unload anaconda
module load python/3.10
unset CUDA_LAUNCH_BLOCKING

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- locate the harness --------------------------------------------------
if [ -n "${COMPARE_DIR:-}" ]; then
  cd "${COMPARE_DIR}"
elif [ -d "${PROJECT_DIR}/shs_demo/compare" ]; then
  cd "${PROJECT_DIR}/shs_demo/compare"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "$(dirname "$0")"
fi
[ -f run_shs.py ] || { [ -f shs_demo/compare/run_shs.py ] && cd shs_demo/compare; }
if [ ! -f run_shs.py ]; then
  echo "run_shs.py not found under $(pwd)" >&2
  echo "set COMPARE_DIR=/path/to/AIME/shs_demo/compare" >&2
  exit 2
fi
echo "harness: $(pwd)"

# ---- preflight -----------------------------------------------------------
python - <<'PYPRE'
import importlib.util as u, sys
print("python:", sys.executable, flush=True)
missing = [m for m in ("torch", "numpy", "scipy", "sklearn", "matplotlib")
           if u.find_spec(m) is None]
if missing:
    sys.exit(f"missing deps {missing} in {sys.executable} -- activate the conda "
             "env on a login node and: pip install scikit-learn matplotlib")
if all(u.find_spec(m) is None for m in ("pypolyagamma", "polyagamma")):
    sys.exit("need `polyagamma` for the TrSLDS Polya-Gamma draws -- on a login "
             "node with the env active: pip install polyagamma")
import numpy, scipy, torch
print(f"numpy {numpy.__version__}  scipy {scipy.__version__}  "
      f"torch {torch.__version__}", flush=True)
print("env preflight OK", flush=True)
PYPRE
[ $? -eq 0 ] || exit 2

for flag in --init-block --prior-persist --bias-prior-var --sweep-every --no-birth; do
  python run_shs.py --help 2>/dev/null | grep -q -- "$flag" || {
    echo "deployed run_shs.py lacks $flag -- deploy the current run_shs.py" >&2
    echo "(and the regime_head.py that forwards rstick_bias_var) first" >&2
    exit 2
  }
done

FRESH="${FRESH:-0}"
if [ "${FRESH}" = "1" ]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  for d in results figures data_cache; do
    if [ -d "$d" ]; then
      mv "$d" "${d}_backup_${STAMP}"
      echo "[fresh] archived $d -> ${d}_backup_${STAMP}" >&2
    fi
  done
fi

SEEDS="${SEEDS:-0 1 2}"
STANDARD_DATASETS="${STANDARD_DATASETS:-toyark13 nascar}"
QUICK="${QUICK:-0}"
EXTRA="${EXTRA:-}"
STANDARD="${STANDARD:-1}"   # nascar + toyark13
MOCAP="${MOCAP:-1}"         # mocap6, its own configuration

# mocap6 base: recurrent gate over a calibrated tight noise prior, on the
# contiguous 100-frame init that is the dataset default inside run_shs.py.
# --kappa 0 is documentation only; regime_head zeroes the sticky mass whenever
# the gate is on.
MOCAP_BASE=(--recurrent --kappa 0 --b0-mode calibrate --sF 0.1
            --prior-persist 0.99 --bias-prior-var 0.10)

Q=(); [ "${QUICK}" = "1" ] && Q=(--quick)
read -r -a EX <<< "${EXTRA}"

FAILED=()
run() {  # run <script> <dataset> <seed> <tag> [flags...] -- skip done, keep going
  if [ -f "results/$2/$4.npz" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[sweep] skip $2/$4 (result exists; FORCE=1 to redo)" >&2
    return 0
  fi
  echo "[sweep] $1 dataset=$2 seed=$3 tag=$4 ${*:5}" >&2
  python "$1" --dataset "$2" --seed "$3" --tag "$4" "${@:5}" "${Q[@]}" "${EX[@]}" \
    || FAILED+=("$1:$2:$3")
}

# ---- 1. nascar and toyark13: package defaults + recurrent gate -----------
if [ "${STANDARD}" = "1" ]; then
  for s in ${SEEDS}; do
    for d in ${STANDARD_DATASETS}; do
      run run_shs.py    "$d" "$s" "shs_seed${s}" --recurrent
      run run_trslds.py "$d" "$s" "trslds_seed${s}"
    done
  done
fi

# ---- 2. mocap6: its own configuration ------------------------------------
if [ "${MOCAP}" = "1" ]; then
  for s in ${SEEDS}; do
    run run_trslds.py mocap6 "$s" "trslds_seed${s}"

    # headline: recovers K = 12, best purity
    run run_shs.py mocap6 "$s" "shsPersist_seed${s}" "${MOCAP_BASE[@]}"

    # ablation: pure pruning -- longest segments, best one-to-one, fewer states
    run run_shs.py mocap6 "$s" "shsPrune_seed${s}" "${MOCAP_BASE[@]}" \
        --laps 20 --sweep-every 1 --no-birth
  done
fi

python make_figures.py --dataset all --latex

# ---- report --------------------------------------------------------------
# top-pair share = fraction of all switches carried by the single most frequent
# state pair.  High means near-duplicate states alternating; an even spread
# means genuinely fast dynamics.
echo "[report] mocap6 -- truth: K=12, 37 segments, median 51 frames, top pair ~5%"
for f in results/mocap6/*.npz; do
  [ -f "$f" ] || continue
  python - "$f" <<'PYREP'
import sys
import numpy as np
sys.path.insert(0, ".")
import datasets
from metrics import all_metrics

zt = np.concatenate(datasets.load("mocap6")["z_true"])
r = np.load(sys.argv[1], allow_pickle=True)
z = np.asarray(r["z_pred"], int)
m, _ = all_metrics(zt, z)
occ = np.bincount(z, minlength=int(z.max()) + 1) / len(z)
seg = np.diff(np.flatnonzero(np.r_[True, z[1:] != z[:-1], True]))

pairs = {}
for i in np.flatnonzero(z[1:] != z[:-1]):
    key = tuple(sorted((int(z[i]), int(z[i + 1]))))
    pairs[key] = pairs.get(key, 0) + 1
total = sum(pairs.values()) or 1
top_n = max(pairs.values()) if pairs else 0

print(f"  {sys.argv[1].split('/')[-1]:26s} K={int((occ > 0.005).sum()):2d} "
      f"segs={len(seg):4d} med={np.median(seg):3.0f}f "
      f"short(<5f)={100 * (seg < 5).mean():3.0f}% "
      f"top-pair={100 * top_n / total:3.0f}% "
      f"m2o={m['m2o']:.3f} 1-to-1={1 - m['hamming']:.3f}")
PYREP
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[sweep] FAILED runs:" >&2
  printf '    %s\n' "${FAILED[@]}" >&2
  exit 1
fi
echo "[sweep] done -- tables in results/<dataset>/, figures in figures/"

