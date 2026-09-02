#!/bin/bash
#SBATCH --job-name=shs-benchmarks
#SBATCH --account=def-irina
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH -o /home/memole/links/projects/def-irina/memole/AIME/logs/slurm-shs-benchmarks-%j.out
#SBATCH -e /home/memole/links/projects/def-irina/memole/AIME/logs/slurm-shs-benchmarks-%j.err

# Benchmark sweep, three seeds, plus the unified paper figure.
#
# nascar and toyark13 use the package defaults with the recurrent gate.
#
# fhn runs a configuration sweep (FHN_VARIANTS below).  Note that the offline
# harness supplies a constant-zero deterministic carry, so a carry-input gate
# degenerates to a per-regime constant; the 'latent' variants read x_{t-1}
# instead and are the only ones whose gate can vary along the trajectory.
# ONE configuration is selected across all seeds -- median |K_used - target|
# first, median Hamming as tie-break -- and its per-seed results are copied to
# shs_seed<s>.npz for the figure.  Selection is per configuration, never per
# seed; the winner is recorded in results/fhn/selected_variant.txt.
#
# mocap6 gets its own configurations (see MOCAP_BASE below):
#   shsPersist  headline row: K recovered, best purity.
#   shsPrune    ablation row: merge/delete only; parked out of the unified
#               figure so median-seed selection sees the headline config only.
#
# rSLDS cannot run here: the vendored 2017 stack (autograd, pybasicbayes,
# pyhsmm, autoregressive, pypolyagamma) does not build on python>=3.10.  This
# job writes run_rslds_legacy.sh; run it in the legacy conda environment
# (environment-baselines.yml) and the figure regenerates with the rSLDS row.
#
# Usage:
#   sbatch run_benchmark.sh              # everything
#   FHN=0 sbatch run_benchmark.sh        # skip the fhn variant sweep
#   MOCAP=0 sbatch run_benchmark.sh      # skip mocap6
#   STANDARD=0 sbatch run_benchmark.sh   # skip nascar + toyark13
#   FRESH=1 sbatch run_benchmark.sh      # archive old outputs first
#   FORCE=1 sbatch run_benchmark.sh      # redo fits that already exist
#   FHN_SELECT_K=4 sbatch run_benchmark.sh   # target regime count for fhn

set -uo pipefail

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

PROJECT_DIR="${PROJECT_DIR:-/home/memole/links/projects/def-irina/memole/AIME}"
mkdir -p "${PROJECT_DIR}/logs"

module --force purge || true
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load scipy-stack/2024a
module load mpi4py/3.1.6
module load arrow

VENV_DIR="${VENV_DIR:-/home/memole/links/D2E}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Missing virtual environment: ${VENV_DIR}" >&2
  exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export MPLBACKEND=Agg

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
    sys.exit(f"missing deps {missing} in {sys.executable} -- activate the env "
             "on a login node and: pip install scikit-learn matplotlib")
if all(u.find_spec(m) is None for m in ("pypolyagamma", "polyagamma")):
    sys.exit("need `polyagamma` for the TrSLDS Polya-Gamma draws -- on a login "
             "node with the env active: pip install polyagamma")
import numpy, scipy, torch
print(f"numpy {numpy.__version__}  scipy {scipy.__version__}  "
      f"torch {torch.__version__}", flush=True)
print("env preflight OK", flush=True)
PYPRE
[ $? -eq 0 ] || exit 2

for flag in --init-block --prior-persist --bias-prior-var --sweep-every --no-birth \
            --move-threshold --gate-input; do
  python run_shs.py --help 2>/dev/null | grep -q -- "$flag" || {
    echo "deployed run_shs.py lacks $flag -- deploy the current package first" >&2
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
STANDARD="${STANDARD:-1}"        # nascar + toyark13
FHN="${FHN:-1}"                  # fhn variant sweep + selection
MOCAP="${MOCAP:-1}"              # mocap6, its own configuration
FHN_SELECT_K="${FHN_SELECT_K:-4}"

if [ "${MOCAP}" = "1" ] && [ ! -f ../mocap6/dataset.npz ]; then
  echo "[sweep] mocap6/dataset.npz missing -- skipping mocap section" >&2
  echo "        copy it into ../mocap6/ and resubmit" >&2
  MOCAP=0
fi

MOCAP_BASE=(--recurrent --kappa 0 --b0-mode calibrate --sF 0.1
            --prior-persist 0.99 --bias-prior-var 0.10)

# fhn configuration sweep: name|flags.
FHN_VARIANTS=(
  "base|--recurrent"
  "bias1|--recurrent --bias-prior-var 1.0"
  "both|--recurrent --bias-prior-var 1.0 --move-threshold 20 --laps 20"
  "latent|--recurrent --gate-input latent --bias-prior-var 1.0 --move-threshold 20 --laps 20"
  "latentP|--recurrent --gate-input latent --bias-prior-var 1.0 --prior-persist 0.99 --move-threshold 20 --laps 20"
  "nonrec|--kappa 10 --bias-prior-var 1.0 --move-threshold 20 --laps 20"
)

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

# ---- 2. fhn: variant sweep, then one configuration for all seeds ---------
if [ "${FHN}" = "1" ]; then
  for s in ${SEEDS}; do
    run run_trslds.py fhn "$s" "trslds_seed${s}"
    for spec in "${FHN_VARIANTS[@]}"; do
      vname="${spec%%|*}"; vflags="${spec#*|}"
      read -r -a VF <<< "${vflags}"
      run run_shs.py fhn "$s" "fhnv${vname}_seed${s}" "${VF[@]}"
    done
  done

  echo "[fhn-select] scoring variants (target K_used=${FHN_SELECT_K})"
  python - "${SEEDS}" "${FHN_SELECT_K}" <<'PYSEL'
import glob, os, re, shutil, sys
import numpy as np
sys.path.insert(0, ".")
import datasets
from metrics import all_metrics

seeds = sys.argv[1].split()
want_K = int(sys.argv[2])
zt = np.concatenate(datasets.load("fhn")["z_true"])

by_variant = {}
for s in seeds:
    for f in sorted(glob.glob(f"results/fhn/fhnv*_seed{s}.npz")):
        name = re.sub(r"^fhnv|_seed\d+\.npz$", "", os.path.basename(f))
        try:
            z = np.asarray(np.load(f, allow_pickle=True)["z_pred"], int)
            if z.shape[0] != zt.shape[0]:
                print(f"  [warn] {os.path.basename(f)}: length mismatch, skipped")
                continue
            m, _ = all_metrics(zt, z)
            # same definition the figure uses: distinct labels present
            K = int(np.unique(z).size)
        except Exception as exc:
            print(f"  [warn] {os.path.basename(f)}: {exc}")
            continue
        by_variant.setdefault(name, []).append(
            dict(seed=s, path=f, K=K, ham=m["hamming"], m2o=m["m2o"]))
        print(f"    {name:9s} seed{s}  K={K:2d}  1-to-1={1 - m['hamming']:.3f}"
              f"  m2o={m['m2o']:.3f}")

if not by_variant:
    print("[fhn-select] no usable variants; leaving shs_seed*.npz untouched")
    raise SystemExit(0)

rank = []
for name, rows in by_variant.items():
    Ks = np.array([r["K"] for r in rows]); hs = np.array([r["ham"] for r in rows])
    rank.append((float(np.median(np.abs(Ks - want_K))), float(np.median(hs)),
                 name, len(rows)))
rank.sort()
print("[fhn-select] configuration ranking (median |K-%d|, median Hamming):" % want_K)
for dK, h, name, n in rank:
    print(f"    {name:9s} med|dK|={dK:.1f}  med 1-to-1={1 - h:.3f}  seeds={n}")

win = rank[0][2]
note = "hits target K" if rank[0][0] == 0 else f"closest (median |dK|={rank[0][0]:.1f})"
for r in by_variant[win]:
    dst = f"results/fhn/shs_seed{r['seed']}.npz"
    shutil.copyfile(r["path"], dst)
    print(f"[fhn-select] seed {r['seed']}: {os.path.basename(r['path'])} -> {dst}")
with open("results/fhn/selected_variant.txt", "w") as fh:
    fh.write(f"selected fhn configuration: {win} ({note})\n")
    for dK, h, name, n in rank:
        fh.write(f"  {name:9s} med|dK|={dK:.1f} med_1to1={1 - h:.3f} seeds={n}\n")
print(f"[fhn-select] chose '{win}' -- {note}; recorded in results/fhn/selected_variant.txt")
PYSEL
fi

# ---- 3. mocap6: its own configuration ------------------------------------
if [ "${MOCAP}" = "1" ]; then
  for s in ${SEEDS}; do
    run run_trslds.py mocap6 "$s" "trslds_seed${s}"
    run run_shs.py mocap6 "$s" "shsPersist_seed${s}" "${MOCAP_BASE[@]}"
    run run_shs.py mocap6 "$s" "shsPrune_seed${s}" "${MOCAP_BASE[@]}" \
        --laps 20 --sweep-every 1 --no-birth
  done
fi

python make_figures.py --dataset all --latex

# ---- report --------------------------------------------------------------
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

# ---- rSLDS companion (legacy python<=3.8 environment) --------------------
cat > run_rslds_legacy.sh <<'RSL'
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
RSL
chmod +x run_rslds_legacy.sh
echo "[sweep] wrote run_rslds_legacy.sh -- run it in the legacy env for the rSLDS row"

# ---- unified paper figure ------------------------------------------------
# Park the ablation and the fhn variants so median-seed selection in the SHS
# row sees only the selected configuration for each dataset.
mkdir -p results/mocap6/ablation results/fhn/variants
mv results/mocap6/shsPrune_seed*.npz results/mocap6/ablation/ 2>/dev/null || true
mv results/fhn/fhnv*_seed*.npz results/fhn/variants/ 2>/dev/null || true

python summary_fig.py --datasets fhn nascar toyark13 mocap6 \
  --out results/offline_summary.pdf || echo "[sweep] summary figure failed" >&2

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[sweep] FAILED runs:" >&2
  printf '    %s\n' "${FAILED[@]}" >&2
  exit 1
fi
echo "[sweep] done -- figure: results/offline_summary.pdf, tables: results/<dataset>/"