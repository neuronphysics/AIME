#!/bin/bash
# Build an ISOLATED environment for the rSLDS baseline.  D2E is never written
# to: this script only ever *reads* it, in step 4, to generate the dataset
# cache (no pip command in this file targets D2E).  A pip freeze of D2E is
# taken before and after and compared at the end to prove it.
#
# Run on a LOGIN node (ssm comes from github; compute nodes have no network):
#
#   cd /home/memole/scratch/AIME
#   bash install_rslds_venv.sh
#
# Eleven checked stages; any failure stops immediately and says what broke and
# what to do.  Nothing is left half-installed without you being told.

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
D2E="${VENV_DIR:-/home/memole/D2E}"
RSLDS_VENV="${RSLDS_VENV:-$HOME/scratch/rslds-venv}"
COMPARE_DIR="${COMPARE_DIR:-${PROJECT_DIR}/shs_demo/compare}"
LOGDIR="${PROJECT_DIR}/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
SSM_URL="${SSM_URL:-git+https://github.com/lindermanlab/ssm.git}"

TOTAL=11
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "=== [${STEP}/${TOTAL}] $* ==="; }
die()  { echo; echo "FAILED at step ${STEP}/${TOTAL}: $1" >&2; shift
         while [ $# -gt 0 ]; do echo "  $1" >&2; shift; done
         echo >&2; echo "Nothing after this point ran.  D2E is unchanged." >&2
         exit 1; }

mkdir -p "${LOGDIR}" || die "cannot create ${LOGDIR}"

# ---------------------------------------------------------------------------
step "login node + network"
[ -z "${SLURM_JOB_ID:-}" ] || die \
  "running inside Slurm job ${SLURM_JOB_ID}" \
  "Compute nodes have no outbound network and ssm comes from github." \
  "Run on a login node:  bash install_rslds_venv.sh"
timeout 20 git ls-remote https://github.com/lindermanlab/ssm.git HEAD >/dev/null 2>&1 \
  || die "cannot reach github.com" \
         "Test by hand:  git ls-remote https://github.com/lindermanlab/ssm.git HEAD"
echo "login node, github reachable"

# ---------------------------------------------------------------------------
step "module stack"
module --force purge || true
for m in StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2024a; do
  module load "$m" || die "module load $m failed" \
      "Run 'module spider ${m%%/*}' and update this script to an available version."
done
python -V 2>&1 | grep -q "3\.11" \
  || die "expected python 3.11, got $(python -V 2>&1)"
echo "$(python -V 2>&1)"

# ---------------------------------------------------------------------------
step "check the fhn cache patch is applied"
[ -d "${COMPARE_DIR}" ] || die "no compare directory at ${COMPARE_DIR}"
cd "${COMPARE_DIR}" || die "cannot cd to ${COMPARE_DIR}"
grep -q 'fhn_N' datasets.py \
  || die "datasets.py does not cache fhn to disk" \
         "load_fhn() calls fhn_demo, which imports torch -- absent from the" \
         "isolated venv by design.  Apply the patch first:" \
         "    git apply fhn_cache.patch" \
         "Then rerun this script."
for f in run_rslds_ssm.py datasets.py io_utils.py metrics.py; do
  [ -f "$f" ] || die "missing ${f} in ${COMPARE_DIR}" \
                     "All runners must share this directory so they write to" \
                     "the same results/ tree."
done
echo "harness OK in $(pwd)"

# ---------------------------------------------------------------------------
step "generate the dataset cache using D2E (READ-ONLY: no pip runs here)"
[ -f "${D2E}/bin/activate" ] || die "no venv at ${D2E}"
D2E_BEFORE="${LOGDIR}/d2e-freeze-before-${STAMP}.txt"
(
  source "${D2E}/bin/activate"
  pip freeze > "${D2E_BEFORE}" 2>/dev/null
  python - <<'PY'
import sys
sys.path.insert(0, ".")
import datasets
ok = True
for name, kw in [("fhn", dict(n_seq=6)), ("nascar", dict(n_seq=5, seed=0)),
                 ("toyark13", dict(n_seq=12)), ("mocap6", {})]:
    try:
        b = datasets.load(name, **kw)
        n = sum(len(s) for s in b["seqs"])
        print(f"  {name:10s} seqs={len(b['seqs']):3d} steps={n:6d} "
              f"D={b['D']} K_true={b['K_true']}")
    except Exception as exc:
        print(f"  {name:10s} FAILED: {exc}")
        ok = False
sys.exit(0 if ok else 1)
PY
) || die "dataset caching failed under D2E" \
        "The rSLDS venv cannot regenerate these itself (no torch)." \
        "Fix the loader error above, then rerun."
ls data_cache/fhn_*.npz >/dev/null 2>&1 \
  || die "no data_cache/fhn_*.npz was written" \
         "The patch check passed but nothing landed; check write permissions" \
         "on ${COMPARE_DIR}/data_cache"
echo "cache:"; ls -la data_cache/ | tail -n +2

# ---------------------------------------------------------------------------
step "create the isolated venv at ${RSLDS_VENV}"
if [ -d "${RSLDS_VENV}" ] && [ "${RECREATE:-0}" = "1" ]; then
  rm -rf "${RSLDS_VENV}" && echo "removed existing venv (RECREATE=1)"
fi
if [ ! -f "${RSLDS_VENV}/bin/activate" ]; then
  virtualenv --no-download "${RSLDS_VENV}" \
    || die "virtualenv failed at ${RSLDS_VENV}" \
           "Check the path is writable and you have quota:  diskusage_report"
else
  echo "reusing existing venv (RECREATE=1 to rebuild from scratch)"
fi
source "${RSLDS_VENV}/bin/activate" || die "cannot activate ${RSLDS_VENV}"
# Compare resolved paths: $HOME/scratch is usually a symlink, so the literal
# strings differ even when the interpreter is the right one.
[ "$(readlink -f "$(command -v python)")" = "$(readlink -f "${RSLDS_VENV}/bin/python")" ] \
  || die "python resolves to $(command -v python), not ${RSLDS_VENV}/bin/python" \
         "Something later on PATH is shadowing the venv; check module order."
echo "python: $(command -v python)"

# ---------------------------------------------------------------------------
step "dependencies (wheelhouse, pinned for ssm)"
pip install --no-index --upgrade pip >/dev/null 2>&1 || true
# numpy is pinned <2 here because ssm predates numpy 2 and uses aliases it
# removed.  This venv is separate precisely so that pin cannot reach D2E/torch.
DEPS=('numpy<2' 'scipy<1.14' 'cython<3' autograd tqdm numba scikit-learn matplotlib seaborn)
for d in "${DEPS[@]}"; do
  echo "--- ${d}"
  pip install --no-index "${d}" \
    || die "pip install --no-index ${d} failed" \
           "If the wheelhouse has no matching build, install that ONE package" \
           "from PyPI by hand (network is available on this node) and rerun."
done
python -c "import numpy,scipy,sklearn,Cython,autograd; print(f'numpy {numpy.__version__} scipy {scipy.__version__}')" \
  || die "dependencies did not import after install" "Run: pip check"

# ---------------------------------------------------------------------------
step "install ssm from source"
pip install --no-build-isolation --no-cache-dir "${SSM_URL}" \
  || die "building ssm failed" \
         "It compiles messages.pyx and cstats.pyx.  Usual causes:" \
         "  * cython 3        -> pip install 'cython<3' and rerun" \
         "  * no compiler     -> module load gcc/12.3" \
         "  * numpy headers   -> --no-build-isolation needs numpy installed" \
         "The first 'error:' line in the log above is the real one."

# ---------------------------------------------------------------------------
step "verify ssm works"
python - <<'PY' || die "ssm installed but does not work" \
    "If the traceback names a removed numpy alias, pin numpy lower:" \
    "    pip install --no-index 'numpy<1.24' && pip install --no-build-isolation --force-reinstall ${SSM_URL}"
import ssm
print("  ssm version:", getattr(ssm, "__version__", "unknown"))
for t in ("recurrent", "recurrent_only"):
    ssm.SLDS(4, 3, 2, transitions=t, dynamics="gaussian",
             emissions="gaussian_orthog", single_subspace=True)
    print(f"  transitions={t} OK")
PY

# ---------------------------------------------------------------------------
step "end-to-end smoke fit in the isolated venv"
python run_rslds_ssm.py --dataset nascar --nseq 1 --iters 5 --seed 0 \
    --tag _smoke_rslds \
  || die "smoke fit failed" \
         "This is exactly the path run_rslds_job.sh takes." \
         "Rerun the command above to see the full traceback."
python - <<'PY' || die "smoke result does not match the shared schema"
import numpy as np
d = np.load("results/nascar/_smoke_rslds.npz", allow_pickle=True)
need = {"model", "dataset", "z_pred", "doc_range", "K_used", "wall_time", "params"}
assert not (need - set(d.files)), f"missing {need - set(d.files)}"
print(f"  schema OK | K_used={int(d['K_used'])} wall={float(d['wall_time']):.1f}s")
PY
rm -f results/nascar/_smoke_rslds.npz
# fhn must load WITHOUT torch in this venv -- that is the whole point.
python -c "
import sys; sys.path.insert(0,'.')
assert 'torch' not in sys.modules
import datasets; b = datasets.load('fhn', n_seq=6)
assert 'torch' not in sys.modules, 'fhn still pulled in torch'
print(f'  fhn loads from cache without torch: {len(b[\"seqs\"])} seqs, K_true={b[\"K_true\"]}')
" || die "fhn does not load in the isolated venv" \
        "Delete data_cache/fhn_*.npz, rerun step 4 under D2E, and try again."
deactivate

# ---------------------------------------------------------------------------
step "confirm D2E is untouched"
D2E_AFTER="${LOGDIR}/d2e-freeze-after-${STAMP}.txt"
(
  source "${D2E}/bin/activate"
  pip freeze > "${D2E_AFTER}" 2>/dev/null
  python -c "import numpy,scipy,torch;print(f'  numpy {numpy.__version__} scipy {scipy.__version__} torch {torch.__version__}')"
) || die "D2E is broken" "That should be impossible here; compare ${D2E_BEFORE} and ${D2E_AFTER}."
if diff -q "${D2E_BEFORE}" "${D2E_AFTER}" >/dev/null; then
  echo "  D2E package list identical before and after"
else
  echo "  D2E CHANGED -- this script does not install there, investigate:" >&2
  diff "${D2E_BEFORE}" "${D2E_AFTER}" | sed 's/^/    /' >&2
fi

# ---------------------------------------------------------------------------
step "done"
cat <<EOF

Two environments, cleanly separated:
  ${D2E}   SHS + TrSLDS + all figures   (untouched)
  ${RSLDS_VENV}   rSLDS only, numpy<2 + ssm, no torch

Next:
    cd ${COMPARE_DIR}
    sbatch run_rslds_job.sh       # 4 datasets x 3 seeds -> results/<d>/rslds_seed<s>.npz
    bash  refresh_figures.sh      # tables + offline_summary.pdf + comparison table

run_rslds_job.sh already defaults RSLDS_VENV to ${RSLDS_VENV}; override with
    RSLDS_VENV=/other/path sbatch run_rslds_job.sh
EOF