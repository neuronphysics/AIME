# CARL experiments on Mila

## What this branch runs

`dreamer.py` is the training entry point. Its CARL path constructs `envs/carl.py`,
normalizes actions, and adds the episode time limit. No demonstration dataset is
needed: training collects its own replay episodes.

The presets in `configs.yaml` train from 64×64 RGB images. Physics context is
selected at **episode reset**, not switched mid-episode. Training samples contexts
randomly; evaluation cycles through held-out contexts. `state` and `context` are
recorded in replay but excluded from the default encoder. The optional
`carl_context_visible` overlay adds context to the encoder, not privileged state.

One full array runs **5 context grids × 2 models × 5 seeds = 50 jobs**:

| SHS indices | Vanilla indices | SHS preset | Train contexts | Held-out evaluation contexts |
|---|---|---|---|---|
| 0–4 | 25–29 | `carl_walker_shs` | gravity: 4.905, 9.81, 19.62 | gravity: 2.4525, 39.24 |
| 5–9 | 30–34 | `carl_walker_gravity_friction_shs` | gravity above × tangential friction: 0.5, 1, 2 | held-out gravity above × friction: 0.25, 4 |
| 10–14 | 35–39 | `carl_quadruped_walk_shs` | gravity: 4.905, 9.81, 19.62 | gravity: 2.4525, 39.24 |
| 15–19 | 40–44 | `carl_quadruped_actuator_shs` | actuator strength: 0.5, 1, 2 | actuator strength: 0.25, 4 |
| 20–24 | 45–49 | `carl_finger_spin_shs` | gravity: 4.905, 9.81, 19.62 | gravity: 2.4525, 39.24 |

Within each five-index block, seeds are 1, 2, 3, 4, 5. Feature grids are Cartesian
products: the gravity/friction experiment has 9 training and 4 evaluation contexts;
the others have 3 and 2. The vanilla arm appends `carl_vanilla` to the same preset,
setting `use_shs=False` and `dyn_discrete=32` without changing its context grid.
The adapter also lists Fish, but this branch has no Fish experiment preset.

Production defaults are 1,000,000 environment steps, action repeat 2, four
environments, batch size 16, sequence length 64, train ratio 512, and 12 evaluation
episodes every 10,000 environment steps. The loop rounds to collection/evaluation
chunks, so the final step count can exceed the requested budget. Compilation is
disabled by the launcher. These are preset settings, not benchmarked runtime or
memory guarantees.

## 1. Prepare the cluster checkout and Python environment

Copy these uncommitted launcher/documentation changes to your cluster checkout, or
commit and push them yourself before updating that checkout. Switching an existing
cluster checkout to `carl` alone does not transfer local changes.

Create the environment on an allocated compute node, not as a training workload on
the login node. From the cluster's AIME checkout:

```bash
module load miniconda/3
conda create -n aime-carl python=3.10 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aime-carl
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-carl.txt
python -m pip check
python -m pip freeze > requirements-carl-resolved.txt
```

`requirements-carl.txt` is a CARL-only starting environment, not a complete lockfile.
It omits Procgen, Crafter, memory-maze, and the unrelated offline baselines. It
retains this branch's MuJoCo/dm_control pins, selects CARL 1.1.1, and uses NumPy 1.x
with the legacy Gym stack. PyTorch 2.4.1 is a conservative choice for this code,
including its legacy checkpoint loading. Validate the environment with the GPU
smoke job below before committing a full experiment budget; it has not been
end-to-end validated on Mila by this change.

If your environment already works, reuse it instead. The launcher accepts:

- `CONDA_ENV=aime-carl`: activates the named environment or Conda prefix. If Conda
  is not on PATH, loads `CONDA_MODULE` (default `miniconda/3`). Set
  `CONDA_MODULE=anaconda/3` if that is the module available to your account.
- `VENV_PATH=/absolute/path/to/venv`: activates a Python virtual environment.
- Neither variable: uses the already-active Python from the submission environment.

Set only one of `CONDA_ENV` and `VENV_PATH`. No packages are installed by the batch
jobs, and no Python module is loaded over an activated environment.

## 2. Inspect and smoke-test

Always submit from the repository root, or export an absolute `PROJECT_DIR`.
The main script defaults to `SLURM_SUBMIT_DIR` inside a job. Slurm output goes to
the submission directory without requiring a pre-existing `logs/bootstrap` path.

Inspect commands locally without Python dependencies, CUDA, or Slurm:

```bash
DRY_RUN=1 bash run_dreamer_carl_mila.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=25 bash run_dreamer_carl_mila.sh
DRY_RUN=1 bash run_carl_smoke_mila.sh
```

On Mila, run the short paired Walker check first:

```bash
export CONDA_ENV=aime-carl
sbatch run_carl_smoke_mila.sh
```

This submits indices 0 and 25 sequentially (SHS and vanilla, seed 1), each with one
L40S, four CPUs, 32 GB RAM, and a 30-minute limit on `unkillable`. The reduced run
uses 2,000 steps, one environment, smaller batches, a shorter episode limit,
minimal pretraining, and disabled SHS figures/video predictions. It still exercises
training, held-out evaluation, and checkpoint writing. Inspect for `latest.pt` and
`metrics.jsonl` under `logdir/smoke/<array-job-id>/<run>/seed_1/`, and check Slurm exit
status. The time limit is a starting allocation, not a guaranteed completion time.

To smoke-test every preset and model, one seed each:

```bash
sbatch --array=0,5,10,15,20,25,30,35,40,45%1 run_carl_smoke_mila.sh
```

Every job first checks CUDA and training imports, then resets, renders, and steps
through **all train and evaluation contexts of its preset** using round-robin
selection. It verifies image/action shapes, finite state/reward, context coverage,
and physical gravity where applicable. This preflight does not change the training
selector. For context/render checks without training:

```bash
PREFLIGHT_ONLY=1 sbatch --array=0,5,10,15,20%1 run_carl_smoke_mila.sh
```

Rendering defaults to EGL. A failed environment preflight retries with OSMesa;
OSMesa requires its system library and may be slower. Missing packages and CUDA
failures stop before the renderer retry. You can explicitly select
`MUJOCO_GL=osmesa`. For EGL device-selection errors, inspect `CUDA_VISIBLE_DEVICES`
and set `MUJOCO_EGL_DEVICE_ID` for the node's EGL device enumeration; GPU UUIDs are
not converted blindly to numeric device IDs.

## 3. Submit production runs

After the smoke jobs succeed, start with the five-seed Walker comparison:

```bash
export CONDA_ENV=aime-carl
sbatch --array=0-4,25-29%2 run_dreamer_carl_mila.sh
```

Or submit the full 50-run comparison:

```bash
sbatch run_dreamer_carl_mila.sh
```

The production allocation is one L40S, eight CPUs, 64 GB RAM, and 41 hours per job
on `long`, with at most two array tasks running concurrently. Adjust concurrency to
your allocation. `long` is preemptible. The smaller smoke allocation fits the
published `unkillable` limits; production's 64 GB does not. Resource policies can
change; consult [Mila's partition reference](https://docs.mila.quebec/technical_reference/clusters/mila/partitions/).

Additional examples:

```bash
sbatch --array=10 --time=2-00:00:00 run_dreamer_carl_mila.sh
VISIBLE=1 sbatch --array=0-4,25-29%2 run_dreamer_carl_mila.sh
LOG_ROOT="$PWD/logdir/batch8" sbatch --array=0,25%1 run_dreamer_carl_mila.sh --batch_size 8
STEPS=2000000 sbatch --array=0 run_dreamer_carl_mila.sh
```

Pass Slurm arguments **before** the script name and Dreamer arguments **after** it.
Prefer script arguments over the legacy whitespace-split `FLAGS` environment
variable for quoted values. Keep preset, seed, and log directory selection in the
launcher rather than overriding them with trailing Dreamer arguments. Overrides
are appended last; use a separate `LOG_ROOT` for changed hyperparameters. Do not
set a single `LOGDIR` or override `SEED` across an array unless you deliberately
restrict it to one task: otherwise jobs can share and corrupt outputs.

The visible-context condition has separate `_visible` directories. Smoke runs
have their own root and cannot accidentally resume production checkpoints unless
you explicitly override their output paths.

**Short runs do not test the full SHS curriculum.** The first phase lasts 500,000
environment steps without structural moves; the recurrent phase starts after
650,000. Walker enables birth/split only after 800,000, while Quadruped/Finger enable
them after 500,000. Use the full budget for the intended comparison. Increasing the
step budget does not rescale these curriculum boundaries.

## 4. Monitor, inspect results, and restart

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -f slurm-aime-carl-JOB_ID_0.out
tensorboard --logdir logdir
```

Replace `JOB_ID` with the ID returned by `sbatch`. Each job prints its configs,
seed, exact command, Git revision, Python executable, and key dependency versions.
Slurm stdout/stderr are separate and append across restarts. Training results use:

```text
logdir/<preset-or-vanilla-name>[_visible]/seed_<1..5>/
  latest.pt
  metrics.jsonl
  events.out.tfevents.*
  train_eps/
  eval_eps/
```

`eval_return` is the equally weighted mean of returns across the observed held-out
contexts. `eval_return_ctx0`, `eval_return_ctx1`, etc. give per-context returns.
Context IDs follow Cartesian-product order within the evaluation specification;
they do not identify the same physics as training IDs. For gravity-only evaluation,
ID 0 means 2.4525 and ID 1 means 39.24. `train_return` is from training contexts,
so do not interpret it as an in-distribution evaluation-policy score.

To resume a stopped run, resubmit its index with the **same environment, flags, and
output root**. `STEPS` is a total target, not an additional budget. The trainer
loads `latest.pt` and on-disk replay automatically; no `--resume` flag is required.
Do not submit a second writer while the original job is running or requeued.

Checkpoints are written after training chunks, not on a Slurm termination signal.
An interruption can lose recent updates, and checkpoint writes are not atomic.
This is restart support, not a guarantee of bitwise-identical continuation or
preemption-safe saving. Keep replay and checkpoints on persistent cluster storage,
not solely in node-local temporary storage, and back up important results.

## References

- [Mila Python environments](https://docs.mila.quebec/technical_reference/general_theory/portability/)
- [Mila single-GPU training workflow](https://docs.mila.quebec/getting_started/train_first_model/)
- [PyTorch previous-version CUDA wheels](https://pytorch.org/get-started/previous-versions/)
- [CARL installation and environment extras](https://github.com/automl/CARL/blob/main/README.md)
