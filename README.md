# AIME · SHS-RSSM

**Sticky nonparametric switching dynamics inside a world model.**

AIME extends [DreamerV3 (PyTorch)](https://github.com/NM512/dreamerv3-torch) with an
SHS-RSSM world model: the neural transition prior of the recurrent state-space model is
replaced by a nonparametric mixture of conjugate linear-Gaussian dynamical *regimes*,
mixed by a sticky, state-dependent Markov chain. The number of regimes is not a fixed
hyperparameter; it is adapted during reinforcement learning by variational birth, split,
merge and delete moves. Everything outside the transition prior — encoder, decoder,
deterministic backbone, actor–critic, imagination, replay and training defaults — follows
upstream DreamerV3-torch.

The repository also contains an offline harness that fits the same regime model to
classical switching-system benchmarks and compares it against published baselines
(rSLDS, TrSLDS).

---

## Overview of the model

DreamerV3 learns a recurrent state `s_t = (h_t, x_t)`: a deterministic GRU carry `h_t`
and a stochastic latent `x_t`, trained by an evidence lower bound with an amortised
encoder, a decoder and reward/continuation heads. Imagination rolls out the transition
prior alone, so the prior determines the latents on which the actor and critic are
trained. SHS-RSSM keeps this architecture and changes only the prior:

| Component | DreamerV3 | SHS-RSSM |
|---|---|---|
| Stochastic latent | categorical (`dyn_discrete > 0`) | continuous (`dyn_discrete: 0`) |
| Transition prior | neural map `p(x_t \| h_t)` | mixture of `K` linear-Gaussian regimes `N(M_k g_t + C h_t, Q_k)`, `g_t = [x_{t-1}; a_{t-1}; 1]` |
| Regime process | — | stick-breaking HDP base with a recurrent, disentangled persistence gate (stay/switch decided by a logistic function of `x_{t-1}`) |
| Number of regimes | — | truncation `K` adapted by birth / split / merge / delete |

Inference is split by conjugacy:

- **Amortised** — the encoder `q(x_t | h_t, o_t)` and all neural heads are trained by
  stochastic gradients, exactly as in DreamerV3.
- **Structured, exact given potentials** — the regime chain `q(z, ζ)` is computed by
  forward–backward from closed-form emission and transition potentials; the logistic
  gate is made conjugate by Pólya–Gamma augmentation, so no discrete-variable gradient
  estimator is needed anywhere.
- **Conjugate, closed form** — regime parameters (Normal–Gamma), transition rows
  (Dirichlet) and gate weights (Gaussian) are refit from expected sufficient statistics;
  the stick-breaking factor uses the surrogate bound of Hughes, Stephenson & Sudderth
  (NeurIPS 2015), optimised by L-BFGS.

Because the encoder and the policy keep changing during training, the global sufficient
statistics are not summed over the corpus but **blended with a constant rate**
(`shs_ema_tau`). This blend is the exact conjugate update of a recency-weighted
(tempered) posterior, so no projection error compounds; a *memoized* mode provides exact
whole-corpus updates for offline fits and consolidation passes. The full derivation,
including the bound, the gate bound and the streaming-update propositions, is in the
paper appendix.

---

## Repository layout

```
.
├── dreamer.py                 # training entry point (DreamerV3-torch loop)
├── models.py, networks.py     # world model, actor-critic, encoders/decoders
├── tools.py, parallel.py      # logging, replay, utilities, parallel envs
├── exploration.py             # exploration heads (upstream)
├── configs.yaml               # all presets; SHS options and curricula
├── envs/                      # environment wrappers (dmc, procgen, crafter, atari, …)
├── shs_rssm/                  # SHS-RSSM regime model
│   ├── shs_rssm.py            #   RSSM subclass: routes kl_loss / img_step through the regime head
│   ├── regime_head.py         #   regime head: E-step, dynamics KL, global updates, imagination
│   ├── regimes.py             #   per-regime Normal-Gamma dynamics (diagonal / low-rank noise)
│   ├── regimes_shared.py      #   shared carry map C
│   ├── sticky_hdp.py          #   stick-breaking HDP allocation model (surrogate bound, L-BFGS)
│   ├── recurrent_stick.py     #   Pólya-Gamma persistence gate (carry input)
│   ├── latent_stick.py, recurrent_stick_z.py   # persistence gate on x_{t-1} (default)
│   ├── forward_backward.py, structured_elbo.py # chain inference and bound terms
│   ├── moves.py, delete_planner.py             # birth / split / merge / delete proposals
│   ├── online_vb.py           #   sufficient-statistic stores: ema, streaming, memoized, full_batch
│   ├── teacher.py             #   EMA copy of the encoder and RSSM posterior path,
│   │                          #   used only as a drift probe for move-buffer staleness
│   ├── continuous_smoother.py, smoother_integration.py  # block-tridiagonal smoother
│   ├── offline_trainer.py, adaptive_fit.py     # offline fitting path used by the benchmarks
│   └── shs_diagnostics.py, shs_filmstrip.py    # regime diagnostics and figures
├── shs_demo/                  # offline benchmarks
│   ├── compare/               #   nascar / toyark13 / mocap6 vs rSLDS and TrSLDS
│   ├── fhn_benchmark.py       #   FitzHugh-Nagumo multi-step prediction
│   ├── rslds/, trslds/        #   vendored baselines (byte-identical to upstream)
│   └── mocap6/, toyark13/     #   datasets
├── sim_multiregime.py         # synthetic end-to-end check of birth / merge / delete
├── demo_shapes_disentangle.py # small pixel-domain demo
├── plot_returns.py            # learning-curve plots from metrics.jsonl
├── run_*.sh                   # SLURM launch scripts (see below)
├── Dockerfile
└── requirements.txt
```

---

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended for online training.

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers the training stack (PyTorch ≥ 2.4, dm_control, MuJoCo,
procgen, crafter, …) and the offline benchmark stack (SciPy, scikit-learn, numba,
polyagamma, matplotlib). Atari and Minecraft need additional setup; see
`envs/setup_scripts/`.

**Headless rendering.** DeepMind Control renders through MuJoCo. On machines without a
display, set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU) before launching.

**Docker.** A CUDA image with all system dependencies is provided:

```bash
docker build -f Dockerfile -t aime .
docker run -it --rm --gpus all -v "$PWD":/workspace aime \
  python dreamer.py --configs dmc_vision dmc_walker_shs --logdir logdir/walker
```

**Legacy baseline environment.** The 2017 rSLDS baseline (`shs_demo/rslds/`) depends on
`pyhsmm`, `pybasicbayes` and `pypolyagamma`, which require Python ≤ 3.8. Build that
environment separately from `shs_demo/compare/environment-baselines.yml`; everything
else runs in the main environment.

---

## Quick start: online training

Train SHS-RSSM on DeepMind Control *Walker Walk* from pixels:

```bash
python dreamer.py --configs dmc_vision dmc_walker_shs \
  --seed 0 --logdir logdir/walker/seed_0
```

Configs are applied left to right on top of `defaults`; any key can be overridden on the
command line (`--steps 2e6`, `--batch_size 16`, `--shs_K 24`, …). If `torch.compile`
causes problems on your platform, add `--compile False`.

### Shipped presets

| Preset | Task | Notes |
|---|---|---|
| `dmc_walker_shs` | `dmc_walker_walk` | 1M env frames, `K=24`, carry projection 256, low-rank noise rank 8, `sF=0.1`; recurrent hierarchical gate from step 0, merge/delete from 100k, births from 400k |
| `dmc_hopper_shs` | `dmc_hopper_hop` | open-loop sticky phase first, gate and moves later in the run |
| `dmc_cheetah_shs` | `dmc_cheetah_run` | as hopper |
| `dmc_acrobot_shs` | `dmc_acrobot_swingup` | swing-up vs. balance, single actuator |
| `dmc_pendulum_shs` | `dmc_pendulum_swingup` | sparse two-regime benchmark |
| `dmc_finger_turn_hard_shs` | `dmc_finger_turn_hard` | contact / no-contact with the spinner |
| `dmc_quadruped_walk_shs` | `dmc_quadruped_walk` | 12-dim action, gait phases, `K=24` |
| `dmc_humanoid_shs` | `dmc_humanoid_walk` | 5M frames; the phase boundaries are inherited from a 1M-frame schedule, scale them if you use it |
| `procgen_shs`, `procgen_bigfish_shs`, `procgen_fruitbot_shs` | Procgen | discrete actions (15-way one-hot), 50M steps |
| `atari100k_shs` + `atari_{breakout,demon_attack,ms_pacman,alien,seaquest,hero}_shs` | Atari100k | compose as `atari100k atari100k_shs atari_breakout_shs`; `shs_action_dim: -1` inherits each game's minimal action set |
| `dmc_vanilla_compare` | any DMC task | upstream-faithful DreamerV3 arm for paired comparisons |

For DMC, combine a domain config with a preset (`dmc_vision dmc_walker_shs`); for
Procgen use `procgen procgen_bigfish_shs`; for Atari compose all three. To run the
unmodified DreamerV3 baseline with identical batch geometry, replace the preset by
`dmc_vanilla_compare` (DMC) or drop the SHS presets (Procgen, Atari).

### Cluster launchers

Three SLURM array scripts cover the three suites. Each takes `ARM=shs|vanilla`, honours
`SEED`, `STEPS`, `BATCH_SIZE`, `BATCH_LENGTH`, `PROJECT_DIR`, `VENV_DIR` and `LOGDIR`
from the environment, and forwards anything in `FLAGS` to `dreamer.py`.

```bash
# DMC: 7 tasks x 5 seeds
sbatch run_dreamer_shs-rssm_dmc.sh                        # SHS arm
ARM=vanilla sbatch run_dreamer_shs-rssm_dmc.sh            # DreamerV3 arm

# Atari100k: 6 games x 5 seeds
sbatch run_dreamer_shs-rssm_atari100k.sh
ARM=vanilla sbatch run_dreamer_shs-rssm_atari100k.sh

# Procgen: 2 games x 5 seeds
sbatch run_dreamer_shs-rssm_procgen.sh
ARM=vanilla sbatch run_dreamer_shs-rssm_procgen.sh
```

Subsets and single runs override the array on the command line:

```bash
# walker only, 4 seeds
TASK=walker NSEEDS=4 sbatch --array=0-3 run_dreamer_shs-rssm_dmc.sh

# one task, one seed
TASK=walker SEED=2 sbatch --array=0 run_dreamer_shs-rssm_dmc.sh
GAME=seaquest SEED=0 sbatch --array=0 run_dreamer_shs-rssm_atari100k.sh

# extra flags forwarded to dreamer.py
FLAGS="--shs_live_moves False" sbatch run_dreamer_shs-rssm_dmc.sh
```

Each run writes `${LOGROOT}/dreamerv3_<arm>_<task>_<seed>.out` and `.err`
(`LOGROOT` defaults to `${PROJECT_DIR}/logs`), for example
`dreamerv3_shs-rssm_walker_0.out` or `dreamerv3_vanilla_walker_0.err`; `PILOT=1`
appends `_pilot`. SLURM expands `#SBATCH --output` at submit time, before the task
and seed are known, so the `%x_%A_%a` files left in the submit directory are
near-empty stubs that only capture failures before the redirect.

### First run on a new cluster

Before committing a full array, do one short run in a fresh log directory to check the
installation, the device and the environment bindings:

```bash
python dreamer.py --configs dmc_vision dmc_walker_shs \
  --steps 20000 --compile False --seed 0 \
  --logdir logdir/pilot_walker_seed0
```

Watch `shs_K`, `shs_live_sweeps_run` and `shs_gate_hier_persist` in TensorBoard. Training
resumes automatically from `<logdir>/latest.pt`, so a pilot must use its own directory.

### Reproducibility switches

Three keys change the algorithm rather than its hyperparameters and are the ones to fix
when comparing arms:

| Key | Default | Effect |
|---|---|---|
| `shs_live_moves` | `True` | `False` disables scheduled birth/split/merge/delete sweeps; `K` stays at `shs_K` |
| `shs_posterior_samples` | 4 | sequential posterior draws averaged into the global statistics; 1 uses the single training draw |
| `shs_gate_inner_iters` | 1 | `> 1` is an experimental batch-adaptive refinement; see `configs.yaml` before enabling |

Checkpoints carry the regime shapes, curriculum counters, calibration state, the pending
statistics transaction and a bounded slice of the move buffer, so a resumed run continues
with the same model. Retaining only part of the buffer means move *timing* after a resume
is close to, but not guaranteed identical to, an uninterrupted run.

---

## Configuration

All options live in `configs.yaml` under `defaults` and can be overridden per preset or
from the command line. The SHS-specific keys are prefixed `shs_`.

### Core model

| Key | Default | Meaning |
|---|---|---|
| `use_shs` | `False` | enable the SHS-RSSM prior (`dyn_discrete` must be `0`) |
| `shs_K` | 16 | initial truncation; over-provision and let merge/delete prune |
| `shs_action_dim` | 0 | width of the per-regime action term `B_k a_{t-1}` (set to the task's action dimension) |
| `shs_shared_carry` | `True` | share the carry map `C` across regimes |
| `shs_q_rank` | 0 | rank of a low-rank residual factor added to the diagonal noise `Q_k` |
| `shs_gamma`, `shs_alpha`, `shs_kappa` | 5, 8, 50 | HDP stick concentration, row concentration, sticky bias (`kappa` is set to 0 once the recurrent gate is active) |
| `shs_a0`, `shs_b0`, `shs_calibrate_b0` | 3, 2, `True` | Gamma prior on regime precisions; `calibrate_b0` scales `b0` to the data |

### Persistence gate

| Key | Default | Meaning |
|---|---|---|
| `shs_recurrent` | `True` | state-dependent stay/switch gate (disentangled sticky construction) |
| `shs_gate_input` | `latent` | `latent`: gate reads `x_{t-1}` (exact moments under the encoder posterior); `carry`: gate reads a projection of `h_t` |
| `shs_prior_persist` | 0.871 | prior mean persistence `E[π_jj]`, matched to the logit-bias prior of the gate |
| `shs_rstick_bias_var`, `shs_rstick_weight_var` | 4.0, 1.0 | prior variances of the gate bias and weights (with `shs_rstick_hier` the bias value is the hyper-prior variance) |
| `shs_rstick_hier` | `True` | hierarchical prior on the gate bias, `r_j ~ N(μ, 1/λ)`, `μ ~ N(logit(prior_persist), bias_var)`, `λ ~ Gamma(a0, b0)`; `E[μ]` = population persistence, `1/E[λ]` = its spread across regimes, both learned by conjugate updates after every gate refit (logged as `shs_gate_hier_*`) |
| `shs_rstick_hier_a0`, `shs_rstick_hier_b0` | 2.0, `null` | Gamma prior on λ; `null` sets `b0 = a0 · bias_var` so the hierarchical gate starts from the fixed prior |
| `shs_rstick_hier_used_only`, `shs_rstick_hier_min_evidence` | `False`, 1.0 | restrict the hyper-update to rows with accumulated gate evidence ≥ threshold (faster adaptation under heavy over-provisioning; the fixed point is the same) |
| `shs_gate_inner_iters` | 1 | **experimental when > 1** (`gate_input: latent` only): each extra pass refits the gate on the current batch, re-runs the chain, then restores the persistent gate. The returned factor is optimal for the temporary gate, not the persistent one, so the reported normaliser is not comparable and the posterior-draw batching stops being an exact rearrangement. Keep 1 |
| `shs_posterior_samples` | 4 | sequential posterior draws averaged in the global-statistics update (regime M-step, transition counts, gate statistics); loss, actor and move buffer use the single training draw |
| `shs_moment_consistent` | `True` | E-step, dynamics KL and move scoring condition on the sampled `z_{t-1}` the recognition GRU consumed (no independent-marginals predecessor variance) |
| `shs_live_moves` | `True` | `False` disables scheduled live merge/delete/birth sweeps (K stays at `shs_K`) |
| `shs_pg_iters` | 4 | inner Pólya–Gamma / Gaussian fixed-point iterations |

### Streaming updates

| Key | Default | Meaning |
|---|---|---|
| `shs_online_mode` | `ema` | `ema`: constant-rate blend of sufficient statistics (live training); `memoized`: exact whole-corpus statistics on a frozen dataset (offline fits, consolidation); `streaming`, `full_batch` also available |
| `shs_ema_tau` | 0.02 | blend rate `τ` (memory ≈ `1/τ` global updates) |
| `shs_ema_memory_env` | `null` | alternatively fix the EMA memory in environment steps |
| `shs_global_update_every` | 1 | global step cadence (gradient steps) |
| `shs_teacher`, `shs_teacher_tau` | `True`, 0.005 | EMA copy of the encoder and the RSSM posterior path (`_img_in_layers`, `_cell`, `_obs_out_layers`, `_obs_stat_layer`), used **only** to detect representation drift so stale move-buffer batches are purged; no latent is ever taken from it |
| `shs_consolidate_*` | — | periodic memoized re-encoding passes on a sampled sub-corpus |

### Structure adaptation

| Key | Default | Meaning |
|---|---|---|
| `shs_move_every` | 0 | move-sweep cadence in gradient steps (0 = off); `move_every_env` in curricula uses env steps |
| `shs_move_birth`, `shs_move_split` | `True` | enable birth / split proposals (merge and delete are always on) |
| `shs_merge_select`, `shs_delete_mode` | `hughes` | shortlist and acceptance rules following Hughes et al. (2015) |
| `shs_move_buffer`, `shs_move_max_age` | 8, 0 | number of recent batches scored per sweep and their maximum age |
| `move_threshold` (curriculum) | 10 nats | required improvement of the scored bound for acceptance |

### Strict ELBO mode

`--configs ... shs_strict` switches to a single-copy bound (`shs_strict_elbo: True`, no
free bits, memoized statistics) for objective-level checks. It requires a stable batch
partition (`shs_expected_batches`) and is not intended for RL runs.

---

## Monitoring, checkpoints and plots

Each run writes to `<logdir>/`:

| Path | Content |
|---|---|
| `metrics.jsonl` | scalar metrics per logging step (returns, losses, regime counts, move statistics) |
| TensorBoard events | the same scalars plus video predictions (`tensorboard --logdir logdir`) |
| `latest.pt` | checkpoint (networks, optimisers, regime head state, curriculum counters) |
| `train_eps/`, `eval_eps/` | replay episodes |
| regime diagnostics | filmstrips of inferred regimes over evaluation episodes and diagnostic figures when `shs_diag_figures: True` |

Learning curves across seeds and environments:

```bash
python plot_returns.py --logdir logdir --metric eval_return --smooth 5 --out returns.png
python plot_returns.py --logdir logdir --separate --csv summary.csv   # one panel per env
```

Seeds are grouped automatically from directory names (`seed_0`, `s1`, …).

---

## Offline benchmarks

The regime model can be fit offline, without the RL loop, through
`shs_rssm/offline_trainer.py`. Three harnesses are provided.

### Switching-system benchmarks (`shs_demo/compare/`)

Fits SHS-RSSM, TrSLDS (Nassar et al., 2019) and rSLDS (Linderman et al., 2017) to
`nascar` (synthetic), `toyark13` (13 autoregressive regimes) and `mocap6` (annotated
motion-capture exercises), and reports Hungarian-matched Hamming distance, purity, NMI
and ARI.

```bash
cd shs_demo/compare
python run_all.py --datasets nascar toyark13 mocap6 --seeds 0 1 2 --latex
python run_shs.py --dataset toyark13 --nseq 12       # single fit
python run_rslds.py --dataset mocap6                 # legacy environment only
python make_figures.py --dataset all --latex         # aggregate results/ into tables and figures
```

Outputs go to `results/<dataset>/` (tables, per-run `.npz`) and `figures/` (ground-truth
vs model regime ribbons, latent trajectories, objective curves, and the combined
`offline_summary.pdf`). See `shs_demo/compare/README.md` for the result schema and for
which metrics are comparable across model classes.

### FitzHugh–Nagumo multi-step prediction (`shs_demo/fhn_benchmark.py`)

Reproduces the protocol of Becker-Ehmck et al. (ICML 2019): 100 trajectories of length
430, last 30 steps held out, normalised RMSE as a function of horizon.

```bash
python shs_demo/fhn_benchmark.py --models shs trslds --seeds 0 1 2 --shs-K 12
python shs_demo/fhn_benchmark.py --plot-only
```

### Synthetic structure recovery (`sim_multiregime.py`)

A self-contained check that the move machinery does what it claims: a generator with a
late-onset regime (birth), a duplicated regime (merge), a rare regime (delete), ragged
sequence lengths and continuation chunks. Reports recovered `K`, segmentation accuracy,
dynamics error and bound monotonicity.

```bash
python sim_multiregime.py
```

---

## Citation

This implementation builds on: DreamerV3 (Hafner et al., 2023), the
sticky HDP-HMM (Fox et al., 2011) and its disentangled variant (Zhou et al., 2020),
memoized variational inference with birth/merge/delete moves (Hughes & Sudderth, 2013;
Hughes, Stephenson & Sudderth, 2015), recurrent switching linear dynamical systems
(Linderman et al., 2017; Nassar et al., 2019) and Pólya–Gamma augmentation
(Polson, Scott & Windle, 2013).

---

## License

This repository is released under the MIT License (see `LICENSE`). It includes code
from [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) (MIT, © NM512) and
vendored copies of the rSLDS and TrSLDS reference implementations under their original
licenses.