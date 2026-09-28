# AIME · SHS-RSSM

**A world model whose transition prior is a sticky, nonparametric mixture of linear-Gaussian
regimes, learned online inside DreamerV3.**

AIME keeps the DreamerV3 architecture (encoder, decoder, recurrent backbone, actor-critic,
imagination, replay) and replaces one component: the neural transition prior of the
recurrent state-space model becomes a mixture of conjugate linear-Gaussian dynamical
*regimes*, selected by a sticky, state-dependent Markov chain with a hierarchical
Dirichlet-process base. Regime parameters are updated in closed form from
responsibility-weighted sufficient statistics while the neural parts train by gradient
descent. Each regime forgets only as fast as its own data contradict it, and the number of
regimes can adapt during training through validated birth, split, merge and delete moves.
The unmodified DreamerV3 baseline lives in the same code base.

The repository contains the model, presets for DeepMind Control, Atari, Procgen and CARL
contextual control, a transfer runner, SLURM launch scripts, offline benchmarks against
published switching-system models, and drift experiments on the streaming update.

## Contents

1. [Model](#model)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Benchmarks and presets](#benchmarks-and-presets)
7. [Transfer](#transfer)
8. [Cluster scripts](#cluster-scripts)
9. [Monitoring](#monitoring)
10. [Offline experiments](#offline-experiments)
11. [References and license](#references)

---

## Model

DreamerV3 learns a recurrent state `s_t = (h_t, x_t)`: a deterministic GRU carry `h_t` and
a stochastic latent `x_t`, trained by an evidence lower bound with an amortised encoder, a
decoder and reward/continuation heads. Imagination rolls out the transition prior alone, so
the prior determines the latents on which the actor and critic are trained. SHS-RSSM changes
only the prior:

| Component | DreamerV3 | SHS-RSSM |
|---|---|---|
| Stochastic latent | categorical (`dyn_discrete > 0`) | continuous Gaussian (`dyn_discrete: 0`) |
| Transition prior | neural map `p(x_t \| h_t)` | mixture of `K` linear-Gaussian regimes |
| Regime process | — | sticky HDP-HMM with a state-dependent persistence gate |
| Number of regimes | — | fixed, or adapted by birth / split / merge / delete |

### Generative model

With a regime indicator `z_t ∈ {1..K}` and regressor `g_t = [x_{t-1}; a_{t-1}; 1]`,

```
x_t | z_t = k, h_t   ~  N( M_k g_t + C P h_t ,  Q_k )
z_t | z_{t-1} = j    ~  σ_j(φ_t) · δ_j  +  (1 − σ_j(φ_t)) · π_j
π_j                  ~  Dir(α β + κ e_j),     β ~ GEM(γ)
```

`M_k` is a regime-specific map on the previous latent, the action and a bias; `C` is a map
of a learned projection `P h_t` of the GRU carry, shared across regimes (`shs_shared_carry`)
or per regime; `Q_k` is diagonal with an optional low-rank factor (`shs_q_rank`). The stay
probability `σ_j(φ_t)` is a logistic function of the previous latent and action
(`shs_gate_input: latent`) or of the carry (`carry`), with a Gaussian prior on its weights.
The first latent of an episode starts from a learned `z0`.

### Inference

- **Neural parts** (encoder, GRU, projection, heads, actor, critic) train by stochastic
  gradients exactly as in DreamerV3. The KL term of the world-model loss is the variational
  expectation of the posterior under the regime mixture,
  `−Σ_k γ_tk E_q[log N(x_t | M_k g_t + C P h_t, Q_k)] − H[q(x_t)]`, plus the KL of the regime
  path, of the gate weights and of `z0`; it is split into dynamics and representation terms
  with DreamerV3's stop-gradients, free bits and scales.
- **Regime path** `q(z)`: exact forward-backward given `q(x)`, using closed-form expected
  emission potentials that include the posterior variance of `x_t` and `x_{t-1}`.
- **Regime parameters**: Normal-Gamma posteriors for `(M_k, Q_k)`, a Gaussian posterior for
  the shared `C`, a Gaussian posterior for the low-rank factor, Dirichlet posteriors for the
  transition rows, and a Gaussian posterior for the gate weights through Pólya-Gamma
  augmentation. The HDP stick weights `β` use the surrogate bound of Hughes, Stephenson &
  Sudderth (2015), optimised by L-BFGS; numerically invalid updates are refused and counted.
- **Filtering** during acting keeps a per-environment regime belief that is propagated by the
  transition model and updated by the emission likelihood of each posterior latent.

### Streaming sufficient statistics

The encoder and the policy keep changing, so global statistics `S̄` are blended from each
replay minibatch `S` with a rate `τ` (`shs_ema_tau`). A normalised blend is the conjugate
update of the implicit transition model of Masegosa et al. (2020): `S̄ = τ Σ`, where `Σ` is a
discounted sum of batch statistics with forgetting factor `1 − τ`. Two mechanisms make
forgetting regime-specific.

- **Responsibility-gated rates** (`shs_ema_gate`). As in the responsibility-weighted updates
  of MOLe (Nagabandi, Finn & Levine, 2019), regime `k` blends at
  `τ_k = τ · min(1, ρ_k / ρ_ref)`, where `ρ_k` is its share of the batch responsibility and
  `ρ_ref = 1/K` unless set. A regime that is not used is not updated, so its dynamics are
  kept and recalled when its mode recurs. Transition rows and gate statistics use the same
  per-regime rates.
- **Change-point retention** (`shs_forget_mode: evidence`). The installed update of every
  factor is read as the fractional posterior, with power `τ`, of a discounted-sum stream in
  the sense of Masegosa et al. (2020): with `S̄ = τ D`,
  `D_k ← r_k (1 − τ_k) D_k + (τ_k/τ) S_k` is exactly `S̄_k ← (1 − τ_k) r_k S̄_k + τ_k S_k`.
  Memory and batch therefore enter the installed factor and the inference for `r` with the
  same relative weights, and every batch enters `D` with a power `τ_k/τ ≤ 1` (its gated share
  when `shs_ema_gate` is on). The retention is inferred in the stream model `D`: the memory
  `M_k = (1 − τ_k)/τ · S̄_k` forms a normalised power prior
  `exp{(λ₀ + r M_k)·t(θ) − A(λ₀ + r M_k)}`, and the batch gives the evidence
  `L(r) = A(λ₀ + r M_k + (τ_k/τ) S_k) − A(λ₀ + r M_k)`. Scoring the installed factor itself
  would temper the batch by a further `τ` and leave the score unable to register even a
  complete change of the dynamics. The retention has a change-point prior: with probability
  `1 − u_k π` the regime has not drifted and `r = 1`, otherwise `r ~ U(0, 1)`. The exposure
  `u_k` is the responsibility share `min(1, ρ_k/ρ_ref)` in every configuration, so a regime or
  row without data never forgets; with `shs_ema_gate` off, evidence mode also gives such a row
  a zero gain, keeping its statistics bitwise unchanged. The drift rate `π` has a Jeffreys
  `Beta(½, ½)` prior, is shared by the regimes of a factor and is learned from every batch; a
  fresh transfer starts a new posterior for the target. `L(r)` is evaluated in closed form for
  every factor, given the current posteriors of the shared carry, the low-rank noise and the
  Pólya-Gamma variables: Normal-Gamma for the emission statistics, Dirichlet-multinomial for
  each transition row and Gaussian for each augmented gate row. For the two Gaussian families
  a simultaneous diagonalisation of the prior and memory precisions makes every `L(r)` an
  `O(G)` sum after one factorisation, written so that the memory-scale terms cancel
  analytically; log-gamma differences at large arguments use the Stirling series. The integral
  of `exp(L(r) − L(1))` over the slab uses globally adaptive Gauss-Kronrod quadrature with an
  estimated error of `10⁻¹⁰` (or the rounding floor of `L`), and the posterior of `π` is
  evaluated on a logit grid whose interpolation error is at machine precision. A regime whose
  evidence is not finite or not integrated to `10⁻⁶` is treated as unused and counted as a
  guard event. The update carries the posterior mean forward,
  `S̄_k ← (1 − τ_k) E[r_k | data] S̄_k + τ_k S_k`, which minimises the posterior expectation
  of `KL(q ‖ q_r)` over conjugate `q`. All of it runs in float64 on the model's device. The
  initial-state counts use the plain gated rate. The inference treats replayed minibatches as
  fresh observations of the stream; overlapping replay makes this an approximation. When one
  batch cannot resolve `r` against a much larger memory, `L(r)` is nearly flat over most of
  the slab, the drift probability stays near `u_k E[π]` and the retention follows the prior;
  `E[π]` then decreases only as stationary exposure accumulates, so early updates and rows
  with few counts forget more than the base rate.

A memoized mode (`shs_online_mode: memoized`) keeps exact whole-corpus statistics for
offline fits on a frozen dataset.

### Imagination

Each imagined step forms the regime mixture from the current belief and draws the next
latent according to `shs_imag_mode`:

| `shs_imag_mode` | Regime choice | Latent sample | Gradient through the regime choice |
|---|---|---|---|
| `actor_moment` | belief kept as a distribution | moment-matched Gaussian of the mixture | pathwise |
| `st_sample` | one regime drawn per step | Gaussian of that regime | straight-through |
| `eval_sample` | one regime drawn per step | Gaussian of that regime | none |
| `reinforce_sample` | one regime drawn per step | Gaussian of that regime | score function |

`imag_particles` repeats every start state to average the actor objective over several
imagined futures.

### Structure adaptation

With `shs_structure_moves: True`, a sweep runs every `move_every_env` agent steps of the
curriculum and proposes, in order: removal of unoccupied regimes, merges (shortlisted by the
Hughes et al. merge criterion), deletions (Hughes et al. planner), births (sequential
creation from a segment of one sequence, or an interval proposal) and splits. The sweep
works on a window of recent training batches (`shs_move_buffer`) whose latents were all
produced under the same version of a slow EMA copy of the encoder
(`shs_move_version: teacher`), so no comparison mixes two latent coordinate systems.

**Fit and validation sets.** Every replay row carries the identity and acquisition time of
the episodes it was cut from. The most recently collected episodes form the validation set
(at least `shs_move_holdout` batches of rows); every row that touches a validation episode is
removed from the fitting set, so the two sets share no episode. If a disjoint split with
enough rows on both sides cannot be formed, the sweep is deferred.

**Acceptance.** A proposal is committed only if it passes three tests, in this order:

1. *Proposal screen.* The candidate and the unchanged model are refit on the fitting set and
   the candidate must raise the variational bound of the transition prior by
   `move_threshold` nats (birth, split) or `shrink_threshold` nats (merge, delete), the
   memoized acceptance rule of Hughes et al. restricted to the window.
2. *Installed-state bound test.* The structure that would actually run is built: the live
   streaming statistics, transition counts and gate statistics are mapped through the
   merge/delete pattern, new regimes are seeded with their window statistics scaled to one
   batch of mass, and the parameters are refit from the mapped statistics. Its bound on the
   fitting set must exceed the bound of the running model by the same threshold. The
   comparison is between two states that received identical optimisation, the running model
   and the running model with the edit.
3. *Predictive test.* The validation sequences are rolled out open-loop for
   `shs_move_holdout_horizon` steps with their logged actions, starting from the causal
   (prefix-only) regime belief, and scored by the per-step log density of the posterior
   latents under the moment-propagated predictive. With paired per-sequence differences `d`
   against the running model, `z = shs_move_holdout_z` and `δ = shs_move_holdout_tol`, a
   growth move needs `mean(d) − z·se(d) > 0` and a simplification needs
   `mean(d) − z·se(d) > −δ` (non-inferiority). Missing or non-finite scores refuse the move.

A refused proposal is rolled back completely (regimes, HDP, transition counts, gate,
statistic store and belief maps) before the next one. `shs_move_K_headroom` caps growth at
`K_initial + headroom` for every candidate.

**Beliefs across edits.** Every accepted edit starts a new structure generation, and every
regime belief carries the generation it was computed in. A belief from an earlier generation
is mapped through the composition of all edits since then, so a belief on a deleted regime
never becomes a belief on a newborn one, including after cycles such as K = 3 → 2 → 3.
Deferred global updates computed under an earlier structure are recomputed under the current
one before they are applied.

**Scope of the tests.** The bound tests are exact for the conjugate transition prior on the
window. The predictive test measures latent forecasts, not images, rewards or returns; the
validation episodes are new relative to the fitting episodes but have been seen by the
neural networks and by earlier streaming updates, which affects the running model and the
candidate alike. Whether an accepted structure improves control is an empirical question
for the multi-seed comparisons below.

---

## Repository layout

```
.
├── dreamer.py                 # training entry point (DreamerV3-torch loop)
├── transfer_walker.py         # transfer runner (source checkpoint -> target task)
├── models.py, networks.py     # world model, actor-critic, encoders/decoders
├── tools.py, parallel.py      # logging, replay with episode identities, utilities
├── configs.yaml               # defaults, presets, curricula, overlays
├── envs/                      # environment wrappers (dmc, atari, procgen, crafter, carl, ...)
├── shs_rssm/
│   ├── shs_rssm.py            # RSSM subclass: KL, filtering, imagination, move scheduling
│   ├── regime_head.py         # E-step, dynamics KL, streaming updates, imagination mixture
│   ├── regimes.py, regimes_shared.py            # Normal-Gamma regimes, shared carry, low rank
│   ├── sticky_hdp.py, recurrent_stick.py, latent_stick.py  # HDP base and persistence gate
│   ├── forward_backward.py, structured_elbo.py  # chain inference and bound terms
│   ├── forgetting.py          # change-point retention: exact evidence, quadrature, drift rate
│   ├── moves.py, delete_planner.py, remap.py    # proposals, acceptance, commit, belief maps
│   ├── online_vb.py           # sufficient-statistic stores (ema, streaming, memoized)
│   ├── teacher.py             # slow EMA copy of the encoder
│   └── offline_trainer.py, shs_diagnostics.py   # offline fitting, diagnostics and figures
├── shs_demo/
│   ├── compare/               # switching-system benchmarks vs rSLDS and TrSLDS
│   ├── rslds/, trslds/        # reference implementations of the two baselines
│   ├── drift/                 # drift experiments on the streaming update
│   ├── fhn_benchmark.py       # FitzHugh-Nagumo multi-step prediction
│   └── toyark13/, mocap6/     # datasets
├── sim_multiregime.py         # synthetic structure recovery check
├── plot_returns.py            # learning-curve plots from metrics.jsonl
├── run_*.sh                   # SLURM launch scripts
└── Dockerfile, requirements.txt
```

---

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended for online training.

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers the training stack (PyTorch ≥ 2.4, dm_control, MuJoCo, procgen,
crafter), CARL (`carl-bench`) and the offline stack (SciPy, scikit-learn, numba,
polyagamma, matplotlib). Atari needs `gym[atari]` with ALE ROMs (`envs/setup_scripts/atari.sh`)
and Minecraft its own setup (`envs/setup_scripts/minecraft.sh`).

**Headless rendering.** DeepMind Control renders through MuJoCo. Without a display, set
`MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU); the SLURM scripts try EGL first.

**Docker.**

```bash
docker run -it --rm --gpus all pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime nvidia-smi
docker build -f Dockerfile -t aime .
docker run -it --rm --gpus all -v "$PWD":/workspace aime \
  python dreamer.py --configs dmc_vision dmc_walker_shs_moves --logdir logdir/walker
tensorboard --logdir logdir
```

If the first command cannot see the GPU, add `--privileged` and run
`sh -c 'ldconfig -v && nvidia-smi'`.

**rSLDS environment.** The rSLDS reference implementation depends on `pyhsmm`,
`pybasicbayes` and `pypolyagamma`, which require Python ≤ 3.8; build it from
`shs_demo/compare/environment-baselines.yml`. Everything else runs in the main environment.

---

## Quick start

```bash
python dreamer.py --configs dmc_vision dmc_walker_shs_moves --seed 0 --logdir logdir/walker/seed_0

python dreamer.py --configs dmc_vision dmc_walker_shs_moves --shs_structure_moves False \
  --seed 0 --logdir logdir/walker_fixed/seed_0

python dreamer.py --configs dmc_vision dmc_vanilla_compare --task dmc_walker_walk \
  --seed 0 --logdir logdir/walker_dreamer/seed_0

python plot_returns.py --logdir logdir --metric eval_return --smooth 5 --out returns.png
```

The first command trains SHS-RSSM with structure adaptation on Walker Walk from pixels, the
second the same model with its regime set fixed, the third the DreamerV3 baseline on the same
budget. Configs are applied left to right on top of `defaults`; any key can be overridden on
the command line (`--steps 2e6`, `--shs_K 24`, `--shs_imag_mode st_sample`, ...). Add
`--compile False` if `torch.compile` is not available.

**Checkpoints and restarts.** `dreamer.py` writes `<logdir>/latest.pt` after every evaluation
interval and, when it receives `SIGUSR1` or `SIGTERM`, at the next 100-step boundary, after
which it exits with status 75. The file is written to a temporary name and renamed, so an
interruption never leaves a partial checkpoint. It holds the model and optimizers, the regime
shapes, the curriculum, gate, calibration, teacher and deferred-update state, the agent step,
the position inside the current evaluation interval, the training and logging schedules, the
random-number generator states and the list of replay episodes. A restart continues at the
checkpoint's step: episodes and metric records written after the checkpoint are moved to
`train_eps_after_checkpoint/` and `metrics_after_checkpoint.jsonl`, TensorBoard hides the events
after that step, the evaluation grid is kept, and the environments and the replay sampler get
new seeds derived from `seed` and the restart count. Episodes that were in progress are not
kept, so a restarted run continues from the checkpoint rather than reproducing the
uninterrupted run step for step. A replay episode listed in the checkpoint but missing from
`train_eps/` stops the run. Each run directory keeps `launch_config.json`, the resolved
configuration of the first launch, and `run_config.json`, that of the latest launch. A launch
whose settings differ from those stored in `latest.pt` in anything other than the run controls
(`logdir`, `steps`, evaluation and logging settings, `device`, `compile`, `parallel`) exits with
status 78 and names the keys, so a run directory is never continued under another
configuration; use another directory instead.

---

## Configuration

All options live in `configs.yaml` under `defaults`; SHS-specific keys are prefixed `shs_`.

### Core model

| Key | Default | Meaning |
|---|---|---|
| `use_shs` | `False` | enable the SHS-RSSM prior (`dyn_discrete` must be `0`) |
| `shs_K` | 16 | initial number of regimes |
| `shs_action_dim` | 0 | width of the per-regime action term (`-1` inherits the environment's action width) |
| `shs_proj_dim` | 64 | width of the projection of `h_t` seen by the regime prior (`null` = no projection) |
| `shs_shared_carry` | `True` | share the carry map `C` across regimes |
| `shs_q_rank` | 0 | rank of the low-rank factor of `Q_k` |
| `shs_gamma`, `shs_alpha`, `shs_kappa` | 5, 8, 50 | HDP stick concentration, row concentration, sticky bias (`kappa` is 0 while the gate is active) |
| `shs_a0`, `shs_b0`, `shs_calibrate_b0`, `shs_calibrate_sF` | 3, 2, `True`, 1.0 | Gamma prior on regime precisions; calibration scales `b0` to the data once, at the end of the first curriculum phase |

### Persistence gate

| Key | Default | Meaning |
|---|---|---|
| `shs_recurrent` | `True` | state-dependent stay/switch gate |
| `shs_gate_input` | `latent` | `latent`: gate reads `x_{t-1}`; `carry`: gate reads a projection of `h_t` |
| `shs_rstick_use_action` | `True` | the gate also reads `a_{t-1}` |
| `shs_prior_persist` | 0.871 | prior mean persistence `E[σ]` |
| `shs_rstick_bias_var`, `shs_rstick_weight_var` | 4.0, 1.0 | prior variances of the gate bias and weights |
| `shs_pg_iters` | 4 | Pólya-Gamma fixed-point iterations of a gate posterior update |

### Streaming updates

| Key | Default | Meaning |
|---|---|---|
| `shs_online_mode` | `ema` | `ema`: streaming blend of sufficient statistics; `memoized`: exact whole-corpus statistics on a frozen dataset; `streaming`, `full_batch` also available |
| `shs_ema_tau` | 0.02 | blend rate `τ` (memory ≈ `1/τ` updates for a fully used regime) |
| `shs_ema_memory_env` | `null` | alternatively set the memory in agent steps |
| `shs_ema_gate` | `True` | responsibility-gated per-regime rates |
| `shs_ema_gate_ref`, `shs_ema_gate_floor` | 0 (= `1/K`), 0.0 | share that receives the full rate; lowest rate of a regime or row with data, as a fraction of `τ` (without data the rate is 0) |
| `shs_forget_mode` | `fixed` | `evidence`: per-regime, per-factor change-point retention with a learned drift rate |
| `shs_teacher`, `shs_teacher_tau`, `shs_teacher_drift_tol` | `True`, 0.005, 0.05 | slow EMA copy of the encoder that stamps the move window |
| `shs_consolidate_*` | — | periodic memoized re-encoding passes on a sampled sub-corpus |

### Structure adaptation

| Key | Default | Meaning |
|---|---|---|
| `shs_structure_moves` | `False` | master switch; when `False` the regime set is fixed and every move schedule is inert |
| `shs_curriculum` | `[]` | phases with `until_env`, `move_every_env`, `recurrent`, `kappa`, `birth`, `split`, `move_threshold`, `shrink_threshold`, `create_bonus` |
| `shs_schedule_unit` | `agent` | unit of every `*_env` key: `agent` steps (frames / `action_repeat`) or raw `frame`s |
| `shs_move_first`, `shs_move_first_env` | `null` | one additional sweep at this gradient / agent step |
| `shs_move_version` | `teacher` | coordinate stamp of the move window (`teacher` or `repr`) |
| `shs_move_birth`, `shs_move_split` | `True` | enable birth / split proposals |
| `shs_birth_style` | `seqcreate` | `seqcreate`, `interval` or residual birth proposals |
| `shs_merge_select`, `shs_delete_mode` | `hughes` | merge shortlist and delete planner |
| `shs_move_buffer`, `shs_move_max_age` | 8, 0 | batches in the move window and their maximum age in gradient steps |
| `shs_move_holdout` | 0 | validation rows, in batches, taken from the newest episodes (needs `shs_move_buffer ≥ 2 × holdout`) |
| `shs_move_holdout_horizon`, `shs_move_holdout_z` | 15, 2.0 | open-loop horizon and confidence multiplier of the predictive test |
| `shs_move_holdout_tol` | 1.0 | non-inferiority margin of a merge or delete (nats per step) |
| `shs_move_commit` | `remap` | `remap`: install the edit on the streaming memory; `window`: install the window refit (complete-corpus consolidation) |
| `shs_move_K_headroom` | `null` | cap `K_initial + headroom` on birth and split |
| `shs_delete_empty` | `False` | propose the removal of regimes with no window mass and no remaining memory |
| `move_threshold`, `shrink_threshold` (curriculum) | 10, = `move_threshold` | bound improvement required for growth and for simplification |

### Imagination

| Key | Default | Meaning |
|---|---|---|
| `shs_imag_mode` | `actor_moment` | `actor_moment`, `st_sample`, `eval_sample`, `reinforce_sample` |
| `shs_imag_propagate_var` | `False` | feed each imagined step's variance into the next step |
| `imag_particles` | 1 | imagined rollouts per start state |

### Overlays

| Overlay | Effect |
|---|---|
| `shs_small` | `K = 18`, projection 128, low-rank rank 4 |
| `shs_walker_recipe` | walker model: `K = 18`, rank 8, projection 128, `τ = 0.02`, `b0` calibrated with `sF = 1`, `κ = 50`, `actor_moment`, gate from 650k frames, moves from 500k |
| `shs_quadruped_recipe` | quadruped model: `K = 24`, rank 4, projection 128, `τ = 0.005`, `b0 = 0.2`, `sF = 0.1`, `κ = 10`, `actor_moment`, gate without the action from the start, moves from 50k |
| `shs_imag_st` | `shs_imag_mode: st_sample` |
| `shs_fixed_k` | fixed regime set and gate off throughout (a stricter ablation than `--shs_structure_moves False`) |
| `shs_strict` | single-copy bound with memoized statistics for objective-level checks (not for RL runs) |
| `dmc_vanilla_compare`, `atari_vanilla`, `carl_vanilla` | DreamerV3 baseline on the same task and budget |
| `carl_moves` | structure-adaptation schedule for the CARL presets |
| `carl_context_visible` | feeds the context vector to the encoder |

---

## Benchmarks and presets

### DeepMind Control

`--configs dmc_vision dmc_<task>_shs_moves`. Frame counts assume `action_repeat: 2`.

| Preset | Task | Budget | merge + delete | gate | birth + split | imagination |
|---|---|---|---|---|---|---|
| `dmc_walker_shs_moves` | walker walk | 1M | 300k | 450k | 600k | `actor_moment` |
| `dmc_cheetah_shs_moves` | cheetah run | 1M | 300k | 450k | 600k | `actor_moment` |
| `dmc_hopper_shs_moves` | hopper hop | 1M | 300k | 450k | 600k | `actor_moment` |
| `dmc_pendulum_shs_moves` | pendulum swingup | 1M | 300k | 450k | 600k | `actor_moment` |
| `dmc_acrobot_shs_moves` | acrobot swingup | 1M | 300k | 450k | 600k | `actor_moment` |
| `dmc_finger_turn_hard_shs_moves` | finger turn hard | 2M | 600k | 800k | 1.0M | `st_sample` |
| `dmc_reacher_hard_shs_moves` | reacher hard | 2M | 600k | 800k | 1.0M | `st_sample` |
| `dmc_quadruped_run_shs_moves` | quadruped run | 2M | 600k | 800k | 1.0M | `st_sample` |
| `dmc_quadruped_walk_shs_moves` | quadruped walk | 2M | 600k | 800k | 1.0M | `st_sample` |

All nine use responsibility-gated rates, change-point retention, sweeps every 20.8k frames
once enabled, a 64-batch window with 8 batches of validation rows, `move_threshold: 10`,
`shrink_threshold: 0` and at most `shs_K + 8` regimes. With `--shs_structure_moves False`
the same presets give the fixed-regime model with an identical gate schedule. The
`dmc_*_shs` and `dmc_*_shs_gate` presets train the same tasks with one ungated rate and no
change-point retention; `dmc_humanoid_shs` covers humanoid walk.

### Atari

`--configs atari100k atari_<game>_shs_moves`: 64×64 colour frames, action repeat 4, minimal
action set, one-hot actor with REINFORCE, one gradient step per agent step, 1M frames, ten
evaluation episodes every 20k frames.

| Preset | Actions | `shs_K` | `shs_kappa` | imagination |
|---|---|---|---|---|
| `atari_pong_shs_moves` | 6 | 8 | 25 | `actor_moment` |
| `atari_breakout_shs_moves` | 4 | 8 | 25 | `actor_moment` |
| `atari_alien_shs_moves` | 18 | 16 | 10 | `actor_moment` |
| `atari_demon_attack_shs_moves` | 6 | 12 | 10 | `st_sample` |
| `atari_hero_shs_moves` | 18 | 24 | 10 | `st_sample` |
| `atari_frostbite_shs_moves` | 18 | 16 | 10 | `st_sample` |

All six share one schedule: no moves for 100k frames, then sweeps every 20.8k frames,
merge/delete only until 200k frames, births from 200k, gate and splits from 500k. The gate
reads the one-hot action. `atari100k_shs` with an `atari_<game>_shs` overlay trains the same
games with one ungated rate and no change-point retention.

### CARL contextual control

The `carl_*_shs` presets train on a grid of physics contexts drawn per episode and evaluate
on contexts outside that grid, so the dynamics change from episode to episode and again at
evaluation:

| Preset | Task | Train contexts | Eval contexts |
|---|---|---|---|
| `carl_walker_shs` | walker walk | gravity 4.9, 9.8, 19.6 | gravity 2.45, 39.2 |
| `carl_walker_gravity_friction_shs` | walker walk | gravity × tangential friction 0.5, 1, 2 | gravity 2.45, 39.2 × friction 0.25, 4 |
| `carl_quadruped_walk_shs` | quadruped walk | gravity 4.9, 9.8, 19.6 | gravity 2.45, 39.2 |
| `carl_quadruped_actuator_shs` | quadruped walk | actuator strength 0.5, 1, 2 | actuator strength 0.25, 4 |
| `carl_finger_spin_shs` | finger spin | gravity 4.9, 9.8, 19.6 | gravity 2.45, 39.2 |

Any context feature of the CARL dm_control environments can be used (`gravity`,
`friction_*`, `actuator_strength`, `joint_damping`, `joint_stiffness`, `density`,
`viscosity`, `geom_density`, `wind_*`, `timestep`; finger adds limb and spinner geometry).
The base presets keep the regime set fixed; `carl_moves` adds the schedule (no moves for 75k
frames, merge/delete until 150k, births from 150k, gate and splits from 450k) and the move
checks (held-out forecasts on 8 episode-disjoint batches, K at most the preset K + 8, removal
of empty regimes). Evaluation returns are logged per context (`eval_return_ctx<i>`) and as
their mean. The CARL launcher applies `shs_walker_recipe` to the walker presets and
`shs_quadruped_recipe` to the quadruped presets unless `RECIPE=0`, and `carl_moves` after
them, so the `shs_moves` arm keeps the CARL move schedule and checks.

The CARL launcher works through a prioritised run list with a few GPU lanes. Each array task
is a lane: a 12-hour job that continues the highest-priority run that is neither finished nor
held by another lane, starts the next run when one reaches `STEPS`, and ten minutes before its
time limit has `dreamer.py` save, queues its own next slice and ends. Runs therefore finish in
priority order, and `DRY_RUN=1 bash run_dreamer_carl_mila.sh` lists them with their progress.
`SETUPS` gives the order as `arm:forgetting` pairs (default
`shs:gated shs_moves:gated shs:retain shs_moves:retain`); within a setup the list goes by seed,
then preset. `shs:preset` with `RECIPE=0` gives the plain presets under their original run
names, and `vanilla` adds DreamerV3.

### Other domains

Procgen (`procgen procgen_bigfish_shs`, `procgen_fruitbot_shs`) and Atari 100k
(`atari100k atari100k_shs`) presets with their own curricula are included.

---

## Transfer

`transfer_walker.py` loads a checkpoint trained on one task and continues on another with
fresh replay and fresh optimisers. The architecture (regime count, projection width,
low-rank rank, action width, gate, network widths) is read from the checkpoint's tensors,
the remaining hyperparameters from its stored configuration (or, for checkpoints that do not
store one, from the first matching recipe in `--source-recipes`), and the live gate and
calibration state from its runtime record. Four arms share the network and the budget:

| Arm | `--transfer-mode` | Preset | Starts from |
|---|---|---|---|
| full | `full` | `dmc_walker_run_shs_transfer_moves` | world model, reward head, actor and critic of the source |
| world model | `world_model` | `dmc_walker_run_shs_transfer_moves` | world model of the source; fresh reward head, actor and critic |
| scratch | `scratch` | `dmc_walker_run_shs_scratch_moves` | random weights, fixed regime set |
| scratch + moves | `scratch` | `dmc_walker_run_shs_scratch_moves` | random weights, structure adaptation |

The transfer arms keep the source's live gate; the scratch arms learn their representation
within the target budget with the gate off. The regime set stays fixed unless
`--shs_structure_moves True` is passed; the transfer schedule then allows no moves for 50k
frames, merge/delete until 100k frames and births afterwards, with at most `K_source + 4`
regimes. With change-point retention, a source regime whose dynamics the target data
contradict discards its source statistics at the rate the evidence supports, while source
regimes that the target does not use keep theirs. The drift-rate posterior is not carried
over: the target learns its own.

```bash
python transfer_walker.py --configs dmc_vision dmc_walker_run_shs_transfer_moves \
  --task dmc_walker_run --source-task dmc_walker_walk \
  --transfer-mode full --checkpoint logdir/shs_moves/walker/seed_1/latest.pt \
  --shs_structure_moves True --steps 200000 --logdir logdir/transfer/full_moves_seed_1
```

**Source runs.** The transfer launchers read the Walker Walk runs of
`run_dreamer_shs-rssm_dmc_best.sh` (`dmc_walker_shs shs_walker_recipe`), that is
`logdir/dmc_walker_walk_shs_<arm>_seed_<n>`, with `<arm>` = `SOURCE_ARM`, which defaults to the
target's forgetting arm `FORGET` (`fixed`, `gated`, `retain`, `slow`, `fast`, as in that
script) and `SOURCE_DIR` the `RUN_ROOT` of those runs. Each target run is named after its arm
and its source. `dmc_walker_shs_moves` sources
from `run_dreamer_shs-rssm_dmc_mila.sh` (`NAME=walker`, `logdir/shs_<arm>/walker/seed_<n>`) or
any other checkpoint are used with `SOURCE_CKPT=…`.

**Runner behaviour.** `--prefill-policy transferred` collects the initial replay with the
loaded actor instead of uniform random actions. `transfer_curriculum_clock` chooses whether
the curriculum continues from the source's gradient step or restarts. A run checkpoints at
every evaluation and every `--save-every` frames, stops cleanly on `SIGUSR1`/`SIGTERM` with
exit status 75 and continues with `--resume`, which restores the experiment definition from
its own checkpoint and refuses flags that would change it. A resumed run sets aside the replay
episodes and metric records written after the checkpoint, restores the random-number
generators and draws new environment and sampler seeds, as described for `dreamer.py`; an
evaluation that was interrupted is repeated from the checkpoint taken before it. Each run directory keeps
`launch_config.json`, `transfer_config.json` and `transfer_metadata.json` with the resolved
source, arm, structure switch, forgetting mode, commit mode and imagination mode.

---

## Cluster scripts

| Script | Cluster | Runs |
|---|---|---|
| `run_dreamer_shs-rssm_dmc_best.sh` | Vulcan or Mila | walker walk, finger turn hard, quadruped walk and run, reacher hard × four seeds with the recipes below and the forgetting arms `ARMS="fixed gated retain slow"` (`fast` optional), `MOVES=0/1`, `MOVE_GATE=1/0`; run with `bash` on a login node, it submits itself with 4 CPUs, 24G and one GPU other than V100; ten minutes before the time limit a slice queues the next one (the new job id is printed) and has `dreamer.py` checkpoint and exit with 75, until the run is done; a slice that could not queue its successor ends with status 75; run directories go under `RUN_ROOT` (`PROJECT_DIR/logdir`) |
| `run_dreamer_shs-rssm_dmc_mila.sh` | Mila | nine DMC tasks × four seeds; `ARM=moves\|fixedk\|moves_st\|fixedk_st` differ only in the structure switch and imagination mode, `ARM=fixedk_nogate` adds `shs_fixed_k` |
| `run_dreamer_shs-rssm_atari_mila.sh`, `..._atari_vulcan.sh` | Mila, Vulcan | six Atari games × four seeds; `ARM=fixedk\|moves\|moves_st\|vanilla` |
| `run_dreamer_carl_mila.sh` | Mila | 80 runs (five CARL presets × `shs:gated`, `shs_moves:gated`, `shs:retain`, `shs_moves:retain` × four seeds) worked through in priority order by 12-hour lanes that resubmit themselves (`--array=1-2` gives two GPUs at a time); walker and quadruped presets use the recipe overlays (`RECIPE=0` for the plain presets); `SETUPS`, `CARL_PRESETS`, `SEEDS` change the list; logs go to `./logs`, which must exist before `sbatch` |
| `run_dreamer_shs-rssm_walker_vulcan.sh`, `..._walker_vulcan_minimum.sh` | Vulcan | Walker Walk source runs; `MOVES=1` (default) or `MOVES=0` |
| `run_dreamer_shs-rssm_walker_transfer_mila.sh` | Mila | four transfer arms × four seeds in chained 3-hour slices; `MOVES=1` enables structure adaptation in the full and world-model arms (`scratch` stays fixed, `scratch_moves` always adapts); `FORGET` selects the forgetting arm and the source; defaults `MOVES=0`, `FORGET=fixed` |
| `run_dreamer_shs-rssm_walker_transfer_vulcan_minimum.sh` | Vulcan | full, world-model and scratch arms; `MOVES=0/1` applies to all three; `FORGET` selects the forgetting arm and the source; defaults `MOVES=0`, `FORGET=fixed` |
| `run_dreamer_gaussian_latents_mila.sh` | Mila | DreamerV3 with Gaussian latents on four DMC tasks |
| `run_dreamer_shs-rssm_<task>_vulcan_minimum.sh` | Vulcan | single-task runs |
| `shs_demo/drift/run_prop2.sh`, `run_prop2_segmnist.sh` | Vulcan (CPU) | drift experiments |

The other scripts accept `STEPS`, `SEED`, `FLAGS` and `LOGDIR` overrides; `..._dmc_best.sh`
takes `STEPS`, `FLAGS`, `TASKS`, `SEEDS` and `RUN_ROOT` (run `bash run_dreamer_shs-rssm_dmc_best.sh --help`);
the transfer launchers read their sources from `SOURCE_DIR`, which should be that `RUN_ROOT`.
Every script checks the GPU and the renderer before training, and a resumed run keeps the
structure switch it was started with.

Recipes of `run_dreamer_shs-rssm_dmc_best.sh`, all with `shs_imag_mode: actor_moment` and 16 × 64
batches:

| Task | Presets | Overrides |
|---|---|---|
| walker walk | `dmc_walker_shs shs_walker_recipe` | none |
| finger turn hard | `dmc_finger_turn_hard_shs_gate` | `shs_K 16` |
| quadruped walk | `dmc_quadruped_walk_shs_gate shs_quadruped_recipe` | none |
| quadruped run | `dmc_quadruped_run_shs_gate shs_quadruped_recipe` | `shs_b0 0.1` |
| reacher hard | `dmc_reacher_hard_shs` | none |

With `MOVES=1` the launcher also applies `shs_move_gate` (held-out forecasts on 8
episode-disjoint batches, K at most the recipe K + 8, removal of empty regimes); `MOVE_GATE=0`
runs the moves without these checks under `<arm>_noholdout`. The schedules count agent steps
(frames / 2). Finger and quadruped merge and delete from agent step 50k (frame 100k), add
births from 200k (frame 400k) and splits from 300k (frame 600k). The walker and reacher
schedules keep the first phase (no moves, non-recurrent gate, `κ = 10`) until agent step 500k,
which is frame 1M, so within the 1M-frame budget those two tasks keep K fixed whatever `MOVES`
says.

The arms change only the forgetting of the conjugate statistics:

| Arm | `shs_ema_gate` | `shs_forget_mode` | Rate |
|---|---|---|---|
| `fixed` | `False` | `fixed` | `τ` of the recipe for every regime |
| `gated` | `True` | `fixed` | `τ·u_k` |
| `retain` | `True` | `evidence` | `τ·u_k`, and the kept fraction `(1 − τ u_k) E[r_k]` |
| `slow`, `fast` | `False` | `fixed` | `τ/4`, `4τ` |

`fixed` against `slow` and `fast` measures the effect of the base rate, `gated` against `fixed`
the effect of sparing unused regimes, and `retain` against `gated` the effect of the inferred
retention; `shs_retention_*`, `shs_drift_*` and `shs_ema_gate_mean` record the forgetting each
run applied.

---

## Monitoring

Each run writes to `<logdir>/`: `metrics.jsonl` (scalars per logging step), TensorBoard
events with video predictions, `latest.pt`, replay episodes, and regime filmstrips and
figures when `shs_diag_figures` is on. With `shs_diag_log: True` the log includes:

| Metric | Meaning |
|---|---|
| `shs_current_K`, `shs_active_regimes`, `shs_active_regimes_window` | regime count; regimes with more than 1% of the responsibility mass on the diagnostic batch / on the move window |
| `shs_move_<kind>_accepted`, `shs_move_<kind>_gain` | outcome and proposal-screen gain of the last sweep per move kind |
| `shs_move_installed_gain` | bound gain of the installed edits of the last sweep on the fitting set |
| `shs_move_fit_vetoes`, `shs_move_holdout_vetoes` | proposals refused by the installed-state bound test / by the predictive test |
| `shs_move_holdout_base`, `shs_move_holdout_episodes` | validation score of the running model; number of validation episodes |
| `shs_move_sweeps_deferred` | sweeps skipped because no disjoint validation set could be formed (the reason is printed) |
| `shs_move_refit_gain` | bound gain of refitting the unchanged model on the window |
| `shs_empty_regimes_removed`, `shs_move_K_cap`, `shs_move_K_capped` | accepted removals of unoccupied regimes; the growth cap |
| `shs_ema_gate_mean`, `shs_ema_gate_at_floor` | mean responsibility gate; regimes at the floor rate |
| `shs_retention_<factor>_mean`, `shs_retention_<factor>_min` | posterior mean retention `E[r_k]` for `emission`, `transition`, `gate` (1 = memory kept) |
| `shs_drift_exposure_<factor>_mean` | mean responsibility share `u_k` that exposes a regime or row to drift |
| `shs_drift_prob_<factor>_max`, `shs_drift_rate_<factor>`, `shs_drift_events_<factor>` | largest drift probability of the last update; posterior mean drift rate per unit of usage; expected number of drifts so far |
| `shs_retention_quad_err`, `shs_retention_guard_hits` | largest numerical error estimate of the retention integrals; factor updates or regimes whose evidence could not be formed (treated as unused) |
| `shs_guard_reject_rate_hdp`, `shs_guard_reject_rate_pg` | share of HDP / gate updates refused as numerically invalid |
| `shs_curriculum_phase`, `shs_recurrent_live`, `shs_kappa_live`, `shs_move_every_live` | live curriculum state |
| `shs_imag_between_var_frac`, `shs_imag_top_weight`, `shs_imag_weight_entropy` | share of the one-step imagined variance due to regime disagreement; mixture weight statistics |

A `shs_drift_rate_emission` that stays high during stationary training indicates that the
encoder is moving faster than the regime statistics; a guard rejection rate above a few
percent means the regime prior is stale. `plot_returns.py` groups seeds by directory name
and plots evaluation returns across tasks.

---

## Offline experiments

### Switching-system benchmarks (`shs_demo/compare/`)

Fits SHS-RSSM, TrSLDS (Nassar et al., 2019) and rSLDS (Linderman et al., 2017) to `nascar`
(synthetic), `toyark13` (13 autoregressive regimes) and `mocap6` (annotated motion-capture
exercises); reports Hungarian-matched Hamming distance, purity, NMI and ARI.

```bash
cd shs_demo/compare
python run_all.py --datasets nascar toyark13 mocap6 --seeds 0 1 2 --latex
python run_rslds.py --dataset mocap6
python make_figures.py --dataset all --latex
```

`run_rslds.py` needs the rSLDS environment described under Installation.

### FitzHugh-Nagumo multi-step prediction (`shs_demo/fhn_benchmark.py`)

Protocol of Becker-Ehmck et al. (2019): 100 trajectories of length 430, last 30 steps held
out, normalised RMSE as a function of horizon.

```bash
python shs_demo/fhn_benchmark.py --models shs trslds --seeds 0 1 2 --shs-K 12
```

### Synthetic structure recovery (`sim_multiregime.py`)

A generator with a late-onset regime, a duplicated regime and a rare regime; reports the
recovered `K`, segmentation accuracy, dynamics error and bound monotonicity.

### Drift experiments (`shs_demo/drift/`)

CPU experiments on the streaming update of the conjugate factors, run directly against
`shs_rssm/regimes_shared.py`, `recurrent_stick.py`, `forward_backward.py` and `forgetting.py`
(their hashes are recorded in `checks.json`). Every experiment includes AIME's production
update (`retention_arm.py`): responsibility-gated gains, with and without change-point
retention on the emission, transition and gate statistics of each regime. The gated update
with retention is the proposed model (`shs_forget_mode: evidence`, "AIME (proposed)" in the
figures): τ is only its base rate, and regime k keeps a fraction (1 − τu_k)·E[r_k] of its
statistics per update. The plain EMA arms use one ungated rate, as in the tracking bound:

- `run_drift.py`: three identifiable linear regimes whose maps rotate and whose fixed points
  migrate, and a recurring condition in which one regime is absent for `rotation_steps`
  updates and then returns unchanged; compares the streaming update (two rates, with
  structure moves, gated, gated with change-point retention) against streaming VB, a sliding
  window and a cumulative mean by excess predictive NLL over the true dynamics (figure 3: the
  migrating condition over time, and the post-change mean in the stationary, migrating and
  recurring conditions), checks the population-level tracking bound and plots the learned
  basins and the retention and drift rate of each factor.
- `prop2_experiment.py`: latent-coordinate drift (the physics is fixed, the model sees a
  slowly rotating coordinate system); the ramp study of the tracking bound, a rate sweep,
  power-prior and hierarchical power-prior streaming VB (Masegosa et al., 2020), a matched
  sliding window, and the gated update with and without change-point retention.
- `prop2_segmentation_moving_mnist.py`: the same protocol on rendered Moving MNIST frames
  through a fixed PCA encoder, with an angle-wise batch-refit reference.

```bash
python shs_demo/drift/run_drift.py --out shs_demo/drift/results --seeds 20 --workers 16
python shs_demo/drift/prop2_experiment.py --out shs_demo/drift/results_prop2 --seeds 20 --workers 16
python -c "from torchvision.datasets import MNIST; MNIST('data', train=True, download=True)"
python shs_demo/drift/prop2_segmentation_moving_mnist.py --source-root . --mnist-root data \
  --out shs_demo/drift/results_visual --seeds 20 --workers 16
```

Each script writes per-seed `.npz` files, `summary.json`/`.csv`, LaTeX tables and figures
into `--out`, and skips seeds that are already there.

---

## References

- Hafner, Pasukonis, Ba & Lillicrap (2023). *Mastering diverse domains through world models* (DreamerV3).
- Fox, Sudderth, Jordan & Willsky (2011). *A sticky HDP-HMM with application to speaker diarization.*
- Hughes & Sudderth (2013); Hughes, Stephenson & Sudderth (2015). Memoized variational inference with birth, merge and delete moves for nonparametric mixtures and HMMs.
- Masegosa, Ramos-López, Salmerón, Langseth & Nielsen (2020). *Variational inference over nonstationary data streams for exponential family models.*
- Nagabandi, Finn & Levine (2019). *Deep online learning via meta-learning: continual adaptation for model-based RL* (MOLe).
- Bürkner, Gabry & Vehtari (2020). *Approximate leave-future-out cross-validation for Bayesian time series models.*
- Kass & Raftery (1995). *Bayes factors.*
- Linderman et al. (2017). *Bayesian learning and inference in recurrent switching linear dynamical systems*; Nassar et al. (2019). *Tree-structured recurrent switching linear dynamical systems.*
- Polson, Scott & Windle (2013). *Bayesian inference for logistic models using Pólya-Gamma latent variables.*

## License

This repository is released under the MIT License (see `LICENSE`). It includes code from
[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) (MIT, © NM512) and copies of the
rSLDS (Linderman et al.) and TrSLDS (Nassar et al.) reference implementations under their
original licenses.
