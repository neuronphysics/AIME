# SHS-RSSM: A Sticky-HDP Switching World Model for DreamerV3

This repository implements the **Sticky Hierarchical-Dirichlet-Process Switching Recurrent State-Space Model (SHS-RSSM)**, a structured Bayesian nonparametric replacement for the latent dynamics prior of DreamerV3. The codebase is a fork of [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), the PyTorch implementation of DreamerV3 [1], and trains end to end with the standard Dreamer actor-critic pipeline on DeepMind Control Suite, Atari 100k, Crafter, Memory Maze, and Minecraft.

Stock DreamerV3 can be trained from this repository unchanged (`use_shs: False`, the default), so the switching model can be evaluated against its exact baseline under identical training code.

## Motivation

The DreamerV3 world model learns a single amortized transition prior over its latent state. This is a strong general-purpose choice, but it represents all of the environment's dynamics with one homogeneous conditional density. Many control domains are instead piecewise regular: locomotion alternates among stance, swing, contact, and recovery phases, each locally coherent and recurring. SHS-RSSM makes this structure explicit. A discrete regime variable selects among a set of Bayesian linear-Gaussian latent transition models, the regime sequence follows sticky hierarchical-Dirichlet-process Markov dynamics, and the number of regimes is itself adapted during training by data-driven structure moves.

## What is new relative to DreamerV3

DreamerV3 [1] and this model share the encoder, decoder, reward head, continuation head, actor, critic, replay, and imagination-based policy optimization. The differences are confined to the latent dynamics prior and its inference:

| Component | DreamerV3 | SHS-RSSM (this repository) |
|---|---|---|
| Latent state | 32 x 32 categorical | Continuous Gaussian (`dyn_discrete: 0`) |
| Transition prior | Single amortized network `p(z_t \| h_t)` | Mixture of `K` regime-conditional Bayesian linear-Gaussian transitions, marginalized over a discrete regime `s_t` |
| Regime dynamics | Not modeled | Sticky HDP-HMM transition posterior [3, 4] with a stick-breaking root, plus optional state-dependent recurrent stickiness [6, 9, 10] |
| Dynamics parameters | Point estimates via SGD | Conjugate variational posteriors over regime dynamics `(A_k, Q_k)`, updated in closed form |
| Number of latent modes | Fixed by architecture | Adapted online by birth, split, merge, and delete moves scored on a memoized variational bound [7, 8] |
| Sequence objective | Per-step KL balancing | Structured variational objective with an exact HMM forward-backward E-step over the regime chain |
| Imagination | Samples from the amortized prior | Bayesian imagination: propagates latent mean and covariance, samples the regime mixture, and includes parameter uncertainty |

The intent is a world model whose discrete structure is identifiable and interpretable (regimes align with behavioral phases), whose capacity adapts to the data, and whose uncertainty is carried consistently from inference into imagination.

## Model

At each step the world model maintains DreamerV3's deterministic carry `h_t` (GRU) and a continuous stochastic latent `z_t`. SHS-RSSM introduces a discrete regime `s_t in {1, ..., K}` and a regressor built from the previous latent and the carry:

```math
g_t = [\, z_{t-1},\; P h_t,\; 1 \,], \qquad
z_t \mid g_t,\, s_t = k,\, \Theta_k \;\sim\; \mathcal N\!\left(A_k g_t,\; Q_k\right).
```

The prior over `z_t` marginalizes the regime, and the actor and critic consume this mixture prior during imagined rollouts.

**Shared-carry parameterization** (`shs_shared_carry: True`, recommended). The carry-dependent drift is tied across regimes and regime-specific maps model only the residual dependence on `z_{t-1}`:

```math
z_t = C \tilde h_t + A_k r_t + \epsilon_t .
```

This closes a collapse mode in which each regime absorbs its own high-capacity carry map and the discrete variable becomes redundant.

**Process noise.** Each regime covariance is diagonal (`shs_q_rank: 0`) or low-rank plus diagonal, `Q_k = diag(d_k) + U_k U_k^T` (`shs_q_rank > 0`). In the low-rank case all inverse and log-determinant computations use Woodbury identities, so the off-diagonal precision structure enters the variational evidence, the structure-move scores, and Bayesian imagination.

**Regime transitions.** The base transition rows follow a sticky HDP-HMM variational posterior in the spirit of Fox et al. [4], with the proper stick-breaking treatment of the root weights from Hughes et al. [7]:

```math
\pi_i \sim \mathrm{Dir}(\alpha \beta + \kappa \delta_i).
```

**State-dependent recurrent stickiness.** Optionally (`shs_recurrent: True`), each origin regime `i` carries a logistic persistence model on a low-dimensional feature `phi_t` of the world-model state:

```math
\rho_{t,i} = \sigma(w_i^\top \phi_t + b_i), \qquad
M_t[i,j] = \rho_{t,i}\,\mathbf 1[j{=}i] + (1-\rho_{t,i})\,\pi_{ij}.
```

Dwell time therefore becomes input-dependent and regime-specific rather than governed by a single global `kappa`. The logistic weights receive statewise Polya-Gamma variational updates [9], following the recurrent-stickiness construction of the RS-HDP-HMM [6]; the per-regime separation of persistence from transition similarity follows the disentangled sticky HDP-HMM [5]. With `shs_rstick_stopgrad: True` the transition term does not backpropagate into the Dreamer GRU, which prevents recurrent stickiness from substituting for the switching dynamics.

The recurrent transition is treated fully variationally on both sides of the coordinate ascent. The E-step and every ELBO evaluation use the Polya-Gamma / Jaakkola-Jordan lower bound on the expected log augmented transition (`RecurrentStickiness.bound_log_trans`), evaluated at the coordinate-ascent-optimal PG parameter `c_{t,i}^2 = E_q[psi_{t,i}^2]`: the persistence branch contributes `m/2 + log sigma(c) - c/2`, the switch branch `-m/2 + log sigma(c) - c/2 + E[log pibar_{ij}]`, and the two branches are marginalized inside the potential by log-sum-exp. The base branch uses the raw sub-stochastic Dirichlet-posterior expectation `E[log pibar]` (never softmax-renormalized), so the forward-backward log-partition is a genuine lower bound on the marginal evidence, exactly mirroring Linderman et al.'s PG-augmented mean-field updates [10] transplanted to the disentangled-persistence parameterization [5, 6]. Pairwise marginals decompose exactly into persistence and switch branches (`attribute_bound`), giving the fractional-binomial statistics of the PG M-step and the base-branch counts of the non-sticky HDP update. The probit moment approximation survives only in generative imagination rollouts, where no bound is claimed.

**Inference.** The continuous posterior `q(z_t | h_t, x_t)` remains amortized by the Dreamer encoder. The regime posterior `q(s_{1:T})` is a structured chain computed exactly by forward-backward, yielding regime marginals `gamma_t(k)` and pairwise marginals `xi_t(i, j)`. Regime dynamics parameters have conjugate variational posteriors updated in closed form. A single fully Bayesian local evidence

```math
\ell_{t,k} = \mathbb E_{q(z_t)\, q(z_{t-1})\, q(\Theta_k)}\big[\log p(z_t \mid z_{t-1}, h_t, \Theta_k)\big]
```

is shared by the E-step, the dynamics loss, and structure-move scoring, and accounts for uncertainty in the target latent, the regressor, and the regime parameters. The dynamics loss is the negative structured variational objective over the latent trajectory and regime chain, including the HMM log normalizer and, when active, the recurrent-stickiness posterior KL.

**Adaptive regime complexity.** The truncation `K` adapts during training through Hughes-style structure moves [7, 8]: birth creates a regime from poorly explained timesteps, split re-differentiates a broad regime, merge combines redundant regimes, and delete removes regimes with negligible mass. Seeds are data-driven proposals in Hughes' sense; acceptance never is: a proposal lands only if it improves the exact structured variational bound aggregated over a memoized buffer of recent latent batches, with the global complexity terms (regime parameter KL, stickiness KL, HDP allocation with its exact linear slack) counted once per scored corpus. Merge selection uses Hughes' entropy-free test (`data_elbo_from_stats` + allocation, NIPS 2015 Sec. 4), which is guaranteed to shortlist every ELBO-improving pair because path entropy can only decrease under a merge; delete drops the state's statistics and count row/column and refits the candidate on the buffer so its mass is reassigned data-driven, before the exact whole-buffer verification. When recurrent stickiness is active, base and candidates are scored under the same PG/JJ-bounded recurrent ELBO with properly row-mapped, buffer-refit stickiness posteriors, so structure search remains exact under the full model (the earlier base-transition-only scoring limitation is removed). 

## Training curriculum

The reference configuration `dmc_walker_shs` stages the model within a single run. Phase boundaries are counted in world-model gradient steps and the live phase is logged (`shs_curriculum_phase`, `shs_move_every_live`, `shs_kappa_live`, ...):

1. **Formation.** Fixed `K`, strong sticky prior (`kappa`), no structure moves. The encoder and latent geometry converge before any structure search runs on top of them.
2. **Discovery.** Birth, split, merge, and delete are enabled under the plain variational bound; the stationary sticky base transition provides persistence.
3. **Full model.** Recurrent, state-specific stickiness switches on (`kappa` is internally zeroed to avoid double-counted persistence) while structure moves stay active.
4. **Stable full model.** All components remain active; the move cadence is reduced to lower overhead.

Boundaries in `configs.yaml` are placeholders and should be aligned with where the reconstruction and dynamics-KL curves flatten on your task.

## Installation

Python 3.10 or 3.11 is recommended with the pinned requirements. The Docker image below is the reference environment.

```bash
git clone <repository-url>
cd dreamerv3-shs-rssm
pip install -r requirements.txt
```

DeepMind Control Suite renders through MuJoCo with OSMesa by default (`MUJOCO_GL=osmesa` is set in `dreamer.py`). On a headless machine, either install OSMesa (`libosmesa6`) or wrap commands with the provided `xvfb_run.sh`. Setup scripts for Atari and Minecraft are in `envs/setup_scripts/`.

### Docker

```bash
docker build -f Dockerfile -t shs-rssm .
docker run -it --rm --gpus all -v "$PWD":/workspace shs-rssm \
  sh xvfb_run.sh python3 dreamer.py \
  --configs dmc_walker_shs --logdir ./logdir/dmc_walker_walk_shs
```

## Usage

Train SHS-RSSM on Walker Walk with the staged curriculum (action repeat 2, 4 parallel environments):

```bash
python3 dreamer.py --configs dmc_walker_shs --logdir ./logdir/dmc_walker_walk_shs
```

Train the unchanged DreamerV3 baseline on the same task:

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk \
  --logdir ./logdir/dmc_walker_walk_baseline
```

Apply SHS-RSSM to any other DMC vision task by overriding the two structural flags:

```bash
python3 dreamer.py --configs dmc_vision --task dmc_cheetah_run \
  --use_shs True --dyn_discrete 0 \
  --logdir ./logdir/dmc_cheetah_run_shs
```

Proprioceptive DMC, Atari 100k, and Crafter follow the upstream interface:

```bash
python3 dreamer.py --configs dmc_proprio --task dmc_walker_walk --logdir ./logdir/walker_proprio
python3 dreamer.py --configs atari100k --task atari_pong --logdir ./logdir/atari_pong
python3 dreamer.py --configs crafter --logdir ./logdir/crafter
```

Monitor training:

```bash
tensorboard --logdir ./logdir
```

Any key in `configs.yaml` can be overridden on the command line, for example `--seed 3`, `--steps 1e6`, or `--shs_K 8`. Named configs compose left to right after `defaults`.

## Key configuration flags

| Flag | Default | Meaning |
|---|---|---|
| `use_shs` | `False` | Swap the stock RSSM for SHS-RSSM. Requires `dyn_discrete: 0`. |
| `shs_K` | `16` | Initial regime truncation. |
| `shs_shared_carry` | `True` | Tie the carry drift across regimes (collapse fix). Structural; set at construction. |
| `shs_q_rank` | `0` | `0` for diagonal process noise, `> 0` for low-rank-plus-diagonal `Q_k`. Structural. |
| `shs_kappa` | `50.0` | Sticky self-transition bias of the base HDP transition. |
| `shs_recurrent` | `True` | Build the state-dependent recurrent-stickiness module. |
| `shs_rstick_dim` | `8` | Feature dimension for recurrent stickiness. |
| `shs_rstick_stopgrad` | `True` | Do not push transition gradients into the Dreamer GRU carry. |
| `shs_move_every` | `0` | `0` disables structure moves (fixed `K`); `> 0` runs a move sweep on that gradient-step interval after `shs_move_warmup`. |
| `shs_move_birth`, `shs_move_split` | `True` | Enable birth and split proposals (merge and delete are always considered during sweeps). |
| `shs_move_buffer` | `8` | Held-out minibatches over which the move bound is scored (memoized set). |
| `shs_move_confirm_top` | `null` | `null` confirms all merge/split candidates with the full bound; an integer shortlists for speed. |
| `shs_imag_sample_mixture` | `True` | Sample the true regime mixture in imagination rather than a moment-matched Gaussian. |
| `shs_analytic_estep` | `True` | E-step uses the expected log-likelihood under `q(z)` rather than a sample. |
| `shs_curriculum` | `[]` | In-run staging of `{move_every, recurrent, kappa, move_threshold, create_bonus}` by gradient step. |
| `shs_diag_log`, `shs_diag_figures` | `False`, `True` | Log regime diagnostics as scalars and PNG figures (occupancy, t-SNE, filmstrips) on the logging cadence. |

`shs_shared_carry` and `shs_q_rank` allocate buffers at construction and cannot be changed by the curriculum mid-run.

## Demonstrations

Two self-contained demonstrations exercise the switching machinery without a full Dreamer training run.

Regime discovery and disentanglement diagnostics on a synthetic moving-shapes dataset, using an untrained encoder so that any recovered structure is genuinely discovered:

```bash
PYTHONPATH=. python3 demo_shapes_disentangle.py
```

Figures are written to `./demo_outputs` (override with the `SHS_OUT` environment variable).

Full SHS-RSSM on a multi-object switching scene with a composition change, producing latent-clustering, reconstruction and open-loop imagination, and true-versus-inferred dynamics figures:

```bash
python3 shs_demo/recon_demo.py 2000
```

## Repository structure

| Path | Purpose |
|---|---|
| `dreamer.py` | Training entry point (agent loop, environments, logging). |
| `configs.yaml` | All named configurations, including `dmc_walker_shs`. |
| `models.py` | World model and behavior; selects stock RSSM or SHS-RSSM from the config. |
| `networks.py` | Stock DreamerV3 networks. |
| `shs_rssm/shs_rssm.py` | SHS-RSSM: RSSM subclass and training/imagination integration. |
| `shs_rssm/regime_head.py` | Regime inference, structured dynamics objective, Bayesian imagination. |
| `shs_rssm/regimes.py`, `shs_rssm/regimes_shared.py` | Regime-specific Bayesian linear dynamics; shared-carry variant. |
| `shs_rssm/sticky_hdp.py` | Sticky HDP transition posterior and stick-breaking root. |
| `shs_rssm/recurrent_stick.py` | State-specific recurrent stickiness with Polya-Gamma updates. |
| `shs_rssm/forward_backward.py` | HMM forward-backward with episode-reset handling. |
| `shs_rssm/moves.py` | Birth, split, merge, and delete structure moves with memoized scoring. |
| `shs_rssm/lowrank.py` | Low-rank-plus-diagonal Gaussian operations (Woodbury). |
| `shs_rssm/structured_elbo.py`, `shs_rssm/mixture_prior.py` | Structured sequence objective and mixture-prior KL. |
| `shs_rssm/shs_diagnostics.py`, `shs_rssm/shs_filmstrip.py`, `shs_rssm/shs_disentangle.py` | Regime diagnostics, filmstrip rendering, disentanglement metrics. |
| `envs/` | Environment wrappers (DMC, Atari, Crafter, Memory Maze, Minecraft, synthetic shapes). |

## Reproducibility

Runs are seeded with `--seed`; `--deterministic_run True` additionally enables deterministic PyTorch kernels. `torch.compile` is disabled automatically when `use_shs: True` because the model performs Python-side conjugate updates, buffer mutation, and shape-changing structure moves that the compiler cannot trace.

## References

1. D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. Mastering Diverse Domains through World Models. arXiv:2301.04104, 2023.
2. Y. W. Teh, M. I. Jordan, M. J. Beal, D. M. Blei. Hierarchical Dirichlet Processes. Journal of the American Statistical Association, 101(476), 2006.
3. M. J. Beal, Z. Ghahramani, C. E. Rasmussen. The Infinite Hidden Markov Model. NeurIPS, 2002.
4. E. B. Fox, E. B. Sudderth, M. I. Jordan, A. S. Willsky. A Sticky HDP-HMM with Application to Speaker Diarization. Annals of Applied Statistics, 5(2A), 2011.
5. D. Zhou, Y. Gao, L. Paninski. Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model. ECML-PKDD, 2020. arXiv:2004.03019.
6. M. Słupiński, P. Lipiński. The Recurrent Sticky Hierarchical Dirichlet Process Hidden Markov Model. arXiv:2411.04278, 2024.
7. M. C. Hughes, D. I. Kim, E. B. Sudderth. Reliable and Scalable Variational Inference for the Hierarchical Dirichlet Process. AISTATS, 2015.
8. M. C. Hughes, W. Stephenson, E. B. Sudderth. Scalable Adaptation of State Complexity for Nonparametric Hidden Markov Models. NeurIPS, 2015.
9. N. G. Polson, J. G. Scott, J. Windle. Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables. Journal of the American Statistical Association, 108(504), 2013.
10. S. W. Linderman, M. J. Johnson, A. C. Miller, R. P. Adams, D. M. Blei, L. Paninski. Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems. AISTATS, 2017.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dreamerv3_shs_rssm,
  author = {Sheikhbahaee, Zahra},
  title  = {SHS-RSSM: A Sticky-HDP Switching State Space World Model for DreamerV3},
  year   = {2026},
  url    = {https://github.com/neuronphysics/AIME.git}
}
```

## License and acknowledgements

Released under the MIT License (see `LICENSE`). This repository builds on [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) by NM512, which reimplements DreamerV3 by Hafner et al. [1]; the stock training pipeline, environment wrappers, and baseline networks originate there. The SHS-RSSM model, structured variational inference, structure moves, recurrent stickiness, diagnostics, and curriculum are contributions of this repository.

## Claim scope (what is and is not variational)

Variational, with entropy/KL terms in one bound and exact/bounded coordinate updates:
HDP sticks u, transition rows pi-bar, recurrent-stick weights beta (PG/JJ), regime
dynamics (M_k, Q_k), the start distribution, the exact forward-backward switch posterior
q(s), and the persistence indicators w. Point-estimated or heuristic, documented and
optional where feasible: neural encoder/decoder/GRU weights, the amortized factorized
q(z), ARD empirical Bayes (`ard=False` to freeze; the source of ELBO non-monotonicity in
the Lorenz attribution study), optional ML low-rank process noise, the frozen stickiness
projection, stop-grads / variance caps / curricula (rejected by the strict-ELBO
profile). Monte Carlo appears only in generative rollouts and optional diagnostics;
fitting and move acceptance are deterministic. 

## Benchmarks

Per-domain configs live in `benchmarks/<domain>/configs.yaml` and are merged into
the same named-config namespace as the root `configs.yaml` at startup, so
`--configs metaworld_proprio_shs` works exactly like a built-in config. Name
collisions raise rather than silently override. See `benchmarks/README.md`.

Meta-World (`benchmarks/metaworld/`) is new: 50 manipulation tasks, a wrapper
verified against the real `metaworld` package, the Seo et al. difficulty tiers,
a sweep launcher, and an aggregation script that reports IQM with stratified
bootstrap CIs from `metrics.jsonl`.

Note that `use_shs: True` requires `dyn_discrete: 0`, so an SHS-vs-DreamerV3
comparison changes both the dynamics prior and the latent type. Each benchmark
folder therefore also defines a `*_gauss` control arm (continuous latent, stock
amortised prior) which is what makes a difference attributable to the switching
prior. Details in `benchmarks/README.md`.
