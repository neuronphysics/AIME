# SHS-RSSM

A switching state-space world model for DreamerV3.

SHS-RSSM replaces DreamerV3's latent transition prior with a Bayesian nonparametric
switching linear dynamical system. A discrete regime variable selects among `K`
linear-Gaussian transition models, the regime sequence follows sticky HDP-HMM
dynamics with optional state-dependent persistence, and `K` adapts during training
through birth, merge and delete moves accepted on a variational bound.

The repository is built on [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch).
Everything outside the transition prior is unchanged, and baseline DreamerV3 trains
from this repository with `use_shs: False`, so the two models can be compared under
identical training code.

## Model

Continuous latent $z_t$, deterministic carry $h_t$ from the GRU, discrete regime $s_t$:

```math
\begin{aligned}
s_t \mid s_{t-1} &\sim \pi_{s_{t-1}}
  &&\text{sticky HDP-HMM} \\
z_t \mid s_t, g_t &\sim \mathcal{N}\!\left(A_{s_t} g_t,\ Q_{s_t}\right),
  \qquad g_t = \left[\, z_{t-1},\ h_t,\ a_{t-1},\ 1 \,\right] \\
o_t \mid z_t, h_t &\sim \mathrm{decoder}(z_t, h_t)
  &&\text{unchanged from DreamerV3}
\end{aligned}
```

Hierarchical Dirichlet process prior on the transition matrix:

```math
\beta \sim \mathrm{GEM}(\gamma),
\qquad
\hat\theta_{k\ell} = M_{k\ell} + \alpha\, \mathbb{E}\!\left[\beta_\ell\right]
                     + \kappa\, \mathbb{1}\!\left[k = \ell\right],
\qquad
(A_k, Q_k) \sim \mathrm{MNIW}
```

With `shs_recurrent: True` the self-persistence probability becomes state- and
context-dependent, $\rho_{k,t} = \sigma\!\left(w_k^\top \phi(h_t)\right)$, rendered
conjugate by Pólya-Gamma augmentation; the HDP's own stickiness is set to zero in
that mode so that persistence is modelled in one place.

Inference is structured variational: exact forward–backward over $s_{1:T}$,
closed-form conjugate M-steps for $(A_k, Q_k)$ and the transition Dirichlets, a
Pólya-Gamma-conjugate update for the persistence gate, and a numerical update for
the stick-breaking root $q(u_k) = \mathrm{Beta}\!\left(\hat\rho_k \hat\omega_k,\
(1 - \hat\rho_k)\hat\omega_k\right)$. The bound decomposes as

```math
\mathcal{L} = \mathcal{L}_{\mathrm{data}} + \mathcal{L}_{\mathrm{entropy}}
            + \mathcal{L}_{\mathrm{alloc}} + \mathcal{L}_{\mathrm{slack}}
```

Structure moves are accepted only when they increase $\mathcal{L}$.

## Layout

```text
dreamer.py            training loop, unchanged from dreamerv3-torch
models.py             world model; routes to SHS-RSSM when use_shs is set
networks.py           encoder, decoder, RSSM, actor, critic
configs.yaml          hyperparameters and presets
run_benchmark.sh      offline benchmark sweep (SLURM or plain bash)

shs_rssm/
  shs_rssm.py         drop-in replacement for the RSSM
  regime_head.py      E-step, global variable updates, variational bound
  regimes.py          per-regime linear dynamics, conjugate M-step
  regimes_shared.py   shared-carry / low-rank variant
  sticky_hdp.py       HDP transition prior, stick-breaking root
  forward_backward.py message passing over the regime chain
  recurrent_stick.py  state-dependent persistence (Pólya-Gamma gate)
  continuous_smoother.py  information-form smoother for the continuous state
  moves.py            birth, merge, delete, and the move sweep
  delete_planner.py   delete eligibility and target selection
  offline_trainer.py  frozen-corpus fitting and consolidation
  online_vb.py        streaming / EMA sufficient-statistic store
  init_data.py        contiguous-block initialisation
  checks.py           runtime invariant assertions

shs_demo/
  fhn_demo.py         FitzHugh-Nagumo demonstration
  toyark13/           13-regime autoregressive benchmark (Hughes protocol)
  mocap6/             six annotated CMU motion-capture sequences
  compare/            baseline comparison harness (SHS vs TrSLDS vs rSLDS)
  trslds/, rslds/     vendored baselines, byte-identical to upstream
```

## Running

```bash
pip install -r requirements.txt
```

`polyagamma` is required only by the TrSLDS/rSLDS baselines in `shs_demo/compare/`.
On clusters without outbound network access from compute nodes, install it from a
login node first.

### 1. Reinforcement learning (DreamerV3 with the switching prior)

```bash
python dreamer.py --configs dmc_vision --task dmc_walker_walk                 # baseline DreamerV3
python dreamer.py --configs dmc_vision --task dmc_walker_walk --use_shs True  # SHS-RSSM
python dreamer.py --configs dmc_humanoid_shs --seed 1 --steps 3000000         # tuned preset
```

Runs resume from `--logdir`, so a multi-day target can be chained across jobs. Use
one logdir per seed; two jobs sharing a logdir will continue each other's
checkpoints.

### 2. Segmentation benchmarks (offline, frozen corpus)

The comparison harness fits SHS-RSSM's regime head and the published baselines on
the same data and writes tables and figures:

```bash
cd shs_demo/compare
python run_shs.py    --dataset toyark13 --seed 0 --recurrent --tag shs_seed0
python run_trslds.py --dataset toyark13 --seed 0 --tag trslds_seed0
python make_figures.py --dataset all --latex
```

Datasets: `toyark13`, `nascar`, `mocap6`. Each run writes
`results/<dataset>/<tag>.npz`; `make_figures.py` aggregates whatever is present into
`results/<dataset>/table.{csv,md,tex}` and `figures/`. Metrics are label-based
(Hamming after Hungarian matching, many-to-one, NMI, ARI); model objectives are
recorded but are not comparable across model classes.

The whole sweep, across three seeds, with a summary report:

```bash
sbatch run_benchmark.sh
FRESH=1 sbatch run_benchmark.sh  # archive previous results/ and figures/ first
```

Finished fits are skipped on re-submission (`FORCE=1` redoes them), so an
interrupted sweep resumes. rSLDS needs the 2017 Linderman stack and runs from a
separate environment; see `shs_demo/compare/environment-baselines.yml`. Its results
merge into the same `results/` folder.

Note on the gate in this harness: the offline runner supplies no deterministic
carry ($h_t$ is a zero channel), so $\phi(h_t)$ is constant and `--recurrent`
reduces to a learned constant persistence per state, $\rho_k = \sigma(b_k)$.
Genuine context dependence requires feeding an informative signal in the `deter`
slot — the previous observation, or a smoothed speed feature. Inside DreamerV3 the
carry is the real GRU state, so the gate is context-dependent there.

### 3. Protocol reproduction

```bash
cd shs_demo/toyark13
NSEQ=12 LAPS=30 python run_hughes_protocol.py
```

Over-provisions at `K=25` with contiguous-block initialisation and runs
merge/delete/birth on a frozen corpus, printing `K` and Hamming per lap.

## Worked example: mocap6

Six CMU motion-capture sequences, 12 annotated exercise behaviours, 12 channels.
Ground truth is 37 segments with a median duration of 51 frames — long, slow
behaviours. The example is instructive because the two things that can go wrong
are independent: the model can find the wrong *number* of regimes, and it can find
the right number but switch between them too quickly.

```bash
cd shs_demo/compare

# (a) package defaults
python run_shs.py --dataset mocap6 --seed 0 --recurrent --tag m_default

# (b) recovers the annotated regime count
python run_shs.py --dataset mocap6 --seed 0 --recurrent \
    --b0-mode calibrate --sF 0.1 --tag m_k12

# (c) as (b), with a firmer persistence prior
python run_shs.py --dataset mocap6 --seed 0 --recurrent \
    --b0-mode calibrate --sF 0.1 \
    --prior-persist 0.98 --bias-prior-var 0.25 --tag m_persist

python make_figures.py --dataset mocap6
```

What each flag does. `--b0-mode calibrate --sF 0.1` sets the Normal-Gamma noise
rate from the data as $b_0 = 0.1 \cdot \mathrm{Var}(x)$ instead of the package
default $b_0 = 2.0$: a tighter noise prior makes regimes more distinguishable, and
it is the ingredient that lets `K` settle at the annotated 12 rather than
collapsing to 5–6. Run (b) starts from `K0 = 20` and prunes by merge and delete —
the count is inferred, not imposed. `--prior-persist` sets the gate's prior mean
self-persistence and `--bias-prior-var` how firmly that prior is held: the default
pair $(0.9,\ 4.0)$ puts the logit bias at $2.20 \pm 2.0$, i.e. persistence anywhere
in $[0.55,\ 0.985]$, which thousands of per-frame likelihood terms easily overrule;
$(0.98,\ 0.25)$ holds the gate near a long dwell unless the data insists otherwise.

Reading the result. `results/mocap6/table.md` gives `K used` and the label metrics.
Segment durations are the diagnostic for flicker and are worth checking directly:

```python
import numpy as np

z = np.load("results/mocap6/m_k12.npz", allow_pickle=True)["z_pred"]
seg = np.diff(np.flatnonzero(np.r_[True, z[1:] != z[:-1], True]))
print(len(seg), np.median(seg), (seg < 5).mean())   # target: ~37, ~51, ~0
```

Many-to-one accuracy near 0.65 with a one-to-one Hamming near 0.50 is the
signature of correct labels at the wrong timescale: the states are right, the
transitions are too frequent. If durations stay short at every setting of the
persistence prior, the limitation is the emission model rather than the prior —
a first-order AR with diagonal noise cannot represent a volatile regime, so the
fit uses extra state switches in its place. `--q-rank 3` (correlated noise) and a
higher AR order address that directly.

## Configuration

| setting | meaning |
|---|---|
| `use_shs` | route the transition prior through SHS-RSSM |
| `shs_K` | truncation (an upper bound, not a fixed count) |
| `shs_kappa` | sticky-HDP self-transition mass (forced to 0 when recurrent) |
| `shs_alpha`, `shs_gamma` | HDP concentration parameters |
| `shs_prior_persist` | prior mean persistence for the recurrent gate |
| `shs_rstick_bias_var` | prior variance of the gate's logit bias; small = firmly persistent |
| `shs_recurrent` | state-dependent persistence via the Pólya-Gamma gate |
| `shs_move_every` | structure-move interval (0 disables live moves) |
| `shs_consolidate_every_episodes` | frozen-representation consolidation interval |
| `shs_learn_b0` | conjugate Gamma hierarchy on the emission noise rate |
| `shs_q_rank` | rank of correlated emission noise (0 = diagonal) |
| `shs_online_mode` | `ema` (streaming) or `memoized` (Hughes replace semantics) |
| `shs_strict_elbo` | enforce the single-objective contract (see Scope) |

Two settings deserve attention on new data. Initialisation: the automatic
contiguous-block length (about $T/2K$) is short relative to long behavioural
segments, so `shs_demo/compare/run_shs.py` defaults mocap6 to `init_block=100`.
Emission tightness: the Normal-Gamma rate $b_0$ sets segmentation granularity;
`regimes.calibrate_b0_from_data(z, sF)` follows the bnpy convention
$b_0 = s_F \cdot \mathrm{Var}(x)$, and `shs_learn_b0` replaces the hand-set value
with a conjugate hierarchy that tracks the representation as it drifts.

## Structure moves

`K` is a truncation, not a fixed count. Moves propose changes and are accepted only
if the bound improves.

- **Birth** seeds two regimes on a subwindow of a contiguous block of the current
  segmentation, splitting at the cut that maximises the conjugate marginal evidence.
- **Merge** combines a pair by statistic surgery,
  $M'_{ii} = M_{ii} + M_{jj} + M_{ij} + M_{ji}$; the recurrent gate's Pólya-Gamma
  statistics are additive in the same way, so the merged gate row is their sum,
  refit and refined before scoring.
- **Delete** removes a rarely-used regime and redistributes its mass, refining only
  on the sequences that used it.

The sweep runs prune-first: merge, delete, then birth and split.

Following Hughes, Stephenson & Sudderth (2015), exact acceptance requires a fixed
corpus: statistics computed under different encoder versions are not comparable.
Run moves through `offline_trainer.py` on a frozen snapshot, or through scheduled
consolidation during training.

## Scope

Three operating modes carry different guarantees, worth distinguishing when
reporting results:

- **Offline, frozen corpus** (`offline_trainer.py`, the `compare/` harness) —
  coordinate ascent on one bound over a complete corpus, with whole-corpus move
  acceptance. This is the mode the variational guarantees describe.
- **Consolidation during training** — the representation is frozen, replay is
  re-encoded under it, and moves are accepted on that frozen corpus, restoring
  exact acceptance inside a training loop.
- **Default online training** — EMA sufficient statistics, free bits, Dreamer's KL
  scaling, a moving encoder, and optionally live moves on a recent window. These
  are engineering choices that work well in practice, but they are a
  forgetting-factor approximation rather than coordinate ascent on one fixed
  objective, and live moves on an incomplete buffer are a recent-window
  approximation.

`shs_strict_elbo` tightens the contract toward the first mode. Memoized mode
additionally requires a stable batch identifier per replay partition: call
`SHSRSSM.set_batch_id(...)` with your replay partition id and set
`shs_expected_batches` before using it. The head refuses an update with no batch id
rather than appending a new batch on every replay visit, which would silently break
the replace semantics that memoization depends on.

Without structure moves a single fit under-uses its truncation on hard
autoregressive data: moves are the mechanism by which `K` is recovered, not a
refinement of it. Treat inferred `K` from a move-free fit as a lower bound.

## References

- Hughes, Stephenson & Sudderth. Scalable adaptation of state complexity for
  nonparametric hidden Markov models. NIPS 2015.
- Bryant & Sudderth. Truly nonparametric online variational inference for
  hierarchical Dirichlet processes. NIPS 2012.
- Fox, Sudderth, Jordan & Willsky. A sticky HDP-HMM with application to speaker
  diarization. Annals of Applied Statistics, 2011.
- Zhou, Gao & Paninski. Disentangled sticky hierarchical Dirichlet process hidden
  Markov model. ECML PKDD 2020.
- Linderman, Johnson, Miller, Adams, Blei & Paninski. Bayesian learning and
  inference in recurrent switching linear dynamical systems. AISTATS 2017.
- Nassar, Linderman, Bugallo & Park. Tree-structured recurrent switching linear
  dynamical systems for multi-scale modeling. ICLR 2019.
- Costacurta, Duncker, Sheffer, Gillis, Weinreb, Markowitz, Datta, Williams &
  Linderman. Distinguishing discrete and continuous behavioral variability using
  warped autoregressive HMMs. 2022.
- Polson, Scott & Windle. Bayesian inference for logistic models using Pólya-Gamma
  latent variables. JASA, 2013.
- Hafner, Pasukonis, Ba & Lillicrap. Mastering diverse domains through world
  models. 2023.

## License

MIT, as in the upstream dreamerv3-torch repository.
