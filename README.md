# AIME · SHS-RSSM

A DreamerV3 world model whose transition prior is a sticky, nonparametric mixture of
linear-Gaussian regimes, learned online. The unmodified DreamerV3 baseline is in the same code.

## Architecture

AIME keeps DreamerV3 (encoder, decoder, GRU backbone, reward and continuation heads,
actor-critic, imagination, replay) and replaces only the neural transition prior:

```math
\begin{aligned}
x_t \mid z_t = k,\, h_t &\sim \mathcal{N}\!\left(M_k\,[x_{t-1};\, a_{t-1};\, 1] + C P h_t,\; Q_k\right),\\
z_t \mid z_{t-1} = j &\sim \sigma_j(\phi_t)\,\delta_j + \bigl(1 - \sigma_j(\phi_t)\bigr)\,\pi_j,\\
\pi_j &\sim \mathrm{Dir}(\alpha\beta + \kappa e_j),\\
\beta &\sim \mathrm{GEM}(\gamma).
\end{aligned}
```

- **Regimes.** `K` linear-Gaussian dynamics with Normal-Gamma posteriors, a shared map of the
  GRU carry and optional low-rank noise, updated in closed form from responsibility-weighted
  sufficient statistics. The neural parts train by gradient descent as in DreamerV3.
- **Switching.** A sticky HDP-HMM with a state-dependent persistence gate (Pólya-Gamma
  logistic regression); the regime path is inferred exactly by forward-backward.
- **Forgetting.** From each replay batch, regime $k$ updates its statistics as
  $\bar S_k \leftarrow (1 - \tau u_k)\,\mathbb{E}[r_k]\,\bar S_k + \tau u_k S_k$: $u_k$ is its share
  of the batch, so unused regimes keep their memory (`gated`, Nagabandi et al., 2019), and
  $\mathbb{E}[r_k]$ is the fraction of memory kept, inferred from the batch evidence under a
  change-point power prior (`retain`, Masegosa et al., 2020); `fixed` uses $u_k = r_k = 1$.
- **Structure adaptation** (`shs_structure_moves`). Birth, split, merge and delete moves on a
  window of recent batches, accepted only if they raise the variational bound of the installed
  model and pass a held-out latent-forecast test on episode-disjoint data.
- **Imagination.** Rolls out the regime mixture (`shs_imag_mode`: moment-matched or sampled).

## What is inside

```
dreamer.py            training loop (DreamerV3-torch) with signal-safe checkpoints and resume
transfer_walker.py    transfer runner: source checkpoint -> target task
models.py networks.py world model, actor-critic, encoders and decoders
tools.py parallel.py  logging, replay, utilities
configs.yaml          defaults, task presets, curricula and recipe overlays
envs/                 DMC, Atari, Procgen, Crafter, CARL, ... wrappers
shs_rssm/             the regime prior: regimes, sticky HDP and gate, forward-backward,
                      forgetting (forgetting.py), structure moves, diagnostics
shs_demo/             offline benchmarks (vs rSLDS, TrSLDS), FitzHugh-Nagumo, drift experiments
run_*.sh              Slurm launchers
```

## Install and run

```bash
pip install -r requirements.txt
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl        # osmesa for both on a CPU node
python dreamer.py --configs dmc_vision dmc_walker_shs shs_walker_recipe --seed 1 --logdir logdir/walker
python dreamer.py --configs dmc_vision dmc_vanilla_compare --task dmc_walker_walk --seed 1 --logdir logdir/walker_dreamer
python plot_returns.py --logdir logdir --metric eval_return --out returns.png
```

Configs apply left to right on top of `defaults`; any key can be overridden
(`--shs_K 24`, `--shs_forget_mode evidence`, ...). `dreamer.py` saves `<logdir>/latest.pt`
after each evaluation interval and, on `SIGUSR1`/`SIGTERM`, at the next 100-step boundary
(exit status 75); rerunning the same command continues from the checkpoint. A checkpoint
trained with other settings is refused (exit status 78).

## Experiments

| Launcher | Cluster | Runs |
|---|---|---|
| `run_dreamer_shs-rssm_dmc_best.sh` | Vulcan | `sbatch run_dreamer_shs-rssm_dmc_best.sh`: `fixed`, `gated`, `retain` × walker walk, finger turn hard, quadruped walk and run, reacher hard × 4 seeds (60 runs, 1M frames, one L40S, 4 CPUs, 24G each); `bash` on the script lists the commands |
| `run_dreamer_carl_mila.sh` | Mila | `mkdir -p logs && sbatch run_dreamer_carl_mila.sh`: five CARL tasks × `shs`/`shs_moves` × `gated`/`retain` × 4 seeds (80 runs), worked through in priority order by 12-hour lanes on `long`; `DRY_RUN=1 bash run_dreamer_carl_mila.sh` shows progress |
| `run_dreamer_shs-rssm_walker_transfer_mila.sh`, `..._vulcan_minimum.sh` | Mila, Vulcan | walker walk → walker run transfer (full, world model, scratch); set `FORGET` and `MOVES` |
| other `run_*.sh` | Mila, Vulcan | single tasks, Atari, DMC suites, Gaussian-latent DreamerV3 |

The DMC, CARL and Mila transfer launchers save ten minutes before the time limit, queue the
next slice and continue from the checkpoint until the run is done.

Hyperparameters of the DMC runs, taken from the best learning curves:

| Task | Presets |
|---|---|
| walker walk | `dmc_walker_shs shs_walker_recipe` |
| finger turn hard | `dmc_finger_turn_hard_shs_gate` with `shs_K 16`, `shs_q_rank 4`, `shs_b0 0.2`, `shs_calibrate_sF 0.1` |
| quadruped walk / run | `dmc_quadruped_{walk,run}_shs_gate shs_quadruped_recipe` (run: `shs_b0 0.1`) |
| reacher hard | `dmc_reacher_hard_shs` with `shs_q_rank 4` |

CARL walker and quadruped runs use the same recipe overlays. The walker and reacher schedules
start structure moves after 1M frames, so those runs keep `K` fixed.

## References

- Hafner et al. (2023). *Mastering diverse domains through world models* (DreamerV3).
- Fox et al. (2011). *A sticky HDP-HMM with application to speaker diarization.*
- Hughes, Stephenson & Sudderth (2015). *Scalable adaptation of state complexity for nonparametric hidden Markov models.*
- Masegosa et al. (2020). *Variational inference over nonstationary data streams for exponential family models.*
- Nagabandi, Finn & Levine (2019). *Deep online learning via meta-learning: continual adaptation for model-based RL.*
- Linderman et al. (2017); Nassar et al. (2019). Recurrent and tree-structured switching linear dynamical systems.
- Polson, Scott & Windle (2013). *Bayesian inference for logistic models using Pólya-Gamma latent variables.*

## License

MIT (see `LICENSE`). Includes code from [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
(MIT, © NM512) and the rSLDS and TrSLDS reference implementations under their original licenses.