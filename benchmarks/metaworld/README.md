# Benchmarks

Per-domain configs. `dreamer.py` merges every `benchmarks/*/configs*.yaml` into
the same flat namespace as the root `configs.yaml`, so a config here is used
exactly like a built-in one. Name collisions raise at startup.

```bash
python3 -u dreamer.py --configs metaworld_proprio_shs --task metaworld_door-open \
    --logdir ./logdir/shs/door-open/seed0
```

| Folder | Status |
|---|---|
| `metaworld/` | working, extensively run |
| `eval/` | `aggregate.py` (IQM + bootstrap CIs), `summarize.py` (compact curves) |
| `dmc/` | control arms only; DMC configs are in the root `configs.yaml` |
| `atari100k/`, `crafter/` | configs only, never run |
| `minecraft/`, `procgen/` | scaffolding; `envs/procgen.py` is untested |

## The three-arm rule, and why one arm is broken

`use_shs: True` asserts `dyn_discrete == 0` — it swaps the dynamics prior AND
replaces the 32x32 categorical latent with a 32-dim Gaussian. So baseline-vs-SHS
changes two things. The intended control is `*_gauss` (Gaussian latent, stock
prior).

**That control does not work in this codebase.** Two independent 500k runs gave
`actor_grad_norm ~1e-6` and `actor_entropy` frozen at 5.675 (exactly
sigma = max_std) — the actor never trained. Bit-identical with and without
`torch.compile`. No original config combines `dyn_discrete: 0` with
`use_shs: False`, so `networks.RSSM`'s continuous branch appears to be dead
code. Until fixed, use a K=1 SHS config as the control.

## Results so far (door-open, seed 0, 500k steps)

| Arm | goals | success | return | % of expert (4492) |
|---|---|---|---|---|
| baseline | random | 0.00 | ~1000 | 22% |
| baseline | fixed | 0.00 | 1973 | 44% |
| SHS | random | 0.00 | 1089 | 24% |
| **SHS** | **fixed** | **1.00** | **4270** | **95%** |
| SHS + action_dim 4 | fixed | 1.00 | 4300 | 96% |

Two open caveats: the result exists only under fixed goals (the easier variant,
`_freeze_rand_vec=True`), and `shs_current_K` collapsed to 1 in every run, so
the switching prior was inert and the win is attributable to the
continuous-latent + sticky parameterisation rather than to switching.

`shs_init_scale` (new) addresses the second: all K regimes were previously
constructed identically, so one took ~all responsibility mass immediately and
`shs_active_regimes` read 1 from the first diagnostic, before any structure
move. See `shs_rssm/regimes_shared.py` and `metaworld/configs_switching.yaml`.

## Workflow

```bash
python3 benchmarks/metaworld/launch.py --suite suite6 --seeds 3   # inspect
sbatch --job-name=mw-shs-do benchmarks/metaworld/sbatch_run.sh \
    metaworld_proprio_shs door-open 0 500000 fixedgoal --mw_randomize_goal False
python3 benchmarks/eval/summarize.py door-open                     # compact curves
python3 benchmarks/eval/aggregate.py ./logdir --arms shs baseline \
    --metric eval_log_success --at 500000                          # IQM + CIs
```
