# Benchmarks

Each subfolder holds the named configs for one domain. `dreamer.py` merges every
`benchmarks/*/configs.yaml` into the same flat namespace as the root
`configs.yaml`, so a config defined here is used exactly like a built-in one:

```bash
python3 dreamer.py --configs metaworld_proprio_shs --task metaworld_assembly \
    --logdir ./logdir/shs/assembly/seed0
```

Name collisions with the root `configs.yaml` raise at startup rather than
silently overriding, so nothing that already works can be shadowed.

| Folder | Status | Notes |
|---|---|---|
| `metaworld/` | **new, smoke-tested** | 50 manipulation tasks. Wrapper verified against the real `metaworld` package. |
| `dmc/` | control arms only | DMC configs live in the root `configs.yaml`; this folder adds the missing continuous-latent controls. |
| `atari100k/` | configs only | Confounded, see below. |
| `crafter/` | configs only | Confounded, see below. |
| `procgen/` | scaffolding | `envs/procgen.py` is new and untested end to end; `procgen` is not in `requirements.txt`. |
| `minecraft/` | scaffolding | 100M steps x 10 seeds in the original paper. Budget before starting. |
| `eval/` | **new** | `aggregate.py`: IQM / median / mean with stratified bootstrap CIs from `metrics.jsonl`. |

## The three-arm rule

`use_shs: True` asserts `dyn_discrete == 0`. It does not just swap the dynamics
prior — it also replaces DreamerV3's 32x32 categorical latent with a continuous
Gaussian one. So a `baseline` vs `shs` comparison changes **two** things at once.

Every domain folder therefore defines three arms:

| Arm | Latent | Prior |
|---|---|---|
| `baseline` | categorical (32x32) | amortised |
| `gauss` | continuous Gaussian | amortised |
| `shs` | continuous Gaussian | switching (SHS-RSSM) |

Without the `gauss` arm you cannot say whether a difference came from the
switching prior or from the latent type. On DMC the latent swap is cheap; on
discrete-action pixel domains (Atari, ProcGen, Crafter, Minecraft) the
categorical latent is one of the components DreamerV3's own ablations credit
heavily, so the confound there is large and runs in the direction that hurts you.

## Which domains are worth running

**Meta-World is the strongest target.** It is not in the DreamerV3 paper, so
there is no official baseline to beat — every published DreamerV3 Meta-World
number is somebody else's re-run, and those re-runs are consistently weak
(TD-MPC2's appendix notes DreamerV3 often fails to converge; other papers report
0.0 success on several tasks even at 1M steps). Clearing that bar is likely but
a reviewer will point out the baseline was under-tuned. Run your own baseline in
this codebase (`use_shs: False`, identical pipeline) and report that.

It is also the best *scientific* fit: manipulation is genuinely phase-structured
(approach / contact / grasp / transport / release), which is closer to the
switching-regime inductive bias than walker gait phases are. Regime occupancy
aligning with manipulation phases is a real interpretability result independent
of the score.

**Atari / ProcGen / Crafter / Minecraft** cost three arms each, hit the
categorical-latent confound, and run with `torch.compile` disabled (the SHS
model does Python-side conjugate updates and shape-changing structure moves that
cannot be traced). DreamerV3's own ProcGen numbers use one seed per game, which
makes it noisy in both directions as a target. If you want one discrete-domain
data point, Crafter is the cheapest: 1M steps, single env.

## Workflow

```bash
# 1. See the sweep before running it
python3 benchmarks/metaworld/launch.py --suite suite15 --seeds 3
#    -> 135 jobs (15 tasks x 3 arms x 3 seeds)

# 2. Pilot one task first to place the curriculum boundaries
python3 dreamer.py --configs metaworld_proprio_shs --task metaworld_assembly \
    --logdir ./logdir/pilot

# 3. Launch
python3 benchmarks/metaworld/launch.py --sbatch | bash

# 4. Aggregate
python3 benchmarks/eval/aggregate.py ./logdir --arms shs baseline \
    --metric eval_log_success --at 1000000
```
