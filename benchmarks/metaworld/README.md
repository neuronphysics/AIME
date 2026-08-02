# Meta-World

50 Sawyer manipulation tasks. Not part of the DreamerV3 paper — see
`../README.md` for what "beating DreamerV3 here" does and does not mean.

## Setup

```bash
pip install metaworld
```

The wrapper (`envs/metaworld.py`) targets the Farama `metaworld` package and
prefers the v3 task registry, falling back to v2 if only that is present. It
converts gymnasium's 5-tuple `step` to the old-gym 4-tuple this repo's rollout
loop expects.

Rendering needs a GL backend. `dreamer.py` sets `MUJOCO_GL=osmesa`; on a
headless node either install `libosmesa6` or use `sh xvfb_run.sh`. Proprioceptive
runs set `mw_render: false` and skip rendering entirely.

## Configs

| Config | Obs | Latent | Prior |
|---|---|---|---|
| `metaworld_proprio` | 39-d state | categorical | amortised |
| `metaworld_proprio_gauss` | 39-d state | Gaussian | amortised |
| `metaworld_proprio_shs` | 39-d state | Gaussian | SHS-RSSM |
| `metaworld_vision` | 64x64 RGB | categorical | amortised |
| `metaworld_vision_shs` | 64x64 RGB | Gaussian | SHS-RSSM |

Start with proprio. It is 3-5x faster (no MuJoCo render per step), it isolates
the dynamics prior from representation learning, and the phase structure the
switching prior is supposed to exploit is present in the state observation
regardless. Add vision only once proprio shows something.

## Task selection

`tasks.py` holds the Seo et al. (MWM) difficulty partition — easy (28),
medium (11), hard tier (11) — and two pre-committed suites.

```bash
python3 tasks.py --suite suite15 --with-steps   # recommended, 5 per tier
python3 tasks.py --suite suite6                 # minimum, skewed hard
python3 tasks.py --suite all                    # all 50
```

**Fix the task list before you look at any results.** It is the cheapest thing
you can do to make the comparison credible and the first thing a reviewer asks
about. `SUITE_15` is deliberately not cherry-picked: the easy five are
near-ceiling for everything and exist to catch regressions, not to generate a
win.

Step budgets follow Seo et al.: 500K / 1M / 2M env steps at action repeat 2 for
easy / medium / hard.

Provenance note: the easy and medium lists are published verbatim. The hard tier
is the remaining 11 tasks; 7 of the 11 hard-vs-very-hard assignments are
confirmed from secondary sources, 4 are not. Report the 11 as one tier unless
you have checked Seo et al. Appendix F yourself.

## Protocol decisions baked into the wrapper

Three choices that change the numbers. All are flags; the defaults are the
harder, more defensible setting.

**Goal randomisation (`mw_randomize_goal`, default `True`).** The
`*-goal-observable` classes ship with `_freeze_rand_vec = True`, which silently
pins the goal to one position for the entire run — verified: two consecutive
resets return an identical goal. That is the *easy* single-goal variant. The
default here randomises per episode, matching TD-MPC / TD-MPC2, which describe
the goal-conditioned version as harder than the single-goal variant often used
in related work. If you are comparing against a paper that used fixed goals,
set this to `False` or the comparison is invalid in your favour.

**No terminal state (`mw_terminate_on_success`, default `False`).** Meta-World
episodes end by time limit; success does not terminate. `is_terminal` is
therefore always `False`, so the continuation head is not taught a spurious
absorbing state. Terminating on success changes the benchmark definition and
makes numbers non-comparable.

**Camera (`mw_camera`, default `corner2`).** The modified `corner2` viewpoint
used by MWM, MoDem, and TD-MPC2, including the standard
`cam_pos[2] = [0.75, 0.075, 0.7]` adjustment so the workspace fits a 64x64 frame.

## Success metric — read this before changing `mlp_keys`

Success is emitted as `log_success`, raised exactly once (the first success
step), so the episode sum is a 0/1 "did this episode ever succeed" indicator —
the standard Meta-World success rate. `tools.simulate` strips `log_` keys before
the agent sees them.

**However:** `save_episodes` writes the cache *before* the `log_` keys are
popped, so `log_success` is present in the replay `.npz`. With
`mlp_keys: '.*'` — which is what `dmc_proprio` uses — the encoder would read the
success flag as an observation and the agent could learn to look up the answer.
The Meta-World configs pin `mlp_keys: 'state'` for exactly this reason. Do not
relax that regex.

## Curriculum boundaries are placeholders

The `shs_curriculum` phase boundaries (10k / 20k / 30k world-model gradient
steps) are copied from `dmc_walker_shs`. The root README is explicit that these
should sit where reconstruction and dynamics-KL curves flatten, and manipulation
almost certainly flattens somewhere else. Run one pilot task, look at
`image_loss` / dynamics KL and the logged `shs_curriculum_phase`, and re-fit
before launching the sweep. Launching 135 jobs on DMC-derived boundaries is the
most likely way to burn a week for nothing.

## Full workflow

```bash
python3 benchmarks/metaworld/launch.py --suite suite15 --seeds 3   # inspect
python3 benchmarks/metaworld/launch.py --sbatch | bash             # launch
python3 benchmarks/eval/aggregate.py ./logdir --arms shs baseline \
    --metric eval_log_success --at 1000000
```
