# Changes: Meta-World benchmark + evaluation tooling

Diffed against the pristine `AIME-main.zip`. Five existing files touched
(101 lines added, 6 changed), 13 new files.

## Modified files

### `tools.py` (+21) — **bug fix, review first**

Per-episode `log_*` metrics were written with `logger.scalar` on every episode,
but the eval branch flushes the logger exactly once, after the final episode.
Each call overwrote the previous, so an eval `log_*` metric recorded only the
**last episode's value**. For a 0/1 metric like Meta-World success that is a
coin flip, not a rate over `eval_episode_num` episodes.

Now: per-episode values are collected into `episode_logs`; train writes them per
episode as before (train flushes every episode, so nothing changes there); eval
accumulates into `eval_logs` and emits the mean as `eval_<key>` at flush time.

New metric name: **`eval_log_success`**. Affects any env emitting `log_*`
(Crafter's achievement logs go through the same path).

### `configs.yaml` (+15, 6 changed)

New `defaults` keys so argparse registers them: `mw_camera`, `mw_render`,
`mw_randomize_goal`, `mw_terminate_on_success`, `procgen_distribution_mode`,
`procgen_num_levels`, `procgen_start_level`.

Pre-existing bugs fixed:

| Config | Was | Now | Effect |
|---|---|---|---|
| `crafter` | `step: 1e6` | `steps: 1e6` | none (default was already 1e6) |
| `minecraft` | `step: 1e8` | `steps: 1e8` | **ran to 1e6, 100x short** |
| `crafter`, `minecraft` | `value: {layers: 5}` | `critic: {layers: 5}` | code reads `config.critic`; the 5-layer critic was never applied |

### `dreamer.py` (+40)

`make_env`: two new suites, `metaworld` and `procgen`. No existing branch
touched.

Config loader: merges every `benchmarks/*/configs.yaml` into the same flat
named-config namespace. Raises on a name collision with the root file rather
than silently overriding.

### `README.md` (+18), `requirements.txt` (+7)

Docs and the `metaworld` dependency. `procgen` deliberately not pinned — it
needs its own build toolchain and does not install cleanly next to gym 0.22.

## New files

| File | Lines | Status |
|---|---|---|
| `envs/metaworld.py` | 208 | smoke-tested against the real package |
| `envs/procgen.py` | 102 | **untested**, scaffolding |
| `benchmarks/metaworld/{configs.yaml,tasks.py,launch.py,README.md}` | 383 | tested |
| `benchmarks/eval/aggregate.py` | 174 | tested on synthetic logs |
| `benchmarks/{atari100k,crafter,dmc,minecraft,procgen}/configs.yaml` | 218 | config only, unrun |
| `benchmarks/README.md` | 85 | — |

## What was verified

Against the installed `metaworld` package:

- Task-name normalisation (`reach` / `pick_place` / `pick-place-v3`)
- Full episode = 250 agent steps x action_repeat 2 = 500 env steps = `max_path_length`
- `is_terminal` never fires
- `randomize_goal=True` gives different goals across resets; `False` gives identical ones
- Meta-World's scripted expert on `reach`: `log_success` sums to exactly 1.0 per episode, 3/3 episodes
- The `tools.py` fix: fake 10-episode eval with 7 successes yields `eval_log_success = 0.7`
- `dreamer.py`'s config-loading path reproduced for all 11 new configs; every `use_shs: True` config has `dyn_discrete: 0`
- All modified and new Python files compile

Never run: a real training step. No GPU, no torch in the sandbox.

## Review checklist

**Correctness — read the code**

1. `tools.py` diff. It changes logging for *all* domains, not just Meta-World.
   Confirm your existing Crafter/DMC dashboards still read what you expect.
2. `envs/metaworld.py` `step()`: reward accumulates across `action_repeat`,
   `terminated`/`truncated` break the loop, success latches once.
3. `envs/metaworld.py` `_setup_renderer()`: swaps `env.mujoco_renderer` to force
   camera and resolution. Untested — **no GL in the sandbox.** First thing to
   verify on a real node if you run `metaworld_vision`.

**Protocol — these change the numbers**

4. `mw_randomize_goal: True`. The goal-observable classes ship with
   `_freeze_rand_vec = True` (fixed goal, easier). Default here randomises per
   episode, matching TD-MPC2. If your DreamerV3 comparison point used fixed
   goals, this is not a fair fight in your favour.
5. `mw_terminate_on_success: False`. Success does not end the episode.
6. `mlp_keys: 'state'` in the Meta-World configs. `save_episodes` writes
   `log_success` into the replay `.npz` before `log_*` keys are popped, so
   `mlp_keys: '.*'` would feed the success flag to the encoder. **Do not relax
   this regex.**
7. `benchmarks/metaworld/tasks.py`: easy (28) and medium (11) are published
   lists verbatim; the hard tier is the remaining 11, of which 4 have unverified
   hard-vs-very-hard assignments. Do not publish a hard/very-hard split without
   checking Seo et al. Appendix F.

**Before launching the sweep**

8. `shs_curriculum` boundaries in `benchmarks/metaworld/configs.yaml` are copied
   from `dmc_walker_shs`. At `batch_size 26 x batch_length 64 / train_ratio 512`
   there is one gradient step per 3.25 agent steps, so 10k/20k/30k grad steps =
   65k/130k/195k env steps. Refit from a pilot before committing.
9. Decide whether the `*_gauss` control arms stay. `use_shs: True` forces
   `dyn_discrete: 0`, so baseline-vs-shs changes the latent type *and* the
   prior. Without `gauss` the result is not attributable.

## Untouched

`models.py`, `networks.py`, all of `shs_rssm/`, `envs/wrappers.py`, and every
existing env wrapper. No SHS-RSSM code was modified. Existing DMC configs
(`dmc_walker_shs` and friends) are byte-identical, so prior DMC results remain
reproducible.
