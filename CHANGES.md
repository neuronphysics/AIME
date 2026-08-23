# Changes to AIME: Meta-World benchmark, eval tooling, and SHS symmetry breaking

Diffed against the original `AIME-main.zip`.

## Modified (7 files)

| File | What |
|---|---|
| `dreamer.py` | `metaworld` + `procgen` suites in `make_env`; merges `benchmarks/*/configs*.yaml`; passes `log_video` to eval; imports `compat` |
| `tools.py` | eval `log_*` averaging fix; `log_video` flag; imports `compat` |
| `models.py` | two guards so SHS diagnostics skip image figures on proprio configs |
| `configs.yaml` | `mw_*` / `procgen_*` defaults; `shs_init_scale`; 3 pre-existing bug fixes |
| `shs_rssm/{regimes_shared,regimes,regime_head,shs_rssm}.py` | `init_scale` symmetry breaking |
| `.gitignore` | venv patterns (`git clean -fdx` would otherwise delete an in-repo venv) |
| `README.md`, `requirements.txt` | docs, `metaworld` dependency |

### Pre-existing bugs fixed in `configs.yaml`

| Config | Was | Now | Effect |
|---|---|---|---|
| `minecraft` | `step: 1e8` | `steps: 1e8` | **ran to 1e6, 100x short** |
| `crafter` | `step: 1e6` | `steps: 1e6` | none (default matched) |
| `crafter`, `minecraft` | `value: {layers: 5}` | `critic: {layers: 5}` | code reads `config.critic`; the 5-layer critic was never applied |

### `tools.py` eval logging — read this if you have old Crafter numbers

Per-episode `log_*` metrics were written with `logger.scalar` on every episode,
but the eval branch flushes the logger once, after the final episode. Each call
overwrote the previous, so an eval `log_*` recorded only the **last episode's
value**. For a 0/1 metric that is a coin flip, not a rate. Now accumulated and
emitted as `eval_<key>`. Affects any env emitting `log_*`, including Crafter.

### `shs_rssm` symmetry breaking (new `shs_init_scale`, default 0.0)

All K regimes were constructed identically (same `M0`, same `lam`, zero
sufficient statistics), so responsibilities were driven only by the HDP stick
weights and one regime took ~all the mass immediately. `shs_active_regimes` read
**1 at the first diagnostic in every run**, long before any structure move — so
the later merges were garbage collection on empty components, not a model
selection verdict. That is why merge gains were near-identical across different
tasks (`22.0/22.9/18.7` on reach vs `21.97/22.88/18.74` on door-open).

`init_scale` perturbs the POSTERIOR mean `M` per regime. It deliberately does
not touch `M0`, the shared prior mean used in the KL and m-step. Measured at
L=32, K=12: the perturbation inflates `max|eig(A_k)|` by ~`init_scale*sqrt(L)`
(0.005 -> 1.04, 0.01 -> 1.06, 0.02 -> 1.14, 0.05 -> 1.36 expansive). **0.01 is
the recommended start.** Default 0.0 leaves existing behaviour bit-identical.

## New (17 files)

`compat.py` (numpy 2 shims), `envs/metaworld.py`, `envs/procgen.py` (untested),
`benchmarks/` (7 config folders, `tasks.py`, `launch.py`, `sbatch_run.sh`,
`eval/aggregate.py`, `eval/summarize.py`, READMEs), `CHANGES.md`.

## Verified

- All 50 Meta-World tasks: action dim **4**, obs dim **39**, `max_path_length` 500
- Wrapper: 250 agent steps/episode, `is_terminal` never fires, `log_success` sums to exactly 1 under the scripted expert, fixed vs randomised goals behave as documented
- Eval fix: 10-episode eval with 7 successes yields `eval_log_success = 0.7`
- `init_scale`: 0.0 leaves regimes identical, 0.01 differentiates them
- All 29 named configs load through `dreamer.py`'s exact argparse path; every `use_shs: True` config has `dyn_discrete: 0`
- Everything compiles

Never run here: a real training step (no GPU in the authoring environment).

## Known broken

`*_gauss` arms (`dyn_discrete: 0` + `use_shs: False`). Two independent 500k runs
gave `actor_grad_norm ~1e-6`, `actor_entropy` frozen at 5.675. Bit-identical
with and without `torch.compile`. No original config exercises this path.

## Known dead config keys

`dmc_acrobot_shs` sets `shs_rho1`, `shs_rho2`, `shs_recur_scale`,
`shs_switch_settle`. Grep finds zero references in any `.py`. Left untouched.
