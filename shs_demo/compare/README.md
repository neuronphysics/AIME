# Baseline comparison harness

Fits SHS-RSSM and the published switching-system baselines on the same three
benchmarks and produces the tables and regime-change figures for the paper.

| model  | reference                              | code                            | environment |
|--------|----------------------------------------|---------------------------------|-------------|
| shs    | this repository (SHS-RSSM regime head) | `shs_rssm/` (offline path)      | modern      |
| trslds | Nassar et al., ICLR 2019               | `../trslds/` vendored, byte-identical to `tree_structured_rslds` upstream | modern |
| rslds  | Linderman et al., AISTATS 2017         | `../rslds/` vendored, byte-identical to `recurrent-slds` upstream | legacy (`environment-baselines.yml`) |

Datasets: `nascar` (synthetic, Linderman et al.; cached to `data_cache/` so
every environment sees the same realisation), `toyark13` (Hughes/Sudderth
x-hdphmm-nips2015, 13 AR regimes), `mocap6` (Fox et al., 12 annotated
exercise regimes).  Loaders in `datasets.py` return one shared bundle format.

## Workflow

```bash
# everything runnable in this environment, one seed, then figures/tables
python run_all.py

# publication sweep
python run_all.py --seeds 0 1 2 --latex

# individual fits
python run_shs.py    --dataset toyark13 --nseq 12
python run_trslds.py --dataset nascar --samples 200 --burnin 100
python run_rslds.py  --dataset mocap6            # legacy env only

# aggregate whatever is in results/
python make_figures.py --dataset all --latex
```

Every runner writes `results/<dataset>/<tag>.npz` in the schema documented in
`io_utils.py`.  `make_figures.py` never refits anything; it only aggregates,
so results produced in the legacy rSLDS environment combine with modern runs.
Multi-seed runs use tags `<model>_seed<k>` and are grouped automatically
(tables report mean +/- std; ribbons show the median-Hamming seed).
On nascar the synthetic realisation is pinned by `--data-seed` (default 0, shared by all three runners); `--seed` varies initialisation only, so seed rows measure fit variance on one common dataset.

Outputs: `results/<dataset>/table.{csv,md,tex}`, `figures/<dataset>_ribbons.png`
(the regime-change figure: truth ribbon, one Hungarian-matched ribbon per
model, an observation channel), `figures/nascar_latents.png` (inferred 2-D
latents coloured by regime next to the true track), and
`figures/<dataset>_objectives.png`.

## What is comparable, and what is not

Headline numbers are label-based: Hamming distance after optimal one-to-one
(Hungarian) matching -- the NPBayesHMM / bnpy protocol -- plus many-to-one
purity, NMI and ARI (`metrics.py`).  Raw training objectives (SHS variational
bound, rSLDS/TrSLDS joint log-likelihoods) are recorded per model for
convergence panels but are **not comparable across model classes**: different
latent spaces, different likelihoods.  Context to report with the numbers:
TrSLDS/rSLDS partition a continuous latent space, which matches NASCAR's
generative structure; ToyARK13 and mocap6 switch by a Markov chain, the
regime the HDP-HMM family targets.  K is fixed for rSLDS and TrSLDS (tree
leaves), adapted by SHS -- compare segmentation quality, and treat inferred
K as an SHS-only quantity.

## Environment shims (documented deviations)

Vendored baseline sources are byte-identical to upstream; environment drift
is absorbed at runtime in the runners instead of by editing model code:

* `io_utils.ensure_pypolyagamma()` -- registers a `pypolyagamma` module
  backed by the maintained `polyagamma` package (same PG(b, c)
  parameterisation) when the 2017 C extension is absent.
* `io_utils.ensure_legacy_scipy()` -- restores `scipy.signal.gaussian` and
  `scipy.ndimage.filters` aliases removed from modern scipy.
* `../trslds/__init__.py` is the one deliberate *addition* to the vendored
  directory (no upstream file is modified): it re-installs the scipy and
  Polya-Gamma shims at import time, because trslds's `joblib.Parallel`
  (loky backend) spawns fresh worker processes that re-import
  `trslds.conditionals` when unpickling tasks -- launcher-level shims do not
  reach them.
* `run_trslds.py` clamps trslds's module-level `n_cpu = cpu_count()//2` into
  `[1, allowed CPUs]` via `sched_getaffinity` (upstream is 0 on single-core
  machines, and on clusters counts the whole node rather than the cgroup
  allocation, oversubscribing Slurm jobs).
* `run_trslds.py` drops index 0 of `model.z` / `model.x` per sequence:
  upstream carries the initial latent state, so those arrays have length
  T+1 against T observations.

## Adding a model

Fit however you like, then `io_utils.save_result(dataset, tag, z_pred,
doc_range, wall_time, params, objective=..., x_latent=...)`.  Anything in
`results/<dataset>/` is picked up by `make_figures.py`.
