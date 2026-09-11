"""Result-file conventions + environment compatibility helpers.

Every runner writes ``results/<dataset>/<model>.npz`` with a fixed schema so
``make_figures.py`` can aggregate runs performed in *different* environments
(the 2017 rSLDS stack must run under python<=3.8; SHS-RSSM and TrSLDS run in
the modern env).  Schema:

    model        str            'shs' | 'rslds' | 'trslds' (+ variant suffix)
    dataset      str
    z_pred       (T_total,) int concatenated predicted regimes
    doc_range    (S+1,) int
    K_used       int
    wall_time    float seconds
    objective    (n,) float     per-model training objective trace (own scale!)
    x_latent     (T_total, D_lat) or absent   inferred continuous latents
    params       str            json of run settings
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def result_path(dataset, model):
    d = RESULTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}.npz"


def save_result(dataset, model, z_pred, doc_range, wall_time, params,
                objective=None, x_latent=None, **extra):
    payload = dict(model=model, dataset=dataset,
                   z_pred=np.asarray(z_pred, dtype=np.int64),
                   doc_range=np.asarray(doc_range, dtype=np.int64),
                   K_used=int(np.unique(z_pred).size),
                   wall_time=float(wall_time),
                   params=json.dumps(params, default=str))
    if objective is not None:
        payload["objective"] = np.asarray(objective, dtype=np.float64)
    if x_latent is not None:
        payload["x_latent"] = np.asarray(x_latent, dtype=np.float64)
    payload.update(extra)
    path = result_path(dataset, model)
    np.savez(path, **payload)
    print(f"[{model}] result -> {path}")
    return path


def load_results(dataset):
    d = RESULTS / dataset
    out = {}
    if d.exists():
        for f in sorted(d.glob("*.npz")):
            if f.stem == "dataset_cache":
                continue
            out[f.stem] = dict(np.load(f, allow_pickle=True))
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


# --------------------------------------------------------------- scipy shim
def ensure_legacy_scipy():
    """Restore two pre-1.13 scipy aliases used by the vendored trslds package.

    The vendored code is kept byte-identical to upstream (Nassar et al.), so
    environment drift is absorbed here instead:

      * ``scipy.signal.gaussian``    -> ``scipy.signal.windows.gaussian``
      * ``scipy.ndimage.filters``    -> ``scipy.ndimage`` (module alias)

    No-ops when the running scipy still provides them natively.
    """
    import sys
    import types

    import scipy.ndimage as ndi
    import scipy.signal as sig

    patched = []
    if not hasattr(sig, "gaussian"):
        from scipy.signal.windows import gaussian
        sig.gaussian = gaussian
        patched.append("signal.gaussian")
    if not hasattr(ndi, "filters"):
        mod = types.ModuleType("scipy.ndimage.filters")
        for name in dir(ndi):
            if not name.startswith("_"):
                setattr(mod, name, getattr(ndi, name))
        sys.modules["scipy.ndimage.filters"] = mod
        ndi.filters = mod
        patched.append("ndimage.filters")
    return "native" if not patched else "shim:" + ",".join(patched)


# ------------------------------------------------------------------ PG shim
def ensure_pypolyagamma(seed_default=0):
    """Make ``import pypolyagamma`` work on modern Python.

    The original ``pypolyagamma`` C extension (2017) does not build on
    python>=3.10.  TrSLDS only needs ``PyPolyaGamma(seed)`` objects and the
    module-level ``pgdrawvpar(ppgs, b, c, out)`` /  ``pgdrawv`` samplers, all
    of which sample omega ~ PG(b, c) elementwise.  When the real package is
    absent we register a drop-in module backed by the maintained ``polyagamma``
    package (same PG(b, c) exponential-tilting parameterisation).  If the real
    extension is importable it is used untouched.
    """
    try:
        import pypolyagamma  # noqa: F401  real extension available
        return "native"
    except ImportError:
        pass
    import types

    from polyagamma import random_polyagamma

    class PyPolyaGamma:
        def __init__(self, seed=seed_default):
            self._rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

        def pgdraw(self, b, c):
            return float(random_polyagamma(h=b, z=c, random_state=self._rng))

        def pgdrawv(self, b, c, out):
            out[...] = random_polyagamma(h=np.asarray(b), z=np.asarray(c),
                                         random_state=self._rng)

    def pgdrawvpar(ppgs, b, c, out):
        rng = ppgs[0]._rng if ppgs else np.random.default_rng()
        out[...] = random_polyagamma(h=np.asarray(b), z=np.asarray(c),
                                     random_state=rng)

    mod = types.ModuleType("pypolyagamma")
    mod.PyPolyaGamma = PyPolyaGamma
    mod.pgdrawvpar = pgdrawvpar
    mod.pgdrawv = lambda ppg, b, c, out: ppg.pgdrawv(b, c, out)
    mod.__version__ = "shim-polyagamma"
    sys.modules["pypolyagamma"] = mod
    return "shim"
