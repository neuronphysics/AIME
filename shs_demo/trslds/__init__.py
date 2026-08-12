"""Environment shims for the vendored TrSLDS package (Nassar et al.).

The .py modules in this directory are byte-identical to upstream
``tree_structured_rslds``; this ``__init__`` is the one deliberate addition.
It exists because the runner-level shims in ``compare/io_utils.py`` live only
in the *launching* interpreter, while ``models.py`` fans work out with
``joblib.Parallel`` whose default loky backend spawns **fresh** worker
processes: each worker re-imports ``trslds.conditionals`` while unpickling its
task, and on a modern stack that import dies at
``from scipy.signal import gaussian`` (removed from ``scipy.signal``'s
top level) or at ``import pypolyagamma`` (the 2017 C extension).  Putting the
shims here means every process that imports anything from this package --
launcher or worker -- installs them first.

Both shims are idempotent no-ops when the running environment provides the
real thing (old scipy, or a built pypolyagamma), so this file is safe in the
legacy rSLDS environment too.  The logic deliberately duplicates
``io_utils.ensure_legacy_scipy`` / ``ensure_pypolyagamma`` rather than
importing them, so the package has no dependency on the harness being on
``sys.path`` inside workers.
"""
import sys as _sys
import types as _types


def _ensure_legacy_scipy():
    import scipy.ndimage as _ndi
    import scipy.signal as _sig

    if not hasattr(_sig, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        _sig.gaussian = _gaussian
    if not hasattr(_ndi, "filters"):
        _mod = _types.ModuleType("scipy.ndimage.filters")
        for _name in dir(_ndi):
            if not _name.startswith("_"):
                setattr(_mod, _name, getattr(_ndi, _name))
        _sys.modules["scipy.ndimage.filters"] = _mod
        _ndi.filters = _mod


def _ensure_pypolyagamma(seed_default=0):
    try:
        import pypolyagamma  # noqa: F401  real extension available
        return
    except ImportError:
        pass

    import numpy as _np
    from polyagamma import random_polyagamma as _rpg

    class PyPolyaGamma:
        def __init__(self, seed=seed_default):
            self._rng = _np.random.default_rng(int(seed) & 0xFFFFFFFF)

        def pgdraw(self, b, c):
            return float(_rpg(h=b, z=c, random_state=self._rng))

        def pgdrawv(self, b, c, out):
            out[...] = _rpg(h=_np.asarray(b), z=_np.asarray(c),
                            random_state=self._rng)

    def pgdrawvpar(ppgs, b, c, out):
        rng = ppgs[0]._rng if ppgs else _np.random.default_rng()
        out[...] = _rpg(h=_np.asarray(b), z=_np.asarray(c), random_state=rng)

    _mod = _types.ModuleType("pypolyagamma")
    _mod.PyPolyaGamma = PyPolyaGamma
    _mod.pgdrawvpar = pgdrawvpar
    _mod.pgdrawv = lambda ppg, b, c, out: ppg.pgdrawv(b, c, out)
    _mod.__version__ = "shim-polyagamma"
    _sys.modules["pypolyagamma"] = _mod


_ensure_legacy_scipy()
_ensure_pypolyagamma()
