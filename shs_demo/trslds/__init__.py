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
        import pypolyagamma
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
