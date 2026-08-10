"""Small compatibility shims for dependency-version drift.

Import this early (tools.py does) and it is a no-op on environments that do not
need it. Delete the file and its import once the upstream versions are pinned.

Both shims below are the same class of problem: a dependency that has not caught
up with numpy 2. They are stopgaps -- upgrading the offending library is the real
fix, since restoring a removed name cannot help with the *other* ways an old
library may break on numpy 2.

--------------------------------------------------------------------------
1. numpy >= 2.1 removed the `newshape=` keyword of np.reshape
--------------------------------------------------------------------------
torch.utils.tensorboard._utils._prepare_video still calls
`np.reshape(V, newshape=(...))`, so any run that logs a video crashes at the
first eval with

    TypeError: reshape() got an unexpected keyword argument 'newshape'

This hits every vision config (dmc_vision, crafter, atari100k, minecraft,
metaworld_vision), not just Meta-World.

--------------------------------------------------------------------------
2. numpy 2.0 removed a batch of long-deprecated aliases
--------------------------------------------------------------------------
`np.Inf`, `np.NaN`, `np.float_`, `np.trapz` and friends were removed. Any
matplotlib still shipping `get_tight_layout_figure` (< 3.6) uses `np.Inf`
internally, so plotting crashes with

    AttributeError: `np.Inf` was removed in the NumPy 2.0 release.

which takes down the SHS diagnostic figures mid-run.
"""

import numpy as np


def _patch_numpy_reshape_newshape():
    try:
        np.reshape(np.zeros(4), newshape=(2, 2))
        return False  # keyword still supported, nothing to do
    except TypeError:
        pass

    _orig_reshape = np.reshape

    def reshape(a, *args, **kwargs):
        # Translate the removed `newshape=` alias to `shape=`, but only when the
        # caller did not already supply the shape positionally or by keyword --
        # otherwise np.reshape would get two values for the same parameter.
        newshape = kwargs.pop("newshape", None)
        if newshape is not None and not args and "shape" not in kwargs:
            kwargs["shape"] = newshape
        return _orig_reshape(a, *args, **kwargs)

    reshape.__doc__ = _orig_reshape.__doc__
    np.reshape = reshape
    return True


# name -> replacement, for aliases numpy 2.0 removed. Only names that are
# genuinely missing get restored, so this is a no-op on numpy 1.x.
_REMOVED_ALIASES = {
    "Inf": np.inf, "Infinity": np.inf, "infty": np.inf,
    "PINF": np.inf, "NINF": -np.inf,
    "NaN": np.nan, "NAN": np.nan,
    "PZERO": 0.0, "NZERO": -0.0,
    "float_": np.float64, "complex_": np.complex128,
    "unicode_": np.str_, "string_": np.bytes_,
    "bool8": np.bool_, "object0": object,
    "int0": np.intp, "uint0": np.uintp,
    "round_": np.round, "product": np.prod, "cumproduct": np.cumprod,
    "sometrue": np.any, "alltrue": np.all,
}


def _patch_numpy_removed_aliases():
    restored = []
    for name, value in _REMOVED_ALIASES.items():
        try:
            getattr(np, name)
        except AttributeError:
            setattr(np, name, value)
            restored.append(name)
    # np.trapz was renamed rather than deleted.
    if hasattr(np, "trapezoid"):
        try:
            np.trapz
        except AttributeError:
            np.trapz = np.trapezoid
            restored.append("trapz")
    return restored


PATCHED_NUMPY_RESHAPE = _patch_numpy_reshape_newshape()
RESTORED_NUMPY_ALIASES = _patch_numpy_removed_aliases()
