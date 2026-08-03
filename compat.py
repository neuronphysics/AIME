"""Small compatibility shims for dependency-version drift.

Import this early (tools.py does) and it is a no-op on environments that do not
need it. Delete the file and its import once the upstream versions are pinned.

--------------------------------------------------------------------------
1. numpy >= 2.1 removed the `newshape=` keyword of np.reshape
--------------------------------------------------------------------------
torch.utils.tensorboard._utils._prepare_video still calls
`np.reshape(V, newshape=(...))`, so any run that logs a video crashes at the
first eval with:

    TypeError: reshape() got an unexpected keyword argument 'newshape'

This hits every vision config (dmc_vision, crafter, atari100k, minecraft,
metaworld_vision) and is unrelated to the model. The real fix is to upgrade
torch or hold numpy < 2.1; this shim restores the removed alias so an existing
environment keeps working either way. It only patches when the keyword is
genuinely missing, so it disappears the moment the dependency is fixed.
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


PATCHED_NUMPY_RESHAPE = _patch_numpy_reshape_newshape()
