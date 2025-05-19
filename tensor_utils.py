import collections
import numbers
import numpy as np
import gym
import gym.spaces
import torch
import nest

def _as_array(nested):
    """Convert numbers in ``nested`` to np.ndarray."""

    def __as_array(x):
        if isinstance(x, numbers.Number):
            return np.array(x)
        return x

    return nest.map_structure(__as_array, nested)



def tensor_spec_from_gym_space(space,
                               simplify_box_bounds=True,
                               float_dtype=np.float32):
    """
    Construct tensor spec from gym space.
    Args:
        space (gym.Space): An instance of OpenAI gym Space.
        simplify_box_bounds (bool): if True, will try to simplify redundant
            arrays to make logging and debugging less verbose when printed out.
        float_dtype (np.float32 | np.float64 | None): the dtype to be used for
            the floating numbers. If None, it will use dtypes of gym spaces.
    """

    # We try to simplify redundant arrays to make logging and debugging less
    # verbose and easier to read since the printed spec bounds may be large.
    from tensor_specs import BoundedTensorSpec
    def try_simplify_array_to_value(np_array):
        """If given numpy array has all the same values, returns that value."""
        first_value = np_array.item(0)
        if np.all(np_array == first_value):
            return np.array(first_value, dtype=np_array.dtype)
        else:
            return np_array

    if isinstance(space, gym.spaces.Discrete):
        # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
        # are inclusive on their bounds.
        maximum = space.n - 1
        return BoundedTensorSpec(
            shape=(), dtype=space.dtype.name, minimum=0, maximum=maximum)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        maximum = try_simplify_array_to_value(
            np.asarray(space.nvec - 1, dtype=space.dtype))
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=space.dtype.name,
            minimum=0,
            maximum=maximum)
    elif isinstance(space, gym.spaces.MultiBinary):
        shape = (space.n,)
        return BoundedTensorSpec(
            shape=shape, dtype=space.dtype.name, minimum=0, maximum=1)
    elif isinstance(space, gym.spaces.Box):

        if float_dtype is not None and "float" in space.dtype.name:
            dtype = np.dtype(float_dtype)
        else:
            dtype = space.dtype

        minimum = np.asarray(space.low, dtype=dtype)
        maximum = np.asarray(space.high, dtype=dtype)
        if simplify_box_bounds:
            minimum = try_simplify_array_to_value(minimum)
            maximum = try_simplify_array_to_value(maximum)
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=dtype.name,
            minimum=minimum,
            maximum=maximum)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple([tensor_spec_from_gym_space(s) for s in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
        return collections.OrderedDict([(key, tensor_spec_from_gym_space(s))
                                        for key, s in space.spaces.items()])
    else:
        raise ValueError(
            'The gym space {} is currently not supported.'.format(space))


