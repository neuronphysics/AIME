import gin
import numpy as np

import torch


def torch_dtype_to_str(dtype):
    assert isinstance(dtype, torch.dtype)
    return dtype.__str__()[6:]


@gin.configurable
class TensorSpec(object):
    """Describes a torch.Tensor.
    A TensorSpec allows an API to describe the Tensors that it accepts or
    returns, before that Tensor exists. This allows dynamic and flexible graph
    construction and configuration.
    """

    __slots__ = ["_shape", "_dtype"]

    def __init__(self, shape, dtype=torch.float32):
        """
        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
        """
        self._shape = tuple(shape)
        if isinstance(dtype, str):
            self._dtype = getattr(torch, dtype)
        else:
            assert isinstance(dtype, torch.dtype)
            self._dtype = dtype

    @classmethod
    def from_spec(cls, spec):
        assert isinstance(spec, TensorSpec)
        return cls(spec.shape, spec.dtype)

    @classmethod
    def from_tensor(cls, tensor, from_dim=0):
        """Create TensorSpec from tensor.
        Args:
            tensor (Tensor): tensor from which the spec is extracted
            from_dim (int): use tensor.shape[from_dim:] as shape
        Returns:
            TensorSpec
        """
        assert isinstance(tensor, torch.Tensor)
        return TensorSpec(tensor.shape[from_dim:], tensor.dtype)

    @classmethod
    def from_array(cls, array, from_dim=0):
        """Create TensorSpec from numpy array.
        Args:
            array (np.ndarray|np.number): array from which the spec is extracted
            from_dim (int): use ``array.shape[from_dim:]`` as shape
        Returns:
            TensorSpec
        """
        assert isinstance(array, (np.ndarray, np.number))
        return TensorSpec(array.shape[from_dim:], str(array.dtype))

    @classmethod
    def is_bounded(cls):
        del cls
        return False

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def numel(self):
        """Returns the number of elements."""
        return int(np.prod(self._shape))

    @property
    def ndim(self):
        """Return the rank of the tensor."""
        return len(self._shape)

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @property
    def is_discrete(self):
        """Whether spec is discrete."""
        return not self.dtype.is_floating_point

    @property
    def is_continuous(self):
        """Whether spec is continuous."""
        return self.dtype.is_floating_point

    def __repr__(self):
        return "TensorSpec(shape={}, dtype={})".format(self.shape,
                                                       repr(self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    def __ne__(self, other):
        return not self == other

    def __reduce__(self):
        return TensorSpec, (self._shape, self._dtype)

    def _calc_shape(self, outer_dims):
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return shape

    def constant(self, value, outer_dims=None):
        """Create a constant tensor from the spec.
        Args:
            value : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return self.ones(outer_dims) * value

    def zeros(self, outer_dims=None):
        """Create a zero tensor from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return torch.zeros(self._calc_shape(outer_dims), dtype=self._dtype)

    def numpy_constant(self, value, outer_dims=None):
        """Create a constant np.ndarray from the spec.
        Args:
            value (Number) : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return np.ones(shape, dtype=torch_dtype_to_str(self._dtype)) * value

    def numpy_zeros(self, outer_dims=None):
        """Create a zero numpy.ndarray from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        return self.numpy_constant(0, outer_dims)

    def ones(self, outer_dims=None):
        """Create an all-one tensor from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return torch.ones(self._calc_shape(outer_dims), dtype=self._dtype)

    def randn(self, outer_dims=None):
        """Create a tensor filled with random numbers from a std normal dist.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.randn(*shape, dtype=self._dtype)
    
def _random_uniform_int(shape, outer_dims, minval, maxval, dtype, seed=None):
  """Iterates over n-d tensor minval, maxval limits to sample uniformly."""
  # maxval in BoundedTensorSpec is bound inclusive.
  # tf.random_uniform is upper bound exclusive, +1 to fix the sampling
  # behavior.
  # However +1 could cause overflow, in such cases we use the original maxval.
  maxval = torch.broadcast_to(maxval, minval.shape).to(dtype)
  minval = torch.broadcast_to(minval, maxval.shape).to(dtype)

  sampling_maxval = maxval

  if not torch.all(shape[-len(minval.shape):] == minval.shape):
    raise ValueError(
        "%s == shape[-%d:] != minval.shape == %s.  shape == %s." %
        (shape[len(minval.shape):], len(minval.shape), minval.shape, shape))

  # Example:
  #  minval = [1.0, 2.0]
  #  shape = [3, 2]
  #  outer_dims = [5]
  # Sampling becomes:
  #  sample [5, 3] for minval 1.0
  #  sample [5, 3] for minval 2.0
  #  stack on innermost axis to get [5, 3, 2]
  #  reshape to get [5, 3, 2]
  samples = []
  shape = torch.as_tensor(shape, dtype=torch.int32)
  sample_shape = torch.cat((outer_dims, shape[:-len(minval.shape)]), dim=0)
  full_shape = torch.cat((outer_dims, shape), dim=0)
  if seed==None:
      torch.manual_seed(0)
  else:
      torch.manual_seed(seed) 
  for (single_min, single_max) in zip(minval.flat, sampling_maxval.flat):
    samples.append(torch.tensor(sample_shape,dtype=dtype).uniform_(single_min, single_max))
  samples = torch.stack(samples, axis=-1)
  samples = torch.reshape(samples, full_shape)
  return samples

def sample_bounded_spec(spec, seed=None, outer_dims=None):
  """Samples uniformily the given bounded spec.
  Args:
    spec: A BoundedSpec to sample.
    seed: A seed used for sampling ops
    outer_dims: An optional `Tensor` specifying outer dimensions to add to the
      spec shape before sampling.
  Returns:
    A Tensor sample of the requested spec.
    #based onn https://github.com/PeterJaq/optical-film-maker/blob/04db98357ed3ba7b0830b7e48fb3ddb4c6dc9194/agents/tf_agents/specs/tensor_spec.py
  """
  minval = spec.minimum
  maxval = spec.maximum
  dtype = spec.dtype

  # To sample uint8 we will use int32 and cast later. This is needed for two
  # reasons:
  #  - tf.random_uniform does not currently support uint8
  #  - if you want to sample [0, 255] range, there's no way to do this since
  #    tf.random_uniform has exclusive upper bound and 255 + 1 would overflow.
#   print(dtype)
  is_uint8 = dtype == torch.uint8
  sampling_dtype = torch.int32 if is_uint8 else dtype


  if outer_dims is None:
    outer_dims = torch.tensor([], dtype=torch.int32)
  else:
    outer_dims = torch.as_tensor(outer_dims, dtype=torch.int32)

  def _unique_vals(vals):
    if vals.size > 0:
      if vals.ndim > 0:
        return np.all(vals == vals[0])
    return True

  if (minval.ndim != 0 or
      maxval.ndim != 0) and not (_unique_vals(minval) and _unique_vals(maxval)):
    # tf.random_uniform can only handle minval/maxval 0-d tensors.

    res = _random_uniform_int(
        shape=spec.shape,
        outer_dims=outer_dims,
        minval=minval,
        maxval=maxval,
        dtype=sampling_dtype,
        seed=seed)
  else:
    minval = minval.item(0) if minval.ndim != 0 else minval
    maxval = maxval.item(0) if maxval.ndim != 0 else maxval
    # BoundedTensorSpec are bounds inclusive.
    # tf.random_uniform is upper bound exclusive, +1 to fix the sampling
    # behavior.
    # However +1 will cause overflow, in such cases we use the original maxval.

    shape = torch.as_tensor(spec.shape, dtype=torch.int32)
    full_shape = torch.cat((outer_dims, shape), dim=0)
    g = torch.Generator()
    if not seed is None:
      g.manual_seed(seed)
    else:
      g.manual_seed(0) 
    
    # print(f"type {type(minval)}")
    res=torch.tensor(full_shape,dtype=sampling_dtype).uniform_(float(minval), float(maxval))

    if is_uint8:
     res =torch.tensor(res, dtype=dtype)

  return res


@gin.configurable
class BoundedTensorSpec(TensorSpec):
    """A `TensorSpec` that specifies minimum and maximum values.
    Example usage:
    .. code-block:: python
        spec = BoundedTensorSpec((1, 2, 3), torch.float32, 0, (5, 5, 5))
        torch_minimum = torch.as_tensor(spec.minimum, dtype=spec.dtype)
        torch_maximum = torch.as_tensor(spec.maximum, dtype=spec.dtype)
    Bounds are meant to be inclusive. This is especially important for
    integer types. The following spec will be satisfied by tensors
    with values in the set {0, 1, 2}:
    .. code-block:: python
        spec = BoundedTensorSpec((3, 5), torch.int32, 0, 2)
    """

    __slots__ = ("_minimum", "_maximum")

    def __init__(self, shape, dtype=torch.float32, minimum=0, maximum=1):
        """
        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
            minimum: numpy number or sequence specifying the minimum element
                bounds (inclusive). Must be broadcastable to `shape`.
            maximum: numpy number or sequence specifying the maximum element
                bounds (inclusive). Must be broadcastable to `shape`.
        """
        super(BoundedTensorSpec, self).__init__(shape, dtype)

        try:
            min_max = np.broadcast(minimum, maximum, np.zeros(self.shape))
            for m, M, _ in min_max:
                assert m <= M, "Min {} is greater than Max {}".format(m, M)
        except ValueError as exception:
            raise ValueError(
                "minimum or maximum is not compatible with shape. "
                "Message: {!r}.".format(exception))

        self._minimum = np.array(
            minimum, dtype=torch_dtype_to_str(self._dtype))
        self._minimum.setflags(write=False)

        self._maximum = np.array(
            maximum, dtype=torch_dtype_to_str(self._dtype))
        self._maximum.setflags(write=False)

    @classmethod
    def is_bounded(cls):
        del cls
        return True

    @classmethod
    def from_spec(cls, spec):
        assert isinstance(spec, BoundedTensorSpec)
        minimum = getattr(spec, "minimum")
        maximum = getattr(spec, "maximum")
        return BoundedTensorSpec(spec.shape, spec.dtype, minimum, maximum)

    @property
    def minimum(self):
        """Returns a NumPy array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self):
        """Returns a NumPy array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def __repr__(self):
        s = "BoundedTensorSpec(shape={}, dtype={}, minimum={}, maximum={})"
        return s.format(self.shape, repr(self.dtype), repr(self.minimum),
                        repr(self.maximum))

    def __eq__(self, other):
        tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
        return (tensor_spec_eq and np.allclose(self.minimum, other.minimum)
                and np.allclose(self.maximum, other.maximum))

    def __reduce__(self):
        return BoundedTensorSpec, (self._shape, self._dtype, self._minimum,
                                   self._maximum)

    def sample(self, outer_dims=None):
        """Sample uniformly given the min/max bounds.
        Args:
            outer_dims (list[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape

        if self.is_continuous:
            uniform = torch.rand(shape, dtype=self._dtype)
            return ((1 - uniform) * torch.as_tensor(self._minimum) +
                    torch.as_tensor(self._maximum) * uniform)
        else:
            # torch.randint cannot have multi-dim lows and highs; currently only
            # support a scalar minimum and maximum
            assert (np.shape(self._minimum) == ()
                    and np.shape(self._maximum) == ())
            return torch.randint(
                low=self._minimum.item(),
                high=self._maximum.item() + 1,
                size=shape,
                dtype=self._dtype)

    def numpy_sample(self, outer_dims=None):
        """Sample numpy arrays uniformly given the min/max bounds.
        Args:
            outer_dims (list[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            np.ndarray: an array of ``self._dtype``
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape

        if self.is_continuous:
            uniform = np.random.rand(*shape).astype(
                torch_dtype_to_str(self._dtype))
            return (1 - uniform) * self._minimum + self._maximum * uniform
        else:
            return np.random.randint(
                low=self._minimum,
                high=self._maximum + 1,
                size=shape,
                dtype=torch_dtype_to_str(self._dtype))
