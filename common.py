# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various functions used by different alf modules."""
from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter
from absl import flags, logging
from collections import OrderedDict
import copy
import functools
from functools import wraps, partial
import atexit
import threadpoolctl
import traceback
import gin
import glob
import math
import numpy as np
import os
import pprint
import random
import shutil
import subprocess
import sys
import time
import torch
import torch.distributions as td
import torch.nn as nn
import types
from numbers import Number
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union
import gin.config
import nest as nest
from tensor_specs import TensorSpec, BoundedTensorSpec
import runpy
import gym_wrappers
import gym
import numbers
import multiprocessing
from enum import Enum
from multiprocessing import dummy as mp_threads
######################################################
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
import inspect
from inspect import Parameter
import torch.multiprocessing as mp
from alf_environment import AlfEnvironment, BatchedTensorWrapper, TensorWrapper, MultitaskWrapper, BatchEnvironmentWrapper
from . import _penv
_CONF_TREE = {}
_PRE_CONFIGS = []
_HANDLED_PRE_CONFIGS = []

def _ensure_wrappability(fn):
    """Make sure `fn` can be wrapped cleanly by functools.wraps.

    Adapted from gin-config/gin/config.py
    """
    # Handle "builtin_function_or_method", "wrapped_descriptor", and
    # "method-wrapper" types.
    unwrappable_types = (type(sum), type(object.__init__),
                         type(object.__call__))
    if isinstance(fn, unwrappable_types):
        # pylint: disable=unnecessary-lambda
        wrappable_fn = lambda *args, **kwargs: fn(*args, **kwargs)
        wrappable_fn.__name__ = fn.__name__
        wrappable_fn.__doc__ = fn.__doc__
        wrappable_fn.__module__ = ''  # These types have no __module__, sigh.
        wrappable_fn.__wrapped__ = fn
        return wrappable_fn

    # Otherwise we're good to go...
    return fn


def _make_config(signature, whitelist, blacklist):
    """Create a dictionary of _Config for given signature.

    Args:
        signature (inspec.Signature): function signature
        whitelist (list[str]): allowed configurable argument names
        blacklist (list[str]): disallowed configurable argument names
    Returns:
        dict: name => _Config
    """
    configs = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if ((not blacklist and not whitelist)
                or (whitelist and name in whitelist)
                or (blacklist and name not in blacklist)):
            config = _Config()
            configs[name] = config
            if param.default is not inspect.Parameter.empty:
                config.set_default_value(param.default)

    return configs


def _handle_pre_configs(path, node):
    def _handle1(item):
        name, value = item
        parts = name.split('.')
        if len(parts) > len(path):
            return True
        for i in range(-len(parts), 0):
            if parts[i] != path[i]:
                return True
        node.set_value(value)
        node.set_mutable(False)
        _HANDLED_PRE_CONFIGS.append(item)
        return False

    global _PRE_CONFIGS
    _PRE_CONFIGS = list(filter(_handle1, _PRE_CONFIGS))

def _add_to_conf_tree(module_path, func_name, arg_name, node):
    """Add a config object to _CONF_TREE.

    Args:
        module_path (list[str]): module path of this function
        func_name (str): name of the function
        node (_Config): config object for this value
        arg_name: (str): name of the argument
    """

    tree = _CONF_TREE
    path = module_path + func_name.split('.') + [arg_name]
    names = []
    for name in reversed(path[1:]):
        if not isinstance(tree, dict):
            raise ValueError("'%s' conflicts with existing config name '%s'" %
                             ('.'.join(path), '.'.join(names)))
        if name not in tree:
            tree[name] = {}
        tree = tree[name]
        names.insert(0, name)

    if not isinstance(tree, dict):
        raise ValueError("'%s' conflicts with existing config name '%s'" %
                         ('.'.join(path), '.'.join(names)))
    if path[0] in tree:
        if isinstance(tree[path[0]], dict):
            leaves = _get_all_leaves(tree)
            raise ValueError(
                "'%s' conflicts with existing config name '%s'" %
                ('.'.join(path), '.'.join([leaves[0][0]] + names)))
        else:
            raise ValueError("'%s' has already been defined." % '.'.join(path))

    tree[path[0]] = node

    _handle_pre_configs(path, node)


def _find_class_construction_fn(cls):
    """Find the first __init__ or __new__ method in the given class's MRO.

    Adapted from gin-config/gin/config.py
    """
    for base in type.mro(cls):
        if '__init__' in base.__dict__:
            return base.__init__
        if '__new__' in base.__dict__:
            return base.__new__

def _get_all_leaves(conf_dict):
    """
    Returns:
        list[tuple[path, _Config]]
    """
    leaves = []
    for k, v in conf_dict.items():
        if not isinstance(v, dict):
            leaves.append((k, v))
        else:
            leaves.extend(
                [(name + '.' + k, node) for name, node in _get_all_leaves(v)])
    return leaves


class _Config(object):
    """Object representing one configurable value."""

    def __init__(self):
        self._configured = False
        self._used = False
        self._has_default_value = False
        self._mutable = True
        self._sole_init = False

    def set_default_value(self, value):
        self._default_value = value
        self._has_default_value = True

    def has_default_value(self):
        return self._has_default_value

    def get_default_value(self):
        return self._default_value

    def is_configured(self):
        return self._configured

    def set_mutable(self, mutable):
        self._mutable = mutable

    def is_mutable(self):
        return self._mutable

    def set_sole_init(self, sole_init):
        self._sole_init = sole_init

    def get_sole_init(self):
        return self._sole_init

    def set_value(self, value):
        self._configured = True
        self._value = value

    def get_value(self):
        assert self._configured
        return self._value

    def get_effective_value(self):
        assert self._configured or self._has_default_value
        return self._value if self._configured else self._default_value

    def set_used(self):
        self._used = True

    def is_used(self):
        return self._used

    def reset(self):
        self._used = False
        self._configured = False
        self._mutable = True

def _make_wrapper(fn, configs, signature, has_self, config_only_args):
    """Wrap the function.

    Args:
        fn (Callable): function to be wrapped
        configs (dict[_Config]): config associated with the arguments of function
            ``fn``
        signature (inspect.Signature): Signature object of ``fn``. It is provided
            as an argument so that we don't need to call ``inspect.signature(fn)``
            repeatedly, with is expensive.
        has_self (bool): whether the first argument is expected to be self but
            signature does not contains parameter for self. This should be True
            if fn is __init__() function of a class.
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects.
    Returns:
        The wrapped function
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        unspecified_positional_args = []
        unspecified_kw_args = {}
        num_positional_args = len(args)
        num_positional_args -= has_self

        set_positional_args = []
        for i, (name, param) in enumerate(signature.parameters.items()):
            config = configs.get(name, None)
            if config is None:
                continue
            elif i < num_positional_args:
                set_positional_args.append(name)
            elif param.kind in (Parameter.VAR_POSITIONAL,
                                Parameter.VAR_KEYWORD):
                continue
            elif param.kind == Parameter.POSITIONAL_ONLY:
                if config.is_configured():
                    unspecified_positional_args.append(config.get_value())
                    config.set_used()
            elif name not in kwargs and param.kind in (
                    Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                if config.is_configured():
                    unspecified_kw_args[name] = config.get_value()
                config.set_used()

        for config_only_arg in config_only_args:
            if config_only_arg in set_positional_args or config_only_arg in kwargs:
                raise ValueError(
                    f"The arg '{config_only_arg}' of {fn.__qualname__} is guarded but has been modified. "
                    f"Most likely partial() was used to change this value, which is not allowed."
                )

        return fn(*args, *unspecified_positional_args, **kwargs,
                  **unspecified_kw_args)

    return _wrapper


def _decorate(fn_or_cls, name, whitelist, blacklist, config_only_args):
    """decorate a function or class.

    Args:
        fn_or_cls (Callable): a function or a class
        name (str): name for the function. If None, ``fn_or_cls.__qualname__``
            will be used.
        whitelist (list[str]): A whitelisted set of kwargs that should be configurable.
            All other kwargs will not be configurable. Only one of ``whitelist`` or
            `blacklist` should be specified.
        blacklist (list[str]): A blacklisted set of kwargs that should not be
            configurable. All other kwargs will be configurable. Only one of
            ``whitelist` or ``blacklist`` should be specified.
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects.
    Returns:
        The decorated function
    """
    signature = inspect.signature(fn_or_cls)
    configs = _make_config(signature, whitelist, blacklist)

    orig_name = name

    if name is None or '.' not in name:
        module_path = fn_or_cls.__module__.split('.')
    else:
        parts = name.split('.')
        module_path = parts[:-1]
        name = parts[-1]

    if name is None:
        name = fn_or_cls.__qualname__

    for arg_name, node in configs.items():
        _add_to_conf_tree(module_path, name, arg_name, node)

    if inspect.isclass(fn_or_cls):
        # cannot use _make_wrapper() directly on fn_or_cls. This is because
        # _make_wrapper() returns a function. But we want to return a class.
        construction_fn = _find_class_construction_fn(fn_or_cls)
        has_self = construction_fn.__name__ != '__new__'
        decorated_fn = _make_wrapper(
            _ensure_wrappability(construction_fn), configs, signature,
            has_self, config_only_args)
        if construction_fn.__name__ == '__new__':
            decorated_fn = staticmethod(decorated_fn)
        setattr(fn_or_cls, construction_fn.__name__, decorated_fn)
    else:
        fn_or_cls = _make_wrapper(
            fn_or_cls,
            configs,
            signature,
            has_self=0,
            config_only_args=config_only_args)

    if fn_or_cls.__module__ != '<run_path>' and os.environ.get(
            'ALF_USE_GIN', "1") == "1":
        # If a file is executed using runpy.run_path(), the module name is
        # '<run_path>', which is not an acceptable name by gin.
        return gin.configurable(
            orig_name, whitelist=whitelist, blacklist=blacklist)(fn_or_cls)
    else:
        return fn_or_cls




def configurable(fn_or_name=None,
                 whitelist=[],
                 blacklist=[],
                 config_only_args=[]):
    """Decorator to make a function or class configurable.

    This decorator registers the decorated function/class as configurable, which
    allows its parameters to be supplied from the global configuration (i.e., set
    through ``alf.config()``). The decorated function is associated with a name in
    the global configuration, which by default is simply the name of the function
    or class, but can be specified explicitly to avoid naming collisions or improve
    clarity.

    If some parameters should not be configurable, they can be specified in
    ``blacklist``. If only a restricted set of parameters should be configurable,
    they can be specified in ``whitelist``. Furthermore, parameters can be
    guarded by being specified in ``config_only_args`` so that their values can only be
    changed globally via alf.config(). This prevents unintended side effects
    that may arise from having inconsistent parameter values caused by local
    changes (e.g., partial()).

    The decorator can be used without any parameters as follows:

    .. code-block: python

        @alf.configurable
        def my_function(param1, param2='a default value'):
            ...

    In this case, the function is associated with the name
    'my_function' in the global configuration, and both param1 and param2 are
    configurable.

    The decorator can be supplied with parameters to specify the configurable name
    or supply a whitelist/blacklist:

    .. code-block: python

        @alf.configurable('my_func', whitelist=['param2'])
        def my_function(param1, param2='a default value'):
            ...

    In this case, the configurable is associated with the name 'my_func' in the
    global configuration, and only param2 is configurable.

    Classes can be decorated as well, in which case parameters of their
    constructors are made configurable:

    .. code-block:: python

        @alf.configurable
        class MyClass(object):
            def __init__(self, param1, param2='a default value'):
                ...

    In this case, the name of the configurable is 'MyClass', and both `param1`
    and `param2` are configurable.

    The full name of a configurable value is MODULE_PATH.FUNC_NAME.ARG_NAME. It
    can be referred using any suffixes as long as there is no ambiguity. For
    example, assuming there are two configurable values "abc.def.func.a" and
    "xyz.uvw.func.a", you can use "abc.def.func.a", "def.func.a", "xyz.uvw.func.a"
    or "uvw.func.a" to refer these two configurable values. You cannot use
    "func.a" because of the ambiguity. Because of this, you cannot have a config
    name which is the strict suffix of another config name. For example,
    "A.Test.arg" and "Test.arg" cannot both be defined. You can supply a different
    name for the function to avoid conflict:

    .. code-block:: python

        @alf.configurable("NewTest")
        def Test(arg):
            ...

    or

    .. code-block:: python

        @alf.configurable("B.Test")
        def Test(arg):
            ...


    Note: currently, to maintain the compatibility with gin-config, all the
    functions decorated using alf.configurable are automatically configurable
    using gin. The values specified using ``alf.config()`` will override
    values specified through gin. Gin wrapper is quite convoluted and can make
    debugging more challenging. It can be disabled by setting environment
    variable ALF_USE_GIN to 0 if you are not using gin.

    Args:
        fn_or_name (Callable|str): A name for this configurable, or a function
            to decorate (in which case the name will be taken from that function).
            If not set, defaults to the name of the function/class that is being made
            configurable. If a name is provided, it may also include module components
            to be used for disambiguation. If the module components is provided,
            the original module name of the function will not be used to compose
            the full name.
        whitelist (list[str]): A whitelisted set of kwargs that should be configurable.
            All other kwargs will not be configurable. Only one of ``whitelist`` or
            ``blacklist`` should be specified.
        blacklist (list[str]): A blacklisted set of kwargs that should not be
            configurable. All other kwargs will be configurable. Only one of
            ``whitelist`` or ``blacklist`` should be specified. An entry that is in
            ``blacklist`` cannot be in ``config_only_args``.
        config_only_args (list[str]): list of args that should be guarded. In other
            words, their values can only be set / changed globally via ``alf.config()``.
            This protects against local untracked changes as a result of 1) using
            ``partial()`` or 2) setting an argument when calling the function, which
            can cause unintended side effects. An entry that is in ``config_only_args``
            cannot be in ``blacklist``.
    Returns:
        decorated function if fn_or_name is Callable.
        a decorator if fn is not Callable.
    Raises:
        ValueError: Can be raised
            1) If a configurable with ``name`` (or the name of `fn_or_cls`) already exists
            2) If both a whitelist and blacklist are specified.
            3) If the same entry is found in both blacklist and config_only_args.
            4) If an arg listed in config_only_args is changed without using alf.config().
    """

    if callable(fn_or_name):
        name = None
    else:
        name = fn_or_name

    if whitelist and blacklist:
        raise ValueError("Only one of 'whitelist' and 'blacklist' can be set.")

    for entry in blacklist:
        if entry in config_only_args:
            raise ValueError(
                f"Entry '{entry}' found in both blacklist and config_only_args. "
                f"An entry can only be in one of these lists.")

    if not callable(fn_or_name):

        def _decorator(fn_or_cls):
            return _decorate(fn_or_cls, name, whitelist, blacklist,
                             config_only_args)

        return _decorator
    else:
        return _decorate(fn_or_name, name, whitelist, blacklist,
                         config_only_args)
    
class InverseTransformSampling(object):
    """Interface for defining inverse transform sampling."""

    @staticmethod
    def cdf(x):
        """Cumulative distribution function of this distribution."""
        raise NotImplementedError

    @staticmethod
    def icdf(x):
        """Inverse of the CDF"""
        raise NotImplementedError

    @staticmethod
    def log_prob(x):
        """Log probability density."""
        raise NotImplementedError


class NormalITS(InverseTransformSampling):
    """Normal distribution.

    .. math::

        p(x) = 1/sqrt(2*pi) * exp(-x^2/2)

    """

    @staticmethod
    def cdf(x):
        # sqrt(0.5) = 0.7071067811865476
        return 0.5 * (1 + torch.erf(x * 0.7071067811865476))

    @staticmethod
    def icdf(x):
        # sqrt(2) = 1.4142135623730951
        return torch.erfinv(2 * x - 1) * 1.4142135623730951

    @staticmethod
    def log_prob(x):
        # log(sqrt(2 * pi)) = 0.9189385332046727
        return -0.5 * (x**2) - 0.9189385332046727

class TruncatedDistribution(td.Distribution):
    r"""The base class of truncated distributions.

    A truncated distribution :math:`q(x)` is defined as a standard base distribution :math:`p(x)` and
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s}) if l \le x le u
        q(x) = 0 otherwise

    where :math:`P` is the cdf of :math:`p`.

    Args:
        loc: the location parameter. Its shape is batch_shape + event_shape.
        scale: the scale parameter. Its shape is batch_shape + event_shape.
        lower_bound: the lower bound. Its shape is event_shape.
        upper_bound: the upper bound. Its shape is event_shape.
        its: the standard distribution to be used.
    """

    arg_constraints = {
        'loc': td.constraints.real,
        'scale': td.constraints.positive
    }
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, lower_bound: Tensor,
                 upper_bound: Tensor, its: InverseTransformSampling):
        event_shape = torch.broadcast_shapes(lower_bound.shape,
                                             upper_bound.shape)
        batch_shape = torch.broadcast_shapes(scale.shape, loc.shape,
                                             event_shape)
        if len(event_shape) > 0:
            batch_shape = batch_shape[:-len(event_shape)]

        self._scale = scale
        self._loc = loc

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self._its = its
        self._lower_bound = lower_bound.to(loc.device)
        self._upper_bound = upper_bound.to(loc.device)
        self._cdf_lb = self._its.cdf((self._lower_bound - loc) / scale)
        self._cdf_ub = self._its.cdf((self._upper_bound - loc) / scale)
        self._logz = (scale * (self._cdf_ub - self._cdf_lb + 1e-30)).log()

    @property
    def scale(self):
        """Scale parameter of this distribution."""
        return self._scale

    @property
    def loc(self):
        """Location parameter of this distribution."""
        return self._loc

    @property
    def lower_bound(self):
        """Lower bound of this distribution."""
        return self._lower_bound

    @property
    def upper_bound(self):
        """Upper bound of this distribution."""
        return self._upper_bound

    @property
    def mode(self):
        """Mode of this distribution."""
        result = torch.maximum(self._lower_bound, self._loc)
        result = torch.minimum(self._upper_bound, result)

        return result

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        Args:
            sample_shape: sample shape
        Returns:
            Tensor of shape ``sample_shape + batch_shape + event_shape``
        """
        r = torch.rand(sample_shape + self._batch_shape + self._event_shape)
        r = (1 - r) * self._cdf_lb + r * self._cdf_ub
        r = r.clamp(0.001, 0.999)
        x = self._its.icdf(r) * self._scale + self._loc
        assert torch.isfinite(x).all()
        # because of the r.clamp() above, x may be out of bound
        x = torch.maximum(x, self._lower_bound)
        x = torch.minimum(x, self._upper_bound)
        return x

    def log_prob(self, value: Tensor):
        """The log of the probability density evaluated at ``value``.

        Args:
            value: its shape should be ``sample_shape + batch_shape + event_shape``
        Returns:
            Tensor of shape ``sample_shape + batch_shape``
        """
        y = self._its.log_prob((value - self._loc) / self._scale) - self._logz
        assert torch.isfinite(y).all()
        n = len(self._event_shape)
        if n > 0:
            return y.sum(dim=list(range(-n, 0)))
        else:
            return y


class TruncatedNormal(TruncatedDistribution):
    r"""Truncated normal distribution.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p` and :math:`P` are the pdf and cdf of the standard normal
    distribution respectively.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, NormalITS())


class CauchyITS(InverseTransformSampling):
    """Cauchy distribution.

    .. math::

        p(x) = 1 / (pi * (1 + x*x))

    """

    @staticmethod
    def cdf(x):
        return torch.atan(x) / math.pi + 0.5

    @staticmethod
    def icdf(x):
        return torch.tan(math.pi * (x - 0.5))

    @staticmethod
    def log_prob(x):
        return -(math.pi * (1 + x**2)).log()


class TruncatedCauchy(TruncatedDistribution):
    r"""Truncated Cauchy distribution.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p` and :math:`P` are the pdf and cdf of the standard Cauchy
    distribution respectively.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, CauchyITS())


class TruncatedT2(TruncatedDistribution):
    r"""Truncated Student's t distribution with degree of freedom 2.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p(x)=1 / (2 * (1 + x^2)^1.5)` and :math:`P` is the cdf of
    :math:`p(x)`.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, T2ITS())

def get_invertible(cls):
    """A helper function to turn on the cache mechanism for transformation.
    This is useful as some transformations (say :math:`g`) may not be able to
    provide an accurate inversion therefore the difference between :math:`x` and
    :math:`g^{-1}(g(x))` is large. This could lead to unstable training in
    practice. For a torch transformation :math:`y=g(x)`, when ``cache_size`` is
    set to one, the latest value for :math:`(x, y)` is cached and will be used
    later for future computations. E.g. for inversion, a call to
    :math:`g^{-1}(y)` will return :math:`x`, solving the inversion error issue
    mentioned above. Note that in the case of having a chain of transformations
    (:math:`G`), all the element transformations need to turn on the cache to
    ensure the composite transformation :math:`G` satisfy:
    :math:`x=G^{-1}(G(x))`.
    """

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, cache_size=1)

    return NewCls


"""
WARNING: If you need to train policy gradient with a ``TransformedDistribution``,
then make sure to detach the sampled action when the transforms have trainable
parameters.

For detailed reasons, please refer to ``alf/docs/notes/compute_probs_of_transformed_dist.rst``.
"""

AbsTransform = get_invertible(td.AbsTransform)
ExpTransform = get_invertible(td.ExpTransform)
PowerTransform = get_invertible(td.PowerTransform)
SigmoidTransform = get_invertible(td.SigmoidTransform)
SoftmaxTransform = get_invertible(td.SoftmaxTransform)


class AffineTransform(get_invertible(td.AffineTransform)):
    """Overwrite PyTorch's ``AffineTransform`` to provide a builder to be
    compatible with ``DistributionSpec.build_distribution()``.
    """

    def get_builder(self):
        return functools.partial(
            AffineTransform, loc=self.loc, scale=self.scale)



class DiagMultivariateNormal(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate normal distribution with diagonal variance.

        Args:
            loc (Tensor): mean of the distribution
            scale (Tensor): standard deviation. Should have same shape as ``loc``.
        """
        # set validate_args to False here to enable the construction of Normal
        # distribution with zero scale.
        super().__init__(
            td.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1)

    @property
    def stddev(self):
        return self.base_dist.stddev


@configurable(whitelist=['eps'])
class Beta(td.Beta):
    r"""Beta distribution parameterized by ``concentration1`` and ``concentration0``.

    Note: we need to wrap ``td.Beta`` so that ``self.concentration1`` and
    ``self.concentration0`` are the actual tensors passed in to construct the
    distribution. This is important in certain situation. For example, if you want
    to register a hook to process the gradient to ``concentration1`` and ``concentration0``,
    ``td.Beta.concentration0.register_hook()`` will not work because gradient will
    not be backpropped to ``td.Beta.concentration0`` since it is sliced from
    ``td.Dirichlet.concentration`` and gradient will only be backpropped to
    ``td.Dirichlet.concentration`` instead of ``td.Beta.concentration0`` or
    ``td.Beta.concentration1``.

    """

    def __init__(self,
                 concentration1,
                 concentration0,
                 eps=None,
                 validate_args=None):
        """
        Args:
            concentration1 (float or Tensor): 1st concentration parameter of the distribution
                (often referred to as alpha)
            concentration0 (float or Tensor): 2nd concentration parameter of the distribution
                (often referred to as beta)
            eps (float): a very small value indicating the interval ``[eps, 1-eps]``
                into which the sampled values will be clipped. This clipping can
                prevent ``NaN`` and ``Inf`` values in the gradients. If None,
                a small value defined by PyTorch will be used.
        """
        self._concentration1 = concentration1
        self._concentration0 = concentration0
        super().__init__(concentration1, concentration0, validate_args)
        if eps is None:
            self._eps = torch.finfo(self._dirichlet.concentration.dtype).eps
        else:
            self._eps = float(eps)

    @property
    def concentration0(self):
        return self._concentration0

    @property
    def concentration1(self):
        return self._concentration1

    @property
    def mode(self):
        alpha = self.concentration1
        beta = self.concentration0
        mode = torch.where((alpha > 1) & (beta > 1),
                           (alpha - 1) / (alpha + beta - 2),
                           torch.where(alpha < beta, torch.zeros(()),
                                       torch.ones(())))
        return mode

    def rsample(self, sample_shape=()):
        """We override the original ``rsample()`` in order to clamp the output
        to avoid `NaN` and `Inf` values in the gradients. See Pyro's
        ``rsample()`` implementation in
        `<https://docs.pyro.ai/en/dev/_modules/pyro/distributions/affine_beta.html#AffineBeta>`_.
        """
        x = super(Beta, self).rsample(sample_shape)
        return torch.clamp(x, min=self._eps, max=1 - self._eps)


class DiagMultivariateBeta(td.Independent):
    def __init__(self, concentration1, concentration0):
        """Create multivariate independent beta distribution.

        Args:
            concentration1 (float or Tensor): 1st concentration parameter of the
                distribution (often referred to as alpha)
            concentration0 (float or Tensor): 2nd concentration parameter of the
                distribution (often referred to as beta)
        """
        super().__init__(
            Beta(concentration1, concentration0), reinterpreted_batch_ndims=1)


class AffineTransformedDistribution(td.TransformedDistribution):
    r"""Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    The reason of not using ``td.TransformedDistribution`` is that we can implement
    ``entropy``, ``mean``, ``variance`` and ``stddev`` for ``AffineTransforma``.
    """

    def __init__(self, base_dist: td.Distribution, loc, scale):
        """
        Args:
            loc (Tensor or float): Location parameter.
            scale (Tensor or float): Scale parameter.
        """
        super().__init__(
            base_distribution=base_dist,
            transforms=AffineTransform(loc, scale))
        self.loc = loc
        self.scale = scale

        # broadcast scale to event_shape if necessary
        s = torch.ones(base_dist.event_shape) * scale
        self._log_abs_scale = s.abs().log().sum()

    def entropy(self):
        """Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return self._log_abs_scale + self.base_dist.entropy()

    @property
    def mean(self):
        """Returns the mean of the distribution."""
        return self.scale * self.base_dist.mean + self.loc

    @property
    def variance(self):
        """Returns the variance of the distribution."""
        return self.scale * self.scale * self.base_dist.variance

    @property
    def stddev(self):
        """Returns the variance of the distribution."""
        return self.scale * self.base_dist.stddev


class StableCauchy(td.Cauchy):
    def rsample(self, sample_shape=torch.Size(), clipping_value=0.49):
        r"""Overwrite Pytorch's Cauchy rsample for a more stable result. Basically
        the sampled number is clipped to fall within a reasonable range.


        For reference::

            > np.tan(math.pi * -0.499)
            -318.30883898554157
            > np.tan(math.pi * -0.49)
            -31.820515953773853

        Args:
            clipping_value (float): suppose eps is sampled from ``(-0.5,0.5)``.
                It will be clipped to ``[-clipping_value, clipping_value]`` to
                avoid values with huge magnitudes.
        """
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).uniform_()
        eps = torch.clamp(eps - 0.5, min=-clipping_value, max=clipping_value)
        return torch.tan(eps * math.pi) * self.scale + self.loc


class DiagMultivariateCauchy(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate cauchy distribution with diagonal scale matrix.

        Args:
            loc (Tensor): median of the distribution. Note that Cauchy doesn't
                have a mean (divergent).
            scale (Tensor): also known as "half width". Should have the same
                shape as ``loc``.
        """
        super().__init__(StableCauchy(loc, scale), reinterpreted_batch_ndims=1)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


class OneHotCategoricalStraightThrough(td.OneHotCategoricalStraightThrough):
    """Provide an additional property ``mode`` with gradient enabled.
    """

    @property
    def mode(self):
        mode = torch.nn.functional.one_hot(
            torch.argmax(self.logits, -1), num_classes=self.logits.shape[-1])
        return mode.to(self.logits) + self.probs - self.probs.detach()


@configurable
class OneHotCategoricalGumbelSoftmax(td.OneHotCategorical):
    r"""Create a reparameterizable ``td.OneHotCategorical`` distribution based on
    the Gumbel-softmax gradient estimator from

    ::

        Jang et al., "CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX", 2017.
    """
    has_rsample = True

    def __init__(self, hard_sample: bool = True, tau: float = 1., **kwargs):
        """
        Args:
            hard_sample: If False, the rsampled result will be a "soft" vector
                of Gumbel softmax distribution, which naturally supports gradient
                backprop. If True, ``argmax`` will be applied on top of it and then
                a straight-through gradient estimator is used.
            tau: the Gumbel-softmax temperature for ``rsample``. A higher
                temperature leads to a more uniform sample.
        """
        super(OneHotCategoricalGumbelSoftmax, self).__init__(**kwargs)
        self._hard_sample = hard_sample
        self._tau = tau

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        # expand additional first dims according to ``sample_shape``
        shape = sample_shape + (1, ) * len(self.param_shape)
        logits = self.logits.repeat(*shape)
        return torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self._tau, hard=self._hard_sample, dim=-1)

    @property
    def mode(self):
        mode = torch.nn.functional.one_hot(
            torch.argmax(self.logits, -1), num_classes=self.logits.shape[-1])
        return mode.to(self.logits) + self.probs - self.probs.detach()


def _builder_independent(base_builder, reinterpreted_batch_ndims_, **kwargs):
    return td.Independent(base_builder(**kwargs), reinterpreted_batch_ndims_)


def _builder_transformed(base_builder, transform_builders, params_,
                         transforms_params_):
    transforms = [
        b(**p) for b, p in zip(transform_builders, transforms_params_)
    ]
    return td.TransformedDistribution(base_builder(**params_), transforms)


def _get_categorical_builder(obj: Union[
        td.Categorical, td.OneHotCategorical, td.
        OneHotCategoricalStraightThrough, OneHotCategoricalStraightThrough]):

    dist_cls = type(obj)

    if 'probs' in obj.__dict__ and id(obj.probs) == id(obj._param):
        # This means that obj is constructed using probs
        return dist_cls, {'probs': obj.probs}
    else:
        return dist_cls, {'logits': obj.logits}


def _get_gumbelsoftmax_categorical_builder(
        obj: OneHotCategoricalGumbelSoftmax):
    builder = functools.partial(
        OneHotCategoricalGumbelSoftmax,
        hard_sample=obj._hard_sample,
        tau=obj._tau)
    if 'probs' in obj.__dict__ and id(obj.probs) == id(obj._param):
        # This means that obj is constructed using probs
        return builder, {'probs': obj.probs}
    else:
        return builder, {'logits': obj.logits}


def _get_independent_builder(obj: td.Independent):
    builder, params = _get_builder(obj.base_dist)
    new_builder = functools.partial(_builder_independent, builder,
                                    obj.reinterpreted_batch_ndims)
    return new_builder, params


def _get_transform_builders_params(transforms):
    """Return a nested structure where each node is a non-composed transform,
    after expanding any composed transform in ``transforms``.
    """

    def _get_transform_builder(transform):
        if hasattr(transform, "get_builder"):
            return transform.get_builder()
        return transform.__class__

    def _get_transform_params(transform):
        if hasattr(transform, 'params') and transform.params is not None:
            # We assume that if a td.Transform has attribute 'params', then they are the
            # parameters we'll extract and store.
            assert isinstance(
                transform.params,
                dict), ("Transform params must be provided as a dict! "
                        f"Got {transform.params}")
            return transform.params
        return {}  # the transform doesn't have any parameter

    if isinstance(transforms, td.Transform):
        if isinstance(transforms, td.ComposeTransform):
            builders, params = _get_transform_builders_params(transforms.parts)
            compose_transform_builder = lambda parts_params: td.ComposeTransform(
                [b(**p) for b, p in zip(builders, parts_params)])
            return compose_transform_builder, {'parts_params': params}
        else:
            builder = _get_transform_builder(transforms)
            params = _get_transform_params(transforms)
            return builder, params

    assert isinstance(transforms, list), f"Incorrect transforms {transforms}!"
    builders_and_params = [
        _get_transform_builders_params(t) for t in transforms
    ]
    builders, params = zip(*builders_and_params)
    return list(builders), list(params)


def _get_transformed_builder(obj: td.TransformedDistribution):
    # 'params' contains the dist params and all wrapped transform params starting
    # 'obj.base_dist' downwards
    builder, params = _get_builder(obj.base_dist)
    transform_builders, transform_params = _get_transform_builders_params(
        obj.transforms)
    new_builder = functools.partial(_builder_transformed, builder,
                                    transform_builders)
    new_params = {"params_": params, 'transforms_params_': transform_params}
    return new_builder, new_params


def _builder_affine_transformed(base_builder, loc_, scale_, **kwargs):
    # 'loc' and 'scale' may conflict with the names in kwargs. So we add suffix '_'.
    return AffineTransformedDistribution(base_builder(**kwargs), loc_, scale_)


def _get_affine_transformed_builder(obj: AffineTransformedDistribution):
    builder, params = _get_builder(obj.base_dist)
    new_builder = functools.partial(_builder_affine_transformed, builder,
                                    obj.loc, obj.scale)
    return new_builder, params

#Needed
def _get_mixture_same_family_builder(obj: td.MixtureSameFamily):
    mixture_builder, mixture_params = _get_builder(obj.mixture_distribution)
    components_builder, components_params = _get_builder(
        obj.component_distribution)

    def _mixture_builder(mixture, components):
        return td.MixtureSameFamily(
            mixture_builder(**mixture), components_builder(**components))

    return _mixture_builder, {
        "mixture": mixture_params,
        "components": components_params
    }


_get_builder_map = {
    td.Categorical:
        _get_categorical_builder,
    td.OneHotCategorical:
        _get_categorical_builder,
    td.OneHotCategoricalStraightThrough:
        _get_categorical_builder,
    OneHotCategoricalStraightThrough:
        _get_categorical_builder,
    OneHotCategoricalGumbelSoftmax:
        _get_gumbelsoftmax_categorical_builder,
    td.Normal:
        lambda obj: (td.Normal, {
            'loc': obj.mean,
            'scale': obj.stddev
        }),
    StableCauchy:
        lambda obj: (StableCauchy, {
            'loc': obj.loc,
            'scale': obj.scale
        }),
    td.Independent:
        _get_independent_builder,
    DiagMultivariateNormal:
        lambda obj: (DiagMultivariateNormal, {
            'loc': obj.mean,
            'scale': obj.stddev
        }),
    DiagMultivariateCauchy:
        lambda obj: (DiagMultivariateCauchy, {
            'loc': obj.loc,
            'scale': obj.scale
        }),
    td.TransformedDistribution:
        _get_transformed_builder,
    AffineTransformedDistribution:
        _get_affine_transformed_builder,
    Beta:
        lambda obj: (Beta, {
            'concentration1': obj.concentration1,
            'concentration0': obj.concentration0
        }),
    DiagMultivariateBeta:
        lambda obj: (DiagMultivariateBeta, {
            'concentration1': obj.base_dist.concentration1,
            'concentration0': obj.base_dist.concentration0
        }),
    TruncatedNormal:
        lambda obj: (functools.partial(
            TruncatedNormal,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    TruncatedCauchy:
        lambda obj: (functools.partial(
            TruncatedCauchy,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    TruncatedT2:
        lambda obj: (functools.partial(
            TruncatedT2,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    td.MixtureSameFamily:
        _get_mixture_same_family_builder,
}


######################################################
def _get_builder(obj):
    return _get_builder_map[type(obj)](obj)

class DistributionSpec(object):
    def __init__(self, builder, input_params_spec):
        """

        Args:
            builder (Callable): the function which is used to build the
                distribution. The returned value of ``builder(input_params)``
                is a ``Distribution`` with input parameter as ``input_params``.
            input_params_spec (nested TensorSpec): the spec for the argument of
                ``builder``.
        """
        self.builder = builder
        self.input_params_spec = input_params_spec

    def build_distribution(self, input_params):
        """Build a Distribution using ``input_params``.

        Args:
            input_params (nested Tensor): the parameters for build the
                distribution. It should match ``input_params_spec`` provided as
                ``__init__``.
        Returns:
            Distribution:
        """
        nest.assert_same_structure(input_params, self.input_params_spec)
        return self.builder(**input_params)

    @classmethod
    def from_distribution(cls, dist, from_dim=0):
        """Create a ``DistributionSpec`` from a ``Distribution``.
        Args:
            dist (Distribution): the ``Distribution`` from which the spec is
                extracted.
            from_dim (int): only use the dimensions from this. The reason of
                using ``from_dim>0`` is that ``[0, from_dim)`` might be batch
                dimension in some scenario.
        Returns:
            DistributionSpec:
        """
        builder, input_params = _get_builder(dist)
        input_param_spec = extract_spec(input_params, from_dim)
        return cls(builder, input_param_spec)

def to_distribution_param_spec(nests):
    """Convert the ``DistributionSpecs`` in nests to their parameter specs.
    Args:
        nests (nested DistributionSpec of TensorSpec):  Each ``DistributionSpec``
            will be converted to a dictionary of the spec of its input ``Tensor``
            parameters.
    Returns:
        nested TensorSpec: Each leaf is a ``TensorSpec`` or a ``dict``
        corresponding to one distribution, with keys as parameter name and
        values as ``TensorSpecs`` for the parameters.
    """

    def _to_param_spec(spec):
        if isinstance(spec, DistributionSpec):
            return spec.input_params_spec
        elif isinstance(spec, TensorSpec):
            return spec
        else:
            raise ValueError("Only TensorSpec or DistributionSpec is allowed "
                             "in nest, got %s. nest is %s" % (spec, nests))

    return nest.map_structure(_to_param_spec, nests)

def params_to_distributions(nests, nest_spec):
    """Convert distribution parameters to ``Distribution``, keep tensors unchanged.
    Args:
        nests (nested Tensor): a nested ``Tensor`` and dictionary of tensor
            parameters of ``Distribution``. Typically, ``nest`` is obtained using
            ``distributions_to_params()``.
        nest_spec (nested DistributionSpec and TensorSpec): The distribution
            params will be converted to ``Distribution`` according to the
            corresponding ``DistributionSpec`` in ``nest_spec``.
    Returns:
        nested Distribution or Tensor:
    """

    def _to_dist(spec, params):
        if isinstance(spec, DistributionSpec):
            return spec.build_distribution(params)
        elif isinstance(spec, TensorSpec):
            return params
        else:
            raise ValueError(
                "Only DistributionSpec or TensorSpec is allowed "
                "in nest_spec, got %s. nest_spec is %s" % (spec, nest_spec))

    return nest.map_structure_up_to(nest_spec, _to_dist, nest_spec, nests)

def zero_tensor_from_nested_spec(nested_spec, batch_size):
    """Create nested zero Tensors or Distributions.
    A zero tensor with shape[0]=`batch_size is created for each TensorSpec and
    A distribution with all the parameters as zero Tensors is created for each
    DistributionSpec.
    Args:
        nested_spec (nested TensorSpec or DistributionSpec):
        batch_size (int|tuple|list): batch size/shape added as the first dimension to the shapes
             in TensorSpec
    Returns:
        nested Tensor or Distribution
    """
    if isinstance(batch_size, Iterable):
        shape = batch_size
    else:
        shape = [batch_size]

    def _zero_tensor(spec):
        return spec.zeros(shape)

    param_spec = to_distribution_param_spec(nested_spec)
    params = nest.map_structure(_zero_tensor, param_spec)
    return params_to_distributions(params, nested_spec)

def add_method(cls):
    """A decorator for adding a method to a class (cls).
    Example usage:
    .. code-block:: python
        class A:
            pass
        @add_method(A)
        def new_method(self):
            print('new method added')
        # now new_method() is added to class A and is ready to be used
        a = A()
        a.new_method()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def as_list(x):
    """Convert ``x`` to a list.
    It performs the following conversion:
    .. code-block:: python
        None => []
        list => x
        tuple => list(x)
        other => [x]
    Args:
        x (any): the object to be converted
    Returns:
        list:
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def tuplify2d(x):
    """Convert ``x`` to a tuple of length two.
    It performs the following conversion:
    .. code-block:: python
        x => x if isinstance(x, tuple) and len(x) == 2
        x => (x, x) if not isinstance(x, tuple)
    Args:
        x (any): the object to be converted
    Returns:
        tuple:
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


class Periodically(nn.Module):
    def __init__(self, body, period, name='periodically'):
        """Periodically performs the operation defined in body.
        Args:
            body (Callable): callable to be performed every time
                an internal counter is divisible by the period.
            period (int): inverse frequency with which to perform the operation.
            name (str): name of the object.
        Raises:
            TypeError: if body is not a callable.
        """
        super().__init__()
        if not callable(body):
            raise TypeError('body must be callable.')
        self._body = body
        self._period = period
        self._counter = 0
        self._name = name

    def forward(self):
        self._counter += 1
        if self._counter % self._period == 0:
            self._body()
        elif self._period is None:
            return


def get_target_updater(models, target_models, tau=1.0, period=1, copy=True):
    r"""Performs a soft update of the target model parameters.
    For each weight :math:`w_s` in the model, and its corresponding
    weight :math:`w_t` in the target_model, a soft update is:
    .. math::
        w_t = (1 - \tau) * w_t + \tau * w_s.
    Args:
        models (Network | list[Network] | Parameter | list[Parameter] ): the
            current model or parameter.
        target_models (Network | list[Network] | Parameter | list[Parameter]):
            the model or parameter to be updated.
        tau (float): A float scalar in :math:`[0, 1]`. Default :math:`\tau=1.0`
            means hard update.
        period (int): Step interval at which the target model is updated.
        copy (bool): If True, also copy ``models`` to ``target_models`` in the
            beginning.
    Returns:
        Callable: a callable that performs a soft update of the target model parameters.
    """
    models = as_list(models)
    target_models = as_list(target_models)

    def _copy_model_or_parameter(s, t):
        if isinstance(s, nn.Parameter):
            t.data.copy_(s)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                wt.data.copy_(ws)

    def _lerp_model_or_parameter(s, t):
        if isinstance(s, nn.Parameter):
            t.data.lerp_(s, tau)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                wt.data.lerp_(ws, tau)

    if copy:
        for model, target_model in zip(models, target_models):
            _copy_model_or_parameter(model, target_model)

    def update():
        if tau != 1.0:
            for model, target_model in zip(models, target_models):
                _lerp_model_or_parameter(model, target_model)
        else:
            for model, target_model in zip(models, target_models):
                _copy_model_or_parameter(model, target_model)

    return Periodically(update, period, 'periodic_update_targets')


def expand_dims_as(x, y):
    """Expand the shape of ``x`` with extra singular dimensions.
    The result is broadcastable to the shape of ``y``.
    Args:
        x (Tensor): source tensor
        y (Tensor): target tensor. Only its shape will be used.
    Returns:
        ``x`` with extra singular dimensions.
    """
    assert x.ndim <= y.ndim
    k = y.ndim - x.ndim
    if k == 0:
        return x
    else:
        assert x.shape == y.shape[:len(x.shape)]
        return x.reshape(*x.shape, *([1] * k))


def reset_state_if_necessary(state, initial_state, reset_mask):
    """Reset state to initial state according to ``reset_mask``.
    Args:
        state (nested Tensor): the current batched states
        initial_state (nested Tensor): batched intitial states
        reset_mask (nested Tensor): with ``shape=(batch_size,), dtype=tf.bool``
    Returns:
        nested Tensor
    """
    return nest.map_structure(
        lambda i_s, s: torch.where(expand_dims_as(reset_mask, i_s), i_s, s),
        initial_state, state)

def create_summary_writer(summary_dir, flush_secs=10, max_queue=10):
    """Creates a SummaryWriter that will write out events to the event file.

    Args:
        summary_dir (str) – Save directory location.
        flush_secs (int) – How often, in seconds, to flush the pending events
            and summaries to disk. Default is every 10 seconds.
        max_queue (int) – Size of the queue for pending events and summaries
            before one of the ‘add’ calls forces a flush to disk.
            Default is ten items.
    Returns:
        SummaryWriter
    """
    return SummaryWriter(
        log_dir=summary_dir, flush_secs=flush_secs, max_queue=max_queue)

_progress = {
    "percent": 0.0,
    "iterations": 0.0,
    "env_steps": 0.0,
    "global_counter": 0.0
}
_global_counter = np.array(0, dtype=np.int64)
_summary_enabled = False
_summary_writer_stack = [None]
_record_if_stack = [
    lambda: True,
]
def update_progress(progress_type: str, value: Number):
    _progress[progress_type] = float(value)

def set_global_counter(counter):
    global _global_counter
    _global_counter.fill(counter)
    update_progress("global_counter", counter)

def get_global_counter():
    """Get the global counter

    Returns:
        the global int64 Tensor counter
    """
    return _global_counter

def is_summary_enabled():
    """Return whether summary is enabled."""
    return _summary_enabled

class push_summary_writer(object):
    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        _summary_writer_stack.append(self._writer)

    def __exit__(self, type, value, traceback):
        _summary_writer_stack.pop()

class record_if(object):
    """Context manager to set summary recording on or off according to `cond`."""

    def __init__(self, cond: Callable):
        """Create the context manager.

        Args:
            cond (Callable): a function which returns whether summary should be
                recorded.
        """
        self._cond = cond

    def __enter__(self):
        _record_if_stack.append(self._cond)

    def __exit__(self, type, value, traceback):
        _record_if_stack.pop()

def run_under_record_context(func,
                             summary_dir,
                             summary_interval,
                             flush_secs,
                             summary_max_queue=10):
    """Run ``func`` under summary record context.
    Args:
        func (Callable): the function to be executed.
        summary_dir (str): directory to store summary. A directory starting with
            ``~/`` will be expanded to ``$HOME/``.
        summary_interval (int): how often to generate summary based on the
            global counter
        flush_secs (int): flush summary to disk every so many seconds
        summary_max_queue (int): the largest number of summaries to keep in a queue;
            will flush once the queue gets bigger than this. Defaults to 10.
    """
    summary_dir = os.path.expanduser(summary_dir)
    summary_writer = create_summary_writer(
        summary_dir, flush_secs=flush_secs, max_queue=summary_max_queue)
    global_step = get_global_counter()

    def _cond():
        # We always write summary in the initial `summary_interval` steps
        # because there might be important changes at the beginning.
        return (is_summary_enabled()
                and (global_step < summary_interval
                     or global_step % summary_interval == 0))

    with push_summary_writer(summary_writer):
        with record_if(_cond):
            func()

    summary_writer.close()


@gin.configurable
def cast_transformer(observation, dtype=torch.float32):
    """Cast observation
    Args:
         observation (nested Tensor): observation
         dtype (Dtype): The destination type.
    Returns:
        casted observation
    """

    def _cast(obs):
        if isinstance(obs, torch.Tensor):
            return obs.type(dtype)
        return obs

    return nest.map_structure(_cast, observation)


@gin.configurable
def image_scale_transformer(observation, fields=None, min=-1.0, max=1.0):
    """Scale image to min and max (0->min, 255->max).
    Args:
        observation (nested Tensor): If observation is a nested structure, only
            ``namedtuple`` and ``dict`` are supported for now.
        fields (list[str]): the fields to be applied with the transformation. If
            None, then ``observation`` must be a ``Tensor`` with dtype ``uint8``.
            A field str can be a multi-step path denoted by "A.B.C".
        min (float): normalize minimum to this value
        max (float): normalize maximum to this value
    Returns:
        Transfromed observation
    """

    def _transform_image(obs):
        assert isinstance(obs, torch.Tensor), str(type(obs)) + ' is not Tensor'
        assert obs.dtype == torch.uint8, "Image must have dtype uint8!"
        obs = obs.type(torch.float32)
        return ((max - min) / 255.) * obs + min

    fields = fields or [None]
    for field in fields:
        observation = nest.transform_nest(
            nested=observation, field=field, func=_transform_image)
    return observation


def _markdownify_gin_config_str(string, description=''):
    """Convert an gin config string to markdown format.
    Args:
        string (str): the string from ``gin.operative_config_str()``.
        description (str): Optional long-form description for this config str.
    Returns:
        string: the markdown version of the config string.
    """

    # This function is from gin.tf.utils.GinConfigSaverHook
    # TODO: Total hack below. Implement more principled formatting.
    def _process(line):
        """Convert a single line to markdown format."""
        if not line.startswith('#'):
            return '    ' + line

        line = line[2:]
        if line.startswith('===='):
            return ''
        if line.startswith('None'):
            return '    # None.'
        if line.endswith(':'):
            return '#### ' + line
        return line

    output_lines = []

    if description:
        output_lines.append("    # %s\n" % description)

    for line in string.splitlines():
        procd_line = _process(line)
        if procd_line is not None:
            output_lines.append(procd_line)

    return '\n'.join(output_lines)




def inoperative_config_str(max_line_length=80, continuation_indent=4):
    """Retrieve the "inoperative" configuration as a config string.
    Args:
        max_line_length (int): A (soft) constraint on the maximum length
            of a line in the formatted string.
        continuation_indent (int): The indentation for continued lines.
    Returns:
        A config string capturing all parameter values configured but not
            used by the current program (override by explicit call).
    """
    inoperative_config = {}
    config = gin.config._CONFIG
    operative_config = gin.config._OPERATIVE_CONFIG
    imported_module = gin.config._IMPORTED_MODULES
    for module, module_config in config.items():
        inoperative_module_config = {}
        if module not in operative_config:
            inoperative_module_config = module_config
        else:
            operative_module_config = operative_config[module]
            for key, value in module_config.items():
                if key not in operative_module_config or \
                        value != operative_module_config[key]:
                    inoperative_module_config[key] = value

        if inoperative_module_config:
            inoperative_config[module] = inoperative_module_config

    # hack below
    # `gin.operative_config_str` only depends on `_OPERATIVE_CONFIG` and `_IMPORTED_MODULES`
    gin.config._OPERATIVE_CONFIG = inoperative_config
    gin.config._IMPORTED_MODULES = {}
    inoperative_str = gin.operative_config_str(max_line_length,
                                               continuation_indent)
    gin.config._OPERATIVE_CONFIG = operative_config
    gin.config._IMPORTED_MODULES = imported_module
    return inoperative_str

def get_gin_confg_strs():
    """
    Obtain both the operative and inoperative config strs from gin.
    The operative configuration consists of all parameter values used by
    configurable functions that are actually called during execution of the
    current program, and inoperative configuration consists of all parameter
    configured but not used by configurable functions. See
    ``gin.operative_config_str()`` and ``inoperative_config_str`` for
    more detail on how the config is generated.
    Returns:
        tuple:
        - md_operative_config_str (str): a markdown-formatted operative str
        - md_inoperative_config_str (str): a markdown-formatted inoperative str
    """
    operative_config_str = gin.operative_config_str()
    md_operative_config_str = _markdownify_gin_config_str(
        operative_config_str,
        'All parameter values used by configurable functions that are actually called'
    )
    md_inoperative_config_str = inoperative_config_str()
    if md_inoperative_config_str:
        md_inoperative_config_str = _markdownify_gin_config_str(
            md_inoperative_config_str,
            "All parameter values configured but not used by program. The configured "
            "functions are either not called or called with explicit parameter values "
            "overriding the config.")
    return md_operative_config_str, md_inoperative_config_str

_SUMMARY_DATA_BUFFER = {}
_scope_stack = ['']
def should_record_summaries():
    """Whether summary should be recorded.

    Returns:
        bool: False means that all calls to scalar(), text(), histogram() etc
            are not recorded.

    """
    return (_summary_writer_stack[-1] and is_summary_enabled()
            and _record_if_stack[-1]())

def _summary_wrapper(summary_func):
    """Summary wrapper

    Wrapper summary function to reduce cost for data computation
    """

    @functools.wraps(summary_func)
    def wrapper(name,
                data,
                average_over_summary_interval=False,
                step=None,
                **kwargs):
        """
        Args:
            average_over_summary_interval: if True, the average value of data during a
                summary interval will be written to summary. If data is None,
                it will be ignored for calculating the average. Note that providing
                a "None" value for data is different from not calling the summary
                function at all. A "None" value for data will cause the summary
                to be generated if ``should_record_summaries()`` returns True
                at the moment.
        """
        if average_over_summary_interval:
            if isinstance(data, torch.Tensor):
                data = data.detach()
            if name.startswith('/'):
                name = name[1:]
            else:
                name = _scope_stack[-1] + name
            if data is not None:
                if name in _SUMMARY_DATA_BUFFER:
                    data_sum, counter = _SUMMARY_DATA_BUFFER[name]
                    _SUMMARY_DATA_BUFFER[name] = data_sum + data, counter + 1
                else:
                    _SUMMARY_DATA_BUFFER[name] = data, 1
            if should_record_summaries() and name in _SUMMARY_DATA_BUFFER:
                data_sum, counter = _SUMMARY_DATA_BUFFER[name]
                del _SUMMARY_DATA_BUFFER[name]
                data = data_sum / counter
                if step is None:
                    step = _global_counter
                summary_func(name, data, step, **kwargs)
        else:
            if should_record_summaries():
                if isinstance(data, torch.Tensor):
                    data = data.detach()
                if step is None:
                    step = _global_counter
                if name.startswith('/'):
                    name = name[1:]
                else:
                    name = _scope_stack[-1] + name
                summary_func(name, data, step, **kwargs)

    return wrapper

@_summary_wrapper
def text(name, data, step=None, walltime=None):
    """Add text data to summary.

    Note that the actual tag will be `name + "/text_summary"` because torch
    adds "/text_summary to tag. See
    https://github.com/pytorch/pytorch/blob/877ab3afe33eeaa797296d2794317b59e5ac90f4/torch/utils/tensorboard/summary.py#L477

    Args:
        name (str): Data identifier
        data (str): String to save
        step (int): Global step value to record. None for using get_global_counter()
        walltime (float): Optional override default walltime (time.time())
            seconds after epoch of event
    """
    _summary_writer_stack[-1].add_text(name, data, step, walltime=walltime)

def summarize_gin_config():
    """Write the operative and inoperative gin config to Tensorboard summary.
    """
    md_operative_config_str, md_inoperative_config_str = get_gin_confg_strs()
    text('gin/operative_config', md_operative_config_str)
    if md_inoperative_config_str:
        text('gin/inoperative_config', md_inoperative_config_str)


def copy_gin_configs(root_dir, gin_files):
    """Copy gin config files to root_dir
    Args:
        root_dir (str): directory path
        gin_files (None|list[str]): list of file paths
    """
    root_dir = os.path.expanduser(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    for f in gin_files:
        shutil.copyfile(f, os.path.join(root_dir, os.path.basename(f)))


def get_gin_file():
    """Get the gin configuration file.
    If ``FLAGS.gin_file`` is not set, find gin files under ``FLAGS.root_dir`` and
    returns them. If there is no 'gin_file' flag defined, return ''.
    Returns:
        the gin file(s)
    """
    if hasattr(flags.FLAGS, "gin_file"):
        gin_file = flags.FLAGS.gin_file
        if gin_file is None:
            root_dir = os.path.expanduser(flags.FLAGS.root_dir)
            gin_file = glob.glob(os.path.join(root_dir, "*.gin"))
            assert gin_file, "No gin files are found! Please provide"
        return gin_file
    else:
        return ''


ALF_CONFIG_FILE = 'alf_config.py'
_IMPORT_STACK = []
_CONFIG_MODULES = {}
_CONF_FILES = {}
def get_conf_file():
    """Get the configuration file.
    If ``FLAGS.conf`` is not set, find alf_config.py or configured.gin under
    ``FLAGS.root_dir`` and returns it. If there is no 'conf' flag defined,
    return None.
    Returns:
        str: the name of the conf file. None if there is no conf file
    """
    if not hasattr(flags.FLAGS, "conf") and not hasattr(
            flags.FLAGS, "gin_file"):
        return None

    conf_file = getattr(flags.FLAGS, 'conf', None)
    if conf_file is not None:
        return conf_file
    conf_file = getattr(flags.FLAGS, 'gin_file', None)
    if conf_file is not None:
        return conf_file

    root_dir = os.path.expanduser(flags.FLAGS.root_dir)
    conf_file = os.path.join(root_dir, ALF_CONFIG_FILE)
    if os.path.exists(conf_file):
        return conf_file
    gin_file = glob.glob(os.path.join(root_dir, "*.gin"))
    if not gin_file:
        return None
    assert len(
        gin_file) == 1, "Multiple *.gin files are found in %s" % root_dir
    return gin_file[0]

def _get_config_node(config_name):
    """Get the _Config object corresponding to config_name."""
    tree = _CONF_TREE
    path = config_name.split('.')
    for name in reversed(path):
        if not isinstance(tree, dict) or name not in tree:
            raise ValueError("Cannot find config name %s" % config_name)
        tree = tree[name]

    if isinstance(tree, dict):
        leaves = _get_all_leaves(tree)
        if len(leaves) > 1:
            # only show at most 3 ambiguous choices
            leaves = leaves[:3]
            names = [name + '.' + config_name for name, node in leaves]
            raise ValueError("config name '%s' is ambiguous. There are %s" %
                             (config_name, names))

        assert len(leaves) == 1
        config_node = leaves[0][1]
    else:
        config_node = tree

    return config_node


def get_config_value(config_name):
    """Get the value of the config with the name ``config_name``.

    Args:
        config_name (str): name of the config or its suffix which can uniquely
            identify the config.
    Returns:
        Any: value of the config
    Raises:
        ValueError: if the value of the config has not been configured and it
            does not have a default value.
    """
    config_node = _get_config_node(config_name)
    if not config_node.is_configured() and not config_node.has_default_value():
        raise ValueError(
            "Config '%s' is not configured nor has a default value." %
            config_name)

    config_node.set_used()
    return config_node.get_effective_value()


@logging.skip_log_prefix
def config1(config_name,
            value,
            mutable=True,
            raise_if_used=True,
            sole_init=False,
            override_sole_init=False,
            override_all=False):
    """Set one configurable value.

    Args:
        config_name (str): name of the config
        value (any): value of the config
        mutable (bool): whether the config can be changed later. If the user
            tries to change an existing immutable config, the change will be
            ignored and a warning will be generated. You can always change a
            mutable config. ``ValueError`` will be raised if trying to set a new
            immutable value to an existing immutable value.
        raise_if_used (bool): If True, ValueError will be raised if trying to
            config a value which has already been used.
        sole_init (bool): If True, the config value can no longer be set again
            after this config call. Any future calls will raise a RuntimeError.
            This is helpful in enforcing a singular point of initialization,
            thus eliminating any potential side effects from possible future
            overrides. For users wanting this to be the default behavior, the
            ALF_SOLE_CONFIG env variable can be set to 1.
        override_sole_init (bool): If True, the value of the config will be set
            regardless of any previous ``sole_init`` setting. This should be used
            only when absolutely necessary (e.g., a teacher-student training loop,
            where the student must override certain configs inherited from the
            teacher). If the config is immutable, a warning will be declared with
            no changes made.
        override_all (bool): If True, the value of the config will be set regardless
            of any pre-existing ``mutable`` or ``sole_init`` settings. This should
            be used only when absolutely necessary (e.g., adjusting certain configs
            such as mini_batch_size for DDP workers.).
    """
    config_node = _get_config_node(config_name)

    if raise_if_used and config_node.is_used():
        raise ValueError(
            "Config '%s' has already been used. You should config "
            "its value before using it." % config_name)

    if override_all:
        if config_node.get_sole_init():
            logging.warning(
                "The value of config '%s' (%s) is protected by sole_init. "
                "It is now being overridden by the overide_all flag to a new value %s. "
                "Use at your own risk." % (config_name,
                                           config_node.get_value(), value))
        if not config_node.is_mutable():
            logging.warning(
                "The value of config '%s' (%s) is immutable. "
                "It is now being overridden by the overide_all flag to a new value %s. "
                "Use at your own risk." % (config_name,
                                           config_node.get_value(), value))
        config_node.set_value(value)
        return
    elif override_sole_init:
        if config_node.is_configured():
            if not config_node.is_mutable():
                logging.warning(
                    "The value of config '%s' (%s) is immutable. "
                    "Override flag with new value %s is ignored. " %
                    (config_name, config_node.get_value(), value))
                return
            elif config_node.get_sole_init():
                logging.warning(
                    "The value of config '%s' (%s) is protected by sole_init. "
                    "It is now being overridden by the override flag to a new value %s. "
                    "Use at your own risk." % (config_name,
                                               config_node.get_value(), value))
    elif config_node.is_configured():
        if config_node.get_sole_init():
            raise RuntimeError(
                "Config '%s' is protected by sole_init and cannot be reconfigured. "
                "If you wish to set this config value, do so the location of the "
                "previous call." % config_name)

        if config_node.is_mutable():
            logging.warning(
                "The value of config '%s' has been configured to %s. It is "
                "replaced by the new value %s" %
                (config_name, config_node.get_value(), value))
        else:
            logging.warning(
                "The config '%s' has been configured to an immutable value "
                "of %s. The new value %s will be ignored" %
                (config_name, config_node.get_value(), value))
            config_node.set_sole_init(sole_init)
            return

    config_node.set_value(value)
    config_node.set_mutable(mutable)
    if not override_sole_init:
        config_node.set_sole_init(sole_init)

@logging.skip_log_prefix
def pre_config(configs):
    """Preset the values for configs before the module defining it is imported.

    This function is useful for handling the config params from commandline,
    where there are no module imports and hence no config has been defined.

    The value is bound to the config when the module defining the config is
    imported later. ``validate_pre_configs()` should be called after the config
    file has been loaded to ensure that all the pre_configs have been correctly
    bound.

    Args:
        configs (dict): dictionary of config name to value
    """
    for name, value in configs.items():
        try:
            config1(name, value, mutable=False, sole_init=False)
            _HANDLED_PRE_CONFIGS.append((name, value))
        except ValueError:
            _PRE_CONFIGS.append((name, value))

class ConfigModule:
    pass


def _get_conf_file_full_path(conf_file):
    if os.path.isabs(conf_file):
        if os.path.exists(conf_file):
            return os.path.realpath(conf_file)
    if len(_IMPORT_STACK) == 0:
        # called from load_config()
        dir = os.getcwd()
    else:
        # callded from import_config()
        dir = os.path.dirname(_IMPORT_STACK[-1])
    candidate = os.path.join(dir, conf_file)
    if os.path.exists(candidate):
        return os.path.realpath(candidate)
    conf_path = os.environ.get("ALF_CONFIG_PATH", None)
    conf_dirs = []
    if conf_path is not None:
        conf_dirs = conf_path.split(':')
    for dir in conf_dirs:
        candidate = os.path.join(dir, conf_file)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    raise ValueError(f"Cannot find conf file {conf_file}")

def _add_conf_file(conf_file):
    if conf_file in _CONF_FILES:
        return
    with open(conf_file, "r") as f:
        _CONF_FILES[conf_file] = f.read()

def _import_config(conf_file):
    if conf_file in _CONFIG_MODULES:
        return _CONFIG_MODULES[conf_file]
    _add_conf_file(conf_file)
    _IMPORT_STACK.append(conf_file)
    kv = runpy.run_path(conf_file)
    _IMPORT_STACK.pop()
    module = ConfigModule()
    for k, v in kv.items():
        setattr(module, k, v)
    _CONFIG_MODULES[conf_file] = module
    return module


def load_config(conf_file):
    """Load config from a file.

    Different from ``import_config()``, ``load_config()`` should only be used by
    your main code to load the config. And it should be only called once unless
     ``reset_configs()`` is called to reset the configuration to default state.

    If ``conf_file`` is a relative path, ``load_config()`` will first try to find it
    in the current working directory. If it cannot be found there, directories in
    the environment varianble ``ALF_CONFIG_PATH`` will be searched in order.

    Args:
        conf_file
    Returns:
        the config module object, which can be used in a similar way as python
        imported module.
    """
    global _ROOT_CONF_FILE
    if _ROOT_CONF_FILE is not None:
        raise ValueError(
            "One process can only call alf.load_config() once. "
            "If you want to call it multiple times, you need to call "
            "alf.reset_configs() between the calls.")
    conf_file = _get_conf_file_full_path(conf_file)
    _ROOT_CONF_FILE = conf_file
    return _import_config(conf_file)

def validate_pre_configs():
    """Validate that all the configs set through ``pre_config()`` are correctly bound."""

    if _PRE_CONFIGS:
        raise ValueError((
            "A pre-config '%s' was not handled, either because its config name "
            +
            "was not found, or there was some error when calling pre_config()")
                         % _PRE_CONFIGS[0][0])

    for (config_name, _) in _HANDLED_PRE_CONFIGS:
        _get_config_node(config_name)

def parse_config(conf_file, conf_params, create_env=True):
    """Parse config file and config parameters

    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().

    Args:
        conf_file (str): The full path of the config file.
        conf_params (list[str]): the list of config parameters. Each one has a
            format of CONFIG_NAME=VALUE.
        create_env (bool): whether to create env. If True, create the env if it is not
            created yet, and the ``ml_type`` is ``rl``.
            For other types (e.g. ``sl``), no env is created.

    """
    global _is_parsing
    _is_parsing = True

    try:
        if conf_params:
            for conf_param in conf_params:
                pos = conf_param.find('=')
                if pos == -1:
                    raise ValueError("conf_param should have a format of "
                                     "'CONFIG_NAME=VALUE': %s" % conf_param)
                config_name = conf_param[:pos]
                config_value = conf_param[pos + 1:]
                config_value = eval(config_value)
                pre_config({config_name: config_value})

        load_config(conf_file)
        validate_pre_configs()
    finally:
        _is_parsing = False

    ml_type = get_config_value('TrainerConfig.ml_type')

    # only create env for ``ml_type`` with value of ``rl``
    create_env = create_env and (ml_type == 'rl')

    if create_env:
        # Create the global environment and initialize random seed
        get_env()




class PerProcessContext(object):
    """A singletone that maintains the per process runtime properties.

    It is used mainly in multi-process distributed training mode,
    where properties such as the rank of the process and the total
    number of processes can be accessed via this interface.
    """
    _instance = None

    def __new__(cls):
        """Construct the singleton instance.

        This initializes the singleton and default values are assigned
        to the properties.
        """
        if cls._instance is None:
            cls._instance = super(PerProcessContext, cls).__new__(cls)
            cls._instance._read_only = False
            cls._instance._ddp_rank = -1
            cls._instance._num_processes = 1
        return cls._instance

    def finalize(self) -> None:
        """Lock the context so that it becomes read only.
        """
        self._read_only = True

    def set_distributed(self, rank: int, num_processes: int) -> None:
        """Set the distributed properties.

        Args:
            rank (int): the ID of the process
            num_processes (int): the total number of processes
        """
        if self._read_only:
            raise AttributeError(
                'Cannot mutate PerProcessContext after it is finalized')
        self._ddp_rank = rank
        self._num_processes = num_processes

    def set_paras_queue(self, paras_queue: mp.Queue):
        """Set the parameter queue.

        The queue is used for checking the consistency of model parameters across
        different worker processes, if multi-gpu training is used.
        """
        if self._read_only:
            raise AttributeError(
                'Cannot mutate PerProcessContext after it is finalized')
        self._paras_queue = paras_queue

    @property
    def paras_queue(self) -> mp.Queue:
        return self._paras_queue

    @property
    def is_distributed(self):
        return self._ddp_rank >= 0

    @property
    def ddp_rank(self):
        return self._ddp_rank

    @property
    def num_processes(self):
        return self._num_processes
    
class SpawnedProcessContext(NamedTuple):
    """Stores context information inherited from the main process.

    """
    ddp_num_procs: int
    ddp_rank: int
    env_id: int
    env_ctor: Callable[..., AlfEnvironment]
    pre_configs: List[Tuple[str, Any]]

    def create_env(self):
        """Creates an environment instance using the stored context."""
        return self.env_ctor(env_id=self.env_id)


_SPAWNED_PROCESS_CONTEXT = None


def set_spawned_process_context(context: SpawnedProcessContext):
    """Sets the context for the current spawned process.

    Should be called at the start of a spawned process by the main ALF process.

    Args:
        context (SpawnedProcessContext): The context to be stored.
    """
    global _SPAWNED_PROCESS_CONTEXT
    _SPAWNED_PROCESS_CONTEXT = context

class ConstantScheduler(object):
    def __init__(self, value):
        self._value = value

    def __call__(self):
        return self._value

    def __repr__(self):
        return str(self._value)

    def final_value(self):
        return self._value

def get_spawned_process_context() -> Optional[SpawnedProcessContext]:
    """Retrieves the spawned process context, if available.

    Returns:
        Optional[SpawnedProcessContext]: The current spawned process context.
    """
    return _SPAWNED_PROCESS_CONTEXT

def as_scheduler(value_or_scheduler):
    if isinstance(value_or_scheduler, Callable):
        return value_or_scheduler
    else:
        return ConstantScheduler(value_or_scheduler)
    
@configurable
class TrainerConfig(object):
    """Configuration for training."""

    def __init__(self,
                 root_dir,
                 conf_file='',
                 ml_type='rl',
                 algorithm_ctor=None,
                 data_transformer_ctor=None,
                 random_seed=None,
                 num_iterations=1000,
                 num_env_steps=0,
                 unroll_length=8,
                 unroll_with_grad=False,
                 use_root_inputs_for_after_train_iter=True,
                 async_unroll: bool = False,
                 max_unroll_length: int = 0,
                 unroll_queue_size: int = 200,
                 unroll_step_interval: float = 0,
                 unroll_parameter_update_period: int = 10,
                 use_rollout_state=False,
                 temporally_independent_train_step=None,
                 mask_out_loss_for_last_step=True,
                 sync_progress_to_envs=False,
                 num_checkpoints=10,
                 confirm_checkpoint_upon_crash=True,
                 no_thread_env_for_conf=False,
                 evaluate=False,
                 num_evals=None,
                 eval_interval=10,
                 epsilon_greedy=0.,
                 eval_uncertainty=False,
                 num_eval_episodes=10,
                 num_eval_environments: int = 1,
                 async_eval: bool = True,
                 save_checkpoint_for_best_eval: Optional[Callable] = None,
                 ddp_paras_check_interval: int = 0,
                 num_summaries=None,
                 summary_interval=50,
                 summarize_first_interval=True,
                 update_counter_every_mini_batch=False,
                 summaries_flush_secs=1,
                 summary_max_queue=10,
                 metric_min_buffer_size=10,
                 debug_summaries=False,
                 profiling=False,
                 enable_amp=False,
                 code_snapshots=None,
                 summarize_grads_and_vars=False,
                 summarize_gradient_noise_scale=False,
                 summarize_action_distributions=False,
                 summarize_output=False,
                 initial_collect_steps=0,
                 num_updates_per_train_iter=4,
                 mini_batch_length=None,
                 mini_batch_size=None,
                 whole_replay_buffer_training=True,
                 replay_buffer_length=1024,
                 priority_replay=False,
                 priority_replay_alpha=0.7,
                 priority_replay_beta=0.4,
                 priority_replay_eps=1e-6,
                 offline_buffer_dir=None,
                 offline_buffer_length=None,
                 rl_train_after_update_steps=0,
                 rl_train_every_update_steps=1,
                 empty_cache: bool = False,
                 normalize_importance_weights_by_max: bool = False,
                 clear_replay_buffer=True,
                 clear_replay_buffer_but_keep_one_step=True,
                 visualize_alf_tree=False,
                 remote_training=False):
        """
        Args:
            root_dir (str): directory for saving summary and checkpoints
            ml_type (str): type of learning task, one of ['rl', 'sl']
            algorithm_ctor (Callable): callable that create an
                ``OffPolicyAlgorithm`` or ``OnPolicyAlgorithm`` instance
            data_transformer_ctor (Callable|list[Callable]): Function(s)
                for creating data transformer(s). Each of them will be called
                as ``data_transformer_ctor(observation_spec)`` to create a data
                transformer. Available transformers are in ``algorithms.data_transformer``.
                The data transformer constructed by this can be access as
                ``TrainerConfig.data_transformer``.
                Important Note: ``HindsightExperienceTransformer``, ``FrameStacker`` or
                any data transformer that need to access the replay buffer
                for additional data need to be before all other data transformers.
                The reason is the following:
                In off policy training, the replay buffer stores raw input w/o being
                processed by any data transformer.  If say ``ObservationNormalizer`` is
                applied before hindsight, then data retrieved by replay will be
                normalized whereas hindsight data directly pulled from the replay buffer
                will not be normalized.  Data will be in mismatch, causing training to
                suffer and potentially fail.
            random_seed (None|int): random seed, a random seed is used if None.
                For a None random seed, all DDP ranks (if multi-gpu training used)
                will have a None random seed set to their ``TrainerConfig.random_seed``.
                This means that the actual random seed used by each rank is purely
                random. A None random seed won't set a deterministic torch behavior.
                If a specific random seed is set, DDP rank>0 (if multi-gpu training
                used) will have a random seed set to a value that is deterministically
                "randomized" from this random seed. In this case, all ranks will
                have a deterministic torch behavior. NOTE: By the current design,
                you won't be able to reproduce a training job if its random seed
                was set as None. For reproducible training jobs, always set the
                random seed in the first place.
            num_iterations (int): For RL trainer, indicates number of update
                iterations (ignored if 0). Note that for off-policy algorithms, if
                ``initial_collect_steps>0``, then the first
                ``initial_collect_steps//(unroll_length*num_envs)`` iterations
                won't perform any training. For SL trainer, indicates the number
                of training epochs. If both `num_iterations` and `num_env_steps`
                are set, `num_iterations` must be big enough to consume so many
                environment steps. And after `num_env_steps` environment steps are
                generated, the training will not interact with environments
                anymore, which means that it will only train on replay buffer.
            num_env_steps (int): number of environment steps (ignored if 0). The
                total number of FRAMES will be (``num_env_steps*frame_skip``) for
                calculating sample efficiency. See alf/environments/wrappers.py
                for the definition of FrameSkip.
            unroll_length (float):  number of time steps each environment proceeds per
                iteration. The total number of time steps from all environments per
                iteration can be computed as: ``num_envs * env_batch_size * unroll_length``.
                If ``unroll_length`` is not an integer, the actual unroll_length
                being used will fluctuate between ``floor(unroll_length)`` and
                ``ceil(unroll_length)`` and the expectation will be equal to
                ``unroll_length``.
            unroll_with_grad (bool): a bool flag indicating whether we require
                grad during ``unroll()``. This flag is only used by
                ``OffPolicyAlgorithm`` where unrolling with grads is usually
                unnecessary and turned off for saving memory. However, when there
                is an on-policy sub-algorithm, we can enable this flag for its
                training. ``OnPolicyAlgorithm`` always unrolls with grads and this
                flag doesn't apply to it.
            use_root_inputs_for_after_train_iter (bool): whether to use root_inputs
                (TimeStep) to call ``after_train_iter()``. If False, the the root_inputs
                argument will be ``None``. When memory usage is a concern, setting
                this to False can save memory.
            async_unroll: whether to unroll asynchronously. If True, unroll will
                be performed in parallel with training.
            max_unroll_length: the maximal length of unroll results for each iteration.
                If the time for one step of training is less than the time for
                unrolling ``max_unroll_length`` steps, the length of the unroll
                results will be less than ``max_unroll_length``. Only used if
                ``async_unroll`` is True and unroll_length==0.
            unroll_queue_size: the size of the queue for transmitting unroll
                results from the unroll process to the main process. Only used
                if ``async_unroll`` is True. If the queue is full, the unroll process
                will wait for the main process to retrieve unroll results from
                the queue before performing more unrolls.
            unroll_step_interval: if not zero, the time interval in second
                between each two environment steps. Only used if ``async_unroll`` is True.
                This is useful if the interaction with the environment happens
                in real time (e.g. real world robot or real time simulation) and
                you want a fixed interaction frequency with the environment.
                Note that this will not have any effect if environment step and
                rollout step together spend more than unroll_step_interval.
            unroll_parameter_update_period: update the parameter for the asynchronous
                unroll every so many iterations. Only used if ``async_unroll`` is True.
            use_rollout_state (bool): If True, when off-policy training, the RNN
                states will be taken from the replay buffer; otherwise they will
                be set to 0. In the case of True, the ``train_state_spec`` of an
                algorithm should always be a subset of the ``rollout_state_spec``.
            mask_out_loss_for_last_step (bool): If True, the loss for the last
                step of each sequence will be masked out. For RL training,
                the last step of each episode is usually a terminal state and
                the loss for it is not meaningful. Note that most RL algorithms
                implemented in ALF implicitly assumes this behavior.
            temporally_independent_train_step (bool|None): If True, the ``train_step``
                is called with all the experiences in one batch instead of being
                called sequentially with ``mini_batch_length`` batches. Only used
                by ``OffPolicyAlgorithm``. In general, this option can only be
                used if the algorithm has no state. For Algorithm with state (e.g.
                ``SarsaAlgorithm`` not using RNN), if there is no need to
                recompute state at train_step, this option can also be used. If
                ``None``, its value is inferred based on whether the algorithm
                has RNN state (``True`` if there is no RNN state, ``False`` if yes).
            sync_progress_to_envs: ALF schedulers need to have progress information
                to calculate scheduled values. For parallel environments, the
                environments are in different processes and the progress information
                needs to be synced with the main in order to use schedulers in
                the environment.
            num_checkpoints (int): how many checkpoints to save for the training
            confirm_checkpoint_upon_crash (bool): whether to prompt for whether
                do checkpointing after crash.
            no_thread_env_for_conf (bool): not to create an unwrapped env for
                the purpose of showing operative configurations. If True, no
                ``ThreadEnvironment`` will ever be created, regardless of the
                value of ``TrainerConfig.evaluate``. If False, a
                ``ThreadEnvironment`` will be created if ``TrainerConfig.evaluate``
                or the training env is a ``ParallelAlfEnvironment`` instance.
                For an env that consume lots of resources, this flag can be set to
                ``True`` if no evaluation is needed to save resources. The decision
                of creating an unwrapped env won't affect training; it's used to
                correctly display inoperative configurations in subprocesses.
            evaluate (bool): A bool to evaluate when training
            num_evals (int): how many evaluations are needed throughout the training.
                If not None, an automatically calculated ``eval_interval`` will
                replace ``config.eval_interval``.
            eval_interval (int): evaluate every so many iteration
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation.
            eval_uncertainty (bool): whether to evaluate uncertainty after training.
            num_eval_episodes (int) : number of episodes for one evaluation.
            num_eval_environments: the number of environments for evaluation.
            async_eval: whether to do evaluation asynchronously in a different
                process. Note that this may use more memory.
            save_checkpoint_for_best_eval: If provided, will be called with a list of
                evaluation metrics. If it returns True, a checkpoint will be saved.
                A possible value of this option is `alf.trainers.evaluator.BestEvalChecker()`,
                which will save a checkpoint if the specified evaluation metric
                are better than the previous best.
            ddp_paras_check_interval: if >0, then every so many iterations the trainer
                will perform a consistency check of the model parameters across
                different worker processes, if multi-gpu training is used.
            num_summaries (int): how many summary calls are needed throughout the
                training. If not None, an automatically calculated ``summary_interval``
                will replace ``config.summary_interval``. Note that this number
                doesn't include the summary steps of the first interval if
                ``summarize_first_interval=True``. In this case, the actual number
                of summaries will be roughly this number plus the calculated
                summary interval.
            summary_interval (int): write summary every so many training steps
            summarize_first_interval (bool): whether to summarize every step of
                the first interval (default True). It might be better to turn
                this off for an easier post-processing of the curve.
            update_counter_every_mini_batch (bool): whether to update counter
                for every mini batch. The ``summary_interval`` is based on this
                counter. Typically, this should be False. Set to True if you
                want to have summary for every mini batch for the purpose of
                debugging. Only used by ``OffPolicyAlgorithm``.
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            metric_min_buffer_size (int): a minimal size of the buffer used to
                construct some average episodic metrics used in ``RLAlgorithm``.
            debug_summaries (bool): A bool to gather debug summaries.
            profiling (bool): If True, use cProfile to profile the training. The
                profile result will be written to ``root_dir``/py_train.INFO.
            enable_amp: whether to use automatic mixed precision for training.
                This can makes the training faster if the algorithm is GPU intensive.
                However, the result may be different (mostly likely due to random
                fluctuation).
            code_snapshots (list[str]): an optional list of code files to write
                to tensorboard text. Note: the code file path should be relative
                to "<ALF_ROOT>/alf", e.g., "algorithms/agent.py". This can be
                useful for tracking code changes when running a job.
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
            summarize_gradient_noise_scale (bool): whether summarize gradient
                noise scale. See ``alf.optimizers.utils.py`` for details.
            summarize_output (bool): If True, summarize output of certain networks.
            initial_collect_steps (int): if positive, number of steps each single
                environment steps before perform first update. Only used
                by ``OffPolicyAlgorithm``.
            num_updates_per_train_iter (int): number of optimization steps for
                one iteration. Only used by ``OffPolicyAlgorithm``.
            mini_batch_size (int): number of sequences for each minibatch. If None,
                it's set to the replayer's ``batch_size``. Only used by
                ``OffPolicyAlgorithm``.
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch. Only used by ``OffPolicyAlgorithm``.
            whole_replay_buffer_training (bool): whether use all data in replay
                buffer to perform one update. Only used by ``OffPolicyAlgorithm``.
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean. Only used by
                ``OffPolicyAlgorithm``.
            clear_replay_buffer_but_keep_one_step (bool): If True, for `clear_replay_buffer=True`,
                keep the last step of each sequence in the replay buffer. This
                is to prevent the last batch of experiences in each iteration
                from never getting properly trained (e.g. trained by the bootstrap
                value from the next step). Only used by ``OffPolicyAlgorithm``.
            replay_buffer_length (int): the maximum number of steps the replay
                buffer store for each environment. Only used by
                ``OffPolicyAlgorithm``.
            priority_replay (bool): Use prioritized sampling if this is True.
            priority_replay_alpha (float|Scheduler): The priority from LossInfo is powered
                to this as an argument for ``ReplayBuffer.update_priority()``.
                Note that the effect of ``ReplayBuffer.initial_priority``
                may change with different values of ``priority_replay_alpha``.
                Hence you may need to adjust ``ReplayBuffer.initial_priority``
                accordingly.
            priority_replay_beta (float|Scheduler): weight the loss of each sample by
                ``importance_weight**(-priority_replay_beta)``, where ``importance_weight``
                is from the BatchInfo returned by ``ReplayBuffer.get_batch()``.
                This is only useful if ``prioritized_sampling`` is enabled for
                ``ReplayBuffer``.
            priority_replay_eps (float): minimum priority for priority replay.
            offline_buffer_dir (str|[str]): path to the offline replay buffer
                checkpoint to be loaded. If a list of strings provided, each
                will represent the directory to one replay buffer checkpoint.
            offline_buffer_length (int): the maximum length will be loaded
                from each replay buffer checkpoint. Therefore the total
                buffer length is offline_buffer_length * len(offline_buffer_dir).
                If None, all the samples from all the provided replay buffer
                checkpoints will be loaded.
            rl_train_after_update_steps (int): only used in the hybrid training
                mode. It is used as a starting criteria for the normal (non-offline)
                part of the RL training, which only starts after so many number
                of update steps (according to ``global_counter``).
            rl_train_every_update_steps (int): only used in the hybrid training
                mode. It is used to control the update frequency of the normal
                (non-offline) part of the RL training  (according to
                ``global_counter``). Through this flag, we can have a more fine
                grained control over the update frequencies of online and offline
                RL training (currently assumes the training frequency of offline
                RL is always higher or equal to the online RL part).
                For example, we can set ``rl_train_every_update_steps = 2``
                to have a train config that executes online RL training at the
                half frequency of that of the offline RL training.
            empty_cache: empty GPU memory cache at the start of every iteration
                to reduce GPU memory usage. This option may slightly slow down
                the overall speed.
            normalize_importance_weights_by_max: if True, normalize the importance
                weights by its max to prevent instability caused by large importance
                weight.
            visualize_alf_tree: if True, will call graphviz to draw a model
                structure of the algorithm.
            remote_training: a flag preserved to indicate remote training mode.
                It will be automatically set by ``alf.bin.train`` and users should
                *not* set this flag. Three possible values: ['trainer', 'unroller',
                False].
        """
        if isinstance(priority_replay_beta, float):
            assert priority_replay_beta >= 0.0, (
                "importance_weight_beta should be non-negative")
        assert ml_type in ('rl', 'sl')
        self.root_dir = root_dir
        self.conf_file = conf_file
        self.ml_type = ml_type
        self.algorithm_ctor = algorithm_ctor
        self.data_transformer_ctor = data_transformer_ctor
        self.data_transformer = None  # to be set by Trainer
        self.random_seed = random_seed
        self.num_iterations = num_iterations
        self.num_env_steps = num_env_steps
        self.unroll_length = unroll_length
        self.unroll_with_grad = unroll_with_grad
        self.use_root_inputs_for_after_train_iter = use_root_inputs_for_after_train_iter
        self.async_unroll = async_unroll
        if async_unroll:
            assert not unroll_with_grad, ("unroll_with_grad is not supported "
                                          "for async_unroll=True")
            assert max_unroll_length > 0, ("max_unroll_length needs to be set "
                                           "for async_unroll=True")
        self.max_unroll_length = max_unroll_length or self.unroll_length
        self.unroll_queue_size = unroll_queue_size
        self.unroll_step_interval = unroll_step_interval
        self.unroll_parameter_update_period = unroll_parameter_update_period
        self.use_rollout_state = use_rollout_state
        self.mask_out_loss_for_last_step = mask_out_loss_for_last_step
        self.temporally_independent_train_step = temporally_independent_train_step
        self.sync_progress_to_envs = sync_progress_to_envs
        self.num_checkpoints = num_checkpoints
        self.confirm_checkpoint_upon_crash = confirm_checkpoint_upon_crash
        self.no_thread_env_for_conf = no_thread_env_for_conf
        self.evaluate = evaluate
        self.num_evals = num_evals
        self.eval_interval = eval_interval
        self.epsilon_greedy = epsilon_greedy
        self.eval_uncertainty = eval_uncertainty
        self.num_eval_episodes = num_eval_episodes
        self.num_eval_environments = num_eval_environments
        self.async_eval = async_eval
        self.save_checkpoint_for_best_eval = save_checkpoint_for_best_eval
        self.ddp_paras_check_interval = ddp_paras_check_interval
        self.num_summaries = num_summaries
        self.summary_interval = summary_interval
        self.summarize_first_interval = summarize_first_interval
        self.update_counter_every_mini_batch = update_counter_every_mini_batch
        self.summaries_flush_secs = summaries_flush_secs
        self.summary_max_queue = summary_max_queue
        self.metric_min_buffer_size = metric_min_buffer_size
        self.debug_summaries = debug_summaries
        self.profiling = profiling
        self.enable_amp = enable_amp
        self.code_snapshots = code_snapshots
        self.summarize_grads_and_vars = summarize_grads_and_vars
        self.summarize_gradient_noise_scale = summarize_gradient_noise_scale
        self.summarize_action_distributions = summarize_action_distributions
        self.summarize_output = summarize_output
        self.initial_collect_steps = initial_collect_steps
        self.num_updates_per_train_iter = num_updates_per_train_iter
        self.mini_batch_length = mini_batch_length
        self.mini_batch_size = mini_batch_size
        self.whole_replay_buffer_training = whole_replay_buffer_training
        self.clear_replay_buffer = clear_replay_buffer
        self.clear_replay_buffer_but_keep_one_step = clear_replay_buffer_but_keep_one_step
        self.replay_buffer_length = replay_buffer_length
        self.priority_replay = priority_replay
        self.priority_replay_alpha = as_scheduler(priority_replay_alpha)
        self.priority_replay_beta = as_scheduler(priority_replay_beta)
        self.priority_replay_eps = priority_replay_eps
        # offline options
        self.offline_buffer_dir = offline_buffer_dir
        self.offline_buffer_length = offline_buffer_length
        self.rl_train_after_update_steps = rl_train_after_update_steps
        self.rl_train_every_update_steps = rl_train_every_update_steps
        self.empty_cache = empty_cache
        self.normalize_importance_weights_by_max = normalize_importance_weights_by_max
        self.visualize_alf_tree = visualize_alf_tree
        self.remote_training = remote_training

def adjust_config_by_multi_process_divider(ddp_rank: int,
                                           multi_process_divider: int = 1):
    """Adjust specific configuration value in multiple process settings
    Alf assumes all configuration files geared towards single process training.
    This means that in multi-process settings such as DDP some of the
    configuration values needs to be adjusted to achieve parity on number of
    processes.
    For example, if we run 64 environments in parallel for single process
    settings, the value needs to be overridden with 16 if there are 4 identical
    processes running DDP training to achieve parity.

    The adjusted configs are

    1. TrainerConfig.mini_batch_size: divided by processes
    2. TrainerConfig.num_env_steps: divided by processes if used
    3. TrainerConfig.mini_batch_size: divided by processes if used
    4. TrainerConfig.evaluate: set to False except for process 0

    Args:
        ddp_rank: the rank of device to the process.
        multi_process_divider: this is equivalent to number of processes
    """
    if multi_process_divider <= 1:
        return

    # Adjust the num of environments per process. The value for single process
    # (before adjustment) is divided by the multi_process_divider and becomes
    # the per-process value.
    tag = 'create_environment.num_parallel_environments'
    num_parallel_environments = get_config_value(tag)
    config1(
        tag,
        math.ceil(num_parallel_environments / multi_process_divider),
        raise_if_used=False,
        override_all=True)

    # Adjust the mini_batch_size. If the original configured value is 64 and
    # there are 4 processes, it should mean that "jointly the 4 processes have
    # an effective mini_batch_size of 64", and each process has a
    # mini_batch_size of 16.
    tag = 'TrainerConfig.mini_batch_size'
    mini_batch_size = get_config_value(tag)
    if isinstance(mini_batch_size, int):
        config1(
            tag,
            math.ceil(mini_batch_size / multi_process_divider),
            raise_if_used=False,
            override_all=True)

    # If the termination condition is num_env_steps instead of num_iterations,
    # we need to adjust it as well since each process only sees env steps taking
    # by itself.
    tag = 'TrainerConfig.num_env_steps'
    num_env_steps = get_config_value(tag)
    if num_env_steps > 0:
        config1(
            tag,
            math.ceil(num_env_steps / multi_process_divider),
            raise_if_used=False,
            override_all=True)

    tag = 'TrainerConfig.initial_collect_steps'
    init_collect_steps = get_config_value(tag)
    config1(
        tag,
        math.ceil(init_collect_steps / multi_process_divider),
        raise_if_used=False,
        override_all=True)

    # Only allow process with rank 0 to have evaluate. Enabling evaluation for
    # other parallel processes is a waste as such evaluation does not offer more
    # information.
    if ddp_rank > 0:
        config1(
            'TrainerConfig.evaluate',
            False,
            raise_if_used=False,
            override_all=True)



def _array_to_tensor(data):
    def _array_to_tensor(obj):
        return torch.as_tensor(obj).unsqueeze(
            dim=0) if isinstance(obj, (np.ndarray, numbers.Number)) else obj

    return nest.map_structure(_array_to_tensor, data)


def _tensor_to_array(data):
    return nest.map_structure(lambda x: x.squeeze(dim=0).cpu().numpy(), data)

@configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         image_channel_first=True):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is applied
            if set to 0 or if there is no max_episode_steps set in the environment's
            spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        image_channel_first (bool): whether transpose image channels to first dimension.

    Returns:
        An AlfEnvironment instance.
    """
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make()

    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    return gym_wrappers.wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=image_channel_first)

class ThreadEnvironment(AlfEnvironment):
    """Create, Step a single env in a separate thread
    """

    def __init__(self, env_constructor):
        """Create a ThreadEnvironment

        Args:
            env_constructor (Callable): env_constructor for the OpenAI Gym environment
        """
        super().__init__()
        self._pool = mp_threads.Pool(1)
        self._env = self._pool.apply(env_constructor)
        assert not self._env.batched
        self._closed = False

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return 1

    def env_info_spec(self):
        return self._apply('env_info_spec')

    def observation_spec(self):
        return self._apply('observation_spec')

    def action_spec(self):
        return self._apply('action_spec')

    def reward_spec(self):
        return self._apply('reward_spec')

    def _step(self, action):
        action = _tensor_to_array(action)
        return _array_to_tensor(self._apply('step', (action, )))

    def _reset(self):
        return _array_to_tensor(self._apply('reset'))

    def close(self):
        if self._closed:
            return
        self._apply('close')
        self._pool.close()
        self._pool.join()
        self._closed = True

    def render(self, mode='rgb_array'):
        return self._apply('render', (mode, ))

    def seed(self, seed):
        self._apply('seed', (seed, ))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _apply(self, name, args=()):
        func = getattr(self._env, name)
        return self._pool.apply(func, args)

def get_default_device():
    return torch._C._get_default_device()
_devece_ddtype_tensor_map = {
    'cpu': {
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
        torch.float16: torch.HalfTensor,
        torch.uint8: torch.ByteTensor,
        torch.int8: torch.CharTensor,
        torch.int16: torch.ShortTensor,
        torch.int32: torch.IntTensor,
        torch.int64: torch.LongTensor,
        torch.bool: torch.BoolTensor,
    },
    'cuda': {
        torch.float32: torch.cuda.FloatTensor,
        torch.float64: torch.cuda.DoubleTensor,
        torch.float16: torch.cuda.HalfTensor,
        torch.uint8: torch.cuda.ByteTensor,
        torch.int8: torch.cuda.CharTensor,
        torch.int16: torch.cuda.ShortTensor,
        torch.int32: torch.cuda.IntTensor,
        torch.int64: torch.cuda.LongTensor,
        torch.bool: torch.cuda.BoolTensor,
    }
}


def set_default_device(device_name):
    """Set the default device.

    Cannot find a native torch function for setting default device. We have to
    hack our own.

    Args:
        device_name (str): one of ("cpu", "cuda")
    """
    torch.set_default_tensor_type(
        _devece_ddtype_tensor_map[device_name][torch.get_default_dtype()])



class _MessageType(Enum):
    """Message types for communication via the pipe.

    The ProcessEnvironment uses pipe to perform IPC, where each of the message
    has a message type. This Enum provides all the available message types.
    """
    READY = 1
    ACCESS = 2
    CALL = 3
    RESULT = 4
    EXCEPTION = 5
    CLOSE = 6
    SYNC_PROGRESS = 7


_is_scheduler_allowed = True

_progress = {
    "percent": 0.0,
    "iterations": 0.0,
    "env_steps": 0.0,
    "global_counter": 0.0
}

FLAGS = flags.FLAGS
def update_all_progresses(progresses):
    _progress.update(progresses)

def disallow_scheduler():
    """Disallow the use of scheduler.

    In some context, scheduler cannot be used due to the lack of the information
    about training progress. This function is called by the framework to prevent
    the use of scheduler in such context.
    """
    global _is_scheduler_allowed
    _is_scheduler_allowed = False

def process_call(conn, env, flatten, action_spec):
    """
    Returns:
        True: continue to work
        False: end the worker
    """
    try:
        # Only block for short times to have keyboard exceptions be raised.
        while True:
            if conn.poll(0.1):
                break
        message, payload = conn.recv()
    except (EOFError, KeyboardInterrupt):
        return False
    if message == _MessageType.ACCESS:
        name = payload
        result = getattr(env, name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.CALL:
        name, args, kwargs = payload
        if flatten and name == 'step':
            args = [nest.pack_sequence_as(action_spec, args[0])]
        result = getattr(env, name)(*args, **kwargs)
        if flatten and name in ['step', 'reset']:
            result = nest.flatten(result)
            assert all([not isinstance(x, torch.Tensor) for x in result
                        ]), ("Tensor result is not allowed: %s" % name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.SYNC_PROGRESS:
        update_all_progresses(payload)
    elif message == _MessageType.CLOSE:
        assert payload is None
        env.close()
        return False
    else:
        raise KeyError('Received message of unknown type {}'.format(message))
    return True

def _init_after_spawn(context: SpawnedProcessContext):
    """Perform necessary initialization of flags and configurations when a new
    subprocess is "spawn"-ed for ``ProcessEnvironment``.

    This function is not needed if the subprocess is created via "fork".
    However, if it is created via "spawn", the subprocess will not automatically
    inherit resources such as the ALF configurations, and this function needs to
    be called to ensure the ALF configurations are initialized.

    Args:
        pre_configs: Specifies the set of pre configs that the parent process uses.

    """
    if context.ddp_rank >= 0:
        PerProcessContext().set_distributed(context.ddp_rank,
                                            context.ddp_num_procs)

    # 0. Update the global context for this spawned process. This will
    #    alter the behavior of ``get_env()``.
    set_spawned_process_context(context)

    # 1. Parse the relevant flags for the current subprocess. The set of
    #    relevant flags are defined below. Note that the command line arguments
    #    and options are inherited from the parent process via ``sys.argv``.
    flags.DEFINE_string(
        "root_dir", None,
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_string("conf", None, "Path to the alf config file.")
    flags.DEFINE_multi_string("conf_param", None, "Config binding parameters.")
    FLAGS(sys.argv, known_only=True)
    FLAGS.mark_as_parsed()
    FLAGS.alsologtostderr = True
    conf_file = get_conf_file()

    # 2. Configure the logging
    logging.set_verbosity(logging.INFO)

    # 3. Load the configuration
    pre_config(dict(context.pre_configs))
    parse_conf_file(conf_file)

def _worker(conn: multiprocessing.connection,
            env_constructor: Callable,
            start_method: str,
            pre_configs: List[Tuple[str, Any]],
            env_id: int = None,
            flatten: bool = False,
            fast: bool = False,
            num_envs: int = 0,
            torch_num_threads_per_env: int = 1,
            ddp_num_procs: int = 1,
            ddp_rank: int = -1,
            name: str = ''):
    """The process waits for actions and sends back environment results.

    Args:
        conn: Connection for communication to the main process.
        env_constructor: callable environment creator.
        start_method: whether this subprocess is created via "fork" or "spawn".
        pre_configs: pre configs that need to be inherited if created via "spawn".
        env_id: the id of the env
        flatten: whether to assume flattened actions and time_steps
          during communication to avoid overhead.
        fast: whether created by ``FastParallelEnvironment`` or not.
        num_envs: number of environments in the ``FastParallelEnvironment``.
            Only used if ``fast`` is True.
        torch_num_threads_per_env: how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best to
            set this number as 1. Leave this as 'None' to skip the change.
            Note that this option also affect the thread num for numpy.
        name: name of the FastParallelEnvironment. Only used if ``fast`` is True.

    Raises:
        KeyError: When receiving a message of unknown type.
    """
    try:
        set_default_device("cpu")
        if torch_num_threads_per_env is not None:
            torch.set_num_threads(torch_num_threads_per_env)
            # recommended way of changing num_threads for numpy:
            # https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra
            threadpoolctl.threadpool_limits(torch_num_threads_per_env)
        if start_method == "spawn":
            _init_after_spawn(
                SpawnedProcessContext(
                    ddp_num_procs=ddp_num_procs,
                    ddp_rank=ddp_rank,
                    env_id=env_id,
                    env_ctor=env_constructor,
                    pre_configs=pre_configs))
            # env may have been created during parse_conf_file called by _init_after_spawn
            # so we should not create it again using env_constructor
            env = get_env()
        else:
            env = env_constructor(env_id=env_id)
        if not get_config_value("TrainerConfig.sync_progress_to_envs"):
            disallow_scheduler()
        action_spec = env.action_spec()
        if fast:
            penv = _penv.ProcessEnvironment(
                env, partial(process_call, conn, env, flatten,
                             action_spec), env_id, num_envs, env.batch_size,
                env.batched, env.action_spec(),
                env.time_step_spec()._replace(env_info=env.env_info_spec()),
                name)
            conn.send(_MessageType.READY)  # Ready.
            try:
                penv.worker()
            except KeyboardInterrupt:
                penv.quit()
            except Exception:
                traceback.print_exc()
                penv.quit()
        else:
            conn.send(_MessageType.READY)  # Ready.
            while True:
                if not process_call(conn, env, flatten, action_spec):
                    break
    except KeyboardInterrupt:
        # When worker receives interruption from keyboard (i.e. Ctrl-C), notify
        # the parent process to shut down quietly by sending the CLOSE message.
        #
        # This is to avoid sometimes tens of environment processes panicking
        # simultaneously.
        conn.send((_MessageType.CLOSE, None))
    except Exception:  # pylint: disable=broad-except
        etype, evalue, tb = sys.exc_info()
        stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
        message = 'Error in environment process: {}'.format(stacktrace)
        logging.error(message)
        conn.send((_MessageType.EXCEPTION, stacktrace))
    finally:
        conn.close()


def get_handled_pre_configs():
    """Return a list of handled pre-config ``(name, value)``."""
    return _HANDLED_PRE_CONFIGS

def get_all_progresses():
    return _progress

class ProcessEnvironment(object):
    def __init__(self,
                 env_constructor: Callable,
                 env_id: int = None,
                 flatten: bool = False,
                 fast: bool = False,
                 num_envs: int = 0,
                 torch_num_threads_per_env: int = 1,
                 start_method: str = "fork",
                 name: str = ""):
        """Step environment in a separate process for lock free parallelism.

        The environment is created in an external process by calling the provided
        callable. This can be an environment class, or a function creating the
        environment and potentially wrapping it. The returned environment should
        not access global variables.

        Args:
            env_constructor: callable environment creator.
            env_id: ID of the the env
            flatten: whether to assume flattened actions and time_steps
                during communication to avoid overhead.
            fast: whether created by ``FastParallelEnvironment`` or not.
            num_envs: number of environments in the ``FastParallelEnvironment``.
                Only used if ``fast`` is True.
            torch_num_threads_per_env: how many threads torch will use for each
                env proc. Note that if you have lots of parallel envs, it's best
                to set this number as 1. Leave this as 'None' to skip the change.
            start_method: whether this subprocess is created via "fork" or "spawn".
            name: name of the FastParallelEnvironment. Only used if ``fast``
                is True.

        Attributes:
            observation_spec: The cached observation spec of the environment.
            action_spec: The cached action spec of the environment.
            time_step_spec: The cached time step spec of the environment.
        """
        self._env_constructor = env_constructor
        self._flatten = flatten
        self._env_id = env_id
        self._observation_spec = None
        self._action_spec = None
        self._reward_spec = None
        self._time_step_spec = None
        self._env_info_spec = None
        self._conn = None
        self._fast = fast
        self._num_envs = num_envs
        self._torch_num_threads = torch_num_threads_per_env
        assert start_method in [
            "fork", "spawn"
        ], (f"Unrecognized start method '{start_method}' specified for "
            "ProcessEnvironment. It should be either 'fork' or 'spawn'.")
        self._start_method = start_method
        self._name = name
        if fast:
            self._penv = _penv.ProcessEnvironmentCaller(env_id, name)

    def start(self, wait_to_start=True):
        """Start the process.

        Args:
            wait_to_start (bool): Whether the call should wait for an env initialization.
        """
        assert not self._conn, "Cannot start() ProcessEnvironment multiple times"
        mp_ctx = multiprocessing.get_context(self._start_method)
        self._conn, conn = mp_ctx.Pipe()

        ddp_num_procs = PerProcessContext().num_processes
        ddp_rank = PerProcessContext().ddp_rank

        self._process = mp_ctx.Process(
            target=_worker,
            args=(conn, self._env_constructor, self._start_method,
                  get_handled_pre_configs(), self._env_id, self._flatten,
                  self._fast, self._num_envs, self._torch_num_threads,
                  ddp_num_procs, ddp_rank, self._name),
            name=f"ProcessEnvironment-{self._env_id}")
        atexit.register(self.close)
        self._process.start()
        if wait_to_start:
            self.wait_start()

    def wait_start(self):
        """Wait for the started process to finish initialization."""
        assert self._conn, "Run ProcessEnvironment.start() first"
        result = self._conn.recv()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result == _MessageType.READY, result

    def env_info_spec(self):
        if not self._env_info_spec:
            self._env_info_spec = self.call('env_info_spec')()
        return self._env_info_spec

    def observation_spec(self):
        if not self._observation_spec:
            self._observation_spec = self.call('observation_spec')()
        return self._observation_spec

    def action_spec(self):
        if not self._action_spec:
            self._action_spec = self.call('action_spec')()
        return self._action_spec

    def reward_spec(self):
        if not self._reward_spec:
            self._reward_spec = self.call('reward_spec')()
        return self._reward_spec

    def time_step_spec(self):
        if not self._time_step_spec:
            self._time_step_spec = self.call('time_step_spec')()
        return self._time_step_spec

    def __getattr__(self, name):
        """Request an attribute from the environment.

        Note that this involves communication with the external process, so it can
        be slow.

        Args:
            name (str): Attribute to access.

        Returns:
            Value of the attribute.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        self._conn.send((_MessageType.ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        Args:
            name (str): Name of the method to call.
            *args: Positional arguments to forward to the method.
            **kwargs: Keyword arguments to forward to the method.

        Returns:
            Promise object that blocks and provides the return value when called.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        payload = name, args, kwargs
        self._conn.send((_MessageType.CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            if self._fast:
                self._penv.close()
            else:
                self._conn.send((_MessageType.CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        """Step the environment.

        Args:
            action (nested tensors): The action to apply to the environment.
            blocking (bool): Whether to wait for the result.

        Returns:
            time step when blocking, otherwise callable that returns the time step.
        """
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        """Reset the environment.

        Args:
            blocking (bool): Whether to wait for the result.

        Returns:
            New observation when blocking, otherwise callable that returns the new
            observation.
        """
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def sync_progress(self):
        """Sync the progress of the environment.
        """
        if self._fast:
            self._penv.call()
        self._conn.send((_MessageType.SYNC_PROGRESS, get_all_progresses()))

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        Raises:
            Exception: An exception was raised inside the worker process.
            KeyError: The received message is of an unknown type.

        Returns:
            Payload object of the message.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        message, payload = self._conn.recv()

        # Re-raise exceptions in the main process.
        if message == _MessageType.EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        elif message == _MessageType.RESULT:
            return payload
        elif message == _MessageType.CLOSE:
            # When notified that the child process is going to shut down, do not
            # panic and handle it quietly.
            return None
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): One of ['rgb_array', 'human']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
        Returns:
            An ndarray of shape [width, height, 3] denoting an RGB image if mode is
            `rgb_array`. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
        return self.call('render', mode)()
    
@configurable
class FastParallelEnvironment(AlfEnvironment):
    """Batch together environments and simulate them in external processes.

    The environments are created in external processes by calling the provided
    callables. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The environments can be different
    but must use the same action and observation specs.

    Different from ``parallel_environment.ParallelAlfEnvironment``, ``FastParallelEnvironment``
    uses shared memory to transfer ``TimeStep`` from each process environment
    to the main process.

    Terminology:

    - main process: the process where ParallelEnvironment is created
    - client process: the process running the actual individual environment created
        using env_constructors

    Design:

    ``FastParallelEnvironment`` uses ``_penv.ParallelEnvironment`` (implemented in C++)
    to coordinate step() and reset().
    Each ``ProcessEnvironment`` maintains one ``_penv.ProcessEnvironmentCaller``
    in the main process and one ``_penv.ProcessEnvironment`` in the client process.

    In the client process, ``_penv.ProcessEnvironment.worker()`` runs in a loop to
    wait for jobs from either ``_penv.ParallelEnvironment`` or ``_penv.ProcessEnvironmentCaller``.

    There are 4 types of job:

    - step: step the environment. Sent from ``_penv.ParallelEnvironment``. The
        result is communicated back using shared memory.
    - reset: reset the environment. Sent from ``_penv.ParallelEnvironment``.
        The result is communicated back using shared memory.
    - close: close the environment. Sent from ``_penv.ProcessEnvironmentCaller``.
        This will cause the worker to finish and quit the process.
    - call: access other methods of the environment. Sent from ``_penv.ProcessEnvironmentCaller``.
        This takes advantage of the pipe mechanism used by  the ``ParallelAlfEnvironment``.
        This is achieved by calling ``call_handler`` to do communication using
        python pipe. The reason of using the original pipe mechanism for other
        types of communication is that it is not easy to handle communication of
        unknown size using shared memory.

    Args:
        env_constructors (list[Callable]): a list of callable environment creators.
            Each callable should accept env_id as the only argument and return
            the created environment.
        start_serially (bool): whether to start environments serially or in parallel.
        blocking (bool): not used. Kept for the same interface as ``ParallelAlfEnvironment``.
        flatten (bool): not used. Kept for the same interface as ``ParallelAlfEnvironment``.
        num_spare_envs_for_reload (int): if positive, these environments will be
            maintained in a separate queue and be used to handle slow env resets.
            The batch_size is ``len(env_constructors) - num_spare_envs_for_reload``
        torch_num_threads_per_env (int): how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best
            to set this number as 1. Leave this as 'None' to skip the change.
            Note that this option also affect the thread num for numpy.
        start_method: either "fork" or "spawn". This specifies how the
            subprocesses for the ``ProcessEnvironment`` is created. Normally you
            should use "fork" because it is faster. There are cases when "fork"
            is too bloated or not safe (e.g. inherit shared resources that may
            cause contention), and using "spawn" can resolve that at the price
            of the process creation (and only the creation) speed.

    Raises:
        ValueError: If the action or observation specs don't match.

    """

    def __init__(
            self,
            env_constructors,
            start_serially=True,
            blocking=False,  # unused
            flatten=True,  # unused
            num_spare_envs_for_reload=0,
            torch_num_threads_per_env=1,
            start_method: str = "fork"):
        super().__init__()
        num_envs = len(env_constructors) - num_spare_envs_for_reload
        name = f"alf_penv_{os.getpid()}_{time.time()}"
        self._envs = []
        self._spare_envs = []
        self._start_method = start_method
        for env_id, ctor in enumerate(env_constructors):
            env = ProcessEnvironment(
                ctor,
                env_id=env_id,
                fast=True,
                num_envs=num_envs,
                torch_num_threads_per_env=torch_num_threads_per_env,
                start_method=start_method,
                name=name)
            if env_id < num_envs:
                self._envs.append(env)
            else:
                self._spare_envs.append(env)
        self._num_envs = len(env_constructors)
        self._num_spare_envs_for_reload = num_spare_envs_for_reload
        self._start_serially = start_serially
        self.start()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._reward_spec = self._envs[0].reward_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._env_info_spec = self._envs[0].env_info_spec()
        self._num_tasks = self._envs[0].num_tasks
        self._task_names = self._envs[0].task_names
        self._batch_size = self._envs[0].batch_size * num_envs
        time_step_with_env_info_spec = self._time_step_spec._replace(
            env_info=self._env_info_spec)
        batch_size_per_env = self._envs[0].batch_size
        batched = self._envs[0].batched
        if any(env.is_tensor_based for env in self._envs):
            raise ValueError(
                'All environments must be array-based environments.')
        if any(env.action_spec() != self._action_spec for env in self._envs):
            raise ValueError(
                'All environments must have the same action spec.')
        if any(env.time_step_spec() != self._time_step_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same time_step_spec.')
        if any(env.env_info_spec() != self._env_info_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same env_info_spec.')
        if any(env.batch_size != batch_size_per_env for env in self._envs):
            raise ValueError('All environments must have the same batch_size.')
        if any(env.batched != batched for env in self._envs):
            raise ValueError('All environments must have the same batched.')
        self._closed = False
        self._penv = _penv.ParallelEnvironment(
            num_envs, num_spare_envs_for_reload, batch_size_per_env, batched,
            self._action_spec, time_step_with_env_info_spec, name)

    @property
    def envs(self):
        """The list of individual environment."""
        return self._envs

    @property
    def num_spare_envs_for_reload(self):
        return self._num_spare_envs_for_reload

    def start(self):
        acting_text = {
            "fork": "Forking",
            "spawn": "Spawning"
        }[self._start_method]
        logging.info(f"{acting_text} all {len(self._envs)} processes.")
        for env in self._envs:
            env.start(wait_to_start=self._start_serially)
        for env in self._spare_envs:
            env.start(wait_to_start=self._start_serially)
        if not self._start_serially:
            logging.info('Waiting for all processes to start.')
            for env in self._envs:
                env.wait_start()
            for env in self._spare_envs:
                env.wait_start()
        logging.info('All processes started.')

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def task_names(self):
        return self._task_names

    def call_envs_with_same_args(self, func_name, *args, **kwargs):
        """Call each environment's named function with the same args.

        Args:
            func_name (str): name of the function to call
            *args: args to pass to the function
            **kwargs: kwargs to pass to the function

        return:
            list: list of results from each environment
        """
        return [env.call(func_name, *args, **kwargs)() for env in self._envs]

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def time_step_spec(self):
        return self._time_step_spec

    def render(self, mode="rgb_array"):
        return self._envs[0].render(mode)

    @property
    def metadata(self):
        return self._envs[0].metadata

    def _to_tensor(self, stacked):
        # we need to do np.copy because the result from _penv.step() or
        # _penv.reset() reuses the same internal buffer.
        stacked = nest.map_structure(
            lambda x: torch.as_tensor(np.copy(x), device='cpu'), stacked)
        if get_default_device() == "cuda":
            cpu = stacked
            stacked = nest.map_structure(lambda x: x.cuda(), cpu)
            stacked._cpu = cpu
        return stacked

    def _step(self, action):
        def _to_numpy(x):
            # When AMP is enabled, the action.dtype can be torch.float16. We
            # need to convert it to torch.float32 to match the dtype from
            # action_spec
            if x.dtype == torch.float16:
                x = x.float()
            x = x.cpu().numpy()
            # parallel_environment.cpp requires the arrays to be contiguous. If
            # x is already contiguous, ascontiguousarray() will simply return x.
            x = np.ascontiguousarray(x)
            return x

        action = nest.map_structure(_to_numpy, action)
        return self._to_tensor(self._penv.step(action))

    def _reset(self):
        return self._to_tensor(self._penv.reset())

    def sync_progress(self):
        """Sync the progress of all environments.

        ALF schedulers need to have progress information to calculate scheduled
        values. For parallel environments, the environments are in different
        processes and the progress information needs to be synced with the main
        in order to use schedulers in the environment.
        """
        [env.sync_progress() for env in self._envs]

    def close(self):
        """Close all external process."""
        if self._closed:
            return
        logging.info('Closing all processes.')
        i = 0
        for env in self._envs:
            env.close()
            i += 1
            if i % 100 == 0:
                logging.info(f"Closed {i} processes")
        for env in self._spare_envs:
            env.close()
            i += 1
            if i % 100 == 0:
                logging.info(f"Closed {i} processes")
        self._closed = True

    def seed(self, seeds):
        """Seeds the parallel environments."""
        envs = self._envs + self._spare_envs
        if len(seeds) != len(envs):
            raise ValueError(
                'Number of seeds should match the number of parallel_envs.')
        promises = [env.call('seed', seed) for seed, env in zip(seeds, envs)]
        # Block until all envs are seeded.
        return [promise() for promise in promises]

def _get_wrapped_fn(fn):
    """Get the function that is wrapped by ``functools.partial``"""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn

def _env_constructor(env_load_fn, env_name, batch_size_per_env, seed, env_id):
    # We need to set random seed before env_load_fn because some environment
    # perform use random numbers in its constructor, so we need to randomize
    # the seed for it.
    set_random_seed(seed)

    # In this case, the environment loader is already batched. Just use it to
    # create an environment with the specified batch size.
    #
    # NOTE: here it ASSUMES that the created batched environment will take the
    # following env IDs: env_id, env_id + 1, ... ,env_id + batch_size - 1
    batched = getattr(_get_wrapped_fn(env_load_fn), 'batched', False)
    if batched:
        return env_load_fn(
            env_name, env_id=env_id, batch_size=batch_size_per_env)
    if batch_size_per_env == 1:
        return env_load_fn(env_name, env_id)
    envs = [
        env_load_fn(env_name, env_id * batch_size_per_env + i)
        for i in range(batch_size_per_env)
    ]
    return BatchEnvironmentWrapper(envs)


@configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=load,
                       eval_env_load_fn=None,
                       for_evaluation=False,
                       num_parallel_environments=30,
                       batch_size_per_env=None,
                       eval_batch_size_per_env=None,
                       nonparallel=False,
                       flatten=True,
                       start_serially=True,
                       num_spare_envs=0,
                       torch_num_threads_per_env=1,
                       parallel_environment_ctor= FastParallelEnvironment,
                       seed=None,
                       batched_wrappers=()):
    """Create a batched environment.

    Args:
        env_name (str|list[str]): env name. If it is a list, ``MultitaskWrapper``
            will be used to create multi-task environments. Each one of them
            consists of the environments listed in ``env_name``.
        env_load_fn (Callable) : callable that create an environment
            If env_load_fn has attribute ``batched`` and it is True,
            ``evn_load_fn(env_name, env_id=env_id, batch_size=batch_size_per_env)``
            will be used to create the batched environment. Otherwise,
            ``env_load_fn(env_name, env_id)`` will be used to create the environment.
            env_id is the index of the environment in the batch in the range of
            ``[0, num_parallel_enviroments / batch_size_per_env)``.
            And if "num_parallel_environments" is in the signature of ``env_load_fn``,
            num_parallel_environments will be provided as a keyword argument.
        eval_env_load_fn (Callable) : callable that create an environment for
            evaluation. If None, use ``env_load_fn``. This argument is useful
            for cases when the evaluation environment is different from the
            training environment.
        for_evaluation (bool): whether to create an environment for evaluation
            (if True) or for training (if False). If True, ``eval_env_load_fn``
            will be used for creating the environment if provided. Otherwise,
            ``env_load_fn`` will be used.
        num_parallel_environments (int): num of parallel environments
        batch_size_per_env (Optional[int]): if >1, will create
            ``num_parallel_environments/batch_size_per_env``
            ``ProcessEnvironment``. Each of these ``ProcessEnvironment`` holds
            ``batch_size_per_env`` environments. If each underlying environment
            of ``ProcessEnvironment`` is itself batched, ``batch_size_per_env``
            will be used as the batch size for them. Otherwise
            ``BatchEnvironmentWrapper`` will be sused to instruct each process
            to run the underlying environments sequentially on operations such
            as ``step()``. The potential benefit of using
            ``batch_size_per_env>1`` is to reduce the number of processes being
            used, or to take advantages of the batched nature of the underlying
            environment.
            If None, it will be `num_parallel_envrironments` if ``env_load_fn``
                is batched and 1 otherwise.
        eval_batch_size_per_env (int): if provided, it will be used as the
            batch size for evaluation environment. Otherwise, use
            ``batch_size_per_env``.
        num_spare_envs (int): num of spare parallel envs for speed up reset.
        nonparallel (bool): force to create a single env in the current
            process. Used for correctly exposing game gin confs to tensorboard.
            If True, ``num_parallel_environments`` will be asserted to be 1, and
            ``batch_size_per_env`` has to be None or 1, to avoid potential mistakes
            in run configuration.
        start_serially (bool): start environments serially or in parallel.
        flatten (bool): whether to use flatten action and time_steps during
            communication to reduce overhead.
        num_spare_envs (int): number of spare parallel environments to speed
            up reset.  Useful when a reset is much slower than a regular step.
        torch_num_threads_per_env (int): how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best
            to set this number as 1. Leave this as 'None' to skip the change.
            Only used if the env is not batched and ``nonparallel==False``.
        parallel_environment_ctor (Callable): used to construct parallel environment.
            Available constructors are: ``fast_parallel_environment.FastParallelEnvironment``
            and ``parallel_environment.ParallelAlfEnvironment``.
        seed (None|int): random number seed for environment.  A random seed is
            used if None.
        batched_wrappers (Iterable): a list of wrappers which can wrap batched
            AlfEnvironment.
    Returns:
        AlfEnvironment:

    """

    # Some environment may take long time to load. So we use GPU before loading
    # environments so that other people knows that this GPU is being used.
    if torch.cuda.is_available():
        tmp = torch.zeros((32, ))
        torch.cuda.synchronize()

    if for_evaluation:
        # for creating an evaluation environment, use ``eval_env_load_fn`` if
        # provided and fall back to ``env_load_fn`` otherwise
        env_load_fn = eval_env_load_fn or env_load_fn
        batch_size_per_env = eval_batch_size_per_env or batch_size_per_env

    # env_load_fn may be a functools.partial, so we need to get the wrapped
    # function to get its attributes
    batched = getattr(_get_wrapped_fn(env_load_fn), 'batched', False)
    no_thread_env = getattr(
        _get_wrapped_fn(env_load_fn), 'no_thread_env', False)

    if nonparallel:
        assert num_parallel_environments == 1, "nonparallel is True"
        if batch_size_per_env is not None:
            assert batch_size_per_env == 1, "nonparallel is True"

    if batch_size_per_env is None:
        if batched:
            batch_size_per_env = num_parallel_environments
        else:
            batch_size_per_env = 1

    assert num_parallel_environments % batch_size_per_env == 0, (
        f"num_parallel_environments ({num_parallel_environments}) cannot be"
        f"divided by batch_size_per_env ({batch_size_per_env})")
    num_envs = num_parallel_environments // batch_size_per_env
    if batch_size_per_env > 1:
        assert num_spare_envs == 0, "Do not support spare environments for batch_size_per_env > 1"
        assert parallel_environment_ctor == FastParallelEnvironment

    if 'num_parallel_environments' in inspect.signature(
            env_load_fn).parameters:
        env_load_fn = functools.partial(
            env_load_fn, num_parallel_environments=num_parallel_environments)

    if isinstance(env_name, (list, tuple)):
        env_load_fn = functools.partial(MultitaskWrapper.load,
                                        env_load_fn)

    if batched and batch_size_per_env == num_parallel_environments:
        alf_env = env_load_fn(env_name, batch_size=num_parallel_environments)
        if not alf_env.is_tensor_based:
            alf_env = TensorWrapper(alf_env)
    elif nonparallel:
        # Each time we can only create one unwrapped env at most
        if no_thread_env:
            # In this case the environment is marked as "not compatible with
            # thread environment", and we will create it in the main thread.
            # BatchedTensorWrapper is applied to make sure the I/O is batched
            # torch tensor based.
            alf_env = BatchedTensorWrapper(env_load_fn(env_name))
        else:
            # Create and step the env in a separate thread. env `step` and
            #   `reset` must run in the same thread which the env is created in
            #   for some simulation environments such as social_bot(gazebo)
            alf_env = ThreadEnvironment(lambda: env_load_fn(
                env_name))

        if seed is None:
            alf_env.seed(np.random.randint(0, np.iinfo(np.int32).max))
        else:
            alf_env.seed(seed)
    else:
        if seed is None:
            seeds = list(
                map(
                    int,
                    np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      num_envs + num_spare_envs)))
        else:
            seeds = [seed + i for i in range(num_envs + num_spare_envs)]
        ctors = [
            functools.partial(_env_constructor, env_load_fn, env_name,
                              batch_size_per_env, seed) for seed in seeds
        ]
        # flatten=True will use flattened action and time_step in
        #   process environments to reduce communication overhead.
        alf_env = parallel_environment_ctor(
            ctors,
            flatten=flatten,
            start_serially=start_serially,
            num_spare_envs_for_reload=num_spare_envs,
            torch_num_threads_per_env=torch_num_threads_per_env)
        alf_env.seed(seeds)

    for wrapper in batched_wrappers:
        alf_env = wrapper(alf_env)

    return alf_env

def get_env():
    """Get the global training environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Note: random seed will be initialized in this function.

    Returns:
        AlfEnvironment
    """
    global _env
    if _env is None:
        # When ``get_env()`` is called in a spawned process (this is almost
        # always due to a ``ProcessEnvironment`` created with "spawn" method),
        # use the environment constructor from the context to create the
        # environment. This is to avoid creating a parallel environment which
        # leads to infinite recursion.
        ctx = get_spawned_process_context()
        if isinstance(ctx, SpawnedProcessContext):
            _env = ctx.create_env()
            return _env

        if _is_parsing:
            random_seed = get_config_value('TrainerConfig.random_seed')
        else:
            # We construct a TrainerConfig object here so that the value
            # configured through gin-config can be properly retrieved.
            train_config = TrainerConfig(root_dir='')
            random_seed = train_config.random_seed

        # Override the random seed based on ddp rank. With the following logic:
        #
        # - Rank 0 will get the original random seed specified in the config
        # - Rank 1 will get the 1st pseudo random integer as seed (using original seed)
        # - Rank 2 will get the 2nd pseudo random integer as seed (using original seed)
        # - ...
        #
        # The random.seed(random_seed) is temporary and will be overridden by
        # set_random_seed() below.
        random.seed(random_seed)

        if random_seed is not None:
            # If random seed is None, we will have None for other ranks, too.
            # A 'None' random seed won't set a deterministic torch behavior.
            for _ in range(PerProcessContext().ddp_rank):
                random_seed = random.randint(0, 2**32)
        config1(
            "TrainerConfig.random_seed",
            random_seed,
            raise_if_used=False,
            override_sole_init=True)

        # We have to call set_random_seed() here because we need the actual
        # random seed to call create_environment.
        seed = set_random_seed(random_seed)
        # In case when running in multi-process mode, the number of environments
        # per process need to be adjusted (divided by number of processes).
        adjust_config_by_multi_process_divider(
            PerProcessContext().ddp_rank,
            PerProcessContext().num_processes)
        _env = create_environment(seed=seed)
    return _env

def parse_conf_file(conf_file):
    """Parse config from file.
    It also looks for FLAGS.gin_param and FLAGS.conf_param for extra configs.
    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().
    Args:
        conf_file (str): the full path to the config file
    """
    if conf_file.endswith(".gin"):
        gin_params = getattr(flags.FLAGS, 'gin_param', None)
        gin.parse_config_files_and_bindings([conf_file], gin_params)
        ml_type = get_config_value('TrainerConfig.ml_type')
        if ml_type == 'rl':
            # Create the global environment and initialize random seed
            get_env()
    else:
        conf_params = getattr(flags.FLAGS, 'conf_param', None)
        parse_config(conf_file, conf_params)

def get_operative_configs():
    """Get all the configs that have been used.

    A config is operative if a function call does not explicitly specify the value
    of that config and hence its default value or the value provided through
    alf.config() needs to be used.

    Returns:
        list[tuple[config_name, Any]]
    """
    configs = [(name, config.get_effective_value())
               for name, config in _get_all_leaves(_CONF_TREE)
               if config.is_used()]
    return sorted(configs, key=lambda x: x[0])

def summarize_config():
    """Write config to TensorBoard."""

    def _format(configs):
        paragraph = pprint.pformat(dict(configs))
        return "    ".join((os.linesep + paragraph).splitlines(keepends=True))

    conf_file = get_conf_file()
    if conf_file is None or conf_file.endswith('.gin'):
        return summarize_gin_config()

    operative_configs = get_operative_configs()
    inoperative_configs = get_inoperative_configs()
    text('config/operative_config', _format(operative_configs))
    if inoperative_configs:
        text('gin/inoperative_config',
                         _format(inoperative_configs))


def write_config(root_dir):
    """Write config to a file under directory ``root_dir``
    Configs from FLAGS.conf_param are also recorded.
    Args:
        root_dir (str): directory path
    """
    conf_file = get_conf_file()
    if conf_file is None or conf_file.endswith('.gin'):
        return write_gin_configs(root_dir, 'configured.gin')

    root_dir = os.path.expanduser(root_dir)
    alf_config_file = os.path.join(root_dir, ALF_CONFIG_FILE)
    os.makedirs(root_dir, exist_ok=True)
    conf_params = getattr(flags.FLAGS, 'conf_param', None)
    config = ''
    if conf_params:
        config += "########### config from commandline ###########\n\n"
        config += "import alf\n"
        config += "alf.pre_config({\n"
        for conf_param in conf_params:
            pos = conf_param.find('=')
            if pos == -1:
                raise ValueError("conf_param should have a format of "
                                 "'CONFIG_NAME=VALUE': %s" % conf_param)
            config_name = conf_param[:pos]
            config_value = conf_param[pos + 1:]
            config += "    '%s': %s,\n" % (config_name, config_value)
        config += "})\n\n"
        config += "########### end config from commandline ###########\n\n"
    f = open(conf_file, 'r')
    config += f.read()
    f.close()
    f = open(alf_config_file, 'w')
    f.write(config)
    f.close()


def get_initial_policy_state(batch_size, policy_state_spec):
    """
    Return zero tensors as the initial policy states.
    Args:
        batch_size (int): number of policy states created
        policy_state_spec (nested structure): each item is a tensor spec for
            a state
    Returns:
        state (nested structure): each item is a tensor with the first dim equal
            to ``batch_size``. The remaining dims are consistent with
            the corresponding state spec of ``policy_state_spec``.
    """
    return zero_tensor_from_nested_spec(policy_state_spec, batch_size)


def get_initial_time_step(env, first_env_id=0):
    """Return the initial time step.
    Args:
        env (AlfEnvironment):
        first_env_id (int): the environment ID for the first sample in this
            batch.
    Returns:
        TimeStep: the init time step with actions as zero tensors.
    """
    time_step = env.current_time_step()
    return time_step._replace(env_id=time_step.env_id + first_env_id)


_env = None


def set_global_env(env):
    """Set global env."""
    global _env
    _env = env


@gin.configurable
def get_raw_observation_spec(field=None):
    """Get the ``TensorSpec`` of observations provided by the global environment.
    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    specs = _env.observation_spec()
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


_transformed_observation_spec = None


def set_transformed_observation_spec(spec):
    """Set the spec of the observation transformed by data transformers."""
    global _transformed_observation_spec
    _transformed_observation_spec = spec


@gin.configurable
def get_observation_spec(field=None):
    """Get the spec of observation transformed by data transformers.
    The data transformers are specified by ``TrainerConfig.data_transformer_ctor``.
    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    assert _transformed_observation_spec is not None, (
        "This function should be "
        "called after the global variable _transformed_observation_spec is set"
    )

    specs = _transformed_observation_spec
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


@gin.configurable
def get_states_shape():
    """Get the tensor shape of internal states of the agent provided by
      the global environment.
      Returns:
        0 if internal states is not part of observation; otherwise a
        ``torch.Size``. We don't raise error so this code can serve to check
        whether ``env`` has states input.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    if isinstance(_env.observation_spec(),
                  dict) and ('states' in _env.observation_spec()):
        return _env.observation_spec()['states'].shape
    else:
        return 0


@gin.configurable
def get_action_spec():
    """Get the specs of the tensors expected by ``step(action)`` of the global
    environment.
    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each tensor
        expected by ``step()``.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.action_spec()


@gin.configurable
def get_reward_spec():
    """Get the specs of the reward tensors of the global environment.
    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each reward
        tensor.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.reward_spec()

@gin.configurable
def get_done_spec():
    """Get the specs of the done tensors of the global environment.
    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each done
        tensor.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.done_spec()

def get_env():
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env


@gin.configurable
def get_vocab_size():
    """Get the vocabulary size of observations provided by the global environment.
    Returns:
        int: size of the environment's/teacher's vocabulary. Returns 0 if
        language is not part of observation. We don't raise error so this code
        can serve to check whether the env has language input
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    if isinstance(_env.observation_spec(),
                  dict) and ('sentence' in _env.observation_spec()):
        # return _env.observation_spec()['sentence'].shape[0]
        # is the sequence length of the sentence.
        return _env.observation_spec()['sentence'].maximum + 1
    else:
        return 0


@gin.configurable
def active_action_target_entropy(active_action_portion=0.2, min_entropy=0.3):
    """Automatically compute target entropy given the action spec. Currently
    support discrete actions only.
    The general idea is that we assume :math:`Nk` actions having uniform probs
    for a good policy. Thus the target entropy should be :math:`log(Nk)`, where
    :math:`N` is the total number of discrete actions and k is the active action
    portion.
    TODO: incorporate this function into ``EntropyTargetAlgorithm`` if it proves
    to be effective.
    Args:
        active_action_portion (float): a number in :math:`(0, 1]`. Ideally, this
            value should be greater than ``1/num_actions``. If it's not, it will
            be ignored.
        min_entropy (float): the minimum possible entropy. If the auto-computed
            entropy is smaller than this value, then it will be replaced.
    Returns:
        float: the target entropy for ``EntropyTargetAlgorithm``.
    """
    assert active_action_portion <= 1.0 and active_action_portion > 0
    action_spec = get_action_spec()
    assert action_spec.is_discrete(
        action_spec), "only support discrete actions!"
    num_actions = action_spec.maximum - action_spec.minimum + 1
    return max(math.log(num_actions * active_action_portion), min_entropy)


def write_gin_configs(root_dir, gin_file):
    """
    Write a gin configration to a file. Because the user can
    1) manually change the gin confs after loading a conf file into the code, or
    2) include a gin file in another gin file while only the latter might be
       copied to ``root_dir``.
    So here we just dump the actual used gin conf string to a file.
    Args:
        root_dir (str): directory path
        gin_file (str): a single file path for storing the gin configs. Only
            the basename of the path will be used.
    """
    root_dir = os.path.expanduser(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    file = os.path.join(root_dir, os.path.basename(gin_file))

    md_operative_config_str, md_inoperative_config_str = get_gin_confg_strs()
    config_str = md_operative_config_str + '\n\n' + md_inoperative_config_str

    # the mark-down string can just be safely written as a python file
    with open(file, "w") as f:
        f.write(config_str)


@logging.skip_log_prefix
def warning_once(msg, *args):
    """Generate warning message once.
    Note that the current implementation resembles that of the ``log_every_n()```
    function in ``logging`` but reduces the calling stack by one to ensure
    the multiple warning once messages generated at difference places can be
    displayed correctly.
    Args:
        msg: str, the message to be logged.
        *args: The args to be substitued into the msg.
    """
    caller = logging.get_absl_logger().findCaller()
    count = logging._get_next_log_count_per_token(caller)
    logging.log_if(logging.WARNING, msg, count == 0, *args)


def set_random_seed(seed):
    """Set a seed for deterministic behaviors.
    Note: If someone runs an experiment with a pre-selected manual seed, he can
    definitely reproduce the results with the same seed; however, if he runs the
    experiment with seed=None and re-run the experiments using the seed previously
    returned from this function (e.g. the returned seed might be logged to
    Tensorboard), and if cudnn is used in the code, then there is no guarantee
    that the results will be reproduced with the recovered seed.
    Args:
        seed (int|None): seed to be used. If None, a default seed based on
            pid and time will be used.
    Returns:
        The seed being used if ``seed`` is None.
    """
    if seed is None:
        seed = int(np.uint32(hash(str(os.getpid()) + '|' + str(time.time()))))
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def log_metrics(metrics, prefix=''):
    """Log metrics through logging.
    Args:
        metrics (list[alf.metrics.StepMetric]): list of metrics to be logged
        prefix (str): prefix to the log segment
    """
    log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
    logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))

class OUProcess(nn.Module):
    """A zero-mean Ornstein-Uhlenbeck process for generating noises."""

    def __init__(self, initial_value, damping=0.15, stddev=0.2):
        """
        The Ornstein-Uhlenbeck process is a process that generates temporally
        correlated noise via a random walk with damping. This process describes
        the velocity of a particle undergoing brownian motion in the presence of
        friction. This can be useful for exploration in continuous action
        environments with momentum.
        The temporal update equation is:
        .. code-block:: python
            x_next = (1 - damping) * x + N(0, std_dev)
        Args:
            initial_value (Tensor): Initial value of the process.
            damping (float): The rate at which the noise trajectory is damped
                towards the mean. We must have :math:`0 <= damping <= 1`, where
                a value of 0 gives an undamped random walk and a value of 1 gives
                uncorrelated Gaussian noise. Hence in most applications a small
                non-zero value is appropriate.
            stddev (float): Standard deviation of the Gaussian component.
        """
        super(OUProcess, self).__init__()
        self._damping = damping
        self._stddev = stddev
        self._x = initial_value.clone().detach()

    def forward(self):
        noise = torch.randn_like(self._x) * self._stddev
        return self._x.data.copy_((1 - self._damping) * self._x + noise)

def create_ou_process(action_spec, ou_stddev, ou_damping):
    """Create nested zero-mean Ornstein-Uhlenbeck processes.
    The temporal update equation is:
    .. code-block:: python
        x_next = (1 - damping) * x + N(0, std_dev)
    Note: if ``action_spec`` is nested, the returned nested OUProcess will not bec
    checkpointed.
    Args:
        action_spec (nested BountedTensorSpec): action spec
        ou_damping (float): Damping rate in the above equation. We must have
            :math:`0 <= damping <= 1`.
        ou_stddev (float): Standard deviation of the Gaussian component.
    Returns:
        nested ``OUProcess`` with the same structure as ``action_spec``.
    """

    def _create_ou_process(action_spec):
        return OUProcess(action_spec.zeros(), ou_damping, ou_stddev)

    ou_process = alf.nest.map_structure(_create_ou_process, action_spec)
    return ou_process


def detach(nests):
    """Detach nested Tensors.
    Args:
        nests (nested Tensor): tensors to be detached
    Returns:
        detached Tensors with same structure as nests
    """
    return nest.map_structure(lambda t: t.detach(), nests)


# A catch all mode.  Currently includes on-policy training on unrolled experience.
EXE_MODE_OTHER = 0
# Unroll during training
EXE_MODE_ROLLOUT = 1
# Replay, policy evaluation on experience and training
EXE_MODE_REPLAY = 2
# Evaluation / testing or playing a learned model
EXE_MODE_EVAL = 3

# Global execution mode to track where the program is in the RL training process.
# This is used currently for observation normalization to only update statistics
# during training (vs unroll).  This is also used in tensorboard plotting of
# network output values, evaluation of the same network during rollout vs eval vs
# replay will be plotted to different graphs.
_exe_mode = EXE_MODE_OTHER
_exe_mode_strs = ["other", "rollout", "replay", "eval"]


def set_exe_mode(mode):
    """Mark whether the current code belongs to unrolling or training. This flag
    might be used to change the behavior of some functions accordingly.
    Args:
        training (bool): True for training, False for unrolling
    """
    global _exe_mode
    _exe_mode = mode


def exe_mode_name():
    """return the execution mode as string.
    """
    return _exe_mode_strs[_exe_mode]


def is_replay():
    """Return a bool value indicating whether the current code belongs to
    unrolling or training.
    """
    return _exe_mode == EXE_MODE_REPLAY


def is_rollout():
    """Return a bool value indicating whether the current code belongs to
    unrolling or training.
    """
    return _exe_mode == EXE_MODE_ROLLOUT


def is_eval():
    """Return a bool value indicating whether the current code belongs to
    evaluation or playing a learned model.
    """
    return _exe_mode == EXE_MODE_EVAL


def mark_eval(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a evaluation/test function.
    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_EVAL)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


def mark_replay(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a experience replay function.
    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_REPLAY)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


def mark_rollout(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a rollout function.
    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_ROLLOUT)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


@gin.configurable
def flattened_size(spec):
    """Return the size of the vector if spec.shape is flattened.
    It's same as np.prod(spec.shape)
    Args:
        spec (alf.TensorSpec): a TensorSpec object
    Returns:
        np.int64: the size of flattened shape
    """
    # np.prod(()) == 1.0, need to convert to np.int64
    return np.int64(np.prod(spec.shape))


def is_inside_docker_container():
    """Return whether the current process is running inside a docker container.
    See discussions at `<https://stackoverflow.com/questions/23513045/how-to-check-if-a-process-is-running-inside-docker-container>`_
    """
    return os.path.exists("/.dockerenv")


def check_numerics(nested):
    """Assert all the tensors in nested are finite.
    Args:
        nested (nested Tensor): nested Tensor to be checked.
    """
    nested_finite = nest.map_structure(
        lambda x: torch.all(torch.isfinite(x)), nested)
    if not all(nest.flatten(nested_finite)):
        bad = nest.map_structure(lambda x, finite: () if finite else x,
                                     nested, nested_finite)
        assert all(nest.flatten(nested_finite)), (
            "Some tensor in nested is not finite: %s" % bad)


def get_all_parameters(obj):
    """Get all the parameters under ``obj`` and its descendents.
    Note: This function assumes all the parameters can be reached through tuple,
    list, dict, set, nn.Module or the attributes of an object. If a parameter is
    held in a strange way, it will not be included by this function.
    Args:
        obj (object): will look for paramters under this object.
    Returns:
        list: list of (path, Parameters)
    """
    all_parameters = []
    memo = set()
    unprocessed = [(obj, '')]
    # BFS for all subobjects
    while unprocessed:
        obj, path = unprocessed.pop(0)
        if isinstance(obj, types.ModuleType):
            # Do not traverse into a module. There are too much stuff inside a
            # module.
            continue
        if isinstance(obj, nn.Parameter):
            all_parameters.append((path, obj))
            continue
        if isinstance(obj, torch.Tensor):
            continue
        if path:
            path += '.'
        if nest.is_namedtuple(obj):
            for name, value in nest.extract_fields_from_nest(obj):
                if id(value) not in memo:
                    unprocessed.append((value, path + str(name)))
                    memo.add(id(value))
        elif isinstance(obj, dict):
            # The keys of a generic dict are not necessarily str, and cannot be
            # handled by nest.extract_fields_from_nest.
            for name, value, in obj.items():
                if id(value) not in memo:
                    unprocessed.append((value, path + str(name)))
                    memo.add(id(value))
        elif isinstance(obj, (tuple, list, set)):
            for i, value in enumerate(obj):
                if id(value) not in memo:
                    unprocessed.append((value, path + str(i)))
                    memo.add(id(value))
        elif isinstance(obj, nn.Module):
            for name, m in obj.named_children():
                if id(m) not in memo:
                    unprocessed.append((m, path + name))
                    memo.add(id(m))
            for name, p in obj.named_parameters():
                if id(p) not in memo:
                    unprocessed.append((p, path + name))
                    memo.add(id(p))
        attribute_names = dir(obj)
        for name in attribute_names:
            if name.startswith('__') and name.endswith('__'):
                # Ignore system attributes,
                continue
            attr = None
            try:
                attr = getattr(obj, name)
            except:
                # some attrbutes are property function, which may raise exception
                # when called in a wrong context (e.g. Algorithm.experience_spec)
                pass
            if attr is None or id(attr) in memo:
                continue
            unprocessed.append((attr, path + name))
            memo.add(id(attr))
    return all_parameters


def generate_alf_root_snapshot(alf_root, dest_path):
    """Given a destination path, copy the local ALF root dir to the path. To
    save disk space, only ``*.py`` files will be copied.
    This function can be used to generate a snapshot of the repo so that the
    exactly same code status will be recovered when later playing a trained
    model or launching a grid-search job in the waiting queue.
    Args:
        alf_root (str): the path to the ALF repo
        dest_path (str): the path to generate a snapshot of ALF repo
    """

    def _is_subdir(path, directory):
        relative = os.path.relpath(path, directory)
        return not relative.startswith(os.pardir)

    def rsync(src, target, includes):
        args = ['rsync', '-rI', '--include=*/']
        args += ['--include=%s' % i for i in includes]
        args += ['--exclude=*']
        args += [src, target]
        # shell=True preserves string arguments
        subprocess.check_call(
            " ".join(args), stdout=sys.stdout, stderr=sys.stdout, shell=True)

    assert not _is_subdir(dest_path, alf_root), (
        "Snapshot path '%s' is not allowed under ALF root! Use a different one!"
        % dest_path)

    # these files are important for code status
    includes = ["*.py", "*.gin", "*.so", "*.json"]
    rsync(alf_root, dest_path, includes)

    # rename ALF repo to a unified dir name 'alf'
    alf_dirname = os.path.basename(alf_root)
    if alf_dirname != "alf":
        os.system("mv %s/%s %s/alf" % (dest_path, alf_dirname, dest_path))


def get_alf_snapshot_env_vars(root_dir):
    """Given a ``root_dir``, return modified env variable dict so that ``PYTHONPATH``
    points to the ALF snapshot under this directory.
    """
    alf_repo = os.path.join(root_dir, "alf")
    alf_cnest = os.path.join(alf_repo,
                             "alf/nest/cnest")  # path to archived cnest.so
    alf_examples = os.path.join(alf_repo, "alf/examples")
    python_path = os.environ.get("PYTHONPATH", "")
    python_path = ":".join([alf_repo, alf_cnest, alf_examples, python_path])
    env_vars = copy.copy(os.environ)
    env_vars.update({"PYTHONPATH": python_path})
    return env_vars


def abs_path(path):
    """Given any path, return the absolute path with expanding the user.
    """
    return os.path.realpath(os.path.expanduser(path))
