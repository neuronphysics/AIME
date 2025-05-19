from abc import ABC
import cloudpickle
import atexit
import traceback
import io
import uuid
import datetime
import itertools
from tqdm import tqdm
import imageio
import pathlib
import random
import functools
from numbers import Number
from collections import namedtuple, deque, defaultdict
from typing import Tuple, Dict, List, Iterable
import gym
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, IterableDataset
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import torch.utils.data.distributed
from agac_torch.agac.configs import AlgorithmConfig,LoggingConfig, ReinforcementLearningConfig
from VRNN.Normalization import compute_normalizer
import numpy as np
import json
import time
from gym_wrappers import wrap_env, FrameSkip
import sys
from pathlib import Path
import operator
import re
import math
from agac_torch.agac.agac_ppo import *

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now


def is_named_tuple_instance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n) == str for n in f)


def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def eplen(episode):
    return len(episode["done"]) - 1


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None, minlen=1):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob("*.npz"))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split("-")[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        episodes[str(filename)] = episode
    return episodes


def convert(value):
    if value[0] is None:
        value = value[1:]
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    px_format = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {px_format} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = step.to(torch.float32)
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.unbind(outputs, dim=0)
    outputs = torch.stack(outputs, dim=1)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)

    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def make_episodes(states, actions, next_states, dones):
    pairs, inputs, score, targets = [], [], [], []
    for i in range(len(dones)):
        if not dones[i]:
            pairs.append(torch.cat([states[i], actions[i]]))
            score.append(next_states[i])
        else:
            pairs.append(torch.cat([states[i], actions[i]]))
            score.append(next_states[i])
            inputs.append(pairs)
            pairs = []
            targets.append(score)
            score = []
    if len(pairs) != 0:
        inputs.append(pairs)
        targets.append(score)
    return inputs, targets


def pad(episodes, repeat=True):
    """Pads episodes to all be the same length by repeating the last exp.
    Args:
        episodes (list[list[Experience]]): episodes to pad.
    Returns:
        padded_episodes (list[list[Experience]]): now of shape
            (batch_size, max_len)
        mask (torch.BoolTensor): of shape (batch_size, max_len) with value 0 for
            padded experiences.
    """
    max_len = max(len(episode) for episode in episodes)

    mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
    padded_episodes = []
    for i, episode in enumerate(episodes):
        if repeat:
            padded = episode + [episode[-1]] * (max_len - len(episode))
        else:
            padded = episode + [torch.zeros_like(episode[-1])] * (max_len - len(episode))
        padded_episodes.append(padded)
        mask[i, :len(episode)] = True
    return padded_episodes, mask


def build_dict_from_named_tuple(x):
    res = dict()
    for name in x[0]._fields:
        if isinstance(getattr(x[0], name), np.ndarray):
            res.update({name: np.stack([getattr(o, name) for o in x])})
        elif isinstance(getattr(x[0], name), np.bool_):
            res.update({name: np.stack([np.array(getattr(o, name), dtype=bool) for o in x])})

    return res


def build_dict_from_named_tuple_list(x):
    res = {}
    for name in x[0]._fields:
        if isinstance(getattr(x[0], name), np.ndarray):
            res[name] = np.stack([getattr(o, name) for o in x])
        elif isinstance(getattr(x[0], name), np.bool_):
            res[name] = np.stack([np.array(getattr(o, name), dtype=bool) for o in x])
        else:
            res[name] = [getattr(o, name) for o in x]
    return res


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(nn.Softplus()(x))
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(name)


def dict_to_tran(data):
    return Transition(
        s1=(data['s1']),
        s2=(data['s2']),
        a1=data['a1'],
        a2=data['a2'],
        reward=data['reward'],
        discount=data['discount'],
        done=data['done'])


def deprecated_preprocess_transition_data(s1, a1, s2, done):
    inputs, outputs = make_episodes(s1, a1, s2, done)
    padded_trajectories_inputs, mask_input = pad(inputs, repeat=False)
    padded_trajectories_outputs, mask_output = pad(outputs, repeat=False)
    final_inputs = torch.stack(
        [torch.stack([padded_trajectories_inputs[i][j] for j in range(mask_input.shape[1])], dim=0) for i in
         range(mask_input.shape[0])], dim=0)  # (batch_size , max_len, state_dim)
    final_outputs = torch.stack(
        [torch.stack([padded_trajectories_outputs[i][j] for j in range(mask_output.shape[1])], dim=0) for i in
         range(mask_output.shape[0])], dim=0)  # (batch_size , max_len, state_dim)
    # shape of outputs: (B , max_seq_len, D)
    return final_inputs, final_outputs


class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out


class Async:
    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.


class IterableWrapper(IterableDataset, ABC):
    def __init__(self, iterator):
        super().__init__()
        self.iterator = iterator

    def __iter__(self):
        return self.iterator


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


@functools.total_ordering
class NCounter:
    def __init__(self, initial=0):
        self.value = initial

    def __int__(self):
        return int(self.value)

    def __eq__(self, other):
        return int(self) == other

    def __ne__(self, other):
        return int(self) != other

    def __lt__(self, other):
        return int(self) < other

    def __add__(self, other):
        return int(self) + other

    def increment(self, amount=1):
        self.value += amount


class TerminalOutput:
    def __call__(self, summaries):
        def _format_value(value):
            if value == 0:
                return "0"
            elif 0.01 < abs(value) < 10000:
                value = f"{value:.2f}"
                value = value.rstrip("0")
                value = value.rstrip("0")
                value = value.rstrip(".")
                return value
            else:
                value = f"{value:.1e}"
                value = value.replace(".0e", "e")
                value = value.replace("+0", "")
                value = value.replace("+", "")
                value = value.replace("-0", "-")
            return value

        step = max(s for s, _, _, in summaries)
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        formatted = {k: _format_value(v) for k, v in scalars.items()}
        print(f"[{step}]", " / ".join(f"{k} {v}" for k, v in formatted.items()))


class JSONLOutput:
    def __init__(self, logdir):
        self._logdir = pathlib.Path(logdir).expanduser()

    def __call__(self, summaries):
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        step = max(s for s, _, _, in summaries)
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **scalars}) + "\n")


class TensorBoardOutput:
    def __init__(self, logdir, summary_writer, fps=20):
        self._logdir = os.path.expanduser(logdir)
        self._writer = summary_writer
        self._fps = fps

    def __call__(self, summaries):
        for step, name, value in summaries:
            if len(value.shape) == 0:
                self._writer.add_scalar(name, value, step)
            elif len(value.shape) == 2:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 3:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 4:
                self.video_summary(name, value, step)
        self._writer.flush()

    def video_summary(self, name, video, step):
        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        T, H, W, C = video.shape
        video = video.transpose(0, 3, 1, 2).reshape((1, T, C, H, W))

        self._writer.add_video(name, video, step, 4)

    def log_dict(self, info, prefix, step):
        res = []
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            temp = (step, prefix + "/" + key, value)
            res.append(temp)
        self(res)


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()

        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - nn.functional.softplus(-2.0 * x))


class TruncatedStandardNormal(Distribution, ABC):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
                                        self._little_phi_b * little_phi_coeff_b
                                        - self._little_phi_a * little_phi_coeff_a
                                ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
                1
                - self._lpbb_m_lpaa_d_Z
                - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal, ABC):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = (samples,)

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torch.distributions.OneHotCategorical, ABC):
    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample = sample + (probs - probs.detach())
        return sample

    def log_prob(self, value):
        int_tensor = value.long()
        # False is [1, 0], True is [0, 1]
        onehot_rep = torch.nn.functional.one_hot(int_tensor, num_classes=2)
        return super().log_prob(onehot_rep.squeeze(2))


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class StreamNorm(nn.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        super().__init__()
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = nn.Parameter(torch.ones(shape, dtype=torch.float64), requires_grad=False)

    def forward(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics["mean"] = inputs.mean().detach().cpu()
        metrics["std"] = inputs.std().detach().cpu()
        outputs = self.transform(inputs)
        metrics["normed_mean"] = outputs.mean().detach().cpu()
        metrics["normed_std"] = outputs.std().detach().cpu()
        return outputs, metrics

    def reset(self):
        self.mag.data = torch.ones_like(self.mag)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0).type(torch.float64)
        self.mag.data = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag.data.type(inputs.dtype)[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)


class VideoRecorder:
    def __init__(self, root_dir, render_size=64, fps=20):
        self.enabled = None
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class Logger:
    def __init__(self, outputs, ):
        self._step = 0
        self._outputs = outputs
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, prefix=None):
        step = self._step
        for name, value in dict(mapping).items():
            name = f"{prefix}_{name}" if prefix else name
            value = np.array(value)
            if len(value.shape) not in (0, 2, 3, 4):
                raise ValueError(
                    f"Shape {value.shape} for name '{name}' cannot be "
                    "interpreted as scalar, image, or video."
                )
            self._metrics.append((step, name, value))
        step += 1

    def scalar(self, name, value):
        self.add({name: value})

    def image(self, name, value):
        self.add({name: value})

    def video(self, name, value):
        self.add({name: value})

    def write(self, fps=False):
        # fps and self.scalar("fps", self._compute_fps())
        if not self._metrics:
            return
        for output in self._outputs:
            output(self._metrics)
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self._step) * self._multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class RequiresGrad:
    def __init__(self, parameters):
        # Expects an iterable of parameters, not a model
        self.parameters = parameters

    def __enter__(self):
        # Enable gradients for all parameters in the collection
        for param in self.parameters:
            param.requires_grad_(True)

    def __exit__(self, *args):
        # Disable gradients for all parameters in the collection
        for param in self.parameters:
            param.requires_grad_(False)


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        step = int(step)
        if not self._every:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        step = int(step)
        if not self._until:
            return True
        return step < self._until


class Scale(nn.Module):
    """
    Maps inputs from [space.low, space.high] range to [-1, 1] range.
    Parameters
    ----------
    space : gym.Space
        Space to map from.
    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """

    def __init__(self, space):
        super(Scale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [space.low, space.high] to [-1, 1].
        Parameters
        ----------
        x : torch.tensor
            Input to be scaled
        """
        return 2.0 * ((x - self.low) / (self.high - self.low)) - 1.0


class Unscale(nn.Module):
    """
    Maps inputs from [-1, 1] range to [space.low, space.high] range.
    Parameters
    ----------
    space : gym.Space
        Space to map from.
    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """

    def __init__(self, space):
        super(Unscale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [-1, 1] to [space.low, space.high].
        Parameters
        ----------
        x : torch.tensor
            Input to be unscaled
        """
        return self.low + (0.5 * (x + 1.0) * (self.high - self.low))


class VideoRenderWrapper(gym.Wrapper):
    """A wrapper to enable 'rgb_array' rendering for the gym wrapper of
    ``dm_control``. To do this, the ``metadata`` field needs to be updated.
    """
    _metadata = {'render.modes': ["rgb_array"]}

    def __init__(self, env):
        super().__init__(env)
        self.metadata.update(self._metadata)


def append_action_and_latent_obs(s1, a1, s2, n_step):
    latent_shape = s1.shape[-1]
    obs = torch.reshape(s1, (-1, n_step, latent_shape))
    next_obs = torch.reshape(s2, (-1, n_step, latent_shape))
    inputs = torch.cat((obs, a1), dim=-1)
    return inputs, next_obs

def create_agac_config(config, env_name, layers_dim, adv_layers_dim):
    """Create and return a properly configured ExperimentConfig for AGAC."""
    
    # Create ExperimentConfig for AGAC with all required parameters
    agac_config = ExperimentConfig(
        algorithm=AlgorithmConfig(
            max_steps=config.max_steps if hasattr(config, 'max_steps') else 1000000,
            eval_freq=config.eval_freq if hasattr(config, 'eval_freq') else 10000,
            num_episodes_eval=1,  # Number of episodes to evaluate
            train_freq=config.train_every,  # Same as main config
            num_epochs=5,  # Number of PPO epochs per update
            seed=config.seed,
            env_name=env_name,
            discrete=False,  # For continuous control tasks
            gpu=-1  # Default to CPU
        ),
        reinforcement_learning=ReinforcementLearningConfig(
            batch_size=config.batch_size,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            adversary_learning_rate=3e-4,  # Same as actor
            discount=config.discount if hasattr(config, 'discount') else 0.99,
            lambda_gae=0.95,  # Standard GAE lambda
            layers_dim=layers_dim,
            intrinsic_reward_coefficient=0.01,  # Start with small coefficient
            entropy_coefficient=0.01,
            clipping_epsilon=0.2,  # Standard PPO clip
            value_loss_clip=0.2,
            value_loss_coeff=0.5,
            adversary_loss_coeff=0.1,
            layers_num_channels=[64, 64, 32],  # For CNN if used
            adv_layers_dim=adv_layers_dim,
            clip_grad_norm=0.5,
            cnn_extractor=False,  # No CNN extractor by default
            nb_stack=1,  # Default frame stacking
            episodic_count_coefficient=0.0  # Disable episodic counts by default
        ),
        logging=LoggingConfig(
            save_models=True,
            save_models_freq=100,
            use_neptune=False,  # No Neptune logging
            experiment_name=f"D2E_{env_name}",
            neptune_user_name="",
            neptune_project_name="",
            log_grid=False  # No grid logging
        )
    )
    
    return agac_config