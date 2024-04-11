from WorldModel_D2E_Utils import *

Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward, done')


class Optimizer(nn.Module):
    def __init__(self, name, parameters, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r".*", opt="adam",
                 use_amp=False):
        super().__init__()
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._params = list(parameters)
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(self._params, lr=lr, eps=eps),
            "adamw": lambda: torch.optim.AdamW(self._params, lr=lr, betas=(0.9, 0.999), amsgrad=True),
            "adamax": lambda: torch.optim.Adamax(self._params, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(self._params, lr=lr),
            "momentum": lambda: torch.optim.SGD(self._params, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(self._params, self._clip)
        if self._wd:
            self.apply_weight_decay_(self._params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def apply_weight_decay_(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data

    def zero_grad_(self):
        self._opt.zero_grad()

    def step_(self):
        self._opt.step()


class Replay:
    def __init__(
            self,
            directory,
            capacity=0,
            ongoing=False,
            minlen=1,
            maxlen=0,
            prioritize_ends=False,
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._ongoing = ongoing
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._random = np.random.RandomState()
        # filename -> key -> value_sequence
        self._complete_eps = load_episodes(self._directory, capacity, minlen)
        # worker -> key -> value_sequence

        self._ongoing_eps = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
        }

    def add_step(self, transition, worker=0):
        episode = self._ongoing_eps[worker]
        if is_named_tuple_instance(transition):
            for key, value in transition._asdict().items():
                episode[key].append(value)
            if transition.done:
                self.add_episode(episode)
                episode.clear()
        elif isinstance(transition, dict):
            for key, value in transition.items():
                episode[key].append(value)
            if transition["is_last"]:
                self.add_episode(episode)
                episode.clear()

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()

    def load_episodes(self):
        episodes = load_episodes(self._directory, capacity=self._capacity)
        total_length = 0
        for k, v in episodes.items():
            total_length += len(v['done'])

        self._total_steps = total_length
        self._loaded_steps = total_length
        self._total_episodes = len(episodes)
        self._loaded_episodes = len(episodes)
        self._complete_eps = episodes

    def dataset(self, batch, length):
        # example = next(iter(self._generate_chunks(length)))
        dataset = IterableWrapper(iter(self._generate_chunks(length)))
        dataloader = DataLoader(dataset, batch_size=batch)
        return dataloader

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["done"])
                if len(sequence["done"]) < 1:
                    sequence = self._sample_sequence()
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _sample_sequence(self):
        episodes = list(self._complete_eps.values())
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values() if eplen(x) >= self._minlen
            ]
        episode = self._random.choice(episodes)
        total = len(episode["done"])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: convert(v[index: index + length])
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["is_first"] = np.zeros(len(sequence["done"]), bool)
        sequence["is_first"][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence["done"]) <= self._maxlen
        return sequence

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]


class Driver:
    def __init__(self, envs, **kwargs):
        self._state = None
        self._eps = None
        self._obs = None
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.action_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            # start data collection or restart done env to collect data
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs)
                if ob is None or ob.done
            }

            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                tran = Transition(
                    s1=ob.observation,
                    s2=None,
                    a1=None,
                    a2=None,
                    discount=None,
                    reward=None,
                    done=None
                )
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                self._eps[i] = [tran]

            obs = build_dict_from_named_tuple(self._obs)

            # get actions for each of the env
            actions, self._state = policy(obs, self._state, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i].cpu()) for k in actions}
                for i in range(len(self._envs))
            ]

            removed_actions = [actions[i].pop('logprob') for i in range(len(actions))]

            assert len(actions) == len(self._envs)
            obs_next = [e.step(a['action']) for e, a in zip(self._envs, actions)]

            obs_next = [ob() if callable(ob) else ob for ob in obs_next]
            for i, (act, ob, ob_next) in enumerate(zip(actions, self._obs, obs_next)):
                tran = Transition(
                    s1=ob.observation,
                    s2=ob_next.observation,
                    a1=ob.prev_action,
                    a2=act['action'],
                    discount=ob_next.discount,
                    reward=ob_next.reward,
                    done=ob_next.done
                )

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)

                step += 1
                if ob_next.done:
                    ep = self._eps[i]
                    ep = [Transition(*[self._convert(getattr(t, field)) for field in t._fields]) for t in ep if
                          not any(getattr(t, field) is None for field in t._fields)]
                    ep_dict = build_dict_from_named_tuple_list(ep)
                    [fn(ep_dict, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            self._obs = obs_next

    def _convert(self, value):
        if isinstance(value, bool):
            return np.array(value, dtype=np.bool_)
        else:
            value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value


class NormLayer(nn.Module):
    def __init__(self, dim, name):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if not self._layer:
            return features
        return self._layer(features)


class Module(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self._lazy_modules = nn.ModuleDict({})
        self.device = device
        self.to(device)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if name not in self._lazy_modules:
            self._lazy_modules[name] = ctor(*args, **kwargs).to(self.device)
        return self._lazy_modules[name]


class DistLayer(Module):
    def __init__(self, shape, device, dist="mse", min_std=0.1, init_std=0.0):
        super().__init__(device)
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def forward(self, inputs):
        out = self.get("out", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
        out = torch.reshape(out, inputs.shape[:-1] + self._shape)

        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", nn.Linear, inputs.shape[-1], np.prod(self._shape))(inputs)
            std = torch.reshape(std, inputs.shape[:-1] + self._shape)
        if self._dist == "mse":
            dist = torch.distributions.Normal(out, 1.0)
            return ContDist(torch.distributions.Independent(dist, len(self._shape)))
        if self._dist == "normal":
            dist = torch.distributions.Normal(out, std)
            return ContDist(torch.distributions.Independent(dist, len(self._shape)))
        if self._dist == "binary":
            dist = Bernoulli(
                torch.distributions.Independent(torch.distributions.Bernoulli(logits=out), len(self._shape)))
            return dist
        if self._dist == "tanh_normal":
            mean = 5 * torch.tanh(out / 5)
            std = nn.Softplus()(std + self._init_std) + self._min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, len(self._shape))
            return SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = TruncatedNormal(torch.tanh(out), std, -1, 1)
            return ContDist(torch.distributions.Independent(dist, 1))
        if self._dist == "onehot":
            # TODO out is the input with shape (16, 15, 106), how can we convert it as logit and prob
            return OneHotDist(logits=out)
        raise NotImplementedError(self._dist)


class MLP(Module):
    def __init__(self, shape, layers, units, device, act="elu", norm="none", **out):
        super().__init__(device)
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self.device = device
        self._out = out
        self.to(device)

    def forward(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + (x.shape[-1],))
        return self.get("out", DistLayer, self._shape, self.device, **self._out)(x)


class D2EDataset(Dataset):
    def __init__(self, REPLAY_SIZE):
        self.buffer = deque(maxlen=REPLAY_SIZE)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, indx):
        item = self.buffer[indx]
        return item.s1, item.s2, item.reward, item.discount, item.a1, item.a2, item.done

    def append(self, Transition):
        self.buffer.append(Transition)


class StandardScaler(object):

    def __init__(self, device):
        self.input_mu = torch.zeros(1).to(device)
        self.input_std = torch.ones(1).to(device)
        self.target_mu = torch.zeros(1).to(device)
        self.target_std = torch.ones(1).to(device)
        self.device = device

    def fit(self, inputs, targets, scale_dim=0):
        """
        Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the input
        targets : torch.Tensor
            A torch Tensor containing the input
        """
        self.input_mu = torch.mean(inputs, dim=scale_dim, keepdims=True).to(self.device)
        self.input_std = torch.std(inputs, dim=scale_dim, keepdims=True).to(self.device)
        self.input_std[self.input_std < 1e-8] = 1.0
        self.target_mu = torch.mean(targets, dim=scale_dim, keepdims=True).to(self.device)
        self.target_std = torch.std(targets, dim=scale_dim, keepdims=True).to(self.device)
        self.target_std[self.target_std < 1e-8] = 1.0

    def transform(self, inputs, targets=None):
        """
        Transforms the input matrix data using the parameters of this scaler.
        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the points to be transformed.
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.
        Returns
        -------
        norm_inputs : torch.Tensor
            Normalized inputs
        norm_targets : torch.Tensor
            Normalized targets
        """
        norm_inputs = (inputs - self.input_mu) / self.input_std
        norm_targets = None
        if targets is not None:
            norm_targets = (targets - self.target_mu) / self.target_std
        return norm_inputs, norm_targets

    def inverse_transform(self, targets):
        """
        Undoes the transformation performed by this scaler.
        Parameters
        ----------
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.
        Returns
        -------
        output : torch.Tensor
            The transformed wm_image_replay_buffer.
        """
        output = self.target_std * targets + self.target_mu
        return output


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})
