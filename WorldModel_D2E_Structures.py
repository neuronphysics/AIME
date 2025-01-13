from WorldModel_D2E_Utils import *
from dpgmm_stickbreaking_prior_vae import DPGMMVariationalAutoencoder

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

            # note that obs here is 255 based, we store it as 1 based
            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                tran = Transition(
                    s1=self._obs[i].observation / 255.0,
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
            actions, self._state = policy(obs['observation'] / 255.0, None, obs['done'],
                                          self._state, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i].cpu()) for k in actions}
                for i in range(len(self._envs))
            ]

            # removed_actions = [actions[i].pop('logprob') for i in range(len(actions))]

            assert len(actions) == len(self._envs)
            obs_next = [e.step(a['action']) for e, a in zip(self._envs, actions)]

            obs_next = [ob() if callable(ob) else ob for ob in obs_next]
            for i, (act, ob, ob_next) in enumerate(zip(actions, self._obs, obs_next)):
                tran = Transition(
                    s1=ob.observation / 255.0,
                    s2=ob_next.observation / 255.0,
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
    def __init__(self, shape, device, input_shape, dist="mse", min_std=0.1, init_std=0.0):
        super().__init__(device)
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

        self.get("out", nn.Linear, input_shape, int(np.prod(self._shape)))

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
            return OneHotDist(logits=out)
        raise NotImplementedError(self._dist)


class MLP(Module):
    def __init__(self, shape, layers, units, input_shape, device, act="elu", norm="none", **out):
        super().__init__(device)
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self.device = device
        self._out = out
        self.to(device)

        for index in range(self._layers):
            self.get(f"dense{index}", nn.Linear, input_shape, self._units)
            input_shape = self._units
            self.get(f"norm{index}", NormLayer, input_shape, self._norm)
        self.get("out", DistLayer, self._shape, self.device, self._units, **self._out)

    def forward(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + (x.shape[-1],))
        return self.get("out", DistLayer, self._shape, self.device, self._units, **self._out)(x)


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


class WorldModel(nn.Module):
    def __init__(self,
                 hyperParams,
                 sequence_length,
                 env_name='Hopper-v2',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 writer=None,
                 restore=False):
        super(WorldModel, self).__init__()

        self.sequence_len = sequence_length
        self.optimizer = None
        self.discriminator_optim = None
        self.model_path = os.path.abspath(os.getcwd()) + '/model'

        try:
            os.makedirs(self.model_path, exist_ok=True)
            print("Directory '%s' created successfully" % self.model_path)
        except OSError as error:
            print("Directory '%s' can not be created")
        self.ckpt_path = self.model_path + '/best_model'
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.save_path = os.path.join(self.ckpt_path, 'WorldModel.pth')

        self._params = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self._discount = self._params.GAMMA
        # set the environment
        self._env_name = env_name
        domain_name, task_name = env_name.split("_")
        env = DMCGYMWrapper(
            domain_name=domain_name,
            task_name=task_name,
            visualize_reward=False,
            from_pixels=True,
            height=100,
            width=100,
            camera_id=0,
        )

        self._env = env
        self.n_discriminator_iter = self._params.n_critic
        self._use_amp = False

        # set the sizes
        self.state_dim = self._params.latent_dim
        self.action_dim = env.action_spec().shape[0]

        i = 0
        for key, value in env.observation_spec().items():
            attr_name = f"observation_{key}_dim"
            setattr(self, attr_name, value.shape)
            if i == 0:
                self.observation_dim = tuple(0 for i in range(len(getattr(self, attr_name))))
            self.observation_dim = tuple(map(operator.add, self.observation_dim, getattr(self, attr_name)))
            i += 1

        self.device = device
        self._clip_rewards = "tanh"
        self.heads = nn.ModuleDict()
        input_shape = self.state_dim + self.action_dim
        self.heads["reward"] = MLP(shape=1, layers=4, units=400, input_shape=input_shape, act="ELU", norm="none",
                                   dist="mse", device=self.device)
        self.heads["discount"] = MLP(shape=1, layers=4, units=400, input_shape=input_shape, act="ELU", norm="none",
                                     dist="binary", device=self.device)
        self.heads["done"] = MLP(shape=2, layers=4, units=400, input_shape=input_shape, act="ELU", norm="none",
                                 dist="onehot", device=self.device)

        self.mse_loss = nn.MSELoss()

        # inside VRNN folder we have main.py script : the transition model
        modelstate = ModelState(seed=self._params.seed,
                                nu=self.state_dim + self.action_dim,
                                ny=self.state_dim,
                                sequence_length=sequence_length,
                                h_dim=self._params.hidden_transit,
                                z_dim=self.state_dim,
                                n_layers=2,
                                n_mixtures=self._params.number_of_mixtures,
                                device=device,
                                optimizer_type=self._params.VRNN_Optimizer_Type,
                                )
        self.standard_scaler = StandardScaler(self.device)
        self.transition_model = modelstate.model
        # getting sensory information and building latent state

        self.variational_autoencoder = DPGMMVariationalAutoencoder(max_components=self._params.max_components,
                                                                   input_dim=self._params.image_width,
                                                                   latent_dim=self._params.latent_dim,
                                                                   hidden_dim=self._params.hidden_dim,
                                                                   img_disc_channels=self._params.img_disc_channels,
                                                                   img_disc_layers=self._params.img_disc_layers,
                                                                   latent_disc_layers=self._params.latent_disc_layers,
                                                                   device=device,
                                                                   use_actnorm=self._params.use_actnorm,
                                                                   learning_rate=self._params.lr,
                                                                   grad_clip=self._params.grad_clip)
        self.encoder = self.variational_autoencoder.encoder
        self.decoder = self.variational_autoencoder.decoder
        self.image_discriminator = self.variational_autoencoder.image_discriminator
        self.latent_discriminator = self.variational_autoencoder.latent_discriminator

        self.grad_heads = ["reward", "discount", "done"]
        self.loss_scales = {"reward": 1.0, "discount": 1.0, "done": 1.0}
        for name in self.grad_heads:
            assert name in self.heads, name

        self.parameters = itertools.chain(
            self.transition_model.parameters(),
            self.heads["reward"].parameters(),
            self.heads["discount"].parameters(),
            self.heads["done"].parameters(),
        )

        if restore:
            self.load_checkpoints()

        self.writer = writer
        self.writer_counter = 0

        print("End of initializing the world model")

    def initialize_optimizer(self):
        self.optimizer = Optimizer("world_model",
                                   self.parameters,
                                   lr=self._params.lr,
                                   wd=self._params.weight_decay,
                                   opt="adamw",
                                   use_amp=self._use_amp)

    def preprocess(self, observation, reward, done):
        if isinstance(observation, np.ndarray):
            if observation.max() > 1:
                observation = observation / 255.0
            observation = torch.from_numpy(observation)

        if isinstance(observation, torch.Tensor) and observation.max() > 1:
            observation = observation.to(torch.float32) / 255.0

        if reward is not None and isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)

        if done is not None and isinstance(done, np.ndarray):
            done = torch.from_numpy(done)

        if reward is not None and done is not None:
            data = {"reward": {
                "identity": lambda x: x,
                "sign": torch.sign,
                "tanh": torch.tanh,
            }[self._clip_rewards](reward).unsqueeze(-1), "discount": 1.0 - done.float().unsqueeze(-1)}
            data["discount"] *= self._discount
        else:
            data = {}
        data["observation"] = observation

        data["done"] = done
        return data

    def init_normalizer(self, full_train_set):
        batch_size = 32
        num_batch = full_train_set.size // batch_size
        stacked_in, stacked_out = None, None
        for i in range(num_batch):
            batch = full_train_set.get_batch(np.arange(i * batch_size, (i + 1) * batch_size))

            obs = torch.reshape(batch.s1, (-1, self._params.n_channel, self._params.image_width,
                                           self._params.image_width)).to(self.device)
            next_obs = torch.reshape(batch.s2, (-1, self._params.n_channel, self._params.image_width,
                                                self._params.image_width)).to(self.device)
            _, _, _, z_real, _, _, _ = self.encoder(obs)
            _, _, _, z_next, _, _, _ = self.encoder(next_obs)
            inputs, outputs = append_action_and_latent_obs(z_real, batch.a1.to(self.device), z_next,
                                                           self.sequence_len)
            if i == 0:
                stacked_in = inputs.cpu().detach()
                stacked_out = outputs.cpu().detach()
            else:
                stacked_in = torch.cat((stacked_in, inputs.cpu().detach()), dim=0)
                stacked_out = torch.cat((stacked_out, outputs.cpu().detach()), dim=0)

        normalizer_input, normalizer_output = compute_normalizer(stacked_in, stacked_out)
        self.transition_model.normalizer_input = normalizer_input
        self.transition_model.normalizer_output = normalizer_output

    def train_(self, data):
        self.discriminator.train()

        obs = torch.reshape(data.s1, (-1, self._params.n_channel, self._params.image_width,
                                      self._params.image_width)).to(self.device)
        next_obs = torch.reshape(data.s2, (-1, self._params.n_channel, self._params.image_width,
                                           self._params.image_width)).to(self.device)

        losses, metrics, z_real = self.variational_autoencoder.training_step(obs, self._params.beta,
                                                                             self._params.n_critic,
                                                                             self._params.lambda_img,
                                                                             self._params.lambda_latent)

        self.transition_model.train()
        self.optimizer.zero_grad()
        z_next, z_next_mean, z_next_logvar = self.encoder(next_obs)

        # Prepare & normalize the input/output data for the transition model
        inputs, outputs = append_action_and_latent_obs(z_real, data.a1.to(self.device), z_next, self.sequence_len)
        reshaped_inputs = inputs.permute(0, 2, 1)
        reshaped_outputs = outputs.permute(0, 2, 1)

        with torch.autocast(device_type='cuda', dtype=torch.float32) and torch.backends.cudnn.flags(enabled=False):
            transition_loss, transition_disc_loss, hidden, real_embed, fake_embed, latent = self.transition_model(
                reshaped_inputs, reshaped_outputs)
        with torch.backends.cudnn.flags(enabled=False):
            transition_gradient_penalty = self.transition_model.wgan_gp_reg(real_embed, fake_embed)
        # reward prediction and computing discount factor
        likes, losses = {}, {}
        for name, head in self.heads.items():
            grad_head = name in self.grad_heads
            inp = inputs if grad_head else inputs.detach()
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(getattr(data, key).unsqueeze(-1).to(self.device))
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(self.loss_scales.get(k, 1.0) * v for k, v in losses.items())

        # reward and discount losses
        for k, v in losses.items():
            metrics.update({k + "_loss": v.detach().cpu()})

        metrics["transition_total_loss"] = transition_loss + transition_disc_loss + transition_gradient_penalty

        metrics["total_model_loss"] = metrics["total_observe_loss"] + metrics["transition_total_loss"] + model_loss
        metrics["total_model_loss"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters, self._params.MAX_GRAD_NORM, norm_type=2)
        self.optimizer.step_()

        self.writer.add_scalar('WM/variational_autoencoder_loss', metrics["total_observe_loss"].item(),
                               global_step=self.writer_counter)
        self.writer.add_scalar('WM/transition_loss', metrics["transition_total_loss"].item(),
                               global_step=self.writer_counter)
        self.writer.add_scalar('WM/reward_loss', metrics["reward_loss"].item(), global_step=self.writer_counter)
        self.writer.add_scalar('WM/discount_loss', metrics["discount_loss"].item(), global_step=self.writer_counter)
        self.writer.add_scalar('WM/total_world_model_loss', metrics["total_model_loss"].item(),
                               global_step=self.writer_counter)
        self.writer_counter += 1

        outs = dict(
            embedding=z_real,
            feature=latent,
            like=metrics["reward_loss"] + metrics["discount_loss"],
            prior=hidden,
            post=z_next,
        )

        return real_embed, outs, metrics

    def imagine(self,
                agent: Agent,
                start_state: Dict[str, torch.Tensor],
                horizon: int,
                done: torch.Tensor = None,
                ):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start_state["action"] = agent._p_fn(start_state["feat"])[1]
        seq = {k: [v] for k, v in start_state.items()}

        for i in range(horizon):
            features = torch.cat([seq["feat"][i], seq["action"][i]], dim=-1).unsqueeze(-1)
            state, sample_mu, sample_sigma, hidden = self.transition_model.generate(features, seq_len=1)

            next_latent = state[:, :, -1]
            seq['feat'].append(next_latent)

            action = agent._p_fn(next_latent)[1]
            seq["action"].append(action)

        latent_seq = torch.stack(seq['feat']).transpose(0, 1)
        act_seq = torch.stack(seq['action']).transpose(0, 1)
        seq_feat = torch.cat((latent_seq, act_seq), dim=-1)

        disc = self.heads["discount"](seq_feat[:, :-1, :]).mode().cpu().squeeze(-1)
        seq["reward"] = self.heads["reward"](seq_feat[:, :-1, :]).mode()

        seq['latent'] = latent_seq.cpu()
        seq['action'] = act_seq.cpu()
        seq['feat'] = seq_feat.cpu()

        if done is not None:
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = 1.0 - flatten(done).to(disc.dtype)
            true_first *= self._discount
            true_first = true_first.unsqueeze(-1)
            disc = torch.concat((true_first, disc[:, 1:]), -1)
        else:
            done = torch.zeros_like(disc, dtype=torch.bool)
            done[disc <= self._discount ** DC.MUJOCO_ENVS_LENGTH[self._env_name]] = True
        seq["weight"] = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0).unsqueeze(-1)
        pre_done = self.heads["done"](seq_feat[:, :-1, :]).mode().cpu()
        seq['done'] = torch.argmax(pre_done, dim=-1).to(torch.bool)
        seq['discount'] = disc

        # seq contains all the imagine data
        return seq

    def video_pred(self, observation, action):
        # TODO why we need all these +0.5 here???
        truth = (observation[:6] + 0.5).to(self.device)
        inputs = observation[:5]
        actions = action[:5].to(self.device)
        embed, embed_mean, embed_sigma, _, _, _, _, _, _, recon = self.variational_autoencoder(inputs.to(self.device))

        next_state, sample_mu, sample_sigma, state = self.transition_model.generate(torch.cat([embed, actions],
                                                                                              dim=1)[-1, :].unsqueeze(
            -1).unsqueeze(0))
        openl = self.decoder(next_state.squeeze(-1))
        model = torch.concat([(recon + 0.5), (openl + 0.5)], dim=0)
        # concatenate on a new dimension, type
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)
        # T, C, H, W = video.shape
        return video.permute(0, 2, 3, 1).cpu()

    def save(self):
        torch.save(
            {'transition_model': self.transition_model.state_dict(),
             'reward_model': self.heads['reward'].state_dict(),
             'observation_model': self.variational_autoencoder.state_dict(),
             'observation_discriminator_model': self.discriminator.state_dict(),
             'discount_model': self.heads['discount'].state_dict(),
             'done_model': self.heads['done'].state_dict(),
             'discriminator_optimizer': self.discriminator_optim.state_dict(),
             'world_model_optimizer': self.optimizer.state_dict(), }, self.save_path)

    def load_checkpoints(self):
        if os.path.isfile(self.save_path):
            model_dicts = torch.load(self.save_path, map_location=self.device)
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.variational_autoencoder.load_state_dict(model_dicts['observation_model'])
            self.discriminator.load_state_dict(model_dicts['observation_discriminator_model'])
            self.heads['reward'].load_state_dict(model_dicts['reward_model'])
            self.heads['discount'].load_state_dict(model_dicts['discount_model'])
            self.heads['done'].load_state_dict(model_dicts['done_model'])
            self.optimizer.load_state_dict(model_dicts['world_model_optimizer'])
            self.discriminator_optim.load_state_dict(model_dicts['discriminator_optimizer'])
            print("Loading models checkpoints!")
        else:
            print("Checkpoints not found!")
