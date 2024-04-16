import torch.utils.data.distributed
from WorldModel_D2E_Structures import *
from planner_D2E_regularizer_n_step import *
from gym import spaces
from alf_gym_wrapper import tensor_spec_from_gym_space

sys.path.append(os.path.join(os.getcwd(), 'VRNN'))

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, os.getcwd())

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

"""
The purpose of this script is to integrate three key components of our dream to explore algorithm:
(1) latent representation learning, 
(2) environment dynamics learning (prediction), 
(3) planning. 
The first component involves the use of an infinite Gaussian mixture variational autoencoder to learn latent representations (implemented in Hierarchical_StickBreaking_GMMVAE.py). 
The second component focuses on learning the dynamics of the environment through a variational sequential neural process (VSNP) architecture, which can be found in the VRNN directory and main.py file. 
The third component involves building a variational model-based actor critic algorithm using planner_D2E_regularizer.py.

We create a world model that combines the first two components and adds two extra heads to compute reward and discount. 
Next, we develop the D2E algorithm that brings together the world model and the planning algorithm. 
The goal is to learn the latent dynamics of the environment through the world model and then simulate imaginative trajectories from the model. 
Finally, we use the planning algorithm to determine the best policy that maximizes the expected reward from the simulated trajectories.

Overall, this script aims to provide a comprehensive framework for dream to explore algorithm that incorporates various key components and techniques to enable effective learning of the world model and planning.

"""

HYPER_PARAMETERS = {"batch_size": 240,
                    "input_d": 1,
                    "prior_alpha": 7.,  # gamma_alpha
                    "prior_beta": 1.,  # gamma_beta
                    "K": 25,
                    "image_width": 100,
                    "hidden_d": 300,
                    "latent_d": 100,
                    "latent_w": 200,
                    "hidden_transit": 100,
                    "LAMBDA_GP": 10,  # hyperparameter for WAE with gradient penalty
                    "LEARNING_RATE": 2e-4,
                    "CRITIC_ITERATIONS": 5,
                    "GAMMA": 0.99,
                    "PREDICT_DONE": False,
                    "seed": 1234,
                    "number_of_mixtures": 8,
                    "weight_decay": 1e-5,
                    "n_channel": 3,
                    "VRNN_Optimizer_Type": "MADGRAD",
                    "MAX_GRAD_NORM": 100.,
                    "expl_behavior": "greedy",
                    "expl_noise": 0.0,
                    "eval_noise": 0.0,
                    "replay_buffer_size": int(1e4),
                    }


def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount = amount.to(action.dtype)
    if hasattr(act_space, "n"):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return OneHotDist(probs=probs).sample()
    else:
        return torch.clip(torch.distributions.Normal(action, amount).sample(), -1, 1)


class D2EAlgorithm(nn.Module):
    # https://github.com/sai-prasanna/dreamerv2_torch/tree/main/dreamerv2_torch/agent.py
    def __init__(self,
                 hyperParams,
                 train_data,
                 agent_reply_buffer,
                 sequence_length,
                 obs_space,
                 act_space,
                 latent_space,
                 step: NCounter,
                 env_name: str,
                 log_dir,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model_params=(((200, 200),), 2, 1),
                 optimizers=((0.0001, 0.5, 0.99),),
                 batch_size=50,
                 update_freq=1,
                 update_rate=0.005,
                 discount=0.99,
                 replay_buffer_size=int(1e6),
                 eval_state_mean=False,
                 precision=32,
                 **kwarg):
        super().__init__()
        self.parameter = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self._env = env_name
        self._discount = discount
        self.batch_size = batch_size

        self.obs_space = obs_space
        self.act_space = act_space
        self.latent_space = latent_space

        self.step = step
        self.device = device
        self.precision = precision
        self.eval_state_mean = eval_state_mean
        self.sequence_length = sequence_length
        self.register_buffer("tfstep", torch.ones(()) * int(self.step))
        self.wm = WorldModel(hyperParams,
                             train_data=train_data,
                             sequence_length=self.sequence_length,
                             env_name=self._env,
                             device=self.device,
                             **kwarg)

        self.RewardNorm = StreamNorm(momentum=0.99, scale=1.0, eps=1e-8)
        self.imag_horizon = sequence_length
        self._use_amp = True if self.precision == 16 else False

        train_summary_dir = os.path.join(log_dir, 'train')
        self.train_summary_writer = SummaryWriter(train_summary_dir)
        # Construct agent.
        # Initialize wm_image_replay_buffer.

        agent_flags = utils.Flags(
            observation_spec=latent_space,
            action_spec=act_space,
            model_params=model_params,
            optimizers=optimizers,
            batch_size=batch_size,
            weight_decays=(self.parameter.weight_decay,),
            update_freq=update_freq,
            update_rate=update_rate,
            discount=discount,
            done=self.parameter.PREDICT_DONE,
            env_name=env_name,
            train_data=agent_reply_buffer
        )
        agent_args = Config(agent_flags).agent_args
        self._task_behavior = D2EAgent(**vars(agent_args))
        self._task_behavior_params = itertools.chain(
            self._task_behavior._get_q_source_vars(),
            self._task_behavior._get_p_vars(),
            self._task_behavior._get_c_vars(),
            self._task_behavior._get_v_source_vars(),
            self._task_behavior._a_vars,
            self._task_behavior._ae_vars,
            self._task_behavior._transit_discriminator.parameters()
        )

    def train_(self, data, state=None):
        """
        List of operations happen here in this module 
        1)train the world model
        2)imagine state, action, reward, and discount from the world model
        3) normalize reward
        4) update reward and put them as an appropriate format to train D2E agent
        5) train the policy
        6) we should minimize the difference between the reward and discount (maybe observation) here too???
        """

        metrics = {}

        real_embed, outs, mets = self.wm.train_(data)
        z_real = outs['embedding']
        z_next = outs['post']
        metrics.update(mets)

        # we take a slice of (batch, seq, ...)
        slice_index = 0
        z_real_slice = torch.reshape(z_real, (self.batch_size, self.sequence_length, -1))[:, slice_index, :]
        done_slice = data.done[:, slice_index]
        start = {"feat": z_real_slice}

        with RequiresGrad(self._task_behavior_params):
            with torch.cuda.amp.autocast(self._use_amp):
                seq = self.wm.imagine(
                    self._task_behavior,
                    start,
                    self.imag_horizon,
                    done_slice,
                )

                rewards, mets1 = self.RewardNorm(seq["reward"])
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}

                # add seq to task behavior reply buffer
                rewards = rewards.squeeze(-1).cpu()
                trans = Transition(s1=seq['latent'][:, :-1, :], s2=seq['latent'][:, 1:, :], a1=seq['action'][:, :-1, :],
                                   a2=seq['action'][:, 1:, :], reward=rewards, done=seq['done'],
                                   discount=seq['discount'])
                self._task_behavior._train_data.add_transitions_batch(trans)

                # train the policy
                self._task_behavior.train_step()
                # step = self._task_behavior.global_step
                # get all the loss values from policy network
                # here: https://github.com/neuronphysics/AIME/blob/1077b972d2ce85882cf54558370b21d83f687ce5/planner_D2E_regularizer.py#L1329
                mets2 = self._task_behavior._all_train_info
                metrics.update({"expl_" + key: value for key, value in mets2.items()})
                metrics.update(**mets1)

        return state, metrics

    def report(self, data):
        report = {}
        processed_data = self.wm.preprocess(data['s1'], data['reward'], data['done'])
        batch, seq, channel, image_width, _ = data['s1'].shape
        obs = torch.reshape(processed_data['observation'], (-1, channel, image_width, image_width))
        action = torch.reshape(data['a1'], (-1, data['a1'].shape[-1]))
        report[f"openl_gym"] = self.wm.video_pred(obs, action)
        return report

    def initialize_lazy_modules(self, full_train_set):
        self.wm.initialize_optimizer()
        self.wm.init_normalizer(full_train_set)

    def save(self):
        self.wm.save()
        self._task_behavior._build_checkpointer()

    def load(self):
        self.wm.load_checkpoints()
        self._task_behavior._load_checkpoint()

    def policy(self, observation, reward, done, state=None, mode="train"):
        obs = self.wm.preprocess(observation, reward, done)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])

        _, _, _, z_x, _, _, _ = self.wm.encoder(obs["observation"].to(self.device))

        start = {"feat": z_x}
        seq = self.wm.imagine(self._task_behavior, start, self.imag_horizon, torch.from_numpy(done))
        policy_state = seq['latent']

        # action = self._task_behavior._p_fn(z_x)[1]
        # latent, _, _, _ = self.wm.transition_model.generate(
        #     torch.cat([z_x, action], dim=1).unsqueeze(-1),
        #     seq_len=1
        # )
        # policy_state = latent.clone().detach().squeeze(-1)
        if mode == "eval":
            a_tanh_mode, action, log_pi_a = self._task_behavior._p_fn(policy_state)
            noise = self.parameter.eval_noise
        elif mode in ["explore", "train"]:
            action = self._task_behavior._p_fn(policy_state)[1]
            noise = self.parameter.expl_noise
        action = action[:, 0, :]
        action = action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = policy_state
        return outputs, state


def main(config):
    def per_episode_record(ep, mode):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)

        if should(step) or True:
            episode_name = str(replay.stats['loaded_episodes'])
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}_ep_{episode_name}", np.transpose(ep[key], (0, 2, 3, 1)))

        logger.write()

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            domain_name, task_name = task.split("_")
            new_env = DMCGYMWrapper(
                domain_name=domain_name,
                task_name=task_name,
                visualize_reward=False,
                from_pixels=True,
                height=100,
                width=100,
                camera_id=1,
            )
            new_env = FrameSkip(new_env, config.action_repeat)
            new_env = wrap_env(new_env,
                               env_id=None,
                               discount=config.discount,
                               max_episode_steps=config.max_episode_steps,
                               gym_env_wrappers=(),
                               alf_env_wrappers=(),
                               image_channel_first=False,
                               )
            new_env.seed(config.seed)
        else:
            raise NotImplementedError(suite)

        return new_env

    # set up seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)

    # set up log dir
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # set up reply buffer
    train_replay_param = dict({'capacity': 2e6, 'ongoing': False, 'minlen': 50, 'maxlen': 50, 'prioritize_ends': True})
    train_replay = Replay(logdir / "train_episodes", **train_replay_param, )
    eval_replay = Replay(logdir / "eval_episodes", **dict(
        capacity=train_replay_param["capacity"] // 10,
        minlen=train_replay_param['minlen'],
        maxlen=train_replay_param['maxlen'], ), )

    if config.load_prefill == 1:
        train_replay.load_episodes()
        eval_replay.load_episodes()

    # set up logger
    step = NCounter(train_replay.stats["total_steps"])
    outputs = [
        TerminalOutput(),
        JSONLOutput(logdir),
        TensorBoardOutput(logdir),
    ]
    logger = Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    # set up flow control utils
    should_train = Every(config.train_every)
    should_log = Every(config.log_every)
    should_video_train = Every(config.eval_every)
    should_video_eval = Every(config.eval_every)
    should_expl = Until(config.expl_until // config.action_repeat)

    # set up envs
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none":
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        def make_async_env(mode):
            return Async(functools.partial(make_env, mode), config.envs_parallel)

        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(num_eval_envs)]

    # obtain all the data shape
    act_space = train_envs[0]._env._action_spec
    obs_space = train_envs[0]._env._observation_spec
    latent_space = tensor_spec_from_gym_space(
        spaces.Box(low=0, high=1, shape=(HYPER_PARAMETERS['latent_d'],), dtype=np.float32))

    # set up train and eval drivers
    train_driver = Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode_record(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode_record(ep, mode="eval"))
    eval_driver.on_episode(eval_replay.add_episode)

    # prefill the reply buffer if needed
    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill and config.load_prefill != 1:
        print(f"Prefill dataset ({prefill} steps).")

        proto_act_space = train_envs[0]._env.gym.action_space
        random_actor = torch.distributions.independent.Independent(
            torch.distributions.uniform.Uniform(torch.Tensor(proto_act_space.low)[None],
                                                torch.Tensor(proto_act_space.high)[None]), 1)

        def random_agent(o, r, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {'action': action, 'logprob': logprob}, None

        train_driver(policy=random_agent, steps=prefill, episodes=1)
        eval_driver(policy=random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    pre_train_data_generation_config = dict({'batch': config.length, 'length': config.sequence_size})
    pre_train_dataset_generator = iter(train_replay.dataset(**pre_train_data_generation_config))

    full_train_dataset = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.length,
                                    group_size=config.sequence_size)
    data = next(pre_train_dataset_generator)
    full_train_dataset.add_transitions_batch(dict_to_tran(data))

    task_behavior_reply_buffer = DC.Dataset(observation_spec=latent_space, action_spec=act_space,
                                            size=config.policy_reply_buffer_size,
                                            group_size=config.sequence_size)
    task_behavior_reply_buffer.current_size += config.batch_size

    wm_imagine_reply_buffer = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.batch_size,
                                         group_size=config.sequence_size)
    wm_imagine_reply_buffer.current_size += config.batch_size

    environment_name = config.task.partition('_')[2]
    model = D2EAlgorithm(HYPER_PARAMETERS, wm_imagine_reply_buffer, task_behavior_reply_buffer, config.sequence_size,
                         obs_space, act_space, latent_space, step, environment_name, batch_size=config.batch_size,
                         log_dir=logdir)

    model.initialize_lazy_modules(full_train_dataset)

    task_behavior_reply_buffer.current_size = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
    task_behavior_reply_buffer._current_idx = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
    model.requires_grad_(False)

    train_agent = CarryOverState(model.train_)
    if config.load_model == 1:
        model.load()
    else:
        print("Pretrain agent")
        num_batch = config.length // config.batch_size
        for ep in range(config.num_pretrain_epoch):
            for i in range(num_batch):
                train_agent(full_train_dataset.get_batch(np.arange(i * config.batch_size, (i + 1) * config.batch_size)))
            if ep % 50 == 0:
                model.save()
        model.save()

    # enter train part
    def train_policy(*args):
        return model.policy(*args, mode="explore" if should_expl(step) else "train")

    def eval_policy(*args):
        return model.policy(*args, mode="eval")

    def train_model(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(dict_to_tran(next(train_dataset_generator)))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(torch.tensor(values, device='cpu'), np.float64).mean())
                metrics[name].clear()
            logger.add(model.report(next(report_dataset_generator)), prefix="train")
            logger.write(fps=True)

    train_data_generation_config = dict({'batch': config.batch_size, 'length': config.sequence_size})
    train_dataset_generator = iter(train_replay.dataset(**train_data_generation_config))
    report_dataset_generator = iter(train_replay.dataset(**train_data_generation_config))
    eval_dataset_generator = iter(eval_replay.dataset(**train_data_generation_config))

    train_driver.on_step(train_model)

    cur_step = 0
    while cur_step < config.num_train_epoch:
        logger.write()
        # print("Start evaluation.")
        # logger.add(model.report(next(eval_dataset_generator)), prefix="eval")
        # eval_driver(eval_policy, episodes=config.eval_eps)
        print("Start training.")
        train_driver(train_policy, steps=config.eval_every)

        if cur_step % 50 == 0:
            model.save()
    model.save()

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your search algorithms.")
    parser.add_argument('--logdir', type=str, default=os.path.join(os.getcwd(), 'logs'),
                        help='a path to the log directory')
    parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
    parser.add_argument('--task', type=str, default="dmc_cheetah_run", help="name of the environment")
    parser.add_argument("--envs", type=int, default=1, help='should be updated??')
    parser.add_argument('--model_params', type=tuple, default=(200, 200), help='number of layers in the actor network')
    parser.add_argument("--envs_parallel", type=str, default="none", help='??')

    parser.add_argument("--action_repeat", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--eval_every', type=int, default=1e5)
    parser.add_argument('--log_every', type=int, default=1e4, help='print train info frequency')
    parser.add_argument('--train_every', type=int, default=5, help='frequency of training')
    parser.add_argument('--train_steps', type=int, default=1, help='number of steps for formal training')
    parser.add_argument("--prefill", type=int, default=50000,
                        help="generate (prefill / 500) num of episodes for pretrain")
    parser.add_argument('--length', type=int, default=32, help='num of sequence length chunk from pre fill data')
    parser.add_argument('--policy_reply_buffer_size', type=int, default=10000, help='length of policy reply buffer')
    parser.add_argument('--sequence_size', type=int, default=15, help='n step size')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for pre train')
    parser.add_argument('--expl_until', type=int, default=0, help='frequency of explore....')
    parser.add_argument("--max_episode_steps", type=int, default=1e3, help='an episode corresponds to 1000 steps')
    parser.add_argument("--eval_eps", type=int, default=1, help='??')
    parser.add_argument("--log_keys_video", type=list, default=['s2'], help='??')
    parser.add_argument('--log_keys_sum', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_mean', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_max', type=str, default='^$', help='??')
    parser.add_argument('--discount', type=int, default=0.99, help="??")
    parser.add_argument('--load_prefill', type=int, default=0, help='use exist prefill or not, 1 mean load')
    parser.add_argument('--num_pretrain_epoch', type=int, default=1, help='number of pretraining epochs')
    parser.add_argument('--num_train_epoch', type=int, default=100, help='number of formal training epochs')
    parser.add_argument('--load_model', type=int, default=1, help='if 1 we load pre trained model')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main(parse_args())
