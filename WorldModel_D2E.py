import torch.utils.data.distributed
from WorldModel_D2E_Structures import *
from planner_D2E_regularizer_n_step import *
from gym import spaces
from alf_gym_wrapper import tensor_spec_from_gym_space
import os
os.environ['MUJOCO_GL'] = 'egl'  # or ‘osmesa’ for software rendering

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

HYPER_PARAMETERS = {
                    "input_d": 1,
                    "prior_alpha": 7.,  # gamma_alpha
                    "prior_beta": 1.,  # gamma_beta
                    "image_width": 100,

                    "hidden_transit": 100,
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

                    'max_components': 38,
                    'latent_dim': 40,
                    'hidden_dim': 35,
                    'batch_size': 25,
                    'lr': 1e-4,
                    'beta': 1.0,
                    'lambda_img': 1.0,
                    'lambda_latent': 3.0,
                    'n_critic': 3,
                    'grad_clip': 0.5,
                    'img_disc_channels': 16,
                    'img_disc_layers': 5,
                    'latent_disc_layers': 3,
                    'use_actnorm': True
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
                 config,
                 obs_space,
                 act_space,
                 latent_space,
                 step: NCounter,
                 env_name: str,
                 log_dir,
                 prefill_data_generator,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model_params=(((200, 200),), 2, 1),
                 optimizers=((0.0001, 0.5, 0.99),),
                 update_freq=1,
                 update_rate=0.005,
                 discount=0.99,
                 eval_state_mean=False,
                 precision=32,
                 **kwarg):
        super().__init__()
        self.parameter = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self._env = env_name
        self._discount = discount
        self.config = config
        self.batch_size = config.batch_size

        self.obs_space = obs_space
        self.act_space = act_space
        self.latent_space = latent_space

        self.step = step
        self.device = device
        self.precision = precision
        self.eval_state_mean = eval_state_mean
        self.sequence_length = config.sequence_size
        self.register_buffer("tfstep", torch.ones(()) * int(self.step))

        # every next will give 32 (hard coded) data sample
        self.prefill_generator = prefill_data_generator

        self.task_behavior_reply_buffer = DC.Dataset(observation_spec=latent_space, action_spec=act_space,
                                                     size=config.policy_reply_buffer_size,
                                                     group_size=config.sequence_size)
        self.task_behavior_reply_buffer.current_size += config.batch_size

        self.wm = WorldModel(hyperParams,
                             sequence_length=self.sequence_length,
                             env_name=self._env,
                             device=self.device,
                             writer=config.writer,
                             **kwarg)

        self.RewardNorm = StreamNorm(momentum=0.99, scale=1.0, eps=1e-8)
        self.imag_horizon = config.sequence_size
        self.use_amp = True if self.precision == 16 else False

        # Construct agent.
        agent_flags = utils.Flags(
            observation_spec=latent_space,
            action_spec=act_space,
            model_params=model_params,
            optimizers=optimizers,
            batch_size=self.batch_size,
            weight_decays=(self.parameter.weight_decay,),
            update_freq=update_freq,
            update_rate=update_rate,
            discount=discount,
            done=self.parameter.PREDICT_DONE,
            env_name=env_name,
            train_data=self.task_behavior_reply_buffer
        )
        agent_args = Config(agent_flags).agent_args
        self.task_behavior = D2EAgent(**vars(agent_args))
        self.task_behavior_params = itertools.chain(
            self.task_behavior._get_q_source_vars(),
            self.task_behavior._get_p_vars(),
            self.task_behavior._get_c_vars(),
            self.task_behavior._get_v_source_vars(),
            self.task_behavior._a_vars,
            self.task_behavior._ae_vars,
            self.task_behavior._transit_discriminator.parameters()
        )
        self.task_behavior_reply_buffer.current_size -= config.batch_size

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

        with RequiresGrad(self.task_behavior_params):
            with torch.cuda.amp.autocast(self.use_amp):
                seq = self.wm.imagine(
                    self.task_behavior,
                    start,
                    self.imag_horizon,
                    done_slice,
                )

                rewards, mets1 = self.RewardNorm(seq["reward"])
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}

                # add seq to task behavior reply buffer
                if state is None:
                    rewards = rewards.squeeze(-1).cpu()
                    trans = Transition(s1=seq['latent'][:, :-1, :], s2=seq['latent'][:, 1:, :],
                                       a1=seq['action'][:, :-1, :],
                                       a2=seq['action'][:, 1:, :], reward=rewards, done=seq['done'],
                                       discount=seq['discount'])
                    self.task_behavior._train_data.add_transitions_batch(trans)

                # train the policy
                self.task_behavior.train_step()

                mets2 = self.task_behavior._all_train_info
                metrics.update({"expl_" + key: value for key, value in mets2.items()})
                metrics.update(**mets1)

        return metrics

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

        process_batch = self.config.policy_reply_buffer_prefill_batch
        for i in range(self.config.policy_reply_buffer_size // process_batch + 1):
            tran = dict_to_tran(next(self.prefill_generator))
            obs = torch.reshape(tran.s1, (-1, self.parameter.n_channel, self.parameter.image_width,
                                          self.parameter.image_width)).to(self.device)
            next_obs = torch.reshape(tran.s2, (-1, self.parameter.n_channel, self.parameter.image_width,
                                               self.parameter.image_width)).to(self.device)
            _, _, _, z_real, _, _, _ = self.wm.encoder(obs)
            _, _, _, z_next, _, _, _ = self.wm.encoder(next_obs)
            tran.s1 = z_real.reshape(process_batch, -1, self.parameter.latent_dim).detach().cpu()
            tran.s2 = z_next.reshape(process_batch, -1, self.parameter.latent_dim).detach().cpu()
            self.task_behavior_reply_buffer.add_transitions_batch(tran)

    def save(self):
        self.wm.save()
        self.task_behavior.build_checkpointer()

    def load(self):
        self.wm.load_checkpoints()
        self.task_behavior.load_checkpoint()

    def policy(self, observation, reward, done, state=None, mode="train"):
        obs = self.wm.preprocess(observation, reward, done)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])

        _, _, _, z_x, _, _, _ = self.wm.encoder(obs["observation"].to(self.device))

        start = {"feat": z_x}
        seq = self.wm.imagine(self.task_behavior, start, self.imag_horizon, torch.from_numpy(done))
        policy_state = seq['latent']

        # action = self.task_behavior._p_fn(z_x)[1]
        # latent, _, _, _ = self.wm.transition_model.generate(
        #     torch.cat([z_x, action], dim=1).unsqueeze(-1),
        #     seq_len=1
        # )
        # policy_state = latent.clone().detach().squeeze(-1)
        if mode == "eval":
            a_tanh_mode, action, log_pi_a = self.task_behavior._p_fn(policy_state)
            noise = self.parameter.eval_noise
        elif mode in ["explore", "train"]:
            action = self.task_behavior._p_fn(policy_state)[1]
            noise = self.parameter.expl_noise
        action = action[:, 0, :]
        action = action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = policy_state
        return outputs, state


def main(config):
    ep_num = dict(train=0, eval=0)
    video_log_fre = dict(train=10, eval=1)

    def per_episode_record(ep, mode):
        cur_ep_num = ep_num[mode]
        ep_num[mode] = cur_ep_num + 1

        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        writer.add_scalar(f"per_ep_{mode}/return", score, global_step=cur_ep_num)
        writer.add_scalar(f"per_ep_{mode}/length", length, global_step=cur_ep_num)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                writer.add_scalar(f"per_ep_{mode}/sum_{key}", ep[key].sum(), global_step=cur_ep_num)
            if re.match(config.log_keys_mean, key):
                writer.add_scalar(f"per_ep_{mode}/mean_{key}", ep[key].mean(), global_step=cur_ep_num)
            if re.match(config.log_keys_max, key):
                writer.add_scalar(f"per_ep_{mode}/max_{key}", ep[key].max(0).mean(), global_step=cur_ep_num)

        if cur_ep_num % video_log_fre[mode] == 0:
            for key in config.log_keys_video:
                writer_wrapper.video_summary(f"video/{mode}/{key}_{str(cur_ep_num)}",
                                             np.transpose(ep[key], (0, 2, 3, 1)), 1)

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

    environment_name = config.task.partition('_')[2]

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
    writer = SummaryWriter(log_dir=os.path.expanduser(logdir))
    config.writer = writer
    writer_wrapper = TensorBoardOutput(logdir, writer)

    # set up flow control utils
    should_train = Every(config.train_every)
    should_log = Every(config.log_every)
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

    # generate pre train data set
    pretrain_data_generation_config = dict({'batch': config.length, 'length': config.sequence_size})
    pretrain_dataset_generator = iter(train_replay.dataset(**pretrain_data_generation_config))

    pre_train_dataset = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.length,
                                   group_size=config.sequence_size)
    data = next(pretrain_dataset_generator)
    pre_train_dataset.add_transitions_batch(dict_to_tran(data))

    #
    policy_pretrain_generator_config = dict({'batch': config.policy_reply_buffer_prefill_batch,
                                             'length': config.sequence_size})
    policy_pretrain_dataset_generator = iter(train_replay.dataset(**policy_pretrain_generator_config))

    # create the model
    model = D2EAlgorithm(HYPER_PARAMETERS, config, obs_space, act_space, latent_space, step, environment_name,
                         logdir, policy_pretrain_dataset_generator)

    # init lazy module and normalizer based on the pretrain dataset
    model.initialize_lazy_modules(pre_train_dataset)
    model.requires_grad_(False)

    if config.load_model == 1:
        model.load()
        print("Pretrain skipped")
    else:
        print("Pretrain agent")
        num_batch = config.length // config.batch_size
        for ep in range(config.num_pretrain_epoch):
            for i in range(num_batch):
                model.train_(pre_train_dataset.get_batch(np.arange(i * config.batch_size, (i + 1) * config.batch_size)),
                             state="pretrain")
            if ep % config.save_fre == 0:
                model.save()
        model.save()

    # enter train part
    def train_policy(*args):
        return model.policy(*args, mode="explore" if should_expl(step) else "train")

    def eval_policy(*args):
        return model.policy(*args, mode="eval")

    def train_model(tran, worker):
        # every few step in the env, we will train the agent
        if should_train(step):
            for _ in range(config.train_steps):
                mets = model.train_(dict_to_tran(next(train_dataset_generator)))
                if should_log(step):
                    for name, values in mets.items():
                        writer.add_scalar("train_time/" + name,
                                          np.array(torch.tensor(values, device='cpu'), np.float64).mean(), step.value)
                    writer_wrapper.log_dict(model.report(next(report_dataset_generator)), "train_report", step.value)

    train_data_generation_config = dict({'batch': config.batch_size, 'length': config.sequence_size})
    train_dataset_generator = iter(train_replay.dataset(**train_data_generation_config))
    report_dataset_generator = iter(train_replay.dataset(**train_data_generation_config))
    eval_dataset_generator = iter(eval_replay.dataset(**train_data_generation_config))

    train_driver.on_step(train_model)

    print("Start training")
    cur_epoch = 0
    while cur_epoch < config.num_train_epoch:
        writer_wrapper.log_dict(model.report(next(eval_dataset_generator)), "eval_report", cur_epoch)
        eval_driver(eval_policy, episodes=config.eval_eps)
        # if it does not finish an episode, it will continue from where it left
        train_driver(train_policy, steps=config.train_epoch_length)
        cur_epoch += 1
        if cur_epoch % config.save_fre == 0:
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
    parser.add_argument("--policy_reply_buffer_prefill_batch", type=int, default=32, help='how fast we fill out the '
                                                                                          'policy reply buffer')

    parser.add_argument('--expl_until', type=int, default=0, help='frequency of explore....')
    parser.add_argument("--max_episode_steps", type=int, default=1e3, help='an episode corresponds to 1000 steps')
    parser.add_argument("--eval_eps", type=int, default=1, help='??')
    parser.add_argument('--discount', type=int, default=0.99, help="??")
    parser.add_argument("--action_repeat", type=int, default=2, choices=[1, 2, 3])

    parser.add_argument('--log_every', type=int, default=1000, help='print train info frequency, unit is steps')
    parser.add_argument('--train_every', type=int, default=5, help='frequency of training agent, unit is steps')
    parser.add_argument('--train_steps', type=int, default=1, help='number of steps for formal training')
    parser.add_argument('--save_fre', type=int, default=2, help='frequency of saving model, recommend 2')

    parser.add_argument("--prefill", type=int, default=60000,
                        help="generate (prefill / 500) num of episodes for pretrain")
    parser.add_argument('--length', type=int, default=15000, help='num of sequence length chunk generated from prefill '
                                                                  'data for pretrain purpose')
    parser.add_argument('--policy_reply_buffer_size', type=int, default=100, help='length of policy reply buffer')
    parser.add_argument('--sequence_size', type=int, default=10, help='n step size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for pretrain and train process')

    parser.add_argument('--load_prefill', type=int, default=0, help='use exist prefill or not, 1 mean load, '
                                                                    '0 means generate new prefill, we need to generate '
                                                                    'new prefill when running in the first time')
    parser.add_argument('--num_pretrain_epoch', type=int, default=50, help='number of pretraining epochs')
    parser.add_argument('--num_train_epoch', type=int, default=10000, help='number of formal training epochs')
    parser.add_argument('--train_epoch_length', type=int, default=1500, help='total number of steps of 1 train epoch')
    parser.add_argument('--load_model', type=int, default=0, help='if 1 we load pre trained model, 0 means '
                                                                  'generate new model')

    parser.add_argument("--log_keys_video", type=list, default=['s2'], help='??')
    parser.add_argument('--log_keys_sum', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_mean', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_max', type=str, default='^$', help='??')
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main(parse_args())
