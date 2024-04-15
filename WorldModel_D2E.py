import numpy as np
import torch.utils.data.distributed
from WorldModel_D2E_Structures import *
from planner_D2E_regularizer_n_step import *
import DataCollectionD2E_n_step as DC
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


def append_action_and_latent_obs(s1, a1, s2, n_step):
    latent_shape = s1.shape[-1]
    obs = torch.reshape(s1, (-1, n_step, latent_shape))
    next_obs = torch.reshape(s2, (-1, n_step, latent_shape))
    inputs = torch.cat((obs, a1), dim=-1)
    return inputs, next_obs


class WorldModel(nn.Module):
    def __init__(self,
                 hyperParams,
                 train_data,
                 sequence_length,
                 env_name='Hopper-v2',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 log_dir="logs",
                 restore=False):
        super(WorldModel, self).__init__()

        self.sequence_len = sequence_length
        self.optimizer = None
        self.discriminator_optim = None
        self.wm_image_replay_buffer = train_data
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
        self.n_discriminator_iter = self._params.CRITIC_ITERATIONS
        self._use_amp = False

        # set the sizes
        self.state_dim = self._params.latent_d
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

        self.variational_autoencoder = InfGaussMMVAE(hyperParams,
                                                     K=self._params.K,
                                                     nchannel=self._params.n_channel,
                                                     z_dim=self.state_dim,
                                                     w_dim=self._params.latent_w,
                                                     hidden_dim=self._params.hidden_d,
                                                     device=self.device,
                                                     img_width=self._params.image_width,
                                                     batch_size=self._params.batch_size,
                                                     num_layers=4,
                                                     include_elbo2=True,
                                                     use_mse_loss=True)
        self.encoder = self.variational_autoencoder.GMM_encoder
        self.decoder = self.variational_autoencoder.GMM_decoder
        self.discriminator = VAECritic(self.state_dim)

        self.grad_heads = ["reward", "discount", "done"]
        self.loss_scales = {"reward": 1.0, "discount": 1.0, "done": 1.0}
        for name in self.grad_heads:
            assert name in self.heads, name

        self.parameters = itertools.chain(
            self.transition_model.parameters(),
            self.heads["reward"].parameters(),
            self.heads["discount"].parameters(),
            self.heads["done"].parameters(),
            self.variational_autoencoder.get_trainable_parameters()
        )

        if restore:
            self.load_checkpoints()

        self.writer = SummaryWriter(log_dir)
        self.writer_counter = 0

        self.pbar = tqdm(range(self.n_discriminator_iter))
        print("End of initializing the world model")

    def initialize_optimizer(self):
        self.optimizer = Optimizer("world_model",
                                   self.parameters,
                                   lr=self._params.LEARNING_RATE,
                                   wd=self._params.weight_decay,
                                   opt="adamw",
                                   use_amp=self._use_amp)

        self.discriminator_optim = Optimizer("discriminator_model",
                                             self.discriminator.parameters(),
                                             lr=0.5 * self._params.LEARNING_RATE,
                                             opt="adam",
                                             wd=1e-5,
                                             use_amp=self._use_amp)

    def preprocess(self, observation, reward, done):
        if isinstance(observation, np.ndarray):
            if observation.max() > 1:
                observation = observation / 255.0
            observation = torch.from_numpy(observation)

        if isinstance(observation, torch.Tensor) and observation.dtype == torch.uint8:
            observation = observation.to(torch.float32) / 255.0

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)

        if isinstance(done, np.ndarray):
            done = torch.from_numpy(done)

        data = {"reward": {
            "identity": lambda x: x,
            "sign": torch.sign,
            "tanh": torch.tanh,
        }[self._clip_rewards](reward).unsqueeze(-1), "discount": 1.0 - done.float().unsqueeze(-1)}
        data["discount"] *= self._discount
        data["observation"] = observation

        data["done"] = done
        return data

    def train_discriminator_get_latent(self, obs):
        z_real, z_fake = None, None
        for _ in self.pbar:
            z_real, z_x_mean, z_x_sigma, c_posterior, w_x_mean, w_x_sigma, gmm_dist, z_wc_mean_prior, \
            z_wc_logvar_prior, x_reconstructed = self.variational_autoencoder(obs)
            z_fake = gmm_dist.sample()
            critic_real = self.discriminator(z_real).reshape(-1)
            critic_fake = self.discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(self.discriminator, z_real, z_fake, device=self.device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + self._params.LAMBDA_GP * gp)
            self.discriminator_optim.zero_grad_()
            loss_critic.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self._params.MAX_GRAD_NORM, norm_type=2)
            self.discriminator_optim.step_()

        gen_fake = self.discriminator(z_fake).reshape(-1)
        return z_real, gen_fake

    def init_normalizer(self, full_train_set):
        batch_size = 32
        num_batch = full_train_set.size // batch_size
        stacked_in, stacked_out = None, None
        for i in range(num_batch):
            batch = full_train_set.get_batch(np.arange(i * batch_size, (i + 1) * batch_size))

            obs = torch.reshape(batch.s1, (-1, HYPER_PARAMETERS['n_channel'], HYPER_PARAMETERS['image_width'],
                                           HYPER_PARAMETERS['image_width'])).to(self.device)
            next_obs = torch.reshape(batch.s2, (-1, HYPER_PARAMETERS['n_channel'], HYPER_PARAMETERS['image_width'],
                                                HYPER_PARAMETERS['image_width'])).to(self.device)
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

        obs = torch.reshape(data.s1, (-1, HYPER_PARAMETERS['n_channel'], HYPER_PARAMETERS['image_width'],
                                      HYPER_PARAMETERS['image_width'])).to(self.device)
        next_obs = torch.reshape(data.s2, (-1, HYPER_PARAMETERS['n_channel'], HYPER_PARAMETERS['image_width'],
                                           HYPER_PARAMETERS['image_width'])).to(self.device)

        z_real, gen_fake = self.train_discriminator_get_latent(obs)

        self.transition_model.train()
        self.optimizer.zero_grad()
        w_x, w_x_mean, w_x_sigma, z_next, z_next_mean, z_next_sigma, c_posterior = self.encoder(next_obs)

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

        metrics, z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean, w_posterior_sigma, \
        dist, z_prior_mean, z_prior_logvar, X_reconst = self.variational_autoencoder.get_ELBO(obs)
        metrics["wasserstein_gp_loss"] = -torch.mean(gen_fake)
        metrics["total_observe_loss"] = metrics["loss"] + metrics["wasserstein_gp_loss"]

        # reward and discount losses
        for k, v in losses.items():
            metrics.update({k + "_loss": v.detach().cpu()})

        metrics["transition_total_loss"] = transition_loss + transition_disc_loss + transition_gradient_penalty

        metrics["total_model_loss"] = metrics["total_observe_loss"] + metrics["transition_total_loss"] + model_loss
        metrics["total_model_loss"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters, self._params.MAX_GRAD_NORM, norm_type=2)
        self.optimizer.step_()

        self.writer.add_scalar('Variational autoencoder loss', metrics["total_observe_loss"].item(),
                               global_step=self.writer_counter)
        self.writer.add_scalar('transition loss', metrics["transition_total_loss"].item(),
                               global_step=self.writer_counter)
        self.writer.add_scalar('reward loss', metrics["reward_loss"].item(), global_step=self.writer_counter)
        self.writer.add_scalar('discount loss', metrics["discount_loss"].item(), global_step=self.writer_counter)
        self.writer.add_scalar('total world model loss', metrics["total_model_loss"].item(),
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

        for i in tqdm(range(horizon)):
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
        truth = (observation[:5] + 0.5).to(self.device)
        inputs = observation[:5]
        actions = action[:5].to(self.device)
        embed, embed_mean, embed_sigma, _, _, _, _, _, _, recon = self.variational_autoencoder(inputs.to(self.device))

        next_state, sample_mu, sample_sigma, state = self.transition_model.generate(torch.cat([embed, actions], dim=1).unsqueeze(-1))
        openl = self.decoder(next_state.squeeze(-1))

        # concatenate on a new dimension, type
        model = torch.concat([(recon + 0.5).unsqueeze(1), (openl + 0.5).unsqueeze(1)], dim=1)
        error = (recon - truth + 1) / 2
        video = torch.concat([truth.unsqueeze(1), model, error.unsqueeze(1)], 1)
        B, T, C, H, W = video.shape
        return video.permute(1, 3, 0, 4, 2).reshape(T, H, B * W, C).cpu()

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
                step = self._task_behavior.global_step
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

        embed = self.wm.encoder(obs["observation"].to(self.device))
        sample = (mode == "train") or not self.eval_state_mean

        action = self._task_behavior._p_fn(embed)
        latent, _, _, _ = self.wm.transition_model.generate(
            torch.cat([embed, action], dim=1),
            seq_len=embed.shape[-1]
        )
        policy_state = latent.copy()
        if mode == "eval":
            a_tanh_mode, action, log_pi_a = self._task_behavior._p_fn(policy_state)
            noise = self.parameter.eval_noise
        elif mode in ["explore", "train"]:
            action = self._expl_behavior._p_fn(policy_state)[1]
            noise = self.parameter.expl_noise
        action = action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = latent
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
        # check this too: https://github.com/jsikyoon/dreamer-torch/blob/main/dreamer.py
        suite, task = config.task.split("_", 1)  ###??
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

    n_step_data_generation_config = dict({'batch': config.length, 'length': config.sequence_size})
    train_replay_param = dict({'capacity': 2e6, 'ongoing': False, 'minlen': 50, 'maxlen': 50, 'prioritize_ends': True})
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()

    logdir.mkdir(parents=True, exist_ok=True)

    train_replay = Replay(logdir / "train_episodes", **train_replay_param, )
    eval_replay = Replay(logdir / "eval_episodes", **dict(
        capacity=train_replay_param["capacity"] // 10,
        minlen=train_replay_param['minlen'],
        maxlen=train_replay_param['maxlen'], ), )

    if config.load_prefill == 1:
        train_replay.load_episodes()
        eval_replay.load_episodes()

    step = NCounter(train_replay.stats["total_steps"])
    outputs = [
        TerminalOutput(),
        JSONLOutput(logdir),
        TensorBoardOutput(logdir),
    ]

    logger = Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = Every(config.train_every)
    should_log = Every(config.log_every)
    should_video_train = Every(config.eval_every)
    should_video_eval = Every(config.eval_every)
    should_expl = Until(config.expl_until // config.action_repeat)

    print("Create envs")  # debug: stuck here
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none":
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        def make_async_env(mode):
            return Async(functools.partial(make_env, mode), config.envs_parallel)

        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(num_eval_envs)]
    act_space = train_envs[0]._env._action_spec

    obs_space = train_envs[0]._env._observation_spec

    # create latent space
    latent_space = tensor_spec_from_gym_space(
        spaces.Box(low=0, high=1, shape=(HYPER_PARAMETERS['latent_d'],), dtype=np.float32))

    train_driver = Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode_record(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode_record(ep, mode="eval"))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill and config.load_prefill != 1:
        print(f"Prefill dataset ({prefill} steps).")

        random_actor = torch.distributions.independent.Independent(
            torch.distributions.uniform.Uniform(torch.Tensor(act_space.low)[None],
                                                torch.Tensor(act_space.high)[None]), 1)

        def random_agent(o, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {'action': action, 'logprob': logprob}, None

        train_driver(policy=random_agent, steps=prefill, episodes=1)
        eval_driver(policy=random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create model")
    train_dataset_generator = iter(train_replay.dataset(**n_step_data_generation_config))
    report_dataset_generator = iter(train_replay.dataset(**n_step_data_generation_config))
    eval_dataset_generator = iter(eval_replay.dataset(**n_step_data_generation_config))

    environment_name = config.task.partition('_')[2]

    full_train_dataset = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.length,
                                    group_size=config.sequence_size)
    data = next(train_dataset_generator)
    full_train_dataset.add_transitions_batch(Transition(
        s1=(data['s1'] / 255.0),
        s2=(data['s2'] / 255.0),
        a1=data['a1'],
        a2=data['a2'],
        reward=data['reward'],
        discount=data['discount'],
        done=data['done']))

    task_behavior_reply_buffer = DC.Dataset(observation_spec=latent_space, action_spec=act_space,
                                            size=config.policy_reply_buffer_size,
                                            group_size=config.sequence_size)
    task_behavior_reply_buffer.current_size += config.batch_size

    wm_imagine_reply_buffer = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.batch_size,
                                         group_size=config.sequence_size)
    wm_imagine_reply_buffer.current_size += config.batch_size

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
        for ep in range(config.num_epoch):
            for i in range(num_batch):
                train_agent(full_train_dataset.get_batch(np.arange(i * config.batch_size, (i + 1) * config.batch_size)))
            if ep % 50 == 0:
                model.save()
        model.save()

    def train_policy(*args):
        return model.policy(*args, mode="explore" if should_expl(step) else "train")

    def eval_policy(*args):
        return model.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset_generator))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(model.report(next(report_dataset_generator)), prefix="train")
            logger.write(fps=True)

    train_driver.on_step(train_step)

    cur_step = 0
    while cur_step < config.total_train_steps:
        logger.write()
        print("Start evaluation.")
        logger.add(model.report(next(eval_dataset_generator)), prefix="eval")
        eval_driver(eval_policy, episodes=config.eval_eps)
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
    parser.add_argument("--action_repeat", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--eval_every', type=int, default=1e5)
    parser.add_argument('--log_every', type=int, default=1e4, help='print train info frequency')
    parser.add_argument('--train_every', type=int, default=5, help='frequency of training')
    parser.add_argument('--train_steps', type=int, default=1, help='frequency of training')
    parser.add_argument("--prefill", type=int, default=50000, help="generate prefill / 500 num of episodes")
    parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
    parser.add_argument('--length', type=int, default=32, help='num of sequence length chunk')
    parser.add_argument('--policy_reply_buffer_size', type=int, default=10000, help='length of policy reply buffer')
    parser.add_argument('--sequence_size', type=int, default=15, help='n step size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--expl_until', type=int, default=0, help='frequency of explore....')
    parser.add_argument("--max_episode_steps", type=int, default=1e3, help='an episode corresponds to 1000 steps')
    parser.add_argument("--envs", type=int, default=1, help='should be updated??')
    parser.add_argument("--eval_eps", type=int, default=1, help='??')
    parser.add_argument("--envs_parallel", type=str, default="none", help='??')
    parser.add_argument("--log_keys_video", type=list, default=['s2'], help='??')
    parser.add_argument('--log_keys_sum', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_mean', type=str, default='^$', help='??')
    parser.add_argument('--log_keys_max', type=str, default='^$', help='??')
    parser.add_argument('--discount', type=int, default=0.99, help="??")
    parser.add_argument('--task', type=str, default="dmc_cheetah_run", help="name of the environment")
    parser.add_argument('--sequence_length', type=int, default=100, help="the length of an episode")
    parser.add_argument('--model_params', type=tuple, default=(200, 200), help='number of layers in the actor network')
    parser.add_argument('--load_prefill', type=int, default=1, help='use exist prefill or not')
    parser.add_argument('--num_epoch', type=int, default=1, help='number of pretraining epochs')
    parser.add_argument('--total_train_steps', type=int, default=100, help='number of pretraining epochs')
    parser.add_argument('--load_model', type=int, default=1, help='if 1 we load old model')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main(parse_args())
