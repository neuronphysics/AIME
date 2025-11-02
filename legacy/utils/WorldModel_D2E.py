import DataCollectionD2E as DC
from WorldModel_D2E_Structures import *
from WorldModel_D2E_Utils import *
from gym import spaces
from tensor_utils import tensor_spec_from_gym_space
import os
from legacy.agac_torch.agac.agac_ppo import PPO
from legacy.agac_torch.agac.memory import Memory, Transition
from copy import deepcopy
from welford import Welford
from gym.spaces import Box, Discrete
from legacy.agac_torch.agac.utils import DiscreteGrid, compute_advantages_and_returns
import sys
from pathlib import Path
import gym
from legacy.agac_torch.agac.configs import ExperimentConfig
from torch.utils.tensorboard import SummaryWriter
import argparse
from typing import Dict, List, Iterator, Union
import random
import numpy as np
import re
import functools
import pathlib
os.environ['MUJOCO_GL'] = 'egl'  # or ‘osmesa’ for software rendering

sys.path.append(os.path.join(os.getcwd(), 'VRNN'))
sys.path.append(os.path.join(os.getcwd(), 'agac_torch'))
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, os.getcwd())

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()
try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

"""
Dream to Explore (D2E): World Model Implementation

This code implements a world model-based reinforcement learning algorithm inspired by Dreamer/DreamerV2 
with the integration of Adversarially Guided Actor-Critic (AGAC) for enhanced exploration.

## Key Components:

1. World Model:
   - Encoder (DPGMM-VAE): Maps observations to latent states, learning rich representations
   - Transition Model (VRNN): Predicts next latent states using deterministic and stochastic states
   - Prediction Heads: Estimate rewards and episode termination from latent states
   - Discriminator: Trains adversarially to improve representation quality

2. Actor-Critic with Adversarial Guidance:
   - Actor: Policy network mapping latent states to actions
   - Adversary: Secondary policy trained to maximize KL divergence from actor
   - Critic: Value network estimating expected returns
   - Intrinsic Motivation: Rewards based on policy-adversary disagreement

Training Process:

1. Model Learning:
   - Train the world model on real experience from the environment
   - Maximize ELBO for representation learning with additional adversarial objectives

2. Imagination:
   - Use the world model to generate imagined trajectories
   - Sample actions from the current policy and roll out future states

3. Policy Learning:
   - Train actor and critic on imagined trajectories
   - AGAC provides intrinsic rewards based on actor-adversary divergence
   - Use both dynamics and reinforcement gradients for stable training

4. Environment Interaction:
   - Collect real experiences using the latest policy
   - Add new data to replay buffer and continue the training cycle

This approach combines the sample efficiency of model-based methods with the exploration 
benefits of adversarial techniques, enabling effective learning in sparse-reward environments.
"""



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
 
    """
    Dream to Explore (D2E): A world model-based reinforcement learning agent that
    combines DreamerV2's architecture with:
    
    1. Enhanced representation learning via DPGMM prior
    2. Powerful dynamics model using VRNN with attention
    3. Improved exploration via Adversarially Guided Actor-Critic (AGAC)
    
    This implementation follows a modular architecture with clear separation between:
    - World model learning (representation, dynamics, prediction)
    - Policy optimization through imagination
    - Environment interaction for data collection
    """
    def __init__(self,
                 config: ExperimentConfig,
                 obs_space: gym.Space,
                 act_space: gym.Space,
                 latent_space: gym.Space,
                 step: NCounter,
                 env_name: str,
                 log_dir: str,
                 prefill_data_generator:Iterator[Dict[str, Union[torch.Tensor, np.ndarray]]],
                 device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **kwargs):



        super().__init__()
        """
        Initialize the D2E agent with its components and parameters.
        
        Args:
            config: Configuration object containing hyperparameters
            obs_space: Observation space of the environment
            act_space: Action space of the environment
            latent_space: Specification for the latent space
            step: Step counter object
            env_name: Name of the environment
            log_dir: Directory for logging
            prefill_data_generator: Iterator for prefill data
            device: Device to run computations on (CPU or GPU)
        """        
        # Store parameters
        self._env = env_name
        self.config = config
        self.batch_size = config.batch_size
        self.obs_space = obs_space
        self.act_space = act_space
        self.latent_space = latent_space
        self.log_dir = log_dir
        self.step = step
        self.device = device
        self.sequence_length = config.sequence_size
        self.register_buffer("tfstep", torch.ones(()) * int(self.step))

        # Data sources
        self.prefill_generator = prefill_data_generator
        self._discount = config.discount if hasattr(config, 'discount') else 0.99
        
        # Task behavior replay buffer for policy learning
        self.task_behavior_reply_buffer = DC.Dataset(
            observation_spec=latent_space, 
            action_spec=act_space,
            size=config.policy_reply_buffer_size
        )
        self.task_behavior_reply_buffer.current_size += config.batch_size

        # Initialize World Model
        # This is the primary model for learning representations and dynamics
        self.wm = WorldModel(
            config.world_model,
            sequence_length=self.sequence_length,
            env_name=self._env,
            device=self.device,
            writer=config.writer,
            **kwargs
        )

        # Stream normalization for reward processing
        self.reward_norm = StreamNorm(momentum=0.99, scale=1.0, eps=1e-8)
        
        # Imagination horizon for planning
        self.imag_horizon = config.sequence_size
        # Initialize AGAC memory for storing imagined trajectories
        self.agac_memory = Memory()
        self.task_behavior = PPO(latent_space, act_space, agac_config)

        # Configure precision
        agac_config = create_agac_config(config=config, env_name=env_name)
        self.precision = getattr(agac_config.reinforcement_learning, 'precision', 32) if hasattr(config, 'reinforcement_learning') else 32
        self.use_amp = True if self.precision == 16 else False
        
        # Initialize AGAC policy (Adversarially Guided Actor-Critic)
        self.intrinsic_coef = getattr(agac_config.reinforcement_learning, 'intrinsic_reward_coefficient', 0.01) if hasattr(config, 'reinforcement_learning') else 0.01
        self._lambda_gae = getattr(agac_config.reinforcement_learning, 'lambda_gae', 0.95) if hasattr(config, 'reinforcement_learning') else 0.95
        
        
        
        # Monitoring metrics
        self._metrics = {}

    def initialize_lazy_modules(self, full_train_set):
        """
        Initialize lazy-loaded modules and normalizers.
        
        Args:
            full_train_set: Dataset to use for initialization
        """
        # Initialize world model optimizer
        self.wm.initialize_optimizer()
        
        # Initialize normalizers based on training data
        self.wm.init_normalizer(full_train_set)
        
        # Prefill policy replay buffer with encoded observations
        process_batch = self.config.policy_reply_buffer_prefill_batch
        for i in range(self.config.policy_reply_buffer_size // process_batch + 1):
            # Get next transition
            tran = dict_to_tran(next(self.prefill_generator))
            
            # Encode observations
            obs = torch.reshape(
                tran.s1, 
                (-1, self.parameter.n_channel, self.parameter.image_width, self.parameter.image_width)
            ).to(self.device)
            
            next_obs = torch.reshape(
                tran.s2, 
                (-1, self.parameter.n_channel, self.parameter.image_width, self.parameter.image_width)
            ).to(self.device)
            
            # Get latent representations
            z_real, _, _ = self.wm.encoder(obs)
            z_next, _, _ = self.wm.encoder(next_obs)
            
            # Store latent transitions
            tran.s1 = z_real.reshape(process_batch, -1, self.parameter.latent_dim).detach().cpu()
            tran.s2 = z_next.reshape(process_batch, -1, self.parameter.latent_dim).detach().cpu()
            
            # Add to replay buffer
            self.task_behavior_reply_buffer.add_transitions_batch(tran)

    def train_world_model(self, data):
        """
        Train the world model on a batch of real experience.
        
        This function handles the representation and dynamics learning parts
        of the D2E algorithm, training the DPGMM-VAE and VRNN models.
        
        Args:
            data: Batch of experience data
            
        Returns:
            Dictionary with world model outputs and metrics
        """
        metrics = {}
        
        # Train world model components
        real_embed, outs, world_model_metrics = self.wm.train_(data)
        
        # Extract latent states from outputs
        z_real = outs['embedding']  # Current latent state
        z_next = outs['post']      # Next latent state
        
        # Store metrics
        metrics.update(world_model_metrics)
        
        return {
            'z_real': z_real,
            'z_next': z_next,
            'real_embed': real_embed,
            'outs': outs,
            'metrics': metrics
        }

    def imagine_trajectories(self, start_state, horizon, policy=None, done=None):
        """
        Generate imagined trajectories using the world model.
        
        This is a key part of the DreamerV2 algorithm - using the learned model
        to generate imagined experiences for policy improvement.
        
        Args:
            start_state: Initial latent state
            horizon: Number of steps to imagine
            policy: Policy to use for action selection (uses self.task_behavior if None)
            done: Terminal indicators for initial states
            
        Returns:
            Dictionary of imagined trajectory data
        """
        with torch.no_grad():
            # Use world model to imagine trajectories
            seq = self.wm.imagine(
                self.task_behavior if policy is None else policy,
                start_state,
                horizon,
                done
            )
            
            # Normalize rewards using streaming normalization
            normalized_rewards, reward_metrics = self.reward_norm(seq["reward"])
            seq["normalized_reward"] = normalized_rewards
            
            return seq, reward_metrics

    def prepare_agac_transitions(self, imagined_trajectories):
        """
        Convert imagined trajectories to AGAC transition format for policy training.
        
        Args:
            imagined_trajectories: Trajectories from imagine_trajectories
            
        Returns:
            Number of transitions added to memory
        """
        # Extract trajectory components
        latent_states = imagined_trajectories['latent']
        
        rewards = imagined_trajectories['reward']
        dones = imagined_trajectories['done']
        
        batch_size, seq_len = latent_states.shape[0], latent_states.shape[1] - 1
        
        # Get policy outputs for all states
        pi_outputs = []
        adv_outputs = []
        values = []
        intrinsic_rewards = []
        actions_list = []
        
        # Generate policy outputs for each state in the trajectory
        for t in range(seq_len):
            state = latent_states[:, t]
            
            # Get policy and value outputs
            action, log_pi, adv_log_pi, params_pi, params_adv = self.task_behavior.select_action(
                state.cpu().numpy(), deterministic=False)
            value = self.task_behavior.compute_values(state.cpu().numpy())
            
            # Store policy outputs
            pi_outputs.append((log_pi, params_pi))
            adv_outputs.append((adv_log_pi, params_adv))
            values.append(value)
            
            # Calculate intrinsic rewards (KL divergence between policy and adversary)
            intrinsic_rewards.append(log_pi - adv_log_pi)
            actions_list.append(action.copy())
        
        # Determine value for terminal states
        if torch.all(dones[:, -1] == 1):
            last_r = 0.0
        else:
            # Use critic for value estimation of non-terminal states
            last_state = latent_states[:, -1][dones[:, -1] == 0]
            if last_state.shape[0] > 0:  # Only if there are non-terminal states
                last_r = self.task_behavior.compute_values(last_state.cpu().numpy()).mean()
            else:
                last_r = 0.0
        
        # Calculate advantages using GAE
        advantages, returns = compute_advantages_and_returns(
            rewards.squeeze(-1).cpu().numpy()[:, :-1],
            np.vstack(values).T,  # Reshape to [batch_size, seq_len]
            last_r,
            self._discount,
            self._lambda_gae
        )
        
        # Add intrinsic rewards to advantages
        intrinsic_rewards = np.array(intrinsic_rewards).T  # Reshape to [batch_size, seq_len]
        agac_advantages = advantages + self.intrinsic_coef * intrinsic_rewards
        
        # Create transitions for each state in each trajectory
        transitions_added = 0
        for i in range(batch_size):
            for t in range(seq_len):
                # Create transition with policy information
                transition = Transition(
                    observation=latent_states[i, t].cpu().numpy(),
                    action=actions_list[t][i],
                    extrinsic_return=returns[i, t],
                    advantage=advantages[i, t],
                    agac_advantage=agac_advantages[i, t],
                    value=values[t][i],
                    log_pi=pi_outputs[t][0][i],
                    adv_log_pi=adv_outputs[t][0][i],
                    logits_pi=pi_outputs[t][1][i],
                    adv_logits_pi=adv_outputs[t][1][i],
                    done=dones[i, t].cpu().numpy()
                )
                
                # Add to AGAC memory
                self.agac_memory.add(transition)
                transitions_added += 1
        
        return transitions_added

    def train_policy(self, num_updates=5):
        """
        Train the AGAC policy on imagined trajectories.
        
        Args:
            num_updates: Number of policy updates to perform
            
        Returns:
            Dictionary with policy training metrics
        """
        policy_metrics = {}
        
        # Skip if not enough data
        if self.agac_memory.num_elements < self.batch_size:
            return {"policy_training_skipped": True}
        
        # Perform multiple updates with current data
        for _ in range(num_updates):
            # Sample batch of transitions
            batch = self.agac_memory.sample(self.batch_size)
            
            # Train AGAC policy
            self.task_behavior.train_on_batch(batch, self.intrinsic_coef)
        
        # Collect metrics from AGAC
        for log in self.task_behavior.logs:
            policy_metrics[log.name] = log.value
        
        return policy_metrics

    def train_step(self, data, state=None):
        """
        Perform one complete training step on both world model and policy.
        
        This is the main training function that follows the DreamerV2 pattern:
        1. Train world model on real experience
        2. Generate imagined trajectories using the world model
        3. Train policy on imagined trajectories
        
        Args:
            data: Batch of real experience
            state: Optional state for recurrent processing
            
        Returns:
            Dictionary with all training metrics
        """
        all_metrics = {}
        
        # 1. Train world model on real experience
        wm_results = self.train_world_model(data)
        all_metrics.update(wm_results['metrics'])
        
        # Extract latent state from first step for imagination
        z_real = wm_results['z_real']
        z_real_slice = torch.reshape(z_real, (self.batch_size, self.sequence_length, -1))[:, 0, :]
        done_slice = data.done[:, 0] if hasattr(data, 'done') else None
        
        # Prepare initial state for imagination
        start_state = {"feat": z_real_slice}
        
        # 2. Generate imagined trajectories
        imagined_traj, reward_metrics = self.imagine_trajectories(
            start_state, 
            self.imag_horizon,
            done=done_slice
        )
        
        # Add reward normalization metrics
        reward_metrics = {f"reward_{k}": v for k, v in reward_metrics.items()}
        all_metrics.update(reward_metrics)
        
        # 3. Prepare trajectories for policy training
        transitions_added = self.prepare_agac_transitions(imagined_traj)
        all_metrics["transitions_added"] = transitions_added
        
        # 4. Train policy on imagined trajectories
        policy_metrics = self.train_policy(num_updates=5)
        all_metrics.update(policy_metrics)
        
        # 5. Manage memory
        # Flush memory if it gets too large to avoid off-policy data
        if self.agac_memory.num_elements > 10000:
            self.agac_memory.reset()
            all_metrics["memory_flushed"] = True
        
        return all_metrics

    def policy(self, observation, reward, done, state=None, mode="train"):
        """
        Select actions from current policy based on observations.
        
        This function handles:
        1. Preprocessing observations
        2. Encoding observations to latent states
        3. Using the policy to select actions
        
        Args:
            observation: Environment observation
            reward: Reward from environment
            done: Terminal indicator
            state: Optional recurrent state
            mode: Policy mode ('train', 'explore', or 'eval')
            
        Returns:
            Selected action and updated state
        """
        # Preprocess observations
        obs = self.wm.preprocess(observation, reward, done)
        
        # Update step counter (used for schedules)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])

        # Encode observation to latent state
        z_x, _, _ = self.wm.encoder(obs["observation"].to(self.device))

        # Set initial state for imagination
        start = {"feat": z_x}
        
        # Generate imagined trajectory (unused here, but could be used for planning)
        seq = self.wm.imagine(
            self.task_behavior, 
            start, 
            self.imag_horizon, 
            torch.from_numpy(done)
        )
        
        # Extract policy state from imagined trajectory
        policy_state = seq['latent']

        # Convert to numpy for AGAC
        z_np = z_x.cpu().numpy()
        
        # Select action based on mode
        if mode == "eval":
            action, _, _, _, _ = self.task_behavior.select_action(z_np, deterministic=True)
            noise = self.parameter.eval_noise
        elif mode in ["explore", "train"]:
            action, _, _, _, _ = self.task_behavior.select_action(z_np, deterministic=False)
            noise = self.parameter.expl_noise
        
        # Apply action noise for exploration
        action = torch.tensor(action, device=self.device)
        action = action_noise(action, noise, self.act_space)
        
        # Package outputs
        outputs = {"action": action}
        return outputs, z_x

    def report(self, data):
        """
        Generate visualization and metrics for logging.
        
        Args:
            data: Batch of data for visualization
            
        Returns:
            Dictionary of visualizations
        """
        report = {}
        
        # Preprocess data
        processed_data = self.wm.preprocess(data['s1'], data['reward'], data['done'])
        
        # Reshape for prediction
        batch, seq, channel, image_width, _ = data['s1'].shape
        obs = torch.reshape(processed_data['observation'], (-1, channel, image_width, image_width))
        action = torch.reshape(data['a1'], (-1, data['a1'].shape[-1]))
        
        # Generate video prediction
        report[f"openl_gym"] = self.wm.video_pred(obs, action)
        
        return report
    
    def save(self):
        """Save model weights to disk"""
        self.wm.save()
        self.task_behavior.build_checkpointer()

    def load(self):
        """Load model weights from disk"""
        self.wm.load_checkpoints()
        self.task_behavior.load_checkpoint()

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
                height=config.image_height, 
                width=config.image_width,    
                camera_id=0,
            )
            new_env = NormalizeAction(new_env)
            
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
        spaces.Box(low=0, high=1, shape=(HYPER_PARAMETERS['latent_dim'],), dtype=np.float32))

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
    print(f"observation space: {obs_space}, action space: {act_space}")
    pre_train_dataset = DC.Dataset(observation_spec=obs_space, action_spec=act_space, size=config.length,
                                   group_size=config.sequence_size)
    data = next(pretrain_dataset_generator)
    pre_train_dataset.add_transitions_batch(dict_to_tran(data))

    #
    policy_pretrain_generator_config = dict({'batch': config.policy_reply_buffer_prefill_batch,
                                             'length': config.sequence_size})
    policy_pretrain_dataset_generator = iter(train_replay.dataset(**policy_pretrain_generator_config))

    # create the model
    model = D2EAlgorithm(config, 
                         obs_space, 
                         act_space, 
                         latent_space, 
                         step, 
                         environment_name,
                         logdir, 
                         policy_pretrain_dataset_generator)

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
    parser.add_argument('--model_params', type=tuple, default=(40, 40), help='number of layers in the actor network')
    parser.add_argument("--envs_parallel", type=str, default="none", help='??')
    parser.add_argument("--policy_reply_buffer_prefill_batch", type=int, default=8, help='how fast we fill out the '
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
    parser.add_argument('--image_height', type=int, default=84, 
                        help='Height of input images')
    parser.add_argument('--image_width', type=int, default=84,
                        help='Width of input images')
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main(parse_args())