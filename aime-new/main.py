import argparse
from math import inf
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, bottle_two_output, Encoder, ObservationModel, RecurrentGP, SampleLayer
from infinite_vae import InfGaussMMVAE
from planner import ActorCriticPlanner
from utils import lineplot, write_video
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['4lane', 'loop'], help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=10000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=50, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=10, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=1, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')

## extra hyperparameters for new model
parser.add_argument('--horizon-size', type=int, default=5, metavar='Ho', help='Horizon size')
parser.add_argument('--lagging-size', type=int, default=2, metavar='La', help='Lagging size')
parser.add_argument('--non-cumulative-reward', action='store_true', help='Model non-cumulative rewards')
parser.add_argument('--num-sample-trajectories', type=int, default=20, metavar='nst', help='number of trajectories sample in the imagination part')
parser.add_argument('--temperature-factor', type=float, default=1, metavar='Temp', help='Temperature factor')
parser.add_argument('--discount-factor', type=float, default=0.999, metavar='Temp', help='Discount factor')
parser.add_argument('--num-mixtures', type=int, default=15, metavar='Mix', help='Number of Gaussian mixtures used in the infinite VAE')
parser.add_argument('--w-dim', type=int, default=10, metavar='w', help='dimension of w')
parser.add_argument('--hidden-size', type=int, default=16, metavar='H', help='Hidden size')
parser.add_argument('--state-size', type=int, default=10, metavar='Z', help='State/latent size')
parser.add_argument('--include-elbo2', action='store_true', help='include elbo 2 loss')
parser.add_argument('--use-regular-vae', action='store_true', help='use vae that uses single Gaussian mixture')
parser.add_argument('--rgp-training-interval-ratio', type=float, default=1.1, metavar='In', help='RGP training interval ratio')

args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'latent_kl_loss': [],
           'reward_loss': [], 'value_loss': [], 'policy_loss': [], 'q_loss': [], 'kl_transition_loss': [],
           'elbo1': [], 'elbo2': [], 'elbo3': [], 'elbo4': [], 'elbo5': []}

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
  D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
  # Initialise dataset D with S random seed episodes
  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      D.append(observation, action, reward, done)
      observation = next_observation
      t += 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)


# Initialise model parameters randomly
#observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
#encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.state_size, args.activation_function).to(device=args.device)
recurrent_gp = RecurrentGP(args.horizon_size, args.state_size, env.action_size, args.lagging_size, args.device).to(device=args.device)

if args.use_regular_vae:
  observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
  encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.state_size, args.activation_function).to(device=args.device)
  param_list = list(observation_model.parameters()) + list(encoder.parameters()) + list(recurrent_gp.parameters())
else:
  hyperParams = {"batch_size": args.batch_size,
                "input_d": 1,
                "prior": 1, #dirichlet_alpha'
                "K": args.num_mixtures,
                "hidden_d": args.hidden_size,
                "latent_d": args.state_size,
                "latent_w": args.w_dim}
  infinite_vae = InfGaussMMVAE(hyperParams, args.num_mixtures, 3, 4, args.state_size, args.w_dim, args.hidden_size, args.device, 64, hyperParams["batch_size"], args.include_elbo2).to(device=args.device)
  param_list = list(infinite_vae.parameters()) + list(recurrent_gp.parameters())

optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)

actor_critic_planner = ActorCriticPlanner(args.lagging_size, args.state_size, env.action_size, recurrent_gp, env.action_range[0], env.action_range[1], args.num_sample_trajectories, args.hidden_size).to(device=args.device)
planning_optimiser = optim.Adam(actor_critic_planner.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  if args.use_regular_vae:
    observation_model.load_state_dict(model_dicts['observation_model'])
    encoder.load_state_dict(model_dicts['encoder'])
  else:
    infinite_vae.load_state_dict(model_dicts['infinite_vae'])
  recurrent_gp.load_state_dict(model_dicts['recurrent_gp'])
  optimiser.load_state_dict(model_dicts['optimiser'])
  actor_critic_planner.load_state_dict(model_dicts['actor_critic_planner'])
  planning_optimiser.load_state_dict(model_dicts['planning_optimiser'])

global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence

reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.likelihood, recurrent_gp, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size)))

rgp_training_episode = args.seed_episodes

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []
  if ((episode-1) == rgp_training_episode):
    rgp_training_episode = int(rgp_training_episode * args.rgp_training_interval_ratio)
    if (not args.use_regular_vae):
      infinite_vae.batch_size = args.batch_size * args.chunk_size
    for s in tqdm(range(args.collect_interval)):
      with gpytorch.settings.num_likelihood_samples(1):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
        if args.use_regular_vae:
          latent_mean, latent_std = bottle_two_output(encoder, (observations, ))
          latent_dist = Normal(latent_mean, latent_std)
          latent_kl_loss = kl_divergence(latent_dist, global_prior).sum(dim=2).mean(dim=(0, 1))
          latent_states = latent_dist.rsample()
        else:
          reconstructed_observations, latent_states = bottle_two_output(infinite_vae, (observations, ))
        init_states = latent_states[1:-args.horizon_size].unfold(0, args.lagging_size, 1)
        predicted_rewards = recurrent_gp(init_states, actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1))
        if (not args.non_cumulative_reward):
          true_rewards = rewards[args.lagging_size:].unfold(0, args.horizon_size+1, 1).sum(dim=-1)
        else:
          true_rewards = rewards[args.lagging_size+args.horizon_size:]
        reward_loss = -reward_mll(predicted_rewards, true_rewards).mean()
        if args.use_regular_vae:
          observation_loss = F.mse_loss(bottle(observation_model, (latent_states,)), observations, reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else:
          elbo1, elbo2, elbo3, elbo4, elbo5 = infinite_vae.get_ELBO(observations.view(-1, 3, 64, 64))
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
          for group in optimiser.param_groups:
            group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
        # Update model parameters
        optimiser.zero_grad()
        if args.use_regular_vae:
          (observation_loss + reward_loss + latent_kl_loss).backward()
        else:
          (elbo1 + elbo2 + elbo3 + elbo4 + elbo5 + reward_loss).backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimiser.step()
        # Store loss
        if args.use_regular_vae:
          losses.append([observation_loss.item(), reward_loss.item(), latent_kl_loss.item()])
        else:
          losses.append([elbo1.item(), elbo2.item(), elbo3.item(), elbo4.item(), elbo5.item(), reward_loss.item()])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    if args.use_regular_vae:
      metrics['observation_loss'].append(losses[0])
      metrics['reward_loss'].append(losses[1])
      metrics['latent_kl_loss'].append(losses[2])
      lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['latent_kl_loss']):], metrics['latent_kl_loss'], 'latent_kl_loss', results_dir)
    else:
      metrics['elbo1'].append(losses[0])
      metrics['elbo2'].append(losses[1])
      metrics['elbo3'].append(losses[2])
      metrics['elbo4'].append(losses[3])
      metrics['elbo5'].append(losses[4])
      metrics['reward_loss'].append(losses[5])
      lineplot(metrics['episodes'][-len(metrics['elbo1']):], metrics['elbo1'], 'elbo1', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo2']):], metrics['elbo2'], 'elbo2', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo3']):], metrics['elbo3'], 'elbo3', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo4']):], metrics['elbo4'], 'elbo4', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo5']):], metrics['elbo5'], 'elbo5', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)

  # Data collection
  if args.use_regular_vae:
    encoder.eval()
  else:
    infinite_vae.eval()
    infinite_vae.batch_size = 1
  observation, total_reward, time_step = env.reset(), 0, 0
  with torch.no_grad():
    if args.use_regular_vae:
      current_latent_mean, current_latent_std = encoder(observation.to(device=args.device))
      current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
    else:
      _, current_latent_state = infinite_vae(observation.to(device=args.device))
  episode_states = torch.zeros(args.lagging_size, args.state_size, device=args.device)
  episode_states[-1] = current_latent_state
  episode_actions = torch.zeros(args.lagging_size, env.action_size, device=args.device) + torch.tensor((env.action_range[0] + env.action_range[1]) / 2).to(device=args.device)
  episode_values = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_q_values = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_rewards = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_policy_kl = torch.zeros(args.lagging_size, 1, device=args.device)
  pbar = tqdm(range(args.max_episode_length // args.action_repeat))
  time_steps = 0
  for t in pbar:
    action, action_log_prob, value, q_value = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device)
    episode_policy_kl = torch.cat([episode_policy_kl, (-action_log_prob).sum(dim=-1, keepdim=True)], dim=0)
    observation, reward, done = env.step(action[0].cpu())
    with torch.no_grad():
      if args.use_regular_vae:
        current_latent_mean, current_latent_std = encoder(observation.to(device=args.device))
        current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
      else:
        _, current_latent_state = infinite_vae(observation.to(device=args.device))
    episode_states = torch.cat([episode_states, current_latent_state], dim=0)
    episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
    episode_values = torch.cat([episode_values, value], dim=0)
    episode_q_values = torch.cat([episode_q_values, q_value], dim=0)
    episode_rewards = torch.cat([episode_rewards, torch.Tensor([[reward]]).to(device=args.device)], dim=0)
    D.append(observation, action.detach().cpu(), reward, done)
    total_reward += reward
    
    #if (time_step % args.num_planning_steps == 0) or done:
      # compute returns in reverse order in the episode reward list
    if args.render:
      env.render()
    if done:
      pbar.close()
      break
  
  episode_actions = episode_actions[args.lagging_size:]
  episode_values = episode_values[args.lagging_size:]
  episode_q_values = episode_q_values[args.lagging_size:]
  episode_rewards = episode_rewards[args.lagging_size:]
  episode_policy_kl = episode_policy_kl[args.lagging_size:]
  episode_states = episode_states[args.lagging_size-1:]
  # transition loss for gp, adapted from dlgpd repo
  episode_length = episode_actions[args.lagging_size:].size(0)
  index_numbers = np.arange(0, episode_length-1, args.horizon_size)
  for start in index_numbers:
    if start >= episode_length-2:
      continue
    scaled_X = actor_critic_planner.transition_gp.set_data(curr_states=episode_states[start:min(start+args.horizon_size, episode_length-1)], next_states=episode_states[start+1:min(start+args.horizon_size+1, episode_length)], actions=episode_actions[start:min(start+args.horizon_size, episode_length-1)])
    with gpytorch.settings.prior_mode(True):
        outputs = actor_critic_planner.transition_gp.predict(
            scaled_X, suppress_eval_mode_warning=False
        )
        sample_outputs = torch.transpose(torch.stack([o.rsample(sample_shape=torch.Size([args.num_sample_trajectories])) for o in outputs]), 0, 2).mean(dim=-2)
        episode_state_kl = F.binary_cross_entropy(torch.sigmoid(sample_outputs), torch.sigmoid(episode_states[start+1:min(start+args.horizon_size+1, episode_length)]), reduction='mean').mean(dim=-1, keepdim=True)
        kl_transition_loss = -torch.sum(torch.stack(actor_critic_planner.transition_gp.get_mll(outputs, episode_states[start+1:min(start+args.horizon_size+1, episode_length)])))
    ##
    soft_v_values = episode_q_values[start:min(start+args.horizon_size, episode_length)] - episode_state_kl - episode_policy_kl[start:min(start+args.horizon_size, episode_length)]
    target_q_values = args.temperature_factor * episode_rewards[start:min(start+args.horizon_size-1, episode_length-1)] + args.discount_factor * episode_values[start+1:min(start+args.horizon_size, episode_length)]
    value_loss = F.mse_loss(episode_values[start:min(start+args.horizon_size, episode_length)], soft_v_values, reduction='none').mean()
    q_loss = F.mse_loss(episode_q_values[start:min(start+args.horizon_size-1, episode_length-1)], target_q_values, reduction='none').mean()
    policy_loss = (episode_policy_kl[start:min(start+args.horizon_size, episode_length)] - episode_q_values[start:min(start+args.horizon_size, episode_length)] + episode_values[start:min(start+args.horizon_size, episode_length)]).mean()
    
    planning_optimiser.zero_grad()
    # may add an action entropy
    (value_loss + q_loss + policy_loss + kl_transition_loss).backward(retain_graph=True)
    nn.utils.clip_grad_norm_(actor_critic_planner.parameters(), args.grad_clip_norm, norm_type=2)
    planning_optimiser.step()
  
  metrics['value_loss'].append(value_loss.item())
  metrics['policy_loss'].append(policy_loss.item())
  metrics['q_loss'].append(q_loss.item())
  metrics['kl_transition_loss'].append(kl_transition_loss.item())
  # Update and plot train reward metrics
  metrics['steps'].append(t + metrics['steps'][-1])
  metrics['episodes'].append(episode)
  metrics['train_rewards'].append(total_reward)
  lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)
  
  lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['q_loss']):], metrics['q_loss'], 'q_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['kl_transition_loss']):], metrics['kl_transition_loss'], 'kl_transition_loss', results_dir)

  if args.use_regular_vae:
    encoder.train()
  else:
    infinite_vae.train()

  # Test model
  if episode % args.test_interval == 0:
    # Set models to eval mode
    if args.use_regular_vae:
      observation_model.eval()
      encoder.eval()
    else:
      infinite_vae.eval()
    recurrent_gp.eval()
    actor_critic_planner.eval()
    # Initialise parallelised test environments
    test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)
    
    with torch.no_grad():
      observation, total_rewards, video_frames, time_step = test_envs.reset(), np.zeros((args.test_episodes, )), [], 0
      episode_states = torch.zeros(args.lagging_size, args.state_size, device=args.device)
      if args.use_regular_vae:
        current_latent_mean, current_latent_std = encoder(observation.to(device=args.device))
        current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
      else:
        reconstructed_observation, current_latent_state = infinite_vae(observation.to(device=args.device))
      episode_states[-1] = current_latent_state
      episode_actions = torch.zeros(args.lagging_size, env.action_size, device=args.device) + torch.tensor((env.action_range[0] + env.action_range[1]) / 2).to(device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        action, _, _, _ = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device)
        observation, reward, done = test_envs.step(action.cpu())
        if args.use_regular_vae:
          current_latent_mean, current_latent_std = encoder(observation.to(device=args.device))
          current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
        else:
          _, current_latent_state = infinite_vae(observation.to(device=args.device))
        episode_states = torch.cat([episode_states, current_latent_state], dim=0)
        episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
        total_rewards += reward.numpy()
        if not args.symbolic_env:  # Collect real vs. predicted frames for video
          if args.use_regular_vae:
            video_frames.append(make_grid(torch.cat([observation, observation_model(current_latent_state).cpu()], dim=3), nrow=5).numpy())  # Decentre
          else:
            video_frames.append(make_grid(torch.cat([observation, reconstructed_observation.permute(0, 3, 1, 2).cpu()], dim=3), nrow=5).numpy())  # Decentre
        if done.sum().item() == args.test_episodes:
          pbar.close()
          break
    
    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics['test_episodes'].append(episode)
    metrics['test_rewards'].append(total_rewards.tolist())
    lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
    if not args.symbolic_env:
      episode_str = str(episode).zfill(len(str(args.episodes)))
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Set models to train mode
    if args.use_regular_vae:
      observation_model.train()
      encoder.train()
    else:
      infinite_vae.train()
    recurrent_gp.train()
    actor_critic_planner.train()
    # Close test environments
    test_envs.close()


  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    if args.use_regular_vae:
      torch.save({
        'observation_model': observation_model.state_dict(),
        'encoder': encoder.state_dict(),
        'recurrent_gp': recurrent_gp.state_dict(),
        'optimiser': optimiser.state_dict(),
        'actor_critic_planner': actor_critic_planner.state_dict(),
        'planning_optimiser': planning_optimiser.state_dict(),
        }, os.path.join(results_dir, 'models_%d.pth' % episode))
    else:
      torch.save({
        'infinite_vae': infinite_vae.state_dict(),
        'recurrent_gp': recurrent_gp.state_dict(),
        'optimiser': optimiser.state_dict(),
        'actor_critic_planner': actor_critic_planner.state_dict(),
        'planning_optimiser': planning_optimiser.state_dict(),
        }, os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
