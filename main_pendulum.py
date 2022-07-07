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
#from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
#from infGaussianVAE import InfGaussMMVAE, VAECritic, gradient_penalty
from planner import ActorCriticPlanner
from utils import lineplot, write_video, AdaBound
from recurrent_gp import RecurrentGP
from main_utils import *
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, PredictiveLogLikelihood
# Hyperparameters
parser = argparse.ArgumentParser(description='AIME')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['4lane', 'loop'], help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=10000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=100, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--initial-collect-interval', type=int, default=10000, metavar='C', help='First collect interval')
parser.add_argument('--batch-size', type=int, default=8, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=8, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value') 

parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=1, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--norm-reward', action='store_true', help='nomormalize reward')

## extra hyperparameters for new model
parser.add_argument('--horizon-size', type=int, default=3, metavar='Ho', help='Horizon size')
parser.add_argument('--lagging-size', type=int, default=2, metavar='La', help='Lagging size')
parser.add_argument('--non-cumulative-reward', action='store_true', help='Model non-cumulative rewards')
parser.add_argument('--num-sample-trajectories', type=int, default=10, metavar='nst', help='number of trajectories sample in the imagination part')
parser.add_argument('--temperature-factor', type=float, default=1, metavar='Temp', help='Temperature factor')
parser.add_argument('--discount-factor', type=float, default=0.999, metavar='Temp', help='Discount factor')
parser.add_argument('--hidden-size', type=int, default=10, metavar='H', help='Hidden size')
parser.add_argument('--state-size', type=int, default=3, metavar='Z', help='State/latent size')
parser.add_argument('--use-ada-bound', action='store_true', help='use AdaBound as the optimizer')
parser.add_argument('--rgp-training-interval-ratio', type=float, default=1.1, metavar='In', help='RGP training interval ratio')
parser.add_argument('--num-gp-likelihood-samples', type=int, default=1, metavar='GP', help='Number of likelihood samples for GP')
parser.add_argument('--num-inducing-recurrent-gp', type=int, default=16, metavar='GP', help='Number of inducing points for DGPHiddenLayer in Recurrent GP')
parser.add_argument('--num-inducing-planner', type=int, default=16, metavar='GP', help='Number of inducing points for DGPHiddenLayer in Planner')
parser.add_argument('--result-dir', type=str, default="results", help='result directory')
parser.add_argument('--lineplot', action='store_true', help='lineplot metrics')
parser.add_argument('--imaginary-rollout-softplus', action='store_true', help='add a softplus operation on imaginary_rollout of planner')
parser.add_argument('--input-size', type=int, default=3, metavar='InpS', help='observation input size')

args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

# return the latent state through encoder
def get_latent(observation):
  return observation.to(device=args.device)

# test the agent
def test(episode, metrics): 
  print("Testing at episode", episode)
  recurrent_gp.eval()
  actor_critic_planner.eval()
  # Initialise parallelised test environments
  test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.norm_reward, args.input_size), {}, args.test_episodes)
  
  with torch.no_grad():
    observation, total_rewards, video_frames, time_step = test_envs.reset(), np.zeros((args.test_episodes, )), [], 0
    episode_states = torch.zeros(args.lagging_size, args.state_size, device=args.device)
    current_latent_state = get_latent(observation)
    episode_states[-1] = current_latent_state
    episode_actions = torch.zeros(args.lagging_size, env.action_size, device=args.device) + torch.tensor((env.action_range[0] + env.action_range[1]) / 2).to(device=args.device)
    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    true_rewards = []
    imagined_rewards = []
    for t in pbar:
      with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples):
        action, _, _, _, _, _, imagined_reward = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device, softplus=args.imaginary_rollout_softplus)
      observation, reward, done = test_envs.step(action.cpu())
      current_latent_state = get_latent(observation)
      episode_states = torch.cat([episode_states, current_latent_state], dim=0)
      episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
      total_rewards = total_reward + reward.numpy()
      imagined_rewards.append(imagined_reward)
      true_rewards.append(reward)
      if done.sum().item() == args.test_episodes:
        pbar.close()
        break
  
  # Update and plot reward metrics (and write video if applicable) and save metrics
  metrics['test_episodes'].append(episode)
  metrics['test_rewards'].append(total_rewards.tolist())
  lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
  lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
  print("total test rewards", total_rewards)
  print("imagined_rewards", imagined_rewards)
  print("true_rewards", true_rewards)
  torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

  del episode_states, episode_actions

  # Set models to train mode
  recurrent_gp.train()
  actor_critic_planner.train()
  # Close test environments
  test_envs.close()

def rgp_step(n_iter, losses):
    recurrent_gp.train()
    policy.train()
    gp_pbar = tqdm(range(n_iter))
    for s in gp_pbar:
      with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples): # to do: make this a hyperparameter as well
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)    # Transitions start at time t = 0
        latent_states = observations
        # print("raw rewards", rewards)
        # print("actions", actions)
        # latent_states : <chunk_size, batch_size, z_dim>
        init_states = latent_states[1:-args.horizon_size].unfold(0, args.lagging_size, 1)
        # init_states: <chunk_size - horizon_size - lagging_size, batch_size, latent_size, lagging_size>
        init_actions = actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1)
        # init_actions: <chunk_size - horizon_size - lagging_size, batch_size, action_size, lagging_size> 

        # without transpose, the result will be [t0, t1, t2, t0, t1, t2,...]
        # We want [t0, t0, t1, t1, t2, t2, ...] so the lagging_actions[..., self.action_size:] works as expected
        init_states = torch.transpose(init_states, -2, -1)
        # init_states: <chunk_size - horizon_size - lagging_size, batch_size, lagging_size, latent_size>
        init_actions = torch.transpose(init_actions, -2, -1)
        # init_actions: <chunk_size - horizon_size - lagging_size, batch_size, lagging_size, action_size> 

        # true_rewards size: <chunk_size - horizon_size - lagging_size, batch_size>
        if (not args.non_cumulative_reward):
          true_rewards = rewards[args.lagging_size:].unfold(0, args.horizon_size+1, 1).sum(dim=-1)
        else:
          true_rewards = rewards[args.lagging_size+args.horizon_size:]
        # true_latent size: <chunk_size - lagging_size, batch_size, latent_size>
        # true_action size: <chunk_size - lagging_size, batch_size, action_size>
        true_rewards = true_rewards.reshape((true_rewards.size(0) * true_rewards.size(1)))
        #print("true_rewards", true_rewards, true_rewards.shape)
        true_action = actions[args.lagging_size-1:-1].unfold(0, 1, 1)
        true_action = true_action.reshape(true_action.size(0), true_action.size(1), -1)
        policy_input = latent_states[:-1].unfold(0, args.lagging_size, 1)
        policy_input = policy_input.reshape(policy_input.size(0), policy_input.size(1), -1)
        true_latent = latent_states[args.lagging_size:].unfold(0, 1, 1)
        true_latent = true_latent.reshape(true_latent.size(0), true_latent.size(1), -1)
        true_latent = true_latent.reshape((true_latent.size(0) * true_latent.size(1), -1))
        actions_input = actions[:-1].unfold(0, args.lagging_size, 1)
        actions_input = actions_input.reshape(actions_input.size(0), actions_input.size(1), -1)
        transition_input = torch.cat([policy_input, actions_input], dim=-1)
        transition_input = transition_input.reshape((transition_input.size(0) * transition_input.size(1), -1))
        true_rewards = rewards[:-1]
        true_rewards = true_rewards.reshape(true_rewards.size(0) * true_rewards.size(1))

        optimizer.zero_grad()
        policy.zero_grad()
        #predicted_rewards, _, _ = recurrent_gp(init_states, init_actions, policy)
        predicted_states = recurrent_gp.transition_module(transition_input)
        predicted_rewards = recurrent_gp.reward_module(transition_input)
        
        loss_reward = -reward_mll(predicted_rewards, true_rewards)#.mean()
        loss_transition = -transition_mll(predicted_states, true_latent)#.mean()
        recurrent_gaussian_loss = loss_reward + loss_transition
        recurrent_gaussian_loss.backward()

        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
          for group in optimiser.param_groups:
            group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
        
        # Update model parameters 
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimizer.step()
        gp_pbar.set_postfix(rgpLoss=recurrent_gaussian_loss.item(), rLoss=loss_reward.item(), TtLoss=loss_transition.item())
        losses.append([loss_reward.item(), loss_transition.item(), 0])
        del true_latent, true_action, predicted_rewards, predicted_states, init_states, latent_states, init_actions, actions, observations
    return losses

def test_recurrent_gp(n_test):
  recurrent_gp.eval()
  policy.eval()
  rmse_rewards_gp = 0
  rmse_state = 0
  rmse_rewards = 0
  for s in range(n_test):
    with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples): # to do: make this a hyperparameter as well
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
      observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
      latent_states = observations
      # latent_states : <chunk_size, batch_size, z_dim>
      init_states = latent_states[1:-args.horizon_size].unfold(0, args.lagging_size, 1)
      # init_states: <chunk_size - horizon_size - lagging_size, batch_size, latent_size, lagging_size>
      init_actions = actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1)
      # init_actions: <chunk_size - horizon_size - lagging_size, batch_size, action_size, lagging_size> 

      # without transpose, the result will be [t0, t1, t2, t0, t1, t2,...]
      # We want [t0, t0, t1, t1, t2, t2, ...] so the lagging_actions[..., self.action_size:] works as expected
      init_states = torch.transpose(init_states, -2, -1)
      # init_states: <chunk_size - horizon_size - lagging_size, batch_size, lagging_size, latent_size>
      init_actions = torch.transpose(init_actions, -2, -1)
      # init_actions: <chunk_size - horizon_size - lagging_size, batch_size, lagging_size, action_size> 
      # print("init_states", init_states.shape, init_states)
      # print("init_actions", init_actions.shape, init_actions)
    
      true_rewards_gp = rewards[args.lagging_size+args.horizon_size:]
      #print("true_rewards", true_rewards, true_rewards.shape)
      true_rewards_gp = true_rewards_gp.reshape((true_rewards_gp.size(0) * true_rewards_gp.size(1)))
      #print("true_rewards", true_rewards, true_rewards.shape)
      true_action = actions[args.lagging_size-1:-1].unfold(0, 1, 1)
      true_action = true_action.reshape(true_action.size(0), true_action.size(1), -1)
      policy_input = latent_states[:-1].unfold(0, args.lagging_size, 1)
      policy_input = policy_input.reshape(policy_input.size(0), policy_input.size(1), -1)
      true_latent = latent_states[args.lagging_size:].unfold(0, 1, 1)
      true_latent = true_latent.reshape(true_latent.size(0), true_latent.size(1), -1)
      true_latent = true_latent.reshape((true_latent.size(0) * true_latent.size(1), -1))
      actions_input = actions[:-1].unfold(0, args.lagging_size, 1)
      actions_input = actions_input.reshape(actions_input.size(0), actions_input.size(1), -1)
      transition_input = torch.cat([policy_input, actions_input], dim=-1)
      transition_input = transition_input.reshape((transition_input.size(0) * transition_input.size(1), -1))

      true_rewards = rewards[:-1]
      true_rewards = true_rewards.reshape(true_rewards.size(0) * true_rewards.size(1))

      # print(transition_input)
      # print(observations)
      # print(rewards)
      # print(true_rewards)


      predicted_rewards_gp = recurrent_gp.predict_recurrent_gp(init_states, init_actions, policy).mean(0)
      predicted_states = recurrent_gp.predict_transition(transition_input).mean(0)
      predicted_rewards = recurrent_gp.predict_reward(transition_input).mean(0)
      # print("predicted_rewards_gp", predicted_rewards_gp.shape, true_rewards.shape)
      # print("predicted_states", predicted_states.shape, true_latent.shape)
      # print("predicted_rewards", predicted_rewards.shape, true_rewards.shape)
      # print()

      rmse_rewards_gp += torch.mean(torch.pow(predicted_rewards_gp - true_rewards_gp, 2)).sqrt()
      rmse_state += torch.mean(torch.pow(predicted_states - true_latent, 2)).sqrt()
      rmse_rewards += torch.mean(torch.pow(predicted_rewards - true_rewards, 2)).sqrt()

  print(f"RMSE rmse_rewards_gp: {rmse_rewards_gp.item() / n_test}")
  print(f"RMSE rmse_state: {rmse_state.item() / n_test}")
  print(f"RMSE rmse_rewards: {rmse_rewards.item() / n_test}")
  recurrent_gp.train()
  policy.train()

###### Setup
results_dir = os.path.join(args.result_dir, args.id)
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda:0')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'latent_kl_loss': [],
           'transition_imagine_loss':[], 'controller_imagine_loss':[], 'reward_loss': [], 'value_loss': [], 'policy_loss': [], 'q_loss': [], 
           'policy_mll_loss': [], 'transition_mll_loss': [], 'elbo1': [], 'elbo2': [], 'elbo3': [], 'elbo4': [], 'elbo5': []}

###### Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.norm_reward, args.input_size)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
  D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device, args.input_size)
  # Initialise dataset D with S random seed episodes
  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    rewards = []
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      D.append(observation, action, reward, done)
      observation = next_observation
      rewards.append(reward)
      t = t + 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)

# init recurrent GP and VAE models
#recurrent_gp = RecurrentGP(args.horizon_size, args.state_size, env.action_size, args.lagging_size, args.num_inducing_recurrent_gp, args.device).to(device=args.device)
recurrent_gp = RecurrentGP(args.horizon_size, args.state_size, env.action_size, args.lagging_size, args.num_inducing_recurrent_gp, args.device).to(device=args.device)
param_list = list(recurrent_gp.parameters()) 
optimizer = torch.optim.Adam(recurrent_gp.parameters(), lr=0.002)
#optimiser = AdaBound(param_list, lr=0.0001) if args.use_ada_bound else optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
actor_critic_planner = ActorCriticPlanner(args.lagging_size, args.state_size, env.action_size, recurrent_gp, env.action_range[0], env.action_range[1], args.num_sample_trajectories, args.hidden_size, args.num_gp_likelihood_samples, args.num_inducing_planner, args.device).to(device=args.device)
policy = actor_critic_planner.actor
# set up optimizers
planning_optimiser = optim.Adam(actor_critic_planner.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  recurrent_gp.load_state_dict(model_dicts['recurrent_gp'])
  #save the controller & transition in the above? 
  optimiser.load_state_dict(model_dicts['optimiser'])
  actor_critic_planner.load_state_dict(model_dicts['actor_critic_planner'])
  planning_optimiser.load_state_dict(model_dicts['planning_optimiser'])

#reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.likelihood, recurrent_gp, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size)))
#### ELBO objective for the transition and controller GP ### ???? need to be fixed
#transition_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.transition_module.likelihood, recurrent_gp.transition_module, args.batch_size*(args.chunk_size-args.lagging_size), beta=1.0))#what X=action+latent_space
#transition_mll = DeepApproximateMLL(PredictiveLogLikelihood(recurrent_gp.transition_modules.likelihood, recurrent_gp.transition_modules, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size), beta=1.0)) #typically produces better predictive variances than the ELBO
#controller_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.policy_module.likelihood, recurrent_gp.policy_module,args.batch_size*(args.chunk_size-args.lagging_size), beta=1.0))#latent_space
rgp_training_episode = args.seed_episodes

reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.reward_module.likelihood, recurrent_gp.reward_module, args.experience_size))
transition_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.transition_module.likelihood, recurrent_gp.transition_module, args.experience_size))

find_model_size(recurrent_gp)

print("Test Recurrent GP before Pretraining:")
test_recurrent_gp(1000)
rgp_step(args.initial_collect_interval, [])
print("Test Recurrent GP after Pretraining:")
test_recurrent_gp(1000)
###### Training (and testing)

for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []

  # training the observation models and recurrent GP 
  if ((episode-1) == rgp_training_episode):
    rgp_training_episode = int(rgp_training_episode * args.rgp_training_interval_ratio)
    print("Test Recurrent GP before rgp_training_episode:")
    test_recurrent_gp(1000)
    losses = rgp_step(args.collect_interval, losses)
    print("Test Recurrent GP after rgp_training_episode:")
    test_recurrent_gp(1000)

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics = update_gp_loss_metric(args, metrics, losses, results_dir)

  # Data collection and planning
  recurrent_gp.eval()
  observation, total_reward, time_step = env.reset(), 0, 0
  with torch.no_grad():
    current_latent_state = get_latent(observation)

  (
    episode_states,
    episode_actions,
    episode_values,
    episode_q_values,
    episode_rewards,
    episode_policy_kl,
    episode_policy_mll_loss,
    episode_transition_kl,
    episode_transition_mll_loss
  ) = init_planner_states(args, current_latent_state, env)
  imagined_rewards = []
  true_rewards = []

  pbar = tqdm(range(args.max_episode_length // args.action_repeat)) # max_episode_length = 1000
  time_steps = 0
  with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples):
    for t in pbar:
      (
        action,
        action_log_prob,
        policy_mll_loss,
        value, q_value,
        transition_dist,
        imagined_reward
      ) = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device, softplus=args.imaginary_rollout_softplus)
      
      observation, reward, done = env.step(action[0].cpu())

      imagined_rewards.append(imagined_reward)
      true_rewards.append(reward)
      
      with torch.no_grad():
        current_latent_state = get_latent(observation)
      (
        episode_policy_kl,
        episode_policy_mll_loss,
        transition_kl,
        transition_mll_loss,
        episode_transition_kl,
        episode_transition_mll_loss,
        episode_states,
        episode_actions,
        episode_values,
        episode_q_values,
        episode_rewards
      ) = update_planner_states(transition_dist, current_latent_state, actor_critic_planner, action, value, q_value, reward,
                                episode_policy_kl, action_log_prob, episode_policy_mll_loss, policy_mll_loss, episode_transition_kl, 
                                episode_transition_mll_loss, episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, args)

      D.append(observation, action.detach().cpu(), reward, done)
      total_reward = total_reward + reward
      if args.render:
        env.render()
      if done:
        pbar.close()
        break
  
  ### Compute loss and backward-prop
  current_q_values = episode_q_values
  previous_q_values = episode_q_values
  current_rewards = episode_rewards
  current_policy_kl = episode_policy_kl
  current_transition_kl = episode_transition_kl
  current_values = episode_values
  previous_values = episode_values
  soft_v_values = current_q_values - current_transition_kl - current_policy_kl
  target_q_values = args.temperature_factor * current_rewards + args.discount_factor * current_values
  value_loss = F.mse_loss(previous_values, soft_v_values, reduction='none').mean()
  q_loss = F.mse_loss(previous_q_values, target_q_values, reduction='none').mean()
  policy_loss = (current_policy_kl - current_q_values + previous_values).mean()
  
  current_policy_mll_loss = episode_policy_mll_loss.mean()
  current_transition_mll_loss = episode_transition_mll_loss.mean()
  
  planning_optimiser.zero_grad()
  (value_loss + q_loss + policy_loss + current_policy_mll_loss + current_transition_mll_loss).backward()
  print("value_loss", value_loss, "q_loss", q_loss, "policy_loss", policy_loss, "current_policy_mll_loss", current_policy_mll_loss, "current_transition_mll_loss", current_transition_mll_loss)
  print("planning total loss", value_loss + q_loss + policy_loss + current_policy_mll_loss + current_transition_mll_loss)
  print("training rewards", total_reward)
  print("current_rewards", current_rewards)
  print("train observation", observation.shape, observation, "min:", torch.min(observation), "max:", torch.max(observation))
  print("train imagined_rewards", imagined_rewards)
  print("train true_rewards", true_rewards)
  nn.utils.clip_grad_norm_(actor_critic_planner.parameters(), args.grad_clip_norm, norm_type=2)
  planning_optimiser.step()

  metrics = update_plot_planning_loss_metric(metrics, value_loss, policy_loss, q_loss, current_policy_mll_loss, current_transition_mll_loss, t, episode, total_reward, results_dir, args)

  # delete unused variables
  del episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, episode_policy_kl, episode_policy_mll_loss, episode_transition_kl, episode_transition_mll_loss
  del action, action_log_prob, policy_mll_loss, value, q_value, transition_dist
  del current_q_values, previous_q_values, current_rewards, current_policy_kl, current_transition_kl, current_values, previous_values, soft_v_values, target_q_values, value_loss, q_loss, policy_loss 
  del current_policy_mll_loss, current_transition_mll_loss

  # Test model
  if episode % args.test_interval == 0:
    test(episode, metrics)

print("Done!")
# Close training environment
env.close()
