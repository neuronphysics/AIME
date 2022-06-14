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
from models import bottle, bottle_two_output, Encoder, ObservationModel, RecurrentGP
from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
from planner import ActorCriticPlanner
from utils import lineplot, write_video, AdaBound
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
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=10000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=100, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=8, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=8, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
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
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')

## extra hyperparameters for new model
parser.add_argument('--horizon-size', type=int, default=5, metavar='Ho', help='Horizon size')
parser.add_argument('--lagging-size', type=int, default=2, metavar='La', help='Lagging size')
parser.add_argument('--non-cumulative-reward', action='store_true', help='Model non-cumulative rewards')
parser.add_argument('--num-sample-trajectories', type=int, default=10, metavar='nst', help='number of trajectories sample in the imagination part')
parser.add_argument('--temperature-factor', type=float, default=1, metavar='Temp', help='Temperature factor')
parser.add_argument('--discount-factor', type=float, default=0.999, metavar='Temp', help='Discount factor')
parser.add_argument('--num-mixtures', type=int, default=25, metavar='Mix', help='Number of Gaussian mixtures used in the infinite VAE')
parser.add_argument('--w-dim', type=int, default=50, metavar='w', help='dimension of w')
parser.add_argument('--hidden-size', type=int, default=20, metavar='H', help='Hidden size')
parser.add_argument('--state-size', type=int, default=20, metavar='Z', help='State/latent size')
parser.add_argument('--include-elbo2', action='store_true', help='include elbo 2 loss')
parser.add_argument('--use-regular-vae', action='store_true', help='use vae that uses single Gaussian mixture')
parser.add_argument('--use-ada-bound', action='store_true', help='use AdaBound as the optimizer')
parser.add_argument('--rgp-training-interval-ratio', type=float, default=1.1, metavar='In', help='RGP training interval ratio')
parser.add_argument('--num-gp-likelihood-samples', type=int, default=50, metavar='GP', help='Number of likelihood samples for GP')

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
           'transition_imagine_loss':[], 'controller_imagine_loss':[], 'reward_loss': [], 'value_loss': [], 'policy_loss': [], 'q_loss': [], 
           'policy_mll_loss': [], 'transition_mll_loss': [], 'elbo1': [], 'elbo2': [], 'elbo3': [], 'elbo4': [], 'elbo5': []}

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
      t = t + 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)


recurrent_gp = RecurrentGP(args.horizon_size, args.state_size, env.action_size, args.lagging_size, args.device).to(device=args.device)

if args.use_regular_vae:
  observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
  encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.state_size, args.activation_function).to(device=args.device)
  param_list = list(observation_model.parameters()) + list(encoder.parameters()) + list(recurrent_gp.parameters()) 
else:
  hyperParams = {"batch_size": args.batch_size * args.chunk_size,
               "input_d": 1,
               "prior_alpha": 7., #gamma_alpha
               "prior_beta": 1., #gamma_beta
               "K": 25,
               "hidden_d": args.hidden_size,
               "latent_d": args.state_size,
               "latent_w": args.w_dim,
               "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
               "LEARNING_RATE": 1e-4,
               "CRITIC_ITERATIONS" : 5
               }
  infinite_vae = InfGaussMMVAE(hyperParams, hyperParams["K"], 3, hyperParams["latent_d"], hyperParams["latent_w"], hyperParams["hidden_d"], args.device, 64, hyperParams["batch_size"],include_elbo2=True).to(device=args.device)
  
  vae_encoder, vae_decoder, discriminator = infinite_vae.encoder, infinite_vae.decoder, VAECritic(infinite_vae.z_dim)

  enc_optim = torch.optim.Adam(vae_encoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
  dec_optim = torch.optim.Adam(vae_decoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
  dis_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.5 * hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))

  enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size = 30, gamma = 0.5)
  dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size = 30, gamma = 0.5)
  dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size = 30, gamma = 0.5)
  #optimize transition function, controller and reward functions
  ##should we optimize these transition function and controller like the following?
  param_list    = list(recurrent_gp.parameters())
  

optimiser = AdaBound(param_list, lr=0.0001) if args.use_ada_bound else optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)

actor_critic_planner = ActorCriticPlanner(args.lagging_size, args.state_size, env.action_size, recurrent_gp, env.action_range[0], env.action_range[1], args.num_sample_trajectories, args.hidden_size, args.num_gp_likelihood_samples, args.device).to(device=args.device)
planning_optimiser = optim.Adam(actor_critic_planner.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  if args.use_regular_vae:
    observation_model.load_state_dict(model_dicts['observation_model'])
    encoder.load_state_dict(model_dicts['encoder'])
  else:
    infinite_vae.load_state_dict(model_dicts['infinite_vae'])
  recurrent_gp.load_state_dict(model_dicts['recurrent_gp'])
  #save the controller & transition in the above? 
  optimiser.load_state_dict(model_dicts['optimiser'])
  actor_critic_planner.load_state_dict(model_dicts['actor_critic_planner'])
  planning_optimiser.load_state_dict(model_dicts['planning_optimiser'])

global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence

reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.likelihood, recurrent_gp, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size)))
#### ELBO objective for the transition and controller GP ### ???? need to be fixed
transition_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.transition_module.likelihood, recurrent_gp.transition_module, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size), beta=1.0))#what X=action+latent_space
#transition_mll = DeepApproximateMLL(PredictiveLogLikelihood(recurrent_gp.transition_modules.likelihood, recurrent_gp.transition_modules, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size), beta=1.0))
controller_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.policy_module.likelihood, recurrent_gp.policy_module,args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size), beta=1.0))#latent_space
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
      with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples): # to do: make this a hyperparameter as well
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
        if args.use_regular_vae:
          latent_mean, latent_std = bottle_two_output(encoder, (observations, ))
          latent_dist = Normal(latent_mean, latent_std)
          latent_kl_loss = kl_divergence(latent_dist, global_prior).sum(dim=2).mean(dim=(0, 1))
          latent_states = latent_dist.rsample()
        else:
          grad_clip = 1.0
          original_shape = observations.shape
          observations = observations.view(-1, original_shape[-3], original_shape[-2], original_shape[-1])
          for _ in range(hyperParams["CRITIC_ITERATIONS"]):
            X_recons_linear, mu_z, logvar_z, _, _, _, _, _, _, _, _, _ = infinite_vae(observations)
            z_fake = infinite_vae.encoder.reparameterize(mu_z, logvar_z)
            reconstruct_latent_components = infinite_vae.get_component_samples(hyperParams["batch_size"])
            
            critic_real = discriminator(reconstruct_latent_components).reshape(-1)
            critic_fake = discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(discriminator, reconstruct_latent_components, z_fake, device=args.device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + hyperParams["LAMBDA_GP"] * gp
            )
            discriminator.zero_grad()
            loss_critic.backward(retain_graph=True)
            dis_optim.step()
          gen_fake = discriminator(z_fake).reshape(-1)
        
          loss_dict = infinite_vae.get_ELBO(observations)
          loss_dict["wasserstein_loss"] =  -torch.mean(gen_fake)
          vae_encoder.zero_grad()
          vae_decoder.zero_grad()
          
          loss_dict["WAE-GP"]=loss_dict["loss"]+loss_dict["wasserstein_loss"]

          loss_dict["WAE-GP"].backward()
          
          torch.nn.utils.clip_grad_norm_(infinite_vae.parameters(), grad_clip)

          enc_optim.step()
          dec_optim.step()
          
          latent_states = torch.reshape(infinite_vae.get_latent_states(observations), (original_shape[0], original_shape[1], args.state_size))

        init_states = latent_states[1:-args.horizon_size].unfold(0, args.lagging_size, 1)
        predicted_rewards = recurrent_gp(init_states, actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1))
        if (not args.non_cumulative_reward):
          true_rewards = rewards[args.lagging_size:].unfold(0, args.horizon_size+1, 1).sum(dim=-1)
        else:
          true_rewards = rewards[args.lagging_size+args.horizon_size:]
        reward_loss = -reward_mll(predicted_rewards, true_rewards).mean()
        ############### add controller & policy losses #################
        ######## Need to be fixed in terms of inputs and outputs #######
        init_transition  = torch.cat([init_states, actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1)], dim=-1)
        print(f"size of the input for the transition function {init_transition.size()}")
        predicted_latent = recurrent_gp.transition_module(init_transition)#input:action+latent_space ??
        predicted_action = recurrent_gp.policy_module(init_states)#input:latent_space
        true_latent      = latent_states[args.lagging_size:]
        print(f"size of the output for the transition function {true_latent.size()}")
        true_action      = actions[args.lagging_size:]
        transition_imagine_loss = -transition_mll(predicted_latent, true_latent)##??fix
        
        controller_imagine_loss = -controller_mll(predicted_action, true_action)##??fix
        
        ##################
        if args.use_regular_vae:
          observation_loss = F.mse_loss(bottle(observation_model, (latent_states,)), observations, reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
          for group in optimiser.param_groups:
            group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
        # Update model parameters 
        optimiser.zero_grad()
        #compute losses
        if args.use_regular_vae:
          (observation_loss + reward_loss + latent_kl_loss + transition_imagine_loss + controller_imagine_loss).backward()
        else:
          (reward_loss + transition_imagine_loss +  controller_imagine_loss).backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimiser.step()
        # Store loss
        if args.use_regular_vae:
          losses.append([observation_loss.item(), reward_loss.item(), latent_kl_loss.item(), transition_imagine_loss.item(), controller_imagine_loss.item() ])
        else:
          losses.append([0, 0, 0, 0, 0, reward_loss.item(), transition_imagine_loss.item(), controller_imagine_loss.item()])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    if args.use_regular_vae:
      metrics['observation_loss'].append(losses[0])
      metrics['reward_loss'].append(losses[1])
      metrics['latent_kl_loss'].append(losses[2])
      metrics['imaginative_transition_loss'].append(losses[3])
      metrics['imaginative_controller_loss'].append(losses[4])
      lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['latent_kl_loss']):], metrics['latent_kl_loss'], 'latent_kl_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['imaginative_transition_loss']):], metrics['imaginative_transition_loss'], 'imaginative_transition_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['imaginative_controller_loss']):], metrics['imaginative_controller_loss'], 'imaginative_controller_loss', results_dir)
    else:
      metrics['elbo1'].append(losses[0])
      metrics['elbo2'].append(losses[1])
      metrics['elbo3'].append(losses[2])
      metrics['elbo4'].append(losses[3])
      metrics['elbo5'].append(losses[4])
      metrics['reward_loss'].append(losses[5])
      metrics['imaginative_transition_loss'].append(losses[6])
      metrics['imaginative_controller_loss'].append(losses[7])
      lineplot(metrics['episodes'][-len(metrics['elbo1']):], metrics['elbo1'], 'elbo1', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo2']):], metrics['elbo2'], 'elbo2', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo3']):], metrics['elbo3'], 'elbo3', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo4']):], metrics['elbo4'], 'elbo4', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo5']):], metrics['elbo5'], 'elbo5', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['imaginative_transition_loss']):], metrics['imaginative_transition_loss'], 'imaginative_transition_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['imaginative_controller_loss']):], metrics['imaginative_controller_loss'], 'imaginative_controller_loss', results_dir)

  # Data collection
  if args.use_regular_vae:
    encoder.eval()
  else:
    infinite_vae.eval()
    infinite_vae.batch_size = 1
  observation, total_reward, time_step = env.reset(), 0, 0
  with torch.no_grad():
    if args.use_regular_vae:
      current_latent_mean, current_latent_std = encoder(observation.unsqueeze(dim=0).to(device=args.device))
      current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
    else:
      _, current_latent_state = infinite_vae(observation.unsqueeze(dim=0).to(device=args.device))
  episode_states = torch.zeros(args.lagging_size, args.state_size, device=args.device)
  episode_states[-1] = current_latent_state
  episode_actions = torch.zeros(args.lagging_size, env.action_size, device=args.device) + torch.tensor((env.action_range[0] + env.action_range[1]) / 2).to(device=args.device)
  episode_values = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_q_values = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_rewards = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_policy_kl = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_policy_mll_loss = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_transition_kl = torch.zeros(args.lagging_size, 1, device=args.device)
  episode_transition_mll_loss = torch.zeros(args.lagging_size, 1, device=args.device)
  pbar = tqdm(range(args.max_episode_length // args.action_repeat))
  time_steps = 0
  with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples):
    for t in pbar:
      action, action_log_prob, policy_mll_loss, value, q_value, transition_dist = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device)
      episode_policy_kl = torch.cat([episode_policy_kl, (-action_log_prob).unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
      episode_policy_mll_loss = torch.cat([episode_policy_mll_loss, policy_mll_loss.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
      observation, reward, done = env.step(action[0].cpu())
      with torch.no_grad():
        if args.use_regular_vae:
          current_latent_mean, current_latent_std = encoder(observation.unsqueeze(dim=0).to(device=args.device))
          current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
        else:
          _, current_latent_state = infinite_vae(observation.unsqueeze(dim=0).to(device=args.device))
      transition_kl = -transition_dist.log_prob(current_latent_state)
      transition_mll_loss = -actor_critic_planner.transition_mll(transition_dist, current_latent_state)
      episode_transition_kl = torch.cat([episode_transition_kl, transition_kl.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
      episode_transition_mll_loss = torch.cat([episode_transition_mll_loss, transition_mll_loss.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
      episode_states = torch.cat([episode_states, current_latent_state], dim=0)
      episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
      episode_values = torch.cat([episode_values, value], dim=0)
      episode_q_values = torch.cat([episode_q_values, q_value], dim=0)
      episode_rewards = torch.cat([episode_rewards, torch.Tensor([[reward]]).to(device=args.device)], dim=0)
      D.append(observation, action.detach().cpu(), reward, done)
      total_reward = total_reward + reward
      
      if args.render:
        env.render()
      if done:
        pbar.close()
        break

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
  nn.utils.clip_grad_norm_(actor_critic_planner.parameters(), args.grad_clip_norm, norm_type=2)
  planning_optimiser.step()
  
  metrics['value_loss'].append(value_loss.item())
  metrics['policy_loss'].append(policy_loss.item())
  metrics['q_loss'].append(q_loss.item())
  metrics['policy_mll_loss'].append(current_policy_mll_loss.item())
  metrics['transition_mll_loss'].append(current_transition_mll_loss.item())
  # Update and plot train reward metrics
  metrics['steps'].append(t + metrics['steps'][-1])
  metrics['episodes'].append(episode)
  metrics['train_rewards'].append(total_reward)
  lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)
  lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['q_loss']):], metrics['q_loss'], 'q_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['policy_mll_loss']):], metrics['policy_mll_loss'], 'policy_mll_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['transition_mll_loss']):], metrics['transition_mll_loss'], 'transition_mll_loss', results_dir)

  del episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, episode_policy_kl, episode_policy_mll_loss, episode_transition_kl, episode_transition_mll_loss

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
        current_latent_mean, current_latent_std = encoder(observation.unsqueeze(dim=0).to(device=args.device))
        current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
      else:
        reconstructed_observation, current_latent_state = infinite_vae(observation.unsqueeze(dim=0).to(device=args.device))
      episode_states[-1] = current_latent_state
      episode_actions = torch.zeros(args.lagging_size, env.action_size, device=args.device) + torch.tensor((env.action_range[0] + env.action_range[1]) / 2).to(device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples):
          action, _, _, _, _, _ = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device)
        observation, reward, done = test_envs.step(action.cpu())
        if args.use_regular_vae:
          current_latent_mean, current_latent_std = encoder(observation.unsqueeze(dim=0).to(device=args.device))
          current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
        else:
          _, current_latent_state = infinite_vae(observation.unsqueeze(dim=0).to(device=args.device))
        episode_states = torch.cat([episode_states, current_latent_state], dim=0)
        episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
        total_rewards = total_reward + reward.numpy()
        if not args.symbolic_env:  # Collect real vs. predicted frames for video
          if args.use_regular_vae:
            video_frames.append(make_grid(observation, nrow=5).numpy())  # Decentre
          else:
            video_frames.append(make_grid(observation, nrow=5).numpy())  # Decentre
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

    del episode_states, episode_actions

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
