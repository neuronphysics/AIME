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
from models import bottle, bottle_two_output, Encoder, ObservationModel, RegDecoder, RegEncoder
#from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
from infGaussianVAE import InfGaussMMVAE, VAECritic, gradient_penalty
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
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=10000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=100, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--initial-collect-interval', type=int, default=1000, metavar='C', help='First collect interval')
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
parser.add_argument('--skip-vae', action='store_true', help='use observation as latent states, enable for simple symbolic envs')

## extra hyperparameters for new model
parser.add_argument('--horizon-size', type=int, default=5, metavar='Ho', help='Horizon size')
parser.add_argument('--lagging-size', type=int, default=1, metavar='La', help='Lagging size') # changed to 1!!!!!!!!!!!!!!!!!!!!!!!!!
parser.add_argument('--non-cumulative-reward', action='store_true', help='Model non-cumulative rewards')
parser.add_argument('--num-sample-trajectories', type=int, default=10, metavar='nst', help='number of trajectories sample in the imagination part')
parser.add_argument('--temperature-factor', type=float, default=1, metavar='Temp', help='Temperature factor')
parser.add_argument('--discount-factor', type=float, default=0.999, metavar='Temp', help='Discount factor')
parser.add_argument('--num-mixtures', type=int, default=10, metavar='Mix', help='Number of Gaussian mixtures used in the infinite VAE')
parser.add_argument('--w-dim', type=int, default=10, metavar='w', help='dimension of w')
parser.add_argument('--hidden-size', type=int, default=10, metavar='H', help='Hidden size')
parser.add_argument('--state-size', type=int, default=10, metavar='Z', help='State/latent size')
parser.add_argument('--include-elbo2', action='store_true', help='include elbo 2 loss')
parser.add_argument('--use-regular-vae', action='store_true', help='use vae that uses single Gaussian mixture')
parser.add_argument('--use-ada-bound', action='store_true', help='use AdaBound as the optimizer')
parser.add_argument('--rgp-training-interval-ratio', type=float, default=1.1, metavar='In', help='RGP training interval ratio')
parser.add_argument('--num-gp-likelihood-samples', type=int, default=1, metavar='GP', help='Number of likelihood samples for GP')
parser.add_argument('--num-inducing-recurrent-gp', type=int, default=16, metavar='GP', help='Number of inducing points for DGPHiddenLayer in Recurrent GP')
parser.add_argument('--num-inducing-planner', type=int, default=16, metavar='GP', help='Number of inducing points for DGPHiddenLayer in Planner')
parser.add_argument('--input-size', type=int, default=32, metavar='InpS', help='observation input size')
parser.add_argument('--train-all-layers', action='store_true', help='train all layers of infGaussianVAE, default setting will only train encoder and decoder')
parser.add_argument('--result-dir', type=str, default="results", help='result directory')
parser.add_argument('--lineplot', action='store_true', help='lineplot metrics')
parser.add_argument('--imaginary-rollout-softplus', action='store_true', help='add a softplus operation on imaginary_rollout of planner')
parser.add_argument('--warm-up-vae', type=int, default=1000, metavar='Vae', help='warm up vae for some iterations')
parser.add_argument('--recons-from-base', action='store_true', help='use base encoder and decoder in infGaussianVAE to reconstruct an input, instead of doing entire forward pass')

args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

def _backward_infGaussianVAE(l, retain_graph=False):
  if args.train_all_layers:
    infinite_vae.zero_grad()
  else:
    vae_encoder.zero_grad()
    vae_decoder.zero_grad()
    salient_dec.zero_grad()
    salient_en.zero_grad()
  l.backward(retain_graph=retain_graph)
  torch.nn.utils.clip_grad_norm_(infinite_vae.parameters(), args.grad_clip_norm, norm_type=2)

  if args.train_all_layers:
    inf_optim.step()
  else:
    enc_optim.step()
    dec_optim.step()
    salient_en.step()
    salient_dec.step()

# return the latent state through encoder
def get_latent(observation):
  if args.use_regular_vae:
    current_latent_mean, current_latent_std = encoder(observation.unsqueeze(dim=0).to(device=args.device))
    current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
  else:
    current_latent_state = infinite_vae.get_latent_states(observation.unsqueeze(dim=0).to(device=args.device))
  return current_latent_state 

# return the reconstruct observation
def get_reconstruct(observation, unsqueeze=True):
  if unsqueeze:
    observation = observation.unsqueeze(dim=0)
  with torch.no_grad():
    if args.use_regular_vae:
      current_latent_mean, current_latent_std = encoder(observation.to(device=args.device))
      current_latent_state = current_latent_mean + torch.randn_like(current_latent_mean) * current_latent_std
      reconstruct = observation_model(current_latent_state)
    else:
      if args.recons_from_base:
        latent = infinite_vae.get_latent_states(observation.to(device=args.device))
        reconstruct = infinite_vae.decoder(latent)
      else:
        reconstruct = infinite_vae(observation.to(device=args.device))[0]
    reconstruct = reconstruct.view(observation.shape)
  return reconstruct

# test the agent
def test(episode, metrics): 
  print("Testing at episode", episode)
  # Set models to eval mode
  if args.use_regular_vae:
    observation_model.eval()
    encoder.eval()
  else:
    infinite_vae.eval()
  recurrent_gp.eval()
  actor_critic_planner.eval()
  # Initialise parallelised test environments
  test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.norm_reward, args.input_size), {}, args.test_episodes) # Anudeep, new args added
  
  with torch.no_grad():
    observation, total_rewards, video_frames, time_step = test_envs.reset(), np.zeros((args.test_episodes, )), [], 0
    if not args.use_regular_vae:
      observation += 0.5  # scale to [0, 1]
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
      if not args.use_regular_vae:
        observation += 0.5  # scale to [0, 1]
      current_latent_state = get_latent(observation)
      episode_states = torch.cat([episode_states, current_latent_state], dim=0)
      episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
      total_rewards = total_reward + reward.numpy()
      imagined_rewards.append(imagined_reward)
      true_rewards.append(reward)
      if not args.symbolic_env:  # Collect real vs. predicted frames for video
        video_frames.append(make_grid(observation.cpu(), nrow=5).numpy())  # Decentre
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
    save_image(torch.as_tensor(make_grid(observation.cpu(), nrow=5).numpy()), os.path.join(results_dir, 'test_episode_gt_%s.png' % episode_str))
    recons = get_reconstruct(observation)
    save_image(torch.as_tensor(make_grid(recons.cpu(), nrow=5).numpy()), os.path.join(results_dir, 'test_episode_recons_%s.png' % episode_str))
    print("test observation", observation, observation.shape, "min:", torch.min(observation), "max:", torch.max(observation))
    print("test recons", recons, recons.shape, "min:", torch.min(recons), "max:", torch.max(recons))
  print("total test rewards", total_rewards)
  print("imagined_rewards", imagined_rewards)
  print("true_rewards", true_rewards)
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

def vae_test(step):
    t = "_reg" if args.use_regular_vae else "_inf" 
    out_dir = "main_vae_out_" + str(args.input_size) + t
    os.makedirs(out_dir, exist_ok=True)
    
    if args.use_regular_vae:
      observation_model.eval()
      encoder.eval()
    else:
      infinite_vae.eval()
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
    if not args.use_regular_vae:
      observations += 0.5  # scale to [0, 1]
    original_shape = observations.shape
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
    observations = observations.view(-1, original_shape[-3], original_shape[-2], original_shape[-1])
    print("observations size in test", observations.shape)
    reconstruct = get_reconstruct(observations, unsqueeze=False)
    print("observations", observations[0])
    print("reconstruct", reconstruct[0])

    save_image(torch.as_tensor(make_grid(observations.cpu(), nrow=args.chunk_size).numpy()), os.path.join(out_dir, 'test_gt_%s.png' % step))
    save_image(torch.as_tensor(make_grid(reconstruct.cpu(), nrow=args.chunk_size).numpy()), os.path.join(out_dir, 'test_recons_%s.png' % step))
    
    if args.use_regular_vae:
      observation_model.train()
      encoder.train()
    else:
      infinite_vae.train()

def train_vae(n_iter, losses):
    if (not args.use_regular_vae):
      infinite_vae.batch_size = args.batch_size * args.chunk_size
    gp_pbar = tqdm(range(n_iter))
    test_interval = 100
    for s in gp_pbar:
      with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples): # to do: make this a hyperparameter as well
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
        if not args.use_regular_vae:
          observations += 0.5  # scale to [0, 1]
        #print("observations vision gp", observations.shape, observations[0][0], "min:", torch.min(observations), "max:", torch.max(observations))
        # observations: <chunk_size, batch_size, 3, input_dim, input_dim)
        if args.use_regular_vae:
          observation_loss, latent_kl_loss, latent_states = get_regularVAE_loss_and_latent(observations, encoder, global_prior, observation_model, args)
        else:
          original_shape = observations.shape
          observations = observations.view(-1, original_shape[-3], original_shape[-2], original_shape[-1])
          z_fake = train_discriminator(observations, discriminator, infinite_vae, hyperParams, args, dis_optim)
          loss_dict, latent_states = get_infGaussianVAE_loss_and_latent(discriminator, z_fake, infinite_vae, observations, original_shape, args)

        #compute losses
        if args.use_regular_vae:
          (observation_loss  + latent_kl_loss).backward()
          print("observation_loss", observation_loss, "latent_kl_loss", latent_kl_loss) 
          gp_pbar.set_description("observation_loss:"+str(float(observation_loss)) + " | latent_kl_loss:"+str(float(latent_kl_loss)))
        else:
          _backward_infGaussianVAE(loss_dict["WAE-GP"])
          print("infGaussianLoss", loss_dict["WAE-GP"].item())
          print("infGaussianLoss loss_dict", loss_dict)
          losses.append([
            loss_dict["c_cluster_kld"],
            loss_dict["kumar2beta_kld"],
            loss_dict["w_context_kld"],
            loss_dict["z_latent_space_kld"],
            loss_dict["recon"]
          ])
          gp_pbar.set_description("WAE-GP:"+str(float(loss_dict["WAE-GP"].item())))
        # optimizer includes recurrent_gp and VAE is using regular VAE
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimizer.step()
        # optimiser.step()
        if s % test_interval == 0:
          vae_test(s)
    return losses
  
def rgp_step(n_iter, losses):
    if (not args.use_regular_vae):
      infinite_vae.batch_size = args.batch_size * args.chunk_size
    gp_pbar = tqdm(range(n_iter))
    for s in gp_pbar:
      with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples): # to do: make this a hyperparameter as well
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
        if not args.use_regular_vae:
          observations += 0.5  # scale to [0, 1]
        #print("observations vision gp", observations.shape, observations[0][0], "min:", torch.min(observations), "max:", torch.max(observations))
        # observations: <chunk_size, batch_size, 3, input_dim, input_dim)
        if args.use_regular_vae:
          observation_loss, latent_kl_loss, latent_states = get_regularVAE_loss_and_latent(observations, encoder, global_prior, observation_model, args)
        else:
          original_shape = observations.shape
          observations = observations.view(-1, original_shape[-3], original_shape[-2], original_shape[-1])
          z_fake = train_discriminator(observations, discriminator, infinite_vae, hyperParams, args, dis_optim)
          loss_dict, latent_states = get_infGaussianVAE_loss_and_latent(discriminator, z_fake, infinite_vae, observations, original_shape, args)

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

        # predicted_rewards = recurrent_gp(init_states, init_actions)
        
        # true_rewards size: <chunk_size - horizon_size - lagging_size, batch_size>
        if (not args.non_cumulative_reward):
          true_rewards = rewards[args.lagging_size:].unfold(0, args.horizon_size+1, 1).sum(dim=-1)
        else:
          true_rewards = rewards[args.lagging_size+args.horizon_size:]
        # true_latent size: <chunk_size - lagging_size, batch_size, latent_size>
        # true_action size: <chunk_size - lagging_size, batch_size, action_size>
        true_rewards = true_rewards.reshape((true_rewards.size(0) * true_rewards.size(1))) # Anudeep, new
        
        true_action = actions[args.lagging_size-1:-1].unfold(0, 1, 1)
        true_action = true_action.reshape(true_action.size(0), true_action.size(1), -1)
        policy_input = latent_states[:-1].unfold(0, args.lagging_size, 1)
        policy_input = policy_input.reshape(policy_input.size(0), policy_input.size(1), -1)

        true_latent = latent_states[args.lagging_size:].unfold(0, 1, 1)
        true_latent = true_latent.reshape(true_latent.size(0), true_latent.size(1), -1)
        true_latent = true_latent.reshape((true_latent.size(0) * true_latent.size(1), -1)) # Anudeep, new
        actions_input = actions[:-1].unfold(0, args.lagging_size, 1)
        actions_input = actions_input.reshape(actions_input.size(0), actions_input.size(1), -1)
        print(f'policy_input shape: {policy_input.shape}')
        print(f'actions_input shape: {actions_input.shape}')
        transition_input = torch.cat([policy_input, actions_input], dim=-1)
        transition_input = transition_input.reshape((transition_input.size(0) * transition_input.size(1), -1)) # Anudeep, new
        
        true_rewards = rewards[:-1]
        true_rewards = true_rewards.reshape(true_rewards.size(0) * true_rewards.size(1))
        
        optimizer.zero_grad() #  Anudeep, new
        policy.zero_grad()

        # predicted_action = recurrent_gp.policy_module(policy_input)
        # predicted_action = policy(policy_input)
        print(f'transition_input shape: {transition_input.shape}')
        predicted_latent = recurrent_gp.transition_module(transition_input)
        predicted_rewards = recurrent_gp.reward_module(transition_input)

        reward_loss = -reward_mll(predicted_rewards, true_rewards).mean()
        transition_imagine_loss = -transition_mll(predicted_latent, true_latent).mean()
        # controller_imagine_loss = -controller_mll(predicted_action, true_action).mean()
        print("true_rewards", true_rewards, true_rewards.shape)
        print("predicted_rewards", predicted_rewards.rsample(), predicted_rewards.rsample().shape)
        # print("predicted_rewards", predicted_rewards, "mean",  predicted_rewards.mean, "loc", predicted_rewards.loc, "variance" , predicted_rewards.variance, predicted_rewards.rsample(), predicted_rewards.rsample().shape)
        # print("true_action", true_action, true_action.shape)
        # print("predicted_action", predicted_action, "mean", predicted_action.mean, "loc", predicted_action.loc, "variance" , predicted_action.variance, predicted_action.rsample(), predicted_action.rsample().shape)
        # print("true_latent", true_latent, true_latent.shape)
        # print("predicted_latent", predicted_latent, "mean", predicted_latent.mean,"loc",  predicted_latent.loc, "variance" , predicted_latent.variance, predicted_latent.rsample(), predicted_latent.rsample().shape)

        recurrent_gaussian_loss = reward_loss + transition_imagine_loss #+ controller_imagine_loss  # anudeep, changes two lines
        # recurrent_gaussian_loss.backward()
        
        ##################
        print("rgp losses, r, t", reward_loss, transition_imagine_loss, "total:", recurrent_gaussian_loss)
        #gp_pbar.set_description("rgpLoss:"+str(float(recurrent_gaussian_loss)))

        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
          for group in optimiser.param_groups:
            group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
        
        #compute losses
        if args.use_regular_vae:
          (observation_loss  + latent_kl_loss + recurrent_gaussian_loss).backward()
          print("observation_loss", observation_loss, "latent_kl_loss", latent_kl_loss)
          losses.append([observation_loss.item(), reward_loss.item(), latent_kl_loss.item(), transition_imagine_loss.item(), 1.])
          # gp_pbar.set_description("observation_loss:"+str(float(observation_loss)))
          # gp_pbar.set_description("latent_kl_loss:"+str(float(latent_kl_loss)))
          gp_pbar.set_description("observation_loss:"+str(float(observation_loss)) + " | rgpLoss:"+str(float(recurrent_gaussian_loss)) + " | latent_kl_loss:"+str(float(latent_kl_loss)))
        else:
          _backward_infGaussianVAE(loss_dict["WAE-GP"] + recurrent_gaussian_loss)
          print("infGaussianLoss", loss_dict["WAE-GP"].item() + recurrent_gaussian_loss.item())
          print("infGaussianLoss loss_dict", loss_dict)
          losses.append([
            loss_dict["c_cluster_kld"],
            loss_dict["kumar2beta_kld"],
            loss_dict["w_context_kld"],
            loss_dict["z_latent_space_kld"],
            loss_dict["recon"],
            reward_loss.item(),
            transition_imagine_loss.item(),
            # controller_imagine_loss.item()
          ])
          gp_pbar.set_description("WAE-GP:"+str(float(loss_dict["WAE-GP"].item())) + " | rgpLoss:"+str(float(recurrent_gaussian_loss)))
        # optimizer includes recurrent_gp and VAE is using regular VAE
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimizer.step()
        gp_pbar.set_postfix(rgpLoss=recurrent_gaussian_loss.item(), rLoss=reward_loss.item(), TtLoss=transition_imagine_loss.item())
    del true_latent, true_action, predicted_rewards, predicted_latent, init_states, latent_states, init_actions, actions, observations
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
      print(f'init_states prior to unfold, {init_states.shape}')
      init_actions = actions[:-args.horizon_size-1].unfold(0, args.lagging_size, 1)
      print(f'init_states after unfold, {init_states.shape}')
      # init_actions: <chunk_size - horizon_size - lagging_size, batch_size, action_size, lagging_size> 

      # without transpose, the result will be [t0, t1, t2, t0, t1, t2,...]
      # We want [t0, t0, t1, t1, t2, t2, ...] so the lagging_actions[..., self.action_size:] works as expected
      init_states = torch.transpose(init_states, -2, -1)
      print(f'init_states after transpose, {init_states.shape}')
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
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.norm_reward, args.input_size) # Anudeep, new
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
recurrent_gp = RecurrentGP(args.horizon_size, args.state_size, env.action_size, args.lagging_size, args.num_inducing_recurrent_gp, args.device).to(device=args.device)
param_list = list(recurrent_gp.parameters()) 
optimizer = torch.optim.Adam(recurrent_gp.parameters(), lr=0.002)

if args.use_regular_vae:
  if args.input_size == 32:
    observation_model = RegDecoder(args.state_size).to(device=args.device)
    encoder = RegEncoder(args.state_size).to(device=args.device)
  else:
    observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.state_size, args.embedding_size, args.activation_function, args.input_size).to(device=args.device)
    encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.state_size, args.activation_function, args.input_size).to(device=args.device)
  param_list = list(observation_model.parameters()) + list(encoder.parameters()) + list(recurrent_gp.parameters()) 
else:
  hyperParams = {"batch_size": args.batch_size * args.chunk_size,
               "input_d": 1,
               "prior_alpha": 7., #gamma_alpha
               "prior_beta": 1., #gamma_beta
               "K": args.num_mixtures,
               "hidden_d": args.hidden_size,
               "latent_d": args.state_size,
               "latent_w": args.w_dim,
               "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
               "LEARNING_RATE": args.learning_rate,
               "CRITIC_ITERATIONS" : 5
               }
  infinite_vae = InfGaussMMVAE(hyperParams, hyperParams["K"], 3, hyperParams["latent_d"], hyperParams["latent_w"], hyperParams["hidden_d"], args.device, args.input_size, hyperParams["batch_size"], include_elbo2=True).to(device=args.device)
  
  vae_encoder, vae_decoder, discriminator = infinite_vae.encoder, infinite_vae.decoder, VAECritic(infinite_vae.z_dim)

  if args.train_all_layers:
    inf_optim = torch.optim.Adam(infinite_vae.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
  else:
    enc_optim = torch.optim.Adam(vae_encoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
    dec_optim = torch.optim.Adam(vae_decoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
    salient_en    = torch.optim.SGD(infinite_vae.encoder_w.parameters(), lr=hyperParams["LEARNING_RATE"], momentum=0.9, weight_decay=0.0005)
    salient_dec   = torch.optim.SGD(infinite_vae.decoder_w.parameters(), lr=hyperParams["LEARNING_RATE"] * 10, momentum=0.9, weight_decay=0.0005)
  dis_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.5 * hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))

  # scheduler are not used currently
  # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size = 30, gamma = 0.5)
  # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size = 30, gamma = 0.5)
  # dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size = 30, gamma = 0.5)
  #optimize transition function, controller and reward functions
  ##should we optimize these transition function and controller like the following?
  param_list    = list(recurrent_gp.parameters())
  

optimiser = AdaBound(param_list, lr=0.0001) if args.use_ada_bound else optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)

actor_critic_planner = ActorCriticPlanner(args.lagging_size, args.state_size, env.action_size, recurrent_gp, env.action_range[0], env.action_range[1], args.num_sample_trajectories, args.hidden_size, args.num_gp_likelihood_samples, args.num_inducing_planner, args.device).to(device=args.device)
policy = actor_critic_planner.actor

# set up optimizers
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

# reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.likelihood, recurrent_gp, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size)))
#### ELBO objective for the transition and controller GP ### ???? need to be fixed
# transition_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.transition_module.likelihood, recurrent_gp.transition_module, args.batch_size*(args.chunk_size-args.lagging_size), beta=1.0))#what X=action+latent_space
#transition_mll = DeepApproximateMLL(PredictiveLogLikelihood(recurrent_gp.transition_modules.likelihood, recurrent_gp.transition_modules, args.batch_size*(args.chunk_size-args.lagging_size-args.horizon_size), beta=1.0)) #typically produces better predictive variances than the ELBO
# controller_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.policy_module.likelihood, recurrent_gp.policy_module,args.batch_size*(args.chunk_size-args.lagging_size), beta=1.0))#latent_space
rgp_training_episode = args.seed_episodes

reward_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.reward_module.likelihood, recurrent_gp.reward_module, args.experience_size))
transition_mll = DeepApproximateMLL(VariationalELBO(recurrent_gp.transition_module.likelihood, recurrent_gp.transition_module, args.experience_size))

find_model_size(recurrent_gp)
if args.use_regular_vae:
  find_model_size(observation_model)
  find_model_size(encoder)
else:
  find_model_size(infinite_vae)


train_vae(args.warm_up_vae, [])
# print("Test Recurrent GP before Pretraining:")
# test_recurrent_gp(1000)
rgp_step(args.initial_collect_interval, []) # Anudeep, pretraining?
print('Finished rgp_step')
# print("Test Recurrent GP after Pretraining:")
# test_recurrent_gp(1000)

###### Training (and testing)

for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []

  # training the observation models and recurrent GP 
  if ((episode-1) == rgp_training_episode):
    rgp_training_episode = int(rgp_training_episode * args.rgp_training_interval_ratio)
    # print("Test Recurrent GP before rgp_training_episode:")
    # test_recurrent_gp(1000)
    losses = rgp_step(args.collect_interval, losses)
    # print("Test Recurrent GP after rgp_training_episode:")
    # test_recurrent_gp(1000)

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics = update_plot_loss_metric(args, metrics, losses, results_dir)
    print('Plotted')
  
  # Data collection and planning
  print('Data collection and planning begins')
  if args.use_regular_vae:
    encoder.eval()
  else:
    infinite_vae.eval()
    infinite_vae.batch_size = 1
  recurrent_gp.eval()
  observation, total_reward, time_step = env.reset(), 0, 0
  with torch.no_grad():
    if not args.use_regular_vae:
      current_latent_state = get_latent(observation + 0.5)
    else:
      current_latent_state = get_latent(observation)
      print('Got latent')

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
  print('Initialized planner states')
  imagined_rewards = []
  true_rewards = []

  # pbar = tqdm(range(args.max_episode_length // args.action_repeat)) # max_episode_length = 1000
  # time_steps = 0
  # with gpytorch.settings.num_likelihood_samples(args.num_gp_likelihood_samples):
  #   for t in pbar:
  #     (
  #       action,
  #       action_log_prob,
  #       policy_mll_loss,
  #       value, q_value,
  #       transition_dist,
  #       imagined_reward
  #     ) = actor_critic_planner.act(episode_states[-args.lagging_size:], episode_actions[-args.lagging_size:], device=args.device, softplus=args.imaginary_rollout_softplus)
  #     print('actor_critic_planner acted')
  #     observation, reward, done = env.step(action[0].cpu())
  #     print('Stepped')

  #     imagined_rewards.append(imagined_reward)
  #     true_rewards.append(reward)
      
  #     with torch.no_grad():
  #       if not args.use_regular_vae:
  #         current_latent_state = get_latent(observation + 0.5)
  #       else:
  #         current_latent_state = get_latent(observation)
  #       print('Got latent')
  #     (
  #       episode_policy_kl,
  #       episode_policy_mll_loss,
  #       transition_kl,
  #       transition_mll_loss,
  #       episode_transition_kl,
  #       episode_transition_mll_loss,
  #       episode_states,
  #       episode_actions,
  #       episode_values,
  #       episode_q_values,
  #       episode_rewards
  #     ) = update_planner_states(transition_dist, current_latent_state, actor_critic_planner, action, value, q_value, reward,
  #                               episode_policy_kl, action_log_prob, episode_policy_mll_loss, policy_mll_loss, episode_transition_kl, 
  #                               episode_transition_mll_loss, episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, args)
  #     print('Updated planner state')
      
  #     D.append(observation, action.detach().cpu(), reward, done)
  #     print('Added to replay buffer')
  #     total_reward = total_reward + reward
      # if args.render:
      #   env.render()
      # if done:
        # pbar.close()
        # break
  
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
  
  # planning_optimiser.zero_grad()
  # (value_loss + q_loss + policy_loss + current_policy_mll_loss + current_transition_mll_loss).backward()
  print("value_loss", value_loss, "q_loss", q_loss, "policy_loss", policy_loss, "current_policy_mll_loss", current_policy_mll_loss, "current_transition_mll_loss", current_transition_mll_loss)
  print("planning total loss", value_loss + q_loss + policy_loss + current_policy_mll_loss + current_transition_mll_loss)
  print("training rewards", total_reward)
  print("current_rewards", current_rewards)
  print("train observation", observation.shape, observation, "min:", torch.min(observation), "max:", torch.max(observation))
  print("train imagined_rewards", imagined_rewards)
  print("train true_rewards", true_rewards)
  nn.utils.clip_grad_norm_(actor_critic_planner.parameters(), args.grad_clip_norm, norm_type=2)
  # planning_optimiser.step()

  metrics = update_plot_planning_loss_metric(metrics, value_loss, policy_loss, q_loss, current_policy_mll_loss, current_transition_mll_loss, t, episode, total_reward, results_dir, args)

  # delete unused variables
  del episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, episode_policy_kl, episode_policy_mll_loss, episode_transition_kl, episode_transition_mll_loss
  # del action, action_log_prob, policy_mll_loss, value, q_value, transition_dist
  del current_q_values, previous_q_values, current_rewards, current_policy_kl, current_transition_kl, current_values, previous_values, soft_v_values, target_q_values, value_loss, q_loss, policy_loss 
  del current_policy_mll_loss, current_transition_mll_loss

  if args.use_regular_vae:
    encoder.train()
  else:
    infinite_vae.train()
  recurrent_gp.train()
  policy.train() # Anudeep, New

  # Test model
  if episode % args.test_interval == 0:
    test_recurrent_gp(1000)

  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    if args.use_regular_vae:
      save_regularVAE(observation_model, encoder, recurrent_gp, optimiser, actor_critic_planner, planning_optimiser, episode, results_dir)
    else:
      save_infGaussianVAE(infinite_vae, recurrent_gp, optimiser, actor_critic_planner, planning_optimiser, episode, results_dir)
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes
print("Done!")
# Close training environment
env.close()
