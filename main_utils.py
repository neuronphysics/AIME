import torch 
import os
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from models import bottle, bottle_two_output
from utils import lineplot
from Hierarchical_StickBreaking_GMMVAE import gradient_penalty

def free_params(module: nn.Module):
  for p in module.parameters():
    p.requires_grad = True

def frozen_params(module: nn.Module):
  for p in module.parameters():
    p.requires_grad = False

def find_model_size(model):
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

# only called when using infGaussianVAE
def train_discriminator(observations, discriminator, infinite_vae, hyperParams, args, dis_optim):
  free_params(discriminator)
  frozen_params(infinite_vae)
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
  frozen_params(discriminator)
  free_params(infinite_vae)
  return z_fake

def get_regularVAE_loss_and_latent(observations, encoder, global_prior, observation_model, args):
  latent_mean, latent_std = bottle_two_output(encoder, (observations, ))
  latent_dist = Normal(latent_mean, latent_std)
  latent_kl_loss = kl_divergence(latent_dist, global_prior).sum(dim=2).mean(dim=(0, 1))
  latent_states = latent_dist.rsample()
  observation_loss = F.mse_loss(bottle(observation_model, (latent_states,)), observations, reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
  return observation_loss, latent_kl_loss, latent_states

def get_infGaussianVAE_loss_and_latent(discriminator, z_fake, infinite_vae, observations, original_shape, args):
    gen_fake = discriminator(z_fake).reshape(-1)
    loss_dict = infinite_vae.get_ELBO(observations)
    loss_dict["wasserstein_loss"] =  -torch.mean(gen_fake)
    loss_dict["WAE-GP"]=loss_dict["loss"]+loss_dict["wasserstein_loss"]
    latent_states = torch.reshape(infinite_vae.get_latent_states(observations), (original_shape[0], original_shape[1], args.state_size))
    return loss_dict, latent_states

def init_planner_states(args, current_latent_state, env):
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
  return (
    episode_states,
    episode_actions,
    episode_values,
    episode_q_values,
    episode_rewards,
    episode_policy_kl,
    episode_policy_mll_loss,
    episode_transition_kl,
    episode_transition_mll_loss
  )

def update_planner_states(transition_dist, current_latent_state, actor_critic_planner, action, value, q_value, reward,
                          episode_policy_kl, action_log_prob, episode_policy_mll_loss, policy_mll_loss, episode_transition_kl, 
                          episode_transition_mll_loss, episode_states, episode_actions, episode_values, episode_q_values, episode_rewards, args):

  episode_policy_kl = torch.cat([episode_policy_kl, (-action_log_prob).unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
  episode_policy_mll_loss = torch.cat([episode_policy_mll_loss, policy_mll_loss.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
  transition_kl = -transition_dist.log_prob(current_latent_state)
  transition_mll_loss = -actor_critic_planner.transition_mll(transition_dist, current_latent_state)
  episode_transition_kl = torch.cat([episode_transition_kl, transition_kl.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
  episode_transition_mll_loss = torch.cat([episode_transition_mll_loss, transition_mll_loss.unsqueeze(dim=0).mean(dim=-1, keepdim=True)], dim=0)
  episode_states = torch.cat([episode_states, current_latent_state], dim=0)
  episode_actions = torch.cat([episode_actions, action.to(device=args.device)], dim=0)
  episode_values = torch.cat([episode_values, value], dim=0)
  episode_q_values = torch.cat([episode_q_values, q_value], dim=0)
  episode_rewards = torch.cat([episode_rewards, torch.Tensor([[reward]]).to(device=args.device)], dim=0)
  return (
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
  )


def update_plot_planning_loss_metric(metrics, value_loss, policy_loss, q_loss, current_policy_mll_loss, current_transition_mll_loss, t, episode, total_reward, results_dir, args):
  metrics['value_loss'].append(value_loss.item())
  metrics['policy_loss'].append(policy_loss.item())
  metrics['q_loss'].append(q_loss.item())
  metrics['policy_mll_loss'].append(current_policy_mll_loss.item())
  metrics['transition_mll_loss'].append(current_transition_mll_loss.item())
  # Update and plot train reward metrics
  metrics['steps'].append(t + metrics['steps'][-1])
  metrics['episodes'].append(episode)
  metrics['train_rewards'].append(total_reward)
  if args.lineplot:
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['q_loss']):], metrics['q_loss'], 'q_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['policy_mll_loss']):], metrics['policy_mll_loss'], 'policy_mll_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['transition_mll_loss']):], metrics['transition_mll_loss'], 'transition_mll_loss', results_dir)
  return metrics


def update_plot_loss_metric(args, metrics, losses, results_dir):
  if args.use_regular_vae:
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['latent_kl_loss'].append(losses[2])
    metrics['transition_imagine_loss'].append(losses[3])
    metrics['controller_imagine_loss'].append(losses[4])
    if args.lineplot:
      lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['latent_kl_loss']):], metrics['latent_kl_loss'], 'latent_kl_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['transition_imagine_loss']):], metrics['transition_imagine_loss'], 'transition_imagine_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['controller_imagine_loss']):], metrics['controller_imagine_loss'], 'controller_imagine_loss', results_dir)
  else:
    metrics['elbo1'].append(losses[0])
    metrics['elbo2'].append(losses[1])
    metrics['elbo3'].append(losses[2])
    metrics['elbo4'].append(losses[3])
    metrics['elbo5'].append(losses[4])
    metrics['reward_loss'].append(losses[5])
    metrics['transition_imagine_loss'].append(losses[6])
    metrics['controller_imagine_loss'].append(losses[7])
    if args.lineplot:
      lineplot(metrics['episodes'][-len(metrics['elbo1']):], metrics['elbo1'], 'elbo1', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo2']):], metrics['elbo2'], 'elbo2', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo3']):], metrics['elbo3'], 'elbo3', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo4']):], metrics['elbo4'], 'elbo4', results_dir)
      lineplot(metrics['episodes'][-len(metrics['elbo5']):], metrics['elbo5'], 'elbo5', results_dir)
      lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['transition_imagine_loss']):], metrics['transition_imagine_loss'], 'transition_imagine_loss', results_dir)
      lineplot(metrics['episodes'][-len(metrics['controller_imagine_loss']):], metrics['controller_imagine_loss'], 'controller_imagine_loss', results_dir)
  return metrics

def save_regularVAE(observation_model, encoder, recurrent_gp, optimiser, actor_critic_planner, planning_optimiser, episode, results_dir):
  torch.save({
    'observation_model': observation_model.state_dict(),
    'encoder': encoder.state_dict(),
    'recurrent_gp': recurrent_gp.state_dict(),
    'optimiser': optimiser.state_dict(),
    'actor_critic_planner': actor_critic_planner.state_dict(),
    'planning_optimiser': planning_optimiser.state_dict(),
    }, os.path.join(results_dir, 'models_%d.pth' % episode))

def save_infGaussianVAE(infinite_vae, recurrent_gp, optimiser, actor_critic_planner, planning_optimiser, episode, results_dir):
  torch.save({
    'infinite_vae': infinite_vae.state_dict(),
    'recurrent_gp': recurrent_gp.state_dict(),
    'optimiser': optimiser.state_dict(),
    'actor_critic_planner': actor_critic_planner.state_dict(),
    'planning_optimiser': planning_optimiser.state_dict(),
    }, os.path.join(results_dir, 'models_%d.pth' % episode))
