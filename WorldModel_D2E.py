
import os
import numpy as np
import torch
from torch import nn, optim
from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
import torchvision.transforms as T
from VRNN.main import ModelState
import gym
import mujoco_py
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from planner_D2E_regularizer import D2EAgent as agent
#inspired by https://github.com/mahkons/Dreamer/blob/003c3cc7a9430e9fa0d8af9cead88d8f4b06e0f4/dreamer/WorldModel.py
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation_func_module):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dims[0]), activation_func_module()]
        layers += sum([[nn.Linear(ind, outd), activation_func_module()]
            for ind, outd in zip(hidden_dims, hidden_dims[1:])], [])
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class RewardNetwork(MLP):
    def __init__(self, state_dim):
        super(RewardNetwork, self).__init__(in_dim=state_dim, out_dim=1, hidden_dims=[300, 300], activation_func_module=nn.GELU)

    def forward(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)



class DiscountNetwork(MLP):
    def __init__(self, state_dim):
        super(DiscountNetwork, self).__init__(in_dim=state_dim, out_dim=1, hidden_dims=[300, 300], activation_func_module=nn.GELU)

    def forward(self, x):
        assert(False)

    def predict_logit(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)

    @staticmethod # TODO move to config factory method?
    def create(state_dim, predict_done, gamma):
        if predict_done:
            return DiscountNetwork(state_dim)
        else:
            return StubDiscountNetwork(gamma)


class StubDiscountNetwork(nn.Module):
    def __init__(self, gamma):
        super(StubDiscountNetwork, self).__init__()
        self.gamma = gamma
        self.scale = math.log(gamma) - math.log(1 - gamma)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super().to(device)

    def predict_logit(self, x):
        return torch.ones(x.shape[:-1], device=self.device) * self.scale


hyperParams = {"batch_size": 40,
                "input_d": 1,
                "prior_alpha": 7., #gamma_alpha
                "prior_beta": 1., #gamma_beta
                "K": 25,
                "image_width": 96,
                "hidden_d": 300,
                "latent_d": 100,
                "latent_w": 200,
                "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
                "LEARNING_RATE": 1e-4,
                "CRITIC_ITERATIONS" : 5,
                "GAMMA": 0.99,
                "PREDICT_DONE": False,
                "seed": 1234,
                "number_of_mixtures": 8,
                "weight_decay": 1e-5,
                "n_channel": 3,
                "MAX_GRAD_NORM": 100.
                }

class WorldModel(nn.Module):
    def __init__(self, hyperParams , env_name='Hopper-v2', device):
        super(WorldModel, self).__init__()
        env = gym.make(env_name)
        self.n_discriminator_iter = hyperParams["CRITIC_ITERATIONS"]
        self.batch_size      = hyperParams["batch_size"]
        self.state_dim       = hyperParams["latent_d"]
        self.action_dim      = env.action_space.shape[0]
        self.n_channel       = hyperParams["n_channel"]
        self.img_size        = hyperParams["image_width"]
        self.device          = device
        self.PREDICT_DONE    = False
        self.lambda          = hyperParams["LAMBDA_GP"]
        self.criterion       = nn.MSELoss(reduction='sum')
        self.Max_Grad_Norm   = hyperParams["MAX_GRAD_NORM"]
        self._gamma          = hyperParams["GAMMA"]
        self.reward_model    = RewardNetwork(self.state_dim ).to(device)
        self.discount_model  = DiscountNetwork.create(self.state_dim , hyperParams["PREDICT_DONE"], self._gamma).to(device)
        modelstate =  ModelState(seed  = hyperParams["seed"],
                                 nu    = self.state_dim + self.action_dim,
                                 ny    = self.state_dim,
                                 h_dim = hyperParams["latent_d"],
                                 z_dim = hyperParams["hidden_d"],
                                 n_layers = 2,
                                 n_mixtures = hyperParams["number_of_mixtures"],
                                 device = device
                                )

        self.transition_model = modelstate.model
        self.visual_encoder_decoder =  InfGaussMMVAE(hyperParams,
                                                     K          = hyperParams["K"],
                                                     nchannel   = self.n_channel,
                                                     z_dim      = hyperParams["latent_d"],
                                                     w_dim      = hyperParams["latent_w"],
                                                     hidden_dim = hyperParams["hidden_d"],
                                                     device     = self.device,
                                                     img_width  = self.img_size,
                                                     batch_size = hyperParams["batch_size"],
                                                     num_layers = 4,
                                                     include_elbo2=True)
        self.encoder = self.visual_encoder_decoder.encoder
        self.decoder = self.visual_encoder_decoder.decoder
        self.discriminator = VAECritic(hyperParams["latent_d"])
        self.parameters = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.transition_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )


        self.optimizer = torch.optim.AdamW(self.parameters,
                                           lr      = hyperParams["LEARNING_RATE"],
                                           betas   = (0.9, 0.999),
                                           weight_decay = hyperParams["weight_decay"],
                                           amsgrad = True)

        self._scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.5)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr    = 0.5 * hyperParams["LEARNING_RATE"],
                                                    betas = (0.5, 0.9))
        self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optim, step_size = 30, gamma = 0.5)

    def optimize(self, obs, action, next_obs, reward, discount, scaler, writer):
        for _ in range(self.n_discriminator_iter):
            X_recons_linear, mu_z, logvar_z, mu_w, logvae_w, qc, kumar_a, kumar_b, mu_pz, logvar_pz, gamma_alpha, gamma_beta = self.visual_encoder_decoder(obs)
            z_fake = self.encoder.reparameterize(mu_z, logvar_z)
            reconstruct_latent_components = self.visual_encoder_decoder.get_component_samples(self.batch_size)
            critic_real = self.discriminator(reconstruct_latent_components).reshape(-1)
            critic_fake = self.discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(discriminator, reconstruct_latent_components, z_fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + sself.lambda * gp
            )
            self.discriminator_optim.zero_grad()
            loss_critic.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.Max_Grad_Norm)
            self.discriminator_optim.step()
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float32) and torch.backends.cudnn.flags(enabled=False):
                gen_fake = self.discriminator(z_fake).reshape(-1)
                loss_dict = self.visual_encoder_decoder.get_ELBO(X)
                loss_dict["wasserstein_loss"] =  -torch.mean(gen_fake)
                loss_dict["WAE-GP"] = loss_dict["loss"]+loss_dict["wasserstein_loss"]

                loss_dict["WAE-GP"]-= loss_dict["recon"]
                loss_dict["obs_loss"]= loss_dict["WAE-GP"]
                embed = self.visual_encoder_decoder.z_x
                state_action = torch.cat([embed, action], dim=1)
                X_recons_linear, mu_z, logvar_z, _, _, _, _, _, mu_pz, logvar_pz, _, _ = self.visual_encoder_decoder(next_obs)
                next_embed = self.visual_encoder_decoder.z_x
                loss_dict["trans_loss"], hidden = self.transition_model(state_action, next_embed)
                loss_dict["WAE-GP"] += loss_dict["trans_loss"]
                predicted_obs = self.decoder(hidden)
                #predicting reward
                predicted_reward = self.reward_model(hidden)


                loss_dict["recon"] = self.criterion(predicted_obs.view(-1, self.n_channel*self.img_size*self.img_size), obs.view(-1, self.n_channel*self.img_size*self.img_size))
                loss_dict["WAE-GP"]  += loss_dict["recon"]
                loss_dict["obs_loss"]+= loss_dict["recon"]
                loss_dict["reward_loss"] = self.criterion(reward, predicted_reward)

                loss_dict["WAE-GP"] += loss_dict["reward_loss"]
                loss_dict["discount_loss"] = torch.tensor(0.)
                #computing discount factor
                if self.PREDICT_DONE:
                   predicted_discount_logit = self.discount_model.predict_logit(hidden)
                   loss_dict["discount_loss"] = F.binary_cross_entropy_with_logits(predicted_discount_logit, discount * self._gamma)
                loss_dict["WAE-GP"] += loss_dict["discount_loss"]
            # Creates gradients and Scales the loss for autograd.grad's backward pass, producing scaled_grad_params
            scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss_dict["WAE-GP"]),
                                                        inputs= self.parameters,
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        allow_unused=True #Whether to allow differentiation of unused parameters.
                                                        )
            inv_scale = 1./scaler.get_scale()

            grad_params = [ p * inv_scale if p is not None and not torch.isnan(p).any() else torch.tensor(0, device=device, dtype=torch.float32) for p in scaled_grad_params ]
            #grad_params = [p * inv_scale for p in scaled_grad_params]
            # Computes the penalty term and adds it to the loss
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                #grad_norm = torch.tensor(0, device=grad_params[0].device, dtype=grad_params[0].dtype)
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                # Compute the L2 Norm as penalty and add that to loss
                loss_dict["WAE-GP"] = loss_dict["WAE-GP"] + grad_norm

            scaler.scale(loss_dict["WAE-GP"]).backward()
            nn.utils.clip_grad_norm_(self.parameters, self.Max_Grad_Norm)
            scaler.step(self.optimizer)
            scaler.update()
        writer.add_scalar('model_loss', {'observation loss':loss_dict["obs_loss"].item(),
                                         'transition loss':loss_dict["trans_loss"].item(),
                                         'reward loss': loss_dict["reward_loss"].item(),
                                         'discount loss': loss_dict["discount_loss"].item(),
                                         'total loss':loss_dict["WAE-GP"].item()})

        return hidden


    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        for _ in range(horizon):
            _, action, _ = agent._p_fn(state.to(device=self.device))
            state_action = torch.cat([state, action], dim=1)
            sample, sample_mu, sample_sigma, state = self.transition_model.generate(state_action)
            state = torch.cat(state, dim=-1)

            state_list.append(state)
            action_list.append(action)

        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = self.reward_model(state[1:])
        discount = torch.sigmoid(self.discount_model.predict_logit(state[1:]))
        return state, action, reward, discount
