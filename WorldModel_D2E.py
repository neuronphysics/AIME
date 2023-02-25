
import os
import numpy as np
import torch
import itertools
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
from RCGAN import RGANDiscriminator, RGANGenerator
#inspired by https://github.com/mahkons/Dreamer/blob/003c3cc7a9430e9fa0d8af9cead88d8f4b06e0f4/dreamer/WorldModel.py

class RewardNetwork(nn.Module):
    def __init__(self,
                 latent_dim,
                 input_dim,
                 output_dim,
                 sequence_length,
                 batch_size=28,
                 hidden_size_gen=100,
                 num_layer_gen=1,
                 hidden_size_dis=100,
                 num_layer_dis=1,
                 device="cpu"
                 ):
        super(RewardNetwork, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.generator = RGANGenerator(sequence_length = self.sequence_length,
                                       output_size= self.output_dim,
                                       hidden_size = hidden_size_gen,
                                       noise_size = self.latent_dim + self.input_dim,
                                       num_layers=num_layer_gen
                                       )
        self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length,
                                               input_size = self.input_dim + self.output_dim,
                                               hidden_size=hidden_size_dis,
                                               num_layers=num_layer_dis
                                               )

    def wgan_gp_reg(self, x_real, x_fake, center=1., lambda_gp=10.0):
        #eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        batch_size = x_real.shape[0]
        eps = torch.rand(batch_size, 1, 1).to(self.device)
        eps = eps.expand_as(x_real)
        #eps = torch.randn_like(x_real).to(self.device)
        x_interp = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
        d_out = self.discriminator(x_interp)

        gradients = torch.autograd.grad(inputs = x_interp,
                                       outputs  = d_out,
                                       grad_outputs=torch.ones_like(d_out,requires_grad=False, device=self.device),
                                       create_graph=True,
                                       retain_graph=True,
                                       )[0]

        gradients = gradients.view(gradients.size(), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - center) ** 2).mean() * lambda_gp
        return gradient_penalty

    def forward(self, x, y):
        batch_size = x.shape[0]
        z = torch.randn(size=(batch_size, self.sequence_length, self.latent_dim)).to(self.device)
        noise = torch.cat((x.to(self.device), z), dim=2)
        input_feature = torch.cat((x.to(self.device), y.to(self.device)), dim=2)
        fake = self.generator(noise)
        out_fake = fake.clone()###
        fake_ = fake.permute(0,2,1)
        padded = nn.ConstantPad1d((0, x.shape[1] - fake.shape[1]), 0)(fake_)
        fake_ = padded.permute(0,2,1)
        fake = torch.cat((x, fake_), dim=2)
        length_real =  torch.LongTensor([torch.max((y[i,:,0]!=0).nonzero()).item()+1 for i in range(y.shape[0])])

        disc_real = self.discriminator(input_feature, length_real)

        disc_fake = self.discriminator(fake.to(self.device))
        gradient_penalty = self.wgan_gp_reg(input_feature, fake)
        d_loss = -torch.mean(disc_real) + torch.mean(disc_fake) + gradient_penalty
        return d_loss, out_fake

    def sample_from(self, x, return_gen_flag_feature=False):
        batch_size = x.shape[0]
        self.discriminator.eval()
        self.generator.eval()
        z = torch.randn(size=(batch_size, self.sequence_length, self.latent_dim)).to(self.device)
        noise = torch.cat((x.to(self.device), z), dim=2)

        with torch.no_grad():
            features = self.generator(noise)
            features = features.cpu().numpy()
            gen_flags = np.zeros(features.shape[:-1])
            lengths = np.zeros(features.shape[0])
            for i in range(len(features)):
                winner = (features[i, :, -1] > features[i, :, -2])
                argmax = np.argmax(winner == True)
                if argmax == 0:
                    gen_flags[i, :] = 1
                else:
                    gen_flags[i, :argmax + 1] = 1
                lengths[i] = argmax
            if not return_gen_flag_feature:
                features = features[:, :, :-2]
        return features, gen_flags, lengths

class LSTMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation_func_module):
        super(LSTMBlock, self).__init__()
        self.rnn = nn.LSTM( in_dim, hidden_dims[0], bidirectional=False, batch_first=True)

        layers = [nn.Linear(hidden_dims[0], hidden_dims[0]), activation_func_module()]
        layers += sum([[nn.Linear(ind, outd), activation_func_module()]
            for ind, outd in zip(hidden_dims, hidden_dims[1:])], [])
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # (Batch, D, seq_len)
        length =  torch.LongTensor([torch.max((x[i,0,:]!=0).nonzero()).item()+1 for i in range(x.shape[0])])
        packed = nn.utils.rnn.pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        out_packed, (_, _) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.model(out)



class DiscountNetwork(LSTMBlock):
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
    def __init__(self, hyperParams, sequence_length, env_name='Hopper-v2', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        self.reward_model    = RewardNetwork(latent_dim = hyperParams["hidden_d"],
                                             input_dim = self.state_dim + self.action_dim,
                                             output_dim = 1,
                                             sequence_length = sequence_length,
                                             batch_size = hyperParams["batch_size"],
                                             hidden_size_gen = hyperParams["hidden_d"],
                                             num_layer_gen=1,
                                             hidden_size_dis = hyperParams["hidden_d"],
                                             num_layer_dis=1,
                                             device=self.device)

        self.discount_model  = DiscountNetwork.create(self.state_dim + self.action_dim, hyperParams["PREDICT_DONE"], self._gamma).to(device)
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
            self.reward_model.generator.parameters(),
            self.reward_model.discriminator.parameters(),
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
                state_plus_action = torch.cat([embed, action], dim=1)
                X_recons_linear, mu_z, logvar_z, _, _, _, _, _, mu_pz, logvar_pz, _, _ = self.visual_encoder_decoder(next_obs)
                next_embed = self.visual_encoder_decoder.z_x
                loss_dict["trans_loss"], hidden = self.transition_model(state_plus_action, next_embed)
                loss_dict["WAE-GP"] += loss_dict["trans_loss"]
                predicted_obs = self.decoder(state_plus_action)
                #predicting reward
                loss_dict["reward_loss"], predicted_reward = self.reward_model(state_plus_action, reward)


                loss_dict["recon"] = self.criterion(predicted_obs.view(-1, self.n_channel*self.img_size*self.img_size), obs.view(-1, self.n_channel*self.img_size*self.img_size))
                loss_dict["WAE-GP"]  += loss_dict["recon"]
                loss_dict["obs_loss"]+= loss_dict["recon"]

                loss_dict["WAE-GP"] += loss_dict["reward_loss"]
                loss_dict["discount_loss"] = torch.tensor(0.)
                #computing discount factor
                if self.PREDICT_DONE:
                   predicted_discount_logit = self.discount_model.predict_logit(state_plus_action)
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
            features = torch.cat([torch.stack(state_list)[1:], torch.stack(action_list)], dim=1)
            sample, sample_mu, sample_sigma, state = self.transition_model.generate(features)
            state  = torch.cat(state, dim=-1)
            reward, _, _ = self.reward_model.sample_from(features)
            discount = torch.sigmoid(self.discount_model.predict_logit(features))
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            discount_list.append(discount)
        state = torch.stack(state_list)
        action = torch.stack(action_list)
        reward = torch.stack(reward_list)
        discount = torch.stack(discount_list)
        return state, action, reward, discount
