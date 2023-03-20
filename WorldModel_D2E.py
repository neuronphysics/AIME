import os
import numpy as np
import torch
import itertools
from torch import nn, optim, jit
from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
import torchvision.transforms as T
from VRNN.main import ModelState
from collections import namedtuple
import gym
import mujoco_py
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from planner_D2E_regularizer import D2EAgent as agent
from tqdm import tqdm
import logging
import cv2

def write_video(frames, title, path=''):
    frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :,
             ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()

class logger:
      def __init__ (self):
          self.fmt = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
          self.level = logging.INFO
          self.datefmt = '%Y-%m-%d %H:%M:%S'

class Logger(logging.Logger):
    def __init__(self, name: str, level=None) -> None:
        if level is None:
            level = logger.level
        super().__init__(name, level=level)
        formatter = logging.Formatter(
            fmt=logger.fmt,
            datefmt=logger.datefmt,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logger.level)
        self.addHandler(handler)

#inspired by https://github.com/mahkons/Dreamer/blob/003c3cc7a9430e9fa0d8af9cead88d8f4b06e0f4/dreamer/WorldModel.py


class LSTMBlock(jit.ScriptModule):
    def __init__(self, in_dim, out_dim, hidden_dims, activation_func_module):
        super(LSTMBlock, self).__init__()
        self.rnn = nn.LSTM( in_dim, hidden_dims[0], bidirectional=False, batch_first=True)

        layers = [nn.Linear(hidden_dims[0], hidden_dims[0]), activation_func_module()]
        layers += sum([[nn.Linear(ind, outd), activation_func_module()]
            for ind, outd in zip(hidden_dims, hidden_dims[1:])], [])
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.model = nn.Sequential(*layers)

    @jit.script_method
    def forward(self, x):
        # (Batch, D, seq_len)
        length =  [torch.max((x[i,0,:]!=0).nonzero()).item()+1 for i in range(x.shape[0])]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        out_packed, (_, _) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.model(out)


class RewardNetwork(LSTMBlock):
    def __init__(self, state_dim):
        super(RewardNetwork, self).__init__(state_dim, 1, [300, 300], nn.GELU)

    def forward(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)


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


class StubDiscountNetwork(jit.ScriptModule):
    def __init__(self, gamma):
        super(StubDiscountNetwork, self).__init__()
        self.gamma = gamma
        self.scale = math.log(gamma) - math.log(1 - gamma)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super().to(device)
        
    @jit.script_method
    def predict_logit(self, x: torch.Tensor)-> torch.Tensor:
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
                "hidden_transit": 100,
                "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
                "LEARNING_RATE": 2e-4,
                "CRITIC_ITERATIONS" : 5,
                "GAMMA": 0.99,
                "PREDICT_DONE": False,
                "seed": 1234,
                "number_of_mixtures": 8,
                "weight_decay": 1e-5,
                "n_channel": 3,
                "VRNN_Optimizer_Type":"MADGRAD",
                "MAX_GRAD_NORM": 100.
                }

class WorldModel(jit.ScriptModule):
    def __init__(self, hyperParams, sequence_length, env_name='Hopper-v2', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), log_dir="logs"):
        super(WorldModel, self).__init__()
        env = gym.make(env_name)
        self._params = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self.n_discriminator_iter = self._params.CRITIC_ITERATIONS

        self.state_dim       = self._params.latent_d
        self.action_dim      = env.action_space.shape[0]

        self.device          = device
 
        self.criterion       = nn.MSELoss(reduction='sum')

        self.reward_model    = RewardNetwork( self.state_dim + self.action_dim).to(self.device)
        self.discount_model  = DiscountNetwork.create(self.state_dim + self.action_dim, self._params.PREDICT_DONE, self._params.GAMMA).to(device)
        self.logger = Logger(self.__class__.__name__)

        modelstate =  ModelState(seed            = self._params.seed,
                                 nu              = self.state_dim + self.action_dim,
                                 ny              = self.state_dim,
                                 sequence_length = sequence_length,
                                 h_dim           = self._params.hidden_transit,
                                 z_dim           = self.state_dim,
                                 n_layers        = 2,
                                 n_mixtures      = self._params.number_of_mixtures,
                                 device          = device,
                                 optimizer_type  = self._params.VRNN_Optimizer_Type,
                                )

        self.transition_model = modelstate.model
        self.variational_autoencoder = InfGaussMMVAE(hyperParams,
                                                     K          = self._params.K,
                                                     nchannel   = self._params.n_channel,
                                                     z_dim      = self.state_dim,
                                                     w_dim      = self._params.latent_w,
                                                     hidden_dim =self._params.hidden_d,
                                                     device     = self.device,
                                                     img_width  =self._params.image_width,
                                                     batch_size = self._params.batch_size,
                                                     num_layers = 4,
                                                     include_elbo2=True)
        self.encoder = self.variational_autoencoder.encoder
        self.decoder = self.variational_autoencoder.decoder
        self.discriminator = VAECritic(self.state_dim)
        self.parameters = itertools.chain(
            self.reward_model.parameters(),
            self.discount_model.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )
        self.writer = SummaryWriter(log_dir)
        self.optimizer = torch.optim.AdamW(self.parameters,
                                           lr      = self._params.LEARNING_RATE,
                                           betas   = (0.9, 0.999),
                                           weight_decay = self._params.weight_decay,
                                           amsgrad = True)

        self._scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.5)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr    = 0.5 * self._params.LEARNING_RATE,
                                                    betas = (0.5, 0.9))
        self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optim, step_size = 30, gamma = 0.5)
    
    @torch.jit.script_method
    def optimize(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        self.discriminator.train()
        pbar = tqdm(range(self.n_discriminator_iter))
        for _ in pbar:
            z_real, z_x_mean, z_x_sigma, c_posterior, w_x_mean, w_x_sigma, gmm_dist, z_wc_mean_prior, z_wc_logvar_prior, x_reconstructed = self.variational_autoencoder(obs)
            z_fake = gmm_dist.sample()

            critic_real = self.discriminator(z_real).reshape(-1)
            critic_fake = self.discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(self.discriminator, z_real, z_fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) +self._params.LAMBDA_GP * gp
            )
            self.discriminator_optim.zero_grad()
            loss_critic.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self._params.MAX_GRAD_NORM)
            self.discriminator_optim.step()
            
        gen_fake  = self.discriminator(z_fake).reshape(-1)

        self.transition_model.train()
        self.optimizer.zero_grad()
        w_x, w_x_mean, w_x_sigma, z_next, z_next_mean, z_next_sigma, c_posterior = self.variational_autoencoder.GMM_encoder(next_obs)
        inputs = torch.cat([z_real, action], dim=1)

        transition_loss, transition_disc_loss, hidden, real , fake = self.transition_model(inputs, z_next)
        transition_gradient_penalty = self.transition_model.wgan_gp_reg(real, fake)
        
        predicted_reward = self.reward_model(inputs) #reward
        loss_dict, z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean, w_posterior_sigma, dist, z_prior_mean, z_prior_logvar, X_reconst = self.variational_autoencoder.get_ELBO(X)
        loss_dict["wasserstein_gp_loss"] = -torch.mean(gen_fake)
        loss_dict["total_observe_loss"] = loss_dict["loss"]+loss_dict["wasserstein_gp_loss"]
        loss_dict["transition_total_loss"] = transition_loss+ transition_disc_loss + transition_gradient_penalty
        loss_dict["reward_loss"] = self.criterion(reward, predicted_reward)
        discount_loss = torch.tensor(0.)
        #computing discount factor
        if self.PREDICT_DONE:
            predicted_discount_logit = self.discount_model.predict_logit(inputs)
            discount_loss = F.binary_cross_entropy_with_logits(predicted_discount_logit, discount * self._gamma)
        loss_dict["discount_loss"] = discount_loss
        loss_dict["total_model_loss"] = loss_dict["total_observe_loss"]+loss_dict["transition_total_loss"]+loss_dict["reward_loss"]+loss_dict["discount_loss"]
        loss_dict["total_model_loss"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters,self._params.MAX_GRAD_NORM)
        self.optimizer.step()
        
        self.writer.add_scalar('model_loss', {'observation loss': loss_dict["total_observe_loss"].item(),
                                         'transition loss': loss_dict["transition_total_loss"].item(),
                                         'reward loss': loss_dict["reward_loss"].item(),
                                         'discount loss': loss_dict["discount_loss"].item(),
                                         'total loss': loss_dict["total_model_loss"].item()})

        return hidden


    def imagine(self, agent, state, horizon):
        state_list, reward_list, discount_list, action_list = [state], [], [], []
        for _ in range(horizon):
            _, action, _ = agent._p_fn(state.to(device=self.device))
            features = torch.cat([torch.stack(state_list)[:-1], torch.stack(action_list)], dim=1)
            sample, sample_mu, sample_sigma, state = self.transition_model.generate(features)

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
