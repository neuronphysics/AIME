from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal, OneHotCategorical, MixtureSameFamily, Categorical
from torch.distributions.independent import Independent
import torch
import torch.nn as nn
import torch_optimizer
import os
import sys
from .MaskedNorm import MaskedNorm
from .CustomLSTM import LSTMCore
from .attention_encoder import LatentEncoder
from .vrnn_utilities import _strip_prefix_if_present
import random
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# passing the file name and path as argument
sys.path.append(parent_dir)
from RCGAN import RGANDiscriminator


# https://github.com/alafage/deep-traffic-generation/blob/9585e86dfd7644dc194dcd1fe304486aa02d4381/deep_traffic_generation/core/lsr.py
class CustomMSF(MixtureSameFamily):
    """MixtureSameFamily with `rsample()` method for reparametrization.
    Args:
        mixture_distribution (Categorical): Manages the probability of
            selecting component. The number of categories must match the
            rightmost batch dimension of the component_distribution.
        component_distribution (Distribution): Define the distribution law
            followed by the components. Right-most batch dimension indexes
            component.
    """

    def rsample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped reparameterized sample or
        sample_shape shaped batch of reparameterized samples if the
        distribution parameters are batched.
        Method:
            - Apply `Gumbel Sotmax
              <https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html>`_
              on component weights to get a one-hot tensor;
            - Sample using rsample() from the component distribution;
            - Use the one-hot tensor to select samples.
        .. note::
            The component distribution of the mixture should implements a
            rsample() method.
        .. warning::
            Further studies should be made on this method. It is highly
            possible that this method is not correct.
        """
        assert (
            self.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.mixture_distribution._param
        comp = nn.functional.gumbel_softmax(weights, hard=True).unsqueeze(-1)
        samples = self.component_distribution.rsample(sample_shape)
        return (comp * samples).sum(dim=1)


class NormalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, full_cov=True):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.elu = nn.ELU()
        self.mean_net = nn.Sequential(
            nn.Linear(in_dim, out_dim * n_components),
        )
        ##initialize
        # init_weights(self.mean_net[0])
        if full_cov:
            # Cholesky decomposition of the covariance matrix
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, int(out_dim * (out_dim + 1) / 2 * n_components)),
            )
        else:
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_components),
            )
        ##initialize
        # init_weights(self.tril_net[0])

    def forward(self, x):
        mean = self.mean_net(x).reshape(-1, self.n_components, self.out_dim)
        if self.full_cov:
            tril_values = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[2], mean.shape[2]).to(x.device)
            tril[:, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            # diagonal element must be strictly positive
            # use diag = elu(diag) + 1 to ensure positivity
            tril = tril - torch.diag_embed(torch.diagonal(tril, dim1=-2, dim2=-1)) + torch.diag_embed(
                self.elu(torch.diagonal(tril, dim1=-2, dim2=-1)) + 1 + 1e-8)
        else:
            tril = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.diag_embed(self.elu(tril) + 1 + 1e-8)
        return MultivariateNormal(mean, scale_tril=tril)


class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

        ##initialize
        # init_weights(self.network[0])

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)


class VRNN_GMM(nn.Module):
    #################################################################################
    # The main part of the algorithm for learing the dynamics (world model)
    # A Recurrent Latent Variable Model for Sequential Data (Junyoung Chung et al 2016)
    ##################################################################################
    def __init__(self,
                 u_dim,
                 y_dim,
                 h_dim,
                 z_dim,
                 n_layers,
                 n_mixtures,
                 sequence_length,
                 device,
                 output_clamp=(-10, 10),
                 batch_norm=False,
                 masked_norm=True,
                 bias=False,
                 self_attention_type="multihead",
                 num_layer_discriminator=3,
                 learn_init_state=True,
                 bidirectional=False,
                 dropout=0.1,
                 ):
        super(VRNN_GMM, self).__init__()

        self.y_dim = y_dim
        self.u_dim = u_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_mixtures = n_mixtures
        self.device = device
        self.batch_norm = batch_norm
        self._MaskedNorm = masked_norm
        self.sequence_length = sequence_length
        self.num_layer_dis = num_layer_discriminator
        self._learn_init_state = learn_init_state
        self.is_bidirectional = bidirectional
        self.num_directions = 2 if self.is_bidirectional else 1
        self.zero_init = False if self._learn_init_state else True
        ##====(new)====##
        if self._MaskedNorm:
            self.norm_u = MaskedNorm(u_dim)
            self.norm_y = MaskedNorm(y_dim)

        self._latent_encoder = LatentEncoder(
            u_dim + y_dim,
            hidden_dim=h_dim,
            latent_dim=h_dim,
            self_attention_type=self_attention_type,
            n_encoder_layers=n_layers,
            batchnorm=False,
            dropout=dropout,
            attention_dropout=0,
            attention_layers=2,
            use_lstm=True
        )
        self.output_clamp = output_clamp
        self._eps = 1e-5
        self._max_deviation = 2.0
        self.self_attention_type = self_attention_type
        if dropout > 0.0:
            self.input_drop = nn.Dropout(p=dropout)
        else:
            self.input_drop = nn.Identity()
        ###=============###
        # feature-extracting transformations (phi_y, phi_u and phi_z)

        layers_phi_y = [nn.Linear(self.y_dim, self.h_dim)]
        if self.batch_norm:  # Add batch normalization when requested
            layers_phi_y.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_phi_y = layers_phi_y + [
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        ]

        self.phi_y = torch.nn.Sequential(*layers_phi_y)

        ##initialize
        # init_weights(self.phi_y[0])
        # init_weights(self.phi_y[-1])
        ###=============###
        layers_phi_u = [nn.Linear(self.u_dim, self.h_dim)]
        if self.batch_norm:
            layers_phi_u.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_phi_u = layers_phi_u + [
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        ]

        self.phi_u = torch.nn.Sequential(*layers_phi_u)

        ##initialize
        # init_weights(self.phi_u[0])
        # init_weights(self.phi_u[-1])
        ######
        layers_phi_z = [nn.Linear(self.z_dim, self.h_dim)]
        if self.batch_norm:
            layers_phi_z.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_phi_z = layers_phi_z + [
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        ]

        self.phi_z = torch.nn.Sequential(*layers_phi_z)

        ##initialize
        # init_weights(self.phi_z[0])
        # init_weights(self.phi_z[-1])
        # -----------------------------------------
        # encoder function (phi_enc) -> Inference

        encoder_layers = [nn.Linear(self.h_dim + self.h_dim + self.h_dim, self.h_dim)]
        if self.batch_norm:
            encoder_layers.append(torch.nn.BatchNorm1d(self.h_dim))
        encoder_layers = encoder_layers + [
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        ]
        if self.batch_norm:
            encoder_layers = encoder_layers + [nn.BatchNorm1d(self.h_dim)]

        encoder_layers = encoder_layers + [nn.ReLU()]

        self.enc = torch.nn.Sequential(*encoder_layers)

        ##initialize
        # init_weights(self.enc[0])
        # if not self.batch_norm:
        #    init_weights(self.enc[2])
        # else:
        #    init_weights(self.enc[3])
        # -----------------------------------------
        ###  encoder mean
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        # init_weights(self.enc_mean)

        ###  encoder log_var
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)
        # init_weights(self.enc_logvar)

        # -----------------------------------------
        # prior function (phi_prior) -> Prior

        layers_prior = [nn.Linear(self.h_dim, self.h_dim)]
        if self.batch_norm:
            layers_prior.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_prior = layers_prior + [
            nn.ReLU(),
        ]

        self.prior = torch.nn.Sequential(*layers_prior)
        ##initialize
        # init_weights(self.prior[0])

        # prior mean

        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        # init_weights(self.prior_mean)

        # prior logvar
        self.prior_logvar = nn.Linear(self.h_dim, self.z_dim)
        # init_weights(self.prior_logvar)

        # init_weights(self.prior_logvar)

        # -----------------------------------------
        # decoder function (phi_dec) -> Generation
        layers_decoder = [
            nn.Linear(self.h_dim + self.h_dim + self.h_dim, self.h_dim)  # modified
        ]
        if self.batch_norm:
            layers_decoder.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_decoder = layers_decoder + [
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim)
        ]
        if self.batch_norm:
            layers_decoder.append(torch.nn.BatchNorm1d(self.h_dim))
        layers_decoder = layers_decoder + [
            nn.ReLU(),
        ]

        layers_decoder.append(
            nn.Linear(self.h_dim, self.n_mixtures + self.n_mixtures * 2 * self.y_dim)
        )

        ## (decoder) Normal distribution
        self.decoder = torch.nn.Sequential(*layers_decoder)

        # recurrence function (f_theta) -> Recurrence
        # self._rnn =  nn.LSTM(self.h_dim + self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias, batch_first= True)#modified

        self._rnn = LSTMCore(input_size=self.h_dim + self.h_dim + self.h_dim,
                             hidden_size=self.h_dim,
                             num_layers=self.n_layers,
                             batch_first=True,
                             train_truncate=100,
                             train_burn_in=80)

        self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length,
                                               input_size=2 * self.h_dim + self.u_dim + self.y_dim,
                                               hidden_size=self.h_dim,
                                               num_layers=num_layer_discriminator
                                               )
        if self._learn_init_state:
            self.to_init_state_h = nn.Linear(u_dim, self.h_dim * self.n_layers * self.num_directions)
            self.to_init_state_c = nn.Linear(u_dim, self.h_dim * self.n_layers * self.num_directions)
        self.to(self.device)
        # self.apply(self.weight_init)

    def wgan_gp_reg(self, x_real, x_fake, center=1., lambda_gp=10.0):

        batch_size, T, D = x_real.shape

        eps = torch.rand((batch_size, 1, 1)).repeat(1, T, D).to(self.device)

        # eps = torch.randn_like(x_real).to(self.device)
        x_interp = (eps * x_real + (1 - eps) * x_fake)
        x_interp.requires_grad_(True)
        d_out = self.discriminator(x_interp)
        # for name, param in self.discriminator.named_parameters():
        #   if param.requires_grad==True:
        #      print(name, param.data, param.grad_fn)

        gradients = torch.autograd.grad(inputs=x_interp,
                                        outputs=d_out,
                                        grad_outputs=torch.autograd.Variable(torch.ones_like(d_out),
                                                                             requires_grad=False).to(self.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True,
                                        )[0]

        gradients = gradients.view(gradients.size(), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - center) ** 2).mean() * lambda_gp

        return gradient_penalty

    def initialize_state_vectors(self, batch_size, first_obs=None):
        """
        incorporating tips from here to improve the performance of the model
        https://danijar.com/tips-for-training-recurrent-neural-networks/
        """

        if self._learn_init_state:
            c_0 = self.to_init_state_c(torch.squeeze(first_obs[:, :, 0:1], dim=-1))
            c_0 = c_0.reshape(-1, self.n_layers, self.h_dim).transpose(0, 1).contiguous()
            h_0 = self.to_init_state_h(torch.squeeze(first_obs[:, :, 0:1], dim=-1))
            h_0 = h_0.reshape(-1, self.n_layers, self.h_dim).transpose(0, 1).contiguous()

        elif self.zero_init:
            h_0 = torch.nn.Parameter(torch.zeros(self.n_layers, batch_size, self.h_dim).to(self.device),
                                     requires_grad=True)
            c_0 = torch.nn.Parameter(torch.zeros(self.n_layers, batch_size, self.h_dim).to(self.device),
                                     requires_grad=True)
        else:
            h_0 = torch.nn.Parameter(torch.rand(self.n_layers, batch_size, self.h_dim).to(self.device),
                                     requires_grad=True)
            c_0 = torch.nn.Parameter(torch.rand(self.n_layers, batch_size, self.h_dim).to(self.device),
                                     requires_grad=True)

        return (h_0, c_0)

    def forward(self, u, y):
        deterministic_hidden_state = []
        batch_size = y.size(0)
        # input has size (Batch, D, seq_len)
        # seq_len = y.shape[-1]
        seq_len = [torch.max((u[i, 0, :] != 0).nonzero()).item() + 1 for i in range(u.shape[0])]

        noise = torch.randn(size=(batch_size, self.sequence_length, self.z_dim)).to(self.device)
        fake_y = torch.zeros(batch_size, self.sequence_length, self.y_dim).to(self.device)
        attention_latent = torch.zeros(batch_size, self.sequence_length, 2 * self.h_dim, device=self.device).to(
            self.device)
        ##======fiding sequece length of each episode =======##
        # mask = ~u.eq(0.0)

        # sqn  = torch.unique(mask.sum(dim=-1), sorted=False, dim=1).squeeze(-1)
        # sqn  = sqn.cpu()
        # print(seq_len,sqn, u.shape)
        # assert torch.all(torch.eq(seq_len, sqn))
        ##==================== (new) =====================##
        reshaped_u = u.permute(0, 2, 1)
        reshaped_y = y.permute(0, 2, 1)

        assert not torch.isnan(reshaped_u).any()
        assert not torch.isnan(reshaped_y).any()
        assert torch.isfinite(reshaped_u).any()

        # mistmach between input???
        if self._MaskedNorm:
            normed_u = self.norm_u(reshaped_u)
            normed_y = self.norm_y(reshaped_y)
            assert not torch.isnan(normed_u).any()
            assert not torch.isnan(normed_y).any()
            context_u = normed_u
            context_y = normed_y
        else:
            context_u = reshaped_u
            context_y = reshaped_y
        if self.self_attention_type is not "multihead":
            mean_attn, logvar_attn = self._latent_encoder(context_u, context_y)
        else:
            mean_attn, logvar_attn, attn_loss = self._latent_encoder(context_u, context_y)
        # mean_attn [B, :seq_len, D_h])
        ##=============##
        # allocation
        total_loss = 0
        # initialization
        input_ = self.input_drop(u)
        h, c = self.initialize_state_vectors(batch_size, first_obs=input_)

        # for all time steps
        for i in range(len(seq_len)):
            for t in range(seq_len[i]):
                # feature extraction: y_t
                phi_y_t = self.phi_y(y[:, :, t])
                # feature extraction: u_t
                phi_u_t = self.phi_u(u[:, :, t])

                # encoder: u_t, y_t, h_t -> z_t posterior
                encoder_input = torch.cat([phi_u_t, phi_y_t, h[-1]], dim=1)
                encoder_input = self.enc(encoder_input)

                enc_mean_t = torch.clamp(self.enc_mean(encoder_input), *self.output_clamp)
                enc_logvar_t = self.enc_logvar(encoder_input)
                enc_logvar_t = torch.clamp(nn.Softplus()(enc_logvar_t), *self.output_clamp)
                posterior_dist = Independent(Normal(enc_mean_t, enc_logvar_t.exp().sqrt()), 1)
                # prior: h_t -> z_t (for KLD loss)
                prior_input = h[-1]
                prior_input = self.prior(prior_input)

                prior_mean_t = self.prior_mean(prior_input)
                prior_logvar_t = self.prior_logvar(prior_input)
                prior_logvar_t = nn.Softplus()(prior_logvar_t)
                # prior = self.reparametrization(prior_mean_t, prior_logvar_t)

                # sampling and reparameterization: get a new z_t
                # temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
                # z_t = tdist.Normal.rsample(temp)
                z_t = self.reparametrization(enc_mean_t, enc_logvar_t)

                # feature extraction: z_t
                phi_z_t = self.phi_z(z_t)
                ### fake data for discriminator
                fake_phi_z_t = self.phi_z(noise[:, t, :])
                #####=====(new)======######

                # mean attention size :torch.Size([B, time, D_hid])
                c_t_mu = mean_attn[:, t, :]
                c_t_logvar = logvar_attn[:, t, :]
                # phi_z_t:torch.Size([B, D_hid])
                # c_final :torch.Size([B, D_hid])
                c_final = self.reparametrization(c_t_mu, c_t_logvar)

                #####================######
                # decoder: h_t, z_t -> y_t
                decoder_input = torch.cat([phi_z_t, c_final, h[-1]], dim=1)  ##(modified)
                _, dist = self._decode(decoder_input)

                attention_latent[:, t, :] = torch.cat([phi_z_t, c_final], dim=1)
                # fake latent dimension pass through the decoder
                decoder_input_fake = torch.cat([fake_phi_z_t, c_final, h[-1]], dim=1)  ##(modified)
                fake_y[:, t, :], _ = self._decode(decoder_input_fake)

                RNN_inputs = torch.cat([phi_u_t, phi_z_t, c_final], dim=1)
                # input of RNN module torch.Size([B, 3*H]) ==>[B,T,3*H]

                # recurrence: u_t+1, z_t -> h_t+1
                _, (h, c) = self._rnn(torch.unsqueeze(RNN_inputs, 1), (h, c))  ##(modified)Size [2, B, H]

                # computing the loss
                KLD = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)

                assert not torch.isnan(KLD)
                KLD_ATTN = self.kld_attention(c_t_mu, c_t_logvar)  ##(new)
                assert not torch.isnan(KLD_ATTN)
                # Create the observation model (generating distribution) p(y_t|z_t,h_t, c_t)

                log_p = dist.log_prob(y[:, :, t])
                loss_pred = log_p.mean()
                assert not torch.isnan(loss_pred)
                assert torch.isfinite(loss_pred)

                total_loss += - loss_pred + KLD + KLD_ATTN
                deterministic_hidden_state.append(h[-1])

        deterministic_hidden_state = torch.stack(deterministic_hidden_state).view(batch_size, -1, self.h_dim)
        hs = deterministic_hidden_state.permute(0, 2, 1)

        padded = nn.ConstantPad1d((0, reshaped_u.shape[1] - deterministic_hidden_state.shape[1]), 0)(hs)
        hs = padded.permute(0, 2, 1)

        # for Discriminator
        input_feature = torch.cat((attention_latent, reshaped_u, reshaped_y), dim=2)
        disc_real = self.discriminator(input_feature, seq_len)

        fake_input_feature = torch.cat((attention_latent, reshaped_u, fake_y), dim=2)
        disc_fake = self.discriminator(fake_input_feature, seq_len)

        d_loss = -torch.mean(disc_real) + torch.mean(disc_fake)

        if self.self_attention_type is not "multihead":
            return total_loss, d_loss, deterministic_hidden_state, input_feature, fake_input_feature
        else:
            return total_loss + attn_loss, d_loss, deterministic_hidden_state, input_feature, fake_input_feature, \
                   attention_latent

    def reparametrization(self, mu, log_var):

        # Reparametrization trick
        var = torch.exp(log_var * 0.5)
        eps = torch.FloatTensor(var.size()).normal_(mean=0, std=1).to(self.device)
        eps = torch.autograd.Variable(eps)

        return eps.mul(var).add_(mu).add_(1e-7)

    def _decode(self, x):
        x_decoder = self.decoder(x)
        # Get the mixing probabilities from the decoder
        mix_probs = x_decoder[:, : self.n_mixtures]
        # Return all (for every component) means and standard deviations from the decoder
        mu, sigma = x_decoder[:, self.n_mixtures:].chunk(2, dim=-1)

        sigma = nn.Softplus()(sigma)
        batch_n = mu.shape[0]
        mu = mu.view(batch_n, self.n_mixtures, self.y_dim)
        sigma = sigma.view(batch_n, self.n_mixtures, self.y_dim)

        # Ensure that the mixing probabilities are between zero and one and sum to one
        mix_probs = torch.nn.functional.softmax(mix_probs, dim=1)

        # Construct a batch of Gaussian Mixture Modles in input_shape-D consisting of
        # GMM_components equally weighted input_shape-D Gaussian distributions
        mix = Categorical(mix_probs)
        comp = Independent(Normal(mu, sigma), 1)
        gmm = MixtureSameFamily(mix, comp)
        # Return a distribution p(y_t|z_t,h_t)
        out = gmm.sample()
        return out, gmm

    def generate(self, u, seq_len=None):
        # get the batch size
        batch_size, D, T = u.shape
        hd = []
        # length of the sequence to generate
        # seq_len = u.shape[-1]
        if seq_len is None:
            seq_len = [torch.max((u[i, 0, :] != 0).nonzero()).item() + 1 for i in range(u.shape[0])]
        elif isinstance(seq_len, int):
            seq_len = [seq_len] * batch_size

        # allocation
        sample = torch.zeros(batch_size, self.y_dim, T, device=self.device)
        sample_mu = torch.zeros(batch_size, self.y_dim, T, device=self.device)
        sample_sigma = torch.zeros(batch_size, self.y_dim, T, device=self.device)

        ## use the learned initial state based on the first frame.
        input_ = self.input_drop(u)
        h, c = self.initialize_state_vectors(batch_size, first_obs=input_)
        # for all time steps
        for i in range(len(seq_len)):
            for t in range(seq_len[i]):
                # feature extraction: u_t+1
                phi_u_t = self.phi_u(u[:, :, t])

                # prior: h_t -> z_t
                prior_input = h[-1]
                prior_input = self.prior(prior_input)

                prior_mean_t = self.prior_mean(prior_input)
                prior_logvar_t = self.prior_logvar(prior_input)
                prior_logvar_t = nn.Softplus()(prior_logvar_t)

                # sampling and reparameterization: get new z_t
                # temp = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
                # z_t = tdist.Normal.rsample(temp)

                z_t = self.reparametrization(prior_mean_t, prior_logvar_t)
                # feature extraction: z_t
                phi_z_t = self.phi_z(z_t)
                ##=================(new)============##

                c_final = torch.distributions.Normal(torch.zeros_like(phi_z_t), torch.ones_like(phi_z_t)).rsample()

                ##==================================##
                # decoder: z_t, h_t -> y_t

                decoder_input = torch.cat([phi_z_t, c_final, h[-1]], dim=1)  ##(modified)

                sample[:, :, t], gmm = self._decode(decoder_input)

                sample_mu[:, :, t] = gmm.mean
                sample_sigma[:, :, t] = gmm.variance ** 0.5

                # recurrence: u_t+1, z_t -> h_t+1
                RNN_inputs = torch.cat([phi_u_t, phi_z_t, c_final], dim=1)
                # input(B,T,H )

                _, (h, c) = self._rnn(torch.unsqueeze(RNN_inputs, 1), (h, c))  ##(modified)
                hd.append(h[-1])

        hidden_state = torch.stack(hd).view(batch_size, -1, self.h_dim)
        hs = hidden_state.permute(0, 2, 1)

        padded = nn.ConstantPad1d((0, T - hidden_state.shape[1]), 0)(hs)
        hs = padded.permute(0, 2, 1)
        return sample, sample_mu, sample_sigma, hs

    @staticmethod
    def kld_gauss(mu_q, logvar_q, mu_p, logvar_p):
        # Goal: Minimize KL divergence between q_pi(z|xi) || p(z|xi)
        # This is equivalent to maximizing the ELBO: - D_KL(q_phi(z|xi) || p(z)) + Reconstruction term
        # This is equivalent to minimizing D_KL(q_phi(z|xi) || p(z))
        term1 = logvar_p - logvar_q - 1
        term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        kld = 0.5 * torch.sum(term1 + term2)
        return kld

    @staticmethod
    def kl_Normal_MixtureSameFamily(p: MixtureSameFamily, q: Normal):
        NUM_of_MC_SAMPLES = 10
        samples = q.sample(sample_shape=(NUM_of_MC_SAMPLES,))
        px = p.log_prob(samples)
        qx = q.log_prob(samples)
        return torch.mean(qx - px)

    @staticmethod
    def kld_attention(mu_attn, logvar_attn):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar_attn - mu_attn ** 2 - logvar_attn.exp(), dim=1), dim=0)
        return kld_loss


class DynamicModel(VRNN_GMM):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 h_dim=96,
                 z_dim=48,
                 n_layers=2,
                 n_mixtures=10,
                 sequence_length=200,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 learn_init_state=True,
                 bidirectional=False,
                 normalizer_input=None,
                 normalizer_output=None,
                 *args, **kwargs):
        super(DynamicModel, self).__init__(u_dim=num_inputs, y_dim=num_outputs, h_dim=h_dim, z_dim=z_dim,
                                           n_layers=n_layers, n_mixtures=n_mixtures, sequence_length=sequence_length,
                                           device=device, learn_init_state=learn_init_state,
                                           bidirectional=bidirectional)
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.args = args
        self.kwargs = kwargs
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output

        self.to(device)

    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def forward(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)

        vrnn_loss, d_loss, hidden, real_feature, fake_feature, attention_latent = super(DynamicModel, self).forward(u, y)
        return vrnn_loss, d_loss, hidden, real_feature, fake_feature, attention_latent

    def generate(self, u, seq_len=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)

        y_sample, y_sample_mu, y_sample_sigma, hidden = super(DynamicModel, self).generate(u, seq_len)

        if self.normalizer_output is not None:
            y_sample = self.normalizer_output.unnormalize(y_sample)
        if self.normalizer_output is not None:
            y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
        if self.normalizer_output is not None:
            y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)

        return y_sample, y_sample_mu, y_sample_sigma, hidden


class ModelState:
    """
    Container for all model related parameters and optimizer
    model
    optimizer
    """

    def __init__(self,
                 seed,
                 nu,
                 ny,
                 sequence_length,
                 h_dim=80,
                 z_dim=100,
                 n_layers=2,
                 n_mixtures=8,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 optimizer_type="MADGRAD",
                 **kwargs):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_mixtures = n_mixtures

        self.model = DynamicModel(num_inputs=nu, num_outputs=ny, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers,
                                  n_mixtures=n_mixtures, sequence_length=sequence_length, device=device, **kwargs)
        if optimizer_type == "AdaBelief":
            self.optimizer = torch_optimizer.AdaBelief(self.model.parameters(),
                                                       lr=1e-4,
                                                       betas=(0.9, 0.999),
                                                       eps=1e-6,
                                                       weight_decay=0
                                                       )
        elif optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=1e-4,
                                               betas=(0.9, 0.999),
                                               weight_decay=0.001
                                               )
        elif optimizer_type == "SGD":
            self.optimizer = torch_optimizer.SGDW(self.model.parameters(),
                                                  lr=1e-4,
                                                  momentum=0,
                                                  dampening=0,
                                                  weight_decay=1e-2,
                                                  nesterov=False,
                                                  )
        elif optimizer_type == "MADGRAD":
            self.optimizer = torch_optimizer.MADGRAD(self.model.parameters(),
                                                     lr=3e-4,
                                                     momentum=0.0,
                                                     weight_decay=0,
                                                     eps=1e-6,
                                                     )
        else:
            # Optimization parameters
            yogi = torch_optimizer.Yogi(self.model.parameters(),
                                        lr=0.5e-4,
                                        betas=(0.5, 0.999),
                                        eps=1e-3,
                                        initial_accumulator=1e-6,
                                        weight_decay=0, )

            self.optimizer = torch_optimizer.Lookahead(yogi, k=5, alpha=0.5)

    def load_model(self, path, name='VRNN_model.pt', map_location=None):
        file = path if os.path.isfile(path) else os.path.join(path, name)
        try:
            if map_location is None:
                ckpt = torch.load(file, map_location=lambda storage, loc: storage)
            else:
                ckpt = torch.load(file, map_location=map_location)
        except NotADirectoryError:
            raise Exception("Could not find model: " + file)
        loaded_state_dict = _strip_prefix_if_present(ckpt.pop("model"), prefix="module.")
        self.model.load_state_dict(loaded_state_dict, strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt['epoch']
        vloss = ckpt['vloss']
        return epoch, vloss

    def save_model(self, epoch, vloss, elapsed_time, path, name='VRNN_model.pt'):
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'epoch': epoch,
            'model': _strip_prefix_if_present(self.model.state_dict(), 'module.'),
            'optimizer': self.optimizer.state_dict(),
            'vloss': vloss,
            'elapsed_time': elapsed_time,
        },
            os.path.join(path, name))
