# Stick breaking Gaussian Mixture Prior for variational autoencoders
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Independent, Gamma, Beta, MixtureSameFamily
from torch.autograd import Variable
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import os

try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils_VAE import AverageMeter, ResidualBlock, LinearResidual, ResidualBlock_deconv, LayerNorm2d, CustomLinear
from torch import nn, optim

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

local_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SMALL = torch.tensor(np.finfo(np.float32).eps, dtype=torch.float64, device=local_device)
pi_ = torch.tensor(np.pi, dtype=torch.float64, device=local_device)
log_norm_constant = -0.5 * torch.log(2 * pi_)


def epsilon():
    global SMALL
    return SMALL


def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def calculate_layer_size(input_size, kernel_size, stride, padding=0):
    numerator = input_size - kernel_size + (2 * padding)
    denominator = stride
    return (numerator // denominator) + 1


def calculate_channel_sizes(image_channels, max_filters, num_layers):
    channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
    for i in range(1, num_layers):
        prev = channel_sizes[-1][-1]
        new = prev * 2
        channel_sizes.append((prev, new))
    return channel_sizes


def init_mlp(layer_sizes, stdev=.01, bias_init=0.):
    global local_device
    params = {'w': [], 'b': []}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(
            torch.nn.init.normal_(torch.empty(n_in, n_out, device=local_device), std=stdev).requires_grad_(True))
        params['b'].append(torch.empty(n_out, device=local_device).fill_(bias_init).requires_grad_(True))
    return params


def mlp(X, params):
    h = [X]
    activation = nn.Softplus(beta=0.5, threshold=20)
    for w, b in zip(params['w'][:-1], params['b'][:-1]):
        h.append(activation(torch.matmul(h[-1], w) + b))
    return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]


def mean_sum_samples(samples):
    n_dim = samples.dim()
    if n_dim == 4:
        return torch.mean(torch.sum(torch.sum(samples, axis=2), axis=2), axis=1)
    elif n_dim == 3:
        return torch.sum(torch.sum(samples, axis=-1), axis=-1)
    elif n_dim == 2:
        return torch.sum(samples, axis=-1)
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % n_dim)


class DistributionSample(object):
    __metaclass__ = ABCMeta

    def __init__(self, seed=1, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.srng = torch.Generator(device=self.device).manual_seed(seed)
        self.to(device=self.device)

    def set_seed(self, seed=1):
        self.srng = torch.Generator(device=self.device).manual_seed(seed)

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def log_likelihood(self):
        pass


class GammaSample(DistributionSample):
    """
    Gamma distribution
    (beta^alpha)/gamma * x^(alpha-1) * exp^(-beta*x)
    [Naesseth+ 2017]
    [ Marsaglia and Tsang, 2000]
    Rejection Sampling Variational Inference
    http://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
    """

    def __init__(self, iter_sampling=6, rejection_sampling=True, seed=1):
        super(GammaSample, self).__init__(seed)

        self.iter_sampling = iter_sampling
        self.rejection_sampling = rejection_sampling
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def gamma_sample(self, alpha, beta):
        _shape = alpha.shape
        alpha = alpha.flatten()
        output_sample = -torch.ones_like(alpha, device=self.device, dtype=alpha.dtype)
        index = torch.arange(output_sample.shape[0], device=self.device)
        under_one_idx = torch.gt(torch.ones_like(alpha), alpha[index])
        added_alpha = alpha.clone()
        added_alpha[under_one_idx] += 1
        U = torch.empty(under_one_idx[0].shape, device=self.device, dtype=alpha.dtype).uniform_(epsilon(),
                                                                                                1 - epsilon()).to(
            device=self.device)
        if self.rejection_sampling:
            # We don't use theano.scan in order to avoid to use updates.
            for _ in range(self.iter_sampling):
                output_sample, index = self._rejection_sampling(output_sample,
                                                                added_alpha,
                                                                index)
        else:
            output_sample = self._not_rejection_sampling(alpha)
        output_sample = torch.clamp(output_sample, torch.zeros_like(output_sample), output_sample)
        output_sample[under_one_idx] = (U ** (1 / alpha[under_one_idx])) * output_sample[under_one_idx]
        return output_sample.reshape(_shape) / beta

    def log_gamma_pdf(self, samples, alpha, beta):
        output = alpha * torch.log(beta + epsilon()) - torch.lgamma(alpha + epsilon())
        output = output + (alpha - 1) * torch.log(samples + epsilon())
        output = output - beta * samples
        return mean_sum_samples(output)

    def _h(self, alpha, eps):
        d = alpha - 1 / 3.
        c = torch.reciprocal(torch.sqrt(9 * d) + epsilon())
        v = torch.pow((1 + c * eps), 3)
        judge_1 = torch.exp(0.5 * torch.pow(eps, 2) + d - d * v + d * torch.log(v + epsilon()))
        judge_2 = -1 / c
        output = d * v
        return output, judge_1, judge_2

    def _rejection_sampling(self, output_z, alpha, idx):

        eps = torch.empty(idx.shape, device=self.device, dtype=alpha.dtype).normal_(0, 1, generator=self.srng).to(
            device=self.device)
        U = torch.empty(idx.shape, device=self.device, dtype=alpha.dtype).uniform_(epsilon(), 1 - epsilon(),
                                                                                   generator=self.srng).to(
            device=self.device)
        z, judge1, judge2 = self._h(alpha[idx], eps)
        _idx_binary = torch.logical_and(torch.lt(U, judge1), torch.gt(eps, judge2))

        output_z[idx[_idx_binary.nonzero()]] = z[_idx_binary.nonzero()]
        # update idx
        idx = idx[torch.eq(torch.zeros_like(_idx_binary), _idx_binary).squeeze(-1)]
        return output_z, idx

    def _not_rejection_sampling(self, alpha):
        eps = torch.empty(alpha.shape, device=self.device).normal_(0, 1, generator=self.srng).type(alpha.dtype)
        z, _, _ = self._h(alpha, eps)
        return z


class BetaSample(GammaSample):
    """
    Beta distribution
    x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
    """

    def __init__(self, iter_sampling=6, rejection_sampling=True, seed=10):
        super(BetaSample,
              self).__init__(iter_sampling=iter_sampling,
                             rejection_sampling=rejection_sampling,
                             seed=seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def beta_sample(self, alpha, beta):
        z_1 = super(BetaSample,
                    self).gamma_sample(alpha, torch.ones_like(alpha, device=self.device, dtype=alpha.dtype))

        z_2 = super(BetaSample,
                    self).gamma_sample(beta, torch.ones_like(beta, device=self.device, dtype=beta.dtype))

        return torch.div(z_1, (z_1 + z_2) + epsilon())

    def log_beta_pdf(self, samples, alpha, beta):
        output = (alpha - 1) * torch.log(samples + epsilon())
        output = output + (beta - 1) * torch.log(1 - samples + epsilon())
        output = output - self._log_beta_func(alpha, beta)
        return mean_sum_samples(output)

    def _log_beta_func(self, alpha, beta):
        return torch.lgamma(alpha + epsilon()) + torch.lgamma(beta + epsilon()) - torch.lgamma(alpha + beta + epsilon())


def I_function(alpha_q, beta_q, alpha_p, beta_p):
    return - torch.div((alpha_p * beta_p), alpha_q + epsilon()) - \
           beta_q * torch.log(alpha_q + epsilon()) - torch.lgamma(beta_q + epsilon()) + \
           (beta_q - 1) * torch.digamma(beta_p + epsilon()) + \
           (beta_q - 1) * torch.log(alpha_p + epsilon())


def gamma_kl_loss(a, b, c, d):
    a = torch.reciprocal(a + epsilon())
    c = torch.reciprocal(c + epsilon())
    losses = I_function(c, d, c, d) - I_function(a, b, c, d)
    return torch.sum(losses, dim=1)


class VAEEncoder(nn.Module):
    def __init__(self, nchannel, z_dim, hidden_dim, img_width, max_filters=512, num_layers=4, small_conv=False,
                 norm_type='batch', num_groups=1, activation=nn.PReLU()):
        super(VAEEncoder, self).__init__()
        self.nchannel = nchannel
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.img_width = img_width
        self.enc_kernel = 4
        self.enc_stride = 2
        self.enc_padding = 3
        self.res_kernel = 3
        self.res_stride = 1
        self.res_padding = 1
        self.activation = activation
        # ENCODER-CONVOLUTION LAYERS
        if small_conv:
            num_layers += 1
        channel_sizes = calculate_channel_sizes(
            self.nchannel, max_filters, num_layers
        )

        # Encoder
        encoder_layers = nn.ModuleList()
        # Encoder Convolutions
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                # 1x1 Convolution
                encoder_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.enc_kernel,
                    stride=self.enc_stride,
                    padding=self.enc_padding,
                ))
            else:
                encoder_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.enc_kernel,
                    stride=self.enc_stride,
                    padding=self.enc_padding,
                    bias=False,
                ))
            # Batch Norm
            if norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'layer':
                encoder_layers.append(nn.GroupNorm(num_groups, out_channels))

            # ReLU
            encoder_layers.append(self.activation)

            if (i == num_layers // 2):
                # add a residual Layer
                encoder_layers.append(ResidualBlock(
                    out_channels,
                    self.res_kernel,
                    self.res_stride,
                    self.res_padding,
                    nonlinearity=self.activation
                ))

        # Flatten Encoder Output
        encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate shape of the flattened image
        self.h_dim, self.h_image_dim = self.get_flattened_size(self.img_width)
        # linear layers
        layers = []
        layers.append(nn.Linear(self.h_dim, hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)

        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)

        self.linear_layers = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dim, z_dim, bias=False)

        # variance of z

        self.fc_log_var = nn.Linear(hidden_dim, z_dim, bias=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def forward(self, X):

        # Encode
        h = self.encoder(X)
        # Get latent variables
        self.hlayer = self.linear_layers(h)
        # mean_z
        mu_z = self.fc_mu(self.hlayer)

        # logvar_z
        logvar_z = self.fc_log_var(self.hlayer)
        z = self.reparameterize(mu_z, logvar_z)
        return z, mu_z, logvar_z

    def reparameterize(self, mu, log_var):
        eps = torch.randn_like(log_var, device=self.device)
        return torch.add(mu, torch.mul(log_var.mul(0.5).exp_(), eps))

    def get_flattened_size(self, image_size):
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                filters = layer.out_channels
                image_size = calculate_layer_size(
                    image_size, kernel_size, stride, padding
                )
        return filters * image_size * image_size, image_size


class VAEDecoder(nn.Module):
    def __init__(self, nchannel, z_dim, hidden_dim, extend_dim, h_image_dim, img_width, max_filters=512, num_layers=4,
                 small_conv=False, norm_type='batch', num_groups=1, activation=nn.PReLU()):
        super(VAEDecoder, self).__init__()
        self.nchannel = nchannel
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.img_width = img_width
        self.dec_kernel = 4
        self.dec_stride = 2
        self.dec_padding = 3
        self.res_kernel = 3
        self.res_stride = 1
        self.res_padding = 1
        self.activation = activation
        # Px_given_z LAYERS Decoder P(X|Z)

        if small_conv:
            num_layers += 1
        channel_sizes = calculate_channel_sizes(
            self.nchannel, max_filters, num_layers
        )

        # Decoder
        decoder_layers = nn.ModuleList()
        # Feedforward/Dense Layer to expand our latent dimensions
        decoder_layers.append(nn.Linear(z_dim, hidden_dim))
        if norm_type == 'batch':
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            decoder_layers.append(nn.LayerNorm(hidden_dim))

        decoder_layers.append(self.activation)
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

        if norm_type == 'batch':
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            decoder_layers.append(nn.LayerNorm(hidden_dim))

        decoder_layers.append(self.activation)
        decoder_layers.append(torch.nn.Linear(hidden_dim, extend_dim, bias=False))
        if norm_type == 'batch':
            decoder_layers.append(nn.BatchNorm1d(extend_dim))
        elif norm_type == 'layer':
            decoder_layers.append(nn.LayerNorm(extend_dim))

        decoder_layers.append(self.activation)
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(
            nn.Unflatten(1, (int(extend_dim / (h_image_dim * h_image_dim)), h_image_dim, h_image_dim)))
        # Decoder Convolutions

        for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
            if small_conv and i == num_layers - 1:
                # 1x1 Transposed Convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.dec_kernel,
                        stride=self.dec_stride,
                        padding=self.dec_padding,
                    )
                )
            else:
                # Add Transposed Convolutional Layer
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.dec_kernel,
                        stride=self.dec_stride,
                        padding=self.dec_padding,
                        bias=False,
                    )
                )
            if norm_type == 'batch':
                decoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'layer':
                decoder_layers.append(nn.GroupNorm(num_groups, out_channels))

            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(self.activation)
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
            if (i == num_layers // 2):
                # add a residual Layer
                decoder_layers.append(
                    ResidualBlock_deconv(
                        out_channels,
                        self.res_kernel,
                        self.res_stride,
                        self.res_padding,
                        nonlinearity=self.activation
                    )
                )

        self.decoder = nn.Sequential(*decoder_layers)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
        # print(f"decode network:\n{self.decoder}")

    def forward(self, x):
        return self.decoder(x)


class VAECritic(nn.Module):
    # define the descriminator/critic
    def __init__(self, input_dims, num_layers=4, norm_type='layer', activation=nn.LeakyReLU(0.1)):
        super(VAECritic, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        layers = []
        layers.append(nn.Linear(input_dims, input_dims * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(input_dims * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(input_dims * 2))
        # Activation Function
        layers.append(self.activation)
        size = input_dims * 2
        # Fully Connected Block
        for i in range(num_layers - 2):
            # residual feedforward Layer
            layers.append(nn.Linear(size, size // 2))

            if self.norm_type == 'batch':
                layers.append(nn.BatchNorm1d(size // 2))
            elif self.norm_type == 'layer':
                layers.append(nn.LayerNorm(size // 2))

            layers.append(self.activation)
            if (i == (num_layers // 2 - 1)):
                # add a residual block
                LinearResidual(size // 2)
            size = size // 2
        layers.append(nn.Linear(size, size * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(size * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(size * 2))

        # Activation Function
        layers.append(self.activation)
        # add anther residual block
        LinearResidual(size * 2)
        layers.append(nn.Linear(size * 2, 1))
        self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def forward(self, x):
        return self.model(x)


class LinearBN(nn.Module):
    def __init__(self, in_features, out_features, bias=False, activation=nn.LeakyReLU(0.1), norm_type='layer'):
        super(LinearBN, self).__init__()
        self.norm_type = norm_type
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.norm_type == 'batch':
            self.bn = nn.BatchNorm1d(out_features)
        elif self.norm_type == 'layer':
            self.bn = nn.LayerNorm(out_features)

        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)  # or any other activation function
        return x


class GMMVAE(nn.Module):
    def __init__(self, number_of_mixtures, nchannel, z_dim, w_dim, hidden_dim, device, img_width, batch_size,
                 max_filters=512, num_layers=4, norm_type='layer', small_conv=False, use_mse_loss=False):
        super(GMMVAE, self).__init__()
        self.use_mse_loss = use_mse_loss
        self.K = number_of_mixtures
        self.nchannel = nchannel
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.img_width = img_width
        self.batch_size = batch_size
        self.enc_kernel = 4
        self.enc_stride = 2
        self.enc_padding = 0
        self._to_linear = None
        self.norm_type = norm_type

        self.encoder = VAEEncoder(nchannel, z_dim, hidden_dim, img_width, max_filters, num_layers, small_conv,
                                  norm_type=self.norm_type)
        self.decoder = VAEDecoder(nchannel, z_dim, hidden_dim, self.encoder.h_dim, self.encoder.h_image_dim, img_width,
                                  max_filters, num_layers, small_conv, norm_type=self.norm_type)

        # THE LATENT SPACE (W)
        self.encoder_w = CustomLinear(hidden_dim=[img_width * img_width * nchannel, hidden_dim, hidden_dim],
                                      norm_type=self.norm_type, last_activation=nn.LeakyReLU(0.1), flatten=True)

        # mean of w
        self.encoder_w_mean = CustomLinear(hidden_dim=[hidden_dim, w_dim], norm_type=self.norm_type,
                                           last_activation=nn.LeakyReLU(0.1), flatten=False)

        # logvar_w
        self.encoder_w_logvar = CustomLinear(hidden_dim=[hidden_dim, w_dim], norm_type=self.norm_type,
                                             last_activation=nn.Softplus(), flatten=False)

        # number of mixtures (c parameter) gets the input from feedforward layers of w and convolutional layers of z
        self.encoder_c = CustomLinear(hidden_dim=[2 * hidden_dim, hidden_dim, number_of_mixtures],
                                      norm_type=self.norm_type, last_activation=nn.Softmax(dim=-1), flatten=False)

        # (GENERATOR)
        # CreatePriorGenerator_Given Z LAYERS P(z|w,c)
        self.encoder_z_given_w = CustomLinear(hidden_dim=[w_dim, hidden_dim], norm_type=self.norm_type,
                                              last_activation=nn.Tanh(), flatten=False)

        self.pz_wc_mean = nn.ModuleList([LinearBN(self.hidden_dim, self.z_dim, bias=False) for i in range(self.K)])
        self.pz_wc_logvar = nn.ModuleList(
            [LinearBN(self.hidden_dim, self.z_dim, bias=False, activation=nn.Softplus()) for i in range(self.K)])

        self.encoder_kumar_a = init_mlp([hidden_dim, self.K - 1], 1e-7)  # Q(pi|x)
        self.encoder_kumar_b = init_mlp([hidden_dim, self.K - 1], 1e-7)  # Q(pi|x)

        # create prior over alpha variable of stick-breaking
        # psterior distribution of Gamma distribution variables
        self.encoder_gamma_a = init_mlp([hidden_dim, self.K - 1], 1e-8)  # Q(pi|x)
        self.encoder_gamma_b = init_mlp([hidden_dim, self.K - 1], 1e-8)  # Q(pi|x)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def GMM_decoder(self, z_input):
        # Decoder:
        # Generate X: P(X|Z)
        # TODO the reconstruction is outputting 94 * 94, we manually pad it to 100 * 100
        Generated_X = self.decoder(z_input)
        # Generated_X = F.pad(input=Generated_X, pad=(1, 1, 1, 1), mode='constant', value=0)
        if self.use_mse_loss:
            Generated_X = Generated_X - 0.5  # map to [-0.5, 0.5]
        return Generated_X

    def GMM_encoder(self, data):
        # 1) posterior Q(z|X)
        """
        compute z = z_mean + z_var * eps1
        """
        h = data
        # 1) Cmpute posterior of latent variables Q(Z|x) --->Encoder  (mean_z, logvar_z)
        z_x, z_x_mean, z_x_logvar = self.encoder(h)
        z_x_sigma = torch.exp(0.5 * z_x_logvar)
        # 2) posterior Q(w|X)
        # Create Recogniser for W
        hw = self.encoder_w(h)
        # mean of W
        w_x_mean = self.encoder_w_mean(hw)
        # variance of Q(w|x) distribution

        # log(variance of W)
        logvar_w = self.encoder_w_logvar(hw)
        w_x_sigma = torch.exp(0.5 * logvar_w)
        # Build a two-layer-MLP to compute Q(w|x)
        w_x = self.encoder.reparameterize(w_x_mean, logvar_w)

        # 3) posterior P(c|w,z)=Q(c|X)
        # posterior distribution of P(c|w,z^{i})=Q(c|x) where z^{i}~q(z|x)
        # combine hidden layers after convolutional layers for Z latent space and feed forward layers of W latent space
        hc = torch.cat([self.encoder.hlayer, hw], 1)
        c_posterior = self.encoder_c(hc).to(device=self.device)

        # 4) posterior of Kumaraswamy given input images
        # P(kumar_a,kumar_b|X)

        self.kumar_a = torch.exp(mlp(self.encoder.hlayer, self.encoder_kumar_a))
        self.kumar_b = torch.exp(mlp(self.encoder.hlayer, self.encoder_kumar_b))

        # 5) posterir of gamma
        # P(alpha, beta| X)
        self.gamma_alpha = torch.exp(mlp(self.encoder.hlayer, self.encoder_gamma_a))
        self.gamma_beta = torch.exp(mlp(self.encoder.hlayer, self.encoder_gamma_b))

        return w_x, w_x_mean, w_x_sigma, z_x, z_x_mean, z_x_sigma, c_posterior

    def GMM_prior(self, w_x, post_c):

        # priorGenerator(w_sample)
        # prior generator from mixture of Gaussians P(Z|w,c)
        # P(z_i|w,c_i)
        h = self.encoder_z_given_w(w_x)
        z_wc_mean_list = list()
        for module in self.pz_wc_mean:
            z_wc_mean_list.append(module(h))

        z_wc_logvar_list, z_wc_sigma_list = list(), list()

        for module in self.pz_wc_logvar:
            Pz_given_wc_logvar = module(h)
            z_wc_logvar_list.append(Pz_given_wc_logvar)
            Pz_given_wc_sigma = torch.exp(Pz_given_wc_logvar / 2) + epsilon()
            z_wc_sigma_list.append(Pz_given_wc_sigma)

        z_wc_mean = torch.stack(z_wc_mean_list, dim=1)  # [ batch_size, K, z_dim]
        z_wc_logvar = torch.stack(z_wc_logvar_list, dim=1)  # [batch_size, K, z_dim]
        z_wc_sigma = torch.stack(z_wc_sigma_list, dim=1)
        # categorical variable y from the Gumbel-Softmax distribution
        self.p_c_given_z = Categorical(logits=post_c)  # [batch_size,  K]

        # print(z_wc_mean.shape, post_c.shape)
        comp = Independent(Normal(z_wc_mean, z_wc_sigma), 1)
        gmm = MixtureSameFamily(self.p_c_given_z, comp)
        # pdb.set_trace()
        return gmm, z_wc_mean.permute(1, 0, 2), z_wc_logvar.permute(1, 0, 2)  # [k, batch_size, z_dim]

    def get_trainable_parameters(self):
        params = []
        for module in self.modules():
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                for sub_module in module:
                    params += [p for p in sub_module.parameters() if p.requires_grad]
            else:
                params += [p for p in module.parameters() if p.requires_grad]
        return params


def compute_nll(x, x_recon_linear):
    return F.binary_cross_entropy_with_logits(x_recon_linear, x)


def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    # KL divergence between two gaussian distributions
    d = torch.sub(mu_post, mu_prior)
    temp = (0.5 * torch.div(d.pow(2) + sigma_post.pow(2), sigma_prior.pow(2)) - 0.5 + torch.log(
        torch.div(sigma_prior, sigma_post))).mean(dim=0)
    return temp.sum()


def to_tensor(x, dType):
    global local_device
    if isinstance(x, torch.Tensor):
        return x.to(local_device).to(dType)
    else:
        return torch.tensor(x, dtype=torch.float64, device=local_device)


def beta_fn(a, b):
    global local_device
    ta = to_tensor(a, torch.float64)
    tb = to_tensor(b, torch.float64)

    return torch.exp(torch.lgamma(ta + epsilon()) + torch.lgamma(tb + epsilon()) - torch.lgamma(ta + tb + epsilon()))


def gradient_penalty(critic, real, fake, device):
    # wasserstein distance with gradient penalty model
    BATCH_SIZE = real.shape[0]
    # one epsilon value for each
    epsilon = torch.rand((BATCH_SIZE, 1)).to(device)
    epsilon = epsilon.expand_as(real)
    # Get random interpolation between real and fake samples
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    # set it to require grad info
    interpolated_images = torch.autograd.Variable(interpolated_images, requires_grad=True).to(device)
    # calculate critic score
    mixed_scores = critic(interpolated_images)
    # calculate gradient of mixed score w.r.t. interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean(torch.pow(gradient_norm - 1, 2))
    return gradient_penalty


def compute_kumar2beta_kld(a, b, alpha, beta):
    global local_device

    EULER_GAMMA = torch.tensor(0.5772156649015328606, dtype=torch.float, device=local_device)

    upper_limit = 10000.0

    ab = torch.mul(a, b)
    a_inv = torch.reciprocal(a)
    b_inv = torch.reciprocal(b)

    # compute taylor expansion for E[log (1-v)] term
    kl = torch.mul(torch.pow(1 + ab, -1), torch.clamp(beta_fn(a_inv, b), epsilon(), upper_limit))

    for idx in range(10):
        kl = kl + torch.mul(torch.pow(idx + 2 + ab, -1),
                            torch.clamp(beta_fn(torch.mul(idx + 2., a_inv), b), epsilon(), upper_limit))
    kl = torch.mul(torch.mul(beta - 1, b), kl)

    # psi_b = torch.log(b + SMALL) - 1. / (2 * b + SMALL) -    1. / (12 * b**2 + SMALL)
    psi_b = torch.digamma(b + epsilon())
    kl = kl + torch.mul(torch.div(a - alpha, torch.clamp(a, epsilon(), upper_limit)), -EULER_GAMMA - psi_b - b_inv)

    # add normalization constants
    kl = kl + torch.log(torch.clamp(ab, epsilon(), upper_limit)) + torch.log(
        torch.clamp(beta_fn(alpha, beta), epsilon(), upper_limit))

    #  final term
    kl = kl + torch.div(-(b - 1), torch.clamp(b, epsilon(), upper_limit))

    # pdb.set_trace()
    return torch.clamp(kl, min=0.)


def log_normal_pdf(sample, mean, sigma):
    global local_device
    global SMALL
    d = torch.sub(sample, mean).to(device=local_device)
    d2 = torch.mul(-1, torch.mul(d, d)).to(device=local_device)
    s2 = torch.mul(2, torch.mul(sigma, sigma)).to(device=local_device)
    return torch.sum(torch.div(d2, s2 + SMALL) - torch.log(
        torch.mul(sigma, torch.sqrt(2 * torch.tensor(pi_, dtype=torch.float, device=local_device))) + SMALL), dim=1)


def log_beta_pdf(v, alpha, beta):
    global SMALL
    return torch.sum((alpha - 1) * torch.log(v + SMALL) + (beta - 1) * torch.log(1 - v + SMALL) - torch.log(
        beta_fn(alpha, beta) + SMALL), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    global SMALL
    return torch.sum(
        torch.mul(a - 1, torch.log(v + SMALL)) + torch.mul(b - 1, torch.log(1 - torch.pow(v, a) + SMALL)) + torch.log(
            a + SMALL) + torch.log(b + SMALL), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    global SMALL
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in range(K - 1):
        s = s + torch.mul(pi_samples[k + 1], torch.exp(log_normal_pdf(z[k + 1], mu[k + 1], sigma[k + 1])))
    return -torch.log(s + SMALL)


def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    global local_device
    # Sample from Gumbel
    U = torch.rand(logits.shape).to(device=local_device)
    g = -Variable(torch.log(-torch.log(U + eps) + eps))
    # Gumbel-Softmax sample
    y = logits + g
    return nn.Softmax(dim=-1)(y / temperature).to(device=local_device)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim):
    """
    https://github.com/dev4488/VAE_gumble_softmax
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


class ElboReturn:
    def __init__(self, loss_dict, z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean,
                 w_posterior_sigma, dist, z_prior_mean, z_prior_logvar, x_reconst):
        self.loss_dict = loss_dict
        self.z_posterior = z_posterior
        self.z_posterior_mean = z_posterior_mean
        self.z_posterior_sigma = z_posterior_sigma
        self.c_posterior = c_posterior
        self.w_posterior_mean = w_posterior_mean
        self.w_posterior_sigma = w_posterior_sigma
        self.dist = dist
        self.z_prior_mean = z_prior_mean
        self.z_prior_logvar = z_prior_logvar
        self.x_reconst = x_reconst


# Gaussian Mixture Model VAE Class
class InfGaussMMVAE(GMMVAE, BetaSample):
    # based on this implementation :https://github.com/enalisnick/mixture_density_VAEs
    def __init__(self, hyperParams, K, nchannel, z_dim, w_dim, hidden_dim, device, img_width, batch_size, num_layers=4,
                 include_elbo2=True, use_mse_loss=False):
        self.prior_nu = None
        global local_device
        local_device = device
        super(InfGaussMMVAE, self).__init__(K, nchannel, z_dim, w_dim, hidden_dim, device, img_width, batch_size,
                                            max_filters=512, num_layers=num_layers, small_conv=False,
                                            use_mse_loss=use_mse_loss)
        BetaSample.__init__(self)

        self.priors = {"alpha": hyperParams['prior_alpha'], "beta": hyperParams['prior_beta']}
        self.K = K
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_width
        self.include_elbo2 = include_elbo2
        self.to(device=self.device)
        self.use_mse_loss = use_mse_loss

    # def __len__(self):
    #     return len(self.X)

    def forward(self, X):
        # init variational params

        w_x, w_x_mean, w_x_sigma, z_x, z_x_mean, z_x_sigma, c_posterior = self.GMM_encoder(X)
        gmm, z_wc_mean_prior, z_wc_logvar_prior = self.GMM_prior(w_x, c_posterior)
        # prior_z = gmm.sample()
        x_reconstructed = self.GMM_decoder(z_x)
        self.prior_nu = self.gamma_sample(self.gamma_alpha, self.gamma_beta)

        return z_x, z_x_mean, z_x_sigma, c_posterior, w_x_mean, w_x_sigma, gmm, z_wc_mean_prior, z_wc_logvar_prior, \
               x_reconstructed

    def compose_stick_segments(self, v):
        segments = []
        self.remaining_stick = [torch.ones((v.shape[0], 1)).to(device=self.device)]
        for i in range(self.K - 1):
            curr_v = torch.unsqueeze(v[:, i], 1)
            segments.append(torch.mul(curr_v, self.remaining_stick[-1]))
            self.remaining_stick.append(torch.mul(1 - curr_v, self.remaining_stick[-1]))
        segments.append(self.remaining_stick[-1])
        return segments

    def GenerateMixtures(self):
        """
        #KL divergence P(c|z,w)=Q(c|x) while P(c|pi) is the prior
        """
        global SMALL
        a_inv = torch.pow(self.kumar_a, -1)
        b_inv = torch.pow(self.kumar_b, -1)
        # compose into stick segments using pi = v \prod (1-v)
        v_means = torch.mul(self.kumar_b, beta_fn(1. + a_inv, self.kumar_b)).to(device=self.device)
        # u       = (r1 - r2) * torch.rand(a_inv.shape[0],self.K-1) + r2
        u = torch.distributions.uniform.Uniform(low=SMALL, high=1 - SMALL).sample([1]).squeeze()
        v_samples = torch.pow(1 - torch.pow(u, b_inv), a_inv).to(device=self.device)
        if v_samples.ndim > 2:
            v_samples = v_samples.squeeze()
        v0 = v_samples[:, -1].pow(0).reshape(v_samples.shape[0], 1)
        v1 = torch.cat([v_samples[:, :self.z_dim - 1], v0], dim=1)
        n_samples = v1.size()[0]
        n_dims = v1.size()[1]
        self.pi_samples = torch.zeros((n_samples, n_dims)).to(device=self.device)

        for k in range(n_dims):
            if k == 0:
                self.pi_samples[:, k] = v1[:, k]
            else:
                self.pi_samples[:, k] = v1[:, k] * torch.stack([(1 - v1[:, j]) for j in range(n_dims) if j < k]).prod(
                    axis=0)

        return self.pi_samples

    def DiscreteKL(self, P, Q, epsilon=1e-8):

        # KL(q(z)||p(z)) =  - sum_k q(k) log p(k)/q(k)
        # let's p(k) = 1/K???
        logQ = torch.log(Q + epsilon).to(device=self.device)

        logP = torch.log(P + epsilon).to(device=self.device)
        element_wise = (Q * torch.sub(logQ, logP))
        # pdb.set_trace()
        return torch.sum(element_wise, dim=-1).mean().to(device=self.device)

    def ExpectedKLDivergence(self, q_c, mean_z, logvar_z, mean_mixture, logvar_mixture):

        # 4. E_p(c|w,z)[KL(q(z)|| p(z|c,w))]
        z_wc = mean_z.unsqueeze(-1)
        z_wc = z_wc.expand(-1, self.z_dim, self.K)
        z_wc = z_wc.permute(2, 0, 1)

        logvar_zwc = logvar_z.unsqueeze(-1)
        logvar_zwc = logvar_zwc.expand(-1, self.z_dim, self.K)
        logvar_zwc = logvar_zwc.permute(2, 0, 1)
        KLD_table = 0.5 * (((logvar_mixture - logvar_zwc) + (
                (logvar_zwc.exp() + (z_wc - mean_mixture).pow(2)) / logvar_mixture.exp())) - 1)
        KLD_table = KLD_table.permute(0, 2, 1)

        qc = q_c.unsqueeze(-1)
        qc = qc.expand(-1, self.K, 1)
        qc = qc.permute(1, 0, 2)
        return torch.sum(torch.bmm(KLD_table, qc)).to(device=self.device)

    def get_ELBO(self, X):
        z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean, w_posterior_sigma, dist, \
        z_prior_mean, z_prior_logvar, x_reconst = self.forward(X)

        loss_dict = OrderedDict()
        # 1) Computes the KL divergence between two categorical distributions
        PriorC = self.GenerateMixtures()

        # this term KL divergence of two discrete distributions
        elbo1 = self.DiscreteKL(PriorC, c_posterior)

        # compose elbo of Kumaraswamy-beta
        elbo2 = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.include_elbo2:
            for k in range(self.K - 1):
                elbo2 = elbo2 - compute_kumar2beta_kld(self.kumar_a[:, k], self.kumar_b[:, k], self.prior_nu[:, k],
                                                       (self.K - 1 - k) * self.prior_nu[:, k]).mean()

        # 3)need this term of w (context)
        # 0.5 * sum(1 + 2*log(sigma) - mu^2 - sigma^2)
        elbo3 = -0.5 * torch.sum(
            1 + 2 * torch.log(w_posterior_sigma) - w_posterior_mean.pow(2) - w_posterior_sigma.pow(2)).to(
            device=self.device)  # VAE_KLDCriterion

        # 4)compute E_{p(w|x)p(c|x)}[D_KL(Q(z|x)||p(z|c,w))]

        elbo4 = self.ExpectedKLDivergence(c_posterior, z_posterior_mean, 2 * torch.log(z_posterior_sigma), z_prior_mean,
                                          z_prior_logvar)

        # 5) compute Reconstruction Cost = E_{q(z|x)}[P(x|z)]
        #
        if self.use_mse_loss:
            criterion = nn.MSELoss(reduction='sum')
        else:
            criterion = nn.BCELoss(reduction='sum')

        elbo5 = criterion(x_reconst.reshape(-1, self.nchannel * self.img_size * self.img_size),
                          X.reshape(-1, self.nchannel * self.img_size * self.img_size))

        assert torch.isfinite(elbo5)
        # 4)compute D_KL(Q(nu|x)||p(nu|alpha,beta)) --> gamma distribution
        elbo6 = gamma_kl_loss(self.gamma_alpha, self.gamma_beta,
                              self.priors['alpha'] * torch.ones_like(self.gamma_alpha, device=self.device,
                                                                     requires_grad=False),
                              self.priors['beta'] * torch.ones_like(self.gamma_beta, device=self.device,
                                                                    requires_grad=False)).mean()

        loss_dict['recon'] = elbo5
        loss_dict['c_cluster_kld'] = elbo1
        loss_dict['kumar2beta_kld'] = elbo2
        loss_dict['w_context_kld'] = elbo3
        loss_dict['z_latent_space_kld'] = elbo4
        loss_dict['gamma_params_kld'] = elbo6
        # 6.)  CV = H(C|Z, W) = E_q(z,w) [ E_p(c|z,w)[ - log P(c|z,w)] ]
        # Conditional entropy loss

        logits = self.p_c_given_z.logits
        probs = F.softmax(logits, dim=-1)
        loss_dict['CV_entropy'] = (- probs * torch.log(probs)).sum()

        # print(f" device of CV Etropy: {loss_dict['CV_entropy'].get_device()}")
        # loss_dict['loss'] = elbo1 + elbo2 + elbo3 + elbo4 + elbo5
        # excluding the KL loss terms of latent dimension Z since we have a dicriminator to take care of it.
        if self.include_elbo2:
            loss_dict['loss'] = elbo1 + elbo2 + elbo3 + elbo5 + elbo6
        else:
            loss_dict['loss'] = elbo1 + elbo3 + elbo5 + elbo6
        return ElboReturn(loss_dict, z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean,
                          w_posterior_sigma, dist, z_prior_mean, z_prior_logvar, x_reconst)

    def get_log_margLL(self, X, X_reconst, z_post, z_post_mean, z_post_sigma, gmm, z_prior_mean, z_prior_logvar,
                       w_post_mean, w_post_sigma):
        batchSize = X.shape[0]
        a_inv = torch.pow(self.kumar_a, -1)
        b_inv = torch.pow(self.kumar_b, -1)
        # compute Kumaraswamy samples

        uni_samples = torch.FloatTensor(a_inv.shape[0], self.K - 1).uniform_(epsilon(), 1 - epsilon()).to(
            device=self.device)
        # Samples from a two-parameter Kumaraswamy distribution with a, b parameters. Or equivalently,
        # U ~ U(0,1)
        # X = (1 - (1 - U)^(1 / b))^(1 / a)
        #  based on https://arxiv.org/pdf/1905.12052.pdf
        # https://github.com/astirn/MV-Kumaraswamy/blob/master/mv_kumaraswamy_sampler.py
        v_samples = torch.pow(1 - torch.pow(1 - uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index

        uni_samples = torch.FloatTensor(a_inv.shape[0], self.K).uniform_(epsilon(), 1 - epsilon()).to(
            device=self.device)
        gumbel_samples = -torch.log(-torch.log(uni_samples + epsilon()) + epsilon())

        component_samples = torch.argmax(torch.log(torch.cat(self.pi_samples, 1) + epsilon()) + gumbel_samples, 1).to(
            device=self.device, dtype=torch.int64)
        # print(f'index of stick should be used for the mixture model:\n{component_samples}')
        component_samples = torch.cat(
            [torch.arange(0, batchSize, device=self.device).unsqueeze(1), component_samples.unsqueeze(1)], 1)

        # calc likelihood term for chosen components
        ll = -compute_nll(X, X_reconst)
        # calc prior terms
        all_log_gauss_priors = []

        for k in range(self.K):
            all_log_gauss_priors.append(
                log_normal_pdf(gmm.sample(), z_prior_mean[k], torch.exp(z_prior_logvar[k] / 2) + epsilon()))

        all_log_gauss_priors = torch.stack(all_log_gauss_priors).to(device=self.device)
        # print(f"size of all components of Gaussian prior: {all_log_gauss_priors.size()}")
        # print(f"size of components of Z: {component_samples.size()}")
        log_gauss_prior = gather_nd(torch.t(all_log_gauss_priors), component_samples)
        log_gauss_prior = log_gauss_prior.unsqueeze(1)

        log_gauss_post = log_normal_pdf(z_post, z_post_mean, z_post_sigma)
        # ****need this term :
        # log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        log_beta_prior = log_beta_pdf(v_samples[:, 0].unsqueeze(1), self.prior_nu[:, 0].unsqueeze(1),
                                      (self.K - 1) * self.prior_nu[:, 0].unsqueeze(1))
        for k in range(self.K - 2):
            # log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])
            log_beta_prior = log_beta_prior + log_beta_pdf(v_samples[:, k + 1].unsqueeze(1),
                                                           self.prior_nu[:, k + 1].unsqueeze(1),
                                                           (self.K - 2 - k) * self.prior_nu[:, k + 1].unsqueeze(1))

        # ****need this term :calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)
        # ******compute likelihod terms of gamma distribution
        log_gamma_post = self.log_gamma_pdf(self.prior_nu, self.gamma_alpha, self.gamma_beta)

        log_gamma_prior = self.log_gamma_pdf(self.prior_nu, self.priors['alpha'] * torch.ones_like(self.gamma_alpha,
                                                                                                   device=self.device,
                                                                                                   requires_grad=False),
                                             self.priors['beta'] * torch.ones_like(self.gamma_beta, device=self.device,
                                                                                   requires_grad=False))
        # ****need this term :cal prior and posterior over w
        w_x = self.encoder.reparameterize(w_post_mean, 2.0 * torch.log(w_post_sigma))
        w_prior_dist = Normal(torch.zeros_like(w_x).to(device=self.device), torch.ones_like(w_x).to(device=self.device))
        log_w_prior = log_normal_pdf(w_prior_dist.sample(), torch.zeros(w_x.size()).to(device=self.device),
                                     torch.eye(w_x.size()[0], w_x.size()[1], device=self.device))
        log_w_post = log_normal_pdf(w_x, w_post_mean, w_post_sigma)
        return ll + log_beta_prior + log_gauss_prior + log_w_prior + log_gamma_prior - log_kumar_post - log_gauss_post - log_w_post - log_gamma_post

    @torch.no_grad()
    def get_component_samples(self, batchSize, z_prior_mean, z_prior_logvar):
        # get the components of the latent space
        # a_inv = torch.pow(self.kumar_a,-1)
        # b_inv = torch.pow(self.kumar_b,-1)

        # ensure stick segments sum to 1
        # v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b)).to(device=self.device)
        # m = Beta(self.kumar_a, self.kumar_b)
        if torch.all(self.prior_nu.isfinite()):
            pass
        else:
            print(f" there is a NAN value in the prior over alpha variable")

        v_means = self.beta_sample(torch.ones((batchSize, self.K - 1), device=self.device, requires_grad=False),
                                   self.prior_nu)
        beta1m_cumprod = (1 - v_means).cumprod(-1)
        self.pi_samples = torch.mul(F.pad(v_means, (0, 1), value=1), F.pad(beta1m_cumprod, (1, 0), value=1))[:, :-1]

        # compose into stick segments using pi = v \prod (1-v)

        # p(z^{(i)}|c^{(i)}=j;theta)=N(z^{(i)}|mu_j,sigma_j)
        # p(c^{(i)}=j)=w_j
        # p(z^{(i)}|theta)=sum_{j=1}^{k} p(c^{(i)}=j) p(z^{(i)}|c^{(i)}=j;theta)
        # sample a component index
        component = torch.argmax(self.pi_samples, 1).to(device=self.device, dtype=torch.int64)
        # print(f'size of each latent component: {self.z_sample_list[0].size()}')

        component = torch.cat([torch.arange(0, batchSize, device=self.device).unsqueeze(1), component.unsqueeze(1)], 1)

        all_z = []
        for d in range(self.z_dim):
            z = self.encoder.reparameterize(z_prior_mean[:, :, d], z_prior_logvar[:, :, d])
            temp_z = torch.cat([z[k, :].unsqueeze(1) for k in range(self.K)], dim=1)
            all_z.append(gather_nd(temp_z, component).unsqueeze(1))

        self.pi_samples = self.pi_samples.unsqueeze(-1)
        self.pi_samples = F.pad(input=self.pi_samples, pad=(0, 0, 1, 0), mode='constant', value=0)

        self.pi_samples = self.pi_samples.expand(-1, self.K, 1)
        self.pi_samples = self.pi_samples.permute(0, 2, 1)
        out = torch.stack(all_z).to(device=self.device)
        out = out.permute(1, 0, 2)

        self.concatenated_latent_space = torch.bmm(out, self.pi_samples)
        return torch.squeeze(torch.mean(self.concatenated_latent_space, 2, True))
