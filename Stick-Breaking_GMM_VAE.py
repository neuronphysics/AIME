import logging
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Independent, Uniform
from copy import deepcopy
from torch.autograd import Variable
from types import SimpleNamespace as SN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#tf.reduce_mean -> tensor.mean
#tf.expand_dims -> tensor.expand
#tf.transpose -> tensor.permute
#author: Zahra Sheikhbahaee
'''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
'''
local_device = torch.device('cuda')

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    #source https://github.com/dd-iuonac/object-detector-in-carla/blob/fb900f7a1dcd366e326d044fcd8dc2c6ddc697fb/pointpillars/torchplus/ops/array_ops.py
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def init_mlp(layer_sizes, std=.01, bias_init=0., device=None):
    params = {'w': [], 'b': []}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(nn.Parameter(torch.tensor(Normal(torch.zeros(n_in, n_out), std * torch.ones(n_in, n_out)).rsample(), requires_grad=True, device=device)))
        params['b'].append(nn.Parameter(torch.tensor(torch.mul(bias_init, torch.ones([n_out,])), requires_grad=True, device=device)))
    return params

class MLP(nn.Module):

    def __init__(self, layer_sizes, std=.01, bias_init=0.):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            fc_layer = nn.Linear(n_in, n_out)
            fc_layer.weight.data.normal_(0., std)   # Initialize with values drawn from normal distribution
            fc_layer.bias.data.fill_(bias_init)     # Initialize with constant
            self.layers.append(fc_layer)

    def forward(self, X):
        h = X
        for fc_layer in self.layers:
            h = F.relu(fc_layer(h))
        return h

class GMMVAE(nn.Module):
    #Used this repository as a base https://github.com/psanch21/VAE-GMVAE/blob/e176d24d0e743f109ce37834f71f2f9067aae9bc/Alg_GMVAE/GMVAE_graph.py
    def __init__(self, K, nchannel, base_channels, z_dim, w_dim, hidden_dim,  device, img_width, batch_size):
        super(GMMVAE, self).__init__()

        self.K = K
        self.nchannel = nchannel
        self.base_channels = base_channels
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.img_width = img_width
        self.batch_size = batch_size
        self.enc_kernel = 4
        self.enc_stride = 2
        self._to_linear = None
        # ENCODER-DECODER LAYERS
        self.conv0   = nn.Conv2d(nchannel, base_channels, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_0  = nn.BatchNorm2d(self.base_channels)
        out_width = np.floor((self.img_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv1   = nn.Conv2d(base_channels, base_channels*2, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_1  = nn.BatchNorm2d(base_channels*2)
        out_width = np.floor((out_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv2   = nn.Conv2d(base_channels*2, base_channels*4, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_2  = nn.BatchNorm2d(base_channels*4)
        out_width = np.floor((out_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv3   = nn.Conv2d(base_channels*4, base_channels*8, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_3  = nn.BatchNorm2d(base_channels*8)
        out_width = int(np.floor((out_width - self.enc_kernel) / self.enc_stride + 1))
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear((out_width**2) * base_channels * 8, base_channels*8*4*4, bias=False)
        self.bn1d    = nn.BatchNorm1d(base_channels*8*4*4)
        self.fc1     = nn.Linear(base_channels*8*4*4, z_dim, bias=False)     # mean of z
        self.bn1d_1 = nn.BatchNorm1d(z_dim)
        self.fc2     = nn.Linear(base_channels*8*4*4, z_dim, bias=False)     # variance of z
        self.bn1d_2 = nn.BatchNorm1d(z_dim)
        #for encoding W
        self.flatten_raw_img = nn.Flatten()
        self.fc3     = nn.Linear(img_width * img_width * nchannel, hidden_dim)
        self.bn1d_3  = nn.BatchNorm1d(hidden_dim)
        self.fc4     = nn.Linear(hidden_dim, w_dim, bias=False)
        self.bn1d_4  = nn.BatchNorm1d(w_dim)
        self.fc5     = nn.Linear(hidden_dim, w_dim, bias=False)
        self.bn1d_5  = nn.BatchNorm1d(w_dim)
        self.fc6     = nn.Linear(z_dim+w_dim, hidden_dim)
        self.bn1d_6  = nn.BatchNorm1d(hidden_dim)
        self.fc7     = nn.Linear(hidden_dim, K)
        self.bn1d_7  = nn.BatchNorm1d(K)

        # Pz_given_wc LAYERS P(z|w,c)
        self.pz_wc_fc0 = nn.Linear(self.w_dim, self.hidden_dim, bias=False)
        self.pz_wc_bn1d_0 = nn.BatchNorm1d(hidden_dim)
        self.pz_wc_fc_mean = nn.ModuleList([nn.Linear(self.hidden_dim, self.z_dim, bias=False) for i in range(self.K)])
        self.pz_wc_bn_mean = nn.ModuleList([nn.BatchNorm1d(self.z_dim) for i in range(self.K)])
        self.pz_wc_fc_var = nn.ModuleList([nn.Linear(self.hidden_dim, self.z_dim, bias=False) for i in range(self.K)])
        self.pz_wc_bn_var = nn.ModuleList([nn.BatchNorm1d(self.z_dim) for i in range(self.K)])

        # Px_given_z LAYERS Decoder P(X|Z)
        conv2d_transpose_kernels, conv2d_transpose_input_width = self.determine_decoder_params(self.z_dim, self.img_width)
        self.px_z_fc = nn.Linear(self.z_dim, conv2d_transpose_input_width ** 2)
        #self.unflatten = nn.Unflatten(1, (1, conv2d_transpose_input_width, conv2d_transpose_input_width))
        self.conv2d_transpose_input_width = conv2d_transpose_input_width
        self.px_z_conv_transpose2d = nn.ModuleList()
        self.px_z_bn2d = nn.ModuleList()
        self.n_conv2d_transpose = len(conv2d_transpose_kernels)
        self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(1, self.base_channels * (self.n_conv2d_transpose - 1),
                                                             kernel_size=conv2d_transpose_kernels[0], stride=2))
        self.px_z_bn2d.append(nn.BatchNorm2d(self.base_channels * (self.n_conv2d_transpose - 1)))
        for i in range(1, self.n_conv2d_transpose - 1):
            self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(self.base_channels * (self.n_conv2d_transpose - i),
                                                                 self.base_channels*(self.n_conv2d_transpose - i - 1),
                                                                 kernel_size=conv2d_transpose_kernels[i], stride=2))
            self.px_z_bn2d.append(nn.BatchNorm2d(self.base_channels * (self.n_conv2d_transpose - i - 1)))
        self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(self.base_channels, self.nchannel,
                                                             kernel_size=conv2d_transpose_kernels[-1], stride=2))

    def Pz_given_wc(self, w_input, hidden_dim):
        #prior P(Z|w,c)
        h = self.pz_wc_fc0(w_input)
        h = F.relu(self.pz_wc_bn1d_0(h))
        z_wc_mean_list = list()
        for i in range(self.K):
            Pz_given_wc_mean = F.relu(self.pz_wc_bn_mean[i](self.pz_wc_fc_mean[i](h)))
            z_wc_mean_list.append(Pz_given_wc_mean)

        z_wc_var_list = list()
        for i in range(self.K):
            Pz_given_wc_var  = F.softplus(self.pz_wc_bn_var[i](self.pz_wc_fc_var[i](h)))
            z_wc_var_list.append(Pz_given_wc_var)

        z_wc_mean_stack = torch.stack(z_wc_mean_list) # [K, batch_size, z_dim]
        z_wc_var_stack  = torch.stack(z_wc_var_list) # [K, batch_size, z_dim]
        return z_wc_mean_stack, z_wc_var_stack

    def Px_given_z(self, z_input):
        #prior P(x|z)
        flattened_h = self.px_z_fc(z_input)
        h = flattened_h.view(flattened_h.size()[0], 1, self.conv2d_transpose_input_width, self.conv2d_transpose_input_width)
        for i in range(self.n_conv2d_transpose - 1):
            h = F.relu(self.px_z_bn2d[i](self.px_z_conv_transpose2d[i](h)))
        # h = F.relu(self.px_z_bn2d_0(self.px_z_conv_transpose2d_0(h)))
        # h = F.relu(self.px_z_bn2d_1(self.px_z_conv_transpose2d_1(h)))
        # h = F.relu(self.px_z_bn2d_2(self.px_z_conv_transpose2d_2(h)))
        # h = F.relu(self.px_z_bn2d_3(self.px_z_conv_transpose2d_3(h)))
        x_recons_mean_flat = torch.sigmoid(self.px_z_conv_transpose2d[self.n_conv2d_transpose - 1](h))
        return x_recons_mean_flat


    def encoder_decoder_fn(self, X):
        #posterior Q(z|X)
        h = X
        h = F.relu(self.bn2d_0(self.conv0(h)))
        h = F.relu(self.bn2d_1(self.conv1(h)))
        h = F.relu(self.bn2d_2(self.conv2(h)))
        h = F.relu(self.bn2d_3(self.conv3(h)))

        h = F.relu(self.bn1d(self.fc0(self.flatten(h))))
        self.sigma= 0.0001
        #create bottleneck
        # h = F.relu(self.bn2d_3(self.conv3(h)))
        self.z_x_mean    = F.relu(self.bn1d_1(self.fc1(h)))


        self.z_x_var     = F.softplus(self.bn1d_2(self.fc2(h))) + 1e-20
        self.z_x_logvar  = torch.log(self.z_x_var)

        eps1             = Normal(loc=torch.zeros(self.z_x_mean.shape,), scale=torch.ones(self.z_x_logvar.shape,)).rsample().to(self.device)
        self.z_x         = self.z_x_mean+torch.mul(torch.sqrt(self.z_x_var), eps1)


        #Build a two layers MLP to compute Q(w|x)
        hw = self.flatten_raw_img(X)
        hw = F.relu(self.bn1d_3(self.fc3(hw)))
        #mean of Q(w|x) distribution

        self.w_x_mean    = F.relu(self.bn1d_4(self.fc4(hw)))
        #variance of Q(w|x) distribution

        self.w_x_var     = F.softplus(self.bn1d_5(self.fc5(hw)))
        self.w_x_logvar  = torch.log(self.w_x_var)

        eps2             = Normal(loc=torch.zeros(self.w_x_mean.shape,), scale=torch.ones(self.w_x_logvar.shape,)).rsample().to(self.device)
        self.w_x         = torch.add(self.w_x_mean,torch.mul(torch.sqrt(self.w_x_var), eps2))

        #P(c|w,z)=Q(c|x)
        zw               = torch.cat([self.w_x,self.z_x],1)
        hc               = zw
        hc               = F.relu(self.bn1d_6(self.fc6(hc)))
        Pc_wz            = F.relu(self.bn1d_7(self.fc7(hc)))
        self.pc_wz       = F.softmax(Pc_wz, dim=-1)
        self.log_pc_wz   = torch.log(1e-20+self.pc_wz)

        #Build P(x|z)
        self.x_recons_mean_flat = self.Px_given_z(self.z_x)
        eps = Normal(loc=torch.zeros(self.x_recons_mean_flat.shape,), scale=torch.ones(self.x_recons_mean_flat.shape,)).rsample().to(self.device)

        self.x_recons_flat = torch.min(torch.max(torch.add(self.x_recons_mean_flat, np.sqrt(self.sigma) * eps), torch.tensor(1e-20, dtype=torch.float).to(self.device)), torch.tensor(1-1e-20, dtype=torch.float).to(self.device))
        self.x_recons = torch.reshape(self.x_recons_flat , [-1,self.img_width, self.img_width, self.nchannel])


        #P(w)
        self.w_sample = Normal(loc=torch.zeros((self.batch_size, self.w_dim)), scale=torch.ones((self.batch_size, self.w_dim))).rsample().to(self.device)

        #P(z_i|w,c_i)
        self.z_wc_mean_list_sample, self.z_wc_var_list_sample = self.Pz_given_wc(self.w_sample,self.hidden_dim)
        #self.z_wc_mean_list_sample = self.z_wc_mean_list_sample.reshape(self.z_wc_mean_list_sample.shape[1], self.z_wc_mean_list_sample.shape[0], self.z_wc_mean_list_sample.shape[2])
        #self.z_wc_var_list_sample = self.z_wc_var_list_sample.reshape(self.z_wc_var_list_sample.shape[1], self.z_wc_var_list_sample.shape[0], self.z_wc_var_list_sample.shape[2])
        self.z_sample_list = list()
        for i in range(self.K):
            eps = Normal(loc=torch.zeros_like(self.z_wc_mean_list_sample[i]), scale=torch.ones_like(self.z_wc_mean_list_sample[i])).rsample().to(self.device)
            z_sample = torch.add(self.z_wc_mean_list_sample[i], torch.mul(torch.sqrt(self.z_wc_var_list_sample[i]), eps))
            self.z_sample_list.append(z_sample)
        #P(x|z)
        self.x_sample_mean_flat_list = list()
        self.x_sample_flat_list = list()
        self.x_sample_list = list()
        for i in range(self.K):
            x_sample_mean_flat = self.Px_given_z(self.z_sample_list[i])
            self.x_sample_mean_flat_list.append(x_sample_mean_flat)

            eps = Normal(loc=torch.zeros(x_sample_mean_flat.shape), scale=torch.ones(x_sample_mean_flat.shape)).rsample().to(self.device)

            x_sample_flat = torch.add(x_sample_mean_flat, np.sqrt(self.sigma) * eps)
            x_sample = torch.reshape(x_sample_flat , [-1, self.img_width, self.img_width, self.nchannel])

            self.x_sample_flat_list.append(x_sample_flat)
            self.x_sample_list.append(x_sample)

    def determine_decoder_params(self, z_dim, img_width):
        kernels = []
        input_img_dim = downsampled_img_width = img_width
        while downsampled_img_width ** 2 > z_dim:
            kernel = 6 if (downsampled_img_width - 4) % 2 == 0 else 5
            kernels.append(kernel)
            input_img_dim = downsampled_img_width
            downsampled_img_width = (downsampled_img_width - kernel) // 2 + 1
        kernels = kernels[:-1][::-1]
        return kernels, input_img_dim

    # def forward(self, X):
    #     return self.encoder_decoder_fn(X)

def mlp(X, params):
    h = [X]
    for w, b in zip(params['w'][:-1], params['b'][:-1]):
        h.append(F.relu(torch.matmul(h[-1], w) + b))
    return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]


def compute_nll(x, x_recon_linear):
    #return torch.sum(func.binary_cross_entropy_with_logits(x_recon_linear, x), dim=1, keepdim=True)
    return F.binary_cross_entropy_with_logits(x_recon_linear, x)

def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    d = torch.sub(mu_post , mu_prior)
    d = torch.mul(d,d)
    return torch.sum(-torch.div(d + torch.mul(sigma_post,sigma_post),(2.*sigma_prior*sigma_prior)) - torch.log(sigma_prior*2.506628 + 1e-20), dim=1, keepdim=True)


def beta_fn(a,b):
    global local_device
    return torch.exp(torch.lgamma(torch.tensor(a, dtype=torch.float, requires_grad=True).to(device=local_device)) + torch.lgamma(torch.tensor(b, dtype=torch.float, requires_grad=True).to(device=local_device)) - torch.lgamma(torch.tensor(a+b, dtype=torch.float, requires_grad=True).to(device=local_device)))


def compute_kumar2beta_kld(a, b, alpha, beta):
    SMALL = 1e-16
    EULER_GAMMA = 0.5772156649015329

    ab    = torch.mul(a,b)+ SMALL
    a_inv = torch.pow(a + SMALL, -1)
    b_inv = torch.pow(b + SMALL, -1)
    # compute taylor expansion for E[log (1-v)] term
    kl = torch.mul(torch.pow(1+ab,-1), beta_fn(a_inv, b))
    for idx in range(10):
        kl += torch.mul(torch.pow(idx+2+ab,-1), beta_fn(torch.mul(idx+2., a_inv), b))
    kl = torch.mul(torch.mul(beta-1,b), kl)
    #
    #psi_b = torch.log(b + SMALL) - 1. / (2 * b + SMALL) -\
    #    1. / (12 * b**2 + SMALL)
    psi_b = torch.digamma(b+SMALL)
    kl += torch.mul(torch.div(a-alpha,a+SMALL), -EULER_GAMMA - psi_b - b_inv)
    # add normalization constants
    kl += torch.log(ab) + torch.log(beta_fn(alpha, beta) + SMALL)
    #  final term
    kl += torch.div(-(b-1),b +SMALL)
    return kl


def log_normal_pdf(x, mu, sigma):
    d = torch.sub(mu , x)
    d2 = torch.mul(-1., torch.mul(d, d))
    s2 = torch.mul(2., torch.mul(sigma,sigma))
    return torch.sum(torch.div(d2, s2 + 1e-20) - torch.log(torch.mul(sigma, 2.506628) + 1e-20), dim=1, keepdim=True)


def log_beta_pdf(v, alpha, beta):
    return torch.sum((alpha - 1) * torch.log(v + 1e-20) + (beta-1) * torch.log(1 - v + 1e-20) - torch.log(beta_fn(alpha, beta) + 1e-20), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    return torch.sum(torch.mul(a - 1, torch.log(v + 1e-20)) + torch.mul(b - 1, torch.log(1 - torch.pow(v,a) + 1e-20)) + torch.log(a + 1e-20) + torch.log(b + 1e-20), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in range(K-1):
        s += torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s + 1e-20)




def gumbel_softmax_sample(log_pi, temperature, eps=1e-20):
    global local_device
    # Sample from Gumbel
    U = torch.rand(log_pi.shape).to(device=local_device)
    g = -Variable(torch.log(-torch.log(U + eps) + eps))
    # Gumbel-Softmax sample
    y = log_pi + g
    return F.softmax(y / temperature, dim=-1)

### Gaussian Mixture Model VAE Class
class InfGaussMMVAE(GMMVAE):
    # based on this implementation https://github.com/enalisnick/mixture_density_VAEs/blob/ee4e3b766523017a7bfd1c408d682ffd94fd0829/models/gaussMMVAE_collapsed.py
    def __init__(self, hyperParams, K, nchannel, base_channels, z_dim, w_dim, hidden_dim, device, img_width, batch_size, include_elbo2):
        global local_device
        local_device = device
        super(InfGaussMMVAE, self).__init__(K, nchannel, base_channels, z_dim, w_dim, hidden_dim,  device, img_width, batch_size)

        #self.X = Variable(torch.FloatTensor(hyperParams['batch_size'], hyperParams['input_d']))
        #self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior      = hyperParams['prior']
        self.K          = hyperParams['K']
        self.z_dim      = hyperParams['latent_d']
        self.hidden_dim = hyperParams['hidden_d']
        #self.Initialize = self.encoder_decoder_fn(X)
        # self.init_encoder(hyperParams)
        # self.init_decoder(hyperParams)
        self.flatten_x     = nn.Flatten()   # Added by Blake
        self.fc_inf0       = nn.Linear(img_width * img_width * nchannel, hidden_dim)
        self.bn1d_inf      = nn.BatchNorm1d(hidden_dim)
        #self.x_recons_linear = self.f_prop(X)

        self.encoder_kumar_a = init_mlp([self.hidden_dim, self.K-1], 1e-8, device=device) #Q(pi|x)
        self.encoder_kumar_b = init_mlp([self.hidden_dim, self.K-1], 1e-8, device=device) #Q(pi|x)

        #self.elbo_obj = self.get_ELBO()
        self.img_size = img_width
        self.include_elbo2 = include_elbo2

    # def __len__(self):
    #     return len(self.X)

    def init_encoder(self):
        self.encoder_z_mu    = self.z_x_mean #Q(z|x)
        self.encoder_z_sigma = self.z_x_var  #Q(z|x)
        self.encoder_w_mu    = self.w_x_mean #Q(w|x)
        self.encoder_w_sigma = self.w_x_var  #Q(w|x)
        self.encoder_c       = self.log_pc_wz #log P(c|w,z)=Q(c|x)

    def init_decoder(self):
        self.decoder_x = self.x_recons_flat
        self.decoder_z = self.z_sample_list
        self.decoder_w = self.w_sample

    def mlp(self, X, params):
        h = [X]
        for w, b in zip(params['w'][:-1], params['b'][:-1]):
            h.append(F.relu(torch.matmul(h[-1], w) + b))
        return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]


    def forward(self, X):
        self.encoder_decoder_fn(X)
        self.init_encoder()
        self.init_decoder()

        # init variational params
        x_temp         = list()
        z_temp         = list()
        #x              = x_temp.extend(torch.transpose(self.x_sample_list, [1, 0, 2, 3, 4]))
        x_recon_linear = torch.stack(self.x_sample_list, dim=1)

        self.z_mu      = torch.transpose(self.z_wc_mean_list_sample, 0, 1)
        self.z_sigma   = torch.transpose(self.z_wc_var_list_sample, 0, 1)
        # self.z_mu    = z_temp.extend(np.transpose(self.z_wc_mean_list_sample, [1, 0, 2]))
        # self.z_mu    = torch.tensor(self.z_mu)
        # z_temp       = list()
        # self.z_sigma = z_temp.extend(np.transpose(self.z_wc_var_list_sample, [1, 0, 2]))
        # self.z_sigma = torch.tensor(self.z_sigma)
        self.z       = self.z_mu + torch.mul(self.z_sigma, Normal(0., 1.).rsample(self.z_sigma.shape).to(self.device))
        self.w_mu    = self.w_x_mean # TODO Fix this
        self.w_sigma = self.w_x_var # TODO: Fix this
        self.w       = self.w_mu + torch.mul(self.w_sigma, Normal(0., 1.).rsample(self.w_sigma.shape).to(self.device))

        h = F.relu(self.bn1d_inf(self.fc_inf0(self.flatten_x(X))))  # TODO: What goes here???

        self.kumar_a = torch.exp(mlp(h,self.encoder_kumar_a))
        self.kumar_b = torch.exp(mlp(h,self.encoder_kumar_b))
        self.c       = self.pc_wz
        return x_recon_linear.mean(dim=1), self.z_x


    def compose_stick_segments(self, v):

        segments = []
        self.remaining_stick = [torch.ones((v.shape[0],1)).to(self.device)]
        for i in range(self.K-1):
            curr_v = v[:, i]
            segments.append(torch.mul(curr_v, self.remaining_stick[-1]))
            self.remaining_stick.append(torch.mul(1-curr_v, self.remaining_stick[-1]))
        segments.append(self.remaining_stick[-1])

        return segments

    def _encoding(self, X):
      return self.z_x_mean, self.z_x_logvar, self.kumar_a, self.kumar_b, self.c, self.w_x_mean, self.w_x_logvar

    def _decoding(self, z):
        a_inv = torch.pow(self.kumar_a, -1)
        b_inv = torch.pow(self.kumar_b, -1)

        # compute Kumaraswamy means
        v_means = torch.mul(self.kumar_b, beta_fn(1. + a_inv, self.kumar_b))

        # compute Kumaraswamy samples
        uni_samples = torch.FloatTensor(v_means.shape).uniform_(1e-8, 1-1e-8).to(self.device)       
        v_samples  = torch.pow(1 - torch.pow(uni_samples, b_inv), a_inv)
        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means = torch.stack(self.compose_stick_segments(v_means), axis=2)
        self.pi_samples = torch.max(torch.stack(self.compose_stick_segments(v_samples), axis=2), torch.tensor(1e-6, dtype=torch.float).to(self.device))
        #KL divergence P(c|z,w)=Q(c|x) while P(c|pi) is the prior

        prior_c        = gumbel_softmax_sample(torch.log(self.pi_samples), 1, eps=1e-20)
        return self.x_recons_flat, self.z_mu, torch.log(self.z_sigma + 1e-20), prior_c

    def decode(self, z):
        return self._decoding(z)[0]

    def get_ELBO(self, X):
        _, _, _, pc  = self._decoding(self.z)
        log_ratio = torch.sub(torch.log(self.pc_wz+ 1e-20),torch.log(pc+ 1e-20))
        #1)need this term
        elbo1      = torch.sum(self.pc_wz * log_ratio, dim=-1).mean()

        # compose elbo
        elbo2 = torch.tensor(0, dtype=torch.float).to(self.device)
        if self.include_elbo2:
            for k in range(self.K-1):
                #2)need this term
                #elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), \
                #                                   self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])
                #print(self.kumar_b[:, k].unsqueeze(1))
                #print(self.kumar_b[:, k].expand(self.batch_size))
                #elbo2 -= compute_kumar2beta_kld(self.kumar_a[:, k].expand(self.batch_size), self.kumar_b[:, k].expand(self.batch_size), self.prior, (self.K-1-k)* self.prior).mean()
                elbo2 -= compute_kumar2beta_kld(self.kumar_a[:, k], self.kumar_b[:, k], self.prior, (self.K-1-k)* self.prior).mean()
        #elbo += mcMixtureEntropy(self.pi_samples, self.z, self.z_mu, self.z_sigma, self.K)
        #3)need this term
        elbo3 = -0.5 * torch.sum(1 + torch.log(self.w_sigma) - torch.pow(self.w_mu, 2) - self.w_sigma) #KLD_W
        #compute D_KL(Q(z|x)||p(z|c,w))
        #use this term https://github.com/psanch21/VAE-GMVAE/blob/e176d24d0e743f109ce37834f71f2f9067aae9bc/Alg_GMVAE/GMVAE_graph.py#L278
        # KL loss
        #kl_loss = 0.5*torch.sum(1 + z_logstd - z_mean**2 - torch.exp(z_logstd), dim=1)
        # likelihood loss

        z_wc = torch.unsqueeze(self.z_x_mean, -1)
        z_wc = z_wc.expand(-1, self.z_dim, self.K)
        logvar_z = self.z_x_logvar.unsqueeze(-1)
        logvar_z = logvar_z.expand(-1, self.z_dim, self.K)
        
        logvar_pz = torch.log(self.z_wc_var_list_sample.permute(1, 2, 0))
        elbo4 = 0.5 * (((logvar_pz - logvar_z) + ((logvar_z.exp() + (z_wc - self.z_wc_mean_list_sample.permute(1, 2, 0)).pow(2))/logvar_pz.exp())) - 1).sum(dim=(1,2)).mean(dim=0)


        #compute E_{q(z|x)}[P(x|x)] reconstruction loss
        #use this term https://github.com/psanch21/VAE-GMVAE/blob/e176d24d0e743f109ce37834f71f2f9067aae9bc/Alg_GMVAE/GMVAE_graph.py#L256
        criterion = nn.BCELoss(reduction='sum')
        elbo5 = criterion(self.x_recons_flat.view(-1, self.nchannel*self.img_size*self.img_size), X.view(-1, self.nchannel*self.img_size*self.img_size))

        #elbo = F.binary_cross_entropy(input=self.x_recons_flat, target=X, reduction='sum')

        return (elbo1, elbo2, elbo3, elbo4, elbo5)


    def get_log_margLL(self, batchSize, X_recons_linear):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples

        uni_samples =torch.FloatTensor(a_inv.shape[0], self.K-1).uniform_(1e-8, 1-1e-8).to(self.device)
        v_samples   = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index

        uni_samples = torch.FloatTensor(a_inv.shape[0], self.K).uniform_(1e-8, 1-1e-8).to(self.device)
        gumbel_samples = -torch.log(-torch.log(uni_samples + 1e-20) + 1e-20)
        component_samples = torch.IntTensor(torch.argmax(torch.log(torch.cat( self.pi_samples,1) + 1e-20) + gumbel_samples, 1))

        # calc likelihood term for chosen components
        all_ll = []
        for k in range(self.K): all_ll.append(-compute_nll(self.X, X_recons_linear[k]))
        all_ll = torch.cat( all_ll,1)
        #component_samples = tf.concat(1, [tf.expand_dims(tf.range(0,batchSize),1), tf.expand_dims(component_samples,1)])
        component_samples = torch.cat( [torch.range(0, batchSize).expand(1), component_samples.expand(1)], 1)
        ll = gather_nd(all_ll, component_samples)
        #ll = tf.expand_dims(ll,1)
        ll = ll.expand(1)

        # calc prior terms
        all_log_gauss_priors = []

        for k in range(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z_sample_list[k], self.z_wc_mean_list_sample[k], self.z_wc_var_list_sample[k]))
        all_log_gauss_priors = torch.cat(all_log_gauss_priors, dim=1)
        log_gauss_prior = gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = log_gauss_prior.expand(1)

        #****need this term :
        #log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        log_beta_prior = log_beta_pdf(v_samples[:,0].expand(1), self.prior, (self.K - 1) * self.prior)
        for k in range(self.K-2):
            #log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])
            log_beta_prior += log_beta_pdf(v_samples[:, k + 1].expand(1), self.prior, (self.K - 2 - k) * self.prior)

        # ****need this term :calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)


        log_gauss_post = log_normal_pdf(self.z_x, self.z_x_mean, self.z_x_var)

        #****need this term :cal prior and posterior over w
        log_w_prior = log_normal_pdf(self.w, torch.zeros(self.w.size()),torch.eye(self.w.size()))
        log_w_post  = log_normal_pdf(self.w, self.w_mu,self.w_sigma)
        return ll + log_beta_prior + log_gauss_prior +log_w_prior - log_kumar_post - log_gauss_post - log_w_post


    def generate_samples(self):

        w = list()
        z = list()
        x = list()

        w.extend(self.w_sample)
        z.extend(self.z_sample_list)
        x_tmp = self.x_sample_list
        x_tmp.permute(1, 0, 2, 3, 4)
        x.extend(x_tmp)

        x = np.array(x)
        z = np.array(z)
        w = np.array(w)
        return x, z, w

#class of optimizer
#https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py
from torch.optim import Optimizer
class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss



#data analysis
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.utils
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
device = torch.device('cuda')
hyperParams = {"batch_size": 50,
               "input_d": 1,
               "prior": 1, #dirichlet_alpha'
               "K": 25,
               "hidden_d": 80,
               "latent_d": 20,
               "latent_w": 10}


# # Preparing Data : STL10
print("Loading trainset...")
train_dataset = datasets.STL10('../data', split='train', transform=transform,
                              target_transform=None, download=True)
train_loader = DataLoader.DataLoader(train_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=4)
print("Loading testset...")
test_dataset = datasets.STL10('../data', split='test', transform=transform,
                             target_transform=None, download=True)
test_loader  = DataLoader.DataLoader(test_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=4)
print("Done!")
##########

train_dataset_data = train_dataset.data
train_dataset_label = train_dataset.labels

print(train_dataset_label.shape)
label_class = np.array(list(set(train_dataset.labels)))
print(label_class.shape)



data, label  = iter(train_loader).next()


print('data shape', data.shape)
print('label shape', label.shape)
raw_img_width = data.shape[3]
print('image size', raw_img_width)


net = InfGaussMMVAE(hyperParams, 25, 3, 4, 20, 10, 80, device, raw_img_width, hyperParams["batch_size"],include_elbo2=True)
params = list(net.parameters())
#if torch.cuda.device_count() > 1:
#    net = nn.DataParallel(net)
net = net.to(device)

#net.cuda()
#optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8)

optimizer = AdaBound(net.parameters(), lr=0.0001)


def train(epoch):
    train_loss_avg = []
    net.train()
    train_loss = 0
    train_loss_avg.append(0)
    num_batches = 0
    for batch_idx, (X, _) in enumerate(train_loader):
        X = X.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        X_recons_linear, vector_z = net(X)
        loss = sum(net.get_ELBO(X))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1
        if batch_idx % 5 == 0:
            '''
            if epoch % 2 == 0:
                plt.figure(0)
                plt.imshow(X[0].permute(1,2,0).cpu())
                plt.show()
                plt.figure(1)
                plt.imshow(X_recons_linear[0].detach().cpu().numpy())
                plt.show()
            '''

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(X)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_loss_avg[-1] /= num_batches
    print(train_loss_avg)
    return train_loss_avg

avg_train_loss=[]
best_loss = 10**15  # Random big number (bigger than the initial loss)
best_epoch = 0
for epoch in range(1, 200):
        
       average_epoch_loss =train(epoch)
       avg_train_loss.extend(average_epoch_loss)
       """
       if epoch % 50 == 0:
          torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
          }, "results/model_InfGMM_VAE.pt")
       
       if average_epoch_loss[-1] < best_loss:
          best_loss = average_epoch_loss[-1]
          best_epoch = epoch

       if os.path.isfile("results/model_InfGMM_VAE.pt"):
          net.load_state_dict(torch.load('results/model_InfGMM_VAE.pt'))
       """
#plot the loss vvalues
fig = plt.figure()
plt.plot(avg_train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/Loss_InfGMM_VAE.png')
net.eval()

test_loss_avg, num_batches = 0, 0
for image_batch, _ in test_loader:

    with torch.no_grad():

        image_batch = image_batch.to(device)

        # vae reconstruction
        X_recons_linear, vector_z = net(image_batch)
        

        # reconstruction error
        loss = sum(net.get_ELBO(image_batch))

        test_loss_avg += loss.item()
        num_batches += 1

test_loss_avg /= num_batches
print('average reconstruction error: %f' % (test_loss_avg))



def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):

    with torch.no_grad():

        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

images, labels = iter(test_loader).next()
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.savefig('results/Original_Images_InfGMM_VAE.png')
visualise_output(images, net)
plt.savefig('results/Reconstructed_Images_InfGMM_VAE.png')

