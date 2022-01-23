#Stick breaking Gaussian Mixture Prior for variational autoencoders
import logging
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Independent, Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from copy import deepcopy
from torch.autograd import Variable
from types import SimpleNamespace as SN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import OrderedDict
#import pyro
#import pyro.distributions as dist
from numpy.testing import assert_almost_equal
import os
try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#tf.reduce_mean -> tensor.mean
#tf.expand_dims -> tensor.expand
#tf.transpose -> tensor.permute
#author: Zahra Sheikhbahaee
#set randoom seed
torch.random.seed()
np.random.seed()
if torch.cuda.is_available():
   torch.cuda.seed()

local_device = torch.device('cuda')

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
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

def mlp(X, params):
    h = [X]
    for w, b in zip(params['w'][:-1], params['b'][:-1]):
        h.append(F.relu(torch.matmul(h[-1], w) + b))
    return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]


class GMMVAE(nn.Module):
    #Used this repository as a base 
    # https://github.com/bhavikngala/gaussian_mixture_vae 
    # https://github.com/Nat-D/GMVAE
    # https://github.com/psanch21/VAE-GMVAE
    def __init__(self, number_of_mixtures, nchannel, base_channels, z_dim, w_dim, hidden_dim,  device, img_width, batch_size):
        super(GMMVAE, self).__init__()

        self.K          = number_of_mixtures
        self.nchannel   = nchannel
        self.base_channels = base_channels
        self.z_dim      = z_dim
        self.w_dim      = w_dim
        self.hidden_dim = hidden_dim
        self.device     = device
        self.img_width  = img_width
        self.batch_size = batch_size
        self.enc_kernel = 4
        self.enc_stride = 2
        self._to_linear = None
        ########################
        # ENCODER-CONVOLUTION LAYERS
        self.conv0       = nn.Conv2d(nchannel, base_channels, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_0      = nn.BatchNorm2d(self.base_channels)
        self.LeakyReLU_0 = nn.LeakyReLU(0.2)
        out_width        = np.floor((self.img_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv1       = nn.Conv2d(base_channels, base_channels*2, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_1      = nn.BatchNorm2d(base_channels*2)
        self.LeakyReLU_1 = nn.LeakyReLU(0.2)
        out_width        = np.floor((out_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv2       = nn.Conv2d(base_channels*2, base_channels*4, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_2      = nn.BatchNorm2d(base_channels*4)
        self.LeakyReLU_2 = nn.LeakyReLU(0.2)
        out_width        = np.floor((out_width - self.enc_kernel) / self.enc_stride + 1)
        self.conv3       = nn.Conv2d(base_channels*4, base_channels*8, self.enc_kernel, stride=self.enc_stride)
        self.bn2d_3      = nn.BatchNorm2d(base_channels*8)
        self.LeakyReLU_3 = nn.LeakyReLU(0.2)
        out_width        = int(np.floor((out_width - self.enc_kernel) / self.enc_stride + 1))
        ########################
        #ENCODER-USING FULLY CONNECTED LAYERS
        #THE LATENT SPACE (Z) 
        self.flatten     = nn.Flatten()
        self.fc0         = nn.Linear((out_width**2) * base_channels * 8, base_channels*8*4*4, bias=False)
        self.bn1d        = nn.BatchNorm1d(base_channels*8*4*4)
        self.fc1         = nn.Linear(base_channels*8*4*4, hidden_dim, bias=False)     
        self.bn1d_1      = nn.BatchNorm1d(hidden_dim)
        # mean of z
        
        self.fc2         = nn.Linear(hidden_dim, z_dim, bias=False)     
        self.bn1d_2      = nn.BatchNorm1d(z_dim)
        # variance of z

        self.fc3         = nn.Linear(hidden_dim, z_dim, bias=False)     
        self.bn1d_3      = nn.BatchNorm1d(z_dim)
        ########################
        #ENCODER-JUST USING FULLY CONNECTED LAYERS 
        #THE LATENT SPACE (W) 
        self.flatten_raw_img = nn.Flatten()
        self.fc4         = nn.Linear(img_width * img_width * nchannel, hidden_dim)
        self.bn1d_4      = nn.BatchNorm1d(hidden_dim)
        self.fc5         = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn1d_5      = nn.BatchNorm1d(hidden_dim)
        # mean of w
        self.fc6         = nn.Linear(hidden_dim, w_dim, bias=False)
        self.bn1d_6      = nn.BatchNorm1d(w_dim)
        # logvar_w
        self.fc7         = nn.Linear(hidden_dim, w_dim, bias=False)
        self.bn1d_7      = nn.BatchNorm1d(w_dim)
        ########################
        #number of mixtures (c parameter) gets the input from feedforward layers of w and convolutional layers of z
        self.fc8         = nn.Linear(2*hidden_dim, hidden_dim)
        self.bn1d_8      = nn.BatchNorm1d(hidden_dim)
        self.fc9         = nn.Linear(hidden_dim, number_of_mixtures)
        self.bn1d_9      = nn.BatchNorm1d(number_of_mixtures)
        ########################
        #(GENERATOR)
        # CreatePriorGenerator_Given Z LAYERS P(z|w,c)
        self.pz_wc_fc0     = nn.Linear(self.w_dim, self.hidden_dim, bias=False)
        self.pz_wc_bn1d_0  = nn.BatchNorm1d(hidden_dim)
        self.pz_wc_fc_mean = nn.ModuleList([nn.Linear(self.hidden_dim, self.z_dim, bias=False) for i in range(self.K)])
        self.pz_wc_bn_mean = nn.ModuleList([nn.BatchNorm1d(self.z_dim) for i in range(self.K)])
        self.pz_wc_fc_var  = nn.ModuleList([nn.Linear(self.hidden_dim, self.z_dim, bias=False) for i in range(self.K)])
        self.pz_wc_bn_var  = nn.ModuleList([nn.BatchNorm1d(self.z_dim) for i in range(self.K)])
        ########################
        # DECODER: CreateXGenerator
        # Px_given_z LAYERS Decoder P(X|Z)
        conv2d_transpose_kernels, conv2d_transpose_input_width = self.determine_decoder_params(self.z_dim, self.img_width)
        self.px_z_fc_0   = nn.Linear(self.z_dim, conv2d_transpose_input_width ** 2)
        self.px_z_bn1d_0 = nn.BatchNorm1d(conv2d_transpose_input_width ** 2)
        self.px_z_fc_1   = nn.Linear(conv2d_transpose_input_width ** 2, conv2d_transpose_input_width ** 2)
        #self.unflatten = nn.Unflatten(1, (1, conv2d_transpose_input_width, conv2d_transpose_input_width))
        self.conv2d_transpose_input_width = conv2d_transpose_input_width
        self.px_z_conv_transpose2d = nn.ModuleList()
        self.px_z_bn2d   = nn.ModuleList()
        self.n_conv2d_transpose = len(conv2d_transpose_kernels)
        self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(1, self.base_channels * (self.n_conv2d_transpose - 1),
                                                             kernel_size=conv2d_transpose_kernels[0], stride=2))
        self.px_z_bn2d.append(nn.BatchNorm2d(self.base_channels * (self.n_conv2d_transpose - 1)))
        self.px_z_LeakyReLU = nn.ModuleList()
        self.px_z_LeakyReLU.append(nn.LeakyReLU(0.2))
        #print('Number of convoltional layers in decooder: ', self.n_conv2d_transpose)
        for i in range(1, self.n_conv2d_transpose - 1):
            self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(self.base_channels * (self.n_conv2d_transpose - i),
                                                                 self.base_channels*(self.n_conv2d_transpose - i - 1),
                                                                 kernel_size=conv2d_transpose_kernels[i], stride=2))
            self.px_z_bn2d.append(nn.BatchNorm2d(self.base_channels * (self.n_conv2d_transpose - i - 1)))
            self.px_z_LeakyReLU.append(nn.LeakyReLU(0.2))
        self.px_z_conv_transpose2d.append(nn.ConvTranspose2d(self.base_channels, self.nchannel,
                                                             kernel_size=conv2d_transpose_kernels[-1], stride=2))
        #print('Final number of convoltional layers in decooder: ', len(self.px_z_conv_transpose2d))
        self.encoder_kumar_a = init_mlp([hidden_dim,self.K-1], 1e-5, device=device) #Q(pi|x)
        self.encoder_kumar_b = init_mlp([hidden_dim,self.K-1], 1e-5, device=device) #Q(pi|x)
        ##########Device#########
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def Pz_given_wc(self, w_input):
        # Prior
        #prior generator P(Z|w,c)
        h = self.pz_wc_fc0(w_input)
        h = torch.tanh(self.pz_wc_bn1d_0(h))
        z_wc_mean_list = list()
        for i in range(self.K):
            Pz_given_wc_mean = F.relu(self.pz_wc_bn_mean[i](self.pz_wc_fc_mean[i](h)))
            z_wc_mean_list.append(Pz_given_wc_mean)

        z_wc_var_list = list()
        for i in range(self.K):
            Pz_given_wc_var  = F.softplus(self.pz_wc_bn_var[i](self.pz_wc_fc_var[i](h)))
            z_wc_var_list.append(Pz_given_wc_var)

        z_wc_mean_stack    = torch.stack(z_wc_mean_list, dim=0) # [K, batch_size, z_dim]
        z_wc_logvar_stack  = torch.stack(z_wc_var_list, dim=0) # [K, batch_size, z_dim]
        return z_wc_mean_stack, z_wc_logvar_stack

    def Px_given_z(self, z_input):
        #Decoder: 
        #Generate X: P(X|Z)
        h  = F.relu(self.px_z_bn1d_0(self.px_z_fc_0(z_input)))
        flattened_h = self.px_z_fc_1(h)
        h = flattened_h.view(flattened_h.size()[0], 1, self.conv2d_transpose_input_width, self.conv2d_transpose_input_width)
        for i in range(self.n_conv2d_transpose - 1):
            h = self.px_z_LeakyReLU[i](self.px_z_bn2d[i](self.px_z_conv_transpose2d[i](h)))
        # h = F.relu(self.px_z_bn2d_0(self.px_z_conv_transpose2d_0(h)))
        # h = F.relu(self.px_z_bn2d_1(self.px_z_conv_transpose2d_1(h)))
        # h = F.relu(self.px_z_bn2d_2(self.px_z_conv_transpose2d_2(h)))
        # h = F.relu(self.px_z_bn2d_3(self.px_z_conv_transpose2d_3(h)))
        x_recons_mean_flat = torch.sigmoid(self.px_z_conv_transpose2d[self.n_conv2d_transpose - 1](h))
        return x_recons_mean_flat

    def GMM_encoder(self, data):
        #1) posterior Q(z|X)
        """
        compute z = z_mean + z_var * eps1
        """
        h = data
        h = self.LeakyReLU_0(self.bn2d_0(self.conv0(h)))
        h = self.LeakyReLU_1(self.bn2d_1(self.conv1(h)))
        h = self.LeakyReLU_2(self.bn2d_2(self.conv2(h)))
        h = self.LeakyReLU_3(self.bn2d_3(self.conv3(h)))

        h = F.relu(self.bn1d(self.fc0(self.flatten(h))))
        hlayer = F.relu(self.bn1d_1(self.fc1(h)))
        #mean_z
        mu_z        = F.relu(self.bn1d_2(self.fc2(hlayer)))

        #logvar_z
        logvar_z    = F.softplus(self.bn1d_3(self.fc3(hlayer))) 

        #2) posterior Q(w|X)
        #Create Recogniser for W
        hw = self.flatten_raw_img(data)
        hw = F.relu(self.bn1d_4(self.fc4(hw)))
        hlayer_w = F.relu(self.bn1d_5(self.fc5(hw)))
        #mean of W
        mu_w          = F.relu(self.bn1d_6(self.fc6(hlayer_w)))
        #variance of Q(w|x) distribution
        
        #log(variance of W)
        logvar_w      = F.softplus(self.bn1d_7(self.fc7(hlayer_w)))
        
        #3) posterior P(c|w,z)=Q(c|X)
        #posterior distribution of P(c|w,z^{i})=Q(c|x) where z^{i}~q(z|x)
        #combine hidden layers after convolutional layers for Z latent space and feed forward layers of W latent space
        zw               = torch.cat([hlayer,hlayer_w],1)
        hc               = zw
        hc               = F.relu(self.bn1d_8(self.fc8(hc)))
        Pc_wz            = F.relu(self.bn1d_9(self.fc9(hc)))
        c_posterior      = F.softmax(Pc_wz, dim=-1)
        #4) posterior of Kumaraswamy 
        self.kumar_a = torch.exp(mlp(hlayer,self.encoder_kumar_a))
        self.kumar_b = torch.exp(mlp(hlayer,self.encoder_kumar_b))
        return mu_w, logvar_w, mu_z, logvar_z, c_posterior

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space.
        In order for the back-propagation to work, we need to be able to calculate the gradient. 
        This reparameterization trick first generates a normal distribution, then shapes the distribution
        with the mu and variance from the encoder.
        
        This way, we can can calculate the gradient parameterized by this particular random instance.
        """
        eps = torch.randn_like(log_var)
        return torch.add(mu , torch.mul( log_var.mul(0.5).exp_() , eps))

    

    def encoder_decoder_fn(self, X):
        
        #create bottleneck
        """
        compute z = z_mean + z_var * eps1
        """
        
        
        self.w_x_mean, self.w_x_logvar, self.z_x_mean, self.z_x_logvar, self.c_posterior = self.GMM_encoder(X)
        self.z_x_sigma = torch.exp(0.5 * self.z_x_logvar)
        # Sample Z
        self.z_x     = self.reparameterize(self.z_x_mean, self.z_x_logvar)
        
        #Build a two layers MLP to compute Q(w|x)
        self.w_x_sigma = torch.exp(0.5* self.w_x_logvar)

        
        self.w_x    = self.reparameterize(self.w_x_mean, self.w_x_logvar)

        #Build the decoder P(x|z)
        #
        self.x_recons = self.Px_given_z(self.z_x)

                
        #priorGenerator(w_sample)
        #P(z_i|w,c_i)
        #building p_zc
        self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample = self.Pz_given_wc(self.w_x)
        z_sample_list = list()
        for i in range(self.K):
            z_sample = self.reparameterize(self.z_wc_mean_list_sample[i], self.z_wc_logvar_list_sample[i])
            z_sample_list.append(z_sample)
        return z_sample_list
        

    def reconstruct_img(self, img):
        # encode image x
        #building x_recon
        
        _, _, z_loc, z_logvar, _ = self.GMM_encoder(img)
        
        
        # sample in latent space
        z      = self.reparameterize(z_loc, z_logvar)

        reconst_X = self.Px_given_z(z)
        return reconst_X

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

    

def compute_nll(x, x_recon_linear):
    #return torch.sum(func.binary_cross_entropy_with_logits(x_recon_linear, x), dim=1, keepdim=True)
    return F.binary_cross_entropy_with_logits(x_recon_linear, x)

def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    # KL divergence between two gaussian distributions
    d = torch.sub(mu_post , mu_prior)
    temp= ( 0.5 *torch.div( d.pow(2) +sigma_post.pow(2) , sigma_prior.pow(2))- 0.5 + torch.log(torch.div(sigma_prior, sigma_post))).mean(dim=0)
    return temp.sum()

def beta_fn(a,b):
    return torch.exp(torch.lgamma(torch.tensor(a)) + torch.lgamma(torch.tensor(b)) - torch.lgamma(torch.tensor(a+b)))


def compute_kumar2beta_kld(a, b, alpha, beta):

    global local_device
    SMALL = torch.tensor(1e-16, dtype=torch.float64, device=local_device)
    EULER_GAMMA = torch.tensor(0.5772156649015329, dtype=torch.float, device=local_device)

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

def log_normal_pdf(sample, mean, sigma):
    
    d = torch.sub(sample , mean)
    d2=torch.mul(-1,torch.mul(d,d))
    s2=torch.mul(2,torch.mul(sigma,sigma))
    return torch.sum(torch.div(d2,s2)-torch.log(torch.mul(sigma,np.sqrt(2*np.pi))), dim=1)



def log_beta_pdf(v, alpha, beta):
    return torch.sum((alpha - 1) * torch.log(v + 1e-20) + (beta-1) * torch.log(1 - v + 1e-20) - torch.log(beta_fn(alpha, beta) + 1e-20), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    return torch.sum(torch.mul(a - 1, torch.log(v + 1e-20)) + torch.mul(b - 1, torch.log(1 - torch.pow(v,a) + 1e-20)) + torch.log(a + 1e-20) + torch.log(b + 1e-20), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in range(K-1):
        s += torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s + 1e-20)


log_norm_constant  = -0.5 * np.log(2 * np.pi)
def log_gaussian(x, mu=0, logvar=0.):
    global local_device
	# log likelihood: 
    # llh = -0.5 sum_d { (x_i - mu_i)^2/var_i } - 1/2 sum_d (logVar_i) - D/2 ln(2pi) [N]
    return -0.5 * torch.sum(((x - mu).pow(2))/logvar.exp(), dim=1) \
			  - 0.5 * torch.sum(logvar, dim=1) + torch.tensor(log_norm_constant, dtype=torch.float, device=local_device)
        

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
    # based on this implementation :https://github.com/enalisnick/mixture_density_VAEs
    def __init__(self, hyperParams, K, nchannel, base_channels, z_dim, w_dim, hidden_dim, device, img_width, batch_size, include_elbo2):
        global local_device
        local_device = device
        super(InfGaussMMVAE, self).__init__(K, nchannel, base_channels, z_dim, w_dim, hidden_dim,  device, img_width, batch_size)

        self.prior      = hyperParams['prior']
        self.K          = hyperParams['K']
        self.z_dim      = hyperParams['latent_d']
        self.hidden_dim = hyperParams['hidden_d']
        #self.x_recons_linear = self.f_prop(X)
        
        #self.encoder_kumar_a = nn.Linear(self.K-1, self.hidden_dim, bias=True, device=device)
        #self.encoder_kumar_b = nn.Linear(self.K-1, self.hidden_dim, bias=True, device=device)
        #self.elbo_obj = self.get_ELBO()
        self.img_size    = img_width
        self.include_elbo2 = include_elbo2
        self.to(self.device)
    # def __len__(self):
    #     return len(self.X)


    def forward(self, X):
        # init variational params
        self.z_sample_list = self.encoder_decoder_fn(X)
         
        

        return self.x_recons, self.z_x_mean, self.z_x_logvar,\
               self.w_x_mean, self.w_x_logvar, self.c_posterior , self.kumar_a, self.kumar_b, \
               self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample


    def compose_stick_segments( self,v):
        segments = []
        self.remaining_stick = [torch.ones((v.shape[1],1)).to(self.device)]
        for i in range(self.K-1):
            curr_v = torch.squeeze(v[:, i],0)
            segments.append(torch.mul(curr_v, self.remaining_stick[-1]))
            self.remaining_stick.append(torch.mul(1-curr_v, self.remaining_stick[-1]))
        segments.append(self.remaining_stick[-1])
        return segments

    def GenerateMixtures(self):
        """
        #KL divergence P(c|z,w)=Q(c|x) while P(c|pi) is the prior
        """
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)
        r1    = 1e-8
        r2    = 1-1e-8
        # compose into stick segments using pi = v \prod (1-v)
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b)).to(self.device)
        #u       = (r1 - r2) * torch.rand(a_inv.shape[0],self.K-1) + r2
        u       = torch.distributions.uniform.Uniform(low=r1, high=r2).sample([1]).squeeze()
        v_samples  = torch.pow(1 - torch.pow(u, b_inv), a_inv).to(self.device)
        if v_samples.ndim > 2:
            v_samples = v_samples.squeeze()
        v0 = v_samples[:, -1].pow(0).reshape(v_samples.shape[0], 1)
        v1 = torch.cat([v_samples[:, :self.z_dim - 1], v0], dim=1)
        n_samples = v1.size()[0]
        n_dims = v1.size()[1]
        self.pi_samples = torch.zeros((n_samples, n_dims)).to(self.device)

        for k in range(n_dims):
            if k == 0:
                self.pi_samples[:, k] = v1[:, k]
            else:
                self.pi_samples[:, k] = v1[:, k] * torch.stack([(1 - v1[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)
 
        #print(f'size of pi {self.pi_samples.size()}')
        return  self.pi_samples
        


    def DiscreteKL(self, P,Q):
        #KL(q(z)||p(z)) =  - sum_k q(k) log p(k)/q(k)
   	    # let's p(k) = 1/K???
        log_q = torch.log(Q+ 1e-10).to(self.device)
        q     = Q
        log_p = torch.log(P+ 1e-10).to(self.device)
        element_wise = (q * torch.sub(log_q , log_p))
        return torch.sum(element_wise, dim=-1).mean().to(self.device)
    
    def ExpectedKLDivergence(self, q_c, mean_z, logvar_z, mean_mixture, logvar_mixture):
        # 4. E_p(c|w,z)[KL(q(z)|| p(z|c,w))]
        z_wc       = mean_z.unsqueeze(-1)
        z_wc       = z_wc.expand(-1, self.z_dim, self.K)
        z_wc       = z_wc.permute(2, 0, 1)

        logvar_zwc = logvar_z.unsqueeze(-1)
        logvar_zwc = logvar_zwc.expand(-1, self.z_dim, self.K)
        logvar_zwc = logvar_zwc.permute(2, 0, 1)
        KLD_table  = 0.5 * (((logvar_mixture - logvar_zwc) + ((logvar_zwc.exp() + (z_wc - mean_mixture).pow(2))/logvar_mixture.exp())) - 1)
        KLD_table  = KLD_table.permute(0,2,1)
        
        qc         = q_c.unsqueeze(-1)
        qc         = qc.expand(-1, self.K, 1)
        qc         = qc.permute(1,0,2)
        return torch.sum(torch.bmm(KLD_table, qc)).to(self.device)

    def EntropyCriterion(self):
        # CV = H(C|Z, W) = E_q(z,w) [ E_p(c|z,w)[ - log P(c|z,w)] ]
        z_sample =  self.z_x.unsqueeze(-1)
        z_sample =  z_sample.expand(-1, self.z_dim, self.K)
        z_sample =  z_sample.permute(2,0,1)
        
        log_likelihoods = log_gaussian(z_sample, self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample)
        llh=log_likelihoods.sum(-1)
        
        lh = F.softmax(llh, dim=-1)
        # entropy
        CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh)).to(self.device)
        return CV

    def get_ELBO(self, X):
        loss_dict = OrderedDict()
        #1) Computes the KL divergence between two categorical distributions
        PriorC  = self.GenerateMixtures()

        
        #this term KL divergence of two discrete distributions
        elbo1     = self.DiscreteKL(PriorC ,self.c_posterior)
        #print(elbo1)

        # compose elbo of Kumaraswamy-beta 
        elbo2 = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.include_elbo2:
            for k in range(self.K-1):
                #2)need this term
                #elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), \
                #                                   self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])
                #print(self.kumar_b[:, k].unsqueeze(1))
                #print(self.kumar_b[:, k].expand(self.batch_size))
                #elbo2 -= compute_kumar2beta_kld(self.kumar_a[:, k].expand(self.batch_size), self.kumar_b[:, k].expand(self.batch_size), self.prior, (self.K-1-k)* self.prior).mean()
                elbo2 -= compute_kumar2beta_kld(self.kumar_a[:, k], self.kumar_b[:, k], self.prior, (self.K-1-k)* self.prior).mean()
        
        #3)need this term of w (context)
        elbo3 = -0.5 * torch.sum(1 + self.w_x_logvar - self.w_x_mean.pow(2) - self.w_x_logvar.exp()).to(self.device)#VAE_KLDCriterion
        
        # 4)compute E_{p(w|x)p(c|x)}[D_KL(Q(z|x)||p(z|c,w))]
        
        elbo4 = self.ExpectedKLDivergence(self.c_posterior, self.z_x_mean, self.z_x_logvar, self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample)

        #5) compute Reconstruction Cost = E_{q(z|x)}[P(x|z)] 
        #
        criterion = nn.BCELoss(reduction='sum').to(self.device)
        elbo5 = criterion(self.x_recons.view(-1, self.nchannel*self.img_size*self.img_size), X.view(-1, self.nchannel*self.img_size*self.img_size))
        assert torch.isfinite(elbo5)

        
        loss_dict['recon'] = elbo5
        loss_dict['c_clusster_kld'] = elbo1
        loss_dict['kumar2beta_kld'] = elbo2
        loss_dict['w_context_kld'] = elbo3
        loss_dict['z_latent_space_kld'] = elbo4
        # 6.)  CV = H(C|Z, W) = E_q(z,w) [ E_p(c|z,w)[ - log P(c|z,w)] ]
        loss_dict['CV_entropy'] = self.EntropyCriterion()
        #print(f" Entropy {loss_dict['CV_entropy']}")
        loss_dict['loss'] = elbo1 + elbo2 + elbo3 + elbo4 + elbo5
        return loss_dict


    def get_log_margLL(self, batchSize):
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
        component_samples = torch.cat( [torch.range(0, batchSize).unsqueeze(1), component_samples.unsqueeze(1)], 1)
        # calc likelihood term for chosen components
        ll = -compute_nll(self.X, self.x_recons)
        

        # calc prior terms
        all_log_gauss_priors = []

        for k in range(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z_x, self.z_wc_mean_list_sample[k], self.z_wc_logvar_list_sample[k]))
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


        log_gauss_post = log_normal_pdf(self.z_x, self.z_x_mean, self.z_x_sigma)

        #****need this term :cal prior and posterior over w
        log_w_prior = log_normal_pdf(self.w_x, torch.zeros(self.w_x.size()),torch.eye(self.w_x.size()))
        log_w_post  = log_normal_pdf(self.w_x, self.w_x_mean, self.w_x_sigma)
        return ll + log_beta_prior + log_gauss_prior +log_w_prior - log_kumar_post - log_gauss_post - log_w_post


    #TODO??
    @torch.no_grad()
    def get_component_samples(self,batchSize):
        # get the components of the latent space 
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)
        r1=1e-8
        r2=1-1e-8
        # compose into stick segments using pi = v \prod (1-v)
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b)).to(self.device)
        #u       = (r1 - r2) * torch.rand(a_inv.shape[0],self.K-1) + r2
        u       = torch.distributions.uniform.Uniform(low=r1, high=r2).sample([1]).squeeze()
        v_samples  = torch.pow(1 - torch.pow(u, b_inv), a_inv).to(self.device)
        if v_samples.ndim > 2:
            v_samples = v_samples.squeeze()
        v0 = v_samples[:, -1].pow(0).reshape(v_samples.shape[0], 1)
        v1 = torch.cat([v_samples[:, :self.z_dim - 1], v0], dim=1)
        n_samples = v1.size()[0]
        n_dims = v1.size()[1]
        components = torch.zeros((n_samples, n_dims))

        for k in range(n_dims):
            if k == 0:
                components[:, k] = v1[:, k]
            else:
                components[:, k] = v1[:, k] * torch.stack([(1 - v1[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)
        # ensure stick segments sum to 1
        assert_almost_equal(torch.ones(n_samples), components.sum(axis=1).detach().numpy(),
                            decimal=2, err_msg='stick segments do not sum to 1')
        print(f'size of sticks: {components}')
        
        
        components = torch.IntTensor(torch.argmax(torch.cat( self.compose_stick_segments(v_means),1) ,1), dtype=torch.long, device=self.device)
        components = torch.cat( [torch.range(0, batchSize).unsqueeze(1), components.unsqueeze(1)], 1)
        print(f'size of sticks: {components}')
        all_z = []
        for d in range(self.z_dim):
            temp_z = torch.cat(1, [self.z_sample_list[k][:, d].unsqueeze(1) for k in range(self.K)])
            all_z.append(gather_nd(temp_z, components).unsqueeze(1))
        out       = torch.cat( all_z,1)
        x_samples = self.Px_given_z(out)
        return x_samples



#class of optimizer
from utils import AdaBound
#data analysis
from torch import nn, optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.utils
import glob
import re
import sys
import copy
import time
import argparse
#import wandb
#plotting using visdom
from visdom import Visdom
from visdom import server
from subprocess import Popen, PIPE
from pathlib import Path
###***********************Tensorboard************************###
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/STL10_experiment_1')
###**********************************************************###
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Stick Breaking Gaussian Mixture VAE....')
parser.add_argument('--visdom_server', default="http://localhost",help='visdom server of the web display')
parser.add_argument('--visdom_port', default=8097, help='visdom port of the web display')
args = parser.parse_args()
#server.start_server()
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main',server=args.visdom_server, port=args.visdom_port):
        
        print("Initializing visdom env [%s]" % env_name)
        self.port = port
        self.viz  = Visdom(port =port,server=server, log_to_filename='vis_log_file')
        if not self.viz.check_connection():
                self.create_visdom_connections()
        self.env = env_name
        self.plots = {}
    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %s &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    
    def add_image(self, images, win, opts=None):
        """ vis image in visdom
        """
        default_opts = dict(title=win)
        if opts is not None:
            default_opts.update(opts)
        self.viz.image(images, win=win, opts=default_opts, env=self.env)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
device = torch.device('cuda')

hyperParams = {"batch_size": 50,
               "input_d": 1,
               "prior": 1, #dirichlet_alpha'
               "K": 20,
               "hidden_d": 500,
               "latent_d": 200,
               "latent_w": 150}


# # Preparing Data : STL10
print("Loading trainset...")
train_dataset = datasets.STL10('../data', split='train', transform=transform,
                              target_transform=None, download=True)
train_loader = DataLoader.DataLoader(train_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=0)
print("Loading testset...")
test_dataset = datasets.STL10('../data', split='test', transform=transform,
                             target_transform=None, download=True)
test_loader  = DataLoader.DataLoader(test_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=0)

class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


print("Done!")
##########

train_dataset_data = train_dataset.data
train_dataset_label = train_dataset.labels
print("training data labels ", train_dataset_label.shape)
label_class = np.array(list(set(train_dataset.labels)))
print(f"size of label's class {label_class.shape}")


# get some random training images
data, label  = iter(train_loader).next()
###***********************Tensorboard************************###
# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# create grid of images

img_grid = torchvision.utils.make_grid(data)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('STL10_images', img_grid)
writer.close()
###*********************************************************###

test_data, test_label  = iter(test_loader).next()
img_width = test_data.shape[3]
print('image size in test data ', img_width)

print('data shape', data.shape)
print('label shape', label.shape)
raw_img_width = data.shape[3]
print('images size in the training data: ', raw_img_width)


net = InfGaussMMVAE(hyperParams, 20, 3, 4, 200, 150, 500, device, raw_img_width, hyperParams["batch_size"],include_elbo2=True)

params = list(net.parameters())
#if torch.cuda.device_count() > 1:
#    net = nn.DataParallel(net)
net = net.to(device)
for name, param in net.named_parameters():
    if param.device.type != 'cuda':
        print('param {}, not on GPU'.format(name))
###*********************************************************###

#net.cuda()
#optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8)

optimizer = AdaBound(net.parameters(), lr=0.0001)

img_list = []
num_epochs=100
iters=0
def train(epoch):
    global iters
    train_loss_avg = []
    net.train()
    train_loss = 0
    train_loss_avg.append(0)
    num_batches = 0
    datas, targets = [], []
    for batch_idx, (X, classes) in enumerate(train_loader):
        
        X = X.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        
        X_recons_linear, mu_z, logvar_z, mu_w, logvae_w, qc, kumar_a, kumar_b, mu_pz, logvar_pz= net(X)
        
        loss_dict = net.get_ELBO(X)
        
        loss_dict['loss'].backward()
        train_loss += loss_dict['loss'].item()
        optimizer.step()

        train_loss_avg[-1] += loss_dict['loss'].item()
        num_batches += 1
        if batch_idx % 5 == 0:
                
            print('epoch {} --- iteration {}: '
                      'kumar2beta KL = {:.6f} '
                      'Z latent space KL = {:.6f} '
                      'reconstruction loss = {:.6f} '
                      'W context latent space KL = {:.6f}'.format(epoch, batch_idx, loss_dict['kumar2beta_kld'].item(), loss_dict['z_latent_space_kld'].item(), loss_dict['recon'].item(),loss_dict['w_context_kld'].item()))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_dict['loss'].item() / len(X)))
        if (epoch > 1) and (epoch % 50 == 0):
            #plot low dimensional embedding
            z_loc = net.get_component_samples()
            model_tsne = TSNE(n_components=2, random_state=0)
            z_states = z_loc.detach().cpu().numpy()
            z_embed = model_tsne.fit_transform(z_states)
            classes = classes.detach().cpu().numpy()
            order_classes=set()
            ls =[x for x in classes if x not in order_classes and  order_classes.add(x) is None]
            fig = plt.figure()
            for ic in range(len(ls)):
                ind_class = classes == ic
                color = plt.cm.Set1(ic)
                plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=1, color=color)
                plt.title("Latent Variable T-SNE per Class")
                fig.savefig(str(Path().absolute())+"/results/StickBreaking_GMM_VAE_embedding_" + str(ic) + "_epoch_"+str(epoch)+".png")
            fig.savefig(str(Path().absolute())+"/results/StickBreaking_GMM_VAE_embedding_epoch_"+str(epoch)+".png")

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (batch_idx == len(train_loader)-1)):
             
             img_list.append(torchvision.utils.make_grid(X_recons_linear[0].detach().cpu(), padding=2, normalize=True))
        iters += 1
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_loss_avg[-1] /= num_batches
    print(f' average training loss : {train_loss_avg}')
    return train_loss_avg, X_recons_linear, loss_dict['kumar2beta_kld'].item()



avg_train_loss=[]
best_loss = 10**15  # Random big number (bigger than the initial loss)
best_epoch = 0
regex = re.compile(r'\d+')
start_epoch = 0 
if os.path.isfile(str(Path().absolute())+"/results/model_StickBreaking_GMM_VAE_*.pth"):
   list_of_files = glob.glob(str(Path().absolute())+"/results/model_StickBreaking_GMM_VAE*.pth") # * means all if need specific format then *.csv
   latest_file = max(list_of_files, key=os.path.getctime)
   print("latest saved model: ")
   print(latest_file)
   checkpoint = torch.load(latest_file)
   net.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   best_loss   = checkpoint['best_loss']
   start_epoch = int(regex.findall(latest_file)[-1])


####Training

global plotter
plotter = VisdomLinePlotter(env_name='InfGMMVAE_PLOTS')
best_model_wts = copy.deepcopy(net.state_dict())
best_opt       = copy.deepcopy(optimizer.state_dict())
for epoch in range(start_epoch,num_epochs):

       average_epoch_loss, out , elbo2 =train(epoch)
       avg_train_loss.extend(average_epoch_loss)
       if epoch % 25 == 0:

          img =torchvision.utils.make_grid(out.detach().cpu())
          img = 0.5*(img + 1)
          npimg = np.transpose(img.numpy(),(1,2,0))
          fig = plt.figure(dpi=300)
          plt.imshow(npimg)
          plt.imsave(str(Path().absolute())+"/results/reconst_"+str(epoch)+"_epochs.png", npimg)
       # plot beta-kumaraswamy loss
       plotter.plot('KL of beta-kumaraswamy distributions', 'val', 'Class Loss', epoch, elbo2)
       
       # plot loss
       #print(average_epoch_loss)
       #print(epoch)
       plotter.plot('Total loss', 'train', 'Class Loss', epoch, average_epoch_loss[0])
       
       
       if epoch % 25 == 0:
          torch.save({
                    'best_epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': best_opt,
                    'best_loss': best_loss,
          }, str(Path().absolute())+"/results/model_StickBreaking_GMM_VAE_"+str(epoch)+".pth")
          
       if average_epoch_loss[-1] < best_loss:
          best_loss = average_epoch_loss[-1]
          best_epoch = epoch
          best_model_wts = copy.deepcopy(net.state_dict())
          best_opt       = copy.deepcopy(optimizer.state_dict())




print('Finished Trainning')


#plot the loss values of trained model
fig = plt.figure()
plt.plot(avg_train_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(str(Path().absolute())+"/results/Loss_InfGMM_VAE.png")
net.eval()
# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(data[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

###plot original versus fake images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(str(Path().absolute())+"/results/real_vs_fake.jpg")

test_loss_avg, num_batches = 0, 0
net.eval()
for image_batch, _ in test_loader:

        with torch.no_grad():

            image_batch = image_batch.to(device,dtype=torch.float)

            # VAE Reconstruction
            reco_img = net.reconstruct_img(image_batch)
            
            image_numpy = image_batch[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            ###
            reco_numpy = reco_img[0].cpu().float().numpy()
            reco_numpy = (np.transpose(reco_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            
            plotter.add_image( 
                        image_numpy.transpose([2, 0, 1]), 
                        win='Real', 
                        opts={"caption": "test image"},
            )
            plotter.add_image(
                        reco_numpy.transpose([2, 0, 1]),
                        win='Fake',
                        opts={"caption": "reconstructed image"},
            )


        # reconstruction error
        loss_dict = net.get_ELBO(image_batch)

        test_loss_avg += loss_dict['loss'].item()
        num_batches += 1

test_loss_avg /= num_batches
print('average reconstruction error: %f' % (test_loss_avg))
plotter.viz.save([plotter.env])###???



def show_image(img, filename):
    img = img.clamp(0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)

def visualise_output(images, model,filename):

    with torch.no_grad():
        model.eval()
        images = images.to(device)
        
        out, _, _, _, _, _, _, _, _, _ = model(images)
        img =torchvision.utils.make_grid(out.detach().cpu())
        img = 0.5*(img + 1)
        npimg = np.transpose(img.numpy(),(1,2,0))
        fig = plt.figure(dpi=300)
        plt.imshow(npimg)   
    plt.savefig(filename)

images, labels = iter(test_loader).next()
show_image(torchvision.utils.make_grid(images[1:50],10,5), str(Path().absolute())+"/results/Original_Images_InfGMM_VAE.png")
visualise_output(images, net,str(Path().absolute())+"/results/Reconstructed_Images_InfGMM_VAE.png")
