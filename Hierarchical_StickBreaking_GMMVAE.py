#Stick breaking Gaussian Mixture Prior for variational autoencoders
import logging
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Independent, Gamma, Beta, MixtureSameFamily
import torch.nn.init as init
from copy import deepcopy
from torch.autograd import Variable
from types import SimpleNamespace as SN
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import pdb
from typing import TypeVar, List
from abc import ABCMeta, abstractmethod
Tensor = TypeVar('torch.tensor')
from numpy.testing import assert_almost_equal
from einops import rearrange
import os
#mypdb=pdb.Pdb(stdin=open('fifo_stdin','r'), stdout=open('fifo_stdout','w'))
try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import AverageMeter, ResidualBlock, LinearResidual, ResidualBlock_deconv
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
##########################################
#####  author: Zahra Sheikhbahaee   ######
## affiliation: University of Waterloo ###
##########################################
#set randoom seed
#torch.random.seed()
#np.random.seed()
#if torch.cuda.is_available():
#   torch.cuda.seed()

#class of optimizer
from utils import AdaBound
#data analysis
from torch import nn, optim
from torchvision.utils import save_image, make_grid
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
#from visdom import Visdom
#from visdom import server
from subprocess import Popen, PIPE
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
###***********************Tensorboard************************###
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
if torch.cuda.is_available():
   torch.cuda.manual_seed(42)
np.random.seed(42)

local_device = torch.device('cuda')
SMALL = torch.tensor(np.finfo(np.float32).eps, dtype=torch.float64, device=local_device)
pi_   = torch.tensor(np.pi, dtype=torch.float64, device=local_device)
log_norm_constant  = -0.5 * torch.log(2 * pi_)

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

def calculate_layer_size( input_size, kernel_size, stride, padding=0):
        numerator = input_size - kernel_size + (2 * padding)
        denominator = stride
        return (numerator // denominator) + 1

def calculate_channel_sizes( image_channels, max_filters, num_layers):
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
       
       #params['w'].append(torch.nn.init.normal_(torch.empty(n_in, n_out, device=local_device),std= torch.div(1,torch.sqrt(torch.from_numpy(np.asarray(n_in+n_out)).float()/2.))).requires_grad_(True))
       params['w'].append(torch.nn.init.normal_(torch.empty(n_in, n_out, device=local_device), std= stdev).requires_grad_(True))
       params['b'].append(torch.empty(n_out, device=local_device).fill_(bias_init).requires_grad_(True))
    return params

def mlp(X, params):
    h = [X]
    activation = nn.Softplus(beta=0.5, threshold=20)
    for w, b in zip(params['w'][:-1], params['b'][:-1]):
        h.append(activation(torch.matmul(h[-1], w) + b))
    return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]

##################
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
##############
class DistributionSample(object):
    __metaclass__ = ABCMeta
    def __init__(self, seed=1, **kwargs):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.srng  = torch.Generator(device=self.device).manual_seed(seed)
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
        #print(f"device of GammaSample class is {self.device}")

    def gamma_sample(self, alpha, beta):
        _shape = alpha.shape
        alpha = alpha.flatten()
        output_sample = -torch.ones_like(alpha, device=self.device, dtype=alpha.dtype)
        index = torch.arange(output_sample.shape[0], device=self.device)
        under_one_idx = torch.gt(torch.ones_like(alpha), alpha[index])
        added_alpha = alpha.clone()
        added_alpha[under_one_idx]+= 1
        U = torch.empty(under_one_idx[0].shape, device=self.device,  dtype=alpha.dtype).uniform_(epsilon(), 1 - epsilon()).to(device=self.device)
        if self.rejection_sampling:
            # We don't use theano.scan in order to avoid to use updates.
            for _ in range(self.iter_sampling):
                output_sample, index = self._rejection_sampling(output_sample,
                                                                added_alpha,
                                                                index)
        else:
            output_sample = self._not_rejection_sampling(alpha)
        output_sample = torch.clamp(output_sample,torch.zeros_like(output_sample), output_sample)
        output_sample[under_one_idx]= (U ** (1 / alpha[under_one_idx])) *output_sample[under_one_idx]
        return output_sample.reshape(_shape) / beta

    def log_gamma_pdf(self, samples, alpha, beta):
        output = alpha * torch.log(beta + epsilon()) - torch.lgamma(alpha+ epsilon())
        output = output + (alpha - 1) * torch.log(samples + epsilon())
        output = output - beta * samples
        return mean_sum_samples(output)

    def _h(self, alpha, eps):
        d = alpha - 1 / 3.
        c = torch.reciprocal( torch.sqrt(9 * d)+ epsilon())
        v = torch.pow((1 + c * eps),3)
        judge_1 = torch.exp(0.5 * torch.pow(eps,2) + d - d * v + d * torch.log(v+epsilon()))
        judge_2 = -1 / c
        output = d * v
        return output, judge_1, judge_2


    def _rejection_sampling(self, output_z, alpha, idx):
        
        eps = torch.empty(idx.shape, device=self.device, dtype=alpha.dtype).normal_(0, 1, generator=self.srng).to(device=self.device)
        U   = torch.empty(idx.shape, device=self.device, dtype=alpha.dtype).uniform_(epsilon(), 1 - epsilon(), generator=self.srng).to(device=self.device)
        z, judge1, judge2 = self._h(alpha[idx], eps)
        _idx_binary = torch.logical_and(torch.lt(U, judge1), torch.gt(eps, judge2))

        output_z[idx[_idx_binary.nonzero()]]= z[_idx_binary.nonzero()]
        # update idx
        idx = idx[torch.eq(torch.zeros_like(_idx_binary), _idx_binary).squeeze(-1)]
        return output_z, idx


    def _not_rejection_sampling(self, alpha):
        eps = torch.empty(alpha.shape, device=self.device).normal_(0, 1, generator=self.srng).type(alpha.dtype)
        z, _, _ = self._h(alpha, eps)
        return z
##################
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

        return torch.div(z_1 , (z_1 + z_2)+epsilon())

    def log_beta_pdf(self, samples, alpha, beta):
        output = (alpha - 1) * torch.log(samples + epsilon())
        output = output + (beta - 1) * torch.log(1 - samples + epsilon())
        output = output - self._log_beta_func(alpha, beta)
        return mean_sum_samples(output)

    def _log_beta_func(self, alpha, beta):
        return torch.lgamma(alpha+ epsilon()) + torch.lgamma(beta + epsilon()) - torch.lgamma(alpha + beta + epsilon())


##################
def I_function(alpha_q, beta_q, alpha_p, beta_p):
    return - torch.div((alpha_p * beta_p) , alpha_q+ epsilon()) - \
                beta_q * torch.log(alpha_q+ epsilon()) - torch.lgamma(beta_q+ epsilon()) + \
                (beta_q - 1) * torch.digamma(beta_p+ epsilon()) + \
                (beta_q - 1) * torch.log(alpha_p+ epsilon())
    

def gamma_kl_loss( a, b, c, d):
    a      =  torch.reciprocal(a+epsilon()) 
    c      =  torch.reciprocal(c+epsilon()) 
    losses =  I_function(c, d, c, d) - I_function(a, b, c, d)
    return torch.sum(losses, dim=1)


def init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class VAEEncoder(nn.Module):
    def __init__(self,nchannel, z_dim, hidden_dim, img_width, max_filters=512, num_layers=4, small_conv=False):
        super(VAEEncoder,self).__init__()
        self.nchannel    = nchannel
        self.z_dim       = z_dim
        self.hidden_dim  = hidden_dim
        self.img_width   = img_width
        self.enc_kernel  = 4
        self.enc_stride  = 2
        self.enc_padding = 0 
        self.res_kernel  = 3
        self.res_stride  = 1
        self.res_padding = 1
        ########################
        # ENCODER-CONVOLUTION LAYERS
        if small_conv:
            num_layers += 1
        channel_sizes = calculate_channel_sizes(
            self.nchannel, max_filters, num_layers
        )
        #print(f"channel size of encoder:{channel_sizes}")
        # Encoder
        encoder_layers = nn.ModuleList()
        # Encoder Convolutions
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                # 1x1 Convolution
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.enc_kernel,
                        stride=self.enc_stride,
                        padding=self.enc_padding,
                    )
                )
            else:
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.enc_kernel,
                        stride=self.enc_stride,
                        padding=self.enc_padding,
                        bias=False,
                    )
                )
            # Batch Norm
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU
            encoder_layers.append(nn.ReLU())
            if (i==num_layers//2):
                #add a residual Layer
                encoder_layers.append(
                    ResidualBlock(
                        out_channels,
                        self.res_kernel,
                        self.res_stride,
                        self.res_padding,
                        nonlinearity=nn.ReLU()
                    )
                )
            
        # Flatten Encoder Output
        encoder_layers.append(nn.Flatten())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate shape of the flattened image
        self.h_dim, self.h_image_dim = self.get_flattened_size(self.img_width)
        
        self.encoder = nn.Sequential(
        self.encoder, nn.Linear(self.h_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
        ) # ReLU
        
        #print(f"encode network:\n{self.encoder}")
        ########################
        #ENCODER-USING FULLY CONNECTED LAYERS
        #THE LATENT SPACE (Z)
        # mean of z

        self.fc_mu         = nn.Linear(hidden_dim, z_dim, bias=False)
        
        # variance of z

        self.fc_log_var    = nn.Linear(hidden_dim, z_dim, bias=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
    
    def forward(self,X):
            
            # Encode
            self.hlayer = self.encoder(X)
            # Get latent variables
        
            #mean_z
            mu_z        = self.fc_mu(self.hlayer)

            #logvar_z
            logvar_z    = self.fc_log_var(self.hlayer)
            z = self.reparameterize(mu_z, logvar_z)
            return z, mu_z, logvar_z
        
    def reparameterize(self, mu, log_var):
        
        eps = torch.randn_like(log_var, device = self.device)
        return torch.add(mu, torch.mul( log_var.mul(0.5).exp_(), eps))

    def get_flattened_size( self, image_size ):
        #for param in self.encoder.parameters():
        #    print(type(param.data), param.size())
        
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
    #https://github.com/akashsara/fusion-dance
    def __init__(self,nchannel, z_dim, hidden_dim, extend_dim, h_image_dim, img_width, max_filters=512, num_layers=4, small_conv=False):
        super(VAEDecoder,self).__init__()
        self.nchannel   = nchannel
        self.z_dim      = z_dim
        self.hidden_dim = hidden_dim
        self.img_width  = img_width
        self.dec_kernel = 4
        self.dec_stride = 2
        self.dec_padding = 0 
        self.res_kernel  = 3
        self.res_stride  = 1
        self.res_padding = 1
        ########################
        # DECODER: CreateXGenerator
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
        decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(torch.nn.Linear(hidden_dim, extend_dim, bias=False))
        decoder_layers.append(nn.BatchNorm1d(extend_dim))
        decoder_layers.append(nn.ReLU())
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(nn.Unflatten(1, (int(extend_dim / (h_image_dim * h_image_dim)), h_image_dim, h_image_dim)))
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
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(nn.ReLU())
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
            if (i==num_layers//2):
                #add a residual Layer
                decoder_layers.append(
                    ResidualBlock_deconv(
                        out_channels,
                        self.res_kernel,
                        self.res_stride,
                        self.res_padding,
                        nonlinearity=nn.ReLU()
                    )
                )
                
        self.decoder = nn.Sequential(*decoder_layers)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
        #print(f"decode network:\n{self.decoder}")
    def forward(self, x):
        return self.decoder(x)
   
class VAECritic(nn.Module):
    # define the descriminator/critic
    def __init__(self, input_dims, num_layers=4):
        super(VAECritic,self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dims, input_dims * 2, bias=False))

        layers.append(nn.BatchNorm1d(input_dims * 2))
        # Activation Function
        layers.append(nn.LeakyReLU(0.2))
        size = input_dims * 2
        # Fully Connected Block
        for i in range(num_layers-2):
            # residual feedforward Layer
            layers.append( nn.Linear(size, size //2))
            layers.append(nn.BatchNorm1d(size // 2))
            layers.append(nn.LeakyReLU(0.2))
            if (i==(num_layers//2-1)):
                #add a residual block
                LinearResidual(size//2)
            size = size// 2
        layers.append(nn.Linear(size, size * 2, bias=False))
        layers.append(nn.BatchNorm1d(size * 2))
        # Activation Function
        layers.append(nn.LeakyReLU(0.2))
        #add anther residual block
        LinearResidual(size*2)
        layers.append(nn.Linear(size*2, 1))
        self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def forward(self, x):
        return self.model(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class GMMVAE(nn.Module):
    #Used this repository as a base
    # https://github.com/bhavikngala/gaussian_mixture_vae
    # https://github.com/Nat-D/GMVAE
    # https://github.com/psanch21/VAE-GMVAE
    def __init__(self, number_of_mixtures, nchannel, z_dim, w_dim, hidden_dim,  device, img_width, batch_size, max_filters=512, num_layers=4, small_conv=False):
        super(GMMVAE, self).__init__()

        self.K          = number_of_mixtures
        self.nchannel   = nchannel
        self.z_dim      = z_dim
        self.w_dim      = w_dim
        self.hidden_dim = hidden_dim
        self.device     = device
        self.img_width  = img_width
        self.batch_size = batch_size
        self.enc_kernel = 4
        self.enc_stride = 2
        self.enc_padding = 0 
        self._to_linear = None
        ####################### Encoder & Decoder of Z #####################
        self.encoder  = VAEEncoder(nchannel, z_dim, hidden_dim, img_width, max_filters, num_layers, small_conv)
        self.decoder  = VAEDecoder(nchannel, z_dim, hidden_dim, self.encoder.h_dim, self.encoder.h_image_dim, img_width, max_filters, num_layers, small_conv)
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
        
        self.encoder_kumar_a = init_mlp([hidden_dim,self.K-1], 1e-7) #Q(pi|x)
        self.encoder_kumar_b = init_mlp([hidden_dim,self.K-1], 1e-7) #Q(pi|x)
        ########## Gamma #########
        #create prior over alpha variable of stick-breaking
        # psterior distribution of Gamma distribution variables
        self.encoder_gamma_a = init_mlp([hidden_dim,self.K-1], 1e-8) #Q(pi|x)
        self.encoder_gamma_b = init_mlp([hidden_dim,self.K-1], 1e-8) #Q(pi|x)
        

        ########## Device ############# 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    

    def Pz_given_wc(self, w_input):
        # Prior
        #prior generator P(Z|w,c)

        h = self.pz_wc_fc0(w_input)
        h = torch.tanh(self.pz_wc_bn1d_0(h))
        z_wc_mean_list = list()
        for i in range(self.K):
            Pz_given_wc_mean = F.softplus(self.pz_wc_bn_mean[i](self.pz_wc_fc_mean[i](h)))
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
        Generated_X = self.decoder(z_input)
        Generated_X = F.pad(input=Generated_X, pad=(1, 1, 1, 1), mode='constant', value=0)
        #print(f'size of recounstructed image: {Generated_X.size()}')
        return Generated_X

    def GMM_encoder(self, data):
        #1) posterior Q(z|X)
        """
        compute z = z_mean + z_var * eps1
        """
        h = data
        # 1) Cmpute posterior of latent variables Q(Z|x) --->Encoder  (mean_z, logvar_z)
        z_x, mu_z, logvar_z = self.encoder(h)
        
        #2) posterior Q(w|X)
        #Create Recogniser for W
        hw                = self.flatten_raw_img(data)
        hw                = F.relu(self.bn1d_4(self.fc4(hw)))
        hlayer_w          = F.relu(self.bn1d_5(self.fc5(hw)))
        #mean of W
        mu_w              = F.relu(self.bn1d_6(self.fc6(hlayer_w)))
        #variance of Q(w|x) distribution

        #log(variance of W)
        logvar_w          = F.softplus(self.bn1d_7(self.fc7(hlayer_w)))

        #3) posterior P(c|w,z)=Q(c|X)
        #posterior distribution of P(c|w,z^{i})=Q(c|x) where z^{i}~q(z|x)
        #combine hidden layers after convolutional layers for Z latent space and feed forward layers of W latent space
        zw               = torch.cat([self.encoder.hlayer,hlayer_w],1)
        hc               = zw
        hc               = F.relu(self.bn1d_8(self.fc8(hc)))
        Pc_wz            = F.relu(self.bn1d_9(self.fc9(hc)))
        c_posterior      = F.softmax(Pc_wz, dim=-1).to(device=self.device)

        #4) posterior of Kumaraswamy given input images 
        #P(kumar_a,kumar_b|X)

        self.kumar_a = torch.exp(mlp(self.encoder.hlayer,self.encoder_kumar_a))
        self.kumar_b = torch.exp(mlp(self.encoder.hlayer,self.encoder_kumar_b))
        
        #5) posterir of gamma 
        # P(alpha, beta| X)
        self.gamma_alpha = torch.exp(mlp(self.encoder.hlayer,self.encoder_gamma_a)) 
        self.gamma_beta  = torch.exp(mlp(self.encoder.hlayer,self.encoder_gamma_b))
        
        #print(f"size of gamma distribution variable: { self.gamma_alpha.size()}")
        #print(f"kumaraswamy a: {self.kumar_a[:,self.K-2 ]}")
        #print(f"kumaraswamy b: {self.kumar_b[:,self.K-2 ]}")
        return mu_w, logvar_w, z_x, mu_z, logvar_z, c_posterior

    def encoder_decoder_fn(self, X):

        #create bottleneck
        """
        compute z = z_mean + z_var * eps1
        """

        # Sample Z
        self.w_x_mean, self.w_x_logvar, self.z_x, self.z_x_mean, self.z_x_logvar, self.c_posterior = self.GMM_encoder(X)
        self.z_x_sigma = torch.exp(0.5 * self.z_x_logvar)
        
        #Build a two layers MLP to compute Q(w|x)
        self.w_x_sigma = torch.exp(0.5* self.w_x_logvar)

        self.w_x       = self.encoder.reparameterize(self.w_x_mean, self.w_x_logvar)

        #Build the decoder P(x|z)
        #
        self.x_recons  = self.Px_given_z(self.z_x)

        #priorGenerator(w_sample)
        #P(z_i|w,c_i)
        #building p_zc

        self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample = self.Pz_given_wc(self.w_x)
        z_sample_list = list()
        for i in range(self.K):
            z_sample = self.encoder.reparameterize(self.z_wc_mean_list_sample[i], self.z_wc_logvar_list_sample[i])
            z_sample_list.append(z_sample)
        #pdb.set_trace()
        return z_sample_list


    def reconstruct_img(self, img):
        # encode image x
        #building recunstruction of data

        _, _, z_x, _, _, _ = self.GMM_encoder(img)
        
        return self.Px_given_z(z_x)


def compute_nll(x, x_recon_linear):
    #return torch.sum(func.binary_cross_entropy_with_logits(x_recon_linear, x), dim=1, keepdim=True)
    return F.binary_cross_entropy_with_logits(x_recon_linear, x)

def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    # KL divergence between two gaussian distributions
    d = torch.sub(mu_post , mu_prior)
    temp= ( 0.5 *torch.div( d.pow(2) +sigma_post.pow(2) , sigma_prior.pow(2))- 0.5 + torch.log(torch.div(sigma_prior, sigma_post))).mean(dim=0)
    return temp.sum()

def beta_fn(a,b):
    global local_device
    return torch.exp(torch.lgamma(torch.tensor(a + epsilon() , dtype=torch.float64, device=local_device)) + torch.lgamma(torch.tensor(b + epsilon() , dtype=torch.float64, device=local_device)) - torch.lgamma(torch.tensor(a + b + epsilon() , dtype=torch.float64, device=local_device)))


def gradient_penalty(critic, real, fake, device):
    #wasserstein distance with gradient penalty model
    BATCH_SIZE = real.shape[0]
    #one epsilon value for each
    epsilon = torch.rand((BATCH_SIZE,1)).to(device)
    epsilon = epsilon.expand_as(real)
    # Get random interpolation between real and fake samples
    interpolated_images = real*epsilon + fake * (1-epsilon)
    # set it to require grad info
    interpolated_images = torch.autograd.Variable(interpolated_images, requires_grad=True).to(device)
    #calculate critic score
    mixed_scores        = critic(interpolated_images)
    #calculate gradient of mixed score w.r.t. interpolated images
    gradient = torch.autograd.grad(
        inputs      = interpolated_images,
        outputs     = mixed_scores,
        grad_outputs= torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient      =  gradient.view(gradient.shape[0],-1)
    gradient_norm =  gradient.norm(2, dim=1)
    gradient_penalty =  torch.mean(torch.pow(gradient_norm-1 ,2))
    return gradient_penalty

def compute_kumar2beta_kld(a, b, alpha, beta):
    ####### Error occurs here ##############
    global local_device
    
    EULER_GAMMA = torch.tensor(0.5772156649015328606, dtype=torch.float, device=local_device)
    
    upper_limit = 10000.0

    ab    = torch.mul(a,b)
    a_inv = torch.reciprocal(a )
    b_inv = torch.reciprocal(b )
    
    # compute taylor expansion for E[log (1-v)] term
    kl    = torch.mul(torch.pow(1+ab,-1), torch.clamp(beta_fn(a_inv, b), epsilon(), upper_limit))
    #print(f"1st value of KL divergence between kumaraswamy and beta distributions: {kl}")
    for idx in range(10):
        kl = kl + torch.mul(torch.pow(idx+2+ab,-1), torch.clamp(beta_fn(torch.mul(idx+2., a_inv), b), epsilon(), upper_limit))
    kl    = torch.mul(torch.mul(beta-1,b), kl)
    """
    log_taylor = torch.logsumexp(torch.stack([beta_fn(torch.mul(m , a_inv), b) - torch.log(m + ab) for m in range(1, 10 + 1)], dim=-1), dim=-1)
    kl = torch.mul(torch.mul((beta - 1) , b) , torch.exp(log_taylor))
    """
    #print(f"2nd value of KL divergence between kumaraswamy and beta distributions: {kl}")
    #
    #psi_b = torch.log(b + SMALL) - 1. / (2 * b + SMALL) -    1. / (12 * b**2 + SMALL)
    psi_b = torch.digamma(b + epsilon())
    kl   = kl + torch.mul(torch.div(a-alpha, torch.clamp(a, epsilon(), upper_limit)), -EULER_GAMMA - psi_b - b_inv)
    #print(f"3rd value of KL divergence between kumaraswamy and beta distributions: {kl}")
    # add normalization constants
    kl   = kl + torch.log(torch.clamp(ab, epsilon(), upper_limit)) + torch.log(torch.clamp(beta_fn(alpha, beta), epsilon(), upper_limit))
    #print(f"4th value of KL divergence between kumaraswamy and beta distributions: {kl}")
    #  final term
    kl   = kl + torch.div(-(b-1),torch.clamp(b , epsilon(), upper_limit))
    #print(f"final value of KL divergence between kumaraswamy and beta distributions: {kl}")
    #pdb.set_trace()
    return  torch.clamp(kl, min=0.)

def log_normal_pdf(sample, mean, sigma):
    global local_device
    global SMALL 
    d  = torch.sub(sample , mean).to(device=local_device)
    d2 = torch.mul(-1,torch.mul(d,d)).to(device=local_device)
    s2 = torch.mul(2,torch.mul(sigma,sigma)).to(device=local_device)
    return torch.sum(torch.div(d2, s2+SMALL) - torch.log(torch.mul(sigma, torch.sqrt(2*torch.tensor(pi_, dtype=torch.float, device=local_device)))+SMALL), dim=1)



def log_beta_pdf(v, alpha, beta):
    global SMALL 
    return torch.sum((alpha - 1) * torch.log(v + SMALL ) + (beta-1) * torch.log(1 - v + SMALL ) - torch.log(beta_fn(alpha, beta) + SMALL ), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    global SMALL 
    return torch.sum(torch.mul(a - 1, torch.log(v + SMALL )) + torch.mul(b - 1, torch.log(1 - torch.pow(v,a) + SMALL )) + torch.log(a + SMALL ) + torch.log(b + SMALL ), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    global SMALL 
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in range(K-1):
        s = s + torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s + SMALL )



def log_gaussian(x, mu=0, logvar=0.):
    global local_device
    global log_norm_constant 
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
    return F.softmax(y / temperature, dim=-1).to(device=local_device)

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

### Gaussian Mixture Model VAE Class
class InfGaussMMVAE(GMMVAE,BetaSample):
    # based on this implementation :https://github.com/enalisnick/mixture_density_VAEs
    def __init__(self, hyperParams, K, nchannel, z_dim, w_dim, hidden_dim, device, img_width, batch_size, include_elbo2=True):
        global local_device
        local_device = device
        super(InfGaussMMVAE, self).__init__(K, nchannel, z_dim, w_dim, hidden_dim,  device, img_width, batch_size, max_filters=512, num_layers=4, small_conv=False)
        BetaSample.__init__(self)
       
        self.priors          = {"alpha": hyperParams['prior_alpha'], "beta": hyperParams['prior_beta']}
        self.K               = hyperParams['K']
        self.z_dim           = hyperParams['latent_d']
        self.hidden_dim      = hyperParams['hidden_d']
        self.img_size        = img_width
        self.include_elbo2   = include_elbo2
        self.to(device=self.device)
        self.dummy_param = nn.Parameter(torch.empty(0))

    # def __len__(self):
    #     return len(self.X)


    def forward(self, X):
        # init variational params
        # device = self.dummy_param.device
        # print(f"device in the InfGaussMMVAE class is {device}")
        
        self.z_sample_list = self.encoder_decoder_fn(X)

        self.prior_nu = self.gamma_sample(self.gamma_alpha, self.gamma_beta)
        #print(f"size of alpha (variable of beta distribution): {self.prior_nu.size()}")
        return self.x_recons, self.z_x_mean, self.z_x_logvar,\
               self.w_x_mean, self.w_x_logvar, self.c_posterior , self.kumar_a, self.kumar_b, \
               self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample, self.gamma_alpha, self.gamma_beta

    def get_latent_states(self, X):
        self.forward(X)
        return self.z_x

    def compose_stick_segments( self,v):
        segments = []
        self.remaining_stick = [torch.ones((v.shape[0],1)).to(device=self.device)]
        for i in range(self.K-1):
            curr_v = torch.unsqueeze(v[:, i],1)
            segments.append(torch.mul(curr_v, self.remaining_stick[-1]))
            self.remaining_stick.append(torch.mul(1-curr_v, self.remaining_stick[-1]))
        segments.append(self.remaining_stick[-1])
        return segments

    def GenerateMixtures(self):
        """
        #KL divergence P(c|z,w)=Q(c|x) while P(c|pi) is the prior
        """
        global SMALL
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)
        # compose into stick segments using pi = v \prod (1-v)
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b)).to(device=self.device)
        #u       = (r1 - r2) * torch.rand(a_inv.shape[0],self.K-1) + r2
        u       = torch.distributions.uniform.Uniform(low=SMALL, high=1-SMALL).sample([1]).squeeze()
        v_samples  = torch.pow(1 - torch.pow(u, b_inv), a_inv).to(device=self.device)
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
                self.pi_samples[:, k] = v1[:, k] * torch.stack([(1 - v1[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)
        #pdb.set_trace()
        #print(f'size of pi {self.pi_samples.size()}')

        return  self.pi_samples



    def DiscreteKL(self, P,Q, epsilon=1e-8):

        #KL(q(z)||p(z)) =  - sum_k q(k) log p(k)/q(k)
   	    # let's p(k) = 1/K???
        logQ = torch.log(Q+ epsilon).to(device=self.device)
        
        logP = torch.log(P+ epsilon).to(device= self.device)
        element_wise = (Q * torch.sub(logQ , logP))
        #pdb.set_trace()
        return torch.sum(element_wise, dim=-1).mean().to(device=self.device)

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
        return torch.sum(torch.bmm(KLD_table, qc)).to(device=self.device)

    def EntropyCriterion(self):
        #CV = H(C|Z, W) = E_q(c,w) [ E_p(c|z,w)[ - log P(c|z,w)] ]
        
        z_sample =  self.z_x.unsqueeze(-1)
        z_sample =  z_sample.expand(-1, self.z_dim, self.K)
        z_sample =  z_sample.permute(2,0,1)
        
        """
         z [NxD]
         mean [NxD]
         logVar [NxD]
         llh = -0.5 sum_d { (x_i - mu_i)^2/var_i } - 1/2 sum_d (logVar_i) - D/2 ln(2pi) [N]
        """
        log_likelihoods = log_gaussian(z_sample, self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample)
        llh=log_likelihoods.sum(-1)

        lh = F.softmax(llh, dim=-1).to(device=self.device)
        # entropy
        # H(C|Z) = E_q(z) [ E_p(c|z)[ - log Q(c|z)] ]
        CV = torch.sum(torch.mul(torch.log(lh+epsilon()), lh)).to(device=self.device)
        return CV

    def get_ELBO(self, X):
        self.X           = X
        

        loss_dict = OrderedDict()
        #1) Computes the KL divergence between two categorical distributions
        PriorC  = self.GenerateMixtures()


        #this term KL divergence of two discrete distributions
        elbo1     = self.DiscreteKL(PriorC ,self.c_posterior)


        # compose elbo of Kumaraswamy-beta
        #######error occurs here afterward ##############
        elbo2 = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.include_elbo2:
            for k in range(self.K-1):
                #2)need this term
                #elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), \
                #                                   self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])
                #elbo2 -= compute_kumar2beta_kld(self.kumar_a[:, k].expand(self.batch_size), self.kumar_b[:, k].expand(self.batch_size), self.prior, (self.K-1-k)* self.prior).mean()
                #print(f"a kumaraswamy component {k}: {self.kumar_a[:, k]}")
                #print(f"b kumaraswamy component {k}: {self.kumar_b[:, k]}")
                elbo2 = elbo2 - compute_kumar2beta_kld(self.kumar_a[:, k], self.kumar_b[:, k], self.prior_nu[:,k], (self.K-1-k)* self.prior_nu[:,k]).mean()
        
        #3)need this term of w (context)
        #0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        elbo3     = -0.5 * torch.sum(1 + self.w_x_logvar - self.w_x_mean.pow(2) - self.w_x_logvar.exp()).to(device=self.device)#VAE_KLDCriterion
        
        # 4)compute E_{p(w|x)p(c|x)}[D_KL(Q(z|x)||p(z|c,w))]

        elbo4     = self.ExpectedKLDivergence(self.c_posterior, self.z_x_mean, self.z_x_logvar, self.z_wc_mean_list_sample, self.z_wc_logvar_list_sample)
        
        #5) compute Reconstruction Cost = E_{q(z|x)}[P(x|z)]
        #
        criterion = nn.BCELoss(reduction='sum')
        elbo5     = criterion(self.x_recons.view(-1, self.nchannel*self.img_size*self.img_size), self.X.view(-1, self.nchannel*self.img_size*self.img_size))
        
        assert torch.isfinite(elbo5)
        # 4)compute D_KL(Q(nu|x)||p(nu|alpha,beta)) --> gamma distribution
        elbo6     = gamma_kl_loss( self.gamma_alpha, self.gamma_beta, self.priors['alpha']*torch.ones_like(self.gamma_alpha, device=self.device, requires_grad=False), self.priors['beta']*torch.ones_like(self.gamma_beta, device=self.device, requires_grad=False)).mean()
        
        loss_dict['recon'] = elbo5
        loss_dict['c_cluster_kld'] = elbo1
        loss_dict['kumar2beta_kld'] = elbo2
        loss_dict['w_context_kld'] = elbo3
        loss_dict['z_latent_space_kld'] = elbo4
        loss_dict['gamma_params_kld'] = elbo6
        # 6.)  CV = H(C|Z, W) = E_q(z,w) [ E_p(c|z,w)[ - log P(c|z,w)] ]
        loss_dict['CV_entropy'] = self.EntropyCriterion()
        #print(f" device of CV Etropy: {loss_dict['CV_entropy'].get_device()}")
        #loss_dict['loss'] = elbo1 + elbo2 + elbo3 + elbo4 + elbo5
        #excluding the KL loss terms of latent dimension Z since we have a dicriminator to take care of it.  
        if self.include_elbo2:
            loss_dict['loss'] = elbo1 + elbo2 + elbo3  + elbo5 + elbo6
        else:
            loss_dict['loss'] = elbo1 + elbo3  + elbo5 + elbo6
        return loss_dict

    
    def get_log_margLL(self, batchSize):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples

        uni_samples =torch.FloatTensor(a_inv.shape[0], self.K-1).uniform_(epsilon(), 1-epsilon()).to(device=self.device)
        #Samples from a two-parameter Kumaraswamy distribution with a, b parameters. Or equivalently,
        # U ~ U(0,1)
        # X = (1 - (1 - U)^(1 / b))^(1 / a)
        #  based on https://arxiv.org/pdf/1905.12052.pdf
        #https://github.com/astirn/MV-Kumaraswamy/blob/master/mv_kumaraswamy_sampler.py
        v_samples   = torch.pow(1-torch.pow(1-uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index

        uni_samples = torch.FloatTensor(a_inv.shape[0], self.K).uniform_(epsilon(), 1-epsilon()).to(device=self.device)
        gumbel_samples = -torch.log(-torch.log(uni_samples +epsilon()) + epsilon())
        
        component_samples = torch.argmax(torch.log(torch.cat( self.pi_samples,1) + epsilon()) + gumbel_samples, 1).to(device=self.device, dtype=torch.int64)
        #print(f'index of stick should be used for the mixture model:\n{component_samples}')
        component_samples = torch.cat( [torch.arange(0, batchSize, device=self.device).unsqueeze(1), component_samples.unsqueeze(1)], 1)
        
        # calc likelihood term for chosen components
        ll = -compute_nll(self.X, self.x_recons)
        # calc prior terms
        all_log_gauss_priors = []

        for k in range(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z_x, self.z_wc_mean_list_sample[k], self.z_wc_logvar_list_sample[k]))

        
        all_log_gauss_priors = torch.stack(all_log_gauss_priors).to(device=self.device)
        #print(f"size of all components of Gaussian prior: {all_log_gauss_priors.size()}")
        #print(f"size of components of Z: {component_samples.size()}")
        log_gauss_prior = gather_nd(torch.t(all_log_gauss_priors), component_samples)
        log_gauss_prior = log_gauss_prior.unsqueeze(1)
        
        log_gauss_post = log_normal_pdf(self.z_x, self.z_x_mean, self.z_x_sigma)
        #****need this term :
        #log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        log_beta_prior = log_beta_pdf(v_samples[:,0].unsqueeze(1), self.prior_nu[:,0].unsqueeze(1), (self.K - 1) * self.prior_nu[:,0].unsqueeze(1))
        for k in range(self.K-2):
            #log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])
            log_beta_prior = log_beta_prior + log_beta_pdf(v_samples[:, k + 1].unsqueeze(1), self.prior_nu[:, k + 1].unsqueeze(1), (self.K - 2 - k) * self.prior_nu[:, k + 1].unsqueeze(1))

        # ****need this term :calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)
        #******compute likelihod terms of gamma distribution
        log_gamma_post  = self.log_gamma_pdf(self.prior_nu, self.gamma_alpha, self.gamma_beta)

        log_gamma_prior = self.log_gamma_pdf(self.prior_nu, self.priors['alpha']*torch.ones_like(self.gamma_alpha, device=self.device, requires_grad=False), self.priors['beta']*torch.ones_like(self.gamma_beta, device=self.device, requires_grad=False))
        #****need this term :cal prior and posterior over w
        log_w_prior = log_normal_pdf(self.w_x, torch.zeros(self.w_x.size()).to(device=self.device),torch.eye(self.w_x.size()[0],self.w_x.size()[1], device=self.device))
        log_w_post  = log_normal_pdf(self.w_x, self.w_x_mean, self.w_x_sigma)
        return ll + log_beta_prior + log_gauss_prior +log_w_prior +log_gamma_prior - log_kumar_post - log_gauss_post - log_w_post- log_gamma_post
    

    #TODO??
    @torch.no_grad()
    def get_component_samples(self, batchSize):
        # get the components of the latent space
        #a_inv = torch.pow(self.kumar_a,-1)
        #b_inv = torch.pow(self.kumar_b,-1)
        
        # ensure stick segments sum to 1
        #v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b)).to(device=self.device)
        #m = Beta(self.kumar_a, self.kumar_b)
        if torch.all(self.prior_nu.isfinite()):
            pass
        else:
            print(f" there is a NAN value in the prior over alpha variable")
        
        v_means =  self.beta_sample(torch.ones((batchSize,self.K-1), device=self.device, requires_grad=False), self.prior_nu )
        beta1m_cumprod  = (1 - v_means).cumprod(-1)
        self.pi_samples = torch.mul(F.pad(v_means, (0, 1), value=1) , F.pad(beta1m_cumprod, (1, 0), value=1))[:,:-1]
        
        # compose into stick segments using pi = v \prod (1-v)
        
        #p(z^{(i)}|c^{(i)}=j;theta)=N(z^{(i)}|mu_j,sigma_j)
        #p(c^{(i)}=j)=w_j
        #p(z^{(i)}|theta)=sum_{j=1}^{k} p(c^{(i)}=j) p(z^{(i)}|c^{(i)}=j;theta)
        # sample a component index       
        
        
        component = torch.argmax( self.pi_samples  , 1).to(device=self.device, dtype=torch.int64)
        #print(f'size of each latent component: {self.z_sample_list[0].size()}')

        component = torch.cat( [torch.arange(0, batchSize, device=self.device).unsqueeze(1), component.unsqueeze(1)], 1)
        #assert_almost_equal(torch.ones(n_samples,device=self.device).cpu().numpy(), component.sum(axis=1).detach().cpu().numpy(),decimal=4, err_msg='stick segments do not sum to 1')

        all_z = []
        for d in range(self.z_dim):
            temp_z = torch.cat( [self.z_sample_list[k][:, d].unsqueeze(1) for k in range(self.K)], dim=1)
            all_z.append(gather_nd(temp_z, component).unsqueeze(1))
        #print(f"size of pi samples before: {self.pi_samples.size()}")
        self.pi_samples         = self.pi_samples.unsqueeze(-1)
        self.pi_samples         = F.pad(input=self.pi_samples, pad=(0, 0, 1, 0), mode='constant', value=0)
        #print(f"size of pi samples after adding pad: {self.pi_samples.size()}")
        self.pi_samples         = self.pi_samples.expand(-1, self.K, 1)
        self.pi_samples         = self.pi_samples.permute(0,2,1)
        out      = torch.stack( all_z).to(device=self.device)
        out      = out.permute(1,0,2)
        #print(f"size of all z components after stacking: {out.size()}")
        #print(f"size of pi samples after: {self.pi_samples.size()}")
        
        
        self.concatenated_latent_space = torch.bmm(out, self.pi_samples)
        #print(f"size of latent space before reconstruction: {self.concatenated_latent_space.size()}")
        
        #multivariate Gaussian distribution
        #self.concatenated_latent_space = rearrange(self.concatenated_latent_space, 'd0 d1 d2 -> d1 (d0 d2)')
        #out = self.concatenated_latent_space.permute(1,0)
        return torch.squeeze(torch.mean(self.concatenated_latent_space, 2, True))
        

       
        

### Marginal Likelihood Calculation            
def calc_margLikelihood(data, model, nSamples=50):
    

    sample_collector = []
    for s_idx in range(nSamples):
        samples = model.get_log_margLL(model.batch_size)
        #print(f'batch size:{data.shape[0]}')
        if not np.isnan(samples.mean().detach().cpu().numpy()) and not np.isinf(samples.mean().detach().cpu().numpy()):
            sample_collector.append(samples.detach().cpu().numpy())
        
    if len(sample_collector) < 1:
        print("\tMARG LIKELIHOOD CALC: No valid samples were collected!")
        return np.nan

    all_samples = np.hstack(sample_collector)
    m = np.amax(all_samples, axis=1)
    mLL = m + np.log(np.mean(np.exp( all_samples - m[np.newaxis].T ), axis=1))
    return mLL.mean()

'''
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
'''

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

def train(epoch):
    global iters
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    train_loss_avg = []
    net.train()
    vae_encoder.train()
    vae_decoder.train()
    discriminator.train()

    train_loss = 0
    train_loss_avg.append(0)
    num_batches = 0

    start = end = time.time()
    for batch_idx, (X, classes) in enumerate(train_loader):
        #measure time
        data_time.update(time.time()-end)
        
        X = X.to(device=device, dtype=torch.float)
        #print(f'size of input image {X.size()}')
        #optimizer.zero_grad()
        for _ in range(hyperParams["CRITIC_ITERATIONS"]):
            X_recons_linear, mu_z, logvar_z, mu_w, logvae_w, qc, kumar_a, kumar_b, mu_pz, logvar_pz, gamma_alpha, gamma_beta = net(X)
            z_fake = net.encoder.reparameterize(mu_z, logvar_z)
            reconstruct_latent_components = net.get_component_samples(hyperParams["batch_size"])
            
            critic_real = discriminator(reconstruct_latent_components).reshape(-1)
            critic_fake = discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(discriminator, reconstruct_latent_components, z_fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + hyperParams["LAMBDA_GP"] * gp
            )
            discriminator.zero_grad()
            loss_critic.backward(retain_graph=True)
            dis_optim.step()
        gen_fake = discriminator(z_fake).reshape(-1)
        

        loss_dict = net.get_ELBO(X)
        loss_dict["wasserstein_loss"] =  -torch.mean(gen_fake)
        vae_encoder.zero_grad()
        vae_decoder.zero_grad()
        
        loss_dict["WAE-GP"]=loss_dict["loss"]+loss_dict["wasserstein_loss"]

        # ======== Train Discriminator ======== #
        #frozen_params(vae_encoder)
        #frozen_params(vae_decoder)
        #free_params(discriminator)
        #print(f" Entropy {loss_dict['CV_entropy'].is_cuda}")
        #print(f"device of elbo1 is cuda: {loss_dict['c_cluster_kld'].is_cuda}")
        #print(f"device of elbo2 is cuda: {loss_dict['kumar2beta_kld'].is_cuda}")
        #print(f"device of elbo3 is cuda: {loss_dict['w_context_kld'].is_cuda}")
        #print(f"device of elbo4 is cuda: {loss_dict['z_latent_space_kld'].is_cuda}")
        #print(f"device of elbo5 is cuda: {loss_dict['recon'].is_cuda}")
        #print(f"device of total loss is cuda: {loss_dict['loss'].is_cuda}")

        loss_dict["WAE-GP"].backward()
        train_loss += loss_dict['loss'].item()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        #optimizer.step()
        enc_optim.step()
        dec_optim.step()

        train_loss_avg[-1] += loss_dict["WAE-GP"].item()
        num_batches += 1
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:

            print('epoch {} --- iteration {}: '
                      ', Date = {:0.3f}s '
                      ', Time ={:0.3f}s '
                      ', Gamma KL = {:.6f} '
                      ', kumar2beta KL = {:.6f} '
                      ', Z latent space KL = {:.6f} '
                      ', reconstruction loss = {:.6f} '
                      ', W context latent space KL = {:.6f}'
                      ', Wasserstein gradient penalty = {:.6f}'.format(epoch, batch_idx, data_time.avg, batch_time.avg, loss_dict['gamma_params_kld'].item(), loss_dict['kumar2beta_kld'].item(), loss_dict['z_latent_space_kld'].item(), loss_dict['recon'].item(),loss_dict['w_context_kld'].item(),loss_dict["wasserstein_loss"].item()))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_dict["WAE-GP"].item() / len(X)))
        if ((epoch > 1) and (epoch % 50 == 0)) :
            #plot low dimensional embedding
            reconstruct_latent_components = net.get_component_samples(hyperParams["batch_size"])
            reconst_images_concatenate    = net.Px_given_z(reconstruct_latent_components.float())
         
            model_tsne = TSNE(n_components=2, random_state=0)
            z_states = net.z_x.detach().cpu().numpy()
            #print(f'size of the latent dimension {z_states.shape}')
            z_embed = model_tsne.fit_transform(z_states)
            #print(f'size of the embedding dimension {z_embed.shape}')
            classes = classes.detach().cpu().numpy()
            order_classes=set()
            ls =[x for x in classes if x not in order_classes and  order_classes.add(x) is None]
            matplotlib.use("Agg")
            fig = plt.figure()
            for ic in range(len(ls)):
                
                ind_class = classes == ic
                #print(f"index of classes {ind_class}")
                color = plt.cm.Set1(ic)
                plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, edgecolors=color, facecolors='none')
                plt.title("Latent Variable T-SNE per Class")
                fig.savefig(str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Embedding_" + str(ic) + "_epoch_"+str(epoch)+".png")
            fig.savefig(str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Embedding_epoch_"+str(epoch)+".png")
            # create original and reconstructed image-grids for
            d = X.shape[1]
            w = X.shape[2]
            h = X.shape[3]
            orig_img = X.view(len(X), d, w, h)
            #50 x 3 x 96 x 96
            recons_img = reconst_images_concatenate.view(len(X), d, w, h)

            canvas = torch.cat([orig_img, recons_img], 0)
            canvas = canvas.cpu().data
            # save_image(orig_img, f'{directory}orig_grid_sample_{b}.png')
            canvas_grid = make_grid(
                canvas, nrow=hyperParams["batch_size"], range=(-1, 1), normalize=True
            )
            save_image(canvas, str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Reconstruct_GridSample_Prior_Epoch_"+str(epoch)+".png")

        if (iters % 200 == 0) or ((epoch == num_epochs-1) and (batch_idx == len(train_loader)-1)):
            margll_valid = calc_margLikelihood(X, net) 

            print("\n\nValidation Marginal Likelihood: {:.4f}".format(margll_valid))
            img_list.append(torchvision.utils.make_grid(X_recons_linear[0].detach().cpu(), padding=2, normalize=True))
        iters += 1
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_loss_avg[-1] /= num_batches
    print(f' average training loss : {train_loss_avg}')
    print ("Calculating the marginal likelihood...")
    if torch.isfinite(X_recons_linear).all():
        pass
    else:
        print(f"There is either inf or NaN in the recounstructed image....")
    return train_loss_avg, X_recons_linear, loss_dict['kumar2beta_kld'].item(), loss_dict["wasserstein_loss"].item(), loss_dict['z_latent_space_kld'].item()

def test(epoch):
    
    net.eval()
    test_loss = 0
    test_loss_reconstruction = 0
    with torch.no_grad():
        for i, (image_batch, _) in enumerate(test_loader):
            image_batch = image_batch.to(device=device,dtype=torch.float)
            if i == 0 and epoch % 20 == 0:
                # VAE Reconstruction
                reco_img            = net.reconstruct_img(image_batch)
                
                image_numpy = image_batch[-1].cpu().float().numpy()
                
                ###
                reco_numpy  = reco_img[-1].cpu().float().numpy()
                ###
                print(image_numpy.shape)
                writer.add_image('Real images', image_numpy, epoch) 
                writer.add_image('Reconstructed images', reco_numpy, epoch) 
                
                """
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
                """
            # reconstruction error
            loss_dict = net.get_ELBO(image_batch)

            test_loss += loss_dict['loss'].item()
            test_loss_reconstruction += loss_dict['recon'].item()
            
        test_loss /= len(test_loader.dataset)
        test_loss_reconstruction /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set reconstuction loss: {:.4f}'.format(test_loss_reconstruction))
        
        writer.add_scalar('Test reconstruction loss', loss_dict['recon'].item(), epoch)
        writer.add_scalar( 'Test KL loss of Z', loss_dict['z_latent_space_kld'].item(), epoch)

def show_image(img, filename):
    img = img.clamp(0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)

def visualise_output(images, model,filename):

    with torch.no_grad():
        model.eval()
        images = images.to(device=device)

        out, _, _, _, _, _, _, _, _, _, _, _ = model(images)
        img =torchvision.utils.make_grid(out.detach().cpu())
        img = 0.5*(img + 1)
        npimg = np.transpose(img.numpy(),(1,2,0))
        fig = plt.figure(dpi=700)
        plt.imshow(npimg)
    plt.savefig(filename)

if __name__ == "__main__":

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(log_dir='scalar', comment='runs/STL10_experiment_1')
    ###**********************************************************###
    # Training settings
    #parser = argparse.ArgumentParser(description='PyTorch Stick Breaking Gaussian Mixture VAE....')
    #parser.add_argument('--visdom_server', default="http://localhost",help='visdom server of the web display')
    #parser.add_argument('--visdom_port', default=8097, help='visdom port of the web display')
    #args = parser.parse_args()
    #server.start_server()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])
    device = torch.device('cuda')

    hyperParams = {"batch_size": 100,
                "input_d": 1,
                "prior_alpha": 7., #gamma_alpha
                "prior_beta": 1., #gamma_beta
                "K": 25,
                "hidden_d": 500,
                "latent_d": 200,
                "latent_w": 150,
                "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
                "LEARNING_RATE": 1e-4,
                "CRITIC_ITERATIONS" : 5
                }


    # # Preparing Data : STL10
    print("Loading trainset...")
    train_dataset = datasets.STL10('./data', split='train', transform=transform,
                                target_transform=None, download=True)
    train_loader = DataLoader.DataLoader(train_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=0)
    print("Loading testset...")
    test_dataset = datasets.STL10('./data', split='test', transform=transform,
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


    net = InfGaussMMVAE(hyperParams, 25, 3, 200, 150, 500, device, raw_img_width, hyperParams["batch_size"],include_elbo2=True)

    params = list(net.parameters())
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net)
    net = net.to(device=device)
    for name, param in net.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))
    ###*********************************************************###

    #net.cuda()
    vae_encoder, vae_decoder, discriminator = net.encoder, net.decoder, VAECritic(net.z_dim)

    enc_optim = torch.optim.Adam(vae_encoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
    dec_optim = torch.optim.Adam(vae_decoder.parameters(), lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.5 * hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))

    enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size = 30, gamma = 0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size = 30, gamma = 0.5)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size = 30, gamma = 0.5)
    #optimizer = optim.Adam(net.parameters(), lr=hyperParams["LEARNING_RATE"])

    ##optimizer = AdaBound(net.parameters(), lr=0.0001)

    img_list = []
    num_epochs= 2001
    grad_clip = 1.0
    iters=0
    
    avg_train_loss=[]
    best_loss = np.finfo(np.float64).max # Random big number (bigger than the initial loss)
    best_epoch = -1
    regex = re.compile(r'\d+')
    start_epoch = 0
    for file in os.listdir("./results/"):
        if file.startswith("model_Hierarchical_StickBreaking_GMMVAE_") and file.endswith(".pth"):
            print(file)
            list_of_files = glob.glob(str(Path().absolute())+"/results/model_Hierarchical_StickBreaking_GMMVAE*.pth") # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            print("latest saved model: ")
            print(latest_file)
            checkpoint = torch.load(latest_file)
            net.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            enc_optim.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            dec_optim.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
            vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
            discriminator.load_state_dict(checkpoint['citic_state_dict'])
            best_loss   = checkpoint['best_loss']
            start_epoch = int(regex.findall(latest_file)[-1])
            print(f"start of epoch: {start_epoch}")


    ####Training

    #global plotter
    #plotter = VisdomLinePlotter(env_name='HIERARCHICAL_STICKBREAKING_GMMVAE_PLOTS')

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)
    for epoch in range(start_epoch, num_epochs):

            average_epoch_loss, out , elbo2, wasserstein_loss, latent_dimension_kld = train(epoch)
            if (epoch % 10 == 0):
                test(epoch)
            avg_train_loss.extend(average_epoch_loss)
            if (epoch % 50 == 0) and (epoch > 0):

                img = make_grid(out.detach().cpu())
                img = 0.5*(img + 1)
                npimg = np.transpose(img.numpy(),(1,2,0))
                fig = plt.figure(dpi=600)
                plt.imshow(npimg)
                plt.imsave(str(Path().absolute())+"/results/reconst_images_Hierarchical_StickBreaking_GMMVAE_"+str(epoch)+"_epochs.png", npimg)
            # plot beta-kumaraswamy loss
            #plotter.plot('KL of beta-kumaraswamy distributions', 'val', 'Class Loss', epoch, elbo2)

            # plot loss
            #print(average_epoch_loss)
            #print(epoch)
            #plotter.plot('Total loss', 'train', 'Class Loss', epoch, average_epoch_loss[0])
            vae_encoder.eval()
            vae_decoder.eval()
            discriminator.eval()
            if (epoch % 500 == 0) and (epoch//500> 0):
                torch.save({
                        'best_epoch': epoch,
                        'model_state_dict': copy.deepcopy(net.state_dict()),
                        'encoder_optimizer_state_dict':copy.deepcopy(enc_optim.state_dict()),
                        'decoder_optimizer_state_dict':copy.deepcopy(dec_optim.state_dict()),
                        'encoder_state_dict':copy.deepcopy(vae_encoder.state_dict()),
                        'decoder_state_dict':copy.deepcopy(vae_decoder.state_dict()),
                        'citic_state_dict':copy.deepcopy(discriminator.state_dict()),
                        #'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'best_loss': average_epoch_loss[-1],
                }, str(Path().absolute())+"/results/model_Hierarchical_StickBreaking_GMMVAE_"+str(epoch)+".pth")
                print("Saved Best Model avg. Loss : {:.4f}".format(average_epoch_loss[-1]))
            #scheduler.step(average_epoch_loss[-1])
            writer.add_scalar('training average loss', average_epoch_loss[-1], epoch )
            writer.add_scalar('training loss_kld_Kumar_beta', elbo2, epoch )
            writer.add_scalar('training loss_wasserstein', wasserstein_loss, epoch )
            writer.add_scalar('training loss_kld_z',latent_dimension_kld, epoch )

    writer.flush()



    print('Finished Trainning')


    #plot the loss values of trained model
    fig = plt.figure()
    plt.plot(avg_train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(str(Path().absolute())+"/results/Loss_Hierarchical_StickBreakingGMM_VAE.png")

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(make_grid(data[0].to(device=device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    ###plot original versus fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(str(Path().absolute())+"/results/real_vs_fake_images.jpg")

    #plotter.viz.save([plotter.env])###???
    
    images, labels = iter(test_loader).next()
    show_image(torchvision.utils.make_grid(images[1:50],10,5), str(Path().absolute())+"/results/Original_Images_Hierarchical_StickBreaking_GMM_VAE.png")
    visualise_output(images, net,str(Path().absolute())+"/results/Reconstructed_Images_Hierarchical_StickBreaking_VAE.png")