import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, Independent, Uniform
from copy import deepcopy
from torch.autograd import Variable
#tf.reduce_mean -> tensor.mean
#tf.expand_dims -> tensor.expand
#tf.transpose -> tensor.permute

'''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
'''
local_device = "cpu"

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
        params['w'].append(nn.Parameter(torch.tensor(Normal(torch.zeros(n_in, n_out), std * torch.ones(n_in, n_out)).sample(), requires_grad=True, device=device)))
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
            Pz_given_wc_var  = F.relu(self.pz_wc_bn_var[i](self.pz_wc_fc_var[i](h)))
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


        self.z_x_var     = F.relu(self.bn1d_2(self.fc2(h)))
        self.z_x_logvar  = torch.log(self.z_x_var)

        eps1             = Normal(loc=torch.zeros(self.z_x_mean.shape,), scale=torch.ones(self.z_x_logvar.shape,)).sample().to(self.device)
        self.z_x         = self.z_x_mean+torch.mul(torch.sqrt(self.z_x_var), eps1)


        #Build a two layers MLP to compute Q(w|x)
        hw = self.flatten_raw_img(X)
        hw = F.relu(self.bn1d_3(self.fc3(hw)))
        #mean of Q(w|x) distribution

        self.w_x_mean    = F.relu(self.bn1d_4(self.fc4(hw)))
        #variance of Q(w|x) distribution

        self.w_x_var     = F.relu(self.bn1d_5(self.fc5(hw)))
        self.w_x_logvar  = torch.log(self.w_x_var)

        eps2             = Normal(loc=torch.zeros(self.w_x_mean.shape,), scale=torch.ones(self.w_x_logvar.shape,)).sample().to(self.device)
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
        eps = Normal(loc=torch.zeros(self.x_recons_mean_flat.shape,), scale=torch.ones(self.x_recons_mean_flat.shape,)).sample().to(self.device)

        self.x_recons_flat = torch.min(torch.max(torch.add(self.x_recons_mean_flat, np.sqrt(self.sigma) * eps), torch.tensor(0, dtype=torch.float).to(self.device)), torch.tensor(1, dtype=torch.float).to(self.device))
        self.x_recons = torch.reshape(self.x_recons_flat , [-1,self.img_width, self.img_width, self.nchannel])


        #P(w)
        self.w_sample = Normal(loc=torch.zeros((self.batch_size, self.w_dim)), scale=torch.ones((self.batch_size, self.w_dim))).sample().to(self.device)

        #P(z_i|w,c_i)
        self.z_wc_mean_list_sample, self.z_wc_var_list_sample = self.Pz_given_wc(self.w_sample,self.hidden_dim)
        #self.z_wc_mean_list_sample = self.z_wc_mean_list_sample.reshape(self.z_wc_mean_list_sample.shape[1], self.z_wc_mean_list_sample.shape[0], self.z_wc_mean_list_sample.shape[2])
        #self.z_wc_var_list_sample = self.z_wc_var_list_sample.reshape(self.z_wc_var_list_sample.shape[1], self.z_wc_var_list_sample.shape[0], self.z_wc_var_list_sample.shape[2])
        self.z_sample_list = list()
        for i in range(self.K):
            eps = Normal(loc=torch.zeros_like(self.z_wc_mean_list_sample[i]), scale=torch.ones_like(self.z_wc_mean_list_sample[i])).sample().to(self.device)
            z_sample = torch.add(self.z_wc_mean_list_sample[i], torch.mul(torch.sqrt(self.z_wc_var_list_sample[i]), eps))
            self.z_sample_list.append(z_sample)
        #P(x|z)
        self.x_sample_mean_flat_list = list()
        self.x_sample_flat_list = list()
        self.x_sample_list = list()
        for i in range(self.K):
            x_sample_mean_flat = self.Px_given_z(self.z_sample_list[i])
            self.x_sample_mean_flat_list.append(x_sample_mean_flat)

            eps = Normal(loc=torch.zeros(x_sample_mean_flat.shape), scale=torch.ones(x_sample_mean_flat.shape)).sample().to(self.device)

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
    d = (mu_post - mu_prior)
    d = torch.mul(d,d)
    return torch.sum(-torch.div(d + torch.mul(sigma_post,sigma_post),(2.*sigma_prior*sigma_prior)) - torch.log(sigma_prior*2.506628), dim=1, keepdim=True)


def beta_fn(a,b):
    global local_device
    return torch.exp(torch.lgamma(torch.tensor(a, dtype=torch.float).to(device=local_device)) + torch.lgamma(torch.tensor(b, dtype=torch.float).to(device=local_device)) - torch.lgamma(torch.tensor(a+b, dtype=torch.float).to(device=local_device)))


def compute_kumar2beta_kld(a, b, alpha, beta):
    # precompute some terms
    ab    = torch.mul(a,b)
    a_inv = torch.pow(a, -1)
    b_inv = torch.pow(b, -1)

    # compute taylor expansion for E[log (1-v)] term
    kl = torch.mul(torch.pow(1+ab,-1), beta_fn(a_inv, b))
    
    for idx in range(10):
        kl += torch.mul(torch.pow(idx+2+ab,-1), beta_fn(torch.mul(idx+2., a_inv), b))
        
    kl = torch.mul(torch.mul(beta-1,b), kl)

    kl += torch.mul(torch.div(a-alpha,a), -0.57721 - torch.digamma(b) - b_inv)
    # add normalization constants
    kl += torch.log(ab) + torch.log(beta_fn(alpha, beta))
    
    # final term
    kl += torch.div(-(b-1),b)

    return kl


def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = torch.mul(-1., torch.mul(d, d))
    s2 = torch.mul(2., torch.mul(sigma,sigma))
    return torch.sum(torch.div(d2, s2) - torch.log(torch.mul(sigma, 2.506628)), dim=1, keepdim=True)


def log_beta_pdf(v, alpha, beta):
    return torch.sum((alpha - 1) * torch.log(v) + (beta-1) * torch.log(1 - v) - torch.log(beta_fn(alpha, beta)), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    return torch.sum(torch.mul(a - 1, torch.log(v)) + torch.mul(b - 1, torch.log(1 - torch.pow(v,a))) + torch.log(a) + torch.log(b), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in range(K-1):
        s += torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s)




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
    def __init__(self, hyperParams, K, nchannel, base_channels, z_dim, w_dim, hidden_dim, device, img_width, batch_size):
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
        self.z       = self.z_mu + torch.mul(self.z_sigma, Normal(0., 1.).sample(self.z_sigma.shape).to(self.device))
        self.w_mu    = self.w_x_mean # TODO Fix this
        self.w_sigma = self.w_x_var # TODO: Fix this
        self.w       = self.w_mu + torch.mul(self.w_sigma, Normal(0., 1.).sample(self.w_sigma.shape).to(self.device))

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
        uni_samples = torch.rand(v_means.shape).to(self.device)
        v_samples   = torch.pow(1 - torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means = torch.stack(self.compose_stick_segments(v_means), axis=2)
        self.pi_samples = torch.stack(self.compose_stick_segments(v_samples), axis=2)
        #KL divergence P(c|z,w)=Q(c|x) while P(c|pi) is the prior

        prior_c        = gumbel_softmax_sample(torch.log(self.pi_samples), 1, eps=1e-20)
        return self.x_recons_flat, self.z_mu, torch.log(self.z_sigma), prior_c

    def decode(self, z):
        return self._decoding(z)[0]

    def get_ELBO(self, X):
        _, _, _, pc  = self._decoding(self.z)
        log_ratio = torch.sub(torch.log(self.pc_wz+ 1e-20),torch.log(pc+ 1e-20))
        #1)need this term
        elbo      = torch.sum(self.pc_wz * log_ratio, dim=-1).mean()

        # compose elbo
        for k in range(self.K-1):
            #2)need this term
            #elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), \
            #                                   self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])
            #print(self.kumar_b[:, k].unsqueeze(1))
            #print(self.kumar_b[:, k].expand(self.batch_size))
            elbo -= compute_kumar2beta_kld(self.kumar_a[:, k].expand(self.batch_size), self.kumar_b[:, k].expand(self.batch_size), self.prior, (self.K-1-k)* self.prior).mean()
        #elbo += mcMixtureEntropy(self.pi_samples, self.z, self.z_mu, self.z_sigma, self.K)
        #3)need this term
        elbo += -0.5 * torch.sum(1 + torch.log(self.w_sigma + 1e-10) - self.w_mu*self.w_mu - self.w_sigma) #KLD_W
        #compute D_KL(Q(z|x)||p(z|c,w))
        #use this term https://github.com/psanch21/VAE-GMVAE/blob/e176d24d0e743f109ce37834f71f2f9067aae9bc/Alg_GMVAE/GMVAE_graph.py#L278
        # KL loss
        z_mean, z_logstd, _, _, _, _, _ = self._encoding(X)
        #kl_loss = 0.5*torch.sum(1 + z_logstd - z_mean**2 - torch.exp(z_logstd), dim=1)
        # likelihood loss

        logq = -0.5 * torch.sum(self.z_x_logvar, 1) - 0.5 * torch.sum(
                torch.pow(self.z_x - self.z_x_mean, 2) / self.z_x_var, 1)
        z_wy=torch.unsqueeze(self.z_x, 2)


        #compute E_{q(z|x)}[P(x|x)] reconstruction loss
        #use this term https://github.com/psanch21/VAE-GMVAE/blob/e176d24d0e743f109ce37834f71f2f9067aae9bc/Alg_GMVAE/GMVAE_graph.py#L256
        elbo += F.binary_cross_entropy(input=self.x_recons_flat.view(-1, self.img_size*self.img_size), target=X.view(-1, self.img_size*self.img_size), reduction='sum')
        #elbo += F.binary_cross_entropy(input=self.x_recons_flat, target=X, reduction='sum')
        return elbo.mean()


    def get_log_margLL(self, batchSize, X_recons_linear):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples
        uni_samples = torch.rand((a_inv.shape[0], self.K-1))
        v_samples   = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index
        uni_samples = torch.rand((a_inv.shape[0], self.K))
        gumbel_samples = -torch.log(-torch.log(uni_samples))
        component_samples = torch.IntTensor(torch.argmax(torch.log(torch.cat( self.pi_samples,1)) + gumbel_samples, 1))

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
