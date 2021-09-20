import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, Categorical, Independent
from copy import
from torch.autograd import Variable
#tf.reduce_mean -> tensor.mean
#tf.expand_dims -> tensor.expand
#tf.transpose -> tensor.permute

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    #source https://github.com/dd-iuonac/object-detector-in-carla/blob/fb900f7a1dcd366e326d044fcd8dc2c6ddc697fb/pointpillars/torchplus/ops/array_ops.py
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)


def init_mlp(layer_sizes, std=.01, bias_init=0.):
    params = {'w':[], 'b':[]}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(torch.tensor(Normal([n_in, n_out], torch.tensor([std])) ,requires_grad=True))
        params['b'].append(torch.tensor(torch.mul(bias_init, torch.ones([n_out,])),requires_grad=True))
    return params


def mlp(X, params):
    h = [X]
    for w,b in zip(params['w'][:-1], params['b'][:-1]):
        h.append( torch.nn.ReLU( torch.matmul(h[-1], w) + b ) )
    return torch.matmul(h[-1], params['w'][-1]) + params['b'][-1]

def compute_nll(x, x_recon_linear):
    #return torch.sum(func.binary_cross_entropy_with_logits(x_recon_linear, x), dim=1, keepdim=True)
    return func.binary_cross_entropy_with_logits(x_recon_linear, x)

def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    d = (mu_post - mu_prior)
    d = torch.mul(d,d)
    return torch.sum(-torch.div(d + torch.mul(sigma_post,sigma_post),(2.*sigma_prior*sigma_prior)) - torch.log(sigma_prior*2.506628), dim=1, keepdim=True)


def beta_fn(a,b):
    return torch.exp( torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b) )


def compute_kumar2beta_kld(a, b, alpha, beta):
    # precompute some terms
    ab    = torch.mul(a,b)
    a_inv = torch.pow(a, -1)
    b_inv = torch.pow(b, -1)

    # compute taylor expansion for E[log (1-v)] term
    kl = torch.mul(torch.pow(1+ab,-1), beta_fn(a_inv, b))
    for idx in xrange(10):
        kl += torch.mul(torch.pow(idx+2+ab,-1), beta_fn(torch.mul(idx+2., a_inv), b))
    kl = torch.mul(torch.mul(beta-1,b), kl)

    kl += torch.mul(torch.div(a-alpha,a), -0.57721 - torch.polygamma(b) - b_inv)
    # add normalization constants
    kl += torch.log(ab) + torch.log(beta_fn(alpha, beta))

    # final term
    kl += torch.div(-(b-1),b)

    return kl


def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = torch.mul(-1., torch.mul(d,d))
    s2 = torch.mul(2., torch.mul(sigma,sigma))
    return torch.sum(torch.div(d2,s2) - torch.log(torch.mul(sigma, 2.506628)), dim=1, keepdim=True)


def log_beta_pdf(v, alpha, beta):
    return torch.sum((alpha-1)*torch.log(v) + (beta-1)*torch.log(1-v) - torch.log(beta_fn(alpha,beta)), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    return torch.sum(torch.mul(a-1, torch.log(v)) + torch.mul(b-1, torch.log(1-torch.pow(v,a))) + torch.log(a) + torch.log(b), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in xrange(K-1):
        s += torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s)


### Gaussian Mixture Model VAE Class
class GaussMMVAE(nn.Module):
    def __init__(self, hyperParams):
        super().__init__()
        self.X = Variable(torch.FloatTensor(hyperParams['batch_size'], hyperParams['input_d']))
        #self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()

   def __len__(self):
        return len(self.X)

    def init_encoder(self, hyperParams):
        return {'base':nn.Sequential(
                                     nn.Conv2d([hyperParams['input_d'], 32, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(num_feature=32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(num_feature=64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(num_feature=128),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 256, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(num_feature=256),
                                     nn.ReLU(),
                                     Flatten(),
                                     nn.Linear(256*256, hyperParams['hidden_d'],bias=False),
                                     nn.BatchNorm2d(num_feature=hyperParams['hidden_d']),
                                     nn.ReLU()
                                     ),
                'z_mu':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'z_sigma':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'w_mu':init_mlp([hyperParams['hidden_d'], hyperParams['latent_w']]),
                'w_sigma':init_mlp([hyperParams['hidden_d'], hyperParams['latent_w']]),
                'kumar_a':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8),
                'kumar_b':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8)}


    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def f_prop(self):

        # init variational params
        self.z_mu = []
        self.z_sigma = []
        self.kumar_a = []
        self.kumar_b = []
        self.z = []
        x_recon_linear = []

        h1 =  self.encoder_params['base'](self.X)

        for k in xrange(self.K):
            self.z_mu.append(mlp(h1, self.encoder_params['z_mu'][k]))
            self.z_sigma.append(torch.exp(mlp(h1, self.encoder_params['z_sigma'][k])))
            self.z.append(self.z_mu[-1] + torch.mul(self.z_sigma[-1], Normal(self.z_sigma[-1].shape)))##??
            x_recon_linear.append(mlp(self.z[-1], self.decoder_params))

        self.w_mu    = mlp(h1, self.encoder_params['w_mu'])
        self.w_sigma = torch.exp(mlp(h1, self.encoder_params['w_sigma']))
        self.w       = self.w_mu[-1] + torch.mul(self.z_sigma[-1], Normal(self.z_sigma[-1].shape))
        h2           = torch.cat((self.z, self.w), 1)
        self.kumar_a = torch.exp(mlp(h2, self.encoder_params['kumar_a']))
        self.kumar_b = torch.exp(mlp(h2, self.encoder_params['kumar_b']))

        return x_recon_linear


    def compose_stick_segments(self, v):

        segments = []
        self.remaining_stick = [torch.ones((v.shape[0],1))]
        for i in xrange(self.K-1):
            curr_v = tensor.expand(v[:,i],1)
            segments.append( torch.mul(curr_v, self.remaining_stick[-1]) )
            self.remaining_stick.append( torch.mul(1-curr_v, self.remaining_stick[-1]) )
        segments.append(self.remaining_stick[-1])

        return segments


    def get_ELBO(self):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy means
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b))

        # compute Kumaraswamy samples
        uni_samples = torch.rand(v_means.shape)
        v_samples = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means = self.compose_stick_segments(v_means)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # compose elbo
        elbo = torch.mul(self.pi_means[0], -compute_nll(self.X, self.x_recons_linear[0]) + gauss_cross_entropy(self.z_mu[0], self.z_sigma[0], self.prior['mu'][0], self.prior['sigma'][0]))
        for k in xrange(self.K-1):
            elbo += torch.mul(self.pi_means[k+1], -compute_nll(self.X, self.x_recons_linear[k+1]) \
                               + gauss_cross_entropy(self.z_mu[k+1], self.z_sigma[k+1], self.prior['mu'][k+1], self.prior['sigma'][k+1]))
            elbo -= compute_kumar2beta_kld(tensor.expand(self.kumar_a[:,k],1), tensor.expand(self.kumar_b[:,k],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples, self.z, self.z_mu, self.z_sigma, self.K)
        elbo += -0.5 * torch.sum(1 + torch.log(self.w_sigma) - self.w_mu*self.w_mu - self.w_sigma) #KLD_W


        return tensor.mean(elbo)


    def get_log_margLL(self, batchSize):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples
        uni_samples = torch.rand((a_inv.shape[0], self.K-1))
        v_samples = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index
        uni_samples = torch.rand((a_inv.shape[0], self.K))
        gumbel_samples = -torch.log(-torch.log(uni_samples))
        component_samples = torch.IntTensor(torch.argmax(torch.log(torch.cat( self.pi_samples,1)) + gumbel_samples, 1))

        # calc likelihood term for chosen components
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = torch.cat( all_ll,1)

        component_samples = torch.cat( [tensor.expand(torch.range(0,batchSize),1), tensor.expand(component_samples,1)],1)
        ll = gather_nd(all_ll, component_samples)
        ll = tensor.expand(ll,1)

        # calc prior terms
        all_log_gauss_priors = []

        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = torch.cat(1, all_log_gauss_priors)
        log_gauss_prior = gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = tensor.expand(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tensor.expand(v_samples[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tensor.expand(v_samples[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])

        # calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)

        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.z_mu[k], self.z_sigma[k]))
        all_log_gauss_posts = torch.cat(all_log_gauss_posts,1)
        log_gauss_post = gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tensor.expand(log_gauss_post,1)
        #modified and added by Zahra
        log_w_prior = log_normal_pdf(self.w, torch.zeros(self.w.size()),torch.eye(self.w.size()))
        log_w_post  = log_normal_pdf(self.w, self.w_mu,self.w_sigma)
        return ll + log_beta_prior + log_gauss_prior +log_w_prior - log_kumar_post - log_gauss_post - log_w_post


    def get_samples(self, nImages):
        samples_from_each_component = []
        for k in xrange(self.K):
            z = self.prior['mu'][k] + torch.mul(self.prior['sigma'][k], Normal((nImages, self.decoder_params['w'][0].shape[0])))
            samples_from_each_component.append( torch.nn.Sigmoid(mlp(z, self.decoder_params)) )
        return samples_from_each_component

    def get_samples_prior_w(self,nImages):
        #sample from W parameter (Zahra)??
        return

    def get_component_samples(self, latent_dim, batchSize):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compose into stick segments using pi = v \prod (1-v)
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b))
        components = torch.IntTensor(torch.argmax(torch.cat(1, self.compose_stick_segments(v_means)), 1))
        components = torch.cat([tensor.expand(torch.range(0,batchSize),1), tensor.expand(components,1)],1)

        # sample a z
        all_z = []
        for d in xrange(latent_dim):
            temp_z = torch.cat(1, [tensor.expand(self.z[k][:, d],1) for k in xrange(self.K)])
            all_z.append(tensor.expand(gather_nd(temp_z, components),1))

        return torch.cat( all_z,1)



class DPVAE(GaussMMVAE):
    def __init__(self, hyperParams):

        self.X = Variable(torch.FloatTensor(hyperParams['batch_size'], hyperParams['input_d']))
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop() #define q(z|x)???

        self.elbo_obj = self.get_ELBO()



    def get_ELBO(self):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy means
        v_means = torch.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b))

        # compute Kumaraswamy samples
        uni_samples = torch.rand(v_means.shape)
        v_samples = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means = self.compose_stick_segments(v_means)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # compose elbo
        elbo = torch.mul(self.pi_means[0], -compute_nll(self.X, self.x_recons_linear[0]) + gauss_cross_entropy(self.z_mu[0], self.z_sigma[0], self.prior['mu'][0], self.prior['sigma'][0]))
        for k in xrange(self.K-1):
            elbo += torch.mul(self.pi_means[k+1], -compute_nll(self.X, self.x_recons_linear[k+1]) \
                               + gauss_cross_entropy(self.z_mu[k+1], self.z_sigma[k+1], self.prior['mu'][k+1], self.prior['sigma'][k+1]))
            elbo -= compute_kumar2beta_kld(tensor.expand(self.kumar_a[:,k],1), tensor.expand(self.kumar_b[:,k],1), 1., self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples, self.z, self.z_mu, self.z_sigma, self.K)
        elbo += -0.5 * torch.sum(1 + torch.log(self.w_sigma) - self.w_mu*self.w_mu - self.w_sigma)
        return tensor.mean(elbo)


    def get_log_margLL(self, batchSize):
        a_inv = torch.pow(self.kumar_a,-1)
        b_inv = torch.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples
        uni_samples = torch.rand((a_inv.shape[0], self.K-1))
        v_samples = torch.pow(1-torch.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index
        uni_samples = torch.rand((a_inv.shape[0], self.K))
        gumbel_samples = -torch.log(-torch.log(uni_samples))
        component_samples = torch.IntTensor(torch.argmax(torch.log(torch.cat(1, self.pi_samples)) + gumbel_samples, 1))

        # calc likelihood term for chosen components
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = torch.cat( all_ll, 1)

        component_samples = torch.cat( [tensor.expand(torch.range(0,batchSize),1), tensor.expand(component_samples,1)],1)
        ll = gather_nd(all_ll, component_samples)
        ll = tensor.expand(ll,1)

        # calc prior terms
        all_log_gauss_priors = []
        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = torch.cat( all_log_gauss_priors,1)
        log_gauss_prior = gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = tensor.expand(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tensor.expand(v_samples[:,0],1), 1., self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tensor.expand(v_samples[:,k+1],1), 1., self.prior['dirichlet_alpha'])

        # calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)

        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.z_mu[k], self.z_sigma[k]))
        all_log_gauss_posts = torch.cat(1, all_log_gauss_posts)
        log_gauss_post = gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tensor.expand(log_gauss_post,1)

        return ll + log_beta_prior + log_gauss_prior - log_kumar_post - log_gauss_post


### *DEEP* Latent Gaussian MM
class DLGMM(GaussMMVAE):
    def __init__(self, hyperParams):

        self.X = Variable(torch.FloatTensor(hyperParams['batch_size'], hyperParams['input_d']))
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        t_hyperParams = deepcopy(hyperParams)
        t_hyperParams['input_d'] = t_hyperParams['hidden_d']
        #t_hyperParams['hidden_d'] /= 2
        t_hyperParams['latent_d'] /= 2
        self.encoder_params1 = self.init_encoder(hyperParams)
        self.encoder_params2 = self.init_encoder(t_hyperParams)
        t_hyperParams['input_d'] = None
        self.decoder_params2 = self.init_decoder(t_hyperParams)
        self.decoder_params1 = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()

        #self.batch_log_margLL = self.get_log_margLL(hyperParams['batchSize'])


    def init_encoder(self, hyperParams):
        return {'base':init_mlp([hyperParams['input_d'], hyperParams['hidden_d']], 0.00001),
                'mu':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']], 0.00001) for k in xrange(self.K)],
                'sigma':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']], 0.00001) for k in xrange(self.K)],
                'kumar_a':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-5),
                'kumar_b':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-5)}


    def init_decoder(self, hyperParams):
        if hyperParams['input_d']:
            return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['input_d']], 0.00001)
        else:
            return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d']], 0.00001)


    def f_prop(self):
        # init variational params
        self.z_mu1 = []
        self.z_mu2 =[]
        self.z_sigma1 = []
        self.z_sigma2 = []
        self.z1 = []
        self.z2 = []
        self.kumar_a1 = []
        self.kumar_b1 = []
        x_recon_linear = []

        h1 = mlp(self.X, self.encoder_params1['base'])

        # compute z1's sigma params
        for k in xrange(self.K):
            self.z_sigma1.append(torch.exp(mlp(h1, self.encoder_params1['sigma'][k])))

        h2 = mlp(h1, self.encoder_params2['base'])

        # compute z2's param
        for k in xrange(self.K):
            self.z_mu2.append(mlp(h2, self.encoder_params2['mu'][k]))
            self.z_sigma2.append(torch.exp(mlp(h2, self.encoder_params2['sigma'][k])))
            self.z2.append(self.z_mu2[-1] + torch.mul(self.z_sigma2[-1], Normal(self.z_sigma2[-1].shape)))
        self.kumar_a2 = torch.nn.Softplus(mlp(h2, self.encoder_params2['kumar_a']))
        self.kumar_b2 = torch.nn.Softplus(mlp(h2, self.encoder_params2['kumar_b']))

        h3 = []
        for k in xrange(self.K):
            h3.append(mlp(self.z2[k], self.decoder_params2))

        # compute z1's, finally.  KxK of them
        for k in xrange(self.K):
            self.z1.append([])
            self.z_mu1.append([])
            self.kumar_a1.append(torch.nn.Softplus(mlp(h3[k], self.encoder_params1['kumar_a'])))
            self.kumar_b1.append(torch.nn.Softplus(mlp(h3[k], self.encoder_params1['kumar_b'])))
            for j in xrange(self.K):
                self.z_mu1[-1].append(mlp(h3[k], self.encoder_params1['mu'][j]))
                self.z1[-1].append(self.z_mu1[-1][-1] + torch.mul(self.z_sigma1[k], Normal(self.z_sigma1[k].shape)))

        # compute KxK reconstructions
        for k in xrange(self.K):
            x_recon_linear.append([])
            for j in xrange(self.K):
                x_recon_linear[-1].append(mlp(self.z1[k][j], self.decoder_params1))

        # clip kumar params
        self.kumar_a1 = [torch.maximum(torch.minimum(a, 18.), .1) for a in self.kumar_a1]
        self.kumar_b1 = [torch.maximum(torch.minimum(b, 18.), .1) for b in self.kumar_b1]
        self.kumar_a2 = torch.maximum(torch.minimum(self.kumar_a2, 18.), .1)
        self.kumar_b2 = torch.maximum(torch.minimum(self.kumar_b2, 18.), .1)

        return x_recon_linear


    def get_ELBO(self):
        a1_inv = [torch.pow(a,-1) for a in self.kumar_a1]
        a2_inv = torch.pow(self.kumar_a2,-1)
        b1_inv = [torch.pow(b,-1) for b in self.kumar_b1]
        b2_inv = torch.pow(self.kumar_b2,-1)

        # compute Kumaraswamy means
        v_means1 = [torch.mul(self.kumar_b1[k], beta_fn(1.+a1_inv[k], self.kumar_b1[k])) for k in xrange(self.K)]
        v_means2 = torch.mul(self.kumar_b2, beta_fn(1.+a2_inv, self.kumar_b2))

        # compute Kumaraswamy samples
        uni_samples1 = [torch.rand(v_means1[k].shape, minval=1e-3, maxval=1-1e-3) for k in xrange(self.K)]
        uni_samples2 = torch.rand(v_means2.shape)
        v_samples1 = [torch.pow(1-torch.pow(uni_samples1[k], b1_inv[k]), a1_inv[k]) for k in xrange(self.K)]
        v_samples2 = torch.pow(1-torch.pow(uni_samples2, b2_inv), a2_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means1 = [self.compose_stick_segments(v_means1[k]) for k in xrange(self.K)]
        self.pi_samples1 = [self.compose_stick_segments(v_samples1[k]) for k in xrange(self.K)]
        self.pi_means2 = self.compose_stick_segments(v_means2)
        self.pi_samples2 = self.compose_stick_segments(v_samples2)

        # compose elbo
        elbo = torch.zeros((a2_inv.shape[0],1))
        for k in xrange(self.K):
            elbo += torch.mul(self.pi_means2[k], gauss_cross_entropy(self.z_mu2[k], self.z_sigma2[k], self.prior['mu'][k], self.prior['sigma'][k]))
            for j in xrange(self.K):
                elbo += torch.mul(torch.mul(self.pi_means2[k], self.pi_means1[k][j]), -compute_nll(self.X, self.x_recons_linear[k][j]))
                elbo += torch.mul(torch.mul(self.pi_means2[k], self.pi_means1[k][j]), log_normal_pdf(self.z1[k][j], self.prior['mu'][k], self.prior['sigma'][k]))


        kl2 = torch.zeros((a2_inv.shape[0],1))
        for k in xrange(self.K-1):
            kl2 -= compute_kumar2beta_kld(tensor.expand(self.kumar_a2[:,k],1), tensor.expand(self.kumar_b2[:,k],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])

        kl1 = torch.zeros((a2_inv.shape[0],1))
        for k in xrange(self.K):
            for j in xrange(self.K-1):
                kl1 -= torch.mul(self.pi_means2[k], compute_kumar2beta_kld(tensor.expand(self.kumar_a1[k][:,j],1), tensor.expand(self.kumar_b1[k][:,j],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-j)*self.prior['dirichlet_alpha']))

        elbo += kl1
        elbo += kl2


        elbo += mcMixtureEntropy(self.pi_samples2, self.z2, self.z_mu2, self.z_sigma2, self.K)
        for k in xrange(self.K):
            elbo += torch.mul(self.pi_means2[k], mcMixtureEntropy(self.pi_samples1[k], self.z1[k], self.z_mu1[k], self.z_sigma1, self.K))

        return tensor.mean(elbo)


    def get_log_margLL(self, batchSize):
        a1_inv = [torch.pow(a,-1) for a in self.kumar_a1]
        a2_inv = torch.pow(self.kumar_a2,-1)
        b1_inv = [torch.pow(b,-1) for b in self.kumar_b1]
        b2_inv = torch.pow(self.kumar_b2,-1)

        # compute Kumaraswamy samples
        uni_samples1 = [torch.rand((a2_inv.shape[0], self.K-1), minval=1e-3, maxval=1-1e-3) for k in xrange(self.K)]
        uni_samples2 = torch.rand((a2_inv.shape[0], self.K-1))
        v_samples1 = [torch.pow(1-torch.pow(uni_samples1[k], b1_inv[k]), a1_inv[k]) for k in xrange(self.K)]
        v_samples2 = torch.pow(1-torch.pow(uni_samples2, b2_inv), a2_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_samples1 = [self.compose_stick_segments(v_samples1[k]) for k in xrange(self.K)]
        self.pi_samples2 = self.compose_stick_segments(v_samples2)

        # sample a component index, from KxK
        uni_samples1 = torch.rand((a2_inv.shape[0], self.K*self.K))
        uni_samples2 = torch.rand((a2_inv.shape[0], self.K))
        gumbel_samples1 = -torch.log(-torch.log(uni_samples1))
        gumbel_samples2 = -torch.log(-torch.log(uni_samples2))
        log_prod_weights = []
        for k in xrange(self.K):
            log_prod_weights.append( torch.log(self.pi_samples2[k]) + torch.log(torch.cat( self.pi_samples1[k]),1) )

        log_prod_weights = torch.cat( log_prod_weights, 1)
        component_samples1 = torch.IntTensor(torch.argmax(log_prod_weights + gumbel_samples1, 1))
        component_samples1 = torch.cat( [tensor.expand(torch.range(0,batchSize),1), tensor.expand(component_samples1,1)],1)
        component_samples2 = torch.IntTensor(torch.argmax(torch.log(torch.cat( self.pi_samples2,1)) + gumbel_samples2, 1))
        component_samples2 = torch.cat( [tensor.expand(torch.range(0,batchSize),1), tensor.expand(component_samples2,1)],1)

        # calc likelihood term for chosen components
        all_ll = []
        for k in xrange(self.K):
            for j in xrange(self.K):
                all_ll.append(-compute_nll(self.X, self.x_recons_linear[k][j]))
        all_ll = torch.cat(1, all_ll)

        ll = gather_nd(all_ll, component_samples1)
        ll = tensor.expand(ll,1)

        ### TOP LEVEL

        # calc prior terms
        all_log_gauss_priors = []
        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z2[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = torch.cat( all_log_gauss_priors,1)
        log_gauss_prior = gather_nd(all_log_gauss_priors, component_samples2)
        log_gauss_prior = tensor.expand(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tensor.expand(v_samples2[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tensor.expand(v_samples2[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])

        # calc post term
        log_kumar_post = log_kumar_pdf(v_samples2, self.kumar_a2, self.kumar_b2)

        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z2[k], self.z_mu2[k], self.z_sigma2[k]))
        all_log_gauss_posts = torch.cat(1, all_log_gauss_posts)
        log_gauss_post = gather_nd(all_log_gauss_posts, component_samples2)
        log_gauss_post = tensor.expand(log_gauss_post,1)


        ### BOTTOM LEVEL

        # calc prior terms
        all_log_gauss_priors1 = []
        for k in xrange(self.K):
            for j in xrange(self.K):
                all_log_gauss_priors1.append(log_normal_pdf(self.z1[k][j], self.prior['mu'][k], self.prior['sigma'][k]))

        all_log_gauss_priors1 = torch.cat(all_log_gauss_priors1,1)
        log_gauss_prior1 = gather_nd(all_log_gauss_priors1, component_samples1)
        log_gauss_prior1 = tensor.expand(log_gauss_prior1,1)


        all_log_beta_priors1 = []
        for k in xrange(self.K):
            temp = torch.zeros((a2_inv.shape[0],1))
            for j in xrange(self.K-1):
                temp += log_beta_pdf(tensor.expand(v_samples1[k][:,j],1), self.prior['dirichlet_alpha'], (self.K-1-j)*self.prior['dirichlet_alpha'])
            all_log_beta_priors1.append(temp)

        all_log_beta_priors1 = torch.cat(1, all_log_beta_priors1)
        log_beta_prior1 = gather_nd(all_log_beta_priors1, component_samples2)
        log_beta_prior1 = tensor.expand(log_beta_prior1,1)


        # calc post terms
        all_log_gauss_posts1 = []
        for k in xrange(self.K):
            for j in xrange(self.K):
                all_log_gauss_posts1.append(log_normal_pdf(self.z1[k][j], self.z_mu1[k][j], self.z_sigma1[k]))

        all_log_gauss_posts1 = torch.cat( all_log_gauss_posts1, 1)
        log_gauss_post1 = gather_nd(all_log_gauss_posts1, component_samples1)
        log_gauss_post1 = tensor.expand(log_gauss_post1,1)


        all_log_kumar_posts1 = []
        for k in xrange(self.K):
            all_log_kumar_posts1.append( log_kumar_pdf(v_samples1[k], self.kumar_a1[k], self.kumar_b1[k]) )

        all_log_kumar_posts1 = torch.cat(1, all_log_kumar_posts1)
        log_kumar_post1 = gather_nd(all_log_kumar_posts1, component_samples2)
        log_kumar_post1 = tensor.expand(log_kumar_post1,1)


        return ll + log_beta_prior + log_beta_prior1 + log_gauss_prior + log_gauss_prior1 - log_kumar_post - log_kumar_post1 - log_gauss_post - log_gauss_post1
