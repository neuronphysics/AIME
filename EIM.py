from multiprocessing import context
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import normal
import numpy as np
import os
from planner_regulizer_EIM import Split
from tqdm import tqdm
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from typing import List
import math
from itertools import chain
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.modules import ModuleTrainer
from torchsample.metrics import Metric
from torchsample.regularizers import L2Regularizer
from torchsample.initializers import XavierUniform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsample.datasets import TensorDataset
"""
    minimizing Kullback-Leibler divergences by estimating density ratios
    Based on https://github.com/pbecker93/ExpectedInformationMaximization/
"""
def weights_init(modules, type='xavier'):
    "Based on shorturl.at/jmqV3"
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for k, v in m._modules.items():
            if isinstance(v, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    n = v.kernel_size[0] * v.kernel_size[1] * v.out_channels
                    v.weight.data.normal_(0, np.sqrt(2. / n))

                if v.bias is not None:
                    v.bias.data.zero_()
            elif isinstance(v, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    n = v.kernel_size[0] * v.kernel_size[1] * v.out_channels
                    v.weight.data.normal_(0, np.sqrt(2. / n))

                if v.bias is not None:
                    v.bias.data.zero_()
            elif isinstance(v, nn.BatchNorm2d):
                v.weight.data.fill_(1.0)
                v.bias.data.zero_()
            elif isinstance(v, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    v.weight.data.fill_(1.0)

                if v.bias is not None:
                    v.bias.data.zero_()

class RecorderKeys:
    TRAIN_ITER = "train_iteration_module"
    INITIAL = "initial_module"
    MODEL = "model_module"
    DRE = "dre_rec_mod"
    WEIGHTS_UPDATE = "weights_update"
    COMPONENT_UPDATE = "component_update"

class ConfigDict:

    def __init__(self, **kwargs):
        self._adding_permitted = True
        self._modifying_permitted = True
        self._c_dict = {**kwargs}
        self._initialized = True

    def __setattr__(self, key, value):
        if "_initialized" in self.__dict__:
            if self._adding_permitted:
                self._c_dict[key] = value
            else:
                if self._modifying_permitted and key in self._c_dict.keys():
                    self._c_dict[key] = value
                elif key in self._c_dict.keys():
                    raise AssertionError("Tried modifying existing parameter after modifying finalized")
                else:
                    raise AssertionError("Tried to add parameter after adding finalized")
        else:
            self.__dict__[key] = value

    def __getattr__(self, item):
        if "_initialized" in self.__dict__ and item in self._c_dict.keys():
            return self._c_dict[item]
        else:
            raise AssertionError("Tried accessing non existing parameter")

    def __getitem__(self, item):
        return self._c_dict[item]

    @property
    def adding_permitted(self):
        return self.__dict__["_adding_permitted"]

    @property
    def modifying_permitted(self):
        return self.__dict__["_modifying_permitted"]

    def finalize_adding(self):
        self.__dict__["_adding_permitted"] = False

    def finalize_modifying(self):
        if self.__dict__["_adding_permitted"]:
            warnings.warn("ConfigDict.finalize_modifying called while adding still allowed - also deactivating adding!")
            self.__dict__["_adding_permitted"] = False
        self.__dict__["_modifying_permitted"] = False

    def keys(self):
        return self._c_dict.keys()


class NetworkKeys:
    NUM_UNITS = "num_units"
    ACTIVATION = "activation"
    L2_REG_FACT = "l2_reg_fact"
    DROP_PROB = "drop_prob"
    BATCH_NORM = "batch_norm"

def l2_penalty(model, l2_lambda=0.001):
    """Returns the L2 penalty of the params."""
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l2_lambda*l2_norm


def build_dense_network(input_dim, output_dim, output_activation, params, with_output_layer=True):



    activation = params.get(NetworkKeys.ACTIVATION, "relu")
    l2_reg_fact = params.get(NetworkKeys.L2_REG_FACT, 0.0)
    drop_prob = params.get(NetworkKeys.DROP_PROB, 0.0)
    batch_norm = params.get(NetworkKeys.BATCH_NORM, False)
    layers=[]
    last_dim = input_dim
    for i in range(len(params[NetworkKeys.NUM_UNITS])):
        layers.append(nn.Linear(last_dim,params[NetworkKeys.NUM_UNITS][i]))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(params[NetworkKeys.NUM_UNITS][i]))

        if activation=="relu":
            layers.append(nn.ReLU())
        elif activation=="LeakyRelu":
            layers.append(nn.LeakyReLU(0.1,inplace=True))
        else:
            pass

        last_dim = params[NetworkKeys.NUM_UNITS][i]

        if drop_prob > 0.0:
            layers.append(torch.nn.Dropout(p=drop_prob))

    if with_output_layer:

        layers.append(nn.Linear(params[NetworkKeys.NUM_UNITS][-1],output_dim))
    model = nn.Sequential(*layers)
    regularizer = l2_penalty(model, l2_lambda=0.001) if l2_reg_fact > 0 else None
    return model, regularizer
###########################################
#############  Distributions  #############

class Gaussian:

    def __init__(self, mean, covar):
        self._dim = mean.shape[-1]
        self.update_parameters(mean, covar)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log2pi  = torch.log(torch.tensor(2*np.pi, requires_grad=False)).to(self.device)
        self.log2piE = torch.log(torch.tensor(2 * np.pi * np.e, requires_grad=False)).to(self.device)

    def density(self, samples):
        return torch.exp(self.log_density(samples))

    def log_density(self, samples):
        norm_term = self._dim * self.log2pi + self.covar_logdet()
        diff = samples - self._mean
        exp_term = torch.sum(torch.square(diff @ self._chol_precision), dim=-1)
        return -0.5 * (norm_term + exp_term)

    def log_likelihood(self, samples):
        return torch.mean(self.log_density(samples))

    def sample(self, num_samples):
        eps = normal.Normal(0,1).sample(sample_shape=[num_samples, self._dim])
        return self._mean + eps @ self._chol_covar.T

    def entropy(self):
        return 0.5 * (self._dim * self.log2piE + self.covar_logdet())

    def kl(self, other):
        trace_term = torch.sum(torch.square(other.chol_precision.T @ self._chol_covar))
        kl = other.covar_logdet() - self.covar_logdet() - self._dim + trace_term
        diff = other.mean - self._mean
        kl = kl + torch.sum(torch.square(other.chol_precision.T @ diff))
        return 0.5 * kl

    def covar_logdet(self):
        return 2 * torch.sum(torch.log(torch.diag(self._chol_covar) + 1e-25))

    def update_parameters(self, mean, covar):
        try:
            chol_covar = torch.linalg.cholesky(covar)
            inv_chol_covar = torch.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar

            self._chol_precision = torch.linalg.cholesky(precision)
            self._mean = mean
            self._lin_term = precision @ mean
            self._covar = covar
            self._precision = precision

            self._chol_covar = chol_covar

        except Exception as e:
            print("Gaussian Paramameter update rejected:", e)

    @property
    def mean(self):
        return self._mean

    @property
    def covar(self):
        return self._covar

    @property
    def lin_term(self):
        return self._lin_term

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision

class Categorical:

    def __init__(self, probabilities):
        self._p = probabilities
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
    def sample(self, num_samples):
        thresholds = torch.unsqueeze(torch.cumsum(self._p, dim=0), dim=0)
        thresholds[0, -1] = 1.0

        eps = torch.distributions.uniform.Uniform(0, 1).rsample([num_samples, 1])
        samples = torch.argmax((eps < thresholds).long(), dim=-1)
        return samples

    @property
    def probabilities(self):
        return self._p

    @probabilities.setter
    def probabilities(self, new_probabilities):
        self._p = new_probabilities

    @property
    def log_probabilities(self):
        return torch.log(self._p + 1e-25)

    def entropy(self):
        return - torch.sum(self._p * torch.log(self._p + 1e-25))

    def kl(self, other):
        return torch.sum(self._p * (torch.log(self._p + 1e-25) - other.log_probabilities))

class GMM:

    def __init__(self, weights, means, covars):
        self._weight_distribution = Categorical(weights)
        self._components = [Gaussian(means[i], covars[i]) for i in range(means.shape[0])]

    def density(self, samples):
        densities = torch.stack([self._components[i].density(samples) for i in range(self.num_components)], dim=0)
        w = torch.unsqueeze(self.weight_distribution.probabilities, dim=-1)
        return torch.sum(w * densities, dim=0)

    def log_density(self, samples):
        return torch.log(self.density(samples) + 1e-25)

    def log_likelihood(self, samples):
        return torch.mean(self.log_density(samples))

    def sample(self, num_samples):
        w_samples = self._weight_distribution.sample(num_samples)
        samples =[]
        for i in range(self.num_components):
            cns = torch.count_nonzero(w_samples == i)
            if cns > 0:
                samples.append(self.components[i].sample(cns))
        return torch.randperm(torch.cat(samples, dim=0))

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    @property
    def weight_distribution(self):
        return self._weight_distribution



class DensityRatioAccuracy(Metric):

    def __init__(self):
        super(DensityRatioAccuracy, self).__init__()
        self.p_prob = 0
        self.q_prob = 0

        self._name = 'DRAC_metric'

    def reset(self):
        self.p_prob = 0
        self.q_prob = 0

    def __call__(self, pq_outputs):
        #print("pq_outputs", pq_outputs)
        p_outputs, q_outputs = pq_outputs
        self.p_prob = torch.nn.Sigmoid()(p_outputs)
        self.q_prob = torch.nn.Sigmoid()(q_outputs)
        acc         = torch.mean(torch.cat([torch.greater_equal(self.p_prob, 0.5), torch.lt(self.q_prob, 0.5)], 0).type(torch.float32))
        #print("self.p_prob, self.q_prob", self.p_prob, self.q_prob)
        #print("torch.greater_equal(self.p_prob, 0.5)", torch.greater_equal(self.p_prob, 0.5))
        #print("torch.lt(self.q_prob, 0.5)", torch.lt(self.q_prob, 0.5))
        #print("acc", acc)
        return acc.cpu().detach().numpy()





def gaussian_log_density(samples, means, chol_covars):
    covar_logdet = 2 *torch.sum(torch.log(torch.diagonal(chol_covars, dim1=-2, dim2=-1)+1e-15),dim=-1)
    diff = torch.unsqueeze(samples - means, -1)
    exp_term = torch.sum(torch.square(torch.linalg.solve_triangular(chol_covars, diff, upper=False)), (-2, -1))
    return - 0.5 * (samples.size()[-1] * torch.log(2 * torch.tensor(np.pi, requires_grad=False)) + covar_logdet + exp_term)

def gaussian_density(samples, means, chol_covars):
    return torch.exp(ConditionalGaussian.gaussian_log_density(samples, means, chol_covars))



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

###############################################
def SplitLayer(layer,x,y):
    r      = layer.chunk(x+y, dim = 1)
    first  = torch.cat([i for i in r[:x]],dim=1)
    second = torch.cat([i for i in r[x::]], dim=1)
    return first, second


###############################################

#################  conditional ################
class ConditionalGaussian(nn.Module):

    def __init__(self, context_dim, sample_dim, hidden_dict, seed, trainable=True, weight_path=None):
        super(ConditionalGaussian, self).__init__()
        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._seed = seed
        self._hidden_dict = hidden_dict

        self.trainable = trainable


        self._hidden_net, self.regularizer = build_dense_network(self._context_dim, output_dim=-1, output_activation=None,
                                                  params=self._hidden_dict, with_output_layer=False)

        self._model = self._hidden_net
        idx = list(self._model._modules.keys())[-1] #get the index of last component in Sequential
        hidden_dim = self._hidden_net._modules[list(self._hidden_net._modules)[-2]].out_features
        #add a linear layer for a combination of mean and covariance
        self._model._modules[str(int(idx)+1)] = torch.nn.ReLU()
        self._model._modules[str(int(idx)+2)] = nn.Linear(hidden_dim, self._sample_dim+self._sample_dim ** 2)
        #print(next(iter(next(reversed(self._hidden_net._modules.items())))))


        weights_init(self._model, 'xavier')
        #based on this  shorturl.at/pTVZ3
        self._chol_covar =  Lambda(self._create_chol)

        if weight_path is not None:
           self._model.load_state_dict(torch.load(weight_path))


    def forward(self, contexts):
        #print("contexts", contexts.shape)
        
        output      = self._model(contexts)
        #print("output", output.shape)
        mean, covar = SplitLayer(output,self._sample_dim,self._sample_dim ** 2)
        #print("self._sample_dim", self._sample_dim)
        #print("mean", mean.shape)
        #print("covar", covar.shape)
        chol_covar  = self._chol_covar(covar)
        #print("chol_covar", chol_covar.shape, chol_covar[0])
        covariance  = torch.matmul(chol_covar, torch.transpose(chol_covar, -2, -1))
        return mean, covariance, chol_covar

    def mean(self, contexts):
        return self(contexts)[0]

    def covar(self, contexts):
        return self(contexts)[1]


    def _create_chol(self, chol_raw):
        #tensorflow.linalg.band_part(input, num_lower, num_upper, name=None)
        #num_lower: A Tensor. The number of subdiagonals to preserve. Negative values ​​preserve all lower triangles
        #num_upper: A Tensor. The number of superdiagonals to preserve. Negative values ​​preserve all upper triangles. 
        # print("chol_raw", chol_raw.shape, chol_raw[0])
        # print("torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim])", torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]).shape, torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim])[0]) 
        # print("torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]).T", torch.transpose(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), 1, 2).shape, torch.transpose(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), 1, 2)[0])
        # print("torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0)", torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0).shape, torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0)[0])
        # print("torch.tril(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0)", torch.tril(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0).shape, torch.tril(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0)[0])
        # print("torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]).t(), diagonal=0).t()", torch.transpose(torch.triu(torch.transpose(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), 1, 2), diagonal=0), 1, 2).shape, torch.transpose(torch.triu(torch.transpose(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), 1, 2), diagonal=0), 1, 2)[0])
        # samples = torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]).t(), diagonal=0).t()
        samples = torch.tril(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), diagonal=0)
        #print("samples", samples.shape, samples[0])
        #print("torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1)", torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1))[0], torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1)).shape)
        
        samples[:, range(self._sample_dim), range(self._sample_dim)] = torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1))+ 1e-12
        #print("samples", samples.shape, samples[0])
        #output  = samples.fill_diagonal_(torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1))+ 1e-12)
        #print(output.size())
        return samples


    def sample(self, contexts):
        mean, _, chol_covar = self(contexts)
        torch.random.manual_seed(self._seed)
        #print("mean", mean.shape)
        eps = torch.normal(mean=0,std=1,size=(mean.shape[0], mean.shape[1], 1))
        #print("eps", eps.shape)
        #print("ConditionalGaussian sample", mean + torch.reshape(torch.matmul(chol_covar, eps), mean.shape))
        return mean + torch.reshape(torch.matmul(chol_covar, eps), mean.shape)

    def log_density(self, contexts, samples):
        mean, _, chol_covar = self(contexts)
        return gaussian_log_density(samples, mean, chol_covar)

    def density(self, contexts, samples):
        return torch.exp(self.log_density(contexts, samples))

    ##@staticmethod
    def expected_entropy(self, contexts):
        _, _, chol_covars = self(contexts)
        return 0.5 * (self._sample_dim * np.log(torch.tensor(2 * np.e * np.pi, requires_grad=False)) + torch.mean(self._covar_logdets(chol_covars)))

    ##@staticmethod
    def entropies(self, contexts):
        _, _, chol_covars = self(contexts)
        return 0.5 * (self._sample_dim * torch.log(torch.tensor(2 * np.e * np.pi, requires_grad=False)) + self._covar_logdets(chol_covars))

    ##@staticmethod
    def _covar_logdets(self, chol_covars):
        return 2 * torch.sum(torch.log(torch.diagonal(chol_covars, dim1=-2, dim2=-1) + 1e-12), dim=-1)

    ##@staticmethod
    def kls(self, contexts, other_means, other_chol_covars):
        means, _, chol_covars = self(contexts)
        kl = self._covar_logdets(other_chol_covars) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += torch.sum(torch.square(torch.linalg.solve_triangular(other_chol_covars, chol_covars, upper=False)), (-2, -1))
        diff = torch.unsqueeze(other_means - means, -1)
        kl += torch.sum(torch.square(torch.linalg.solve_triangular(other_chol_covars, diff, upper=False)), (-2, -1))
        return 0.5 * kl

    ##@staticmethod
    def kls_other_chol_inv(self, contexts, other_means, other_chol_inv):
        means, _, chol_covars = self(contexts)
        kl = - self._covar_logdets(other_chol_inv) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += torch.sum(torch.square(torch.matmul(other_chol_inv, chol_covars)),( -2, -1))
        diff = torch.unsqueeze(other_means - means, -1)
        kl += torch.sum(torch.square(torch.matmul(other_chol_inv, diff)), (-2, -1))
        return 0.5 * kl

    ##@staticmethod
    def expected_kl(self, contexts, other_means, other_chol_covars):
        return torch.mean(self.kls
        (contexts, other_means, other_chol_covars))

    ##@staticmethod
    def log_likelihood(self, contexts, samples):
        return torch.mean(self.log_density(contexts, samples))

    def conditional_params(self, contexts):
        #get mean and variance parameters of the network separately
        #shorturl.at/amWX4
        res =  self(contexts)
        return res[0], res[2]
        #return map(nn.Parameter, self.build(contexts))

    @property
    def trainable_variables(self) -> List[nn.Parameter]:
        """The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.
        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        param = [x for x in self._model.parameters() if x.requires_grad]
        chol  = [x for x in self._chol_covar.parameters() if x.requires_grad]
        param.extend(chol)

        return param

    @property
    def sample_dim(self):
        return self._sample_dim

    def save_model_params(self, filepath):
        torch.save(self._model.state_dict(), filepath )



class Softmax(nn.Module):

    def __init__(self, context_dim, z_dim, hidden_dict, seed, trainable=True, weight_path=None):
        super(Softmax, self).__init__()
        self._context_dim = context_dim
        self._z_dim = z_dim
        self._seed = seed
        self._hidden_dict = hidden_dict
        self._trainable = trainable

        self._logit_net, self._logit_regularizer = build_dense_network(self._context_dim, self._z_dim, output_activation=None,
                                                 params=self._hidden_dict)
        print("self._context_dim, self._z_dim", self._context_dim, self._z_dim)
        print("self._hidden_dict", self._hidden_dict)

        if weight_path is not None:
           self._logit_net.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        #print("x.shape", x.shape)
        return self._logit_net(x)

    def logits(self, contexts):
        return self(contexts)

    def probabilities(self, contexts):
        return nn.Softmax(dim=1)(self.logits(contexts))

    def log_probabilities(self, contexts):
        return torch.log(self.probabilities(contexts) + 1e-12)

    def expected_entropy(self, contexts):
        p = self.probabilities(contexts)
        return - torch.mean(torch.sum(p * torch.log(p + 1e-12), -1))

    def expected_kl(self, contexts, other_probabilities):
        p = self.probabilities(contexts)
        return p, torch.mean(torch.sum(p * (torch.log(p + 1e-12) - torch.log(other_probabilities + 1e-12)), -1))

    def sample(self, contexts):
        p = self.probabilities(contexts)
        thresholds = torch.cumsum(p, dim=-1)
        # ensure the last threshold is always exactly one - it can be slightly smaller due to numerical inaccuracies
        # of cumsum, causing problems in extremely rare cases if a "n" is samples that's larger than the last threshold
        thresholds = torch.cat([thresholds[..., :-1], torch.ones([thresholds.size()[0], 1])], -1)
        torch.random.manual_seed(self._seed)
        n=torch.distributions.uniform.Uniform(0.0,1.0).rsample((thresholds.size()[0], 1))
        idx = torch.where(torch.lt(n, thresholds), torch.arange(self._z_dim) * torch.ones(thresholds.shape, dtype=torch.int64),
                       self._z_dim * torch.ones(thresholds.shape, dtype=torch.int64))
        return torch.min(idx, -1)[0]

    @property
    def trainable_variables(self)-> List[nn.Parameter]:
        """The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.
        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self._logit_net.parameters() if x.requires_grad]

    def save_model_params(self, filepath):
        torch.save(self._logit_net.state_dict(), filepath )
############################

#############################
class GaussianEMM(nn.Module):
    """gated mixture of experts """
    def __init__(self, context_dim, sample_dim, number_of_components, gating_num_epochs, dre, component_hidden_dict, gating_hidden_dict,
                 seed=0, trainable=True, weight_path=None):
        super().__init__()
        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._number_of_components = number_of_components
        self._gating_num_epochs   = gating_num_epochs
        self._mixture_hidden_dict = gating_hidden_dict
        self._component_hidden_dict = component_hidden_dict
        #self.hidden_params = hidden_params

        wp = None if weight_path is None else os.path.join(weight_path, self._mixture_params_file_name())
        self._gating_distribution = Softmax(self._context_dim, self._number_of_components, gating_hidden_dict,
                                            seed=seed, trainable=trainable, weight_path=wp)

        self._gating_output_size = list(self._gating_distribution._modules.values())[-1][-1].out_features
        # input features: 6
        self._gating_input_size  = list(self._gating_distribution._modules.values())[-1][0].in_features
        print(self._gating_output_size,self._gating_input_size)


        self._components = []
        for i in range(number_of_components):
            h_dict = component_hidden_dict[i] if isinstance(component_hidden_dict, list) else component_hidden_dict
            wp = None if weight_path is None else os.path.join(weight_path, self._component_params_file_name(i))
            c = ConditionalGaussian(self._context_dim, self._sample_dim, h_dict,
                                    trainable=trainable, seed=seed, weight_path=wp)
            self._components.append(c)
        #iput features: 6
        self._component_input_size  = list(self._components[-1]._modules.values())[0][0].in_features
        print(self._component_input_size)
        #number_of_components: 3
        self.trainable_variables = self._gating_distribution.trainable_variables
        # self._net, self._regularizer = build_dense_network(input_dim=input_dim, output_dim=1,
        #                                      output_activation="linear", params=self.hidden_params)
        self._net = dre
        for c in self._components:
            self.trainable_variables += c.trainable_variables


    def forward(self, inputs):
        #print("inputs", inputs.shape)
        if (self._number_of_components > 1):
          
           old_probs = self.gating_distribution.probabilities(inputs)
           self.gating_losses={}
           for i in range(self.gating_num_epochs):
               gating_losses = []
               for j in range(self._number_of_components):
                   samples = self._components[j].sample(inputs)
                   gating_losses.append(- self._net(torch.cat([inputs, samples], dim=-1)))
               gating_losses.append( torch.cat(gating_losses, dim=1))
           self.gating_losses["loss"]=torch.cat(self.gating_losses, dim=2)
           expected_entropy = self.gating_distribution.expected_entropy(inputs)
           _, self.gating_losses["expected_kl"] = self.gating_distribution.expected_kl(inputs, old_probs)
           self.gating_losses["total_loss"] = torch.sum(torch.mean(probabilities * torch.mean(self.gating_losses["loss"],dim=2), 0)) + self.gating_losses["expected_kl"]
        importance_weights = self.gating_distribution.probabilities(inputs)
        importance_weights = importance_weights / torch.sum(importance_weights, dim=0, keepdims=True)
        #is importance_weights a distribution? Test it?
        old_means, old_chol_covars = self.get_component_parameters(inputs)

        print("[forward] old_means, old_chol_covars", old_means.shape, old_chol_covars.shape)
        rhs = torch.eye(old_means.shape[-1], batch_shape=old_chol_covars.shape[:-2])
        stab_fact = 1e-20
        old_chol_inv = torch.linalg.solve_triangular(old_chol_covars + stab_fact * rhs, rhs, upper=False)
        self.components_loss={}
        kls_components=[]
        component_net_losses=[]
        total_losses=[]
        for i in range(self._number_of_components):
            iw_batch = importance_weights[:, i] / torch.sum(importance_weights[:, i])
            samples = self._components[i].sample(inputs)
            component_losses = - torch.squeeze(self._net(torch.cat([inputs, samples], dim=-1)))
            kls =self._components[i].kls_other_chol_inv(inputs, old_means[:, i],old_chol_inv[:, i])
            kls_components.append(kls)
            component_net_losses.append(component_losses)
            loss = torch.mean(iw_batch * (component_losses + kls))
            total_losses.append(loss)
        self.components_loss["KL"]=torch.cat(kls_components,dim=1)
        self.components_loss["each_componnent_network"]=torch.cat(component_net_losses,dim=1)
        self.components_loss["total_loss"]=torch.cat(loss,dim=1)
        comp = D.Independent(D.Normal(
             old_means, old_chol_covars), 1)
        gmm = MixtureSameFamily(importance_weights, comp)
        return gmm

    def density(self, contexts, samples):
        p = self._gating_distribution.probabilities(contexts)
        density = p[:, 0] * self._components[0].density(contexts, samples)
        for i in range(1, self._number_of_components):
            density += p[:, i] * self._components[i].density(contexts, samples)
        return density

    def log_density(self, contexts, samples):
        return torch.log(self.density(contexts, samples) + 1e-12)

    def log_likelihood(self, contexts, samples):
        #print("contexts", contexts, contexts.shape)
        #print("samples", samples, samples.shape)
        return torch.mean(self.log_density(contexts, samples))

    def sample(self, contexts):
        modes = self._gating_distribution.sample(contexts)
        samples = torch.zeros([contexts.shape[0], self._sample_dim])
        #print("modes", modes, "_number_of_components", self._number_of_components)
        for i in range(self._number_of_components):
            idx = modes == i
            #print(idx)
            #print(torch.eq(modes, i))
            if torch.any(idx):
                samples[idx] = self._components[i].sample(contexts[idx])
        #print("GaussianEMM samples", samples, samples.shape)
        return samples

    @property
    def gating_distribution(self):
        return self._gating_distribution

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    def get_component_parameters(self, context):
        means = []
        chol_covars = []
        for c in self._components:
            m, cc = c.conditional_params(context)
            means.append(m)
            chol_covars.append(cc)
        return torch.stack(means, 1), torch.stack(chol_covars, 1)


###################################################



############# Main Class of the Model #############
def LogisticRegressionLoss(pq_outputs):
    p_outputs, q_outputs = pq_outputs
    loss = - torch.mean(torch.log(torch.sigmoid(p_outputs) +1e-12))- torch.mean(torch.log(1 - torch.sigmoid(q_outputs) + 1e-12))
    return  loss

class DensityRatioEstimator(nn.Module):

    def __init__(self, target_train_samples, hidden_params, early_stopping=False, target_val_samples=None,
                 conditional_model=False):
        self.acc_fn = DensityRatioAccuracy()
        super(DensityRatioEstimator,self).__init__()
        self._early_stopping = early_stopping
        self._conditional_model = conditional_model

        if self._conditional_model:
            self._train_contexts = torch.from_numpy(target_train_samples[0]).type(torch.float32)

            self._target_train_samples = torch.cat([torch.from_numpy(x) for x in target_train_samples], -1).type(torch.float32)
        else:
            self._target_train_samples = torch.from_numpy(target_train_samples).type(torch.float32)

        if self._early_stopping:
            assert target_val_samples is not None, \
                "For early stopping validation data needs to be provided via target_val_samples"
            if self._conditional_model:
                self._val_contexts = torch.from_numpy(target_val_samples[0]).type(torch.float32)
                self._target_val_samples = torch.cat([torch.from_numpy(x) for x in target_val_samples], -1).type(torch.float32)
            else:
                self._target_val_samples = torch.from_numpy(target_val_samples).type(torch.float32)
        self.hidden_params=hidden_params
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_dim = self._target_train_samples.shape[-1]

        self._ldre_net, self._ldre_regularizer = build_dense_network(input_dim=input_dim, output_dim=1,
                                             output_activation="linear", params=self.hidden_params)
        #print("_ldre_net input", input_dim)

        self._p_samples = nn.Linear(input_dim,input_dim)
        self._q_samples = nn.Linear(input_dim,input_dim)
        self.to(self.device)

    def forward(self, x, inTrain=True):
        #print("forward x", x.shape, "inTrain", inTrain)
        if inTrain:
          p = self._p_samples(x)
          q = self._q_samples(x)
          p = x[:, 0, :]
          q = x[:, 1, :]
          # print("p.shape", p.shape)
          # print("q.shape", q.shape)
          combined = torch.cat((p.view(p.size(0), -1),
                                q.view(q.size(0), -1)), dim=0)
          self._split_layers = Split(
           self._ldre_net,
           n_parts=2,
           dim = 0
          )
          p_output, q_output =self._split_layers(combined)
          return p_output, q_output
        else:
          return self._ldre_net(x)

    # def __call__(self, samples):
    #     print("__call__ x", samples.shape)
    #     return self._ldre_net(samples)

    

    def train_dre(self, model, batch_size, num_iters):
        #compile the model
        #https://github.com/ncullen93/torchsample
        self.trainer   = ModuleTrainer(self) # model
        self.batch_size= batch_size
        metrics   = [self.acc_fn]
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        initializers = [XavierUniform(bias=False, module_filter='*')]
        regularizers   = [L2Regularizer(scale=1e-4)]
        #LogisticRegressionLoss  = lambda p_outputs, q_outputs: - torch.mean(torch.log(torch.sigmoid(p_outputs) +1e-12))- torch.mean(torch.log(1 - torch.sigmoid(q_outputs) + 1e-12))

        if self._early_stopping:
           model_train_samples, model_val_samples = self.sample_model(model)
           callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                        ReduceLROnPlateau(factor=0.5, patience=5)]
           #print(f" size of validation data : {model_val_samples.size(0)}")
           val_dataset =  TensorDataset([self._target_val_samples, model_val_samples])
           val_loader  = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True)
           #val_data = torch.tensor([[target, train] for target, train in zip(self._target_val_samples, model_val_samples)])
           val_data = torch.stack([self._target_val_samples, model_val_samples], dim=1)
           
        else:
           model_train_samples = self.sample_model(model)
           callbacks  = []
           val_loader = None
           val_data = None
        train_dataset =  TensorDataset([self._target_train_samples, model_train_samples])
        #train_data = torch.tensor([[target, train] for target, train in zip(self._target_train_samples, model_train_samples)])
        #train_data = torch.cat([self._target_train_samples, model_train_samples], dim=1)
        train_data = torch.stack([self._target_train_samples, model_train_samples], dim=1)

        train_loader  = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        # print(batch_size,num_iters)
        # print(self._target_val_samples.shape, model_val_samples.shape)
        # print(self._target_train_samples.shape, model_train_samples.shape)

        self.trainer.compile(loss=LogisticRegressionLoss,
                             optimizer=optimizer,
                             regularizers=regularizers,
                             initializers=initializers,
                             metrics=metrics,
                             callbacks=callbacks)
        #print("shape of input data....")
        #torch.Size([10000, 12]) torch.Size([10000, 12])
        #print(self._target_train_samples.size(), model_train_samples.size())
        # self.trainer.fit_loader(train_loader,
        #                         val_loader,
        #                         num_epoch = num_iters,
        #                         verbose = 1)
        #print("train_data", train_data.shape)
        #print("val_data", val_data.shape)
        last_epoch = self.trainer.fit(train_data, 
                           targets=None,
                           val_data=val_data,
                           num_epoch=num_iters,
                           verbose=1)
        #self.trainer.fit((self._target_train_samples, model_train_samples), targets=None, val_data=)

        return self.trainer.last_epoch_log["last_epoch"]+1, self.trainer.last_epoch_log["loss"], self.trainer.last_epoch_log["DRAC_metric"]



    def eval(self, target_samples, model):
        #Turns off training-time behavior
        if self._conditional_model:
            model_samples = torch.cat([target_samples[0], model.sample(target_samples[0])], dim=-1)
            target_samples = torch.cat([target_samples[0], target_samples[1]], dim=-1)
        else:
            model_samples = model.sample(self._target_train_samples.shape[0])
        target_ldre = self(target_samples, inTrain=False)
        model_ldre  = self(model_samples, inTrain=False)
        target_prob = torch.sigmoid(target_ldre)
        model_prob  = torch.sigmoid(model_ldre)
        #eval_loss   = self.trainer.evaluate(target_ldre, model_ldre, batch_size=self.batch_size, verbose=1, inTrain=False)
        ikl_estem   = torch.mean(- model_ldre)
        acc  = self.acc_fn((target_ldre, model_ldre))
        bce = LogisticRegressionLoss((target_ldre, model_ldre))

        return ikl_estem, bce, acc, torch.mean(target_prob), torch.mean(model_prob)

    def sample_model(self, model):
        """Sample model for density ratio estimator training"""
        if self._conditional_model:
            model_train_samples = torch.cat([self._train_contexts, model.sample(self._train_contexts)], dim=-1)
        else:
            model_train_samples = model.sample(self._target_train_samples.shape[0])
        if self._early_stopping:
            if self._conditional_model:
                model_val_samples = torch.cat([self._val_contexts, model.sample(self._val_contexts)], dim=-1)
            else:
                model_val_samples = model.sample(self._target_val_samples.shape[0])
            return model_train_samples, model_val_samples
        return model_train_samples

class Recorder:

    def __init__(self, modules_dict,
                 plot_realtime, save,
                 save_path=os.path.abspath("rec"),
                 fps=3, img_dpi=1200, vid_dpi=500):
        """
        Initializes and handles recording modules
        :param modules_dict: Dictionary containing all recording modules
        :param plot_realtime: whether to plot in realtime while algorithm is running
        :param save: whether to save plots/images
        :param save_path: path to save data to
        :param fps: video frame rate
        :param img_dpi: image resolution
        :param vid_dpi: video resolution
        """
        self.save_path = save_path
        self._modules_dict = modules_dict

        self._plot_realtime = plot_realtime
        self._save = save

        self._fps = fps
        self._img_dpi = img_dpi
        self._vid_dpi = vid_dpi

        if not os.path.exists(save_path):
             warnings.warn('Path ' + save_path + ' not found - creating')
             os.makedirs(save_path)

    def handle_plot(self, name, plot_fn, data=None):
        if self._plot_realtime:
            plt.figure(name)
            plt.clf()
            plot_fn() if data is None else plot_fn(data)
            plt.pause(0.0001)

    def save_img(self, name, plot_fn, data=None):
        """
        Saves an image
        :param name: file name
        :param plot_fn: function generating the plot
        :param data: data provided to plot_fn to generate plot
        :return:
        """
        fig = plt.figure(name)
        plot_fn() if data is None else plot_fn(data)
        fig.savefig(os.path.join(self.save_path, name + ".pdf"), format="pdf", dpi=self._img_dpi)
        plt.close(name)

    def save_vid(self, name, update_fn, frames):
        """
        Saves a video
        :param name: file name
        :param update_fn: function generating plots from data
        :param frames: list containing data for frames
        :return:
        """
        def _update_fn_wrapper(i):
            plt.clf()
            update_fn(i)

        if self._save:
            fig = plt.figure()
            ani = anim.FuncAnimation(fig,
                                     func=_update_fn_wrapper,
                                     frames=frames)
            writer = anim.writers['imagemagick'](fps=self._fps)
            ani.save(os.path.join(self.save_path, name+".mp4"),
                     writer=writer,
                     dpi=(500 if self._vid_dpi > 500 else self._vid_dpi))

    def save_vid_raw(self, name, data, preprocess_fn=None):
        save_dict = {}
        preprocess_fn = (lambda x: x) if preprocess_fn is None else preprocess_fn
        for i, d in enumerate(data):
            save_dict[str(i)] = preprocess_fn(data[i])
        np.savez(os.path.join(self.save_path, name + "_raw.npz"), **save_dict)

    def initialize_module(self, name, *args, **kwargs):
        module = self._modules_dict.get(name)
        if module is not None:
            module.initialize(self, self._plot_realtime, self._save, *args, **kwargs)

    def __call__(self, module, *args, **kwargs):
        module = self._modules_dict.get(module)
        if module is not None:
            module.record(*args, **kwargs)

    def snapshot(self):
        for key in self._modules_dict.keys():
            self._modules_dict[key].snapshot()

    def finalize_training(self):
        for key in self._modules_dict.keys():
            m = self._modules_dict[key]
            if m.is_initialized:
                m.finalize()

    def get_last_rec(self):
        last_rec = {}
        for key in self._modules_dict.keys():
            last_rec = {**last_rec, **self._modules_dict[key].get_last_rec()}
        return last_rec

class ConditionalMixtureEIM:

    @staticmethod
    def get_default_config() -> ConfigDict:
        c = ConfigDict(
            num_components=1,
            train_epochs=1000,
            # Component
            components_learning_rate=1e-3,
            components_batch_size=1000,
            components_num_epochs=10,
            components_net_reg_loss_fact=0.,
            components_net_drop_prob=0.0,
            components_net_hidden_layers=[50, 50],
            # Gating
            gating_learning_rate=1e-3,
            gating_batch_size=1000,
            gating_num_epochs=10,
            gating_net_reg_loss_fact=0.,
            gating_net_drop_prob=0.0,
            gating_net_hidden_layers=[50, 50],
            # Density Ratio Estimation
            dre_reg_loss_fact=0.0,  # Scaling Factor for L2 regularization of density ratio estimator
            dre_early_stopping=True,  # Use early stopping for density ratio estimator training
            dre_drop_prob=0.0,  # If smaller than 1 dropout with keep prob = 'keep_prob' is used
            dre_num_iters=1000,  # Number of density ratio estimator steps each iteration (i.e. max number if early stopping)
            dre_batch_size=1000,  # Batch size for density ratio estimator training
            dre_hidden_layers=[30, 30]  # width of density ratio estimator  hidden layers
        )
        c.finalize_adding()
        return c

    def __init__(self, config: ConfigDict, train_samples: torch.Tensor, recorder: Recorder,
                 val_samples: torch.Tensor = None, seed: int = 0):
        # supress pytorch casting warnings
        logging.getLogger("pytorch").setLevel(logging.ERROR)

        self.c = config
        self.c.finalize_modifying()

        self._recorder = recorder

        self._context_dim = train_samples[0].shape[-1]
        self._sample_dim = train_samples[1].shape[-1]
        print("train_samples", train_samples[0].shape, train_samples[1].shape, "_context_dim", self._context_dim, "_sample_dim", self._sample_dim)
        self._train_contexts = torch.tensor(train_samples[0], dtype=torch.float32)

        c_net_hidden_dict = {NetworkKeys.NUM_UNITS: self.c.components_net_hidden_layers,
                             NetworkKeys.ACTIVATION: "relu",
                             NetworkKeys.BATCH_NORM: False,
                             NetworkKeys.DROP_PROB: self.c.components_net_drop_prob,
                             NetworkKeys.L2_REG_FACT: self.c.components_net_reg_loss_fact}

        g_net_hidden_dict = {NetworkKeys.NUM_UNITS: self.c.gating_net_hidden_layers,
                             NetworkKeys.ACTIVATION: "relu",
                             NetworkKeys.BATCH_NORM: False,
                             NetworkKeys.DROP_PROB: self.c.gating_net_drop_prob,
                             NetworkKeys.L2_REG_FACT: self.c.gating_net_reg_loss_fact}

        dre_params = {NetworkKeys.NUM_UNITS: self.c.dre_hidden_layers,
                      NetworkKeys.ACTIVATION: "relu",
                      NetworkKeys.DROP_PROB: self.c.dre_drop_prob,
                      NetworkKeys.L2_REG_FACT: self.c.dre_reg_loss_fact}


        self._dre = DensityRatioEstimator(target_train_samples=train_samples,
                                          hidden_params=dre_params,
                                          early_stopping=self.c.dre_early_stopping, target_val_samples=val_samples,
                                          conditional_model=True)
        
        self._model = GaussianEMM(self._context_dim, self._sample_dim, self.c.num_components, self.c.gating_num_epochs, self._dre,
                                  c_net_hidden_dict, g_net_hidden_dict, seed=seed)



        
        self._recorder.initialize_module(RecorderKeys.INITIAL)
        self._recorder(RecorderKeys.INITIAL, "Nonlinear Conditional EIM - Reparametrization", config)
        self._recorder.initialize_module(RecorderKeys.MODEL, self.c.train_epochs)
        self._recorder.initialize_module(RecorderKeys.WEIGHTS_UPDATE, self.c.train_epochs)
        self._recorder.initialize_module(RecorderKeys.COMPONENT_UPDATE, self.c.train_epochs, self.c.num_components)
        self._recorder.initialize_module(RecorderKeys.DRE, self.c.train_epochs)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_emm(self):
        for i in range(self.c.train_epochs):
            self._recorder(RecorderKeys.TRAIN_ITER, i)
            self.train_iter(i)

    #   extra function to allow running from cluster work
    def train_iter(self, i):

        dre_steps, loss, acc = self._dre.train_dre(self._model, self.c.dre_batch_size, self.c.dre_num_iters)
        with torch.no_grad():
          self._recorder(RecorderKeys.DRE, self._dre, self._model, i, dre_steps)

        if self._model.num_components > 1:
            w_res = self.update_gating()
            #print("w_res", w_res)
            with torch.no_grad():
              self._recorder(RecorderKeys.WEIGHTS_UPDATE, w_res)

        c_res = self.update_components()
        with torch.no_grad():
          self._recorder(RecorderKeys.COMPONENT_UPDATE, c_res)

          self._recorder(RecorderKeys.MODEL, self._model, i)

    """component update"""
    def update_components(self):
        with torch.no_grad():
          importance_weights = self._model.gating_distribution.probabilities(self._train_contexts)
          importance_weights = importance_weights / torch.sum(importance_weights, dim=0, keepdims=True)
          old_means, old_chol_covars = self._model.get_component_parameters(self._train_contexts)
  
        rhs = torch.eye(old_means.shape[-1]).repeat(*old_chol_covars.shape[:-2], 1, 1)

        stab_fact = 1e-20
        old_chol_inv = torch.linalg.solve_triangular(old_chol_covars + stab_fact * rhs, rhs, upper=False)
        #print("old_chol_inv", old_chol_inv.shape, old_chol_inv)
        for i in range(self.c.components_num_epochs):
            self._components_train_step(importance_weights, old_means, old_chol_inv)

        res_list = []
        for i, c in enumerate(self._model.components):
            expected_entropy = torch.sum(importance_weights[:, i] * c.entropies(self._train_contexts))
            kls = c.kls_other_chol_inv(self._train_contexts, old_means[:, i], old_chol_inv[:, i])
            expected_kl = torch.sum(importance_weights[:, i] * kls)
            res_list.append((self.c.components_num_epochs, expected_kl, expected_entropy, ""))
        return res_list


    def _components_train_step(self, importance_weights, old_means, old_chol_precisions):
        self._c_opts = [ torch.optim.Adam(self._model.components[i].trainable_variables, lr=self.c.components_learning_rate, betas=(0.5, 0.999)) for i in range(len(self._model.components))]

        for i in range(self._model.num_components):
            dataset = torch.utils.data.TensorDataset(self._train_contexts, importance_weights[:, i], old_means, old_chol_precisions)
            #print("self.c.components_batch_size", self.c.components_batch_size)
            #print("self._train_contexts, importance_weights[:, i], old_means, old_chol_precisions", self._train_contexts.shape, importance_weights[:, i].shape, old_means.shape, old_chol_precisions.shape)
            
            loader = torch.utils.data.DataLoader( dataset, shuffle = True, batch_size=self.c.components_batch_size)
            
            for batch_idx, (context_batch, iw_batch, old_means_batch, old_chol_precisions_batch) in enumerate(loader):
                self._c_opts[i].zero_grad()
                iw_batch = iw_batch / torch.sum(iw_batch)
                samples = self._model.components[i].sample(context_batch)
                #print("torch.cat([context_batch, samples], dim=-1)", torch.cat([context_batch, samples], dim=-1).shape)
                #with torch.no_grad():
                losses = - torch.squeeze(self._dre(torch.cat([context_batch, samples], dim=-1), inTrain=False))
                kls = self._model.components[i].kls_other_chol_inv(context_batch, old_means_batch[:, i], old_chol_precisions_batch[:, i])
                # print("context_batch", context_batch, context_batch.shape)
                # print("old_means_batch", old_means_batch, old_means_batch.shape)
                # print("old_chol_precisions_batch", old_chol_precisions_batch, old_chol_precisions_batch.shape)
                # print("iw_batch", iw_batch)
                # print("losses", losses)
                # print("kls", kls)
                loss = torch.mean(iw_batch * (losses + kls))
                #loss = torch.mean(iw_batch*kls)
                loss.backward()
                self._c_opts[i].step()
            self._c_opts[i].zero_grad()

    """gating update"""
    def update_gating(self):
        with torch.no_grad():
          old_probs = self._model.gating_distribution.probabilities(self._train_contexts)
        for i in range(self.c.gating_num_epochs):
            self._gating_train_step(old_probs)

        expected_entropy = self._model.gating_distribution.expected_entropy(self._train_contexts)
        _, expected_kl = self._model.gating_distribution.expected_kl(self._train_contexts, old_probs)

        return i + 1, expected_kl, expected_entropy, ""

    def _gating_train_step(self, old_probs):
        self._g_opt = torch.optim.Adam(self._model.gating_distribution.trainable_variables, lr=self.c.gating_learning_rate, betas=(0.5, 0.999))
        losses = []
        for i in range(self.c.num_components):
            samples = self._model.components[i].sample(self._train_contexts)
            #print("torch.cat([self._train_contexts, samples], dim=-1)", torch.cat([self._train_contexts, samples], dim=-1).shape)
            with torch.no_grad():
              losses.append(- self._dre(torch.cat([self._train_contexts, samples], dim=-1), inTrain=False))

        losses = torch.cat(losses, dim=1)
        dataset = torch.utils.data.TensorDataset(self._train_contexts, losses, old_probs)
        loader = torch.utils.data.DataLoader( dataset, shuffle = True, batch_size=self.c.gating_batch_size)
        for batch_idx, (context_batch, losses_batch, old_probs_batch) in enumerate(loader):
            self._g_opt.zero_grad()
            #probabilities = self._model.gating_distribution.probabilities(context_batch)
            # print("old_probs_batch",  old_probs_batch)
            # print("losses_batch", losses_batch)
            # print("context_batch", context_batch)
            probabilities, kl = self._model.gating_distribution.expected_kl(context_batch, old_probs_batch)
            loss = torch.sum(torch.mean(probabilities * losses_batch, 0)) + kl
            loss.backward()
            self._g_opt.step()
        self._g_opt.zero_grad()

    @property
    def model(self):
        return self._model
################################################################################
#################  Running an experiment to test the model #####################
################################################################################
from ObstacleData import ObstacleData
import json

class RecorderModule:
    """RecorderModule Superclass"""

    def __init__(self):
        self.__recorder = None
        self._plot_realtime = None
        self._save = None
        self._logger = logging.getLogger(self.logger_name)

    @property
    def _save_path(self):
        return self._recorder.save_path

    @property
    def _recorder(self):
        assert self.__recorder is not None, "recorder not set yet - Recorder module called before proper initialization "
        return self.__recorder

    @property
    def is_initialized(self):
        return self.__recorder is not None

    def initialize(self, recorder, plot_realtime, save, *args):
        self.__recorder = recorder
        self._save = save
        self._plot_realtime = plot_realtime

    def record(self, *args):
        raise NotImplementedError

    @property
    def logger_name(self):
        raise NotImplementedError

    def finalize(self):
        pass

    def get_last_rec(self):
        return {}


class ConfigInitialRecMod(RecorderModule):

    def record(self, name, config):
        self._logger.info((10 + len(name)) * "-")
        self._logger.info("---- " + name + " ----")
        self._logger.info((10 + len(name)) * "-")
        for k in config.keys():
            self._logger.info(str(k) + " : " + str(config[k]))
        if self._save:
            filename = os.path.join(self._save_path, "config.json")
            with open(filename, "w") as file:
                json.dump(config.__dict__, file, separators=(",\n", ": "))

    @property
    def logger_name(self):
        return "Config"

"""Simple recording models - most of them just print information"""
class TrainIterationRecMod(RecorderModule):
    """Prints out current training iteration"""
    def record(self, iteration):
        self._logger.info("--- Iteration {:5d} ---".format(iteration))

    @property
    def logger_name(self):
        return "Iteration"

"""Runs the obstacles experiments described in section 5.4 of the paper"""
#########################################
########################################
"""configure experiment"""
plot_realtime = True
plot_save = False
record_dual_opt = True
record_discriminator = True
num_components = 3

"""generate data"""

data = ObstacleData(10000, 5000, 5000, num_obstacles=3, samples_per_context=10, seed=0)
context_dim, sample_dim = data.dim
print("context_dim, sample_dim ", context_dim, sample_dim )

class Colors:
    """Provides colors for plotting """
    def __init__(self, pyplot_color_cycle=True):
        if pyplot_color_cycle:
            self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            #Todo implement color list with more colors...
            raise NotImplementedError("Not yet implemented")

    def __call__(self, i):
        return self._colors[i % len(self._colors)]

"""eval and recording"""
def save_model(model, path: str, filename: str):
    if isinstance(model, GMM):
        save_gmm(model, path, filename)
    elif isinstance(model, GaussianEMM):
        save_gaussian_gmm(model, path, filename)

    else:
        raise NotImplementedError("Saving not implemented for " + str(model.__class__))
def save_gaussian_gmm(model: GaussianEMM, path: str, filename: str):
    model.save(path, filename)


def save_gmm(model: GMM, path: str, filename: str):
    means = np.stack([c.mean for c in model.components], axis=0)
    covars = np.stack([c.covar for c in model.components], axis=0)
    model_dict = {"weights": model.weight_distribution.p, "means": means, "covars": covars}
    np.savez_compressed(os.path.join(path, filename + ".npz"), **model_dict)


def eval_fn(model):
    contexts = torch.tensor(data.raw_test_samples[0]).float()
    samples = np.zeros([contexts.shape[0], 10, data.dim[1]])
    for i in range(10):
        samples[:, i] = model.sample(contexts).detach().numpy()
    #print("context", contexts.shape, contexts)
    #print("samples", samples.shape, samples)
    return data.rewards_from_contexts(contexts.detach().numpy(), samples)

class ModelRecMod(RecorderModule):
    """Records current eim performance: Log Likelihood and if true (unnormalized) log density is provided the
    i-projection kl (plus constant)"""
    def __init__(self, train_samples, test_samples, true_log_density=None, eval_fn=None,
                 test_log_iters=50, save_log_iters=50):
        super().__init__()
        self._train_samples = train_samples if torch.is_tensor(train_samples) else (torch.tensor(train_samples)[0].float(), torch.tensor(train_samples)[1].float())
        self._test_samples = test_samples if torch.is_tensor(test_samples) else (torch.tensor(test_samples)[0].float(), torch.tensor(test_samples)[1].float())
        self._true_log_density = true_log_density
        self._eval_fn = eval_fn

        self._test_log_iters = test_log_iters
        self._save_log_iters = save_log_iters

        self._train_ll_list = []
        self._train_kl_list = []

        self._test_ll_list = []
        self._test_kl_list = []
        self._test_eval_list = []

        self.__num_iters = None
        self._colors = Colors()

    @property
    def _num_iters(self):
        assert self.__num_iters is not None, "Model Recorder not initialized properly"
        return self.__num_iters

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self.__num_iters = num_iters

    def _log_train(self, model):
        #print("self._train_samples", self._train_samples)
        if isinstance(self._train_samples,torch.Tensor):
            self._train_ll_list.append(torch.tensor(model.log_likelihood(self._train_samples)))
        else:
            self._train_ll_list.append(torch.tensor(model.log_likelihood(self._train_samples[0], self._train_samples[1])))
        log_str = "Training LL: " + str(self._train_ll_list[-1])
        if self._true_log_density is not None:
            if isinstance(self._train_samples, torch.Tensor):
                model_train_samples = model.sample(len(self._train_samples))
                self._train_kl_list.append(torch.mean(model.log_density(model_train_samples) -
                                                   self._true_log_density(model_train_samples)))
            else:
                model_train_samples = model.sample(self._train_samples[0])
                self._train_kl_list.append(torch.mean(model.log_density(self._train_samples[0], model_train_samples) -
                                                   self._true_log_density(self._train_samples[0], model_train_samples)))

            log_str += " KL (MC-Estimate): " + str(self._train_kl_list[-1])
        print(log_str)
        self._logger.info(log_str)
        if self._plot_realtime:
            self._recorder.handle_plot("Loss", self._plot)

    def _log_test(self, model, model_test_samples=None):
        if isinstance(self._test_samples, torch.Tensor):
            self._test_ll_list.append(torch.tensor(model.log_likelihood(self._test_samples)))
        else:
            self._test_ll_list.append(torch.tensor(model.log_likelihood(self._test_samples[0], self._test_samples[1])))

        log_str = "Test: Likelihood " + str(self._test_ll_list[-1])

        if self._true_log_density is not None:
            if isinstance(self._train_samples, torch.Tensor):
                model_test_samples = model.sample(len(self._test_samples))
                self._test_kl_list.append(torch.mean(model.log_density(model_test_samples) -
                                                   self._true_log_density(model_test_samples)))
            else:
                model_test_samples = model.sample(self._test_samples[0])
                self._test_kl_list.append(torch.mean(model.log_density(self._test_samples[0], model_test_samples) -
                                                   self._true_log_density(self._test_samples[0], model_test_samples)))
            log_str += " KL (MC-Estimate): " + str(self._test_kl_list[-1])
        if self._eval_fn is not None:
            self._test_eval_list.append(self._eval_fn(model))
            log_str += " Eval Loss: " + str(self._test_eval_list[-1])
        print(log_str)
        self._logger.info(log_str)

    def record(self, model, iteration):
        if self._save and (iteration % self._save_log_iters == 0):
            save_model(model, self._save_path, "modelAtIter{:05d}".format(iteration))
        self._log_train(model)
        if iteration % self._test_log_iters == 0:
            self._log_test(model)

    def _plot(self):
        plt.subplot(2 if self._true_log_density is not None else 1, 1, 1)
        plt.title("Train Log Likelihood")

        plt.plot(np.arange(0, len(self._train_ll_list)), np.array(self._train_ll_list))
        plt.xlim(0, self._num_iters)
        if self._true_log_density is not None:
            plt.subplot(2, 1, 2)
            plt.title("I-Projection KL (MC-Estimate)")
            plt.plot(np.arange(0, len(self._train_kl_list)), np.array(self._train_kl_list))
            plt.xlim((0, self._num_iters))
        plt.tight_layout()

    def finalize(self):
        if self._save:
            save_dict = {"ll_train": self._train_ll_list,
                         "ll_test": self._test_ll_list}
            if self._true_log_density is not None:
                save_dict["kl_train"] = self._train_kl_list
                save_dict["kl_test"] = self._test_kl_list
                print("_train_kl_list:", self._train_kl_list)
                print("_test_kl_list", self._test_kl_list)
            if self._eval_fn is not None:
                save_dict["eval_test"] = self._test_eval_list
            np.savez_compressed(os.path.join(self._save_path, "losses_raw.npz"), **save_dict)
            self._recorder.save_img("Loss", self._plot)

    def get_last_rec(self):
        res_dict = {"ll_train": self._train_ll_list[-1]}
        if self._true_log_density is not None:
            res_dict["kl_train"] = self._train_kl_list[-1]

        return res_dict

    @property
    def logger_name(self):
        return "Model"

class ModelRecModWithModelVis(ModelRecMod):
    """Superclass - Standard eim recording + eim visualization
    (actual visualization needs to be implemented individually depending on data/task)"""

    def record(self, model, iteration):
        super().record(model, iteration)
        if self._plot_realtime:
            plt_fn = lambda x: self._plot_model(x, title="Iteration {:5d}".format(iteration))
            self._recorder.handle_plot("Model", plt_fn, model)

    def _plot_model(self, model, title):
        raise NotImplementedError("Not Implemented")

    @staticmethod
    def _draw_2d_covariance(mean, covmatrix, chisquare_val=2.4477, return_raw=False, *args, **kwargs):
        (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covmatrix)
        phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

        a = chisquare_val * np.sqrt(largest_eigval)
        b = chisquare_val * np.sqrt(smallest_eigval)

        ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi))
        ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi))

        R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
        if return_raw:
            return mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1]
        else:
            return plt.plot(mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1], *args, **kwargs)

class ObstacleModelRecMod(ModelRecModWithModelVis):

    def __init__(self, obstacle_data, train_samples, test_samples, true_log_density=None, eval_fn=None,
                 test_log_iters=50, save_log_iters=50):
        super().__init__(train_samples, test_samples, true_log_density, eval_fn, test_log_iters, save_log_iters)
        self._data = obstacle_data

    def _plot_model(self, model, title):
        x_plt = np.arange(0, 1, 1e-2)
        color = Colors()
        contexts = torch.tensor(self._data.raw_test_samples[0][:10]).float()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            context = contexts[i:i + 1]
            plt.imshow(self._data.img_from_context(context[0]))
            lines = []
            for k, c in enumerate(model.components):
                m = (c.mean(context)[0] + 1) / 2
                cov = c.covar(context)[0]
                mx, my = m[::2], m[1::2]
                plt.scatter(200 * mx, 100 * my, c=color(k))
                for j in range(mx.shape[0]):
                    mean = np.array([mx[j], my[j]])
                    cov_j = cov[2 * j: 2 * (j + 1), 2 * j: 2 * (j + 1)]
                    plt_cx, plt_cy = self._draw_2d_covariance(mean, cov_j, 1, return_raw=True)
                    plt.plot(200 * plt_cx, 100 * plt_cy, c=color(k), linestyle="dotted", linewidth=2)
                for j in range(10):
                    s = np.array(c.sample(contexts[i:i + 1]))
                    spline = self._data.get_spline(s[0])
                    l, = plt.plot(200 * x_plt, 100 * spline(x_plt), c=color(k), linewidth=1)
                lines.append(l)
            for j in range(10):
                s = self._data.raw_test_samples[1][i, j]
                spline = self._data.get_spline(s)
                plt.plot(200 * x_plt, 100 * spline(x_plt), c=color(model.num_components), linewidth=1, linestyle="dashed")

            weights = model.gating_distribution.probabilities(context)[0]
            strs = ["{:.3f}".format(weights[i]) for i in range(model.num_components)]
            plt.legend(lines, strs, loc=1)
            plt.gca().set_axis_off()
            plt.gca().set_xlim(0, 200)
            plt.gca().set_ylim(0, 100)

def to_numpy(x):
    #print("x", x)
    if torch.is_tensor(x):
      return x.detach().numpy()
    else:
      return np.array(x)
def log_res(res, key_prefix):
    #print("res", res)
    num_iters, kl, entropy, add_text = [to_numpy(x) for x in res]
    #print("num_iters, kl, entropy, add_text", num_iters, kl, entropy, add_text)
    last_rec = {key_prefix + "_num_iterations": num_iters, key_prefix + "_kl": kl, key_prefix + "_entropy": entropy}
    log_string = "Updated for {:d} iterations. ".format(num_iters)
    log_string += "KL: {:.5f}. ".format(kl)
    log_string += "Entropy: {:.5f} ".format(entropy)
    log_string += str(add_text)
    return log_string, last_rec

class ComponentUpdateRecMod(RecorderModule):

    def __init__(self, plot, summarize=True):
        super().__init__()
        self._plot = plot
        self._last_rec = None
        self._summarize = summarize
        self._kls = None
        self._entropies = None
        self._num_iters = -1
        self._num_components = -1
        self._c = Colors()

    def initialize(self, recorder, plot_realtime, save, num_iters, num_components):
        super().initialize(recorder, plot_realtime, save)
        self._num_iters = num_iters
        self._num_components = num_components
        self._kls = [[] for _ in range(self._num_components)]
        self._entropies = [[] for _ in range(self._num_components)]

    def record(self, res_list):
        self._last_rec = {}
        for i, res in enumerate(res_list):
            cur_log_string, cur_last_rec = log_res(res, "component_{:d}".format(i))
            self._last_rec = {**self._last_rec, **cur_last_rec}
            self._logger.info("Component{:d}: ".format(i + 1) + cur_log_string)
            if not self._summarize:
                self._logger.info("Component{:d}: ".format(i + 1) + cur_log_string)
            if self._plot:
                self._kls[i].append(self._last_rec["component_{:d}_kl".format(i)])
                self._entropies[i].append(self._last_rec["component_{:d}_entropy".format(i)])
        if self._summarize:
            self._summarize_results(res_list)
        if self._plot:
            self._recorder.handle_plot("Component Update", self._plot_fn)

    def _summarize_results(self, res_list):
        fail_ct = 0
        for res in res_list:
            if "failed" in str(res[-1]).lower():
                fail_ct += 1
        num_updt = len(res_list)
        log_str = "{:d} components updated - {:d} successful".format(num_updt, num_updt - fail_ct)
        self._logger.info(log_str)
    def _plot_fn(self):
        plt.subplot(2, 1, 1)
        plt.title("Expected KL")
        for i in range(self._num_components):
            plt.plot(self._kls[i], c=self._c(i))
        plt.legend(["Component {:d}".format(i + 1) for i in range(self._num_components)])
        plt.xlim(0, self._num_iters)
        plt.subplot(2, 1, 2)
        plt.title("Expected Entropy")
        for i in range(self._num_components):
            plt.plot(self._entropies[i], c=self._c(i))
        plt.xlim(0, self._num_iters)
        plt.tight_layout()

    @property
    def logger_name(self):
        return "Component Update"

    def get_last_rec(self):
        assert self._last_rec is not None
        return self._last_rec

    def finalize(self):
        if self._plot:
            self._recorder.save_img("ComponentUpdates", self._plot_fn)

#############

class WeightUpdateRecMod(RecorderModule):

    def __init__(self, plot):
        super().__init__()
        self._last_rec = None
        self._plot = plot
        self._kls = []
        self._entropies = []
        self._num_iters = -1

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self._num_iters = num_iters

    def record(self, res):
        log_string, self._last_rec = log_res(res, "weights")
        self._logger.info(log_string)
        if self._plot:
            self._kls.append(self._last_rec["weights_kl"])
            self._entropies.append(self._last_rec["weights_entropy"])
            self._recorder.handle_plot("Weight Update", self._plot_fn)

    def _plot_fn(self):
        plt.subplot(2, 1, 1)
        plt.title("Expected KL")
        plt.plot(self._kls)
        plt.xlim(0, self._num_iters)
        plt.subplot(2, 1, 2)
        plt.title("Expected Entropy")
        plt.plot(self._entropies)
        plt.xlim(0, self._num_iters)
        plt.tight_layout()

    @property
    def logger_name(self):
        return "Weight Update"

    def get_last_rec(self):
        assert self._last_rec is not None
        return self._last_rec

    def finalize(self):
        if self._plot:
            self._recorder.save_img("WeightUpdates", self._plot_fn)


class DRERecMod(RecorderModule):
    """Records current Density Ratio Estimator performance - loss, accuracy and mean output for true and fake samples"""
    def __init__(self, true_samples, target_ld=None):
        super().__init__()
        if isinstance(true_samples, torch.Tensor):
            self._target_samples = true_samples.type(torch.float32)
        else:
            self._target_samples = [x.type(np.float32) for x in true_samples]
        self._steps = []
        self._estm_ikl = []
        self._loss = []
        self._acc = []
        self._true_mean = []
        self._fake_mean = []
        self.__num_iters = None
        self._target_ld = target_ld
        self._dre_type = None
        if self._target_ld is not None:
            self._dre_rmse = []
        self._conditional = not isinstance(true_samples, torch.Tensor)

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self.__num_iters = num_iters

    @property
    def _num_iters(self):
        assert self.__num_iters is not None, "Density Ratio Estimator Recorder not properly initialized"
        return self.__num_iters

    def _dre_rmse_fn(self, idx, dre, model, samples):
        mld = model.log_density(samples)
        return torch.squeeze(dre(samples, idx)) - (self._target_ld(samples) - mld)

    def record(self, dre, model, iteration, steps):
        if iteration == 0 and self._target_ld is not None:
            for _ in model.components:
                self._dre_rmse.append([])
        estm_ikl, loss, acc, true_mean, fake_mean = [torch.tensor(x) for x in dre.eval(self._target_samples, model)]
        log_str = "Density Ratio Estimator ran for " + str(steps) + " steps. "
        log_str += "Loss {:.4f} ".format(loss)
        log_str += "Estimated IKL: {:.4f} ".format(estm_ikl)
        log_str += "Accuracy: {:.4f} ".format(acc)
        log_str += "True Mean IKL: {:.4f} ".format(true_mean)
        log_str += "Fake Mean IKL: {:.4f} ".format(fake_mean)

        if self._target_ld is not None:
            all_errs = []
            for i, c in enumerate(model.components):
                samples = c.sample(1000)
                errs = torch.tensor(self._dre_rmse_fn(i, dre, model, samples))
                all_errs.append(errs)
                self._dre_rmse[i].append(torch.sqrt(torch.mean(errs**2)))
                log_str += "Component {:d}: DRE RMSE: {:.4f} ".format(i, self._dre_rmse[i][-1])
            self._recorder.handle_plot("Err Hist", self._plot_hist, all_errs)
        self._logger.info(log_str)

        self._steps.append(steps)
        self._estm_ikl.append(estm_ikl.item())
        self._loss.append(loss.item())
        self._acc.append(acc.item())
        self._true_mean.append(true_mean.item())
        self._fake_mean.append(fake_mean.item())
        #print("self._estm_ikl", self._estm_ikl)
        print("self._loss", self._loss)
        print("self._acc", self._acc)
        # print("self._true_mean", self._true_mean)
        # print("self._fake_mean", self._fake_mean)
        if self._plot_realtime:
            self._recorder.handle_plot("Discriminator Evaluation", self._plot)

    def finalize(self):
        if self._save:
            save_dict = {"estm_ikl": self._estm_ikl, "loss": self._loss, "acc": self._acc, "true_mean":
                         self._true_mean, "fake_mean": self._fake_mean}
            np.savez_compressed(os.path.join(self._save_path, "DensityRatioEstimatorEval_raw.npz"), **save_dict)
            self._recorder.save_img("DensityRatioEstimatorEval", self._plot)

    def _subplot(self, i, title, data_list, data_list2=None, y_lim=None):
        plt.subplot(5 if self._target_ld is not None else 4, 1, i)
        plt.title(title)
        print("i, title, data_list", i, title, data_list)
        plt.plot(np.array(data_list))
        if data_list2 is not None:
            plt.plot(np.array(data_list2))
        plt.xlim(0, self._num_iters)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])

    def _plot_hist(self, errs):
        for i, err in enumerate(errs):
            plt.subplot(len(errs), 1, i+1)
            plt.hist(err, density=True, bins=25)
        plt.tight_layout()

    def _plot(self):
        self._subplot(1, "Estimated I-Projection", self._estm_ikl)
        self._subplot(2, "Density Ratio Estimator Loss", self._loss)
        self._subplot(3, "Density Ratio Estimator Accuracy", self._acc, y_lim=(-0.1, 1.1))
        self._subplot(4, "", self._true_mean, self._fake_mean, y_lim=(-0.1, 1.1))
        plt.legend(["Mean output true samples", "Mean output fake samples"])
        if self._target_ld is not None:
            plt.subplot(5, 1, 5)
            for i in range(len(self._dre_rmse)):
                self._subplot(5, "DRE RMSE", self._dre_rmse[i])
        plt.tight_layout()

    def get_last_rec(self):
        lr = {"steps": self._steps[-1], "estimated_ikl": self._estm_ikl[-1], "dre_loss": self._loss[-1],
              "accuracy": self._acc[-1], "true_mean": self._true_mean[-1], "fake_mean": self._fake_mean[-1]}
        if self._target_ld is not None:
            for i in range(len(self._dre_rmse)):
                lr["dre_rmse{:d}".format(i)] = self._dre_rmse[i][-1]
        return lr

    @property
    def logger_name(self):
        return "DRE"

"""Recording"""
recorder_dict = {
    RecorderKeys.TRAIN_ITER: TrainIterationRecMod(),
    RecorderKeys.INITIAL: ConfigInitialRecMod(),
    RecorderKeys.MODEL: ObstacleModelRecMod(data,
                                        train_samples=data.train_samples,
                                        test_samples=data.test_samples,
                                        test_log_iters=1,
                                        eval_fn=eval_fn,
                                        save_log_iters=50),
    RecorderKeys.DRE: DRERecMod(torch.from_numpy(np.asarray(data.train_samples))),
    RecorderKeys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=True, summarize=False)}
if num_components > 1:
    recorder_dict[RecorderKeys.WEIGHTS_UPDATE] = WeightUpdateRecMod(plot=True)


recorder = Recorder(recorder_dict, plot_realtime=plot_realtime, save=plot_save, save_path="rec")


"""Configure EIM"""

config = ConditionalMixtureEIM.get_default_config()
config.train_epochs = 1000
config.num_components = num_components

config.components_net_hidden_layers = [64, 64]
config.components_batch_size = 1000
config.components_num_epochs = 10
config.components_net_reg_loss_fact = 0.0
config.components_net_drop_prob = 0.0

config.gating_net_hidden_layers = [64, 64]
config.gating_batch_size = 1000
config.gating_num_epochs = 10
config.gating_net_reg_loss_fact = 0.0
config.gating_net_drop_prob = 0.0

config.dre_reg_loss_fact = 0.0005
config.dre_early_stopping = True
config.dre_drop_prob = 0.0
config.dre_num_iters =  50
config.dre_batch_size = 1000
config.dre_hidden_layers = [128, 128, 128]

"""Build and Run EIM"""
model = ConditionalMixtureEIM(config, train_samples=data.train_samples, seed=42 * 7, recorder=recorder, val_samples=data.val_samples)
model.train_emm()
