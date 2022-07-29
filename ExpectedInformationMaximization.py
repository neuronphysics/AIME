from multiprocessing import context
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import normal
import numpy as np
import shutil
import yaml
import os
from planner_regulizer import Split
from tqdm import tqdm
import warnings
import logging
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
        elif activation=="leaky_relu":
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
#### Distributions:

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

def accuracy(p_outputs, q_outputs):
    p_prob = torch.nn.Sigmoid(p_outputs)
    q_prob = torch.nn.Sigmoid(q_outputs)
    return torch.mean(torch.cat([torch.greater_equal(p_prob, 0.5), torch.lt(q_prob, 0.5)], 0).type(torch.float32))


def logistic_regression_loss(p_outputs, q_outputs):
    return - torch.mean(torch.log(torch.nn.Sigmoid(p_outputs) + 1e-12)) \
           - torch.mean(torch.log(1 - torch.nn.Sigmoid(q_outputs) + 1e-12))

  

def gaussian_log_density(samples, means, chol_covars):
    covar_logdet = 2 *torch.sum(torch.log(torch.diagonal(chol_covars, dim1=-2, dim2=-1)+1e-15),dim=-1)
    diff = torch.unsqueeze(samples - means, -1)
    exp_term = torch.sum(torch.square(torch.linalg.solve_triangular(chol_covars, diff)), (-2, -1))
    return - 0.5 * (samples.size()[-1].type( exp_term.dtype) * torch.log(2 * torch.tensor(np.pi, requires_grad=False)) + covar_logdet + exp_term)

def gaussian_density(samples, means, chol_covars):
    return torch.exp(ConditionalGaussian.gaussian_log_density(samples, means, chol_covars))

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class ConditionalGaussian(nn.Module):

    def __init__(self, context_dim, sample_dim, hidden_dict, seed, trainable=True, weight_path=None):
        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._seed = seed
        self._hidden_dict = hidden_dict

        self.trainable = trainable

        self._model = self._build()
        if self._weight_path is not None:
           self._model.load_state_dict(torch.load(weight_path))
            

    def _build(self):
        self._hidden_net, self.regularizer = build_dense_network(self._context_dim, output_dim=-1, output_activation=None,
                                                  params=self._hidden_dict, with_output_layer=False)

        self._mean_t = nn.Linear(self._hidden_net._modules[next(iter(next(reversed(self._hidden_net._modules.items()))))].out_features, self._sample_dim)
        self._chol_covar_raw = nn.Linear(self._hidden_net._modules[next(iter(next(reversed(self._hidden_net._modules.items()))))].out_features, self._sample_dim ** 2)
        #based on this  shorturl.at/pTVZ3
        self._chol_covar =  LambdaLayer( lambda x:self._create_chol(x))(self._chol_covar_raw)
    
        

    def forward(self, contexts):
        h = nn.ReLu(self._hidden_net(contexts))
        mean=self._mean_t(h)
        chol_covar =self._chol_covar(h)
        covar= torch.matmult(torch.transpose(chol_covar, 0, 1),chol_covar)
        return mean, covar, chol_covar 
    
    def mean(self, contexts):
        return self._model(contexts)[0]

    def covar(self, contexts):
        return self._model(contexts)[1]
           

    def _create_chol(self, chol_raw):
        #tensorflow.linalg.band_part(input, num_lower, num_upper, name=None) 
        #num_lower: A Tensor. The number of subdiagonals to preserve. Negative values ​​preserve all lower triangles
        #num_upper: A Tensor. The number of superdiagonals to preserve. Negative values ​​preserve all upper triangles.
        samples =torch.triu(torch.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]).t(), diagonal=0).t()
        return samples.fill_diagonal_(torch.exp(torch.diagonal(samples, dim1=-2, dim2=-1))+ 1e-12)
        

    def sample(self, contexts):
        mean, _, chol_covar = self._model(contexts)
        torch.random.manual_seed(self._seed)
        eps = torch.normal(mean=0,std=1,size=(torch.shape(mean)[0], torch.shape(mean)[1], 1))
        return mean + torch.reshape(torch.matmul(chol_covar, eps), torch.shape(mean))

    def log_density(self, contexts, samples):
        mean,_, chol_covar = self._model(contexts)
        return gaussian_log_density(samples, mean, chol_covar)

    def density(self, contexts, samples):
        return torch.exp(self.log_density(contexts, samples))

    @staticmethod
    def expected_entropy(self, contexts):
        _, _, chol_covars = self._model(contexts)
        return 0.5 * (self._sample_dim * np.log(torch.tensor(2 * np.e * np.pi, requires_grad=False)) + torch.mean(self._covar_logdets(chol_covars)))
    
    @staticmethod
    def entropies(self, contexts):
        _, _, chol_covars = self._model(contexts)
        return 0.5 * (self._sample_dim * torch.log(torch.tensor(2 * np.e * np.pi, requires_grad=False)) + self._covar_logdets(chol_covars))
    
    @staticmethod
    def _covar_logdets(self, chol_covars):
        return 2 * torch.sum(torch.log(torch.diagonal(chol_covars, dim1=-2, dim2=-1) + 1e-12), dim=-1)

    @staticmethod
    def kls(self, contexts, other_means, other_chol_covars):
        means, _, chol_covars = self._model(contexts)
        kl = self._covar_logdets(other_chol_covars) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += torch.sum(torch.square(torch.linalg.solve_triangular(other_chol_covars, chol_covars)), (-2, -1))
        diff = torch.unsqueeze(other_means - means, -1)
        kl += torch.sum(torch.square(torch.linalg.solve_triangular(other_chol_covars, diff)), (-2, -1))
        return 0.5 * kl

    @staticmethod
    def kls_other_chol_inv(self, contexts, other_means, other_chol_inv):
        means, _, chol_covars = self._model(contexts)
        kl = - self._covar_logdets(other_chol_inv) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += torch.sum(torch.square(torch.matmul(other_chol_inv, chol_covars)),( -2, -1))
        diff = torch.unsqueeze(other_means - means, -1)
        kl += torch.sum(torch.square(torch.matmul(other_chol_inv, diff)), (-2, -1))
        return 0.5 * kl

    @staticmethod
    def expected_kl(self, contexts, other_means, other_chol_covars):
        return torch.mean(self.kls(contexts, other_means, other_chol_covars))

    @staticmethod
    def log_likelihood(self, contexts, samples):
        return torch.mean(self.log_density(contexts, samples))

    def conditional_params(self, contexts):
        #get mean and variance parameters of the network separately 
        #shorturl.at/amWX4
        return map(nn.Parameter, self.build(contexts))

    @property
    def trainable_variables(self):
        return filter(lambda p: p.requires_grad, self._model.parameters())

    @property
    def sample_dim(self):
        return self._sample_dim

    def save_model_params(self, filepath):
        torch.save(self._model.state_dict(), filepath )

class Softmax(nn.Module):

    def __init__(self, context_dim, z_dim, hidden_dict, seed, trainable=True, weight_path=None):
        self._context_dim = context_dim
        self._z_dim = z_dim
        self._seed = seed
        self._hidden_dict = hidden_dict
        self._trainable = trainable

        self._logit_net = build_dense_network(self._context_dim, self._z_dim, output_activation=None,
                                                 params=self._hidden_dict)

        if self._weight_path is not None:
           self._logit_net.load_state_dict(torch.load(weight_path))
        

    def logits(self, contexts):
        return self._logit_net(contexts)

    def probabilities(self, contexts):
        return nn.Softmax(self.logits(contexts))

    def log_probabilities(self, contexts):
        return torch.log(self.probabilities(contexts) + 1e-12)

    def expected_entropy(self, contexts):
        p = self.probabilities(contexts)
        return - torch.mean(torch.sum(p * torch.log(p + 1e-12), -1))

    def expected_kl(self, contexts, other_probabilities):
        p = self.probabilities(contexts)
        return \
            torch.mean(torch.sum(p * (torch.log(p + 1e-12) - torch.log(other_probabilities + 1e-12)), -1))

    def sample(self, contexts):
        p = self.probabilities(contexts)
        thresholds = torch.cumsum(p, dim=-1)
        # ensure the last threshold is always exactly one - it can be slightly smaller due to numerical inaccuracies
        # of cumsum, causing problems in extremely rare cases if a "n" is samples that's larger than the last threshold
        thresholds = torch.cat([thresholds[..., :-1], torch.ones([torch.size(thresholds)[0], 1])], -1)
        torch.random.manual_seed(self._seed)
        n=torch.distributions.uniform.Uniform(0.0,1.0).rsample((torch.size(thresholds)[0], 1))
        idx = torch.where(torch.lt(n, thresholds), torch.range(self._z_dim) * torch.ones(thresholds.shape, dtype=torch.int32),
                       self._z_dim * torch.ones(thresholds.shape, dtype=torch.int32))
        return torch.min(idx, -1)

    @property
    def trainable_variables(self):
        return filter(lambda p: p.requires_grad, self._logit_net.parameters())

    def save_model_params(self, filepath):
        torch.save(self._logit_net.state_dict(), filepath )

class GaussianEMM:

    def __init__(self, context_dim, sample_dim, number_of_components, component_hidden_dict, gating_hidden_dict,
                 seed=0, trainable=True, weight_path=None):

        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._number_of_components = number_of_components

        self._mixture_hidden_dict = gating_hidden_dict
        self._component_hidden_dict = component_hidden_dict

        wp = None if weight_path is None else os.path.join(weight_path, self._mixture_params_file_name())
        self._gating_distribution = Softmax(self._context_dim, self._number_of_components, gating_hidden_dict,
                                            seed=seed, trainable=trainable, weight_path=wp)

        self._components = []
        for i in range(number_of_components):
            h_dict = component_hidden_dict[i] if isinstance(component_hidden_dict, list) else component_hidden_dict
            wp = None if weight_path is None else os.path.join(weight_path, self._component_params_file_name(i))
            c = ConditionalGaussian(self._context_dim, self._sample_dim, h_dict,
                                    trainable=trainable, seed=seed, weight_path=wp)
            self._components.append(c)

        self.trainable_variables = self._gating_distribution.trainable_variables
        for c in self._components:
            self.trainable_variables += c.trainable_variables

    def density(self, contexts, samples):
        p = self._gating_distribution.probabilities(contexts)
        density = p[:, 0] * self._components[0].density(contexts, samples)
        for i in range(1, self._number_of_components):
            density += p[:, i] * self._components[i].density(contexts, samples)
        return density

    def log_density(self, contexts, samples):
        return torch.log(self.density(contexts, samples) + 1e-12)

    def log_likelihood(self, contexts, samples):
        return torch.mean(self.log_density(contexts, samples))

    def sample(self, contexts):
        modes = self._gating_distribution.sample(contexts)
        samples = torch.zeros([contexts.shape[0], self._sample_dim])
        for i in range(self._number_of_components):
            idx = (modes == i)
            if torch.any(idx):
                samples[idx] = self._components[i].sample(contexts[idx])
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




class DensityRatioEstimator:

    def __init__(self, target_train_samples, hidden_params, early_stopping=False, target_val_samples=None,
                 conditional_model=False):

        self._early_stopping = early_stopping
        self._conditional_model = conditional_model

        if self._conditional_model:
            self._train_contexts = target_train_samples[0].type(torch.float32)
            self._target_train_samples = torch.cat(target_train_samples, -1).type(torch.float32)
        else:
            self._target_train_samples = target_train_samples.type(torch.float32)

        if self._early_stopping:
            assert target_val_samples is not None, \
                "For early stopping validation data needs to be provided via target_val_samples"
            if self._conditional_model:
                self._val_contexts = target_val_samples[0].type(torch.float32)
                self._target_val_samples = torch.cat(target_val_samples, -1).type(torch.float32)
            else:
                self._target_val_samples = target_val_samples.type(torch.float32)
        self.hidden_params=hidden_params
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model=self._build()
        
    def _build(self):
        input_dim = self._target_train_samples.shape[-1]
    
        self._ldre_net = build_dense_network(input_dim=input_dim, output_dim=1,
                                             output_activation="linear", params=self.hidden_params)

        self._p_samples = nn.Linear(input_dim,input_dim)
        self._q_samples = nn.Linear(input_dim,input_dim)
        
    def forward(self,x):
        p = self._p_samples(x)
        q = self._q_samples(x)
        combined = torch.cat((p.view(p.size(0), -1),
                              q.view(q.size(0), -1)), dim=1)
        self._split_layers = Split(
         self._ldre_net[-1],
         parts=2,
        )
        p_output, q_output =self._split_layers(combined)
        return p_output, q_output 
    
    def train(self, pre_epoch=30):
        model = self._model.to(self.device)

        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        epoch_bar = tqdm(range(pre_epoch))
        for _ in epoch_bar:
            L = 0
            for batchidx, (x, y) in enumerate(self.train_loader):
                # Send data and labels to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Zero the optimizer
                optimizer.zero_grad()

                # Forward pass
                self._train_model_p_outputs, self._train_model_q_outputs= self.model(x)

                # Calculate loss score
                loss = logistic_regression_loss(self._train_model_p_outputs, self._train_model_q_outputs)
                L   += loss.detach().cpu().numpy()
                # Back prop
                
                loss.backward()
                optimizer.step()    
                self._acc = accuracy(self._train_model_p_outputs, self._train_model_q_outputs)
            epoch_bar.write('\nL={:.4f} accuracy={:.4f}}\n'.format(L / len(self.train_loader), self._acc))
        

    def __call__(self, samples):
        return self._ldre_net(samples)

    def eval(self, target_samples, model):
        if self._conditional_model:
            model_samples = torch.cat([target_samples[0], model.sample(target_samples[0])], dim=-1)
            target_samples = torch.cat(target_samples, dim=-1)
        else:
            model_samples = model.sample(self._target_train_samples.shape[0])
        target_ldre = self(target_samples)
        model_ldre = self(model_samples)
        target_prob = nn.Sigmoid(target_ldre)
        model_prob = nn.Sigmoid(model_ldre)

        ikl_estem = torch.mean(- model_ldre)
        acc = accuracy(target_ldre, model_ldre)
        bce = logistic_regression_loss(target_ldre, model_ldre)
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

    def __init__(self, config: ConfigDict, train_samples: np.ndarray, recorder: Recorder,
                 val_samples: np.ndarray = None, seed: int = 0):
        # supress tensorflow casting warnings
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        self.c = config
        self.c.finalize_modifying()

        self._recorder = recorder

        self._context_dim = train_samples[0].shape[-1]
        self._sample_dim = train_samples[1].shape[-1]
        self._train_contexts = tf.constant(train_samples[0], dtype=tf.float32)

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

        self._model = GaussianEMM(self._context_dim, self._sample_dim, self.c.num_components,
                                  c_net_hidden_dict, g_net_hidden_dict, seed=seed)
        self._c_opts = [k.optimizers.Adam(self.c.components_learning_rate, 0.5) for _ in self._model.components]
        self._g_opt = k.optimizers.Adam(self.c.gating_learning_rate, 0.5)

        dre_params = {NetworkKeys.NUM_UNITS: self.c.dre_hidden_layers,
                      NetworkKeys.ACTIVATION: k.activations.relu,
                      NetworkKeys.DROP_PROB: self.c.dre_drop_prob,
                      NetworkKeys.L2_REG_FACT: self.c.dre_reg_loss_fact}
        self._dre = DensityRatioEstimator(target_train_samples=train_samples,
                                          hidden_params=dre_params,
                                          early_stopping=self.c.dre_early_stopping, target_val_samples=val_samples,
                                          conditional_model=True)

        self._recorder.initialize_module(RecorderKeys.INITIAL)
        self._recorder(RecorderKeys.INITIAL, "Nonlinear Conditional EIM - Reparametrization", config)
        self._recorder.initialize_module(RecorderKeys.MODEL, self.c.train_epochs)
        self._recorder.initialize_module(RecorderKeys.WEIGHTS_UPDATE, self.c.train_epochs)
        self._recorder.initialize_module(RecorderKeys.COMPONENT_UPDATE, self.c.train_epochs, self.c.num_components)
        self._recorder.initialize_module(RecorderKeys.DRE, self.c.train_epochs)

    def train(self):
        for i in range(self.c.train_epochs):
            self._recorder(RecorderKeys.TRAIN_ITER, i)
            self.train_iter(i)

    #   extra function to allow running from cluster work
    def train_iter(self, i):

        dre_steps, loss, acc = self._dre.train(self._model, self.c.dre_batch_size, self.c.dre_num_iters)
        self._recorder(RecorderKeys.DRE, self._dre, self._model, i, dre_steps)

        if self._model.num_components > 1:
            w_res = self.update_gating()
            self._recorder(RecorderKeys.WEIGHTS_UPDATE, w_res)

        c_res = self.update_components()
        self._recorder(RecorderKeys.COMPONENT_UPDATE, c_res)

        self._recorder(RecorderKeys.MODEL, self._model, i)

    """component update"""
    def update_components(self):
        importance_weights = self._model.gating_distribution.probabilities(self._train_contexts)
        importance_weights = importance_weights / torch.sum(importance_weights, dim=0, keepdims=True)

        old_means, old_chol_covars = self._model.get_component_parameters(self._train_contexts)

        rhs = torch.eye(torch.size(old_means)[-1], batch_shape=torch.size(old_chol_covars)[:-2])
        stab_fact = 1e-20
        old_chol_inv = torch.linalg.solve_triangular(old_chol_covars + stab_fact * rhs, rhs)

        for i in range(self.c.components_num_epochs):
            self._components_train_step(importance_weights, old_means, old_chol_inv)

        res_list = []
        for i, c in enumerate(self._model.components):
            expected_entropy = torch.sum(importance_weights[:, i] * c.entropies(self._train_contexts))
            kls = c.kls_other_chol_inv(self._train_contexts, old_means[:, i], old_chol_inv[:, i])
            expected_kl = torch.sum(importance_weights[:, i] * kls)
            res_list.append((self.c.components_num_epochs, expected_kl, expected_entropy, ""))
        return res_list

    @tf.function
    def _components_train_step(self, importance_weights, old_means, old_chol_precisions):
        for i in range(self._model.num_components):
            dt = (self._train_contexts, importance_weights[:, i], old_means, old_chol_precisions)
            data = tf.data.Dataset.from_tensor_slices(dt)
            data = data.shuffle(self._train_contexts.shape[0]).batch(self.c.components_batch_size)

            for context_batch, iw_batch, old_means_batch, old_chol_precisions_batch in data:
                iw_batch = iw_batch / torch.sum(iw_batch)
                with tf.GradientTape() as tape:
                    samples = self._model.components[i].sample(context_batch)
                    losses = - torch.squeeze(self._dre(torch.cat([context_batch, samples], dim=-1)))
                    kls = self._model.components[i].kls_other_chol_inv(context_batch, old_means_batch[:, i],
                                                                  old_chol_precisions_batch[:, i])
                    loss = torch.mean(iw_batch * (losses + kls))
                gradients = tape.gradient(loss, self._model.components[i].trainable_variables)
                self._c_opts[i].apply_gradients(zip(gradients, self._model.components[i].trainable_variables))


    """gating update"""
    def update_gating(self):
        old_probs = self._model.gating_distribution.probabilities(self._train_contexts)
        for i in range(self.c.gating_num_epochs):
            self._gating_train_step(old_probs)

        expected_entropy = self._model.gating_distribution.expected_entropy(self._train_contexts)
        expected_kl = self._model.gating_distribution.expected_kl(self._train_contexts, old_probs)

        return i + 1, expected_kl, expected_entropy, ""

    @tf.function
    def _gating_train_step(self, old_probs):
        losses = []
        for i in range(self.c.num_components):
            samples = self._model.components[i].sample(self._train_contexts)
            losses.append(- self._dre(torch.cat([self._train_contexts, samples], dim=-1)))

        losses = torch.cat(losses, dim=1)
        data = tf.data.Dataset.from_tensor_slices((self._train_contexts, losses, old_probs))
        data = data.shuffle(self._train_contexts.shape[0]).batch(self.c.gating_batch_size)
        for context_batch, losses_batch, old_probs_batch in data:
            with tf.GradientTape() as tape:
                probabilities = self._model.gating_distribution.probabilities(context_batch)
                kl = self._model.gating_distribution.expected_kl(context_batch, old_probs_batch)
                loss = torch.sum(torch.mean(probabilities * losses_batch, 0)) + kl
            gradients = tape.gradient(loss, self._model.gating_distribution.trainable_variables)
            self._g_opt.apply_gradients(zip(gradients, self._model.gating_distribution.trainable_variables))

    @property
    def model(self):
        return self._model
