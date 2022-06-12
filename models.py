# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F

from torch.distributions import constraints

import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
def hidden_size_extract(kwargs, name, delete_from_dict=False):
    if name not in kwargs:
        hidden_size = []
        for i in range(0, 6):
            key = name + '_%d' % i
            if key in kwargs and kwargs[key] != 0:
                hidden_size.append(kwargs[key])

                if delete_from_dict:
                    kwargs.pop(key)
    else:
        hidden_size = kwargs[name].copy()

        if delete_from_dict:
            kwargs.pop(name)

    return hidden_size


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

def bottle_two_output(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y1, y2 = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y1_size = y1.size()
  y2_size = y2.size()
  return y1.view(x_sizes[0][0], x_sizes[0][1], *y1_size[1:]), y2.view(x_sizes[0][0], x_sizes[0][1], *y2_size[1:])

class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  @jit.script_method
  def forward(self, state):
    hidden = self.fc1(state)  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation


def ObservationModel(symbolic, observation_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, -1, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(state_size, embedding_size, activation_function)


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, state_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.state_size = state_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
    self.fc_mean = nn.Linear(embedding_size, state_size)
    self.fc_std = nn.Linear(embedding_size, state_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    embedding = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    latent_mean = self.fc_mean(embedding)
    latent_std = F.softplus(self.fc_std(embedding))
    return latent_mean, latent_std


def Encoder(symbolic, observation_size, embedding_size, state_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, state_size, activation_function)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, device, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(GPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = ConstantMean()
        elif mean_type == 'zero':
            self.mean_module = ZeroMean()
        else:
            self.mean_module = LinearMean(input_dims)
        
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))




class DeepGaussianProcesses(DeepGP):
    def __init__(self, input_size, output_size, device, num_inducing=30, **kwargs):
        # pass hidden layer sizes as separate arguments as well as array
        for k, v in kwargs.items():
            if k=='mean_type':
                setattr(self, k, v)
        hidden_size = hidden_size_extract(kwargs, 'hidden_size')
        
        self.hidden_size = hidden_size
        self.hidden_size.append(output_size)
        
        self.device = device
        first_layer = DGPHiddenLayer(
                      input_dims=input_size,
                      output_dims=self.hidden_size[0],
                      device = self.device,
                      num_inducing=num_inducing,
                      mean_type=self.mean_type
                      )

        # variable count of hidden layers and neurons

        if self.mean_type=='linear' or self.mean_type=='constant':
            hidden_layers = nn.ModuleList(
              [ 
              
                  DGPHiddenLayer(
                     input_dims=self.hidden_size[i],
                     output_dims=self.hidden_size[i + 1],
                     device = self.device,
                     num_inducing=num_inducing,
                     mean_type='constant'
                  )
                  for i in range(len(self.hidden_size) - 1)
              ])
        else:
            hidden_layers = nn.ModuleList(
              [
                
                  DGPHiddenLayer(
                     input_dims=self.hidden_size[i],
                     output_dims=self.hidden_size[i + 1],
                     device = self.device,
                     num_inducing=num_inducing,
                     mean_type='zero'
                  )
                  for i in range(len(self.hidden_size) - 1)
              ])

        super().__init__()

        self.first_layer = first_layer
        self.hidden_layers = hidden_layers
        self.likelihood = GaussianLikelihood()
        self.to(device=self.device)

    def forward(self, inputs):
        out = self.first_layer(inputs, are_samples=True)
        for hidden in self.hidden_layers:
            out = hidden(out)
        return out
        
    #def predict(self, test_loader):
    #    """original predict function"""
    #    with torch.no_grad():
    #        mus = []
    #        variances = []
    #        lls = []
    #        gts = []
    #        for x_batch, y_batch in test_loader:
    #            preds = self.likelihood(self(x_batch))
    #            mus.append(preds.mean)
    #            variances.append(preds.variance)
    #            gts.append(y_batch)
    #            lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))
    #    return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1), torch.cat(gts, dim=-1)

    def predict(self, x):
        with torch.no_grad():
            mus = []
            variances = []
            samples = []
            preds = self.likelihood(self(x))
            mus.append(preds.mean)
            variances.append(preds.variance)
            #Alternative: not sure about the size of input for sample???
            #mvn = MultivariateNormal(preds.mean,preds.variance)
            #x = mvn.sample(sample_shape=torch.Size([self.dim]))
            #x = x.t()
            samples.append(preds.sample())           
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(samples, dim=-1)

    def retrieve_all_hyperparameter(self):
        all_sigma2 = []
        all_sigma2.append(self.first_layer.get_sigma2())
        for h in self.hidden_layers:
            all_sigma2.append(h.get_sigma2())
       

        all_lengthscale = []
        all_lengthscale.append(self.first_layer.get_lengthscale())
        for h in self.hidden_layers:
            all_lengthscale.append(h.get_lengthscale())

        return all_sigma2, all_lengthscale

    def get_K_1(self, x):
        return self.first_layer.covar_module(x).evaluate().data

def compute_EZ_trajectory(K, m, sigma2s, lengthscales):
    def recur(x, sigma2, lengthscale):
        return 2 * (1. - 1. / np.power(1 + x / lengthscale, m / 2.))

    x = K
    EZ = []
    for sigma2, lengthscale in zip(sigma2s, lengthscales):
        temp = recur(x, sigma2, lengthscale)
        EZ.append(temp)
        x = temp
    return EZ


class TransitionGP(DeepGaussianProcesses):
    #transition probability  P(z_{t}|x_{t-k},...x_{t-1})=N(z_{t}|mu_x,sigma_x)---->T(x_{t-k},...x_{t-1})=z_{t}; x_{.}=[z_{.},a_{.}]
    def __init__(self, latent_size, action_size, lagging_size, device, hidden_size=[100], mean_type= 'linear'):
        input_size = (latent_size+action_size)*lagging_size
        super(TransitionGP, self).__init__(input_size=input_size, output_size=latent_size, device=device, hidden_size=hidden_size, mean_type=mean_type) 
      
class PolicyGP(DeepGaussianProcesses):
    #policy P(z_{t},...z_{t-k})=a(t)--- input z:latent_space---> output a: action
    def __init__(self, latent_size, action_size, lagging_size, device, hidden_size=[50], mean_type= 'constant'):
        input_size =latent_size*lagging_size
        super(PolicyGP, self).__init__(input_size=input_size, output_size= action_size, device=device, hidden_size=hidden_size, mean_type=mean_type)
     
class RewardGP(DeepGaussianProcesses):
    def __init__(self, latent_size, action_size, lagging_size,  device, hidden_size=[100], mean_type= 'zero'):
       input_size =(latent_size+action_size)*lagging_size
       super(RewardGP, self).__init__(input_size=input_size, output_size= 1, device=device, hidden_size=hidden_size, mean_type=mean_type)


class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_size, device, num_mixture_samples=1):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_size = lagging_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_module = TransitionGP(latent_size, action_size, lagging_size, device).to(device=device) 
        self.policy_module = PolicyGP(latent_size, action_size, lagging_size, device).to(device=device) 
        self.reward_gp = RewardGP(latent_size, action_size, lagging_size, device).to(device=device)
        self.num_mixture_samples = num_mixture_samples
        self.likelihood = GaussianLikelihood()
    
    def forward(self, init_states, actions):
        # need to stack actions and latent vectors together (also reshape so that the lagging length dimension is stacked as well)
        init_states = init_states.reshape((init_states.size(0), init_states.size(1), -1))
        actions = actions.reshape((actions.size(0), actions.size(1), -1))
        z_hat = torch.cat([init_states, actions], dim=-1)

        lagging_actions = actions
        lagging_states = init_states
        for i in range(self.horizon_size):
            # policy distribution
            a, a_v, a_s = self.policy_module.predict(lagging_states)
            print(f"action mean {a}, variance {a_v}, sample {a_s}")
            lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
            z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
            # transition distribution
            z, z_v, z_s = self.transition_module.predict(z_hat)
            print(f"latent state mean {z}, variance {z_v}, sample {z_s}")
            lagging_states = torch.cat([lagging_states[..., self.latent_size:], z], dim=-1)
        
        # last policy in the horizon
        a, _, _ = self.policy_module.predict(lagging_states)
        lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
        z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
        # output the final reward
        rewards, _, _ = self.reward_gp(z_hat)
        return rewards
