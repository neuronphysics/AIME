# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F

from torch.distributions import constraints
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, SpectralMixtureKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import Interval, GreaterThan, Positive, LessThan

#import debugpy
#debugpy.listen(5678)
#print("Waiting for debugger....")
#debugpy.wait_for_client()
#print("Attached!")


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
  
  def __init__(self, state_size, embedding_size, activation_function='relu', input_size=64):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(state_size, embedding_size)
    self.input_size = input_size
    if input_size == 32:
      self.conv1 = nn.ConvTranspose2d(embedding_size, 64, 6, stride=2)
      self.conv2 = nn.ConvTranspose2d(64, 32, 5, stride=2)
      self.conv3 = nn.ConvTranspose2d(32, 3, 4, stride=2)
      self.conv4 = nn.Identity()
    else:
      self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
      self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
      self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
      self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  @jit.script_method
  def forward(self, state):
    hidden = self.fc1(state)  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    #print("hidden 1 decoder", hidden.shape)
    hidden = self.act_fn(self.conv2(hidden))
    #print("hidden 2 decoder", hidden.shape)
    observation = self.act_fn(self.conv3(hidden))
    #print("hidden 3 decoder", observation.shape)
    observation = self.conv4(observation)
    return observation


def ObservationModel(symbolic, observation_size, state_size, embedding_size, activation_function='relu', input_size=64):
  if symbolic:
    return SymbolicObservationModel(observation_size, -1, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(state_size, embedding_size, activation_function, input_size)


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
  
  def __init__(self, embedding_size, state_size, activation_function='relu', input_size=64):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.state_size = state_size
    self.input_size = input_size
    #self.transform = Resize((input_size, input_size))
    if input_size == 32:
      self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
      self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
      self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
      self.conv4 = nn.Identity()
      self.feature_size = 512
    else:
      self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
      self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
      self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
      self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
      self.feature_size = 1024
    self.fc = nn.Identity() if embedding_size == self.feature_size else nn.Linear(self.feature_size, embedding_size)
    self.fc_mean = nn.Linear(embedding_size, state_size)
    self.fc_std = nn.Linear(embedding_size, state_size)

  @jit.script_method
  def forward(self, observation):
    #if self.input_size != 64:
      #observation = self.transform(observation)
      #print("observation", observation.shape)
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.conv4(hidden)
    if self.input_size == 64:
      hidden = self.act_fn(hidden)
    #print("hidden", hidden.shape)
    hidden = hidden.view(-1, self.feature_size)
    embedding = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    latent_mean = self.fc_mean(embedding)
    latent_std = F.softplus(self.fc_std(embedding))
    return latent_mean, latent_std


def Encoder(symbolic, observation_size, embedding_size, state_size, activation_function='relu', input_size=64):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, state_size, activation_function, input_size)

class DGPHiddenLayer(DeepGPLayer):

    def __init__(self, input_dims, output_dims, device, num_inducing, mean_type='constant', num_mixtures=5):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([output_dims])
        # Sparse Variational Formulation (inducing variables initialised as randn)
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

        super().__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'zero':
            self.mean_module = ZeroMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        base_covar_module = RBFKernel()
        # self.covar_module = ScaleKernel(
        #     SpectralMixtureKernel(num_mixtures=num_mixtures, batch_shape=batch_shape, ard_num_dims=input_dims),
        #     batch_shape=batch_shape, ard_num_dims=None, outputscale_constraint=Interval(1e-8, 1)
        # )
        self.covar_module = ScaleKernel(
            base_covar_module, batch_shape=batch_shape, ard_num_dims=None, outputscale_constraint=Interval(1e-8, 1)
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
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))



class DeepGaussianProcesses(DeepGP):
    def __init__(self, input_size, output_size, device, num_inducing, noise_constraint=None, max_cholesky_size=10000, **kwargs):
        # pass hidden layer sizes as separate arguments as well as array
        for k, v in kwargs.items():
            if k=='mean_type':
                setattr(self, k, v)
        hidden_size = hidden_size_extract(kwargs, 'hidden_size')
        
        self.hidden_size = hidden_size
        self.hidden_size.append(output_size)
        self.max_cholesky_size = max_cholesky_size
        self.device = device
        first_layer = DGPHiddenLayer(
                      input_dims=input_size,
                      output_dims=self.hidden_size[0],
                      device = self.device,
                      num_inducing=num_inducing,
                      mean_type=self.mean_type,
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
                     mean_type='constant',
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
                     mean_type='zero',
                  )
                  for i in range(len(self.hidden_size) - 1)
              ])

        super().__init__()

        self.first_layer = first_layer
        self.hidden_layers = hidden_layers
        if output_size is None:
           self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        else:
           self.likelihood= MultitaskGaussianLikelihood(num_tasks=output_size, noise_constraint=noise_constraint)
        self.to(device=self.device)

    def forward(self, inputs):
        #debugpy.breakpoint()
        #print('break on this line')
        out = self.first_layer(inputs)
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
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            mus = []
            variances = []
            samples = []
            lowers=[]
            uppers=[]
            dist = self(x)
            preds = self.likelihood(dist)
            mus.append(preds.mean)
            variances.append(preds.variance)
            lower, upper = preds.confidence_region()
            lowers.append(lower)
            uppers.append(upper)
            samples.append(preds.rsample())           
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lowers, dim=-1), torch.cat(uppers, dim=-1), torch.cat(samples, dim=-1), dist

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
    def __init__(self, latent_size, action_size, lagging_size, num_inducing, device, hidden_size=[32], mean_type= 'linear'):
        input_size = (latent_size+action_size)*lagging_size
        super(TransitionGP, self).__init__(input_size=input_size, output_size=latent_size, device=device, hidden_size=hidden_size, mean_type=mean_type, num_inducing=num_inducing, noise_constraint=Interval(1e-4, 10)) 
      
class PolicyGP(DeepGaussianProcesses):
    #policy P(z_{t})=a(t)--- input z:latent_space---> output a: action
    def __init__(self, latent_size, action_size, lagging_size, num_inducing, device, hidden_size=[32], mean_type= 'constant'):
        input_size =latent_size*lagging_size
        super(PolicyGP, self).__init__(input_size=input_size, output_size= action_size, device=device, hidden_size=hidden_size, mean_type=mean_type, num_inducing=num_inducing, noise_constraint=Interval(1e-4, 10))
     
class RewardGP(DeepGaussianProcesses):
    def __init__(self, latent_size, action_size, lagging_size, num_inducing, device, hidden_size=[32], mean_type= 'zero'):
       input_size =(latent_size+action_size)*lagging_size
       super(RewardGP, self).__init__(input_size=input_size, output_size= None, device=device, hidden_size=hidden_size, mean_type=mean_type, num_inducing=num_inducing, noise_constraint=Interval(1e-4, 10))


class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_size, num_inducing, device, num_mixture_samples=1):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_size = lagging_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_module = TransitionGP(latent_size, action_size, lagging_size, num_inducing, device).to(device=device) 
        self.policy_module = PolicyGP(latent_size, action_size, lagging_size, num_inducing, device).to(device=device) 
        self.reward_gp = RewardGP(latent_size, action_size, lagging_size, num_inducing, device).to(device=device)
        self.num_mixture_samples = num_mixture_samples
        self.likelihood = GaussianLikelihood()
    
    def forward(self, init_states, actions):
        # need to stack actions and latent vectors together (also reshape so that the lagging length dimension is stacked as well)
        init_states = init_states.reshape((init_states.size(0), init_states.size(1), -1))
        # init_states size <chunk_size - horizon_size - lagging_size, batch_size, latent_size * lagging_size>
        actions = actions.reshape((actions.size(0), actions.size(1), -1))
        # actions size <chunk_size - horizon_size - lagging_size, batch_size, action_size * lagging_size>
        z_hat = torch.cat([init_states, actions], dim=-1)

        lagging_actions = actions
        lagging_states = init_states
        
        for i in range(self.horizon_size):
            # policy distribution      
            _, _, _, _, a_s, a_d = self.policy_module.predict(lagging_states)
            # a_s has size <num_gp_likelihood_samples, chunk_size - horizon_size - lagging_size, batch_size, action_size>
            lagging_actions = torch.cat([lagging_actions[..., self.action_size:], torch.mean(a_s, dim=0)], dim=-1)
            
            z_hat = torch.cat([lagging_states, lagging_actions], dim=-1) # z_hat has size <1, 8, 52>
            # z_hat has shape # <chunk_size - horizon_size - lagging_size, batch_size, (action_size + latent_size)*lagging_size>

            # transition distribution
            _, _, _, _, z_s, z_d = self.transition_module.predict(z_hat)
            # z_s has size <num_gp_likelihood_samples, chunk_size - horizon_size - lagging_size, batch_size, latent_size>
            
            lagging_states = torch.cat([lagging_states[..., self.latent_size:], torch.mean(z_s, dim=0)], dim=-1) 
        
        # last policy in the horizon
        _, _, _, _, a_s, _ = self.policy_module.predict(lagging_states)
        lagging_actions = torch.cat([lagging_actions[..., self.action_size:], torch.mean(a_s, dim=0)], dim=-1) 
        z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
        # output the final reward
        rewards = self.reward_gp.predict(z_hat)[-1] # self.reward_gp(z_hat) will return a MultivariateNormal object
        return rewards
