# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

import torch
import torch.nn as nn 
import torch.nn.functional as F
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, device, num_inducing=16, mean_type='constant'):
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

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
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
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class AIMEDeepGP(DeepGP):
    def __init__(self, input_dim, out_dim, device, hidden_dim=16, num_inducing=16, noise_constraint=None):
        hidden_layer = DeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            device=device,
            num_inducing=num_inducing,
            mean_type='linear',
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=out_dim,
            device=device,
            num_inducing=num_inducing,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        if out_dim is None:
          self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        else:
          self.likelihood = MultitaskGaussianLikelihood(out_dim, noise_constraint=noise_constraint)
        
        self.device = device
        self.to(device=self.device)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output
      
    def predict_sample(self, inputs):
        return self.likelihood(self(inputs)).mean#.sample()

class TransitionGP(AIMEDeepGP):
    #transition probability  P(z_{t}|x_{t-k},...x_{t-1})=N(z_{t}|mu_x,sigma_x)---->T(x_{t-k},...x_{t-1})=z_{t}; x_{.}=[z_{.},a_{.}]
    def __init__(self, latent_size, action_size, lagging_size, num_inducing, device, hidden_dim=16):
        input_size = (latent_size+action_size)*lagging_size
        super(TransitionGP, self).__init__(input_dim=input_size, out_dim=latent_size, device=device, hidden_dim=hidden_dim, num_inducing=num_inducing, noise_constraint=None) 
      
class RewardGP(AIMEDeepGP):
    def __init__(self, latent_size, action_size, lagging_size, num_inducing, device, hidden_dim=16):
       input_size =(latent_size+action_size)*lagging_size
       super(RewardGP, self).__init__(input_dim=input_size, out_dim= None, device=device, hidden_dim=hidden_dim, num_inducing=num_inducing, noise_constraint=None)

class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_size, num_inducing, device, num_mixture_samples=1):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_size = lagging_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_module = TransitionGP(latent_size, action_size, lagging_size, num_inducing, device).to(device=device) 
        self.reward_module = RewardGP(latent_size, action_size, lagging_size, num_inducing, device).to(device=device)

    def predict_transition(self, input):
        with torch.no_grad():
          return self.transition_module.predict_sample(input)

    def predict_reward(self, input):
        with torch.no_grad():
          return self.reward_module.predict_sample(input)

    def predict_recurrent_gp(self, init_states, init_actions, policy):
        with torch.no_grad():
          out, _, _ = self(init_states, init_actions, policy)
          preds = self.reward_module.likelihood(out)
        return preds.mean

    def forward(self, init_states, init_actions, policy):
        init_states = init_states.reshape((init_states.size(0) * init_states.size(1), -1))
        # init_states = init_states.reshape(init_states.size(0), init_states.size(1), -1)
        init_actions = init_actions.reshape((init_actions.size(0) * init_actions.size(1), -1))
        # init_actions = init_actions.reshape(init_actions.size(0), init_actions.size(1), -1)
        print(f'Anudeep, init_states shape: {init_states.shape}')
        print(f'Anudeep, init_actions shape: {init_actions.shape}')
        z_hat = torch.cat([init_states, init_actions], dim=-1).T
        # z_hat = z_hat.reshape((z_hat.size(0) * z_hat.size(1), -1))
        
        future_states = []
        future_actions = []
        lagging_actions = init_actions.T
        lagging_states = init_states.T

        for i in range(self.horizon_size):
            next_state = self.transition_module.predict_sample(z_hat)
            # z_s has size <num_gp_likelihood_samples, chunk_size - horizon_size - lagging_size, batch_size, latent_size>
            print(f'Next state shape: {next_state.shape}')
            print(f'Lagging state view : {lagging_states[..., self.latent_size:].shape}')
            lagging_states = torch.cat([lagging_states[..., self.latent_size:], torch.mean(next_state, dim=0)[:lagging_states.size(0),:]], dim=0) 
            
            next_action = policy.predict_sample(lagging_states)
            # a_s has size <num_gp_likelihood_samples, chunk_size - horizon_size - lagging_size, batch_size, action_size>
            lagging_actions = torch.cat([lagging_actions[..., self.action_size:], torch.mean(next_action, dim=0)], dim=-1)


            # transition distribution
            z_hat = torch.cat([lagging_states, lagging_actions], dim=-1) 
            
            # policy distribution      
            
            # z_hat = torch.cat([lagging_states, lagging_actions], dim=-1) # z_hat has size <1, 8, 52>
            # # z_hat has shape # <chunk_size - horizon_size - lagging_size, batch_size, (action_size + latent_size)*lagging_size>

        
        # last policy in the horizon
        a_s = policy.predict_sample(lagging_states)
        lagging_actions = torch.cat([lagging_actions[..., self.action_size:], torch.mean(a_s, dim=0)], dim=-1) 
        z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
        future_states.append(next_state)
        future_actions.append(next_action)
        # output the final reward
        pred_rewards = self.reward_module(z_hat)
        future_states = torch.stack(future_states, dim=0) if len(future_states) > 0 else None
        future_actions = torch.stack(future_actions, dim=0) if len(future_actions) > 0 else None
        
        return pred_rewards, future_states, future_actions

