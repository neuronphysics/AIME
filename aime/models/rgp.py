# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/DGP_Multitask_Regression.html

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.nn import PyroModule, PyroParam, PyroSample

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=10):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
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

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class TransitionGP(DGPHiddenLayer):
    pass

class PolicyGP(DGPHiddenLayer):
    pass

class RewardGP(DGPHiddenLayer):
    pass

class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_length, use_cuda=True):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_length = lagging_length
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_modules = [TransitionGP(input_dims=(latent_size+action_size)*lagging_length, output_dims=latent_size) for _ in range(horizon_size)]
        self.policy_modules = [PolicyGP(input_dims=latent_size*lagging_length, output_dims=action_size) for _ in range(horizon_size)]
        self.reward_gp = RewardGP(input_dims=(latent_size+action_size)*lagging_length, output_dims=1)
        if use_cuda:
            self.cuda()
    
    def forward(self, data):
        # need to stack actions and latent vectors together (also reshape so that the lagging length dimension is stacked as well)
        z = torch.reshape(data["latent"][:self.lagging_length].transpose(0,1), (-1, self.lagging_length * self.latent_size))
        a = torch.reshape(data["action"][:self.lagging_length].transpose(0,1), (-1, self.lagging_length * self.action_size))
        horizon_actions = []
        horizon_latents= []
        for i in range(self.horizon_size):
            latent = self.transition_modules[i](torch.cat((z, a), dim=-1))
            horizon_latents.append(latent)
            z = torch.cat((z[:, 1:, :], latent.unsqueeze(0)), dim=-2)
            action = self.policy_modules[i](z)
            horizon_actions.append(a)
            a = torch.cat((a[:, 1:, :], action.unsqueeze(0)), dim=-2)
        # output the final reward
        r = self.reward_gp(torch.cat((z, a), dim=-1))
        return r
