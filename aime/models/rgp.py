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
    def __init__(self, horizon_size, latent_size, action_size, train_x_shape=(64, 64, 3)):
        super().__init__()
        self.horizon_size = horizon_size
        self.transition_modules = [None for _ in range(horizon_size)]
        self.policy_modules = [None for _ in range(horizon_size)]
        for i in range(horizon_size):
            self.transition_modules[i] = TransitionGP(
                input_dims=latent_size,
                output_dims=latent_size
            )
            self.policy_modules[i] = PolicyGP(
                input_dims=latent_size,
                output_dims=action_size
            )
        self.reward_gp = RewardGP(input_dims=1, output_dims=1)
    
    def forward(self, inputs):
        # first stack input x and latent vector z together
        z_hat = inputs
        for i in range(self.horizon_size):
            z = self.transition_modules[i](z_hat)
            w = z[:-1]
            a = self.policy_modules[i](w)
            z_hat = torch.stack([z, a])
        # output the final reward
        r = self.reward_gp(z_hat)
        return r
