import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from torch.distributions import MultivariateNormal, Normal, Independent, Uniform
#https://github.com/JasonMa2016/SMODICE/blob/d6e58b0663fe636f313fe761b5473b6891f09f2b/discriminator_pytorch.py
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class Generator(nn.Module):
    def __init__(self,
                input_dim: int,
                output_dim: int,
                hidden_dim: int,
                latent_dim: int,
                device: Any,
                lr: float = 1e-4,
                weight_decay : float = 1e-3,
                opt_betas : tuple = (0.9, 0.999)
                ):
        super(Generator, self).__init__()

        self.output_emb = nn.Sequential(
                                     nn.Linear(input_dim, hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True)
                                     )


        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + hidden_dim, hidden_dim, normalize=False),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        self.device=device
        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.opt_betas,
        )
        self.to(device)

    def forward(self, noise, x):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.output_emb(x), noise), -1)
        y = self.model(gen_input)
        return y

def generate_noise(bs, nz, device):
    loc = torch.zeros(bs, nz).to(device)
    scale = torch.ones(bs, nz).to(device)
    normal = Normal(loc, scale)
    diagn = Independent(normal, 1)
    noise = diagn.sample()
	return noise

class Discriminator(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 latent_dim:int,
                 device:Any,
                 lr: float = 1e-4,
                 ):
        super(Discriminator, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr     = lr
        self.device = device
        self.input_condition_disc = nn.Sequential( nn.Linear(input_dim, hidden_dim),
                                                  nn.Tanh()
                                                  ).to(device)
        self.output_condition_disc = nn.Sequential( nn.Linear(output_dim, hidden_dim),
                                                    nn.Tanh()
                                                  ).to(device)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.train()

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.fake_state_action = Generator(self.input_dim, self.output_dim, self.hidden_dim, self.latent_dim, self.device)
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         next_state,
                         current_state,
                         current_action,
                         lambda_=10):
        """Calculates the gradient penalty loss for WGAN GP"""

        future_data  = self.output_condition_disc(next_state)
        current_data = self.input_condition_disc(torch.cat([current_state, current_action.float()], dim=1))
        alpha = torch.rand(current_data.size(0), 1)
        alpha = alpha.expand_as(current_data).to(current_data.device)
        #interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True).float()
        mixup_data = (alpha  * current_data + (1 - alpha)* future_data).requires_grad_(True).float()


        disc = self.trunk(mixup_data)
        ones =  torch.autograd.Variable(torch.ones(disc.size()), requires_grad=False).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * torch.mean((grad.norm(2, dim=1) - 1).pow(2))
        return grad_pen

    def update(self, loader):
        self.train()
        self.fake_state_action.train()
        div_loss = 0
        penalty = 0
        n = 0
        for item in loader:
            policy_state = torch.FloatTensor(item["obs"]).to(self.device)
            policy_action = torch.FloatTensor(item["action"]).to(self.device)
            current_state_o = self.input_condition_disc( torch.cat([policy_state, policy_action], dim=1) )

            next_state = torch.FloatTensor(item["next_obs"]).to(self.device)

            next_state_o = self.output_condition_disc(next_state)
            #min Es∼d^o [logc(s,a)]+Es∼d^f [log(1−c(s,a))]
            observation_d = self.trunk(
                torch.cat([next_state_o, current_state_o], dim=1))


            real_loss = F.binary_cross_entropy_with_logits(
                observation_d,
                torch.autograd.Variable(torch.ones_like(observation_d), requires_grad=False).to(self.device))


            noise_            = generate_noise(bs=policy_state.shape[0], nz=self.latent_dim, device=self.device)
            next_state_f      = self.fake_state_action(noise_, torch.cat([policy_state, policy_action.float()], dim=1))

            next_state_f      = self.output_condition_disc(next_state_f)
            observation_f     = self.trunk(torch.cat([next_state_f, current_state_o], dim=1))

            fake_loss = F.binary_cross_entropy_with_logits(
                                                           observation_f,
                                                           torch.autograd.Variable(torch.zeros_like(observation_f), requires_grad=False).to(self.device))
            gail_loss = real_loss + fake_loss
            grad_pen = self.compute_grad_pen(next_state_f,
                                             policy_state, policy_action)

            div_loss += gail_loss.item()
            penalty  += grad_pen.item()
            n += 1

        return div_loss / n, penalty/ n

    def predict(self, state, action):
        with torch.no_grad():
            self.eval()
            noise = generate_noise(bs=state.shape[0], nz=self.latent_dim, device=self.device)
            fake_future_state = self.fake_state_action(noise, torch.cat([state, action.float()], dim=1))
        return fake_future_state

class TransitDataset(torch.utils.data.Dataset):

    def __init__(self, states, actions, next_states):
        """
        Custom PyTorch dataset that preprocess each transition dynamics.

        """
        super(TransitDataset, self).__init__()

        self.next_states = next_states
        self.states      = states
        self.actions     = actions


    def __len__(self):
        return len(self.next_states)

    def __getitem__(self, item):
        states  = self.states[item]
        actions = self.actions[item]
        targets = self.targets[item]
        return {"next_obs":targets , "obs": states, "action": actions}

def preprocess_loader(states, actions, next_states, device):
    dataset =  TransitDataset(states, actions, next_states)
    pin_memory = device == 'cuda'
    dloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=states.shape[0]
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory
            )
    return dloader
