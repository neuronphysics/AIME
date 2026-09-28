import numpy as np
import numpy.random as npr
from . import conditionals
from . import utils
from numpy import newaxis as na
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed
from os import cpu_count
import copy
n_cpu = int(cpu_count() / 2)


class TroSLDS:
    'The recurrent only TrSLDS. This was the model showcased in Nassar et al. ICLR (2019)'
    def __init__(self, D_in, D_out, K, dynamics, dynamics_noise, emission, hyper_planes, possible_paths, leaf_path,
                 leaf_nodes, D_bias=1, nu=None, nuy=None, Lambda_x=None, Lambda_y=None, My=None, Vy=None,
                 mu_hyper=None, tau_hyper=None, Mx=None, Vx=None, bern=False, emission_noise=None, normalize=True,
                 rotate=True, P0=None, scale=None, N=1, neg_bin=False):
        self.D_in = D_in
        self.D_out = D_out
        self.K = K
        self.depth = int(np.ceil(np.log2(K)) + 1)
        self.x = []
        self.u = []
        self.z = []
        self.path = []
        self.y = []

        self.mask = []
        
        if scale is None:
            self.scale = 0.9
        else:
            self.scale = scale

        "The following three variables characterize the tree"
        self.poss_paths = possible_paths
        self.leaf_paths = leaf_path
        self.leaf_nodes = leaf_nodes

        self.D_bias = D_bias

        self.A = dynamics
        self.Aleaf = np.ones((D_in, D_in + 1, K))
        self._obtain_leaf_dynamics()
        self.Q = dynamics_noise
        self.C = emission
        if bern != True:
            if emission_noise is None:
                self.S = np.eye(D_out)
            else:
                self.S = emission_noise

        self.R = hyper_planes

        if mu_hyper is tau_hyper is None:
            self.mu_hyper = np.zeros(D_in + 1)
            self.tau_hyper = 1e-4*np.eye(D_in + 1)

        if nu is Lambda_x is Mx is Vx is None:
            self.nux = D_in + 1
            self.lambdax = 1e-8*np.eye(D_in)
            self.Mx = np.zeros((D_in, D_in + D_bias))
            self.Mx[:, :-D_bias] = 0.99 * np.eye(D_in)
            self.Vx = 100*np.eye(D_in + D_bias)
        else:
            self.nux = nu
            assert nu > D_in
            self.lambdax = Lambda_x
            self.Mx = Mx
            self.Vx = Vx

        if nuy is Lambda_y is My is Vy is None:
            self.nuy = D_out + 1
            self.lambday = 1e-8*np.eye(D_out)
            self.My = np.zeros((D_out, D_in + 1))
            self.Vy = 100*np.eye(D_in + 1)
        else:
            self.nuy = nuy
            assert nuy > D_out
            self.lambday = Lambda_y
            self.My = My
            self.Vy = Vy

        if P0 is None:
            self.P0 = 20*np.eye(D_in)
        else:
            self.P0 = P0

        self.normalize = normalize
        self.rotate = rotate
        self.bern = bern
        self.neg_bin = neg_bin
        self.N = N

        self.alphas = []
        self.covs = []
        
        self.omega = []
        if bern:
            self.spike_omega = []

    def _add_data(self, x, y, z, path, mask=None, u=None):
        self.x.append(x)
        self.y.append(y)
        self.path.append(path)
        self.z.append(z)
        if u is None:
            self.u.append(np.ones((1, x[0, :].size)))
        else:
            assert self.D_bias == u[:, 0].size
            self.u.append(u)

        if mask is None:
            self.mask.append(y[0, :] != np.nan)


    def _initialize_polya_gamma(self):
        assert len(self.x) != 0
        self.omega = [None] * len(self.x)
        self.omega = conditionals.pg_tree_posterior(self.x, self.omega, self.R, self.path, self.depth)

        if self.bern:
            self.spike_omega = [None] * len(self.x)
            self._sample_spike_pg()


    def _obtain_leaf_dynamics(self):
        for (level, node, k) in self.leaf_nodes:
            self.Aleaf[:, :, k] = self.A[level][:, :, node]

    def _sample_emission(self):
        if self.bern:
            self.C = conditionals.emission_parameters_spike_train(self.y, self.x, self.spike_omega, self.mask,
                                                                  self.My[0, :], self.Vy, self.normalize, N=self.N,
                                                                  neg_bin=self.neg_bin)
        else:
            self.C, self.S = conditionals.emission_parameters(self.y, self.x, self.mask, self.nuy, self.lambday,
                                                              self.My, self.Vy, self.normalize)

        if self.rotate:
            "Take RQ decomposition of C"
            upper, orthor = scipy.linalg.rq(self.C[:, :-1])

            "Constrain minor diagnoal of upper to be positive to avoid sign flipping"
            rotate = np.eye(self.D_in)
            for j in range(self.D_in):
                if np.sign(upper[self.D_out - self.D_in + j, j]) < 0:
                    rotate[j, j] = -1

            upper = upper @ rotate
            orthor = rotate @ orthor

            self.C[:, :-1] = upper

            "Rotate latent states and dynamics"
            self.x = utils.rotate_latent(self.x, orthor)
            self.A = utils.rotate_dynamics(self.A, orthor, self.depth)
            self._obtain_leaf_dynamics()

    def _sample_hyperplanes(self):
        if self.depth != 1:
            self.R = utils.sample_hyperplanes(self.x, self.omega, self.path, self.depth, self.mu_hyper, self.tau_hyper,
                                              self.poss_paths, self.R)

    def _sample_dynamics(self):
        self.A, self.Q = utils.sample_leaf_dynamics(self.x, self.u, self.z, self.A, self.Q, self.nux, self.lambdax,
                                                    self.Mx, self.Vx, self.scale, self.leaf_nodes)
        self._obtain_leaf_dynamics()
        self.A = utils.sample_internal_dynamics(self.A, self.scale, self.Mx, self.Vx, self.depth)

    def _sample_discrete_latent(self):
        self.z, self.path = conditionals.discrete_latent_recurrent_only(self.z, self.path, self.leaf_paths, self.K,
                                                                        self.x, self.u, self.Aleaf, self.Q, self.R,
                                                                        self.depth, self.D_bias)

    def _sample_pg(self):
        self.omega = conditionals.pg_tree_posterior(self.x, self.omega, self.R, self.path, self.depth)

    def _sample_spike_pg(self):
        self.spike_omega = conditionals.pg_spike_train(self.x, self.y, self.C, self.spike_omega, self.D_out, N=self.N,
                                                       neg_bin=self.neg_bin)

    def _sample_continuous_latent(self):
        if len(self.alphas) == 0:
            max_len = max([self.x[idx][0, :].size for idx in range(len(self.x))])
            self.alphas = np.zeros((self.D_in, max_len))
            self.covs = np.repeat(self.P0[:, :, na], max_len, axis=2)
            self.P = np.repeat(self.P0[:, :, na], self.alphas[0, :].size, axis=2)

        if self.bern:
            self._sample_pg()
            self._sample_spike_pg()
            temps = Parallel(n_jobs=n_cpu)(delayed(conditionals.pg_kalman)(self.D_in, self.D_bias, self.x[n],
                                                                           self.u[n], self.P, self.Aleaf, self.Q,
                                                                           self.C, 0, self.y[n], self.path[n],
                                                                           self.z[n], self.omega[n], self.alphas,
                                                                           self.covs, self.R, self.depth,
                                                                           omegay=self.spike_omega[n], bern=self.bern,
                                                                           N=self.N, neg_bin=self.neg_bin, marker=n)
                                           for n in range(len(self.x)))
            for n in range(len(temps)):
                self.x[temps[n][1]] = temps[n][0]
        else:
            self._sample_pg()
            temps = Parallel(n_jobs=n_cpu)(delayed(conditionals.pg_kalman)(self.D_in, self.D_bias, self.x[n],
                                                                                    self.u[n], self.P, self.Aleaf,
                                                                                    self.Q, self.C, self.S, self.y[n],
                                                                                    self.path[n], self.z[n],
                                                                                    self.omega[n], self.alphas,
                                                                                    self.covs, self.R, self.depth,
                                                                                    marker=n) for n in range(len(self.x)))
            for n in range(len(temps)):
                self.x[temps[n][1]] = temps[n][0]


    def resample(self, no_samples):
        self._initialize_polya_gamma()
        for m in tqdm(range(no_samples)):
            self._sample_emission()
            self._sample_hyperplanes()
            self._sample_dynamics()
            self._sample_discrete_latent()
            self._sample_continuous_latent()

    def _generate_data(self, T, starting_pt, u=None, noise=True):
        if u is None:
            u = np.ones((1, T))
        x = np.zeros((self.D_in, T + 1))
        x[:, 0] = starting_pt
        y = np.zeros((self.D_out, T))
        z = np.zeros(T + 1).astype(int)
        for t in range(T):
            log_p = utils.compute_leaf_log_prob(self.R, x[:, t], self.K, self.depth, self.leaf_paths)
            p_unnorm = np.exp(log_p - np.max(log_p))
            p = p_unnorm/np.sum(p_unnorm)
            if noise:
                choice = npr.multinomial(1, p.ravel(), size=1)
                z[t] = np.where(choice[0, :] == 1)[0][0].astype(int)
                x[:, t + 1] = (self.Aleaf[:, :-self.D_bias, z[t]] @ x[:, t][:, na] +  \
                              self.Aleaf[:, -self.D_bias:, z[t]] @ u[:, t][:, na] + \
                              npr.multivariate_normal(np.zeros(self.D_in), self.Q[:, :, z[t]])[:, na]).flatten()
                y[:, t] = (self.C[:, :-1] @ x[:, t + 1][:, na] + self.C[:, -1][:, na] + \
                          npr.multivariate_normal(np.zeros(self.D_out), self.S)[:, na]).flatten()

            else:
                z[t] = np.argmax(choice)
                x[:, t + 1] = (self.Aleaf[:, :-self.D_bias, z[t]] @ x[:, t][:, na] + \
                               self.Aleaf[:, -self.D_bias:, z[t]] @ u[:, t][:, na]).flatten()
                y[:, t] = (self.C[:, :-1] @ x[:, t + 1][:, na] + self.C[:, -1][:, na]).flatten()

        log_p = utils.compute_leaf_log_prob(self.R, x[:, -1], self.K, self.depth, self.leaf_paths)
        p_unnorm = np.exp(log_p - np.max(log_p))
        p = p_unnorm / np.sum(p_unnorm)
        choice = npr.multinomial(1, p.ravel(), size=1)
        z[-1] = np.where(choice[0, :] == 1)[0][0]

        return y, x, z










