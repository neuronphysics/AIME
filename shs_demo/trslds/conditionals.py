import numpy as np
from . import utils
from numpy import newaxis as na
import numpy.random as npr
import scipy
from numpy.linalg import LinAlgError
from scipy.stats import invwishart
from pypolyagamma import PyPolyaGamma
import pypolyagamma
from numba import njit, jit
import pdb
from os import cpu_count
import time

n_cpu = int(cpu_count() / 2)


def pg_tree_posterior(states, omega, R, path, depth, nthreads=None):
    for idx in range(len(states)):
        T = states[idx][0, :].size
        b = np.ones(T * (depth - 1))
        if nthreads is None:
            nthreads = n_cpu
        v = np.ones((depth - 1, T))
        out = np.empty(T * (depth - 1))
        for d in range(depth - 1):
            for t in range(T):
                index = int(path[idx][d, t] - 1)
                v[d, t] = np.matmul(R[d][:-1, index], np.array(states[idx][:, t])) + R[d][-1, index]
        seeds = np.random.randint(2 ** 16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]
        pypolyagamma.pgdrawvpar(ppgs, b, v.flatten(order='F'), out)
        omega[idx] = out.reshape((depth - 1, T), order='F')

    return omega


def pg_spike_train(X, Y, C, Omega, D_out, nthreads=None, N=1, neg_bin=False):
    for idx in range(len(X)):
        T = X[idx][0, 1:].size
        b = N * np.ones(T * D_out)
        if neg_bin:
            b += Y[idx].flatten(order='F')
        if nthreads is None:
            nthreads = n_cpu
        out = np.empty(T * D_out)
        V = C[:, :-1] @ X[idx][:, 1:] + C[:, -1][:, na]

        seeds = np.random.randint(2 ** 16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]

        pypolyagamma.pgdrawvpar(ppgs, b, V.flatten(order='F'), out)
        Omega[idx] = out.reshape((D_out, T), order='F')

    return Omega


def emission_parameters(obsv, states, mask, nu, Lambda, M, V, normalize=True):

    Y = np.hstack(obsv).T
    X = np.hstack([states[idx][:, 1:] for idx in range(len(states))])
    X = np.vstack((X, np.ones((1, X[0, :].size)))).T

    boolean_mask = np.hstack(mask).T
    Y = Y[boolean_mask, :]
    X = X[boolean_mask, :]

    M_posterior, V_posterior, IW_matrix, df_posterior = utils.compute_ss_mniw(X, Y, nu, Lambda, M, V)

    C, S = utils.sample_mniw(df_posterior, IW_matrix, M_posterior, V_posterior)

    if normalize:
        C_temp = C[:, :-1]
        L = np.diag(C_temp.T @ C_temp)
        L = np.diag(np.power(L, -0.5))
        C_temp = C_temp @ L
        C[:, :-1] = C_temp

    return C, S


def emission_parameters_spike_train(spikes, states, Omega, mask, mu, Sigma, normalize=True, N=1, neg_bin=False):
    X = np.hstack([states[idx][:, 1:] for idx in range(len(states))])
    X = np.vstack((X, np.ones(X[0, :].size)))
    Y = np.hstack(spikes)
    W = np.hstack(Omega)
    boolean_mask = np.hstack(mask)

    X = X[:, boolean_mask]
    Y = Y[:, boolean_mask]
    W = W[:, boolean_mask]
    
    dim_y = Y[:, 0].size
    dim = X[:-1, 0].size
    C = np.zeros((dim_y, dim + 1))

    for neuron in range(dim_y):
        Lambda_post = np.linalg.inv(Sigma)
        temp_mu = np.matmul(mu, Lambda_post)

        xw_tilde = np.multiply(X, np.sqrt(W[neuron, :]))
        Lambda_post += np.einsum('ij,ik->jk', xw_tilde.T, xw_tilde.T)
        if neg_bin:
            kappa = (Y[neuron, :][na, :] - N) / 2
        else:
            kappa = Y[neuron, :][na, :] - N / 2

        temp_mu += np.sum((X * kappa).T, axis=0)
        Sigma_post = np.linalg.inv(Lambda_post)
        mu_post = np.matmul(temp_mu, Sigma_post)
        C[neuron, :] = npr.multivariate_normal(np.array(mu_post).ravel(), Sigma_post)

    if normalize:
        C_temp = C[:, :-1]
        L = np.diag(np.matmul(C_temp.T, C_temp))
        L = np.diag(np.power(L, -0.5))
        C_temp = np.matmul(C_temp, L)
        C[:, :-1] = C_temp

    return C


def hyper_planes(w, x, z, prior_mu, prior_precision, draw_prior):

    if draw_prior:
        return npr.multivariate_normal(prior_mu, prior_precision)
    else:
        J = prior_precision @ prior_mu[:, na]
        xw_tilde = np.multiply(x, np.sqrt(w[na, :]))
        precision = np.einsum('ij,ik->jk', xw_tilde.T,
                              xw_tilde.T)

        k = z % 2 - 0.5
        J += np.sum(x * k[na, :], axis=1)[:, na]

        posterior_cov = np.linalg.inv(precision + prior_precision)
        posterior_mu = posterior_cov @ J

        return npr.multivariate_normal(posterior_mu.flatten(), posterior_cov)


def _internal_dynamics(Mprior, Vparent, Achild, Vchild, N=2):
    assert Mprior.shape == Achild.shape
    precision_parent = np.linalg.inv(np.kron(Vparent, np.eye(Achild[:, 0].size)))
    precision_child = np.linalg.inv(np.kron(Vchild, np.eye(Achild[:, 0].size)))
    
    posterior_sigma = np.linalg.inv(precision_parent + N*precision_child)
    posterior_mu = posterior_sigma @ (precision_parent @ Mprior.flatten(order='F')[:, na] + 
                                      precision_child @ Achild.flatten(order='F')[:, na])
    return npr.multivariate_normal(posterior_mu.flatten(), posterior_sigma).reshape(Achild.shape, order='F')


def leaf_dynamics(Y, X, nu, Lambda, M, V, draw_prior):
    if draw_prior:
        A, Q = utils.sample_mniw(nu, Lambda, M, V)
        return A, Q
    else:
        M_posterior, V_posterior, IW_matrix, df_posterior = utils.compute_ss_mniw(X.T, Y.T, nu, Lambda, M, V)
        A, Q = utils.sample_mniw(df_posterior, IW_matrix, M_posterior, V_posterior)
        return A, Q


def discrete_latent_recurrent_only(Z, paths, leaf_path, K, X, U, A, Q, R, depth, D_input):
    Qinv = Q + 0
    Qlogdet = np.ones(K)
    for k in range(K):
        Qinv[:, :, k] = np.linalg.inv(Q[:, :, k])
        Qlogdet[k] = np.log(np.linalg.det(Q[:, :, k]))

    for idx in range(len(X)):
        log_p = utils.compute_leaf_log_prob_vectorized(R, X[idx], K, depth, leaf_path)

        "Compute transition probability for each leaf"
        temp = X[idx][:, 1:]
        for k in range(K):
            mu_temp = A[:, :-D_input, k] @ X[idx][:, :-1] + A[:, -D_input:, k] @ U[idx][:, :-1]
            log_p[k, :-1] += utils.log_mvn(temp, mu_temp, Qinv[:, :, k], Qlogdet[k])

        post_unnorm = np.exp(log_p - np.max(log_p, 0))
        post_p = post_unnorm / np.sum(post_unnorm, 0)
        for t in range(X[idx][0, :].size):
            choice = npr.multinomial(1, post_p[:, t], size=1)
            paths[idx][:, t] = leaf_path[:, np.where(choice[0, :] == 1)[0][0]].ravel()
            Z[idx][t] = np.where(choice[0, :] == 1)[0][0]

    return Z, paths


def pg_kalman(D_in, D_bias, X, U, P, As, Qs, C, S, Y, paths, Z, omega,
          alphas, Lambdas, R, depth, omegay=None, bern=False, N=1, neg_bin=False, marker=1):

    iden = np.eye(D_in)
    Qinvs = np.zeros((D_in, D_in, Qs[0, 0, :].size))
    for k in range(Qs[0, 0, :].size):
        temp = np.linalg.inv(np.linalg.cholesky(Qs[:, :, k]))
        Qinvs[:, :, k] = temp.T @ temp

    "Filter forward"
    for t in range(X[0, :].size - 1):
        if depth == 1:
            alpha = X[:, t][:, na] + 0
            Lambda = P[:, :, t] + 0
        else:
            J = 0
            temp_mu = 0
            for d in range(depth - 1):
                loc = paths[d, t]
                fin = paths[d + 1, t]
                if ~np.isnan(fin):
                    k = 0.5 * (fin % 2 == 1) - 0.5 * (
                            fin % 2 == 0)
                    tempR = np.expand_dims(R[d][:-1, int(loc - 1)], axis=1)
                    J += omega[d, t] * np.matmul(tempR, tempR.T)
                    temp_mu += tempR.T * (k - omega[d, t] * R[d][-1, int(loc - 1)])

            Pinv = np.linalg.inv(P[:, :, t])
            Lambda = np.linalg.inv(Pinv + J)
            alpha = Lambda @ (Pinv @ X[:, t][:, na] + temp_mu.T)

        alphas[:, t] = alpha.flatten() + 0
        Lambdas[:, :, t] = Lambda + 0
        Q = Qs[:, :, int(Z[t])] + 0
        x_prior = As[:, :-D_bias, int(Z[t])] @ alpha + As[:, -D_bias:, int(Z[t])] @ U[:, t][:, na]
        P_prior = As[:, :-D_bias, int(Z[t])] @ Lambda @ As[:, :-D_bias, int(Z[t])].T + Q
        if bern:
            if neg_bin:
                kt = (Y[:, t] - N) / 2
            else:
                kt = Y[:, t] - N / 2
            S = np.diag(1 / omegay[:, t])
            yt = kt / omegay[:, t]
        else:
            yt = Y[:, t]

        K = P_prior @ np.linalg.solve(C[:, :-1] @ P_prior @ C[:, :-1].T + S, C[:, :-1]).T

        X[:, t + 1] = (x_prior + K @ (yt[:, na] - C[:, :-1] @ x_prior - C[:, -1][:, na])).flatten()
        P_temp = (iden - K @ C[:, :-1]) @ P_prior

        P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden

    "Sample backwards"
    eps = npr.normal(size=X.shape)
    X[:, -1] = X[:, -1] + (np.linalg.cholesky(P[:, :, X[0, :].size - 1]) @ eps[:, -1][:, na]).ravel()

    for t in range(X[0, :].size - 2, -1, -1):
        alpha = alphas[:, t][:, na]
        Lambda = Lambdas[:, :, t]

        A_tot = As[:, :-D_bias, int(Z[t])]
        B_tot = As[:, -D_bias:, int(Z[t])][:, na]
        Q = Qs[:, :, int(Z[t])]
        Qinv = Qinvs[:, :, int(Z[t])]

        Pn = Lambda - Lambda @ A_tot.T @ np.linalg.solve(Q + A_tot @ Lambda @ A_tot.T, A_tot @ Lambda)
        mu_n = Pn @ (np.linalg.solve(Lambda, alpha) + A_tot.T @ Qinv @(X[:, t + 1][:, na] - B_tot @ U[:, t]))

        Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

        X[:, t] = (mu_n + np.linalg.cholesky(Pn) @ eps[:, t][:, na]).ravel()
    return X, marker


def pg_kalman_batch(D_in, D_bias, X, U, P, As, Qs, C, S, Y, paths, Z, omega,
          alphas, Lambdas, R, depth, omegay=None, bern=False, N=1, neg_bin=False):

    iden = np.eye(D_in)
    Qinvs = np.zeros((D_in, D_in, Qs[0, 0, :].size))
    for k in range(Qs[0, 0, :].size):
        temp = np.linalg.inv(np.linalg.cholesky(Qs[:, :, k]))
        Qinvs[:, :, k] = temp.T @ temp

    "Filter forward"
    for idx in range(len(X)):
        for t in range(X[idx][0, :].size - 1):
            if depth == 1:
                alpha = X[idx][:, t][:, na] + 0
                Lambda = P[:, :, t] + 0
            else:
                J = 0
                temp_mu = 0
                for d in range(depth - 1):
                    loc = paths[idx][d, t]
                    fin = paths[idx][d + 1, t]
                    if ~np.isnan(fin):
                        k = 0.5 * (fin % 2 == 1) - 0.5 * (
                                fin % 2 == 0)
                        tempR = np.expand_dims(R[d][:-1, int(loc - 1)], axis=1)
                        J += omega[idx][d, t] * np.matmul(tempR, tempR.T)
                        temp_mu += tempR.T * (k - omega[idx][d, t] * R[d][-1, int(loc - 1)])

                Pinv = np.linalg.inv(P[:, :, t])
                Lambda = np.linalg.inv(Pinv + J)
                alpha = Lambda @ (Pinv @ X[idx][:, t][:, na] + temp_mu.T)

            alphas[:, t] = alpha.flatten() + 0
            Lambdas[:, :, t] = Lambda + 0
            Q = Qs[:, :, int(Z[idx][t])] + 0
            x_prior = As[:, :-D_bias, int(Z[idx][t])] @ alpha + As[:, -D_bias:, int(Z[idx][t])] @ U[idx][:, t][:, na]
            P_prior = As[:, :-D_bias, int(Z[idx][t])] @ Lambda @ As[:, :-D_bias, int(Z[idx][t])].T + Q
            if bern:
                if neg_bin:
                    kt = (Y[idx][:, t] - N) / 2
                else:
                    kt = Y[idx][:, t] - N / 2
                S = np.diag(1 / omegay[idx][:, t])
                yt = kt / omegay[idx][:, t]
            else:
                yt = Y[idx][:, t]

            K = P_prior @ np.linalg.solve(C[:, :-1] @ P_prior @ C[:, :-1].T + S, C[:, :-1]).T

            X[idx][:, t + 1] = (x_prior + K @ (yt[:, na] - C[:, :-1] @ x_prior - C[:, -1][:, na])).flatten()
            P_temp = (iden - K @ C[:, :-1]) @ P_prior

            P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden

        "Sample backwards"
        eps = npr.normal(size=X[idx].shape)
        X[idx][:, -1] = X[idx][:, -1] + (np.linalg.cholesky(P[:, :, X[idx][0, :].size - 1]) @ eps[:, -1][:, na]).ravel()

        for t in range(X[idx][0, :].size - 2, -1, -1):
            alpha = alphas[:, t][:, na]
            Lambda = Lambdas[:, :, t]

            A_tot = As[:, :-D_bias, int(Z[idx][t])]
            B_tot = As[:, -D_bias:, int(Z[idx][t])][:, na]
            Q = Qs[:, :, int(Z[idx][t])]
            Qinv = Qinvs[:, :, int(Z[idx][t])]

            Pn = Lambda - Lambda @ A_tot.T @ np.linalg.solve(Q + A_tot @ Lambda @ A_tot.T, A_tot @ Lambda)
            mu_n = Pn @ (np.linalg.solve(Lambda, alpha) + A_tot.T @ Qinv @(X[idx][:, t + 1][:, na] - B_tot @ U[idx][:, t]))

            Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

            X[idx][:, t] = (mu_n + np.linalg.cholesky(Pn) @ eps[:, t][:, na]).ravel()
    return X
