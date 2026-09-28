import numpy as np
import numpy.random as npr
from numpy import newaxis as na
from scipy.stats import invwishart
from . import conditionals
import torch
from torch.autograd import Variable
import torch.optim as optim
import copy
from tqdm import tqdm
from scipy.ndimage import filters
from scipy.signal import gaussian
from numba import njit, jit


def compute_ss_mniw(X, Y, nu, Lambda, M, V ):
    df_posterior = nu + X[:, 0].size

    Vinv = np.linalg.inv(V)
    Ln = X.T @ X + Vinv
    Bn = np.linalg.solve(Ln, X.T @ Y + Vinv @ M.T)
    IW_matrix = Lambda + (Y - X @ Bn).T @ (Y - X @ Bn) + (Bn - M.T).T @  Vinv @ (Bn - M.T)
    IW_matrix = (IW_matrix + IW_matrix.T) / 2

    M_posterior = Bn.T
    V_posterior = np.linalg.inv(Ln)
    return M_posterior, V_posterior, IW_matrix, df_posterior


def sample_mniw(nu, L, M, S):
    Q = invwishart.rvs(nu, L)
    A = npr.multivariate_normal(M.flatten(order='F'), np.kron(S, Q)).reshape(M.shape, order='F')
    return A, Q


def rotate_latent(states, O):
    return [O @ states[idx] for idx in range(len(states))]


def rotate_dynamics(A, O, depth):
    for level in range(depth):
        for node in range(2 ** level):
            A[level][:, :-1, node] = O @ A[level][:, :-1, node] @ O.T
            A[level][:, -1, node] = (O @ A[level][:, -1, node][:, na]).ravel()
    return A


def sample_hyperplanes(states, omega, paths, depth, prior_mu, prior_tau, possible_paths, R):
    X = np.hstack(states)
    X = np.vstack((X, np.ones((1, X[0, :].size))))
    W = np.hstack(omega)
    path = np.hstack(paths)

    for level in range(depth - 1):
        for node in range(2 ** level):
            if ~np.isnan(possible_paths[level + 1, 2 * node + 1]):
                indices = path[level, :] == (node + 1)
                effective_x = X[:, indices]
                effective_w = W[level, indices]
                effective_z = path[level + 1, indices]

                draw_prior = indices.size == 0

                R[level][:, node] = conditionals.hyper_planes(effective_w, effective_x, effective_z,
                                                              prior_mu, prior_tau, draw_prior)
    return R


def sample_internal_dynamics(A, scale, Mx, Vx, depth):
    for level in range(depth - 2, -1, -1):
        for node in range(2 ** level):
            Achild = 0

            if not np.isnan(A[level + 1][0, 0, 2 * node]):
                if level == 0:
                    Mprior = Mx + 0
                else:
                    Mprior = A[level - 1][:, :, int(np.floor(node/2))] + 0

                for child in range(2):
                    Achild += A[level + 1][:, :, 2 * node + child]

                A[level][:, :, node] = conditionals._internal_dynamics(Mprior, scale ** level * Vx, Achild,
                                                                   scale ** (level + 1) * Vx)
    return A


def sample_leaf_dynamics(states, inputs, discrete_states, A, Q, nu, lambdax, Mx, Vx, scale, leaf_nodes):
    X = np.hstack([states[idx][:, :-1] for idx in range(len(states))])
    U = np.hstack([inputs[idx][:, :-1] for idx in range(len(inputs))])
    X = np.vstack((X, U))

    Y = np.hstack([states[idx][:, 1:] for idx in range(len(states))])

    Z = np.hstack([discrete_states[idx][:-1] for idx in range(len(states))])
    
    for (d, node, k) in leaf_nodes:
        indices = Z == k

        effective_X = X[:, indices]
        effective_Y = Y[:, indices]
        if d == 0:
            Mprior = Mx
        else:
            Mprior = A[d-1][:, :, int(np.floor(node/2))]

        draw_prior = effective_X.size == 0
        A[d][:, :, int(node)], Q[:, :, int(k)] = conditionals.leaf_dynamics(effective_Y, effective_X, nu, lambdax,
                                                                            Mprior, scale ** d * Vx, draw_prior)
    return A, Q


def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def sigmoid_vectorized(x):
    z = np.zeros(x.size)
    z[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
    z[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return z


def log_mvn(x, mu, tau, logdet):
    return np.diag(-0.5 * logdet-0.5 * (x - mu).T @ tau @ (x - mu))


def compute_leaf_log_prob(R, x, K, depth, leaf_paths):
    log_prob = np.zeros(K)
    for k in range(K):
        "Compute prior probabilities of each path"
        for level in range(depth - 1):
            node = int(leaf_paths[level, k])
            child = leaf_paths[level + 1, k]
            if ~np.isnan(child):
                v = np.matmul(R[level][:-1, node - 1], x) + R[level][-1, node - 1]
                if int(child) % 2 == 1:
                    log_prob[k] += np.log(sigmoid(v))
                else:
                    log_prob[k] += np.log(sigmoid(-v))
    return log_prob


def compute_leaf_log_prob_vectorized(R, x, K, depth, leaf_paths):
    log_prob = np.zeros((K, x[0, :].size))
    for k in range(K):
        "Compute prior probabilities of each path"
        for level in range(depth - 1):
            node = int(leaf_paths[level, k])
            child = leaf_paths[level + 1, k]
            if ~np.isnan(child):
                v = (R[level][:-1, node - 1][na, :] @ x).flatten() + R[level][-1, node - 1]
                if int(child) % 2 == 1:
                    log_prob[k, :] = log_prob[k, :] + np.log(sigmoid_vectorized(v))
                else:
                    log_prob[k, :] = log_prob[k, :] + np.log(sigmoid_vectorized(-v))
    return log_prob


def create_balanced_binary_tree(K):
    depth = int(np.ceil(np.log2(K)) + 1)
    K_perf = 2 ** (depth - 1)
    possible_paths = np.ones((depth, K_perf))
    for d in range(1, depth):
        temp = np.arange(0, 2 ** int(d)) + 1
        possible_paths[d, :] = np.repeat(temp, int(K_perf / temp.size))

    right = (2 ** (depth - 1) - K) // 2
    left = (2 ** (depth - 1) - K) - right
    split = K_perf // 2
    possible_paths[-1, :2 * left] = np.nan
    possible_paths[-1, split:split + 2 * right] = np.nan

    leaf_path = np.zeros((depth, K))
    indic = False
    counter = 0
    for n in range(K_perf):
        if n == 0:
            leaf_path[:, counter] = possible_paths[:, n]
            indic = np.isnan(possible_paths[-1, n])
            counter += 1
        elif indic == False:
            leaf_path[:, counter] = possible_paths[:, n]
            indic = np.isnan(possible_paths[-1, n])
            counter += 1
        else:
            indic = False

    leaf_nodes = []
    for d in range(depth - 2, depth):
        for k in range(K):
            if d == depth - 1:
                if not np.isnan(leaf_path[d, k]):
                    leaf_nodes.append((int(d), int(leaf_path[d, k] - 1), int(k)))
            else:
                if np.isnan(leaf_path[d + 1, k]):
                    leaf_nodes.append((int(d), int(leaf_path[d, k] - 1), int(k)))

    return depth, leaf_path, possible_paths, leaf_nodes


def create_batches(batch_size, N):
    if batch_size == N:
        idx = [np.arange(N)]
    else:
        indices = npr.choice(N, N).astype('int')
        numBatches = np.ceil(N / batch_size).astype('int')
        idx = [indices[i * batch_size: (i + 1) * batch_size] for i in range(numBatches)]
    return idx


def compute_residual(X, Y, lds, nu, anc_weights, leaf_weights, y_leafs, temper, num_hp, K):
    counter = 0
    for h in range(num_hp):
        leaf_weights[counter, :] = torch.mul(anc_weights[counter, :],
                                             torch.sigmoid(temper * torch.matmul(X.transpose(0, 1), nu[:, h])))
        leaf_weights[counter + 1, :] = torch.mul(anc_weights[counter + 1, :], torch.sigmoid(
            -temper * torch.matmul(X.transpose(0, 1), nu[:, h])))
        counter += 2

    for k in range(K):
        y_leafs[:, :, k] = torch.mul(leaf_weights[k, :], torch.matmul(lds[:, :, k], X))

    y_pred = torch.sum(y_leafs, 2)
    resid = Y - y_pred
    return resid


def optimize_tree(y, x, LDS, nu, ancestor_weights, K, num_hp, epochs, batch_size, LR, temper):
    N = int(x[:, 0].size)
    dy = y[0, :].size
    input_data = torch.from_numpy(x.T).double()
    output_data = torch.from_numpy(y.T).double()
    lds = Variable(torch.from_numpy(LDS), requires_grad=True).double()
    nu = Variable(torch.from_numpy(nu), requires_grad=True).double()
    prev_weights = torch.from_numpy(ancestor_weights).double()
    losses = []
    optimizer = optim.Adam([lds, nu], lr=LR)
    for epoch in tqdm(range(epochs)):
        "Create mini batches"
        batch_idx = create_batches(batch_size, N)
        for idx in batch_idx:
            optimizer.zero_grad()
            X = input_data[:, idx]
            Y = output_data[:, idx]
            anc_weights = prev_weights[:, idx]
            leaf_weights = torch.zeros(K, idx.size).double()
            y_leafs = torch.zeros(dy, idx.size, K).double()
            resid = compute_residual(X, Y, lds, nu, anc_weights, leaf_weights, y_leafs, temper, num_hp, K)

            loss = 0.5 * torch.matmul(resid, resid.transpose(0, 1)).trace() / idx.size

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

    with torch.no_grad():
        leaf_weights = torch.zeros(K, N).double()
        y_leafs = torch.zeros(dy, N, K).double()
        resid = compute_residual(input_data, output_data, lds, nu, prev_weights, leaf_weights,
                                 y_leafs, temper, num_hp, K)
    return lds, nu, resid.detach().numpy().T, losses


def projection(xreal, xinferr):
    Xreals = np.hstack(xreal).T
    Xrot = np.hstack(xinferr).T
    Xrot = np.hstack((Xrot, np.ones((Xrot[:, 0].size, 1))))
    transform = np.linalg.lstsq(Xrot, Xreals)[0].T
    return transform


def generate_trajectory(A, Q, R, starting_pt, depth, leaf_path, K, T, D_in, noise=True, u=None, D_bias=None):
    if u is D_bias is None:
            u = np.ones((1, T))
            D_bias = 1
    x = np.zeros((D_in, T + 1))
    x[:, 0] = starting_pt
    z = np.zeros(T + 1).astype(int)
    for t in range(T):
        log_p = compute_leaf_log_prob(R, x[:, t], K, depth, leaf_path)
        p_unnorm = np.exp(log_p - np.max(log_p))
        p = p_unnorm/np.sum(p_unnorm)
        if noise:
                choice = npr.multinomial(1, p.ravel(), size=1)
                z[t] = np.where(choice[0, :] == 1)[0][0].astype(int)
                x[:, t + 1] = (A[:, :-D_bias, z[t]] @ x[:, t][:, na] +  \
                              A[:, -D_bias:, z[t]] @ u[:, t][:, na] + \
                              npr.multivariate_normal(np.zeros(D_in), Q[:, :, z[t]])[:, na]).flatten()

        else:
            z[t] = np.argmax(choice)
            x[:, t + 1] = (A[:, :-D_bias, z[t]] @ x[:, t][:, na] + \
                           A[:, -D_bias:, z[t]] @ u[:, t][:, na]).flatten()

    log_p = compute_leaf_log_prob(R, x[:, -1], K, depth, leaf_path)
    p_unnorm = np.exp(log_p - np.max(log_p))
    p = p_unnorm / np.sum(p_unnorm)
    choice = npr.multinomial(1, p.ravel(), size=1)
    z[-1] = np.where(choice[0, :] == 1)[0][0]
    return x, z


def MAP_dynamics(x, u, z, Ainit, Qinit, nux, lambdax, Mx, Vx, scale, leaf_nodes, K, depth, no_samples):
    A_est = []
    Q_est = []
    At = copy.deepcopy(Ainit)
    Qt = copy.deepcopy(Qinit)
    for m in tqdm(range(no_samples)):
        At, Qt = sample_leaf_dynamics(x, u, z, At, Qt, nux, 
                                            lambdax, Mx, Vx, scale, leaf_nodes)
        At = sample_internal_dynamics(At, scale, Mx, Vx, depth)
        if m > no_samples/2:
            A_est.append(copy.deepcopy(At))
            Q_est.append(copy.deepcopy(Qt))
    
    Z = len(A_est)
    for d in range(depth):
        for node in range(2**d):
            At[d][:, :, node] = A_est[0][d][:, :, node] / Z
    Qt = Q_est[0]/Z
    for sample in tqdm(range(1, len(A_est))):
        for k in range(K):
            Qt[:, :, k] += Q_est[sample][:, :, k]/Z
        for d in range(depth):
            for node in range(2**d):
                At[d][:, :, node] += A_est[sample][d][:, :, node] / Z
    return At, Qt


def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y[:, 0].size
    for neuron in range(neurons):
        y_smooth[neuron, :] = filters.convolve1d(y[neuron, :], b)/b.sum()
    return y_smooth
