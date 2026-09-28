import numpy as np

from pybasicbayes.distributions import Regression
from pybasicbayes.util.stats import sample_gaussian
from pypolyagamma.binary_trees import ids, adjacency, depths

class TreeStructuredHierarchicalDynamics(object):

    def __init__(self, tree, Q, S, affine=True, S_scale=1.5):
        self.tree = tree
        self.affine = affine

        self.Q = Q
        self.P = Q.shape[0]
        assert Q.shape == (self.P, self.P)
        self.Q_inv = np.linalg.inv(self.Q)

        self.S = S
        self.D = S.shape[0] - affine
        assert S.shape == (self.D + affine, self.D + affine)

        self.adj = adjacency(tree)
        self.N = self.adj.shape[0]
        self.L = (self.N + 1) // 2

        self._ids = ids(tree)
        self.depths = depths(tree)
        self.leaves = np.array([self._ids[l] for l in range(self.L)])

        J_0 = np.zeros((self.N, self.N))
        J_0[0,0] = 1
        def _prior(node, depth):
            if np.isscalar(node):
                return
            elif isinstance(node, tuple) and len(node) == 2:
                J_0[self._ids[node], self._ids[node]] += S_scale**depth
                J_0[self._ids[node], self._ids[node[0]]] -= S_scale**depth
                J_0[self._ids[node[0]], self._ids[node]] -= S_scale**depth
                J_0[self._ids[node[0]], self._ids[node[0]]] += S_scale**depth
                _prior(node[0], depth + 1)

                J_0[self._ids[node], self._ids[node]] += S_scale**depth
                J_0[self._ids[node], self._ids[node[1]]] -= S_scale**depth
                J_0[self._ids[node[1]], self._ids[node]] -= S_scale**depth
                J_0[self._ids[node[1]], self._ids[node[1]]] += S_scale**depth
                _prior(node[1], depth+1)
        _prior(tree, 0)
        self.J_0 = np.kron(J_0, self.S)

        self.As = np.zeros((self.N, self.P, self.D))
        self.regressions = [Regression(A=self.As[l], sigma=self.Q_inv, affine=self.affine)
                            for l in self.leaves]

        self.resample()

    def resample(self, data=[]):
        N, P, D, L, affine = self.N, self.P, self.D, self.L, self.affine

        if not isinstance(data, list):
            if isinstance(data, tuple) and len(data) == 3:
                data = [data]
            else:
                raise Exception("Expected list or length-3 tuple (x, y, z)!")

        big_J = np.tile(self.J_0[:,:,None], (1, 1, P))
        big_h = np.zeros((N * (D + affine), P))

        for (x, y, z) in data:
            T = x.shape[0]

            if self.affine:
                x = np.column_stack((x, np.ones(T)))

            assert x.shape == (T, D + affine)
            assert y.shape == (T, P)

            assert z.shape == (T,)
            assert z.dtype == int and z.min() >= 0 and z.max() < L

            for l in range(L):
                inds = z == l
                xi = x[inds]
                yi = y[inds]
                xxT = (xi[:, :, None] * xi[:, None, :]).sum(0)
                yxT = (yi[:,:,None] * xi[:, None, :]).sum(0)

                j = self.leaves[l]
                for p in range(P):
                    big_J[j*D:(j+1)*D, j*D:(j+1)*D, p] += self.Q[p,p] * xxT
                    big_h[j*D:(j+1)*D, p] += self.Q[p,p] * yxT[p]

        self.As = np.zeros((N, P, D + affine))
        for p in range(P):
            self.As[:,p,:] = sample_gaussian(J=big_J[:,:,p], h=big_h[:,p]).reshape((N, D + affine))


        for l, regression in zip(self.leaves, self.regressions):
            regression.A = self.As[l]


if __name__ == "__main__":
    np.random.seed(0)

    from pypolyagamma.binary_trees import balanced_binary_tree
    tree = balanced_binary_tree(16)
    P = 1
    D = 2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    lim = 5.25
    nn = 3
    for ii in range(nn):
        for jj in range(nn):
            ax = plt.subplot(nn, nn, ii * nn + jj + 1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            for sp in ax.spines:
                ax.spines[sp].set_visible(False)

            dynamics = TreeStructuredHierarchicalDynamics(tree, np.eye(P), np.eye(D))
            D = max(dynamics.depths.values())
            for n in range(dynamics.N):
                d = dynamics.depths[n]
                plt.plot(dynamics.As[n,0,0], dynamics.As[n,0,1], 'ko', alpha=1-d/(D+1), markersize=8)

            for d in range(D):
                plt.plot(-2*lim, -2*lim, 'ko', alpha=1-d/(D+1), markersize=4, label="depth {}".format(d))

            for (i, j) in zip(*dynamics.adj.nonzero()):
                plt.plot([dynamics.As[i,0,0], dynamics.As[j,0,0]],
                         [dynamics.As[i,0,1], dynamics.As[j,0,1]],
                         '-k', alpha=0.25)

            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)

            if ii == 0 and jj == nn-1:
                plt.legend(loc="lower right")

            plt.title("Tree {}".format(nn*ii + jj + 1))

    plt.tight_layout()
    plt.show()
    
