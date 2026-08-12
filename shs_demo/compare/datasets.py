"""Dataset loaders for the shs_demo comparison suite.

Every loader returns the same bundle so the three runners (SHS-RSSM, rSLDS,
TrSLDS) consume identical data:

    bundle = dict(
        name        str
        seqs        list of (T_i, D) float64 arrays, one per sequence
        z_true      list of (T_i,) int arrays, 0-based ground-truth regimes
        doc_range   (S+1,) int array of cumulative sequence boundaries
        K_true      int
        D           int, observation dimension
        state_names list of str or None
        x_true      list of (T_i, D_lat) arrays or None  (nascar only)
    )

NASCAR is generated once and cached to ``data_cache/`` so that runs in the
modern environment (SHS, TrSLDS) and the legacy rSLDS environment see the
*same* realisation.  The generator is a pure-numpy port of
``recurrent-slds-master/examples/nascar.py`` (Linderman et al., AISTATS 2017):
recurrence-only stick-breaking transitions over four regimes -- two rotations
around (+/-2, 0) plus two straight-line regimes -- observed through a random
linear projection with additive Gaussian noise.
"""
import os
import pathlib

import numpy as np
from scipy.special import digamma, expit

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "data_cache"


# --------------------------------------------------------------------- helpers
def _bundle(name, seqs, z_true, K_true, state_names=None, x_true=None):
    lens = [len(s) for s in seqs]
    return dict(name=name, seqs=seqs, z_true=z_true,
                doc_range=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                K_true=int(K_true), D=int(seqs[0].shape[1]),
                state_names=state_names, x_true=x_true)


def concat(bundle):
    """(X, Z) concatenated over sequences."""
    return (np.concatenate(bundle["seqs"], 0),
            np.concatenate(bundle["z_true"], 0))


# ---------------------------------------------------------------------- NASCAR
def _compute_psi_cmoments(alphas):
    """Stick-breaking asymmetry correction, as pypolyagamma.compute_psi_cmoments.

    mu_k = E[psi_k] under pi ~ Dir(alphas):  digamma(a_k) - digamma(sum_{j>k} a_j).
    """
    K = len(alphas)
    return np.array([digamma(alphas[k]) - digamma(alphas[k + 1:].sum())
                     for k in range(K - 1)])


def _pi_stick_breaking(nu):
    """nu (K-1,) -> pi (K,) via the logistic stick-breaking map."""
    s = expit(nu)
    stick = np.concatenate([[1.0], np.cumprod(1.0 - s)])
    return np.concatenate([s, [1.0]]) * stick


def _simulate_nascar(n_seq=5, T=2000, D_obs=10, seed=0):
    """Faithful numpy port of examples/nascar.py::simulate_nascar (rSLDS(ro))."""
    rng = np.random.RandomState(seed)
    D_lat, K = 2, 4

    def random_rotation(theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        q = np.linalg.qr(rng.randn(D_lat, D_lat))[0]
        return q @ rot @ q.T

    As = [random_rotation(np.pi / 24.), random_rotation(np.pi / 48.)]
    centers = [np.array([+2.0, 0.0]), np.array([-2.0, 0.0])]
    bs = [-(A - np.eye(D_lat)) @ c for A, c in zip(As, centers)]
    As += [np.eye(D_lat), np.eye(D_lat)]
    bs += [np.array([+0.1, 0.0]), np.array([-0.35, 0.0])]

    # recurrence-only stick-breaking transitions (decision-list order 0,1,2,3)
    scale = 100.0
    W = scale * np.array([[+1.0, 0.0], [-1.0, 0.0], [0.0, +1.0]])
    b = scale * np.array([-2.0, -2.0, 0.0]) + _compute_psi_cmoments(np.ones(K))

    C = rng.randn(D_obs, D_lat)
    chol_Q = np.sqrt(1e-4)
    sd_R = np.sqrt(1e-5)

    seqs, zs, xs = [], [], []
    for _ in range(n_seq):
        x = np.zeros((T, D_lat))
        z = np.zeros(T, dtype=int)
        x[0] = np.array([0.0, 1.0]) + np.sqrt(1e-3) * rng.randn(D_lat)
        z[0] = rng.randint(K)
        for t in range(1, T):
            pi = _pi_stick_breaking(W @ x[t - 1] + b)
            z[t] = rng.choice(K, p=pi)
            x[t] = As[z[t]] @ x[t - 1] + bs[z[t]] + chol_Q * rng.randn(D_lat)
        y = x @ C.T + sd_R * rng.randn(T, D_obs)
        seqs.append(y)
        zs.append(z)
        xs.append(x)
    return seqs, zs, xs


def load_nascar(n_seq=5, T=2000, D_obs=10, seed=0, refresh=False):
    """Synthetic NASCAR (Linderman et al. 2017), cached for cross-env identity."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"nascar_seed{seed}_N{n_seq}_T{T}_D{D_obs}.npz"
    if f.exists() and not refresh:
        d = np.load(f)
        S = len(d["doc_range"]) - 1
        cuts = d["doc_range"]
        seqs = [d["Y"][cuts[i]:cuts[i + 1]] for i in range(S)]
        zs = [d["Z"][cuts[i]:cuts[i + 1]] for i in range(S)]
        xs = [d["Xlat"][cuts[i]:cuts[i + 1]] for i in range(S)]
    else:
        seqs, zs, xs = _simulate_nascar(n_seq, T, D_obs, seed)
        np.savez(f, Y=np.concatenate(seqs, 0), Z=np.concatenate(zs, 0),
                 Xlat=np.concatenate(xs, 0),
                 doc_range=np.concatenate([[0], np.cumsum([len(s) for s in seqs])]))
    names = ["curve-right", "curve-left", "straight-fwd", "straight-back"]
    return _bundle("nascar", seqs, zs, 4, names, x_true=xs)


# -------------------------------------------------------------------- ToyARK13
def load_toyark13(n_seq=12, path=None):
    """Hughes/Sudderth ToyARK13 (x-hdphmm-nips2015): 13 AR regimes, 3-D, T=800."""
    from scipy.io import loadmat
    path = path or (HERE.parent / "toyark13" / "HMMdataset.mat")
    M = loadmat(str(path))
    dr = M["doc_range"].ravel()
    n_seq = min(n_seq, len(dr) - 1)
    seqs = [M["X"][dr[i]:dr[i + 1]].astype(np.float64) for i in range(n_seq)]
    Zall = M["TrueZ"].ravel().astype(int)
    zmin = Zall.min()
    zs = [Zall[dr[i]:dr[i + 1]] - zmin for i in range(n_seq)]
    K = int(Zall.max() - zmin + 1)
    return _bundle("toyark13", seqs, zs, K)


# ---------------------------------------------------------------------- mocap6
def load_mocap6(standardize=True, path=None):
    """mocap6 (Fox, Hughes, Sudderth & Jordan, AOAS 2014): 6 seqs, 12 channels,
    12 annotated exercise regimes."""
    path = path or (HERE.parent / "mocap6" / "dataset.npz")
    d = np.load(str(path), allow_pickle=True)
    X, dr = d["X"].astype(np.float64), d["doc_range"].astype(int)
    Z = d["TrueZ"].astype(int)
    zmin = Z.min()
    if standardize:  # per-channel, corpus-wide (bnpy convention)
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    seqs = [X[dr[i]:dr[i + 1]] for i in range(len(dr) - 1)]
    zs = [Z[dr[i]:dr[i + 1]] - zmin for i in range(len(dr) - 1)]
    names = [str(s) for s in d["true_state_names"]] if "true_state_names" in d else None
    return _bundle("mocap6", seqs, zs, int(Z.max() - zmin + 1), names)


LOADERS = dict(nascar=load_nascar, toyark13=load_toyark13, mocap6=load_mocap6)


def load(name, **kw):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; options: {sorted(LOADERS)}")
    return LOADERS[name](**kw)
