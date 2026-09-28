from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, gammaln, logsumexp
import torch

from retention_arm import FACTORS, RetentionRule, summary as retention_summary

K, D, AD = 3, 2, 2
SIGMA, ACTION_SCALE = .15, .30
CENTERS = np.array([[-2., -1.], [2., -1.], [0., 2.]])
BASE_A = np.array([[[.78, -.12], [.04, .32]],
                   [[.60, .20], [-.08, .25]],
                   [[.86, -.08], [.10, .45]]])
FINAL_ANGLES = np.deg2rad([65., -55., 80.])
P_TRUE = .9*np.eye(K) + .1*np.ones((K, K))/K
ARMS = ("ema", "ema_slow", "ema_moves", "ema_gated", "ema_retain", "svb", "window", "cumavg")
KIND = dict(ema="ema", ema_slow="ema", ema_moves="ema", ema_gated="gated", ema_retain="retain", svb="svb",
            window="window", cumavg="cumavg")
GATED = ("gated", "retain")
ORACLE_ARMS = ("ema", "ema_slow", "svb", "window", "cumavg")
COLORS = dict(ema_retain="#2a78d6", ema="#eb6834", ema_gated="#1baf7a", ema_slow="#eda100", window="#e87ba4",
              svb="#008300", ema_moves="#5c5c5c", cumavg="#9b9b9b")
ORDER = ("ema_retain", "ema_gated", "ema", "ema_moves", "ema_slow", "window", "svb", "cumavg")
PAPER_STYLE = {"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "Times",
                                                      "STIXGeneral", "DejaVu Serif"],
               "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10.5, "legend.fontsize": 9,
               "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.linewidth": .8, "axes.edgecolor": ".15",
               "axes.spines.top": True, "axes.spines.right": True, "axes.axisbelow": True, "grid.color": ".88",
               "grid.linewidth": .6, "legend.frameon": False, "savefig.facecolor": "white", "svg.fonttype": "none",
               "pdf.fonttype": 42, "ps.fonttype": 42}
DORMANT = K-1
REGIME_COLORS = ("#0072B2", "#D55E00", "#009E73")
KEYS_K = ("N", "Srr", "Szr", "Szz", "Shh", "Szh", "Srh", "start", "pg_A", "pg_h")
DYN_KEYS = ("N", "Srr", "Szr", "Szz", "Shh", "Szh", "Srh")


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source(root):
    root = Path(root)/"shs_rssm"
    dynamics = import_file(root/"regimes_shared.py", "aime_dynamics")
    gate = import_file(root/"recurrent_stick.py", "aime_gate")
    fb = import_file(root/"forward_backward.py", "aime_fb")
    forgetting = import_file(root/"forgetting.py", "aime_forgetting")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [root/"regimes_shared.py", root/"recurrent_stick.py", root/"forward_backward.py",
                        root/"forgetting.py"]}
    return dynamics.SharedCarryRegimes, gate.RecurrentStickiness, fb.forward_backward, hashes, forgetting


def fraction_at(b, cfg, condition):
    if condition in ("stationary", "recurring"):
        return 0.0
    if condition == "abrupt":
        return float(b >= cfg["change_start"])
    return float(np.clip((b-cfg["change_start"]+1)/cfg["rotation_steps"], 0., 1.))


def centers_at(fraction, cfg):
    """Fixed points revolve about the origin by fraction*center_angle."""
    phi = fraction*np.deg2rad(cfg["center_angle"])
    Rc = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return CENTERS@Rc.T


def maps_at(fraction):
    angles = fraction*FINAL_ANGLES
    R = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angles])
    return R@BASE_A@np.swapaxes(R, -1, -2)


def parameters(b, cfg, condition="moving"):
    fraction = fraction_at(b, cfg, condition)
    A = maps_at(fraction)
    centers = centers_at(fraction, cfg)
    c = centers-np.einsum("kij,kj->ki", A, centers)
    return A, c

def chain_at(b, cfg, condition):
    """Transition matrix and initial law: in the recurring condition the last regime is unreachable
    for rotation_steps updates from change_start, then returns unchanged."""
    if condition == "recurring" and cfg["change_start"] <= b < cfg["change_start"]+cfg["rotation_steps"]:
        P = P_TRUE.copy(); P[:, DORMANT] = 0.; P /= P.sum(-1, keepdims=True)
        init = np.ones(K)/(K-1); init[DORMANT] = 0.
        return P, init
    return P_TRUE, np.ones(K)/K


def simulate_batch(rng, A, c, batch, length, P=P_TRUE, init=None):
    init = np.ones(K)/K if init is None else init
    x = np.empty((batch, length+1, D))
    x[:, 0] = rng.normal(size=(batch, D))
    actions = rng.normal(size=(batch, length, AD))
    eps = rng.normal(scale=SIGMA, size=(batch, length, D))
    uniforms = rng.random((batch, length))
    regimes = np.empty((batch, length), dtype=int)
    z = np.minimum((uniforms[:, 0, None] > np.cumsum(init)).sum(-1), K-1)
    for t in range(length):
        if t:
            z = (uniforms[:, t, None] > np.cumsum(P[z], axis=-1)).sum(-1)
        regimes[:, t] = z
        x[:, t+1] = np.einsum("bij,bj->bi", A[z], x[:, t]) + c[z] + ACTION_SCALE*actions[:, t] + eps[:, t]
    return x, actions, regimes


def prepare(x, actions):
    prev = x[:, :-1]
    y = x[:, 1:]
    g = np.concatenate((prev, actions, np.ones((*prev.shape[:2], 1))), axis=-1)
    phi = np.concatenate((prev/2., np.ones((*prev.shape[:2], 1))), axis=-1)
    return dict(y=torch.from_numpy(y), g=torch.from_numpy(g), phi=torch.from_numpy(phi),
                y_np=y, g_np=g, phi_np=phi)


def fast_fb(log_init, log_trans, ev):
    """Log-space HMM forward-backward smoother."""
    bsz, length, k = ev.shape
    a = np.empty_like(ev)
    a[:, 0] = log_init + ev[:, 0]
    for t in range(1, length):
        a[:, t] = logsumexp(a[:, t-1, :, None]+log_trans[:, t-1], axis=-2)+ev[:, t]
    z = logsumexp(a[:, -1], axis=-1)
    back = np.zeros_like(ev)
    for t in range(length-1, 0, -1):
        back[:, t-1] = logsumexp(log_trans[:, t-1]+(ev[:, t]+back[:, t])[:, None, :], axis=-1)
    gamma = a+back-z[:, None, None]
    gamma = np.exp(gamma-logsumexp(gamma, axis=-1, keepdims=True))
    lx = a[:, :-1, :, None]+log_trans+(ev[:, 1:]+back[:, 1:])[:, :, None, :]
    lx -= logsumexp(lx, axis=(-2, -1), keepdims=True)
    return torch.from_numpy(gamma), torch.from_numpy(np.exp(lx)), z


def clone_stats(s):
    return {k:v.clone() for k,v in s.items()}


class Statistics:
    """Propose/refine one message without counting it once per VB iteration."""
    def __init__(self, rule, initial, tau, window, forgetting=None):
        self.rule, self.tau, self.window = rule, tau, window
        self.total = clone_stats(initial)
        self.count = 1
        self.history = deque([clone_stats(initial)]) if rule == "window" else None
        self.window_sum = clone_stats(initial) if rule == "window" else None
        self.retain = RetentionRule(forgetting, tau, retain=rule == "retain") if rule in GATED else None

    def propose(self, s, r=None):
        if self.rule in GATED:
            return self.retain.propose(self.total, s, r or {})
        if self.rule == "ema":
            return {k:(1-self.tau)*self.total[k]+self.tau*v for k,v in s.items()}
        if self.rule == "svb":
            return {k:self.total[k]+v for k,v in s.items()}
        if self.rule == "cumavg":
            return {k:(self.count*self.total[k]+v)/(self.count+1) for k,v in s.items()}
        if self.rule == "window":
            drop = self.history[0] if len(self.history) == self.window else None
            n = min(len(self.history)+1, self.window)
            return {k:(self.window_sum[k]+v-(drop[k] if drop is not None else 0.))/n for k,v in s.items()}
        raise ValueError(self.rule)

    def commit(self, s, r=None):
        new = self.propose(s, r)
        if self.rule == "window":
            if len(self.history) == self.window:
                old = self.history.popleft()
                for key in s: self.window_sum[key] -= old[key]
            self.history.append(clone_stats(s))
            for key in s: self.window_sum[key] += s[key]
        self.total = new
        self.count += 1


class SwitchingCore:
    def __init__(self, classes, k=K):
        dyn, gate = classes[:2]
        self.K = int(k)
        self.dyn = dyn(K=self.K, L=D, G=D+AD+1, action_dim=AD, ard=False,
                       identity_init=False, learn_b0=False, a0=3., b0=2*SIGMA**2,
                       v0_scale=.2, jitter=0., n_block_iters=1,
                       device=torch.device("cpu"), dtype=torch.float64)
        self.dyn._freeze_C = True
        self.gate = gate(K=self.K, feat_dim=D, prior_persist=.9, weight_prior_var=.25,
                         bias_prior_var=4., pg_iters=1,
                         device=torch.device("cpu"), dtype=torch.float64)
        self.prior_dest = torch.ones(self.K, self.K, dtype=torch.float64)
        self.prior_start = torch.ones(self.K, dtype=torch.float64)
        self.dest = self.prior_dest.clone()
        self.start = self.prior_start.clone()
        self.last_guard_count = 0

    def messages(self, data):
        ev = self.dyn.expected_loglik(data["y"], data["g"])
        elogpi = torch.digamma(self.dest)-torch.digamma(self.dest.sum(-1, keepdim=True))
        eloginit = torch.digamma(self.start)-torch.digamma(self.start.sum())
        logtrans, aux = self.gate.bound_log_trans(elogpi, data["phi"][:, 1:])
        gamma, xi, logz = fast_fb(eloginit.numpy(), logtrans.numpy(), ev.numpy())
        self.last_logz = float(np.sum(logz))
        stats = self.dyn.stats_from_batch(gamma, data["y"], data["g"])
        r, w, C = self.gate.attribute_bound(xi, aux)
        phi = data["phi"][:, 1:].reshape(-1, D+1)
        pg = self.gate.pg_stats_from_batch(phi, r.reshape(-1, self.K), w.reshape(-1, self.K))
        stats.update(C=C, start=gamma[:, 0].sum(0), pg_A=pg["A"], pg_h=pg["h"])
        return stats, gamma.numpy()

    def refit(self, stats, base=None, message=None, tau=None, retain=None):
        if base is not None:
            self.dyn.set_stats({k:base[k] for k in DYN_KEYS})
            self.dyn.ema_update_stats({k:message[k] for k in DYN_KEYS}, tau, retain=retain)
        else:
            self.dyn.set_stats({k:stats[k] for k in DYN_KEYS})
        self.dyn.m_step()
        self.dest = self.prior_dest + stats["C"]
        self.start = self.prior_start + stats["start"]
        self.gate.pg_set_totals(stats["pg_A"], stats["pg_h"])
        if getattr(self.gate, "n_pg_guard_rejects", 0):
            raise FloatingPointError("AIME gate rejected an update; retain and investigate this run")
        if not torch.isfinite(self.dyn.M).all() or not (self.dyn.b > 0).all():
            raise FloatingPointError("Invalid conjugate dynamics refit")

    def predict_nll(self, data):
        g, y, phi = data["g_np"], data["y_np"], data["phi_np"]
        M = self.dyn.M.numpy(); V = self.dyn.V.numpy()
        a, b = self.dyn.a.numpy(), self.dyn.b.numpy()
        mu = np.einsum("klr,btr->btkl", M, g)
        qf = np.einsum("btr,krs,bts->btk", g, V, g)
        varscale = (b/a)[None, None]*(1+qf[..., None])
        df = 2*a
        residual = (y[:, :, None]-mu)**2/varscale
        logev = (gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi)
                 -.5*np.log(varscale)-.5*(df+1)*np.log1p(residual/df)).sum(-1)
        gm = np.einsum("btd,kd->btk", phi, self.gate.m_beta.numpy())
        gv = np.einsum("btd,kde,bte->btk", phi, self.gate.Sigma_beta.numpy(), phi)
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        stay = (expit(gm[..., None]+np.sqrt(2*np.maximum(gv, 0))[..., None]*nodes)*weights).sum(-1)/np.sqrt(np.pi)
        dest = self.dest.numpy(); dest = dest/dest.sum(-1, keepdims=True)
        transition = stay[..., :, None]*np.eye(self.K)+(1-stay[..., :, None])*dest
        init = self.start.numpy(); init = init/init.sum()
        logalpha = np.log(init)[None]+logev[:, 0]
        z = logsumexp(logalpha, axis=-1)
        total = z.copy(); logalpha -= z[:, None]
        for t in range(1, y.shape[1]):
            predicted = logsumexp(logalpha[:, :, None]+np.log(transition[:, t]), axis=-2)
            logalpha = predicted+logev[:, t]
            z = logsumexp(logalpha, axis=-1)
            total += z; logalpha -= z[:, None]
        return -float(total.mean())/y.shape[1]


def align_once(gamma, z):
    pred = gamma.argmax(-1)
    conf = np.array([[(pred[z==truth] == fitted).sum() for fitted in range(K)] for truth in range(K)])
    truth, fitted = linear_sum_assignment(-conf)
    mapping = np.empty(K, dtype=int); mapping[fitted] = truth
    return mapping


def initialize(classes, rng, cfg):
    A, c = parameters(0, cfg, "stationary")
    warm_batches = cfg["warm_batches"]
    x, u, z = simulate_batch(rng, A, c, cfg["batch"]*warm_batches, cfg["length"])
    data = prepare(x, u)
    model = SwitchingCore(classes)
    _, labels = kmeans2(data["y_np"].reshape(-1, D), K, iter=30, minit="++", seed=rng)
    gamma = torch.from_numpy(np.eye(K)[labels].reshape(*z.shape, K))
    stats = model.dyn.stats_from_batch(gamma, data["y"], data["g"])
    stats.update(C=torch.zeros(K, K, dtype=torch.float64),
                 start=gamma[:, 0].sum(0), pg_A=model.gate.pg_A.clone(), pg_h=model.gate.pg_h.clone())
    stats = {k:v/warm_batches for k,v in stats.items()}
    model.refit(stats)
    for _ in range(cfg["warm_iters"]):
        stats, gamma = model.messages(data)
        stats = {k:v/warm_batches for k,v in stats.items()}
        model.refit(stats)
    mapping = align_once(gamma, z)
    return model, stats, mapping, float(np.mean(mapping[gamma.argmax(-1)] != z))



def arm_taus(cfg):
    return dict(ema=cfg["tau"], ema_slow=cfg["tau_slow"], ema_moves=cfg["tau"], ema_gated=cfg["tau"],
                ema_retain=cfg["tau"], svb=None, window=None, cumavg=None)


def dirichlet_kl(q, p):
    q0, p0 = q.sum(-1), p.sum(-1)
    kl = (torch.lgamma(q0)-torch.lgamma(q).sum(-1)-torch.lgamma(p0)+torch.lgamma(p).sum(-1)
          + ((q-p)*(torch.digamma(q)-torch.digamma(q0).unsqueeze(-1))).sum(-1))
    return float(kl.sum())


def global_kl(model):
    """KL(q(Theta)||p(Theta)) of every global factor that the moves change."""
    return (float(model.dyn.param_kl().sum())+float(model.gate.beta_kl().sum())
            + dirichlet_kl(model.dest, model.prior_dest)+dirichlet_kl(model.start, model.prior_start))


def stats_select(s, keep):
    keep = torch.as_tensor(list(keep), dtype=torch.long)
    out = {k: s[k].index_select(0, keep).clone() for k in KEYS_K}
    out["C"] = s["C"].index_select(0, keep).index_select(1, keep).clone()
    return out


def stats_merge(s, i, j):
    """AIME merge semantics: every statistic of j is added to i, then j is dropped.
    Transition counts merge rows and columns, M'_ii = M_ii+M_jj+M_ij+M_ji."""
    out = {k: v.clone() for k, v in s.items()}
    for k in KEYS_K:
        out[k][i] = out[k][i]+out[k][j]
    out["C"][i, :] += out["C"][j, :]
    out["C"][:, i] += out["C"][:, j]
    return stats_select(out, [k for k in range(int(s["N"].shape[0])) if k != j])


def stats_append(s, dyn_new):
    """Append one slot: dynamics statistics from dyn_new, prior (zero) counts and gate totals."""
    k0 = int(s["N"].shape[0])
    out = {}
    for k in KEYS_K:
        new = dyn_new[k] if k in dyn_new else torch.zeros_like(s[k][:1])
        out[k] = torch.cat([s[k], new.reshape(1, *s[k].shape[1:]).to(s[k].dtype)], 0)
    C = torch.zeros(k0+1, k0+1, dtype=s["C"].dtype); C[:k0, :k0] = s["C"]
    out["C"] = C
    return out


def refine_and_score(classes, stats0, window, iters):
    model = SwitchingCore(classes, k=int(stats0["N"].shape[0]))
    model.refit(stats0)
    stats = stats0
    for _ in range(iters):
        acc = None
        for data in window:
            m, _ = model.messages(data)
            acc = m if acc is None else {k: acc[k]+m[k] for k in m}
        stats = {k: v/len(window) for k, v in acc.items()}
        model.refit(stats)
    logz = []
    for data in window:
        model.messages(data)
        logz.append(model.last_logz)
    return model, stats, float(np.mean(logz))-global_kl(model)


def _pooled(window, sel_fn):
    ys, gs = [], []
    for data in window:
        sel = sel_fn(data)
        ys.append(data["y_np"][sel]); gs.append(data["g_np"][sel])
    return np.concatenate(ys), np.concatenate(gs)


def split_proposal(model, stats, window, k, rng):
    """Split slot k: 2-means on (x_t, x_{t-1}) of the steps it owns; child 0 replaces k."""
    owned = {id(d): model.messages(d)[1].argmax(-1) == k for d in window}
    y, g = _pooled(window, lambda d: owned[id(d)])
    if len(y) < 40:
        return None
    _, lab = kmeans2(np.concatenate([y, g[:, :D]], 1), 2, iter=25, minit="++", seed=rng)
    if min((lab == 0).sum(), (lab == 1).sum()) < 20:
        return None
    resp = torch.from_numpy(np.eye(2)[lab])
    child = model.dyn.stats_from_batch(resp, torch.from_numpy(y), torch.from_numpy(g))
    child = {k2: v/len(window) for k2, v in child.items()}
    out = {k2: v.clone() for k2, v in stats.items()}
    for key in DYN_KEYS:
        out[key][k] = child[key][0]
    return stats_append(out, {key: child[key][1] for key in DYN_KEYS})


def birth_proposal(model, stats, window, cfg):
    """New slot from the worst-explained steps of the window (lowest best expected log-lik)."""
    scores = {id(d): model.dyn.expected_loglik(d["y"], d["g"]).max(-1).values.numpy() for d in window}
    cut = np.quantile(np.concatenate([s.ravel() for s in scores.values()]), cfg["birth_frac"])
    y, g = _pooled(window, lambda d: scores[id(d)] <= cut)
    if len(y) < 40:
        return None
    new = model.dyn.stats_from_batch(torch.ones(len(y), 1, dtype=torch.float64),
                                     torch.from_numpy(y), torch.from_numpy(g))
    return stats_append(stats, {key: new[key][0]/len(window) for key in DYN_KEYS})


def move_sweep(model, store, window, classes, cfg, rng):
    try:
        cur_m, cur_s, cur_obj = refine_and_score(classes, store.total, window, cfg["move_refine"])
    except FloatingPointError:
        return None, None, []
    accepted = []

    def attempt(kind, stats0):
        nonlocal cur_m, cur_s, cur_obj
        if stats0 is None:
            return False
        try:
            m, s, obj = refine_and_score(classes, stats0, window, cfg["move_refine"])
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError):
            return False
        if obj > cur_obj+cfg["move_margin"]:
            cur_m, cur_s, cur_obj = m, s, obj
            accepted.append(kind)
            return True
        return False

    kc = int(cur_s["N"].shape[0]); mass = float(cur_s["N"].sum())
    for k in np.argsort(cur_s["N"].numpy()):
        if kc <= 1 or float(cur_s["N"][k]) >= cfg["delete_frac"]*mass:
            break
        if attempt("delete", stats_select(cur_s, [j for j in range(kc) if j != k])):
            break
    kc = int(cur_s["N"].shape[0])
    if kc > 1:
        M = cur_m.dyn.M.numpy()
        pairs = sorted((float(np.linalg.norm(M[i]-M[j])), i, j) for i in range(kc) for j in range(i+1, kc))
        for _, i, j in pairs[:cfg["merge_top"]]:
            if attempt("merge", stats_merge(cur_s, i, j)):
                break
    if int(cur_s["N"].shape[0]) < cfg["k_max"]:
        attempt("split", split_proposal(cur_m, cur_s, window, int(np.argmax(cur_s["N"].numpy())), rng))
    if int(cur_s["N"].shape[0]) < cfg["k_max"]:
        attempt("birth", birth_proposal(cur_m, cur_s, window, cfg))
    return (cur_m, cur_s, accepted) if accepted else (None, None, [])


def aligned_errors(gamma, truth, learned, A_true):
    """Per-batch Hungarian alignment of fitted slots to the true regimes (works for any K).
    Returns Hamming distance and the mean Frobenius error of matched regime maps."""
    pred = gamma.argmax(-1)
    kf = gamma.shape[-1]
    conf = np.array([[np.sum((truth == t) & (pred == k)) for k in range(kf)] for t in range(K)])
    rows, cols = linear_sum_assignment(-conf)
    ham = 1.0-conf[rows, cols].sum()/truth.size
    err = float(np.mean([np.linalg.norm(learned[c]-A_true[r], ord="fro") for r, c in zip(rows, cols)]))
    return float(ham), err


def oracle_nll(data, A, c, P=P_TRUE, init=None):
    init = np.ones(K)/K if init is None else init
    g, y = data["g_np"], data["y_np"]
    x, u = g[..., :D], g[..., D:D+AD]
    mu = np.einsum("kij,btj->btki", A, x)+c[None, None]+ACTION_SCALE*u[:, :, None, :]
    logev = -.5*((y[:, :, None]-mu)**2).sum(-1)/SIGMA**2-D*np.log(SIGMA)-.5*D*np.log(2*np.pi)
    with np.errstate(divide="ignore"):
        logP, loginit = np.log(P), np.log(init)
    logalpha = loginit+logev[:, 0]
    z = logsumexp(logalpha, axis=-1); total = z.copy(); logalpha -= z[:, None]
    for t in range(1, y.shape[1]):
        logalpha = logsumexp(logalpha[:, :, None]+logP[None], axis=-2)+logev[:, t]
        z = logsumexp(logalpha, axis=-1); total += z; logalpha -= z[:, None]
    return -float(total.mean())/y.shape[1]


def true_fixed_points(A, c):
    return np.einsum("kij,kj->ki", np.linalg.inv(np.eye(D)[None]-A), c)


def aligned_basins(model, A, c):
    """Learned fixed points and maps of the occupied slots, matched to the true regimes by
    Hungarian assignment on fixed-point distance (works for any K). Returns [K,D], [K,D,D], error."""
    M, N = model.dyn.M.numpy(), model.dyn.N.numpy()
    occ = np.where(N > .01*N.sum())[0]
    Ahat, bhat = M[occ][:, :, :D], M[occ][:, :, -1]
    mu_hat = np.full((len(occ), D), np.nan)
    for i in range(len(occ)):
        I_A = np.eye(D)-Ahat[i]
        if np.linalg.cond(I_A) < 1e6:
            mu_hat[i] = np.linalg.solve(I_A, bhat[i])
    mu = true_fixed_points(A, c)
    cost = np.nan_to_num(np.linalg.norm(mu[:, None]-mu_hat[None], axis=-1), nan=1e6)
    r, col = linear_sum_assignment(cost)
    fp = np.full((K, D), np.nan); maps = np.full((K, D, D), np.nan)
    fp[r], maps[r] = mu_hat[col], Ahat[col]
    return fp, maps, float(np.nanmean(np.linalg.norm(fp-mu, axis=-1)))


@torch.no_grad()
def run_seed(task):
    seed, cfg, source_root, outdir = task
    torch.set_num_threads(1)
    classes = load_source(source_root)
    model0, initial, mapping, init_ham = initialize(classes, np.random.default_rng(50000+seed), cfg)
    taus = arm_taus(cfg)
    result = {}
    clock = time.perf_counter()
    for condition in cfg["conditions"]:
        models = {a: copy.deepcopy(model0) for a in ARMS}
        stores = {a: Statistics(KIND[a], initial, taus[a], cfg["window"], classes[4]) for a in ARMS}
        rng = np.random.default_rng(100000+seed)
        move_rng = np.random.default_rng(200000+seed)
        window = deque(maxlen=cfg["move_buffer"])
        S, n = cfg["steps"], len(ARMS)
        nll = np.zeros((n, S)); oracle = np.zeros(S)
        ham = nll.copy(); kocc = nll.copy(); fperr = nll.copy()
        fp = np.full((n, S, K, D), np.nan); maps = np.full((n, S, K, D, D), np.nan)
        moves = np.zeros((4, S))
        retention = np.full((len(FACTORS), 5, S), np.nan)
        for b in range(S):
            A, c = parameters(b, cfg, condition)
            P, init = chain_at(b, cfg, condition)
            x, u, truth = simulate_batch(rng, A, c, cfg["batch"], cfg["length"], P, init)
            data = prepare(x, u)
            oracle[b] = oracle_nll(data, A, c, P, init)
            for j, arm in enumerate(ARMS):
                model, store = models[arm], stores[arm]
                nll[j, b] = model.predict_nll(data)
                r, before = None, store.total["N"].clone()
                for iteration in range(cfg["local_iters"]):
                    message, gamma = model.messages(data)
                    if iteration == 0:
                        ham[j, b] = aligned_errors(gamma, truth, model.dyn.M.numpy()[:, :, :D], A)[0]
                    tau = store.tau
                    if KIND[arm] in GATED:
                        r, info = store.retain.factors(model, store.total, message,
                                                       learn=iteration == cfg["local_iters"]-1)
                        tau = store.retain.gain["emission"]
                    proposed = store.propose(message, r)
                    model.refit(proposed, base=store.total if KIND[arm] in ("ema",) + GATED else None,
                                message=message, tau=tau, retain=(r or {}).get("emission"))
                store.commit(message, r)
                if KIND[arm] == "retain":
                    retention[..., b] = retention_summary(store.retain, r, info)
                fp[j, b], maps[j, b], fperr[j, b] = aligned_basins(model, A, c)
                mass = float(model.dyn.N.sum())
                kocc[j, b] = float((model.dyn.N.numpy() > .01*mass).sum())
                wanted = cfg["batch"]*cfg["length"]*(b+2 if arm == "svb" else 1)
                if KIND[arm] in GATED:
                    bound = torch.maximum(before, message["N"])*(1+1e-9)+1e-9
                    if not (mass > 0. and bool((model.dyn.N <= bound).all())):
                        raise AssertionError((arm, b, model.dyn.N.tolist(), bound.tolist()))
                elif not np.isclose(mass, wanted, rtol=1e-9, atol=1e-8):
                    raise AssertionError((arm, b, mass, wanted))
            window.append(data)
            if (b+1) >= cfg["move_warmup"] and (b+1) % cfg["move_every"] == 0 and len(window) == cfg["move_buffer"]:
                new_model, new_stats, accepted = move_sweep(models["ema_moves"], stores["ema_moves"],
                                                            list(window), classes, cfg, move_rng)
                if accepted:
                    models["ema_moves"] = new_model
                    stores["ema_moves"].total = new_stats
                    for kind in accepted:
                        moves[("delete", "merge", "split", "birth").index(kind), b] += 1
        for key, val in dict(nll=nll, oracle=oracle, ham=ham, kocc=kocc, fperr=fperr, fp=fp, maps=maps, moves=moves,
                             retention=retention).items():
            result[f"{condition}_{key}"] = val
    result["initial_hamming"] = init_ham
    result["seconds"] = time.perf_counter()-clock
    path = Path(outdir)/f"seed_{seed:03d}.npz"
    partial = path.with_suffix(".partial")
    with partial.open("wb") as f:
        np.savez_compressed(f, **result)
    partial.replace(path)
    return dict(seed=seed, seconds=float(result["seconds"]), initial_hamming=init_ham)

def expected_stats(A, c, batch, length):
    probs = np.ones(K)/K
    u = np.zeros((K, D)); V = probs[:, None, None]*np.eye(D)
    N = np.zeros(K); Srr = np.zeros((K, D+AD+1, D+AD+1))
    Szr = np.zeros((K, D, D+AD+1)); Szz = np.zeros((K, D))
    for _ in range(length):
        N += probs
        Srr[:, :D, :D] += V
        Srr[:, D:D+AD, D:D+AD] += probs[:, None, None]*np.eye(AD)
        Srr[:, :D, -1] += u; Srr[:, -1, :D] += u; Srr[:, -1, -1] += probs
        Au = np.einsum("kij,kj->ki", A, u)
        next_u = Au+probs[:, None]*c
        AV = A@V
        next_V = AV@np.swapaxes(A, -1, -2)+np.einsum("ki,kj->kij", Au, c)+np.einsum("ki,kj->kij", c, Au)
        next_V += probs[:, None, None]*(np.einsum("ki,kj->kij", c, c)+(ACTION_SCALE**2+SIGMA**2)*np.eye(D))
        Szr[:, :, :D] += AV+np.einsum("ki,kj->kij", c, u)
        Szr[:, :, D:D+AD] += ACTION_SCALE*probs[:, None, None]*np.eye(D)
        Szr[:, :, -1] += next_u
        Szz += np.diagonal(next_V, axis1=-2, axis2=-1)
        u = np.einsum("jk,ji->ki", P_TRUE, next_u)
        V = np.einsum("jk,jil->kil", P_TRUE, next_V)
        probs = probs@P_TRUE
    tensors = dict(N=N, Srr=Srr, Szr=Szr, Szz=Szz,
                   Shh=np.zeros((K, 0, 0)), Szh=np.zeros((K, D, 0)), Srh=np.zeros((K, D+AD+1, 0)))
    return {k:torch.from_numpy(v*batch) for k,v in tensors.items()}


def vector(stats, mass):
    return np.concatenate([stats[k].numpy().ravel() for k in DYN_KEYS])/mass


@torch.no_grad()
def oracle_study(classes, cfg, out):
    mass = cfg["batch"]*cfg["length"]
    model = SwitchingCore(classes)
    taus = arm_taus(cfg)
    messages = [expected_stats(*parameters(b, cfg), cfg["batch"], cfg["length"]) for b in range(cfg["steps"])]
    target = np.stack([vector(s, mass) for s in messages])
    stores = {a: Statistics(KIND[a], messages[0], taus[a], cfg["window"]) for a in ORACLE_ARMS}
    bias = np.zeros((len(ORACLE_ARMS), cfg["steps"]))
    bounds = {a: np.zeros(cfg["steps"]) for a in ("ema", "ema_slow")}
    model.dyn.set_stats(messages[0])
    consistency = 0.
    for b in range(1, cfg["steps"]):
        model.dyn.ema_update_stats(messages[b], cfg["tau"])
        for j, arm in enumerate(ORACLE_ARMS):
            stores[arm].commit(messages[b])
            divisor = mass*stores[arm].count if arm == "svb" else mass
            bias[j, b] = np.linalg.norm(vector(stores[arm].total, divisor)-target[b])
        consistency = max(consistency, np.linalg.norm(vector({k: getattr(model.dyn, k) for k in DYN_KEYS}, mass)
                                                      - vector(stores["ema"].total, mass)))
        delta_b = np.linalg.norm(target[b]-target[b-1])
        for a in bounds:
            bounds[a][b] = (1-taus[a])*(bounds[a][b-1]+delta_b)
    delta = float(np.linalg.norm(np.diff(target, axis=0), axis=1).max())
    steps = np.arange(cfg["steps"])
    constant = {a: (1-taus[a])/taus[a]*delta*(1-(1-taus[a])**steps) for a in bounds}
    for a in bounds:
        if np.max(bias[ORACLE_ARMS.index(a)]-bounds[a]) > 1e-10:
            raise AssertionError(f"Population bias violated drift-sensitive bound for {a}")
    np.savez_compressed(out/"oracle_tracking.npz", target=target, bias=bias, arms=np.array(ORACLE_ARMS),
                        bound=bounds["ema"], bound_slow=bounds["ema_slow"],
                        constant_bound=constant["ema"], constant_bound_slow=constant["ema_slow"])
    return dict(exact_delta=delta, raw_ema_vs_independent_recursion_error=consistency,
                max_population_bias={a: float(bias[ORACLE_ARMS.index(a)].max()) for a in ORACLE_ARMS},
                max_bound={a: float(bounds[a].max()) for a in bounds},
                note="Exact population-moment propagation with known assignments; moves arm excluded (fixed K)")

@torch.no_grad()
def check_implementation(classes, cfg):
    rng = np.random.default_rng(96001)
    bsz,length,k = 4,11,K
    ev=rng.normal(size=(bsz,length,k));trans=rng.normal(size=(bsz,length-1,k,k));init=rng.normal(size=k)
    gam,xi,z=fast_fb(init,trans,ev)
    gg,cc,zz,xx=classes[2](torch.from_numpy(init),torch.from_numpy(trans),torch.from_numpy(ev),return_pairwise=True)
    fb_error=max(float((gam-gg).abs().max()),float((xi-xx).abs().max()),float(np.abs(z-zz.numpy()).max()))
    if fb_error>1e-10:raise AssertionError(fb_error)
    model=SwitchingCore(classes)
    A,c=parameters(0,cfg)
    s=[]
    for _ in range(2):
        x,u,ztrue=simulate_batch(rng,A,c,4,13);d=prepare(x,u)
        s.append(model.dyn.stats_from_batch(torch.from_numpy(np.eye(K)[ztrue]),d["y"],d["g"]))
    model.dyn.set_stats(s[0]);model.dyn.m_step()
    Lold,Mold,aold,bold=[getattr(model.dyn,n).numpy().copy() for n in ("lam","M","a","b")]
    Lnew=Lold+s[1]["Srr"].numpy()
    rhs=np.einsum("klr,krs->kls",Mold,Lold)+s[1]["Szr"].numpy()
    Mnew=np.einsum("klr,krs->kls",rhs,np.linalg.inv(Lnew))
    anew=aold+.5*s[1]["N"].numpy()[:,None]
    bnew=bold+.5*(s[1]["Szz"].numpy()+np.einsum("klr,krs,kls->kl",Mold,Lold,Mold)-np.einsum("klr,krs,kls->kl",Mnew,Lnew,Mnew))
    model.dyn.set_stats({key:s[0][key]+s[1][key] for key in DYN_KEYS});model.dyn.m_step()
    svb_error=max(float(np.max(np.abs(getattr(model.dyn,key).numpy()-val))) for key,val in [("lam",Lnew),("M",Mnew),("a",anew),("b",bnew)])
    if svb_error>1e-9:raise AssertionError(svb_error)
    Ac,cc=parameters(cfg["change_start"]+cfg["rotation_steps"]//2,cfg)
    big=24000
    x,u,ztrue=simulate_batch(rng,Ac,cc,big,12);d=prepare(x,u)
    empirical=model.dyn.stats_from_batch(torch.from_numpy(np.eye(K)[ztrue]),d["y"],d["g"])
    exact=expected_stats(Ac,cc,big,12)
    relative=np.linalg.norm(vector(empirical,big*12)-vector(exact,big*12))/np.linalg.norm(vector(exact,big*12))
    if relative>.04:raise AssertionError(("population moment validation",relative))
    norms=np.linalg.svd(BASE_A,compute_uv=False)[:,0]
    if norms.max()>=.95:raise AssertionError(norms)
    return dict(forward_backward_max_error=fb_error,svb_posterior_as_prior_max_error=svb_error,
                population_moment_monte_carlo_relative_error=float(relative),
                generator_operator_norms=norms.tolist(), population_check_trajectories=big)


def bootstrap_ci(values,rng):
    draws=rng.choice(values,size=(10000,len(values)),replace=True).mean(-1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def summarize(out, cfg):
    files = [out/f"seed_{seed:03d}.npz" for seed in cfg["seed_ids"]]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    data = [np.load(p) for p in files]
    rng = np.random.default_rng(826141)
    cs = cfg["change_start"]
    result = {"n_seeds": len(data), "reference_arm": "ema", "window": f"updates {cs+1}-{cfg['steps']}",
              "primary_metric": "excess predictive NLL (model minus true system), nats per transition",
              "conditions": {}}
    for condition in cfg["conditions"]:
        get = lambda key: np.stack([d[f"{condition}_{key}"] for d in data])
        excess = (get("nll")-get("oracle")[:, None])[:, :, cs:].mean(-1)
        fperr, ham, kocc, mv = get("fperr")[:, :, cs:], get("ham")[:, :, cs:], get("kocc")[:, :, cs:], get("moves")
        row = {}
        for j, arm in enumerate(ARMS):
            diff = excess[:, 0]-excess[:, j]
            row[arm] = {"excess_nll": float(excess[:, j].mean()), "excess_nll_ci95": bootstrap_ci(excess[:, j], rng),
                        "ema_minus_arm": float(diff.mean()), "paired_ci95": bootstrap_ci(diff, rng),
                        "fixed_point_error": float(np.nanmean(fperr[:, j])),
                        "hamming_aligned_per_batch": float(ham[:, j].mean()),
                        "occupied_regimes": float(kocc[:, j].mean())}
        if condition == "recurring":
            back = cs+cfg["rotation_steps"]
            after = (get("nll")-get("oracle")[:, None])[:, :, back:back+50].mean(-1)
            for j, arm in enumerate(ARMS):
                row[arm]["excess_nll_first_50_after_return"] = float(after[:, j].mean())
        row["ema_moves"]["accepted_moves_per_seed"] = {k: float(mv[:, i].sum(-1).mean())
                                                      for i, k in enumerate(("delete", "merge", "split", "birth"))}
        ret = get("retention")
        row["ema_retain"]["retention"] = {
            f: {"mean_retention_before": float(ret[:, i, 0, :cs].mean()),
                "mean_retention_after": float(ret[:, i, 0, cs:].mean()),
                "max_drift_probability_after": float(ret[:, i, 1, cs:].max(-1).mean()),
                "drift_rate_final": float(ret[:, i, 2, -1].mean()),
                "mean_exposure": float(ret[:, i, 3].mean()),
                "failed_regime_updates": float(ret[:, i, 4].sum(-1).mean())} for i, f in enumerate(FACTORS)}
        result["conditions"][condition] = row
    result["initial_hamming"] = [float(d["initial_hamming"]) for d in data]
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result

def plot_results(out, cfg, panel_labels=False):
    """Figures 1-4 of the drift experiment from the saved per-seed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import t as student_t
    plt.rcParams.update(PAPER_STYLE)
    tau, tau_s = cfg["tau"], cfg["tau_slow"]
    labels = dict(ema_retain=r"AIME (proposed): gain $\tau u_k$ + change-point retention $\mathbb{E}[r_k]$",
                  ema_gated=r"AIME ablation: gain $\tau u_k$, no retention",
                  ema=rf"Fixed EMA, $\tau={tau:g}$ for every regime", ema_slow=rf"Fixed EMA, $\tau={tau_s:g}$",
                  ema_moves=rf"Fixed EMA, $\tau={tau:g}$ + birth/split/merge/delete",
                  svb="Sequential SVB (no forgetting)", window=f"Sliding window, W = {cfg['window']}",
                  cumavg="Cumulative mean")
    short = dict(ema_retain="AIME (proposed)", ema_gated="AIME, no retention", ema=rf"Fixed EMA, $\tau={tau:g}$",
                 ema_moves=rf"Fixed EMA, $\tau={tau:g}$ + moves", ema_slow=rf"Fixed EMA, $\tau={tau_s:g}$",
                 window=f"Window, W = {cfg['window']}", svb="Sequential SVB", cumavg="Cumulative mean")
    style = dict(ema_retain="-", ema_gated=(0, (4, 1.6)), ema="-", ema_slow="-", ema_moves="-.", svb="-",
                 window=(0, (1.2, 1.2)), cumavg="--")
    width = {a: 2.6 if a == "ema_retain" else 1.5 for a in ARMS}
    changed = [c for c in cfg["conditions"] if c not in ("stationary", "recurring")][0]

    def letter(ax, s):
        if panel_labels:
            ax.text(-.12, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")

    def shade(ax):
        ax.axvspan(cfg["change_start"]+1, cfg["change_start"]+cfg["rotation_steps"], color=".55", alpha=.11, lw=0)

    o = np.load(out/"oracle_tracking.npz")
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained", gridspec_kw={"width_ratios": [.9, 1.3]})
    ax = axs[0]
    th = np.linspace(0, 2*np.pi, 160)
    circle = np.stack([np.cos(th), np.sin(th)])
    for k, color in enumerate(REGIME_COLORS):
        for f, ls, lw, al in [(0., "--", 1.3, .9), (.5, "-", 1.0, .45), (1., "-", 2.2, 1.)]:
            ctr = centers_at(f, cfg)[k]
            ax.plot(*(ctr[:, None]+maps_at(f)[k]@circle), color=color, ls=ls, lw=lw, alpha=al)
        path = np.stack([centers_at(f, cfg)[k] for f in np.linspace(0, 1, 60)])
        ax.plot(*path.T, color=color, lw=1.0, alpha=.8)
        ax.annotate("", xy=path[-1], xytext=path[-6],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=11))
        ax.scatter(*path[0], s=26, facecolor="white", edgecolor=color, zorder=3, lw=1.3)
        ax.scatter(*path[-1], s=26, color=color, zorder=3)
        ax.text(*(path[0]+np.array([-.28, .28])), f"{k+1}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    ax.set(xlabel=r"State $x_1$", ylabel=r"State $x_2$")
    ax.set_aspect("equal")
    handles = [Line2D([], [], color=".35", ls="--", lw=1.3, label="Start of change"),
               Line2D([], [], color=".35", lw=1.0, alpha=.5, label="Midway"),
               Line2D([], [], color=".35", lw=2.2, label="End of change"),
               Line2D([], [], color=".35", lw=1.0, marker=">", markersize=5, label="Fixed-point path")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8.5)
    letter(ax, "(a)")
    ax = axs[1]
    x = np.arange(1, cfg["steps"]+1)
    arms = list(o["arms"])
    ax.plot(x, o["bias"][arms.index("ema")], color=COLORS["ema"], lw=2, label=labels["ema"])
    ax.plot(x, o["bias"][arms.index("ema_slow")], color=COLORS["ema_slow"], lw=2, label=labels["ema_slow"])
    ax.plot(x, o["bias"][arms.index("window")], color=COLORS["window"], lw=1.6, label=labels["window"])
    ax.plot(x, o["bias"][arms.index("svb")], color=COLORS["svb"], lw=1.6, label="SVB / cumulative mean")
    ax.plot(x, o["bound"], color=COLORS["ema"], ls=(0, (4, 2)), lw=1.1, label=rf"Drift-sensitive bound, $\tau={tau:g}$")
    ax.plot(x, o["bound_slow"], color=COLORS["ema_slow"], ls=(0, (4, 2)), lw=1.1, label=rf"Drift-sensitive bound, $\tau={tau_s:g}$")
    ax.plot(x, o["constant_bound"], color=".55", ls=":", lw=1.2, label=rf"Prop. 2 bound, $\tau={tau:g}$ ($\delta_{{\max}}$)")
    shade(ax)
    ax.set(xlabel="Minibatch update", ylabel="Bias of normalized sufficient statistics")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=8.5)
    letter(ax, "(b)")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out/f"figure_1_tracking.{ext}", dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

    data = [np.load(out/f"seed_{seed:03d}.npz") for seed in cfg["seed_ids"]]
    n = len(data)
    cs, ce = cfg["change_start"], cfg["change_start"]+cfg["rotation_steps"]-1
    order = list(ORDER)

    fp = np.stack([d[f"{changed}_fp"] for d in data])
    mp = np.stack([d[f"{changed}_maps"] for d in data])
    fig, axs = plt.subplots(2, 4, figsize=(14.6, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.99, top=.99, bottom=.14, wspace=.06, hspace=.08)
    circle = np.stack([np.cos(th), np.sin(th)])
    f_end = fraction_at(ce, cfg, changed)
    true_path = np.stack([centers_at(f, cfg) for f in np.linspace(0, 1, 60)])
    steps = np.arange(cs-1, cfg["steps"], 5)
    for ax, arm in zip(axs.ravel(), order):
        j = ARMS.index(arm)
        for k, color in enumerate(REGIME_COLORS):
            ax.plot(*true_path[:, k].T, color=".78", lw=5, solid_capstyle="round", zorder=1)
            ax.plot(*(centers_at(f_end, cfg)[k][:, None]+maps_at(f_end)[k]@circle), color=".55", lw=1.1, ls="--", zorder=2)
            path = np.nanmean(fp[:, j, steps, k], axis=0)
            ax.plot(*path.T, color=color, lw=1.4, zorder=3)
            mu_end = np.nanmean(fp[:, j, ce, k], axis=0)
            A_end = np.nanmean(mp[:, j, ce, k], axis=0)
            if np.all(np.isfinite(mu_end)) and np.all(np.isfinite(A_end)):
                ax.plot(*(mu_end[:, None]+A_end@circle), color=color, lw=2, zorder=4)
                ax.scatter(*mu_end, s=18, color=color, zorder=5)
        ax.text(.03, .97, short[arm], transform=ax.transAxes, va="top", fontsize=9.5, zorder=6,
                fontweight="bold" if arm == "ema_retain" else "normal",
                bbox=dict(boxstyle="square,pad=.25", facecolor="white", edgecolor="none", alpha=.9))
        ax.set_aspect("equal")
        ax.grid(True)
    for ax in axs.ravel()[len(order):]:
        ax.axis("off")
    for ax in axs[1]:
        ax.set_xlabel(r"State $x_1$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"State $x_2$")
    handles = [Line2D([], [], color=".78", lw=5, label="True fixed-point path"),
               Line2D([], [], color=".55", lw=1.1, ls="--", label="True basin, end of change"),
               Line2D([], [], color=".2", lw=1.4, label="Learned fixed-point path"),
               Line2D([], [], color=".2", lw=2, label="Learned basin, end of change")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .01), ncol=4, fontsize=9.5)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out/f"figure_2_basins.{ext}", dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

    bin_size = 10
    starts = list(range(0, cfg["steps"], bin_size))
    xb = np.array([(s+min(s+bin_size, cfg["steps"])+1)/2 for s in starts])

    def excess(condition):
        return np.stack([d[f"{condition}_nll"]-d[f"{condition}_oracle"][None] for d in data])

    floor = 1e-4
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.7), layout="constrained", gridspec_kw={"width_ratios": [1.45, 1]})
    fig.get_layout_engine().set(w_pad=.05, wspace=.03)
    ax = axs[0]
    ex = excess(changed)
    curve = np.stack([ex[..., s:min(s+bin_size, cfg["steps"])].mean(-1) for s in starts], axis=-1).mean(0)
    drawn = [a for a in ORDER if a not in ("ema_moves", "cumavg")]
    for arm in drawn[::-1]:
        c = curve[ARMS.index(arm)]
        ax.plot(xb, np.where(c > floor, c, .1*floor), color=COLORS[arm], lw=width[arm], ls=style[arm],
                label=labels[arm], zorder=3 if arm == "ema_retain" else 2, solid_capstyle="round")
    shade(ax)
    ax.set_yscale("log")
    ax.set_ylim(3*floor, 2*float(curve.max()))
    ax.set_xlim(0, cfg["steps"])
    ax.text((2*cfg["change_start"]+cfg["rotation_steps"])/2, .03, "fixed points migrate", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, style="italic", color=".3")
    k = ARMS.index("ema_retain")
    at = int(np.argmin(np.abs(xb-(cs+cfg["rotation_steps"]*.6))))
    ax.annotate("AIME (proposed)", xy=(xb[at], curve[k, at]), xytext=(xb[at], curve[k, at]/4.5), ha="center",
                fontsize=9, color=".15", arrowprops=dict(arrowstyle="-", color=".45", lw=.7))
    ax.set(xlabel="Minibatch update", ylabel="Excess predictive NLL (nats / transition)")
    ax.grid(True, which="major")
    handles = ax.get_legend_handles_labels()[0][::-1]
    letter(ax, "(a)")
    ax = axs[1]
    rng = np.random.default_rng(826141)
    ypos = np.arange(len(order))[::-1].astype(float)
    marks = [("stationary", "Stationary control", "o", False, .25), (changed, "Migrating basins", "o", True, 0.),
             ("recurring", "Recurring regime", "D", True, -.25)]
    marks = [m for m in marks if m[0] in cfg["conditions"]]
    top = floor
    y_new = ypos[order.index("ema_retain")]
    ax.axhspan(y_new-.47, y_new+.47, color=COLORS["ema_retain"], alpha=.09, lw=0, zorder=0)
    for condition, _, marker, filled, off in marks:
        per_seed = excess(condition)[:, :, cs:].mean(-1)
        for y, arm in zip(ypos, order):
            v = per_seed[:, ARMS.index(arm)]
            lo, hi = bootstrap_ci(v, rng) if n > 1 else (v.mean(), v.mean())
            ax.plot([max(lo, floor), max(hi, floor)], [y+off]*2, color=COLORS[arm], lw=1.3, zorder=2)
            ax.scatter(max(v.mean(), floor), y+off, s=30 if marker == "o" else 24, marker=marker, zorder=3,
                       facecolor=COLORS[arm] if filled else "white", edgecolor=COLORS[arm], lw=1.3)
            top = max(top, hi, v.mean())
    ax.set_yticks(ypos, [short[a] for a in order])
    for tick, arm in zip(ax.get_yticklabels(), order):
        tick.set_fontweight("bold" if arm == "ema_retain" else "normal")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ypos.min()-.6, ypos.max()+.6)
    ax.set_xscale("log")
    ax.set_xlim(.6*floor, 2*top)
    ax.set_xlabel(f"Mean excess NLL, updates {cs+1}\u2013{cfg['steps']}")
    ax.grid(True, axis="x", which="major")
    handles += [Line2D([], [], ls="", marker=m, markersize=6 if m == "o" else 5.3, color=".25",
                       markerfacecolor=".25" if f else "white", label=name) for _, name, m, f, _ in marks]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, handlelength=2.6, columnspacing=2.2)
    letter(ax, "(b)")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out/f"figure_3_excess_nll.{ext}", dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

    fcol = dict(emission="#882255", transition="#117733", gate="#DDAA33")
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")
    fig.get_layout_engine().set(w_pad=.05, wspace=.06)
    for condition, ls in (("stationary", ":"), (changed, "-")):
        ret = np.stack([d[f"{condition}_retention"] for d in data])
        for i, f in enumerate(FACTORS):
            r = np.stack([ret[:, i, 0, s:min(s+bin_size, cfg["steps"])].mean(-1) for s in starts], -1).mean(0)
            axs[0].plot(xb, r, color=fcol[f], ls=ls, lw=1.6)
            axs[1].plot(np.arange(1, cfg["steps"]+1), ret[:, i, 2].mean(0), color=fcol[f], ls=ls, lw=1.6)
    for ax in axs:
        shade(ax)
        ax.grid(True)
        ax.set_xlim(0, cfg["steps"])
    axs[0].set(xlabel="Minibatch update", ylabel=r"Posterior mean retention $\mathbb{E}[r_k]$")
    axs[1].set(xlabel="Minibatch update", ylabel=r"Posterior mean drift rate $\mathbb{E}[\pi]$", yscale="log")
    letter(axs[0], "(a)"); letter(axs[1], "(b)")
    handles = [Line2D([], [], color=fcol[f], lw=1.6, label=f.capitalize()) for f in FACTORS]
    handles += [Line2D([], [], color=".3", lw=1.6, label="Migrating basins"),
                Line2D([], [], color=".3", lw=1.6, ls=":", label="Stationary control")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, handlelength=2.6, columnspacing=2.2)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out/f"figure_4_retention.{ext}", dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    default_source = repo if (repo/"shs_rssm"/"regimes_shared.py").exists() else here/"source_snapshot"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=default_source,
                   help="directory containing shs_rssm/ (default: the AIME repository root)")
    p.add_argument("--out", type=Path, default=here/"results")
    p.add_argument("--stage", choices=["all", "check", "run", "plot"], default="all")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--change-start", type=int, default=300)
    p.add_argument("--rotation-steps", type=int, default=150)
    p.add_argument("--center-angle", type=float, default=40.,
                   help="degrees the fixed points revolve about the origin during the change")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--length", type=int, default=24)
    p.add_argument("--tau", type=float, default=.02)
    p.add_argument("--tau-slow", type=float, default=.005)
    p.add_argument("--window", type=int, default=99)
    p.add_argument("--local-iters", type=int, default=3)
    p.add_argument("--warm-batches", type=int, default=16)
    p.add_argument("--warm-iters", type=int, default=12)
    p.add_argument("--move-every", type=int, default=10, help="updates between move sweeps (moves arm)")
    p.add_argument("--move-buffer", type=int, default=8, help="recent minibatches used to score moves")
    p.add_argument("--move-warmup", type=int, default=20)
    p.add_argument("--move-margin", type=float, default=1.0, help="required window-bound gain, nats per minibatch")
    p.add_argument("--move-refine", type=int, default=2)
    p.add_argument("--merge-top", type=int, default=6)
    p.add_argument("--delete-frac", type=float, default=.02)
    p.add_argument("--birth-frac", type=float, default=.10)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--abrupt", action="store_true", help="also run an abrupt change (Proposition 3)")
    p.add_argument("--panel-labels", action="store_true", help="draw (a)/(b) panel letters")
    args = p.parse_args()
    if not 0 < args.tau <= 1 or not 0 < args.tau_slow <= 1 or not 0 < args.change_start < args.steps or args.rotation_steps <= 0:
        p.error("Require gains in (0,1] and a change point inside the run")
    keys = ["steps", "change_start", "rotation_steps", "center_angle", "batch", "length", "tau", "tau_slow",
            "window", "local_iters", "warm_batches", "warm_iters", "move_every", "move_buffer", "move_warmup",
            "move_margin", "move_refine", "merge_top", "delete_frac", "birth_frac", "k_max"]
    cfg = {k: getattr(args, k) for k in keys}
    cfg["seed_ids"] = list(range(args.seed_start, args.seed_start+args.seeds))
    cfg["conditions"] = ["stationary", "moving", "recurring"]+(["abrupt"] if args.abrupt else [])
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Output directory has a different configuration; use a new --out")
    config_path.write_text(json.dumps(cfg, indent=2)+"\n")
    torch.set_num_threads(1)
    classes = load_source(args.source_root)
    if args.stage in ("all", "check"):
        checks = check_implementation(classes, cfg)
        checks["oracle_study"] = oracle_study(classes, cfg, args.out)
        checks["source_sha256"] = classes[3]
        checks["source_root"] = str(args.source_root.resolve())
        (args.out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
        print(json.dumps(checks, indent=2), flush=True)
    if args.stage in ("all", "run"):
        tasks = [(seed, cfg, str(args.source_root.resolve()), str(args.out.resolve()))
                 for seed in cfg["seed_ids"] if not (args.out/f"seed_{seed:03d}.npz").exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in as_completed([pool.submit(run_seed, t) for t in tasks]):
                print(json.dumps(r.result()), flush=True)
    if args.stage in ("all", "plot"):
        summary = summarize(args.out, cfg)
        plot_results(args.out, cfg, panel_labels=args.panel_labels)
        print(json.dumps(summary, indent=2), flush=True)

if __name__=="__main__":main()
