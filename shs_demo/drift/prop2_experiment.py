from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from scipy.special import digamma, gammaln
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_drift as rd  

MAIN_TAU = 0.02
HPP_GAMMA = 0.1
FAMILY = dict(retain="#2a78d6", ema="#eb6834", gated="#1baf7a", pp="#eda100", window="#e87ba4", svb="#008300",
              hpp="#7b52c7")
NAME = dict(ema="EMA", pp="PP", hpp="HPP", svb="SVB", window="Window", gated="AIME, no retention",
            retain="AIME (proposed)")
LONG = dict(ema="EMA", pp="Power-prior SVB", hpp="Hierarchical power-prior SVB", svb="Streaming VB",
            window="Sliding window", gated="AIME without retention: responsibility-gated EMA",
            retain="AIME (proposed): responsibility-gated EMA with change-point retention")
MARK = dict(ema="o", pp="s", window="^")
LS = dict(ema="-", pp="-", window=(0, (4, 2)), svb="-", hpp=(0, (6, 1.5, 1.2, 1.5)), gated=(0, (4, 1.6)),
          retain="-")
WINDOWS = ("stationary", "drift", "post")


def window_len(t):
    return int(round((2-t)/t))


def arm_list(taus):
    arms = [("ema", t) for t in taus]
    arms += [("pp", t) for t in taus if t < 1] + [("window", t) for t in taus if t < 1]
    return arms+[("svb", None), ("hpp", None), ("gated", MAIN_TAU), ("retain", MAIN_TAU)]


def arm_name(kind, t):
    return kind if t is None else f"{kind}_{t:g}"


def parse_arm(name):
    kind, _, t = name.partition("_")
    return kind, (float(t) if t else None)


def dec(v):
    """0.02 -> '.02' for compact labels."""
    s = f"{v:g}"
    return s[1:] if s.startswith("0.") else s


def short_label(kind, t=MAIN_TAU):
    if kind == "ema":
        return f"EMA ({dec(t)})"
    if kind == "pp":
        return f"PP ({dec(round(1-t, 6))})"
    if kind == "window":
        return f"Window ({window_len(t)})"
    if kind == "gated":
        return f"AIME, no retention ({dec(t)})"
    if kind == "retain":
        return f"AIME (proposed, {dec(t)})"
    return NAME[kind]


def memory(kind, t):
    if kind in ("ema", "pp"):
        return f"tau={t:g}" if kind == "ema" else f"rho={1-t:g}"
    if kind == "window":
        return f"W={window_len(t)}"
    if kind == "gated":
        return f"tau={t:g}, gain tau*min(1, rho_k K)"
    if kind == "retain":
        return f"tau={t:g}, gain tau*min(1, rho_k K), E[r] per regime and factor"
    return "adaptive rho" if kind == "hpp" else "none"


def rotation(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def latent_angle(b, cfg, condition):
    """Rotation of the latent coordinates at update b (0-indexed): zero before the change,
    linear over drift_steps updates, then held at latent_angle degrees."""
    if condition == "stationary":
        return 0.0
    return float(np.clip((b-cfg["change_start"]+1)/cfg["drift_steps"], 0., 1.))*np.deg2rad(cfg["latent_angle"])


def draw_raw(rng, b, cfg, condition):
    """Physics never changes; the model sees the state through a slowly rotating coordinate
    system x = R_b s, as an encoder that is still training would present it."""
    A, c = rd.parameters(0, cfg, "stationary")
    s, u, z = rd.simulate_batch(rng, A, c, cfg["batch"], cfg["length"])
    x = s@rotation(latent_angle(b, cfg, condition)).T
    return rd.prepare(x, u), x, z


def draw(rng, b, cfg, condition):
    return draw_raw(rng, b, cfg, condition)[0]


class PowerPrior:
    """S_b = rho S_{b-1} + s_b, so the posterior is prior + S_b. Fixed rho = 1 - t (SVB-PP)
    or an adaptive rho_b passed at each update (SVB-HPP)."""

    def __init__(self, initial, rho=None):
        self.total, self.rho, self.tau = rd.clone_stats(initial), rho, None

    def propose(self, s, rho=None):
        r = self.rho if rho is None else rho
        return {k: r*self.total[k]+v for k, v in s.items()}

    def commit(self, s, rho=None):
        self.total = self.propose(s, rho)


def make_store(kind, t, initial, forgetting=None):
    if kind in rd.GATED:
        return rd.Statistics(kind, initial, t, None, forgetting)
    if kind == "ema":
        return rd.Statistics("ema", initial, t, None)
    if kind == "window":
        return rd.Statistics("window", initial, None, window_len(t))
    if kind == "svb":
        return rd.Statistics("svb", initial, None, None)
    return PowerPrior(initial, rho=1-t if kind == "pp" else None)


def snapshot(m):
    d = m.dyn
    return dict(M=d.M.numpy().copy(), lam=d.lam.numpy().copy(), a=d.a.numpy().copy(), b=d.b.numpy().copy(),
                mb=m.gate.m_beta.numpy().copy(), Sb=m.gate.Sigma_beta.numpy().copy(),
                dest=m.dest.numpy().copy(), start=m.start.numpy().copy())


def prior_snapshot(classes, initial):
    core = rd.SwitchingCore(classes)
    core.refit({k: torch.zeros_like(v) for k, v in initial.items()})
    return snapshot(core)


def _kl_dir(q, p):
    q0, p0 = q.sum(-1), p.sum(-1)
    return float((gammaln(q0)-gammaln(q).sum(-1)-gammaln(p0)+gammaln(p).sum(-1)
                  + ((q-p)*(digamma(q)-digamma(q0)[..., None])).sum(-1)).sum())


def kl_between(q, p):
    """KL(q || p) summed over all global factors: Normal-Gamma dynamics rows, Gaussian gate
    rows and Dirichlet transition/initial rows."""
    G = q["M"].shape[-1]
    tr = np.einsum("kij,kji->k", p["lam"], np.linalg.inv(q["lam"]))
    logdet = np.linalg.slogdet(q["lam"])[1]-np.linalg.slogdet(p["lam"])[1]
    dm = q["M"]-p["M"]
    maha = np.einsum("klg,kgh,klh->kl", dm, p["lam"], dm)
    gauss = .5*((tr-G+logdet)[:, None]+(q["a"]/q["b"])*maha)
    gam = ((q["a"]-p["a"])*digamma(q["a"])-gammaln(q["a"])+gammaln(p["a"])
           + p["a"]*(np.log(q["b"])-np.log(p["b"]))+q["a"]*(p["b"]-q["b"])/q["b"])
    Sp_inv = np.linalg.inv(p["Sb"]); P = q["mb"].shape[-1]; dmb = p["mb"]-q["mb"]
    gate = .5*(np.einsum("kij,kji->k", Sp_inv, q["Sb"])+np.einsum("ki,kij,kj->k", dmb, Sp_inv, dmb)-P
               + np.linalg.slogdet(p["Sb"])[1]-np.linalg.slogdet(q["Sb"])[1])
    return float((gauss+gam).sum()+gate.sum())+_kl_dir(q["dest"], p["dest"])+_kl_dir(q["start"], p["start"])


def kl_reference(q, p):
    from torch.distributions import Dirichlet, Gamma, MultivariateNormal, kl_divergence
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)
    total = 0.
    K, L, _ = q["M"].shape
    for k in range(K):
        for l in range(L):
            e = q["a"][k, l]/q["b"][k, l]
            total += float(kl_divergence(MultivariateNormal(T(q["M"][k, l]), precision_matrix=T(e*q["lam"][k])),
                                         MultivariateNormal(T(p["M"][k, l]), precision_matrix=T(e*p["lam"][k]))))
            total += float(kl_divergence(Gamma(T(q["a"][k, l]), T(q["b"][k, l])), Gamma(T(p["a"][k, l]), T(p["b"][k, l]))))
        total += float(kl_divergence(MultivariateNormal(T(q["mb"][k]), T(q["Sb"][k])),
                                     MultivariateNormal(T(p["mb"][k]), T(p["Sb"][k]))))
    total += float(kl_divergence(Dirichlet(T(q["dest"])), Dirichlet(T(p["dest"]))).sum())
    total += float(kl_divergence(Dirichlet(T(q["start"])), Dirichlet(T(p["start"]))))
    return total


def trunc_exp_mean(w):
    """E[rho] under q(rho) proportional to exp(w rho) on [0,1] (Masegosa et al. 2020, Eq. 28)."""
    w = float(np.clip(w, -700., 700.))
    return .5+w/12 if abs(w) < 1e-6 else float(1/(-np.expm1(-w))-1/w)


def hpp_fixed_point(refit, kl_prior, kl_prev, r=.5, tol=1e-6, max_iters=200):
    for _ in range(max_iters):
        q = refit(r)
        new = trunc_exp_mean(kl_prior(q)-kl_prev(q)+HPP_GAMMA)
        done, r = abs(new-r) < tol, new
        if done:
            break
    return r


def update_arm(model, store, kind, t, data, prior, iters):
    """Local passes and the global update for minibatch `data`. Returns rho_b for the hierarchical
    power prior (Masegosa et al. 2020, Alg. 1), per-factor retention rows for retain, else nan."""
    if kind == "hpp":
        prev, r = snapshot(model), .5
        for _ in range(iters):
            msg, _ = model.messages(data)

            def refit(rho, msg=msg):
                model.refit(store.propose(msg, rho))
                return snapshot(model)
            r = hpp_fixed_point(refit, lambda q: kl_between(q, prior), lambda q: kl_between(q, prev), r)
        store.commit(msg, r)
        model.refit(store.total)
        return r
    if kind in rd.GATED:
        for i in range(iters):
            msg, _ = model.messages(data)
            r, info = store.retain.factors(model, store.total, msg, learn=i == iters-1)
            model.refit(store.propose(msg, r), base=store.total, message=msg, tau=store.retain.gain["emission"],
                        retain=r.get("emission"))
        store.commit(msg, r)
        return rd.retention_summary(store.retain, r, info) if kind == "retain" else np.nan
    for _ in range(iters):
        msg, _ = model.messages(data)
        model.refit(store.propose(msg), base=store.total if kind == "ema" else None,
                    message=msg, tau=t if kind == "ema" else None)
    store.commit(msg)
    return np.nan


@torch.no_grad()
def ramp_study(classes, cfg):
    n, R = 64, cfg["ramp_reps"]
    x1 = np.repeat([-1., 1.], n//2)
    g1 = np.zeros((1, n, rd.D+rd.AD+1)); g1[0, :, 0] = x1; g1[0, :, -1] = 1.
    core = rd.SwitchingCore(classes)
    resp1 = torch.zeros(1, n, rd.K, dtype=torch.float64); resp1[..., 0] = 1.
    big = rd.SwitchingCore(classes, k=R).dyn
    gR = torch.from_numpy(np.repeat(g1, R, axis=0))
    respR = torch.zeros(R, n, R, dtype=torch.float64)
    respR[torch.arange(R), :, torch.arange(R)] = 1.
    rng = np.random.default_rng(20261)
    steps, delta = cfg["ramp_steps"], cfg["ramp_delta"]
    a_b = -.7+delta*np.minimum(np.arange(steps), cfg["ramp_stop"]-1)

    def exact(a):
        y = np.zeros((1, n, rd.D)); y[0, :, 0] = a*x1
        return core.dyn.stats_from_batch(resp1, torch.from_numpy(y), torch.from_numpy(g1))

    noisy = []
    for a in a_b:
        y = np.zeros((R, n, rd.D)); y[:, :, 0] = a*x1+rd.SIGMA*rng.standard_normal((R, n))
        noisy.append(big.stats_from_batch(respR, torch.from_numpy(y), gR))
    out, tq = dict(a=a_b), student_t.ppf(.975, R-1)
    for t in cfg["ramp_taus"]:
        dyn = copy.deepcopy(core.dyn); dyn.set_stats(exact(a_b[0]))
        mc = copy.deepcopy(big); mc.set_stats(noisy[0])
        err, lag, half = np.zeros(steps), np.zeros(steps), np.zeros(steps)
        for b in range(1, steps):
            dyn.ema_update_stats(exact(a_b[b]), t)
            err[b] = abs(a_b[b]-float(dyn.Szr[0, 0, 0])/n)
            mc.ema_update_stats(noisy[b], t)
            d = a_b[b]-mc.Szr[:, 0, 0].numpy()/n
            lag[b], half[b] = d.mean(), tq*d.std(ddof=1)/np.sqrt(R)
        out[f"error_{t:g}"], out[f"mc_lag_{t:g}"], out[f"mc_half_{t:g}"] = err, lag, half
        out[f"bound_{t:g}"] = (1-t)*delta/t*(1-(1-t)**np.arange(steps))
    return out


def ramp_checks(ramp, cfg):
    stop = cfg["ramp_stop"]
    res = {}
    for t in cfg["ramp_taus"]:
        e, B = ramp[f"error_{t:g}"], ramp[f"bound_{t:g}"]
        res[f"tau_{t:g}"] = dict(max_abs_error_minus_bound_during_ramp=float(np.abs(e[:stop]-B[:stop]).max()),
                                 max_error_minus_bound_overall=float((e-B).max()),
                                 error_at_stop=float(e[stop-1]), error_at_end=float(e[-1]),
                                 monte_carlo_max_abs_lag_minus_exact=float(np.abs(np.abs(ramp[f"mc_lag_{t:g}"])-e).max()),
                                 monte_carlo_max_half_width=float(ramp[f"mc_half_{t:g}"].max()))
    return res


@torch.no_grad()
def kl_checks(classes, cfg, seed=0):
    torch.set_num_threads(1)
    model, initial, _, _ = rd.initialize(classes, np.random.default_rng(50000+seed), cfg)
    prior = prior_snapshot(classes, initial)
    q = snapshot(model)
    ours, aime = kl_between(q, prior), rd.global_kl(model)
    moved = copy.deepcopy(model)
    store = rd.Statistics("ema", initial, .5, None)
    data = draw(np.random.default_rng(7), cfg["change_start"]+cfg["drift_steps"], cfg, "drift")
    update_arm(moved, store, "ema", .5, data, prior, cfg["local_iters"])
    q2 = snapshot(moved)
    ours2, ref2 = kl_between(q2, q), kl_reference(q2, q)
    return dict(kl_to_prior=dict(this_code=ours, aime_global_kl=aime, rel_diff=abs(ours-aime)/abs(aime)),
                kl_between_posteriors=dict(this_code=ours2, torch_distributions=ref2,
                                           rel_diff=abs(ours2-ref2)/abs(ref2)),
                kl_self=kl_between(q, q))


def hpp_beta_binomial(n, seed=0):
    rng = np.random.default_rng(seed)
    p_true = np.r_[np.full(30, .2), np.full(30, .5), np.full(40, .8)]
    alpha = np.ones(2)
    zero = {"ab": torch.zeros(2, dtype=torch.float64)}
    stores = dict(svb=rd.Statistics("svb", zero, None, None), pp_0_9=PowerPrior(zero, .9),
                  pp_0_99=PowerPrior(zero, .99), hpp=PowerPrior(zero))
    mean = {k: np.zeros(len(p_true)) for k in stores}
    rho, ess = np.zeros(len(p_true)), np.zeros(len(p_true))
    for i, p in enumerate(p_true):
        k = float((rng.random(n) < p).sum())
        s = {"ab": torch.tensor([k, n-k], dtype=torch.float64)}
        for name, st in stores.items():
            if name == "hpp":
                prev = alpha+st.total["ab"].numpy()
                r = hpp_fixed_point(lambda rho: alpha+st.propose(s, rho)["ab"].numpy(),
                                    lambda q: _kl_dir(q, alpha), lambda q: _kl_dir(q, prev))
                st.commit(s, r)
                rho[i] = r
            else:
                st.commit(s)
            lam = alpha+st.total["ab"].numpy()
            mean[name][i] = lam[0]/lam.sum()
            if name == "hpp":
                ess[i] = lam.sum()
    return dict(p_true=p_true, rho=rho, ess_hpp=ess, **{f"mean_{k}": v for k, v in mean.items()})


def hpp_checks(out):
    runs = {n: hpp_beta_binomial(n) for n in (100, 1000)}
    np.savez_compressed(out/"hpp_beta_binomial.npz", **{f"n{n}_{k}": v for n, r in runs.items() for k, v in r.items()})
    r = runs[100]
    change = [30, 60]
    steady = np.setdiff1d(np.arange(1, 100), change)
    err = {k[5:]: float(np.abs(v-r["p_true"]).mean()) for k, v in r.items() if k.startswith("mean_")}
    return dict(setup="Masegosa et al. (2020) Sec. 6.1: 100 steps, p=.2/.5/.8 switching at steps 31 and 61, Beta(1,1), gamma=.1",
                rho_at_changes_n100=[float(r["rho"][c]) for c in change],
                rho_median_elsewhere_n100=float(np.median(r["rho"][steady])),
                rho_median_elsewhere_n1000=float(np.median(runs[1000]["rho"][steady])),
                mean_abs_error_of_E_p_n100=err)


@torch.no_grad()
def run_seed(task):
    seed, cfg, source_root, outdir = task
    torch.set_num_threads(1)
    classes = rd.load_source(source_root)
    model0, initial, _, _ = rd.initialize(classes, np.random.default_rng(50000+seed), cfg)
    prior = prior_snapshot(classes, initial)
    arms = arm_list(cfg["taus"])
    names = [arm_name(k, t) for k, t in arms]
    result, clock = {}, time.perf_counter()
    for condition in ("stationary", "drift"):
        models = {nm: copy.deepcopy(model0) for nm in names}
        stores = {nm: make_store(k, t, initial, classes[4]) for (k, t), nm in zip(arms, names)}
        rng = np.random.default_rng(100000+seed)
        nll = np.zeros((len(arms), cfg["steps"])); rho = np.full(cfg["steps"], np.nan)
        ret = np.full((len(rd.FACTORS), 5, cfg["steps"]), np.nan)
        for b in range(cfg["steps"]):
            data = draw(rng, b, cfg, condition)
            for j, ((kind, t), nm) in enumerate(zip(arms, names)):
                nll[j, b] = models[nm].predict_nll(data)
                r = update_arm(models[nm], stores[nm], kind, t, data, prior, cfg["local_iters"])
                if kind == "hpp":
                    rho[b] = r
                elif kind == "retain":
                    ret[..., b] = r
        result[f"{condition}_nll"], result[f"{condition}_hpp_rho"] = nll, rho
        result[f"{condition}_retention"] = ret
    result["seconds"] = time.perf_counter()-clock
    path = Path(outdir)/f"seed_{seed:03d}.npz"
    partial = path.with_suffix(".partial")
    with partial.open("wb") as f:
        np.savez_compressed(f, arms=np.array(names), **result)
    partial.replace(path)
    return dict(seed=seed, seconds=round(result["seconds"], 1))


@torch.no_grad()
def run_speed(task):
    """Supplementary drift-speed test: the EMA sweep on the drift run only, with the same total
    rotation spread over `span` updates. The run stops when the rotation ends."""
    seed, span, cfg, source_root, outdir = task
    torch.set_num_threads(1)
    classes = rd.load_source(source_root)
    c2 = dict(cfg, drift_steps=span, steps=cfg["change_start"]+span)
    model0, initial, _, _ = rd.initialize(classes, np.random.default_rng(50000+seed), cfg)
    taus = cfg["taus"]
    models = [copy.deepcopy(model0) for _ in taus]
    stores = [make_store("ema", t, initial) for t in taus]
    rng = np.random.default_rng(100000+seed)
    nll, clock = np.zeros((len(taus), c2["steps"])), time.perf_counter()
    for b in range(c2["steps"]):
        data = draw(rng, b, c2, "drift")
        for j, t in enumerate(taus):
            nll[j, b] = models[j].predict_nll(data)
            update_arm(models[j], stores[j], "ema", t, data, None, cfg["local_iters"])
    path = Path(outdir)/f"speed_{span:03d}_seed_{seed:03d}.npz"
    partial = path.with_suffix(".partial")
    with partial.open("wb") as f:
        np.savez_compressed(f, taus=np.array(taus), nll=nll, oracle=oracle_curve(seed, c2))
    partial.replace(path)
    return dict(seed=seed, span=span, seconds=round(time.perf_counter()-clock, 1))


def oracle_curve(seed, cfg):
    """Prequential NLL of the true parameters on the same minibatches. Rotating both the data
    and the true model leaves the Gaussian likelihood unchanged, so one curve serves both runs."""
    rng = np.random.default_rng(100000+seed)
    A, c = rd.parameters(0, cfg, "stationary")
    return np.array([rd.oracle_nll(draw(rng, b, cfg, "stationary"), A, c) for b in range(cfg["steps"])])


def fixed_points(model, mapping):
    """Regime fixed points (I - A_k)^{-1} c_k of a fitted model, with actions at their mean,
    reordered to the true regimes with the permutation fixed at initialisation."""
    M = model.dyn.M.numpy()
    out = np.full((rd.K, rd.D), np.nan)
    for j in range(rd.K):
        out[mapping[j]] = np.linalg.solve(np.eye(rd.D)-M[j][:, :rd.D], M[j][:, -1])
    return out


@torch.no_grad()
def trace_fixed_points(classes, cfg, seed, updates):
    """Re-run the drift condition of one seed for EMA (tau=.02) and streaming VB, which gives
    exactly the models of run_seed, and record their fixed points before minibatch b."""
    torch.set_num_threads(1)
    model0, initial, mapping, _ = rd.initialize(classes, np.random.default_rng(50000+seed), cfg)
    prior = prior_snapshot(classes, initial)
    arms = [("ema", MAIN_TAU), ("svb", None)]
    models = [copy.deepcopy(model0) for _ in arms]
    stores = [make_store(k, t, initial) for k, t in arms]
    rng = np.random.default_rng(100000+seed)
    xs, zs, truth = [], [], []
    fp = np.full((len(arms), len(updates), rd.K, rd.D), np.nan)
    for b in range(max(updates)):
        data, x, z = draw_raw(rng, b, cfg, "drift")
        if b+1 in updates:
            i = updates.index(b+1)
            xs.append(x); zs.append(z)
            truth.append(rd.CENTERS@rotation(latent_angle(b, cfg, "drift")).T)
            for j, m in enumerate(models):
                fp[j, i] = fixed_points(m, mapping)
        for (kind, t), m, st in zip(arms, models, stores):
            update_arm(m, st, kind, t, data, prior, cfg["local_iters"])
    R = np.stack([rotation(a) for a in np.linspace(0, np.deg2rad(cfg["latent_angle"]), 91)])
    path = np.einsum("aed,kd->kae", R, rd.CENTERS)
    return dict(updates=np.array(updates), x=np.stack(xs), z=np.stack(zs), truth=np.stack(truth),
                fp=fp, arms=np.array([arm_name(k, t) for k, t in arms]), path=path)


def mean_ci(v):
    """Mean and Student-t 95% interval over seeds."""
    v = np.asarray(v, float); m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(student_t.ppf(.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return m, m-h, m+h


def windows(cfg):
    cs, h = cfg["change_start"], cfg["drift_steps"]
    return dict(stationary=("stationary", slice(cs, cs+h)), drift=("drift", slice(cs, cs+h)),
                post=("drift", slice(cs+h, cfg["steps"])))


def load_seeds(out, cfg):
    return [np.load(out/f"seed_{s:03d}.npz") for s in cfg["seed_ids"]]


def seed_means(data, cfg, orc):
    """Mean prequential NLL and mean excess NLL (minus the true parameters on the same
    minibatches) of every seed and arm in each window, each of shape (seeds, arms)."""
    raw, exc = {}, {}
    for w, (cond, sl) in windows(cfg).items():
        raw[w] = np.stack([d[f"{cond}_nll"][:, sl].mean(-1) for d in data])
        exc[w] = raw[w]-orc[:, sl].mean(-1)[:, None]
    return raw, exc


def oracle_all(out, cfg):
    path = out/"oracle_nll.npy"
    if path.exists():
        orc = np.load(path)
        if orc.shape == (len(cfg["seed_ids"]), cfg["steps"]):
            return orc
    orc = np.stack([oracle_curve(s, cfg) for s in cfg["seed_ids"]])
    np.save(path, orc)
    return orc


def speed_excess(out, cfg, data, orc):
    """Drift-window excess NLL of the EMA sweep for each drift span, shape (seeds, taus). The
    main span comes from the main run; the others from run_speed. Missing spans are skipped."""
    cs, names = cfg["change_start"], list(data[0]["arms"])
    idx = [names.index(arm_name("ema", t)) for t in cfg["taus"]]
    sl = slice(cs, cs+cfg["drift_steps"])
    res = {cfg["drift_steps"]: np.stack([d["drift_nll"][idx, sl].mean(-1) for d in data])-orc[:, sl].mean(-1)[:, None]}
    for span in cfg["speed_steps"]:
        files = [out/f"speed_{span:03d}_seed_{s:03d}.npz" for s in cfg["seed_ids"]]
        if all(f.exists() for f in files):
            sp = [np.load(f) for f in files]
            w = slice(cs, cs+span)
            res[span] = np.stack([r["nll"][:, w].mean(-1)-r["oracle"][w].mean() for r in sp])
    return dict(sorted(res.items()))


def summarize(out, cfg):
    data = load_seeds(out, cfg)
    names = list(data[0]["arms"])
    orc = oracle_all(out, cfg)
    raw, exc = seed_means(data, cfg, orc)
    ref = names.index(arm_name("ema", MAIN_TAU))
    wins = {w: f"updates {sl.start+1}-{sl.stop} of the {cond} run" for w, (cond, sl) in windows(cfg).items()}
    arms = {}
    for j, nm in enumerate(names):
        kind, t = parse_arm(nm)
        row = dict(method=LONG[kind], setting=memory(kind, t))
        for w in WINDOWS:
            row[w] = dict(nll=mean_ci(raw[w][:, j]), excess=mean_ci(exc[w][:, j]),
                          minus_ema=mean_ci(raw[w][:, j]-raw[w][:, ref]))
        arms[nm] = row
    rho = {w: float(np.nanmean(np.stack([d[f"{cond}_hpp_rho"][sl] for d in data]))) for w, (cond, sl) in windows(cfg).items()}
    retention = {w: {f: dict(mean_retention=float(np.nanmean(np.stack([d[f"{cond}_retention"][i, 0, sl] for d in data]))),
                             mean_drift_rate=float(np.nanmean(np.stack([d[f"{cond}_retention"][i, 2, sl] for d in data]))))
                     for i, f in enumerate(rd.FACTORS)} for w, (cond, sl) in windows(cfg).items()}
    speeds = speed_excess(out, cfg, data, orc)
    stat = exc["stationary"][:, [names.index(arm_name("ema", t)) for t in cfg["taus"]]]
    sweep = {f"tau={t:g}": dict(stationary=mean_ci(stat[:, i]),
                                **{f"drift_over_{span}": mean_ci(v[:, i]) for span, v in speeds.items()})
             for i, t in enumerate(cfg["taus"])}
    s = dict(n_seeds=len(data),
             metric="prequential NLL (each minibatch scored before it is used) and excess NLL (minus the true "
                    "parameters on the same minibatches), nats per transition; values are [mean, ci_lo, ci_hi]",
             intervals="Student-t 95% over seeds; minus_ema is paired within seed",
             windows=wins, reference=arm_name("ema", MAIN_TAU), arms=arms, hpp_mean_rho=rho,
             retention=retention,
             true_parameters_nll={w: mean_ci(orc[:, sl].mean(-1)) for w, (_, sl) in windows(cfg).items()},
             ema_tau_sweep_excess=sweep)
    (out/"summary.json").write_text(json.dumps(s, indent=2)+"\n")
    with (out/"summary_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        head = ["arm", "method", "setting"]
        for win in WINDOWS:
            for q in ("nll", "excess", "minus_ema"):
                head += [f"{win}_{q}", f"{win}_{q}_ci_lo", f"{win}_{q}_ci_hi"]
        w.writerow(head)
        for nm, row in arms.items():
            w.writerow([nm, row["method"], row["setting"]]+[f"{v:.4f}" for win in WINDOWS
                                                            for q in ("nll", "excess", "minus_ema") for v in row[win][q]])
    latex_tables(out, cfg, arms, s, speeds, stat)
    return s


def _pm(c):
    return rf"${c[0]:.3f} \pm {(c[2]-c[1])/2:.3f}$"


def latex_tables(out, cfg, arms, s, speeds, stat):
    rows = [(arm_name("ema", MAIN_TAU), rf"EMA, $\tau={MAIN_TAU:g}$"),
            (arm_name("pp", MAIN_TAU), rf"Power-prior SVB, $\rho={1-MAIN_TAU:g}$"),
            (arm_name("window", MAIN_TAU), rf"Sliding window, $W={window_len(MAIN_TAU)}$"),
            ("hpp", r"Hierarchical power-prior SVB"), ("svb", r"Streaming VB"),
            (arm_name("gated", MAIN_TAU), rf"AIME without retention: gated EMA, $\tau={MAIN_TAU:g}$"),
            (arm_name("retain", MAIN_TAU), rf"AIME (proposed): gated EMA, $\tau={MAIN_TAU:g}$, change-point retention")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Update & Stationary & Drift & After drift & Drift minus EMA \\", r"\midrule"]
    for nm, lab in rows:
        r = arms[nm]
        diff = "--" if nm == s["reference"] else _pm(r["drift"]["minus_ema"])
        lines.append(f"{lab} & {_pm(r['stationary']['excess'])} & {_pm(r['drift']['excess'])} & "
                     f"{_pm(r['post']['excess'])} & {diff} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out/"summary_table.tex").write_text("\n".join(lines)+"\n")
    spans = list(speeds)
    cols = "l"+"c"*(1+len(spans))
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf" & & \multicolumn{{{len(spans)}}}{{c}}{{Drift, rotation spread over}} \\",
             rf"\cmidrule(lr){{3-{2+len(spans)}}}",
             r"$\tau$ & Stationary & "+" & ".join(f"{sp} updates" for sp in spans)+r" \\", r"\midrule"]
    for i, t in enumerate(cfg["taus"]):
        lines.append(f"{t:g} & {_pm(mean_ci(stat[:, i]))} & "+" & ".join(_pm(mean_ci(speeds[sp][:, i])) for sp in spans)+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out/"tau_table.tex").write_text("\n".join(lines)+"\n")


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.spines.top": True, "axes.spines.right": True,
                         "axes.linewidth": .8, "axes.edgecolor": ".15", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman",
                                                                "Times", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    return plt


def letter(ax, s, on):
    if on:
        ax.text(-.16, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def save(fig, out, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/f"{stem}.{ext}", dpi=220, bbox_inches="tight", pad_inches=.03)


def plot(out, cfg, panel_labels=True):
    plt = style()
    from matplotlib.lines import Line2D
    ramp = np.load(out/"ramp.npz")
    data = load_seeds(out, cfg)
    orc = oracle_all(out, cfg)
    names = list(data[0]["arms"]); n = len(data)
    cs, h, steps = cfg["change_start"], cfg["drift_steps"], cfg["steps"]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.5), gridspec_kw={"width_ratios": [1, 1.12, 1]})
    fig.subplots_adjust(wspace=.3)

    ax = axs[0]
    x = np.arange(1, cfg["ramp_steps"]+1)
    for t, col in zip(cfg["ramp_taus"], ["#f5b594", "#eb6834", "#a8401a"]):
        lag, half = np.abs(ramp[f"mc_lag_{t:g}"]), ramp[f"mc_half_{t:g}"]
        ax.fill_between(x, np.maximum(lag-half, 0), lag+half, color=col, alpha=.35, lw=0)
        ax.plot(x, lag, color=col, lw=2.2, label=rf"$\tau={t:g}$")
        ax.plot(x, ramp[f"bound_{t:g}"], color="k", lw=.85, ls=(0, (3, 2)))
    ax.axvspan(1, cfg["ramp_stop"], color=".92", lw=0, zorder=0)
    ax.set(xlabel="Minibatch update", ylabel=r"Tracking error $\|\mathbb{E}[\bar s_b]-m_b\|$",
           xlim=(0, cfg["ramp_steps"]), ylim=(0, None))
    ax.grid(axis="y", alpha=.25, lw=.6)
    hand = ax.get_legend_handles_labels()[0]+[Line2D([], [], color="k", lw=.85, ls=(0, (3, 2)), label="Prop. 2 bound")]
    ax.legend(handles=hand, loc="upper left", frameon=False, handlelength=1.8)
    letter(ax, "(a)", panel_labels)

    ax = axs[1]
    shown = [("ema", MAIN_TAU), ("svb", None), ("pp", MAIN_TAU), ("window", MAIN_TAU), ("gated", MAIN_TAU),
             ("retain", MAIN_TAU)]
    exc = np.stack([d["drift_nll"] for d in data])-orc[:, None, :]
    bins = list(range(0, steps, 10))
    xb = np.array([(s+min(s+10, steps)+1)/2 for s in bins])
    curve = np.stack([exc[..., s:min(s+10, steps)].mean(-1) for s in bins], axis=-1)
    mean = curve.mean(0)
    half = student_t.ppf(.975, n-1)*curve.std(0, ddof=1)/np.sqrt(n) if n > 1 else 0*mean
    for kind, t in shown:
        j = names.index(arm_name(kind, t))
        ax.plot(xb, mean[j], color=FAMILY[kind], lw=2.5 if kind == "retain" else 1.6, ls=LS[kind],
                label=short_label(kind, t or MAIN_TAU), zorder=3 if kind == "retain" else 2)
        ax.fill_between(xb, mean[j]-half[j], mean[j]+half[j], color=FAMILY[kind], alpha=.12, lw=0)
    ax.axvspan(cs+1, cs+h, color=".92", lw=0, zorder=0)
    ax.axhline(0, color=".5", lw=.7, zorder=1)
    ax.set(xlabel="Minibatch update", ylabel="Excess NLL (nats / transition)", xlim=(0, steps))
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    letter(ax, "(b)", panel_labels)

    ax = axs[2]
    _, per = seed_means(data, cfg, orc)
    for kind in ("ema", "pp", "window"):
        taus = [t for t in cfg["taus"] if kind == "ema" or t < 1]
        st = np.array([mean_ci(per["drift"][:, names.index(arm_name(kind, t))]) for t in taus])
        ax.errorbar(taus, st[:, 0], yerr=[st[:, 0]-st[:, 1], st[:, 2]-st[:, 0]], color=FAMILY[kind], lw=1.4,
                    ls=LS[kind], marker=MARK[kind], ms=4.5, elinewidth=.8, capsize=0, label=NAME[kind])
    for kind, nm in (("svb", "svb"), ("hpp", "hpp"), ("gated", arm_name("gated", MAIN_TAU)),
                     ("retain", arm_name("retain", MAIN_TAU))):
        ax.axhline(per["drift"][:, names.index(nm)].mean(), color=FAMILY[kind], lw=2.4 if kind == "retain" else 1.3,
                   ls=LS[kind], zorder=1, label=NAME[kind])
    ax.axvline(MAIN_TAU, color=".45", lw=.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set(xlabel=r"Forgetting rate $\tau$", ylabel="Mean excess NLL during drift", ylim=(0, None))
    ax.grid(axis="y", alpha=.25, lw=.6)
    hand = [Line2D([], [], color=FAMILY[k], lw=2.4 if k == "retain" else 1.4, marker=MARK.get(k), ms=4.5, ls=LS[k],
                   label=NAME[k]) for k in ("retain", "gated", "ema", "pp", "window", "svb", "hpp")]
    ax.legend(handles=hand, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, handlelength=2.2)
    letter(ax, "(c)", panel_labels)
    save(fig, out, "figure_prop2")
    plt.close(fig)


REGIME = ("#88CCEE", "#DDCC77", "#AA4499")


def plot_data(out, cfg, trace, panel_labels=True):
    plt = style()
    from matplotlib.lines import Line2D
    ups = list(trace["updates"])
    fig, axs = plt.subplots(1, len(ups), figsize=(3.3*len(ups), 3.5), sharex=True, sharey=True)
    lim = np.abs(trace["x"]).max()*1.05
    for i, ax in enumerate(axs):
        x, z = trace["x"][i][:, 1:], trace["z"][i]
        for k in range(rd.K):
            sel = z == k
            ax.scatter(x[sel, 0], x[sel, 1], s=5, color=REGIME[k], alpha=.55, lw=0, rasterized=True)
        for k in range(rd.K):
            ax.plot(trace["path"][k, :, 0], trace["path"][k, :, 1], color=".6", lw=.7, ls=":", zorder=1)
        ax.plot(*trace["truth"][i].T, ls="", marker="*", ms=11, color="k", zorder=4)
        ax.plot(*trace["fp"][0, i].T, ls="", marker="o", ms=8, mfc="none", mec=FAMILY["ema"], mew=1.6, zorder=5)
        ax.plot(*trace["fp"][1, i].T, ls="", marker="X", ms=7.5, color=FAMILY["svb"], mec="white", mew=.5, zorder=5)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="$x_1$")
        ax.grid(alpha=.2, lw=.6)
        letter(ax, f"({'abcdef'[i]})", panel_labels)
    axs[0].set_ylabel("$x_2$")
    hand = [Line2D([], [], ls="", marker="o", ms=5, color=REGIME[k], label=f"Regime {k+1}") for k in range(rd.K)]
    hand += [Line2D([], [], ls="", marker="*", ms=10, color="k", label="True fixed point"),
             Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=FAMILY["ema"], mew=1.6, label=short_label("ema")),
             Line2D([], [], ls="", marker="X", ms=7, color=FAMILY["svb"], mec="white", mew=.5, label="SVB")]
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), frameon=False, bbox_to_anchor=(.5, -.06))
    save(fig, out, "figure_data")
    plt.close(fig)


def plot_hpp_check(out):
    plt = style()
    r = np.load(out/"hpp_beta_binomial.npz")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
    t = np.arange(1, 101)
    ax = axs[0]
    ax.plot(t, r["n100_p_true"], ls="", marker="o", ms=2.5, color="#C00000", label="True p")
    for k, lab, col, ls in (("svb", "SVB", FAMILY["svb"], "-"), ("pp_0_9", "PP (.9)", FAMILY["pp"], "-"),
                            ("pp_0_99", "PP (.99)", FAMILY["pp"], (0, (3, 2))), ("hpp", "HPP", FAMILY["hpp"], "-")):
        ax.plot(t, r[f"n100_mean_{k}"], color=col, ls=ls, lw=1.5, label=lab)
    ax.set(xlabel="Time step", ylabel="Posterior mean of p")
    ax.legend(frameon=False, loc="upper left")
    ax = axs[1]
    ax.plot(t, r["n100_rho"], color=FAMILY["hpp"], lw=1.5, label="100 draws per step")
    ax.plot(t, r["n1000_rho"], color=FAMILY["hpp"], lw=1.5, ls=(0, (3, 2)), label="1000 draws per step")
    ax.set(xlabel="Time step", ylabel=r"$\mathbb{E}[\rho_t]$", ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right")
    for ax in axs:
        ax.grid(axis="y", alpha=.25, lw=.6)
    save(fig, out, "check_hpp_beta_binomial")
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    default_source = repo if (repo/"shs_rssm"/"regimes_shared.py").exists() else here/"source_snapshot"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, default=default_source)
    p.add_argument("--out", type=Path, default=here/"results_prop2")
    p.add_argument("--stage", choices=["all", "run", "plot"], default="all")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--change-start", type=int, default=300)
    p.add_argument("--drift-steps", type=int, default=150)
    p.add_argument("--latent-angle", type=float, default=60.,
                   help="total rotation of the latent coordinates, degrees. The three regime centres are nearly "
                        "three-fold symmetric, so 60 degrees moves them farthest from any relabelling of themselves")
    p.add_argument("--taus", type=float, nargs="+", default=[.005, .01, .02, .05, .1, .25, 1.])
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--length", type=int, default=24)
    p.add_argument("--local-iters", type=int, default=3)
    p.add_argument("--warm-batches", type=int, default=16)
    p.add_argument("--warm-iters", type=int, default=12)
    p.add_argument("--ramp-steps", type=int, default=700)
    p.add_argument("--ramp-stop", type=int, default=500)
    p.add_argument("--ramp-delta", type=float, default=.001)
    p.add_argument("--ramp-reps", type=int, default=100, help="noisy Monte Carlo runs for panel (a)")
    p.add_argument("--speed-steps", type=int, nargs="*", default=[50, 300],
                   help="extra drift spans (updates) for the drift-speed table; empty to skip")
    p.add_argument("--data-seed", type=int, default=0, help="seed shown in the data figure")
    p.add_argument("--no-panel-labels", action="store_true")
    a = p.parse_args()
    if MAIN_TAU not in a.taus:
        p.error(f"--taus must include {MAIN_TAU}")
    cfg = dict(steps=a.steps, change_start=a.change_start, drift_steps=a.drift_steps, latent_angle=a.latent_angle,
               taus=sorted(a.taus), batch=a.batch, length=a.length, local_iters=a.local_iters,
               warm_batches=a.warm_batches, warm_iters=a.warm_iters, center_angle=0.,
               ramp_steps=a.ramp_steps, ramp_stop=a.ramp_stop, ramp_delta=a.ramp_delta,
               ramp_taus=[.005, .02, .1], ramp_reps=a.ramp_reps,
               speed_steps=sorted(set(a.speed_steps)-{a.drift_steps}), seed_ids=list(range(a.seeds)))
    if a.change_start+a.drift_steps >= a.steps:
        p.error("the run must continue after the drift ends")
    a.out.mkdir(parents=True, exist_ok=True)
    cpath = a.out/"config.json"
    if cpath.exists() and json.loads(cpath.read_text()) != cfg:
        raise ValueError("Output directory has a different configuration; use a new --out")
    cpath.write_text(json.dumps(cfg, indent=2)+"\n")
    torch.set_num_threads(1)
    classes = rd.load_source(a.source_root)
    if a.stage in ("all", "run"):
        ramp = ramp_study(classes, cfg)
        np.savez_compressed(a.out/"ramp.npz", **ramp)
        checks = dict(source_sha256=classes[3], source_root=str(a.source_root.resolve()),
                      proposition2_ramp=ramp_checks(ramp, cfg), kl=kl_checks(classes, cfg),
                      hpp_beta_binomial=hpp_checks(a.out))
        (a.out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
        print(json.dumps({k: checks[k] for k in ("proposition2_ramp", "kl", "hpp_beta_binomial")}, indent=2), flush=True)
        root, dest = str(a.source_root.resolve()), str(a.out.resolve())
        jobs = [(run_seed, (s, cfg, root, dest)) for s in cfg["seed_ids"] if not (a.out/f"seed_{s:03d}.npz").exists()]
        jobs += [(run_speed, (s, span, cfg, root, dest)) for span in cfg["speed_steps"] for s in cfg["seed_ids"]
                 if not (a.out/f"speed_{span:03d}_seed_{s:03d}.npz").exists()]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for r in as_completed([pool.submit(f, t) for f, t in jobs]):
                print(json.dumps(r.result()), flush=True)
    if a.stage in ("all", "plot"):
        s = summarize(a.out, cfg)
        print(json.dumps({k: s[k] for k in ("n_seeds", "windows", "hpp_mean_rho", "retention", "true_parameters_nll")},
                         indent=2), flush=True)
        print((a.out/"summary_table.tex").read_text(), flush=True)
        print((a.out/"tau_table.tex").read_text(), flush=True)
        plot(a.out, cfg, not a.no_panel_labels)
        tpath = a.out/"data_trace.npz"
        if not tpath.exists():
            ups = [cfg["change_start"], cfg["change_start"]+cfg["drift_steps"], cfg["steps"]]
            np.savez_compressed(tpath, **trace_fixed_points(classes, cfg, a.data_seed, ups))
        plot_data(a.out, cfg, dict(np.load(tpath)), not a.no_panel_labels)
        if (a.out/"hpp_beta_binomial.npz").exists():
            plot_hpp_check(a.out)


if __name__ == "__main__":
    main()
