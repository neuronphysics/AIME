from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.special import expit, gammaln, logsumexp
from scipy.stats import t as student_t

from controlled_segmentation_moving_mnist import (
    ControlledSegmentationMovingMNIST, FixedPCAEncoder, StreamSchedule, prepare_latent_for_model,
)
from retention_arm import FACTORS, RetentionRule, summary as retention_summary

MAIN_TAU = 0.02
DYN_KEYS = ("N", "Srr", "Szr", "Szz", "Shh", "Szh", "Srh")
COLOR = {"retain": "#2a78d6", "ema": "#eb6834", "gated": "#1baf7a", "pp": "#eda100", "window": "#e87ba4",
         "svb": "#008300"}
LINESTYLE = {"ema": "-", "svb": "--", "pp": "-.", "window": ":", "gated": (0, (4, 1.6)), "retain": "-"}


def window_len(tau):
    return int(round((2.0-float(tau))/float(tau)))


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_aime(root):
    root = Path(root)/"shs_rssm"
    ps = [root/"regimes_shared.py", root/"recurrent_stick.py", root/"forward_backward.py", root/"forgetting.py"]
    for p in ps:
        if not p.exists():
            raise FileNotFoundError(p)
    d = import_file(ps[0], "visual_p2_dyn")
    g = import_file(ps[1], "visual_p2_gate")
    f = import_file(ps[2], "visual_p2_fb")
    fg = import_file(ps[3], "visual_p2_forgetting")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}
    return d.SharedCarryRegimes, g.RecurrentStickiness, f.forward_backward, sha, fg


def clone_stats(s):
    return {k:v.clone() for k,v in s.items()}


class StatisticsStore:
    def __init__(self, kind, initial, tau=None, rho=None, window=None, forgetting=None):
        self.kind, self.tau, self.rho, self.window = kind, tau, rho, window
        self.total = clone_stats(initial)
        self.history = deque([clone_stats(initial)]) if kind == "window" else None
        self.window_sum = clone_stats(initial) if kind == "window" else None
        self.retain = RetentionRule(forgetting, tau, retain=kind == "retain") if kind in ("gated", "retain") else None

    def propose(self, msg, r=None):
        if self.kind in ("gated", "retain"):
            return self.retain.propose(self.total, msg, r or {})
        if self.kind == "ema":
            t = float(self.tau); return {k:(1-t)*self.total[k]+t*msg[k] for k in msg}
        if self.kind == "svb":
            return {k:self.total[k]+msg[k] for k in msg}
        if self.kind == "pp":
            r = float(self.rho); return {k:r*self.total[k]+msg[k] for k in msg}
        if self.kind == "window":
            drop = self.history[0] if len(self.history) == self.window else None
            n = min(len(self.history)+1, int(self.window))
            return {k:(self.window_sum[k]+msg[k]-(drop[k] if drop is not None else 0.0))/n for k in msg}
        raise ValueError(self.kind)

    def commit(self, msg, r=None):
        new = self.propose(msg, r)
        if self.kind == "window":
            if len(self.history) == self.window:
                old = self.history.popleft()
                for k in msg: self.window_sum[k] -= old[k]
            self.history.append(clone_stats(msg))
            for k in msg: self.window_sum[k] += msg[k]
        self.total = new


class SwitchingCoreND:
    def __init__(self, classes, dim, k, noise_scale=.25):
        dyn, gate, fb = classes
        self.D, self.K, self.fb = int(dim), int(k), fb
        self.dyn = dyn(K=self.K, L=self.D, G=self.D+1, action_dim=0, ard=False,
                       identity_init=False, learn_b0=False, a0=3., b0=2*float(noise_scale)**2,
                       v0_scale=.2, jitter=0., n_block_iters=1,
                       device=torch.device("cpu"), dtype=torch.float64)
        self.dyn._freeze_C = True
        self.gate = gate(K=self.K, feat_dim=self.D, prior_persist=.9,
                         weight_prior_var=.25, bias_prior_var=4., pg_iters=1,
                         device=torch.device("cpu"), dtype=torch.float64)
        self.prior_dest = torch.ones(self.K, self.K, dtype=torch.float64)
        self.prior_start = torch.ones(self.K, dtype=torch.float64)
        self.dest = self.prior_dest.clone(); self.start = self.prior_start.clone()

    def messages(self, data):
        ev = self.dyn.expected_loglik(data["y"], data["g"])
        elogpi = torch.digamma(self.dest)-torch.digamma(self.dest.sum(-1, keepdim=True))
        eloginit = torch.digamma(self.start)-torch.digamma(self.start.sum())
        logtrans, aux = self.gate.bound_log_trans(elogpi, data["phi"][:, 1:])
        gamma, _, _, xi = self.fb(eloginit, logtrans, ev, return_pairwise=True)
        stats = self.dyn.stats_from_batch(gamma, data["y"], data["g"])
        r, w, C = self.gate.attribute_bound(xi, aux)
        phi = data["phi"][:, 1:].reshape(-1, self.D+1)
        pg = self.gate.pg_stats_from_batch(phi, r.reshape(-1,self.K), w.reshape(-1,self.K))
        stats.update(C=C, start=gamma[:,0].sum(0), pg_A=pg["A"], pg_h=pg["h"])
        return stats

    def refit(self, stats, base=None, message=None, tau=None, retain=None):
        if base is not None:
            self.dyn.set_stats({k:base[k] for k in DYN_KEYS})
            self.dyn.ema_update_stats({k:message[k] for k in DYN_KEYS},
                                      tau if torch.is_tensor(tau) else float(tau), retain=retain)
        else:
            self.dyn.set_stats({k:stats[k] for k in DYN_KEYS})
        self.dyn.m_step()
        self.dest = self.prior_dest + stats["C"]
        self.start = self.prior_start + stats["start"]
        self.gate.pg_set_totals(stats["pg_A"], stats["pg_h"])
        if getattr(self.gate, "n_pg_guard_rejects", 0):
            raise FloatingPointError("AIME gate rejected an update")
        if not torch.isfinite(self.dyn.M).all() or not (self.dyn.b > 0).all():
            raise FloatingPointError("Invalid conjugate refit")

    def predict_metrics(self, data, encoder, side):
        """Pre-update latent NLL and reconstructed-frame RMSE in [0,1] pixel units."""
        g, y, phi = data["g_np"], data["y_np"], data["phi_np"]
        M = self.dyn.M.numpy(); V = self.dyn.V.numpy(); a = self.dyn.a.numpy(); b = self.dyn.b.numpy()
        mu = np.einsum("klr,btr->btkl", M, g)
        qf = np.einsum("btr,krs,bts->btk", g, V, g)
        varscale = (b/a)[None,None]*(1+qf[...,None]); df = 2*a
        residual = (y[:,:,None]-mu)**2/varscale
        logev = (gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi)-.5*np.log(varscale)
                 -.5*(df+1)*np.log1p(residual/df)).sum(-1)
        gm = np.einsum("btd,kd->btk", phi, self.gate.m_beta.numpy())
        gv = np.einsum("btd,kde,bte->btk", phi, self.gate.Sigma_beta.numpy(), phi)
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        stay = (expit(gm[...,None]+np.sqrt(2*np.maximum(gv,0))[...,None]*nodes)*weights).sum(-1)/np.sqrt(np.pi)
        dest = self.dest.numpy(); dest /= dest.sum(-1, keepdims=True)
        trans = stay[..., :, None]*np.eye(self.K)+(1-stay[..., :, None])*dest
        init = self.start.numpy(); init /= init.sum()

        B, T = y.shape[:2]
        logalpha = np.log(init)[None]+logev[:,0]
        z = logsumexp(logalpha, axis=-1); total = z.copy()
        pred_prob = np.broadcast_to(init[None], (B,self.K))
        pred_lat = [np.einsum("bk,bkd->bd", pred_prob, mu[:,0])]
        logalpha -= z[:,None]
        for t in range(1,T):
            pred_log = logsumexp(logalpha[:,:,None]+np.log(np.clip(trans[:,t],1e-300,None)), axis=-2)
            pred_prob = np.exp(pred_log-logsumexp(pred_log,axis=-1,keepdims=True))
            pred_lat.append(np.einsum("bk,bkd->bd", pred_prob, mu[:,t]))
            logalpha = pred_log+logev[:,t]
            z = logsumexp(logalpha,axis=-1); total += z; logalpha -= z[:,None]
        pred_lat = np.stack(pred_lat, axis=1)
        pred_px = encoder.decode(pred_lat, side)
        true_px = data["target_pixels"]
        rmse = float(np.sqrt(np.mean((pred_px-true_px)**2)))
        return -float(total.mean())/T, rmse


def arm_spec(taus):
    x = [(f"ema_{t:g}","ema",t) for t in taus]
    x += [("svb","svb",None),("pp","pp",MAIN_TAU),("window","window",MAIN_TAU),("gated","gated",MAIN_TAU),
          ("retain","retain",MAIN_TAU)]
    return x


def make_store(kind,t,initial,forgetting=None):
    if kind in ("gated","retain"): return StatisticsStore(kind, initial, tau=t, forgetting=forgetting)
    if kind=="ema": return StatisticsStore("ema", initial, tau=t)
    if kind=="svb": return StatisticsStore("svb", initial)
    if kind=="pp": return StatisticsStore("pp", initial, rho=1-MAIN_TAU)
    if kind=="window": return StatisticsStore("window", initial, window=window_len(MAIN_TAU))
    raise ValueError(kind)


def choose_indices(idx, cfg):
    if cfg["glyph_mode"] == "fixed":
        return np.full_like(idx, int(cfg["glyph_index"]))
    return idx


def visual_batch(ds, encoder, rng, angle, cfg):
    state, idx = ds.sample_state_batch(rng, cfg["batch"])
    idx = choose_indices(idx, cfg)
    frames = ds.render_batch(state, idx)
    small = ds.downsample(frames, cfg["downsample"])
    rotated = ds.rotate_small(small, angle)
    z = encoder.encode(rotated)
    return prepare_latent_for_model(z, rotated)


def fit_encoder(cfg, out):
    path = out/"visual_encoder.npz"
    if path.exists():
        return FixedPCAEncoder.load(path)
    ds = ControlledSegmentationMovingMNIST(cfg["mnist_root"], frame_size=(cfg["frame"],cfg["frame"]),
        num_frames=cfg["frames"], min_speed=cfg["min_speed"], max_speed=cfg["max_speed"], download=False)
    rng = np.random.default_rng(cfg["encoder_seed"])
    state, idx = ds.sample_state_batch(rng, cfg["encoder_sequences"])
    idx = choose_indices(idx, cfg)
    frames = ds.render_batch(state, idx)
    small = ds.downsample(frames, cfg["downsample"])
    enc = FixedPCAEncoder(cfg["latent_dim"]).fit(small)
    enc.save(path)
    return enc


def initialize(classes, ds, encoder, seed, cfg):
    rng = np.random.default_rng(50000+seed)
    state, idx = ds.sample_state_batch(rng, cfg["batch"]*cfg["warm_batches"])
    idx = choose_indices(idx, cfg)
    small = ds.downsample(ds.render_batch(state, idx), cfg["downsample"])
    z = encoder.encode(small)
    data = prepare_latent_for_model(z, small)
    model = SwitchingCoreND(classes, cfg["latent_dim"], cfg["K"], cfg["noise_scale"])
    flat = data["y_np"].reshape(-1,cfg["latent_dim"])
    _, lab = kmeans2(flat, cfg["K"], iter=40, minit="++", seed=rng)
    gamma = torch.from_numpy(np.eye(cfg["K"])[lab].reshape(*data["y_np"].shape[:2],cfg["K"]))
    stats = model.dyn.stats_from_batch(gamma,data["y"],data["g"])
    stats.update(C=torch.zeros(cfg["K"],cfg["K"],dtype=torch.float64), start=gamma[:,0].sum(0),
                 pg_A=model.gate.pg_A.clone(), pg_h=model.gate.pg_h.clone())
    stats = {k:v/cfg["warm_batches"] for k,v in stats.items()}; model.refit(stats)
    for _ in range(cfg["warm_iters"]):
        msg = model.messages(data); stats = {k:v/cfg["warm_batches"] for k,v in msg.items()}; model.refit(stats)
    return model, stats


def tracking_vector(z):
    """Normalized K=1 Gaussian-regression sufficient-statistic vector from visual latents."""
    prev, y = z[:,:-1], z[:,1:]
    g = np.concatenate([prev,np.ones((*prev.shape[:2],1))],axis=-1)
    G = g.reshape(-1,g.shape[-1]); Y = y.reshape(-1,y.shape[-1]); n = len(G)
    srr = G.T@G/n
    szr = Y.T@G/n
    szz = (Y*Y).mean(0)
    return np.concatenate([srr.ravel(),szr.ravel(),szz.ravel()])


def tracking_study(cfg, out, encoder):
    """Monte-Carlo population statistic under visual rotation and Proposition-2 bound."""
    path = out/"visual_tracking.npz"
    if path.exists(): return np.load(path)
    ds = ControlledSegmentationMovingMNIST(cfg["mnist_root"], frame_size=(cfg["frame"],cfg["frame"]),
        num_frames=cfg["frames"], min_speed=cfg["min_speed"], max_speed=cfg["max_speed"], download=False)
    rng = np.random.default_rng(77123)
    state, idx = ds.sample_state_batch(rng, cfg["tracking_sequences"]); idx = choose_indices(idx,cfg)
    small0 = ds.downsample(ds.render_batch(state,idx),cfg["downsample"])
    angles = np.linspace(0,cfg["final_angle_deg"],cfg["drift_steps"])
    m = []
    for a in angles:
        z = encoder.encode(ds.rotate_small(small0,float(a)))
        m.append(tracking_vector(z))
    m = np.stack(m)
    delta = float(np.linalg.norm(np.diff(m,axis=0),axis=1).max())
    res = {"angles":angles,"m":m,"delta":np.array(delta)}
    for t in cfg["tracking_taus"]:
        ema = m[0].copy(); err = np.zeros(len(m))
        for b in range(1,len(m)):
            ema = (1-t)*ema+t*m[b]
            err[b] = np.linalg.norm(ema-m[b])
        bound = (1-t)*delta/t*(1-(1-t)**np.arange(len(m)))
        res[f"error_{t:g}"] = err; res[f"bound_{t:g}"] = bound
    np.savez_compressed(path,**res)
    return res


def _visual_batch_n(ds, encoder, rng, angle, nseq, cfg):
    """Independent visual batch of nseq sequences at one fixed observation angle."""
    state, idx = ds.sample_state_batch(rng, int(nseq))
    idx = choose_indices(idx, cfg)
    frames = ds.render_batch(state, idx)
    small = ds.downsample(frames, cfg["downsample"])
    rotated = ds.rotate_small(small, float(angle))
    z = encoder.encode(rotated)
    return prepare_latent_for_model(z, rotated)


def _fit_static_reference(classes, data, cfg, rng):
    model = SwitchingCoreND(classes, cfg["latent_dim"], cfg["K"], cfg["noise_scale"])
    flat = data["y_np"].reshape(-1, cfg["latent_dim"])
    _, lab = kmeans2(flat, cfg["K"], iter=40, minit="++", seed=rng)
    gamma = torch.from_numpy(np.eye(cfg["K"])[lab].reshape(*data["y_np"].shape[:2], cfg["K"]))
    stats = model.dyn.stats_from_batch(gamma, data["y"], data["g"])
    stats.update(
        C=torch.zeros(cfg["K"], cfg["K"], dtype=torch.float64),
        start=gamma[:, 0].sum(0),
        pg_A=model.gate.pg_A.clone(),
        pg_h=model.gate.pg_h.clone(),
    )
    model.refit(stats)
    for _ in range(cfg["reference_iters"]):
        stats = model.messages(data)
        model.refit(stats)
    return model


@torch.no_grad()
def angle_reference_study(cfg, out, encoder, classes):
    path = out/"angle_reference.npz"
    if path.exists():
        return np.load(path)

    ds = ControlledSegmentationMovingMNIST(
        cfg["mnist_root"], frame_size=(cfg["frame"], cfg["frame"]),
        num_frames=cfg["frames"], min_speed=cfg["min_speed"], max_speed=cfg["max_speed"],
        download=False,
    )
    angles = np.asarray(cfg["reference_angles"], dtype=float)
    nll = np.zeros((cfg["reference_reps"], len(angles)), dtype=np.float64)
    latent_var = np.zeros_like(nll)

    for r in range(cfg["reference_reps"]):
        for j, angle in enumerate(angles):
            code = int(round(angle * 100))
            train_rng = np.random.default_rng(cfg["reference_seed"] + 100000*r + code)
            test_rng = np.random.default_rng(cfg["reference_seed"] + 5000000 + 100000*r + code)
            train = _visual_batch_n(ds, encoder, train_rng, angle, cfg["reference_train_sequences"], cfg)
            model = _fit_static_reference(classes, train, cfg, train_rng)
            test = _visual_batch_n(ds, encoder, test_rng, angle, cfg["reference_test_sequences"], cfg)
            nll[r, j], _ = model.predict_metrics(test, encoder, cfg["downsample"])
            z = test["y_np"].reshape(-1, cfg["latent_dim"])
            latent_var[r, j] = np.var(z, axis=0, ddof=1).sum()

    np.savez_compressed(path, angles=angles, nll=nll, latent_var=latent_var)

    with (out/"angle_reference.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "reference_nll_mean", "reference_nll_ci95_lo", "reference_nll_ci95_hi",
                    "latent_variance_mean", "latent_variance_ci95_lo", "latent_variance_ci95_hi"])
        for j, angle in enumerate(angles):
            nm, nl, nh = ci95(nll[:, j])
            vm, vl, vh = ci95(latent_var[:, j])
            w.writerow([f"{angle:g}", f"{nm:.6f}", f"{nl:.6f}", f"{nh:.6f}",
                        f"{vm:.6f}", f"{vl:.6f}", f"{vh:.6f}"])
    return np.load(path)


@torch.no_grad()
def run_seed(task):
    seed,cfg,source_root,outdir = task
    torch.set_num_threads(1)
    dyn,gate,fb,sha,fg = load_aime(source_root); classes=(dyn,gate,fb)
    out=Path(outdir); encoder=FixedPCAEncoder.load(out/"visual_encoder.npz")
    ds=ControlledSegmentationMovingMNIST(cfg["mnist_root"],frame_size=(cfg["frame"],cfg["frame"]),
        num_frames=cfg["frames"],min_speed=cfg["min_speed"],max_speed=cfg["max_speed"],download=False)
    schedule=StreamSchedule(cfg["change_start"],cfg["drift_steps"],cfg["final_angle_deg"])
    model0,initial=initialize(classes,ds,encoder,seed,cfg)
    specs=arm_spec(cfg["taus"]); names=[a[0] for a in specs]
    result={}; clock=time.perf_counter()
    for condition in ("stationary","drift"):
        models={nm:copy.deepcopy(model0) for nm in names}; stores={nm:make_store(k,t,initial,fg) for nm,k,t in specs}
        nll=np.zeros((len(specs),cfg["steps"])); rmse=np.zeros_like(nll)
        ret=np.full((len(FACTORS),5,cfg["steps"]),np.nan)
        for b in range(cfg["steps"]):
            rng=np.random.default_rng(100000+1000000*seed+b)
            angle=0.0 if condition=="stationary" else schedule.angle_deg(b)
            data=visual_batch(ds,encoder,rng,angle,cfg)
            for j,(nm,kind,t) in enumerate(specs):
                nll[j,b],rmse[j,b]=models[nm].predict_metrics(data,encoder,cfg["downsample"])
                msg=None; r=None
                for i in range(cfg["local_iters"]):
                    msg=models[nm].messages(data)
                    tau=t
                    if kind in ("gated","retain"):
                        r,info=stores[nm].retain.factors(models[nm],stores[nm].total,msg,learn=i==cfg["local_iters"]-1)
                        tau=stores[nm].retain.gain["emission"]
                    prop=stores[nm].propose(msg,r)
                    models[nm].refit(prop,base=stores[nm].total if kind in ("ema","gated","retain") else None,
                                     message=msg,tau=tau if kind in ("ema","gated","retain") else None,
                                     retain=(r or {}).get("emission"))
                stores[nm].commit(msg,r)
                if kind=="retain": ret[...,b]=retention_summary(stores[nm].retain,r,info)
        result[f"{condition}_nll"]=nll; result[f"{condition}_rmse"]=rmse; result[f"{condition}_retention"]=ret
    tmp=out/f"seed_{seed:03d}.partial"
    with tmp.open("wb") as f: np.savez_compressed(f,arms=np.array(names),source_sha=json.dumps(sha),**result)
    tmp.replace(out/f"seed_{seed:03d}.npz")
    return {"seed":seed,"seconds":round(time.perf_counter()-clock,1)}


def ci95(v):
    v=np.asarray(v,float); m=float(v.mean())
    if len(v)<2:return m,m,m
    h=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))); return m,m-h,m+h


def periods(cfg):
    cs=cfg["change_start"]; de=min(cfg["steps"],cs+cfg["drift_steps"])
    return {"pre":slice(0,cs),"drift":slice(cs,de),"post":slice(de,cfg["steps"])}


def summarize(out,cfg):
    data=[np.load(out/f"seed_{s:03d}.npz") for s in cfg["seed_ids"]]
    names=list(data[0]["arms"]); p=periods(cfg); ref=names.index(f"ema_{MAIN_TAU:g}")
    rows=[]
    for cond in ("stationary","drift"):
        for metric in ("nll","rmse"):
            arr=np.stack([d[f"{cond}_{metric}"] for d in data])
            for j,nm in enumerate(names):
                for period,sl in p.items():
                    vals=arr[:,j,sl].mean(-1); base=arr[:,ref,sl].mean(-1)
                    m,lo,hi=ci95(vals); dm,dlo,dhi=ci95(vals-base)
                    rows.append(dict(condition=cond,method=nm,period=period,metric=metric,mean=m,ci95_lo=lo,
                                     ci95_hi=hi,minus_ema=dm,paired_ci95_lo=dlo,paired_ci95_hi=dhi))
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    main_methods=[f"ema_{MAIN_TAU:g}","svb","pp","window","gated","retain"]
    rrows=[r for r in rows if r["condition"]=="drift" and r["period"]=="drift"
           and r["metric"]=="rmse" and r["method"] in main_methods]
    vals={r["method"]:r["mean"] for r in rrows}
    if vals:
        spread=max(vals.values())-min(vals.values())
        note=("One-step reconstructed-frame RMSE was very similar across the main update rules "
              f"during the drift window (range {min(vals.values()):.6f}–{max(vals.values()):.6f}; "
              f"absolute spread {spread:.6f}). RMSE is therefore reported in summary.csv rather than "
              "used as a main figure panel.\n")
        (out/"rmse_supplement.txt").write_text(note)
    return rows


def _binned(arr,width):
    starts=list(range(0,arr.shape[-1],width)); x=np.array([(s+min(s+width,arr.shape[-1])-1)/2+1 for s in starts])
    y=np.stack([arr[...,s:min(s+width,arr.shape[-1])].mean(-1) for s in starts],axis=-1); return x,y


def _sensitivity_values(data, names, cfg):
    nll=np.stack([d["drift_nll"] for d in data])
    cs=cfg["change_start"];de=min(cfg["steps"],cs+cfg["drift_steps"]);sl=slice(cs,de)
    xs=[];ym=[];lo=[];hi=[]
    for t in cfg["taus"]:
        vals=nll[:,names.index(f"ema_{t:g}"),sl].mean(-1)
        m,l,h=ci95(vals);xs.append(t);ym.append(m);lo.append(m-l);hi.append(h-m)
    refs={}
    for nm in ("svb","pp","window","gated","retain"):
        vals=nll[:,names.index(nm),sl].mean(-1)
        refs[nm]=ci95(vals)
    return np.asarray(xs),np.asarray(ym),np.asarray(lo),np.asarray(hi),refs


def plot_main(out,cfg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":9.5,"axes.spines.top":True,"axes.spines.right":True,"axes.edgecolor":".15",
                         "svg.fonttype":"none","pdf.fonttype":42,"font.family":"serif","mathtext.fontset":"stix",
                         "font.serif":["TeX Gyre Termes","Nimbus Roman","Times New Roman","Times","STIXGeneral","DejaVu Serif"]})
    tracking=np.load(out/"visual_tracking.npz")
    data=[np.load(out/f"seed_{s:03d}.npz") for s in cfg["seed_ids"]]
    names=list(data[0]["arms"]); n=len(data)
    nll=np.stack([d["drift_nll"] for d in data])
    cs=cfg["change_start"]; de=min(cfg["steps"],cs+cfg["drift_steps"])
    shown=[f"ema_{MAIN_TAU:g}","svb","pp","window","gated","retain"]
    style={f"ema_{MAIN_TAU:g}":("ema",rf"EMA ($\tau={MAIN_TAU:g}$)"),"svb":("svb","Streaming VB"),
           "pp":("pp",rf"Power prior ($\rho={1-MAIN_TAU:g}$)"),
           "window":("window",rf"Window ($W={window_len(MAIN_TAU)}$)"),
           "gated":("gated","AIME, no retention"),
           "retain":("retain","AIME (proposed): gated + change-point retention")}

    fig,axs=plt.subplots(1,3,figsize=(13.2,3.8),gridspec_kw={"width_ratios":[1.05,1.2,1.05]})
    fig.subplots_adjust(wspace=.32,bottom=.24)
    def panel(ax,s):
        ax.text(-.13,1.02,s,transform=ax.transAxes,fontweight="bold",fontsize=11,va="bottom")
        ax.grid(axis="y",alpha=.16)

    ax=axs[0]; shades=["#f5b594","#eb6834","#a8401a"]
    x=np.arange(len(tracking["angles"]))+1
    for t,col in zip(cfg["tracking_taus"],shades):
        ax.plot(x,tracking[f"error_{t:g}"],color=col,lw=1.8,label=rf"$\tau={t:g}$")
        ax.plot(x,tracking[f"bound_{t:g}"],color=col,lw=1.0,ls="--")
    ax.set_xlabel("Drift update")
    ax.set_ylabel(r"Statistic tracking error $\|\bar s_b-m_b\|_2$")
    panel(ax,"(a)")
    ax2=ax.twiny();ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([1,len(x)//2,len(x)])
    ax2.set_xticklabels([r"$0^\circ$",r"$45^\circ$",r"$90^\circ$"])
    ax2.tick_params(axis="x",labelsize=8,length=0)
    ax.legend(frameon=False,fontsize=8,loc="upper left")

    ax=axs[1]
    xb,curve=_binned(nll,cfg["bin_width"]);mean=curve.mean(0)
    half=student_t.ppf(.975,n-1)*curve.std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros_like(mean)
    ax.axvspan(cs+1,de,color=".92",lw=0,zorder=0)
    for nm in shown:
        j=names.index(nm);fam,lab=style[nm]
        ax.plot(xb,mean[j],color=COLOR[fam],ls=LINESTYLE[fam],lw=2.5 if fam=="retain" else 1.6,label=lab,zorder=3 if fam=="retain" else 2)
        ax.fill_between(xb,mean[j]-half[j],mean[j]+half[j],color=COLOR[fam],alpha=.10,lw=0)
    ax.axvline(cs+1,color=".65",lw=.8,ls="--");ax.axvline(de,color=".65",lw=.8,ls="--")
    ax.set_xlabel("Minibatch update");ax.set_ylabel("Prequential NLL (nats / transition)")
    panel(ax,"(b)")

    ax=axs[2]
    xs,ym,lo,hi,refs=_sensitivity_values(data,names,cfg)
    ax.errorbar(xs,ym,yerr=np.array([lo,hi]),color=COLOR["ema"],marker="o",lw=1.5,
                capsize=2.5,label="EMA")
    for nm,fam in (("svb","svb"),("pp","pp"),("window","window"),("gated","gated"),("retain","retain")):
        ax.axhline(refs[nm][0],color=COLOR[fam],ls=LINESTYLE[fam],lw=1.25)
    ax.axvline(MAIN_TAU,color=".45",lw=.8,ls=":",zorder=0)
    ax.scatter([MAIN_TAU],[ym[list(xs).index(MAIN_TAU)]],s=42,facecolor="white",
               edgecolor=COLOR["ema"],linewidth=1.2,zorder=4)
    ax.set_xscale("log");ax.set_xlabel(r"EMA gain $\tau$");ax.set_ylabel("Mean drift NLL")
    panel(ax,"(c)")

    handles=[Line2D([],[],color=COLOR[f],ls=LINESTYLE[f],lw=1.8,label=l) for _,(f,l) in style.items()]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.03),ncol=4,frameon=False,fontsize=8.5)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"figure_visual_mnist_prop2.{ext}",dpi=240,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def plot_sensitivity(out,cfg):
    """Panel (c) as a standalone supplementary figure."""
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    data=[np.load(out/f"seed_{s:03d}.npz") for s in cfg["seed_ids"]];names=list(data[0]["arms"])
    xs,ym,lo,hi,refs=_sensitivity_values(data,names,cfg)
    fig,ax=plt.subplots(figsize=(4.4,3.3))
    ax.errorbar(xs,ym,yerr=np.array([lo,hi]),color=COLOR["ema"],marker="o",lw=1.5,
                capsize=2.5,label="EMA")
    for nm,fam,lab in [("svb","svb","Streaming VB"),
                       ("pp","pp",rf"Power prior ($\rho={1-MAIN_TAU:g}$)"),
                       ("window","window",rf"Window ($W={window_len(MAIN_TAU)}$)"),
                       ("gated","gated","AIME, no retention"),
                       ("retain","retain","AIME (proposed): gated + change-point retention")]:
        ax.axhline(refs[nm][0],color=COLOR[fam],ls=LINESTYLE[fam],lw=1.25,label=lab)
    ax.axvline(MAIN_TAU,color=".45",lw=.8,ls=":",zorder=0)
    ax.set_xscale("log");ax.set_xlabel(r"EMA gain $\tau$");ax.set_ylabel("Mean drift NLL")
    ax.grid(axis="y",alpha=.16);ax.legend(frameon=False,fontsize=8)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"figure_visual_mnist_tau_sensitivity.{ext}",dpi=240,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def plot_angle_sanity(out,cfg):
    """Sanity check for angle-dependent density scale of the frozen PCA representation."""
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    r=np.load(out/"angle_reference.npz")
    angles=r["angles"];nll=r["nll"];var=r["latent_var"]
    fig,axs=plt.subplots(1,2,figsize=(7.6,3.1))
    for ax,arr,ylabel,letter in [
        (axs[0],nll,"Angle-wise batch-refit NLL","(a)"),
        (axs[1],var,"Total PCA latent variance","(b)"),
    ]:
        mean=arr.mean(0)
        if arr.shape[0]>1:
            half=student_t.ppf(.975,arr.shape[0]-1)*arr.std(0,ddof=1)/np.sqrt(arr.shape[0])
        else:
            half=np.zeros_like(mean)
        ax.errorbar(angles,mean,yerr=half,color="#4C78A8",marker="o",lw=1.5,capsize=2.5)
        ax.set_xlabel("Image rotation (degrees)");ax.set_ylabel(ylabel)
        ax.grid(axis="y",alpha=.16)
        ax.text(-.13,1.02,letter,transform=ax.transAxes,fontweight="bold",fontsize=10,va="bottom")
    fig.subplots_adjust(wspace=.35)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"figure_visual_mnist_angle_sanity.{ext}",dpi=240,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def plot_protocol(out,cfg):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    ds=ControlledSegmentationMovingMNIST(cfg["mnist_root"],frame_size=(cfg["frame"],cfg["frame"]),num_frames=cfg["frames"],min_speed=cfg["min_speed"],max_speed=cfg["max_speed"],download=False)
    rng=np.random.default_rng(9017);state,idx=ds.sample_state_batch(rng,1);idx=choose_indices(idx,cfg)
    small=ds.downsample(ds.render_batch(state,idx),cfg["downsample"]);frame=small[0,cfg["frames"]//2]
    angles=[0.,cfg["final_angle_deg"]/2,cfg["final_angle_deg"]];fig,axs=plt.subplots(1,3,figsize=(5.7,1.9))
    for ax,a in zip(axs,angles):
        fr=ds.rotate_small(frame[None,None],a)[0,0];ax.imshow(fr,cmap="gray",vmin=0,vmax=1,interpolation="nearest");ax.set_title(rf"${a:.0f}^\circ$",fontsize=9);ax.axis("off")
    fig.subplots_adjust(wspace=.04)
    for ext in ("pdf","png"):fig.savefig(out/f"figure_visual_mnist_protocol.{ext}",dpi=240,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root",type=Path,required=True);p.add_argument("--mnist-root",type=Path,default=here/"data");p.add_argument("--out",type=Path,default=here/"results_visual_mnist_prop2")
    p.add_argument("--stage",choices=["all","run","plot","protocol"],default="all");p.add_argument("--seeds",type=int,default=20);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--steps",type=int,default=600);p.add_argument("--change-start",type=int,default=300);p.add_argument("--drift-steps",type=int,default=150);p.add_argument("--final-angle-deg",type=float,default=90.)
    p.add_argument("--taus",nargs="+",type=float,default=[.005,.01,.02,.05,.1,.25]);p.add_argument("--tracking-taus",nargs="+",type=float,default=[.01,.02,.1])
    p.add_argument("--batch",type=int,default=12);p.add_argument("--frames",type=int,default=20);p.add_argument("--frame",type=int,default=64);p.add_argument("--downsample",type=int,default=16)
    p.add_argument("--latent-dim",type=int,default=8);p.add_argument("--K",type=int,default=4);p.add_argument("--min-speed",type=float,default=2.);p.add_argument("--max-speed",type=float,default=6.)
    p.add_argument("--local-iters",type=int,default=3);p.add_argument("--warm-batches",type=int,default=16);p.add_argument("--warm-iters",type=int,default=10);p.add_argument("--noise-scale",type=float,default=.25);p.add_argument("--bin-width",type=int,default=10)
    p.add_argument("--encoder-sequences",type=int,default=512);p.add_argument("--encoder-seed",type=int,default=61017);p.add_argument("--tracking-sequences",type=int,default=256)
    p.add_argument("--reference-angles",nargs="+",type=float,default=[0,15,30,45,60,75,90])
    p.add_argument("--reference-reps",type=int,default=3);p.add_argument("--reference-train-sequences",type=int,default=256);p.add_argument("--reference-test-sequences",type=int,default=256)
    p.add_argument("--reference-iters",type=int,default=8);p.add_argument("--reference-seed",type=int,default=81231)
    p.add_argument("--glyph-mode",choices=["fixed","random"],default="fixed");p.add_argument("--glyph-index",type=int,default=0)
    a=p.parse_args()
    if MAIN_TAU not in a.taus:p.error("--taus must include 0.02")
    a.out.mkdir(parents=True,exist_ok=True)
    cfg=dict(steps=a.steps,change_start=a.change_start,drift_steps=a.drift_steps,final_angle_deg=a.final_angle_deg,taus=sorted(a.taus),tracking_taus=a.tracking_taus,
             batch=a.batch,frames=a.frames,frame=a.frame,downsample=a.downsample,latent_dim=a.latent_dim,K=a.K,min_speed=a.min_speed,max_speed=a.max_speed,
             local_iters=a.local_iters,warm_batches=a.warm_batches,warm_iters=a.warm_iters,noise_scale=a.noise_scale,bin_width=a.bin_width,encoder_sequences=a.encoder_sequences,
             encoder_seed=a.encoder_seed,tracking_sequences=a.tracking_sequences,reference_angles=a.reference_angles,reference_reps=a.reference_reps,
             reference_train_sequences=a.reference_train_sequences,reference_test_sequences=a.reference_test_sequences,reference_iters=a.reference_iters,
             reference_seed=a.reference_seed,glyph_mode=a.glyph_mode,glyph_index=a.glyph_index,mnist_root=str(a.mnist_root.resolve()),seed_ids=list(range(a.seeds)))
    cpath=a.out/"config.json"
    if cpath.exists() and json.loads(cpath.read_text())!=cfg:raise ValueError("Output directory has a different configuration; choose a new --out")
    cpath.write_text(json.dumps(cfg,indent=2)+"\n")
    if a.stage in ("all","run"):
        enc=fit_encoder(cfg,a.out);tracking_study(cfg,a.out,enc)
        dyn,gate,fb,_,_=load_aime(a.source_root);angle_reference_study(cfg,a.out,enc,(dyn,gate,fb))
        tasks=[(s,cfg,str(a.source_root.resolve()),str(a.out.resolve())) for s in cfg["seed_ids"] if not (a.out/f"seed_{s:03d}.npz").exists()]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(run_seed,t) for t in tasks]): print(json.dumps(f.result()),flush=True)
    if a.stage in ("all","plot"):
        summarize(a.out,cfg);plot_main(a.out,cfg);plot_sensitivity(a.out,cfg);plot_angle_sanity(a.out,cfg)
    if a.stage in ("all","protocol"): plot_protocol(a.out,cfg)

if __name__=="__main__":main()
