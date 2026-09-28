import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
if not hasattr(torch, "softplus"):
    torch.softplus = F.softplus

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_OUT = os.environ.get("SHS_OUT", _HERE)
from shs_rssm.shs_rssm import SHSRSSM
from shs_rssm.shs_diagnostics import (regime_responsibilities, plot_latent_clustering, plot_reconstructions)
from shs_rssm.regime_head import RegimeHead
from shs_rssm.moves import delete_move, merge_move
torch.manual_seed(0); np.random.seed(0)
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
H = W = 32
STOCH, DETER, EMBED, K = 24, 96, 192, 6
FEAT = STOCH + DETER
CH = 3
SHS_RECURRENT = True

ANGS = [0.45, -0.75, 1.10]
stay, Ktrue = 0.97, 3
P = np.full((Ktrue, Ktrue), (1 - stay) / (Ktrue - 1)); np.fill_diagonal(P, stay)

N_OBJ      = 4
T          = 48
T_APPEAR   = T // 2
SHAPES     = ["disk", "square", "triangle", "diamond"]
RADII      = [0.58, 0.62, 0.26, 0.46]
SPEED_MULT = [0.70, 1.25, 0.50, 1.00]
PHASE0     = [0.30, 2.45, 4.00, 5.25]
SIZE       = [4.0, 3.6, 5.6, 4.6]
SPIN_OBJ   = 2
SPIN_RATE  = 0.55

B = 18
pos  = np.zeros((B, T, N_OBJ, 2))
spin = np.zeros((B, T))
S    = np.zeros((B, T), dtype=int)
for b in range(B):
    s = np.random.randint(Ktrue)
    th = np.array(PHASE0) + np.random.uniform(-0.3, 0.3, N_OBJ)
    p = np.stack([np.array(RADII) * np.cos(th), np.array(RADII) * np.sin(th)], axis=1)
    sp = np.random.uniform(0, 2 * np.pi)
    for t in range(T):
        if t > 0:
            s = np.random.choice(Ktrue, p=P[s])
            w = ANGS[s]
            for i in range(N_OBJ):
                a = w * SPEED_MULT[i]
                c, si = np.cos(a), np.sin(a)
                p[i] = np.array([[c, -si], [si, c]]) @ p[i] + 0.004 * np.random.randn(2)
                p[i] = p[i] / (np.linalg.norm(p[i]) + 1e-8) * RADII[i]
            sp += SPIN_RATE
        S[b, t] = s; pos[b, t] = p; spin[b, t] = sp

vis = np.ones((B, T, N_OBJ))
ramp = np.clip((np.arange(T) - T_APPEAR) / 3.0 + 1.0, 0.0, 1.0)
vis[:, :, 3] = ramp[None, :]

ys, xs = np.mgrid[0:H, 0:W]
def _px(pos2d):
    px = (pos2d[..., 0] * 0.5 + 0.5) * (W - 1)
    py = (pos2d[..., 1] * 0.5 + 0.5) * (H - 1)
    return px, py
def _grid(px, py):
    dx = xs[None, None] - px[..., None, None]
    dy = ys[None, None] - py[..., None, None]
    return dx, dy
def shape_field(kind, pos2d, size, sharp=0.7, angle=None):
    px, py = _px(pos2d); dx, dy = _grid(px, py)
    if kind == "disk":
        d = np.sqrt(dx ** 2 + dy ** 2) - size
    elif kind == "square":
        d = np.maximum(np.abs(dx), np.abs(dy)) - size
    elif kind == "diamond":
        d = (np.abs(dx) + np.abs(dy)) - size
    elif kind == "triangle":
        ang = np.zeros_like(px) if angle is None else angle
        ca, sa = np.cos(-ang)[..., None, None], np.sin(-ang)[..., None, None]
        rx, ry = ca * dx - sa * dy, sa * dx + ca * dy
        apo = size * 0.5
        d1 = -ry - apo
        d2 = (-0.86602 * rx + 0.5 * ry) - apo
        d3 = (0.86602 * rx + 0.5 * ry) - apo
        d = np.maximum(np.maximum(d1, d2), d3)
    else:
        raise ValueError(kind)
    return 1.0 / (1.0 + np.exp(d / sharp))

COLORS = np.array([[0.92, 0.16, 0.16],
                   [0.18, 0.78, 0.22],
                   [0.20, 0.42, 0.96],
                   [0.97, 0.82, 0.10]])
frame = np.zeros((B, T, H, W, 3))
for i, kind in enumerate(SHAPES):
    ang = spin if i == SPIN_OBJ else None
    a = (shape_field(kind, pos[:, :, i, :], SIZE[i], sharp=0.5, angle=ang)
         * vis[:, :, i][..., None, None])[..., None]
    frame = a * COLORS[i][None, None, None, None, :] + (1 - a) * frame
frames = torch.tensor(frame.transpose(0, 1, 4, 2, 3), dtype=torch.float32)
action = torch.zeros(B, T, 1)
is_first = torch.zeros(B, T); is_first[:, 0] = 1.0
print(f"video {tuple(frames.shape)}  shapes={SHAPES}  4th appears at t={T_APPEAR}  "
      f"mode fractions {[round((S == k).mean(), 2) for k in range(Ktrue)]}")

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CH, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU())
        self.fc = nn.Linear(128 * 4 * 4, EMBED)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEAT, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, CH, 4, 2, 1))
    def forward(self, f):
        return torch.sigmoid(self.net(self.fc(f).view(-1, 128, 4, 4)))

enc, dec = Enc(), Dec()
rssm = SHSRSSM(stoch=STOCH, deter=DETER, hidden=64, rec_depth=1, discrete=False,
               act="SiLU", norm=True, mean_act="none", std_act="softplus", min_std=0.1,
               unimix_ratio=0.0, initial="learned", num_actions=1, embed=EMBED, device="cpu",
               shs_K=K, shs_proj_dim=16, shs_kappa=40.0, shs_ema_tau=0.05, shs_ard=False,
               shs_hdp_iters=1, shs_global_update_every=4,
               shs_recurrent=SHS_RECURRENT, shs_prior_persist=0.95, shs_pg_iters=4)
params = list(enc.parameters()) + list(dec.parameters()) + list(rssm.parameters())
opt = torch.optim.Adam(params, lr=2e-3)

def encode_all(x):
    BB, TT = x.shape[:2]
    return enc(x.reshape(BB * TT, CH, H, W)).reshape(BB, TT, EMBED)

CKPT = os.path.join(_HERE, 'shs_demo_ckpt.pt')
if ITERS == 0:
    sd = torch.load(CKPT)
    enc.load_state_dict(sd['enc']); dec.load_state_dict(sd['dec']); rssm.load_state_dict(sd['rssm'])
    print("loaded checkpoint (skipping training)")
else:
    if os.environ.get("RESUME") and os.path.exists(CKPT):
        sd = torch.load(CKPT)
        enc.load_state_dict(sd['enc']); dec.load_state_dict(sd['dec']); rssm.load_state_dict(sd['rssm'])
        print(f"resumed from checkpoint; training {ITERS} more iters")
    t0 = time.time()
    for it in range(ITERS):
        embed = encode_all(frames)
        post, prior = rssm.observe(embed, action, is_first)
        feat = rssm.get_feat(post)
        recon = dec(feat.reshape(B * T, FEAT)).reshape(B, T, CH, H, W)
        recon_loss = F.binary_cross_entropy(recon, frames, reduction="none").sum((2, 3, 4)).mean()
        klloss, _, _, _ = rssm.kl_loss(post, prior, free=10.0, dyn_scale=0.5, rep_scale=0.1)
        loss = recon_loss + klloss.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, ITERS // 10) == 0 or it == ITERS - 1:
            with torch.no_grad():
                g = rssm.regime.regime_inference(post["stoch"].float(), post["deter"].float(),
                                                 is_first)[0]
                oc = g.reshape(-1, K).sum(0); nu = int((oc / oc.sum() > 0.02).sum())
            print(f"  it {it:4d}  recon {recon_loss.item():7.3f}  kl {klloss.mean().item():6.3f}  "
                  f"used {nu}/{K}  ({(time.time() - t0) / (it + 1):.2f}s/it)")
    print(f"trained {ITERS} iters in {time.time() - t0:.1f}s")
    torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict(), 'rssm': rssm.state_dict()}, CKPT)


def fit_regimes_posthoc(z, K0=8, iters_burn=18, iters_prune=12):
    B, T, L = z.shape
    head = RegimeHead(stoch=L, deter=1, K=K0, proj_dim=0, ard=False,
                      gamma=6.0, alpha=1.0, kappa=60.0, start_alpha=1.0, a0=2.0, b0=0.5,
                      ema_tau=1.0, hdp_iters=3, identity_init=True, dtype=torch.float32)
    with torch.no_grad():
        for k in range(K0):
            M0 = torch.eye(L)
            if L >= 2:
                a = float(np.random.uniform(-1.2, 1.2)); c, s = np.cos(a), np.sin(a)
                M0[:2, :2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float32) * 0.95
            head.regimes.M[k, :, :L] = M0 + 0.05 * torch.randn(L, L)
            head.regimes.M[k, :, -1] = 0.05 * torch.randn(L)
    d0 = torch.zeros(B, T, 1); isf = torch.zeros(B, T); isf[:, 0] = 1.0
    for _ in range(iters_burn):
        g, c, s0, _ = head.regime_inference(z, d0, isf)
        head.update_globals(z, d0, g, c, s0, isf)
    mm = 0.01 * B * T
    for _ in range(iters_prune):
        for _ in range(2):
            g, c, s0, _ = head.regime_inference(z, d0, isf)
            head.update_globals(z, d0, g, c, s0, isf)
        delete_move(head, z, d0, isf, threshold=0.0)
        merge_move(head, z, d0, isf, threshold=0.0)
    gamma, *_ = head.regime_inference(z, d0, isf)
    return gamma, head.K

def relabel_to_true(gamma, S, ktrue):
    g = gamma.reshape(-1, gamma.shape[-1]).detach().numpy()
    lab = g.argmax(-1); s = S.reshape(-1); Kc = gamma.shape[-1]
    cont = np.zeros((Kc, ktrue))
    for k in range(Kc):
        for j in range(ktrue):
            cont[k, j] = ((lab == k) & (s == j)).sum()
    m = cont.argmax(1)
    merged = torch.zeros(*gamma.shape[:-1], ktrue)
    for k in range(Kc):
        merged[..., int(m[k])] += gamma[..., k]
    acc = (m[lab].reshape(S.shape) == S).mean()
    return merged, float(acc)

def plot_true_vs_inferred_dynamics(S, gamma, pos, angs, mult, Ptrue, path, n_ep=5):
    import matplotlib.pyplot as plt
    Bn, Tn = S.shape; nmode = len(angs); Kc = gamma.shape[-1]
    lab = gamma.argmax(-1).detach().numpy()
    cont = np.zeros((Kc, nmode))
    for k in range(Kc):
        for j in range(nmode):
            cont[k, j] = ((lab == k) & (S == j)).sum()
    mp = cont.argmax(1); inf = mp[lab]
    acc = float((inf == S).mean())
    th = np.arctan2(pos[..., 1], pos[..., 0])
    dth = np.angle(np.exp(1j * (th[:, 1:] - th[:, :-1])))
    obs = (dth / np.array(mult)[None, None, :]).mean(-1)
    inf_av = np.array([obs[inf[:, 1:] == j].mean() if (inf[:, 1:] == j).any() else np.nan
                       for j in range(nmode)])
    cols = np.array([[0.12, 0.47, 0.71], [0.84, 0.15, 0.16], [0.17, 0.63, 0.17],
                     [0.90, 0.60, 0.0], [0.50, 0.30, 0.70]])
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    rows, ylab = [], []
    for e in range(min(n_ep, Bn)):
        rows += [S[e], inf[e], np.full(Tn, -1)]
        ylab += [f"ep {e}  true", "       inferred", ""]
    img = np.stack(rows); rgb = np.ones((*img.shape, 3))
    for j in range(nmode):
        rgb[img == j] = cols[j]
    ax.imshow(rgb, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlabel("time step")
    ax.set_title(f"switching-regime recovery: true vs inferred over time   "
                 f"(mode segmentation acc {acc * 100:.1f}%)")
    ax2 = fig.add_subplot(gs[1, 0])
    Tinf = np.zeros((nmode, nmode))
    for b in range(Bn):
        for t in range(1, Tn):
            Tinf[inf[b, t - 1], inf[b, t]] += 1
    Tinf = Tinf / Tinf.sum(1, keepdims=True).clip(min=1)
    both = np.concatenate([Ptrue, np.full((nmode, 1), np.nan), Tinf], 1)
    ax2.imshow(both, cmap='magma', vmin=0, vmax=1)
    for j in range(nmode):
        for i in range(nmode):
            ax2.text(i, j, f"{Ptrue[j, i]:.2f}", ha='center', va='center',
                     color='w' if Ptrue[j, i] < 0.6 else 'k', fontsize=8)
            ax2.text(i + nmode + 1, j, f"{Tinf[j, i]:.2f}", ha='center', va='center',
                     color='w' if Tinf[j, i] < 0.6 else 'k', fontsize=8)
    ax2.set_xticks([1, nmode + 2]); ax2.set_xticklabels(["true P", "inferred"], fontsize=9)
    ax2.set_yticks(range(nmode)); ax2.set_yticklabels([f"from {j}" for j in range(nmode)], fontsize=8)
    ax2.set_title("sticky transition matrix  (high diagonal = persistent regimes)")
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(nmode)
    ax3.bar(x - 0.2, np.array(angs), 0.4, label="true", color="0.35")
    ax3.bar(x + 0.2, inf_av, 0.4, label="inferred (recovered)", color="tab:orange")
    ax3.axhline(0, color="0.6", lw=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels([f"mode {j}" for j in range(nmode)])
    ax3.set_ylabel("per-step angular velocity (rad)")
    ax3.set_title("dynamics each regime captured"); ax3.legend(fontsize=8)
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()
    print(f"true-vs-inferred dynamics: acc={acc:.3f}, recovered omega={np.round(inf_av, 3)} "
          f"(true {np.round(np.array(angs), 3)})")

with torch.no_grad():
    embed = encode_all(frames)
    post, _ = rssm.observe(embed, action, is_first)
    z_learned = post["mean"].float()

scene = np.concatenate([pos[:, :, :3, :].reshape(B, T, 6),
                        np.cos(spin)[..., None], np.sin(spin)[..., None]], axis=-1)
Zf = z_learned.reshape(-1, STOCH).numpy()
Zaug = np.concatenate([Zf, np.ones((Zf.shape[0], 1))], 1)
yf = scene.reshape(-1, scene.shape[-1])
Wls, *_ = np.linalg.lstsq(Zaug, yf, rcond=None)
pred = Zaug @ Wls
r2 = 1 - ((yf - pred) ** 2).sum() / ((yf - yf.mean(0)) ** 2).sum()
print(f"latent encodes scene state (3 positions + triangle spin): linear R^2 = {r2:.3f}")

mu, sd = z_learned.mean((0, 1), keepdim=True), z_learned.std((0, 1), keepdim=True) + 1e-6
z_std = (z_learned - mu) / sd
gamma, Kf = fit_regimes_posthoc(z_std, K0=8)
occ = gamma.reshape(-1, Kf).sum(0); occ = (occ / occ.sum()).detach().numpy()
n_used = int((occ > 0.02).sum())
g_al, acc = relabel_to_true(gamma, S, Ktrue)
print(f"\nlearned-latent regime clustering (sticky-HDP + birth/merge/delete): "
      f"K 8->{Kf}, {n_used} used, occ={np.round(np.sort(occ)[::-1], 3)}, "
      f"mode segmentation acc={acc:.3f}")
plot_latent_clustering(z_std, g_al, os.path.join(_OUT, 'fig5_latent_clustering.png'),
                       title=f"SHS-RSSM multi-object latent: sticky-HDP regime clustering "
                             f"(recurrent-sticky; K 8->{Kf}, mode acc {acc * 100:.0f}%)", true_labels=S)
plot_true_vs_inferred_dynamics(S, gamma, pos, ANGS, SPEED_MULT, P,
                               os.path.join(_OUT, 'fig7_true_vs_inferred_dynamics.png'))

with torch.no_grad():
    ep, ctx = 0, T_APPEAR + 6
    x = frames[ep:ep + 1]
    emb = encode_all(x)
    post_e, _ = rssm.observe(emb, action[ep:ep + 1], is_first[ep:ep + 1])
    g_e = regime_responsibilities(rssm, post_e, is_first[ep:ep + 1])
    post_e["regime_resp"] = g_e
    recon = dec(rssm.get_feat(post_e).reshape(T, FEAT)).reshape(T, CH, H, W)
    state_ctx = {k: v[:, ctx - 1] for k, v in post_e.items()}
    prior_im = rssm.imagine_with_action(torch.zeros(1, T - ctx, 1), state_ctx)
    im = dec(rssm.get_feat(prior_im).reshape(T - ctx, FEAT)).reshape(T - ctx, CH, H, W)
    imagined = torch.cat([recon[:ctx], im], 0)
    rec_err = ((recon - x[0]) ** 2).mean().item()
    print(f"reconstruction MSE/pixel = {rec_err:.4f}")
    plot_reconstructions(x[0], recon, os.path.join(_OUT, 'fig6_reconstruction.png'),
                         imagined_frames=imagined, n=10, context=ctx,
                         title="SHS-RSSM multi-object: true / reconstructed / open-loop imagined")
print("\nsaved fig5_latent_clustering.png, fig6_reconstruction.png")
