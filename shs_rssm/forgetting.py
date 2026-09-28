"""Retention of streaming conjugate statistics under a change-point normalized power prior.
The evidence of every factor is an exact function of the retention r; r and the drift indicator
are integrated out to quadrature tolerance and the drift rate has an exact posterior in usage time."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64
EPS = torch.finfo(torch.float64).eps
LOG2PI = math.log(2.0 * math.pi)
_ZOOM_POINTS = 64
_BREAKS = 16
_SCALES = tuple(2.0 ** m for m in range(-2, 11))
_TOL = 1e-10
_VETO = 1e-6
_ROUNDS = 200
_MAX_PANELS = 20000
_CHUNK = 4096
_GRADED = 30
_ZOOMS = 4
_XK = (0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
       0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
       0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
       0.207784955007898467600689403773245, 0.0)
_WK = (0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
       0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
       0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
       0.204432940075298892414161999234649, 0.209482141084727828012999174891714)
_WG = (0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
       0.381830050505118944950369775488975, 0.417959183673469387755102040816327)


_RULES = {}


def kronrod(device=None):
    """Nodes on [-1, 1] with 15-point Kronrod weights and the embedded 7-point Gauss weights."""
    key = str(device)
    if key not in _RULES:
        _RULES[key] = _kronrod(device)
    return _RULES[key]


def _kronrod(device):
    x, wk, wg = [], [], []
    for i in range(15):
        j = i if i < 8 else 14 - i
        x.append(-_XK[j] if i < 7 else _XK[j])
        wk.append(_WK[j])
        wg.append(_WG[j // 2] if j % 2 == 1 or j == 7 else 0.0)
    return tuple(torch.tensor(v, dtype=F64, device=device) for v in (x, wk, wg))


def stream_weights(gain, tau):
    """Memory and batch weights (w, v) of the discounted stream D <- r (1 - gain) D + (gain / tau) s whose
    fractional posterior eta0 + tau (eta_D - eta0) is the installed EMA factor S <- r (1 - gain) S + gain s."""
    gain = gain.to(F64).reshape(-1)
    return (1.0 - gain) / float(tau), gain / float(tau)


def memory_stats(reg):
    if getattr(reg, "_residual_only", False):
        out = dict(N=reg.N, Sgg=reg.Sgg, Szg=reg.Szg, Szz=reg.Szz_resid)
        for name in ("Szf", "Sfr", "Sfh", "Sff"):
            if int(getattr(reg, "q_rank", 0)) > 0 and hasattr(reg, name):
                out[name] = getattr(reg, name)
        return out
    return {name: getattr(reg, name) for name in reg.raw_stat_names()}


def regression_stats(reg, stats):
    """(N, Sgg, Szg, Szz) of the Normal-Gamma block given q(C) and q(U), as used by the M-step."""
    stats = {k: v.detach() for k, v in stats.items()}
    if "Sgg" in stats:
        N, Sgg, Szg, Szz = (stats[k].to(F64) for k in ("N", "Sgg", "Szg", "Szz"))
    else:
        C, Vc = reg.Cmean.detach().to(F64), reg.Ccov.detach().to(F64)
        Shh, Szh, Srh = (stats[k].to(F64) for k in ("Shh", "Szh", "Srh"))
        N, Sgg = stats["N"].to(F64), stats["Srr"].to(F64)
        Szg = stats["Szr"].to(F64) - torch.einsum("ih,krh->kir", C, Srh)
        Szz = (stats["Szz"].to(F64) - 2.0 * torch.einsum("ih,kih->ki", C, Szh)
               + torch.einsum("ih,khg,ig->ki", C, Shh, C) + torch.einsum("ihg,khg->ki", Vc, Shh))
    if int(getattr(reg, "q_rank", 0)) > 0 and "Sff" in stats:
        C = reg.Cmean.detach().to(F64)
        U = reg.Umean.detach().to(F64)
        EuuT = reg.Ucov.detach().to(F64) + torch.einsum("kif,kig->kifg", U, U)
        Syf = stats["Szf"].to(F64) - torch.einsum("ih,kfh->kif", C, stats["Sfh"].to(F64))
        Szg = Szg - torch.einsum("kif,kfr->kir", U, stats["Sfr"].to(F64))
        Szz = Szz - 2.0 * (U * Syf).sum(-1) + torch.einsum("kifg,kfg->ki", EuuT, stats["Sff"].to(F64))
    return N, Sgg, Szg, Szz


def _pencils(D, P_extra, W):
    """For B0 = diag(D) and B1 = diag(D) + P_extra: T, mu, log|B| with (B + r W)^-1 = T diag(1 / (1 + r mu)) T^T.
    Both pencils share one batched factorisation; a failed Cholesky gives a NaN log-determinant."""
    K, G = W.shape[0], W.shape[-1]
    diag = torch.diag_embed(D.expand(K, G))
    base = torch.cat([diag, diag + 0.5 * (P_extra + P_extra.transpose(-1, -2))], 0)
    Lc, info = torch.linalg.cholesky_ex(base)
    eye = torch.eye(G, dtype=F64, device=W.device).expand(2 * K, G, G)
    Linv = torch.linalg.solve_triangular(Lc, eye, upper=False)
    Ws = Linv @ torch.cat([W, W], 0) @ Linv.transpose(-1, -2)
    mu, Q = torch.linalg.eigh(0.5 * (Ws + Ws.transpose(-1, -2)))
    logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    logdet = torch.where(info == 0, logdet, torch.full_like(logdet, math.nan))
    T, mu = Linv.transpose(-1, -2) @ Q, mu.clamp_min(0.0)
    return (T[:K], mu[:K], logdet[:K]), (T[K:], mu[K:], logdet[K:])


def lgamma_diff(x, d):
    """log Gamma(x + d) - log Gamma(x) without cancellation, by the Stirling series for large x."""
    big = x > 100.0
    xs = torch.where(big, x, torch.ones_like(x))
    y = xs + d
    def series(z):
        s = 1.0 / z
        t = s * s
        return s * (1.0 / 12.0 - t * (1.0 / 360.0 - t * (1.0 / 1260.0 - t / 1680.0)))
    stirling = (xs - 0.5) * torch.log1p(d / xs) + d * torch.log(y) - d + series(y) - series(xs)
    return torch.where(big, stirling, torch.lgamma(x + d) - torch.lgamma(x))


class _Pencil:
    """x(r)^T (B + r W)^-1 x(r) = r E + A0 - S(r) with x(r) = x0 + r x1 and S(r) = sum_j r g_j^2 / (1 + r mu_j).
    E = x1^T W^+ x1 is the same in every basis, so differences of two pencils never form it."""

    def __init__(self, pencil, W, x0, x1):
        T, self.mu, self.logdet0 = pencil
        a, b = x0 @ T, x1 @ T
        rank = self.mu > W.shape[-1] * EPS * self.mu.max(-1, keepdim=True).values.clamp_min(1e-300)
        root = self.mu.sqrt().unsqueeze(-2)
        inv = torch.where(rank, 1.0 / root.squeeze(-2).clamp_min(1e-300), torch.zeros_like(self.mu)).unsqueeze(-2)
        g = torch.where(rank.unsqueeze(-2), b * inv - a * root, -a * root)
        self.E = (b * b * inv * inv).sum(-1)
        self.A0 = (a * a).sum(-1)
        self.g2 = (g * g).transpose(-1, -2).contiguous()

    def sum(self, r):
        rr = r.unsqueeze(-1)
        return rr * torch.matmul((1.0 / (1.0 + rr * self.mu)).transpose(0, 1), self.g2).transpose(0, 1)

    def logdet(self, r):
        return self.logdet0 + torch.log1p(r.unsqueeze(-1) * self.mu).sum(-1)


class EmissionEvidence:
    """Normal-Gamma evidence of v_k * S_k under the prior plus r * w_k * memory_k, for every regime k."""

    def __init__(self, reg, memory, batch, w, v):
        N0, S0, Z0, V0 = regression_stats(reg, memory)
        dev = S0.device
        v = v.to(F64).to(dev)
        Nb, Sb, Zb, Vb = (t * v.reshape(-1, *([1] * (t.dim() - 1))) for t in regression_stats(reg, batch))
        self.K, self.device = int(S0.shape[0]), dev
        w = w.to(F64).to(dev)
        lam0 = reg.lam0_diag.detach().to(F64).to(dev)
        M0 = reg.M0.detach().to(F64).to(dev)
        R0 = (M0 * lam0).unsqueeze(0)
        m0 = (M0 * M0 * lam0).sum(-1)
        self.a0 = float(reg.a0)
        self.b0 = reg._b0_rate().detach().to(F64).to(dev)
        W = w[:, None, None] * S0
        mZ = w[:, None, None] * Z0
        self.mN, self.Nb = w * N0, Nb
        self.L = Z0.shape[1]
        D = lam0 + float(reg.jitter)
        pp, pn = _pencils(D, Sb, W)
        self.prior = _Pencil(pp, W, R0.expand_as(mZ), mZ)
        self.post = _Pencil(pn, W, R0 + Zb, mZ)
        self.c0 = (m0 - self.prior.A0).clamp_min(0.0)
        self.rss = (w[:, None] * V0 - self.prior.E).clamp_min(0.0)
        self.d0 = Vb + self.prior.A0 - self.post.A0

    def parts(self, r, size=True):
        """L(r) and the absolute size of the terms it is summed from, for its rounding error."""
        sp, sn = self.prior.sum(r), self.post.sum(r)
        ldp, ldn = self.prior.logdet(r), self.post.logdet(r)
        rv = r.unsqueeze(-1)
        ap = self.a0 + 0.5 * r * self.mN
        bp = self.b0 + 0.5 * (self.c0 + rv * self.rss + sp)
        delta = 0.5 * (self.d0 + sn - sp)
        bn = bp + delta
        shift = -ap.unsqueeze(-1) * torch.log1p(delta / bp)
        tail = -0.5 * self.Nb.unsqueeze(-1) * torch.log(bn)
        lg = lgamma_diff(ap, 0.5 * self.Nb)
        value = (0.5 * self.L * (ldp - ldn) + (shift + tail).sum(-1) + self.L * lg
                 - 0.5 * self.Nb * self.L * LOG2PI)
        if not size:
            return value, None
        return value, (0.5 * self.L * (ldp.abs() + ldn.abs()) + (shift.abs() + tail.abs()).sum(-1)
                       + (ap.unsqueeze(-1) * (self.d0.abs() + sn + sp) / bp).sum(-1) + self.L * lg.abs())

    def __call__(self, r):
        return self.parts(r, size=False)[0]


class TransitionEvidence:
    """Dirichlet-multinomial evidence of v_k * c_k under the prior row plus r * w_k * memory row, per row k."""

    def __init__(self, prior_rows, memory, batch, w, v):
        self.a0 = prior_rows.detach().to(F64)
        dev, J = self.a0.device, self.a0.shape[-1]
        pad = lambda C: F.pad(C.detach().to(F64).to(dev), (0, J - C.shape[-1]))
        self.K, self.device = int(self.a0.shape[0]), dev
        self.m = w.to(F64).to(dev)[:, None] * pad(memory)
        self.c = v.to(F64).to(dev)[:, None] * pad(batch)
        self.n = self.c.sum(-1)

    def parts(self, r):
        a = self.a0 + r.unsqueeze(-1) * self.m
        total = lgamma_diff(a.sum(-1), self.n)
        rows = lgamma_diff(a, self.c)
        return rows.sum(-1) - total, total.abs() + rows.abs().sum(-1)

    def __call__(self, r):
        return self.parts(r)[0]


class GateEvidence:
    """Evidence of v_k times the Polya-Gamma augmented gate statistics under the Gaussian prior plus r * w_k * memory."""

    def __init__(self, sigma0_diag, m0, memory_A, memory_h, batch_A, batch_h, w, v):
        dev = memory_A.device
        self.K, self.device = int(memory_A.shape[0]), dev
        w = w.to(F64).to(dev)
        v = v.to(F64).to(dev)
        batch_A = batch_A.detach().to(F64).to(dev) * v[:, None, None]
        batch_h = batch_h.detach().to(F64).to(dev) * v[:, None]
        D = 1.0 / sigma0_diag.detach().to(F64).to(dev)
        eta0 = (D * m0.detach().to(F64).to(dev)).expand(self.K, -1).unsqueeze(1)
        W = w[:, None, None] * memory_A.detach().to(F64)
        mh = (w[:, None] * memory_h.detach().to(F64)).unsqueeze(1)
        pp, pn = _pencils(D, batch_A, W)
        self.prior = _Pencil(pp, W, eta0, mh)
        self.post = _Pencil(pn, W, eta0 + batch_h.unsqueeze(1), mh)
        self.q0 = (self.post.A0 - self.prior.A0)[:, 0]

    def parts(self, r):
        sp, sn = self.prior.sum(r)[..., 0], self.post.sum(r)[..., 0]
        ldp, ldn = self.prior.logdet(r), self.post.logdet(r)
        value = 0.5 * (self.q0 - sn + sp) - 0.5 * (ldn - ldp)
        return value, 0.5 * (self.q0.abs() + sn + sp + ldn.abs() + ldp.abs())

    def __call__(self, r):
        return self.parts(r)[0]


def _evaluate(fn, r):
    return torch.cat([fn(r[i:i + _CHUNK]) for i in range(0, r.shape[0], _CHUNK)], 0)


def _panel_logs(fn, lo, hi, extra):
    """Kronrod-15 and Gauss-7 log integrals of exp(fn) and of exp(fn + g_e) on every panel, g = extra(nodes)."""
    x, wk, wg = kronrod(lo.device)
    half = 0.5 * (hi - lo)
    nodes = (0.5 * (hi + lo)).unsqueeze(1) + half.unsqueeze(1) * x.view(1, -1, 1)
    ell = _evaluate(fn, nodes.reshape(-1, lo.shape[-1])).reshape(nodes.shape)
    g = torch.cat([torch.zeros_like(ell).unsqueeze(-1), extra(nodes)], -1)
    out = []
    for wt in (wk, wg):
        lw = (torch.log(half).unsqueeze(1) + torch.log(wt).view(1, -1, 1) + ell).unsqueeze(-1)
        out.append(torch.logsumexp(lw + g, 1))
    return out


def adaptive_integral(fn, breaks, extra, tol=None):
    """Globally adaptive Gauss-Kronrod for log int exp(fn) and the ratios int exp(fn + g) / int exp(fn).
    Every panel whose embedded Gauss error exceeds its share of the tolerance is bisected."""
    lo, hi = breaks[:-1], breaks[1:]
    tol = torch.full_like(breaks[0], _TOL) if tol is None else tol.clamp_min(_TOL)
    lk, lg = _panel_logs(fn, lo, hi, extra)
    for _ in range(_ROUNDS):
        logz = torch.logsumexp(lk[..., 0], 0)
        e = (torch.exp(lk - logz[None, :, None]) - torch.exp(lg - logz[None, :, None])).abs().sum(-1)
        err = e.sum(0)
        bad = err > tol
        live = (hi > lo).sum(0).clamp_min(1)
        split = bad[None] & ((e > tol[None] / (2.0 * live[None])) | (e >= e.max(0, keepdim=True).values))
        count = split.sum(0)
        S = int(count.max())
        if S == 0 or lo.shape[0] > _MAX_PANELS:
            break
        order = torch.argsort((~split).to(torch.int8), dim=0, stable=True)[:S]
        valid = torch.arange(S, device=lo.device)[:, None] < count[None]
        l0, h0 = lo.gather(0, order), hi.gather(0, order)
        l0, h0 = torch.where(valid, l0, hi[-1:].expand_as(l0)), torch.where(valid, h0, hi[-1:].expand_as(h0))
        m0 = 0.5 * (l0 + h0)
        new_lo, new_hi = torch.cat([l0, m0], 0), torch.cat([m0, h0], 0)
        nk, ng = _panel_logs(fn, new_lo, new_hi, extra)
        dead = torch.full_like(lk[:1], -math.inf)
        lk = torch.where(split[..., None], dead, lk)
        lg = torch.where(split[..., None], dead, lg)
        lo = torch.where(split, hi, lo)
        lo, hi = torch.cat([lo, new_lo], 0), torch.cat([hi, new_hi], 0)
        lk, lg = torch.cat([lk, nk], 0), torch.cat([lg, ng], 0)
    logz = torch.logsumexp(lk[..., 0], 0)
    return logz, torch.exp(torch.logsumexp(lk[..., 1:], 0) - logz.unsqueeze(-1)), err


def _zoom(fn, K, dev):
    """Bracket the maximiser of fn on [0, 1] by nested Chebyshev-Lobatto grids; returns breakpoints from the
    first and last grids, the maximiser and a length scale from the local curvature or end-point slope."""
    j = torch.arange(_ZOOM_POINTS + 1, dtype=F64, device=dev)
    unit = (0.5 * (1.0 - torch.cos(math.pi * j / _ZOOM_POINTS))).unsqueeze(1)
    lo, hi = torch.zeros(K, dtype=F64, device=dev), torch.ones(K, dtype=F64, device=dev)
    cols = torch.arange(K, device=dev)
    first = None
    for _ in range(_ZOOMS):
        x = lo + (hi - lo) * unit
        ell = fn(x)
        idx = ell.argmax(0)
        first = x if first is None else first
        lo, hi = x[(idx - 1).clamp_min(0), cols], x[(idx + 1).clamp_max(_ZOOM_POINTS), cols]
    i0, i1 = (idx - 1).clamp_min(0), (idx + 1).clamp_max(_ZOOM_POINTS)
    x0, xm, x1 = x[i0, cols], x[idx, cols], x[i1, cols]
    f0, fm, f1 = ell[i0, cols], ell[idx, cols], ell[i1, cols]
    s0 = (fm - f0) / (xm - x0).clamp_min(1e-300)
    s1 = (f1 - fm) / (x1 - xm).clamp_min(1e-300)
    curv = 2.0 * (s1 - s0) / (x1 - x0).clamp_min(1e-300)
    interior = (idx > 0) & (idx < _ZOOM_POINTS) & (curv < 0)
    slope = torch.where(idx == 0, s1, s0).abs()
    scale = torch.where(interior, (-curv).clamp_min(1e-300).rsqrt(), 1.0 / slope.clamp_min(1e-300))
    step = _ZOOM_POINTS // _BREAKS
    return torch.cat([first[::step], x[::step]], 0), xm, scale.clamp(1e-300, 1.0)


def slab_integrals(evidence):
    """log I0 = log int_0^1 exp(L(r) - L(1)) dr and E[r | drift, S] for every regime, with the error."""
    K, dev = evidence.K, evidence.device
    L1, S1 = (t[0] for t in evidence.parts(torch.ones(1, K, dtype=F64, device=dev)))
    fn = lambda r: evidence(r) - L1
    grids, mode, scale = _zoom(fn, K, dev)
    floor = EPS * (evidence.parts(grids)[1].max(0).values + S1)
    extra = [mode] + [mode + s * c * scale for c in _SCALES for s in (-1.0, 1.0)]
    graded = (4.0 ** -torch.arange(1, _GRADED + 1, dtype=F64, device=dev)).unsqueeze(1).expand(-1, K)
    breaks, _ = torch.sort(torch.cat([grids, graded, torch.stack(extra, 0).clamp(0.0, 1.0)], 0), 0)
    log0, mean, err = adaptive_integral(fn, breaks, lambda r: torch.log(r).unsqueeze(-1), tol=floor)
    return log0, mean[:, 0].clamp(0.0, 1.0), torch.maximum(err, floor)


class DriftPosterior:
    """Exact posterior of the drift probability per unit of usage under the Jeffreys prior Beta(1/2, 1/2).
    Log density in logit coordinates on a grid; the log likelihood is analytic in the strip |Im x| < pi."""

    def __init__(self, half_width=80.0, spacing=1.0 / 16.0, order=12, device=None):
        n = int(round(2.0 * half_width / spacing)) + 1
        self.half_width, self.spacing, self.order = float(half_width), float(spacing), int(order)
        self.x = torch.linspace(-half_width, half_width, n, dtype=F64, device=device)
        self._bary = torch.tensor([(-1.0) ** k * math.comb(self.order - 1, k) for k in range(self.order)],
                                  dtype=F64, device=device)
        logp = 0.5 * F.logsigmoid(self.x) + 0.5 * F.logsigmoid(-self.x)
        self.logp = logp - logp.max()
        zero = torch.zeros((), dtype=F64, device=device)
        self.events, self.exposure, self.roundoff, self.err = zero.clone(), zero.clone(), zero.clone(), zero.clone()

    def to(self, device):
        for name in ("x", "_bary", "logp", "events", "exposure", "roundoff", "err"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _interp(self, y):
        n, p = self.x.shape[0], self.order
        t = (y - self.x[0]) / self.spacing
        base = (torch.floor(t).long() - (p // 2 - 1)).clamp(0, n - p)
        nodes = torch.arange(p, device=y.device)
        s = t - base.to(F64)
        vals = self.logp[base.unsqueeze(-1) + nodes]
        ref = vals[:, p // 2 - 1:p // 2]
        d = s.unsqueeze(-1) - nodes.to(F64)
        hit = d.abs() < 1e-13
        c = self._bary / torch.where(hit, torch.ones_like(d), d)
        out = ref[:, 0] + (c * (vals - ref)).sum(-1) / c.sum(-1)
        return torch.where(hit.any(-1), (vals * hit).sum(-1), out)

    @staticmethod
    def _loglik(y, b):
        y = y.unsqueeze(-1)
        return torch.logaddexp(F.logsigmoid(-y), F.logsigmoid(y) + b)

    def mean(self):
        empty = torch.zeros(0, dtype=F64, device=self.x.device)
        return self._moments(empty, empty)[1]

    def _moments(self, b, lead):
        grid = self.logp + self._loglik(self.x, b).sum(-1)
        n = self.x.shape[0]
        keep = grid >= grid.max() - 80.0
        first = torch.argmax(keep.to(torch.int8))
        last = n - 1 - torch.argmax(keep.flip(0).to(torch.int8))
        lo, hi = self.x[(first - 2).clamp_min(0)], self.x[(last + 2).clamp_max(n - 1)]
        breaks = (lo + (hi - lo) * torch.linspace(0.0, 1.0, 65, dtype=F64, device=b.device)).unsqueeze(1)
        fn = lambda y: (self._interp(y.reshape(-1)) + self._loglik(y.reshape(-1), b).sum(-1)).reshape(-1, 1)

        def extra(y):
            ls = F.logsigmoid(y).unsqueeze(-1)
            return torch.cat([ls, lead + ls - self._loglik(y, b)], -1)
        floor = self.roundoff + self._rounding(b)
        logz, moments, err = adaptive_integral(fn, breaks, extra, tol=floor.reshape(1))
        tail = 2.0 * torch.exp(torch.logsumexp(grid[[0, -1]], 0) - logz[0])
        return grid, moments[0, 0], moments[0, 1:], torch.maximum(err[0] + tail, floor)

    def _rounding(self, b):
        return EPS * (b.abs() + 2.0 * self.half_width).sum()

    def step(self, log0, usage, learn=True):
        """P(drift_k | all data) for every regime and the posterior mean drift rate after the update."""
        if self.x.device != log0.device:
            self.to(log0.device)
        log0 = log0.detach().to(F64).reshape(-1)
        u = usage.detach().to(device=log0.device, dtype=F64).reshape(-1).clamp(0.0, 1.0)
        pos = u > 0
        b = torch.where(pos, torch.logaddexp(torch.log1p(-u), torch.log(u.clamp_min(1e-300)) + log0),
                        torch.zeros_like(u))
        lead = torch.where(pos, torch.log(u.clamp_min(1e-300)) + log0, torch.full_like(u, -math.inf))
        grid, rate, P, err = self._moments(b, lead)
        P = torch.where(pos, P.clamp(0.0, 1.0), torch.zeros_like(P))
        if learn:
            self.logp = grid - grid.max()
            self.roundoff = self.roundoff + self._rounding(b)
            self.events = self.events + P.sum()
            self.exposure = self.exposure + u.sum()
        self.err = err
        return P, rate, err

    def state_dict(self):
        cpu = lambda t: t.detach().to("cpu")
        return dict(logp=cpu(self.logp), spacing=self.spacing, order=self.order, half_width=self.half_width,
                    events=float(self.events), exposure=float(self.exposure), roundoff=float(self.roundoff))

    def load_state_dict(self, state):
        self.__init__(state.get("half_width", 80.0), state.get("spacing", 1.0 / 16.0), state.get("order", 12))
        self.logp = torch.as_tensor(state["logp"], dtype=F64).clone()
        for name in ("events", "exposure", "roundoff"):
            setattr(self, name, torch.tensor(float(state.get(name, 0.0)), dtype=F64))


def retention(evidence, usage, drift, learn=True):
    """E[r_k | data] under the change-point power prior, with P(drift_k | data) and diagnostics.
    A regime whose evidence is non-finite or not integrated to _VETO is treated as unused: pi is not updated, r = 1."""
    log0, slab_mean, err_r = slab_integrals(evidence)
    ok = torch.isfinite(log0) & torch.isfinite(slab_mean) & (err_r <= _VETO)
    u = torch.where(ok, usage.detach().to(device=log0.device, dtype=F64).reshape(-1), torch.zeros_like(log0))
    log0 = torch.where(ok, log0, torch.zeros_like(log0))
    slab_mean = torch.where(ok, slab_mean, torch.ones_like(slab_mean))
    P, rate, err_p = drift.step(log0, u, learn=learn)
    r = (1.0 - P) + P * slab_mean
    err = torch.maximum(torch.where(ok, err_r, torch.zeros_like(err_r)).max(), err_p)
    return r, dict(P=P, slab=slab_mean, log_bayes=log0, rate=rate, err=err, failed=(~ok).sum())
