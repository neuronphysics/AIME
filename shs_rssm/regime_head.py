from __future__ import annotations

import copy
import threading
import math
import torch

LOG2PI = math.log(2.0 * math.pi)
import torch.nn as nn

from .regimes import DiagARRegimes
from .regimes_shared import SharedCarryRegimes
from .sticky_hdp import StickyHDP
from .forward_backward import forward_backward, start_counts_from
from .mixture_prior import mixture_weights, _diag_gauss_kl
from .recurrent_stick import RecurrentStickiness
from .structured_elbo import diag_gauss_entropy
from .continuous_smoother import chain_potentials, build_blocks, smooth

class RegimeHead(nn.Module):
    def __init__(
        self,
        stoch: int,
        deter: int,
        K: int = 16,
        proj_dim: int | None = 64,
        action_dim: int = 0,
        a0: float = 3.0, b0: float = 2.0, v0_scale: float = 1.0, ard: bool = True,
        learn_b0: bool = False, b0_strength: float = 2.0,
        identity_init: bool = True,
        q_rank: int = 0,
        shared_carry: bool = False,
        gamma: float = 5.0, alpha: float = 1.0, kappa: float = 50.0, start_alpha: float = 1.0,
        recurrent: bool = False, prior_persist: float = 0.9, pg_iters: int = 4,
        rstick_dim: int | None = 8, rstick_stopgrad: bool = True,
        rstick_weight_var: float = 1.0,
        rstick_bias_var: float = 4.0,   # prior variance of the gate's logit bias; small = strongly persistent unless data insists
        rstick_use_action: bool = False,     # feed a_{t-1} to the gate
        ema_tau: float = 0.02, hdp_iters: int = 2,
        online_mode: str = "ema",
        stream_local_iters: int = 1,
        stream_discount: float = 1.0,
        expected_batches: int | None = None,
        expected_ids: set | None = None,
        strict_stream: bool = False,
        hdp_every: int = 1,
        pg_every: int = 1,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self._ctor_kwargs = dict(
            stoch=stoch, deter=deter, K=K, proj_dim=proj_dim,
            action_dim=action_dim, a0=a0, b0=b0,
            learn_b0=learn_b0, b0_strength=b0_strength,
            v0_scale=v0_scale, ard=ard, identity_init=identity_init, q_rank=q_rank,
            shared_carry=shared_carry, gamma=gamma, alpha=alpha, kappa=kappa,
            start_alpha=start_alpha, recurrent=recurrent, prior_persist=prior_persist,
            pg_iters=pg_iters, rstick_dim=rstick_dim, rstick_stopgrad=rstick_stopgrad,
            ema_tau=ema_tau, hdp_iters=hdp_iters, online_mode=online_mode,
            expected_batches=expected_batches,
            expected_ids=(set(expected_ids) if expected_ids is not None else None),
            strict_stream=strict_stream, hdp_every=hdp_every, pg_every=pg_every,
            rstick_weight_var=rstick_weight_var,
            rstick_bias_var=rstick_bias_var,           
            rstick_use_action=rstick_use_action,        
            stream_local_iters=stream_local_iters,
            stream_discount=stream_discount,
            dtype=dtype, device=device)
        self.L = stoch
        self.H = deter
        self.K = K
        self.Hp = deter if (proj_dim is None or proj_dim >= deter) else proj_dim
        self.use_proj = self.Hp < deter
        self.action_dim = int(action_dim)
        self.G = self.L + self.Hp + self.action_dim + 1
        self.ema_tau = ema_tau
        self.online_mode = online_mode
        if online_mode == "ema":
            self.stat_store = None
        else:
            from .online_vb import SuffStatStore
            self.stat_store = SuffStatStore(
                mode=("legacy_ema" if online_mode == "ema" else online_mode),
                ema_tau=ema_tau, expected_batches=expected_batches,
                expected_ids=expected_ids, strict_stream=strict_stream,
                discount=stream_discount)
        self.register_buffer("repr_version", torch.zeros((), dtype=torch.long))
        self._register_load_state_dict_pre_hook(self._structural_pre_load_hook)
        self.register_load_state_dict_post_hook(self._post_load_refresh)
        self.hdp_iters = hdp_iters
        self.recurrent = recurrent
        self.rstick_stopgrad = bool(rstick_stopgrad)
        self.rstick_dim = int(min(deter, rstick_dim)) if rstick_dim is not None else self.Hp
        self.rstick_action_dim = int(action_dim) if rstick_use_action else 0

        if self.use_proj:
            self.P = nn.Linear(deter, self.Hp, bias=False,
                               dtype=dtype, device=device)
        else:
            self.P = None

        if self.rstick_dim < deter:
            self.P_stick = nn.Linear(deter, self.rstick_dim, bias=False)
            nn.init.orthogonal_(self.P_stick.weight, gain=0.5)
        else:
            self.P_stick = None

        self.shared_carry = bool(shared_carry)
        if self.shared_carry:
            self.regimes = SharedCarryRegimes(
                K=K, L=self.L, G=self.G, action_dim=self.action_dim,
                a0=a0, b0=b0, learn_b0=learn_b0, b0_c0=b0_strength,
                v0_scale=v0_scale, ard=ard,
                identity_init=identity_init, q_rank=q_rank, dtype=dtype, device=device,
            )
        else:
            self.regimes = DiagARRegimes(
                K=K, L=self.L, G=self.G, action_dim=self.action_dim,
                a0=a0, b0=b0, learn_b0=learn_b0, b0_c0=b0_strength,
                v0_scale=v0_scale, ard=ard,
                identity_init=identity_init, q_rank=q_rank, dtype=dtype, device=device,
            )
        base_kappa = 0.0 if recurrent else kappa
        self.hdp = StickyHDP(K=K, gamma=gamma, alpha=alpha, kappa=base_kappa,
                             start_alpha=start_alpha, dtype=torch.float64, device=device)

        self.rstick = RecurrentStickiness(
            K=K, feat_dim=self.rstick_dim + self.rstick_action_dim, prior_persist=prior_persist, pg_iters=pg_iters,
            weight_prior_var=rstick_weight_var,
            bias_prior_var=rstick_bias_var,
            dtype=torch.float64, device=device,
        )
        self._estep = None
        self._struct_cache = None
        self.hdp_every = int(hdp_every)
        self.stream_local_iters = int(stream_local_iters)
        self.pg_every = int(pg_every)
        self._gstep_count = 0
        self._episode_cursor = 0
        self._repr_frozen = False
        self.chunk_boundary_mask = True
        self.active_mask = None
        self._shs_online_pairwise = False
        self._async_version = 0
        self._struct_gen = 0
        self._commit_lock = threading.RLock()

        self.z0 = nn.Parameter(torch.zeros(self.L, dtype=dtype, device=device))
        self.z0_logvar = nn.Parameter(torch.full_like(self.z0, -9.210340371976184))
        self._z0_prior_var = 1.0

        self.register_buffer("ema_trans_counts", torch.zeros(K, K, dtype=torch.float64, device=device))
        self.register_buffer("ema_start_counts", torch.zeros(K, dtype=torch.float64, device=device))
        self._counts_initialised = False

    def _proj(self, deter):
        if not self.use_proj:
            return deter
        return self.P(deter.to(self.P.weight.dtype))

    def build_g(self, prev_stoch, deter, action=None):
        htil = self._proj(deter)
        ones = prev_stoch[..., :1] * 0.0 + 1.0
        parts = [prev_stoch, htil]
        if self.action_dim > 0:
            if action is None:
                action = prev_stoch.new_zeros(*prev_stoch.shape[:-1], self.action_dim)
            parts.append(action.to(prev_stoch.dtype))
        parts.append(ones)
        return torch.cat(parts, dim=-1)

    def _shift_action(self, action, is_first):
        if action is None or self.action_dim == 0:
            return None
        if is_first is None:
            return action
        isf = is_first.reshape(*action.shape[:-1], 1).to(action.dtype)
        return action * (1.0 - isf)

    def build_stick_phi(self, deter, action=None, is_first=None):
        h = deter.detach() if self.rstick_stopgrad else deter
        htil = self.P_stick(h) if self.P_stick is not None else h
        htil = torch.tanh(htil[..., :self.rstick_dim])
        ones = htil[..., :1] * 0.0 + 1.0
        parts = [htil]
        if self.rstick_action_dim > 0:
            if action is None:
                action = htil.new_zeros(*htil.shape[:-1], self.rstick_action_dim)
            elif is_first is not None:
                # match build_g: no previous action at an episode start
                isf = is_first.reshape(*action.shape[:-1], 1).to(action.dtype)
                action = action * (1.0 - isf)
            parts.append(action.to(htil.dtype))
        parts.append(ones)
        return torch.cat(parts, dim=-1)

    def _prev_stoch(self, stoch, is_first):
        B, T, L = stoch.shape
        z0 = self.z0.view(1, 1, L).expand(B, 1, L)
        prev = torch.cat([z0, stoch[:, :-1]], dim=1) if T > 1 else z0
        if is_first is not None:
            isf = is_first.reshape(B, T, 1).to(stoch.dtype)
            prev = torch.where(isf > 0.5, self.z0.view(1, 1, L), prev)
        return prev

    def _g_var_from_z_var(self, z_var, g, is_first=None):
        if z_var is None:
            return None
        prev_var = self._shift_var(z_var, is_first)
        zeros_tail = g[..., self.L:] * 0.0
        return torch.cat([prev_var, zeros_tail], dim=-1)

    def _transition_logpotentials(self, deter, dtype=None, device=None, action=None, is_first=None):
        dtype = dtype if dtype is not None else deter.dtype
        device = device if device is not None else deter.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            phi = self.build_stick_phi(deter.to(dtype), action, is_first=is_first)
            log_trans, aux = self.rstick.bound_log_trans(base_elogpi, phi[:, 1:])
            aux = dict(aux, phi_steps=phi[:, 1:])
            log_init, log_trans = self._mask_logpotentials(log_init, log_trans)
            return log_init, log_trans, aux
        log_init, base_elogpi = self._mask_logpotentials(log_init, base_elogpi)
        return log_init, base_elogpi, None

    def _transition_potentials_ondemand(self, deter, dtype=None, device=None, action=None, is_first=None):
        dtype = dtype if dtype is not None else deter.dtype
        device = device if device is not None else deter.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        base = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        phi = self.build_stick_phi(deter.to(dtype), action, is_first=is_first)
        aux = dict(self.rstick.bound_aux_only(base, phi[:, 1:]), phi_steps=phi[:, 1:])
        if self.active_mask is not None:
            m = self.active_mask.to(log_init.device)
            log_init = torch.where(m, log_init, log_init.new_full((), -1e30))
        am = self.active_mask
        rstick = self.rstick

        def trans_fn(t):
            Tt = rstick.trans_slice_from_aux(aux, t)
            if am is not None:
                mm = am.to(Tt.device); K = int(mm.shape[0])
                Tt = torch.where((mm.view(1, K, 1) & mm.view(1, 1, K)), Tt,
                                 Tt.new_full((), -1e30))
            return Tt
        return log_init, aux, trans_fn

    def _vb_evidence(self, z_mean, deter, is_first=None, z_var=None, action=None,
                     z_cov=None, zg_xcov=None, prev_stoch=None):
        _cond_prev = prev_stoch is not None
        prev = self._prev_stoch(prev_stoch if _cond_prev else z_mean, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        zvar = None if z_var is None else z_var
        if zvar is None and z_cov is not None:
            zvar = torch.diagonal(z_cov, dim1=-2, dim2=-1).clamp_min(0)
        gvar = None if _cond_prev else self._g_var_from_z_var(zvar, g, is_first=is_first)
        g_zcov = None if _cond_prev else self._shift_cov(z_cov, is_first=is_first)
        xc = self._align_xcov(zg_xcov, is_first=is_first, T=z_mean.shape[1])
        ev = self.regimes.expected_loglik(z_mean, g, z_var=zvar, g_var=gvar,
                                          g_zcov=g_zcov, zg_xcov=xc, z_cov=z_cov)
        return ev, g, gvar

    def regime_inference(self, stoch, deter, is_first=None, cache_estep: bool = False,
                         z_var=None,
                         action=None, valid=None, z_cov=None, zg_xcov=None,
                         prev_stoch=None):
        with torch.amp.autocast("cuda", enabled=False):
            _keep = (torch.float32, torch.float64)
            _c = lambda t: None if t is None else (t if t.dtype in _keep else t.float())
            z = _c(stoch)
            d = _c(deter)
            zvar = _c(z_var)
            zcov = _c(z_cov)
            zxc = _c(zg_xcov)
            valid = self._chunk_valid(valid, is_first, z.shape[0], z.shape[1],
                                      z.dtype, z.device)
            ev_raw, g, g_var = self._vb_evidence(
                z, d, is_first=is_first, z_var=zvar, action=action, z_cov=zcov,
                zg_xcov=zxc,
                prev_stoch=_c(prev_stoch))
            ev_raw = self._apply_active_mask(ev_raw)
            _ev_shift = ev_raw.max(dim=-1, keepdim=True).values
            ev = ev_raw - _ev_shift
            _online_pw = bool(getattr(self, "_shs_online_pairwise", False)) and self.recurrent
            _trans_fn = None
            if _online_pw:
                log_init, trans_aux, _trans_fn = self._transition_potentials_ondemand(
                    d, dtype=d.dtype, device=d.device, action=action, is_first=is_first)
                _tf64 = (lambda _t: _trans_fn(_t).double()) if _trans_fn is not None else None
                gamma, counts_base, _logZ_re, _la, _lb = forward_backward(
                    log_init.double(), None, ev.double(), is_first=is_first, valid=valid,
                    assume_start_at_t0=not getattr(self, 'chunk_boundary_mask', True),
                    return_messages=True, trans_fn=_tf64)
                gamma = gamma.to(ev.dtype)
                counts_base = counts_base.to(ev.dtype)
                _logZ_re, _la, _lb = (_logZ_re.to(ev.dtype), _la.to(ev.dtype),
                                      _lb.to(ev.dtype))
                xi = None
            else:
                log_init, log_trans, trans_aux = self._transition_logpotentials(
                    d, dtype=d.dtype, device=d.device, action=action, is_first=is_first)
                gamma, counts_base, _logZ_re, xi = forward_backward(
                    log_init.double(), log_trans.double(), ev.double(),
                    is_first=is_first, valid=valid, return_pairwise=True,
                    assume_start_at_t0=not getattr(self, 'chunk_boundary_mask', True))
                gamma = gamma.to(ev.dtype)
                counts_base = counts_base.to(ev.dtype)
                _logZ_re, xi = _logZ_re.to(ev.dtype), xi.to(ev.dtype)
            if valid is not None:
                _vmask = valid.reshape(_ev_shift.shape[0], _ev_shift.shape[1]).to(_ev_shift.dtype)
            else:
                _vmask = _ev_shift.new_ones(_ev_shift.shape[0], _ev_shift.shape[1])
            self._last_logZ = (_logZ_re
                               + (_ev_shift.squeeze(-1) * _vmask).sum(dim=1)).detach()

            if self.recurrent:
                if _online_pw:
                    from shs_rssm.pairwise_online import accumulate_pg_online
                    r_mass, row_weight, counts = accumulate_pg_online(
                        _la, _lb, _logZ_re, _trans_fn, ev, is_first, valid, trans_aux)
                else:
                    r_mass, row_weight, counts = self.rstick.attribute_bound(xi, trans_aux)
                if cache_estep:
                    self._estep = dict(
                        phi_steps=trans_aux["phi_steps"].detach(),
                        r_mass=r_mass.detach(),
                        row_weight=row_weight.detach(),
                        sig=(r_mass / row_weight.clamp_min(1e-12)).detach(),
                        is_first=is_first, valid=valid,
                    )
            else:
                counts = counts_base
                if cache_estep:
                    self._estep = None

            if cache_estep:
                if _online_pw:
                    self._struct_cache = dict(
                        gamma=gamma.detach(), online=True,
                        messages=(_la.detach(), _lb.detach(), _logZ_re.detach()),
                        trans_aux={k: (v.detach() if torch.is_tensor(v) else v)
                                   for k, v in trans_aux.items()},
                        is_first=is_first, valid=valid,
                        evidence=ev_raw.detach(), evidence_shifted=ev.detach(),
                    )
                else:
                    self._struct_cache = dict(
                        gamma=gamma.detach(), xi=xi.detach(), is_first=is_first,
                        evidence=ev_raw.detach(), evidence_shifted=ev.detach(),
                    )

        start_counts = start_counts_from(
            gamma, is_first, valid=valid,
            assume_start_at_t0=not getattr(self, 'chunk_boundary_mask', True))
        return gamma, counts, start_counts, g

    def _discrete_path_kl(self, gamma, xi, deter, is_first=None, action=None):
        B, T, K = gamma.shape
        dtype, device = gamma.dtype, gamma.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        if is_first is None:
            isf = torch.zeros(B, T, dtype=dtype, device=device)
        else:
            isf = is_first.reshape(B, T).to(dtype=dtype, device=device).clone()
        isf[:, 0] = 1.0

        gclamp = gamma.clamp_min(1e-30)
        out = gamma.new_zeros(B, T)

        init_kl = (gamma * (gclamp.log() - log_init.view(1, 1, K))).sum(-1)
        out = out + isf * init_kl

        if T <= 1:
            return out

        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            phi = self.build_stick_phi(deter, action, is_first=is_first)
            log_trans, _ = self.rstick.bound_log_trans(base_elogpi, phi[:, 1:])
        else:
            log_trans = base_elogpi.view(1, 1, K, K).expand(B, T - 1, K, K)

        x = xi.clamp_min(1e-30)
        gp = gamma[:, :-1].clamp_min(1e-30)
        log_q_cond = x.log() - gp[:, :, :, None].log()
        pair_kl = (xi * (log_q_cond - log_trans)).sum(dim=(-2, -1))
        out[:, 1:] = out[:, 1:] + (1.0 - isf[:, 1:]) * pair_kl
        return out

    def _discrete_path_kl_online(self, gamma, deter, is_first, cache, action=None):
        from shs_rssm.pairwise_online import pair_kl_online
        B, T, K = gamma.shape
        dtype, device = gamma.dtype, gamma.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        isf = (torch.zeros(B, T, dtype=dtype, device=device) if is_first is None
               else is_first.reshape(B, T).to(dtype=dtype, device=device).clone())
        isf[:, 0] = 1.0
        gclamp = gamma.clamp_min(1e-30)
        out = gamma.new_zeros(B, T)
        init_kl = (gamma * (gclamp.log() - log_init.view(1, 1, K))).sum(-1)
        out = out + isf * init_kl
        if T <= 1:
            return out
        la, lb, lZ = cache["messages"]
        val = cache.get("valid", None)
        ev = cache["evidence_shifted"]
        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            phi = self.build_stick_phi(deter, action, is_first=is_first)
            aux_diff = self.rstick.bound_aux_only(base_elogpi, phi[:, 1:])
            trans_diff_fn = lambda t: self.rstick.trans_slice_from_aux(aux_diff, t)
            _caux = cache.get("trans_aux", None)
            if _caux is not None:
                trans_det_fn = lambda t: self.rstick.trans_slice_from_aux(_caux, t)
            else:
                trans_det_fn = lambda t: trans_diff_fn(t).detach()
        else:
            _lt = base_elogpi.view(1, 1, K, K).expand(B, T - 1, K, K)
            trans_diff_fn = _lt
            trans_det_fn = _lt.detach()
        pair_kl = pair_kl_online(la, lb, lZ, trans_det_fn, ev, is_first, val,
                                 gamma, trans_diff_fn)
        out[:, 1:] = out[:, 1:] + (1.0 - isf[:, 1:]) * pair_kl
        return out

    def dynamics_kl(self, q_mean, q_std, stoch, deter, gamma, is_first=None,
                    free: float = 1.0, dyn_scale: float = 0.5, rep_scale: float = 0.1,
                    strict_elbo: bool = False, global_scale: float = 1.0, action=None,
                    prev_stoch=None):
        q_var = (q_std ** 2).clamp_min(1e-8)
        gam = gamma.detach()

        evidence, _, _ = self._vb_evidence(q_mean, deter, is_first=is_first, z_var=q_var,
                                           prev_stoch=prev_stoch,
                                           action=action)
        hq = diag_gauss_entropy(q_var)
        cont_vb = -(gam * evidence).sum(-1) - hq

        _c = getattr(self, "_struct_cache", None)
        if _c is not None and _c.get("online", False) \
                and _c.get("gamma", None) is not None and _c["gamma"].shape == gamma.shape:
            disc = self._discrete_path_kl_online(gam, deter, is_first, _c, action=action)
        else:
            xi = None
            if _c is not None and _c.get("gamma", None) is not None \
                    and _c["gamma"].shape == gamma.shape:
                xi = _c.get("xi", None)
            if xi is None:
                with torch.no_grad():
                    ev = evidence.detach()
                    ev = ev - ev.max(dim=-1, keepdim=True).values
                    log_init, log_trans, _ = self._transition_logpotentials(
                        deter.detach(), dtype=ev.dtype, device=ev.device, action=action, is_first=is_first)
                    _, _, _, xi = forward_backward(
                        log_init, log_trans, ev, is_first=is_first, return_pairwise=True)
            disc = self._discrete_path_kl(gam, xi.to(gam.device, gam.dtype), deter, is_first, action=action)

        rstick_kl = q_mean.new_tensor(0.0)
        if self.recurrent and getattr(self, "rstick", None) is not None:
            rstick_kl = self.rstick.beta_kl().to(dtype=q_mean.dtype, device=q_mean.device)
            rstick_kl = rstick_kl / max(1, q_mean.shape[0] * q_mean.shape[1])

        if strict_elbo:
            free = 0.0
        z0kl_b = self.z0_kl().to(dtype=q_mean.dtype, device=q_mean.device) \
            / max(1, q_mean.shape[0] * q_mean.shape[1])
        cont_for_loss = torch.clamp(cont_vb, min=free) if free is not None and free > 0 else cont_vb
        vb = cont_vb + disc + rstick_kl + z0kl_b
        dyn = cont_for_loss + disc + rstick_kl + z0kl_b
        rep = cont_for_loss
        value = vb
        loss = dyn_scale * (cont_for_loss + disc) + rep_scale * rep + rstick_kl + z0kl_b
        if strict_elbo:
            loss = cont_vb + disc + (z0kl_b - z0kl_b.detach())
            with torch.no_grad():
                gneg = -self.bound_global()
            loss = loss + (global_scale * gneg / max(loss.numel(), 1)) * torch.ones_like(loss)
        return loss, value, dyn, rep

    def _Epi(self):
        active = self.hdp.trans_theta[:, :self.K]
        return active / active.sum(-1, keepdim=True)

    @staticmethod
    def _shift_resp(gamma, is_first=None):
        B, T, K = gamma.shape
        u = gamma.new_full((B, 1, K), 1.0 / K)
        prev = torch.cat([u, gamma[:, :-1]], dim=1) if T > 1 else u
        if is_first is not None:
            isf = is_first.reshape(B, T, 1).to(gamma.dtype)
            prev = torch.where(isf > 0.5, gamma.new_full((B, T, K), 1.0 / K), prev)
        return prev

    def _shift_var(self, v, is_first=None):
        if v is None:
            return None
        v0 = torch.exp(self.z0_logvar).clamp_min(1e-10).to(dtype=v.dtype,
                                                           device=v.device)
        v0 = v0.view(1, 1, -1)
        lead = v0.expand(v.shape[0], 1, v.shape[-1])
        sv = torch.cat([lead, v[:, :-1]], dim=1) if v.shape[1] > 1 else lead
        if is_first is not None:
            isf = is_first.reshape(*is_first.shape[:2], 1).to(v.dtype)
            sv = torch.where(isf > 0.5, v0.expand_as(sv), sv)
        return sv

    def svae_estep(self, z_enc, deter, is_first=None, valid=None, action=None,
                   enc_prec=None, prior_prec=1.0, n_iters=2):
        from .continuous_smoother import chain_potentials, build_blocks, smooth

        if self.shared_carry:
            raise NotImplementedError(
                "svae_estep builds its Gaussian chain from the DiagAR natural parameters; "
                "the shared-carry drift needs a matching chain_potentials implementation.")
        L = int(self.regimes.L)
        B, T, _ = z_enc.shape
        if enc_prec is None:
            raise ValueError(
                "svae_estep needs the encoder POTENTIAL precision. Pass "
                "enc_prec = 1/post['std']**2 -- the amortised posterior's precision is "
                "exactly the natural parameter of the evidence potential.")
        if not torch.is_tensor(enc_prec):
            enc_prec = torch.full_like(z_enc, float(enc_prec))
        pp = torch.full_like(z_enc[:, 0], float(prior_prec))

        z0_mean = self.z0.to(z_enc.dtype).reshape(1, L).expand(B, L)
        mean, cov, xcov, logdet, logZ = z_enc, None, None, None, None
        gamma = counts = sc = None
        for it in range(max(1, int(n_iters))):
            z_var = None if cov is None else torch.diagonal(
                cov, dim1=-2, dim2=-1).clamp_min(0)
            gamma, counts, sc, _ = self.regime_inference(
                mean, deter, is_first=is_first, valid=valid,
                z_var=z_var, z_cov=cov, zg_xcov=xcov,
                action=action, cache_estep=True)
            g_full = self.build_g(self._prev_stoch(mean, is_first), deter,
                                  self._shift_action(action, is_first))
            pot = chain_potentials(self.regimes, gamma.to(mean.dtype), g_full)
            D, U, h = build_blocks(pot, z_enc, enc_prec, prior_prec=pp,
                                   prior_mean=z0_mean, is_first=is_first, valid=valid)
            mean, cov, xcov, logZ, logdet = smooth(D, U, h, return_logdet=True)

        gamma, counts, sc, _ = self.regime_inference(
            mean, deter, is_first=is_first, valid=valid,
            z_var=torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0),
            z_cov=cov, zg_xcov=xcov, action=action, cache_estep=True)

        entropy = -0.5 * logdet + 0.5 * T * L * (1.0 + LOG2PI)
        return dict(mean=mean, cov=cov, xcov=xcov, entropy=entropy, gamma=gamma,
                    counts=counts, sc=sc, logZ_z=logZ,
                    z_var=torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0))

    def svae_local_bound(self, res, deter, is_first=None, action=None, valid=None):
        from .forward_backward import forward_backward

        ev, _, _ = self._vb_evidence(
            res["mean"], deter, is_first=is_first, z_var=res["z_var"],
            action=action, z_cov=res["cov"], zg_xcov=res["xcov"])
        if valid is not None:
            ev = ev * valid.to(ev.dtype).unsqueeze(-1)
        log_init, log_trans, _aux = self._transition_logpotentials(
            deter, dtype=ev.dtype, device=ev.device, action=action, is_first=is_first)
        _fb = forward_backward(
            log_init, log_trans, ev, is_first=is_first, valid=valid,
            assume_start_at_t0=not getattr(self, "chunk_boundary_mask", True))
        logZ_s = _fb[2]
        return logZ_s.to(ev.dtype) + res["entropy"].to(ev.dtype)

    @torch.no_grad()
    def smoothed_estep(self, z_enc, deter, is_first=None, valid=None, action=None,
                       enc_prec=100.0, prior_prec=1.0, n_iters=2, cache_estep=True):

        reg = self.regimes
        L = int(reg.L)
        dt = z_enc.dtype
        z_enc = z_enc.to(dt)
        B, T, _ = z_enc.shape
        if not torch.is_tensor(enc_prec):
            enc_prec = torch.full_like(z_enc, float(enc_prec))
        pp = torch.full_like(z_enc[:, 0], float(prior_prec))

        act = self._shift_action(action, is_first)
        g_full = self.build_g(self._prev_stoch(z_enc, is_first), deter, act)
        _refresh_g = (int(getattr(reg, "q_rank", 0)) > 0
                      and not hasattr(reg, "Lr"))

        mean, cov, xcov, logZ_z = z_enc, None, None, None
        gamma = counts = sc = None
        for _it in range(max(1, int(n_iters))):
            z_var = None if cov is None else torch.diagonal(
                cov, dim1=-2, dim2=-1).clamp_min(0)
            gamma, counts, sc, _ = self.regime_inference(
                mean, deter, is_first=is_first, valid=valid, z_var=z_var,
                z_cov=cov, zg_xcov=xcov, action=action, cache_estep=cache_estep)
            if _refresh_g and _it > 0:
                g_full = self.build_g(self._prev_stoch(mean, is_first), deter, act)
            pot = chain_potentials(reg, gamma.to(mean.dtype), g_full)
            D, U, h = build_blocks(pot, z_enc, enc_prec, prior_prec=pp,
                                   is_first=is_first, valid=valid)
            mean, cov, xcov, logZ_z = smooth(D, U, h)

        return dict(gamma=gamma, counts=counts, sc=sc, mean=mean, cov=cov,
                    xcov=xcov, logZ_z=logZ_z,
                    z_var=torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0))

    def _chunk_valid(self, valid, is_first, B, T, dtype, device):
        if not getattr(self, "chunk_boundary_mask", True) or is_first is None:
            return valid
        isf0 = is_first.reshape(B, T)[:, 0].to(dtype)
        if bool((isf0 > 0.5).all()):
            return valid
        v = (torch.ones(B, T, dtype=dtype, device=device) if valid is None
             else valid.reshape(B, T).to(dtype).clone())
        v[:, 0] = v[:, 0] * isf0
        return v

    def _shift_cov(self, cov, is_first=None):
        if cov is None:
            return None
        B, T, L, _ = cov.shape
        dt = torch.promote_types(cov.dtype, self.z0_logvar.dtype)
        cov = cov.to(dt)
        v0 = torch.exp(self.z0_logvar).clamp_min(1e-10).to(dtype=dt, device=cov.device)
        c0 = torch.diag_embed(v0).view(1, 1, L, L)
        lead = c0.expand(B, 1, L, L)
        sc = torch.cat([lead, cov[:, :-1]], dim=1) if T > 1 else lead
        if is_first is not None:
            isf = is_first.reshape(B, T, 1, 1).to(cov.dtype)
            sc = torch.where(isf > 0.5, c0.expand_as(sc), sc)
        return sc

    def _align_xcov(self, xcov, is_first=None, T=None):
        if xcov is None:
            return None
        xcov = xcov.to(torch.promote_types(xcov.dtype, self.z0_logvar.dtype))
        B, Tm1, L, _ = xcov.shape
        T = (Tm1 + 1) if T is None else int(T)
        zero = xcov.new_zeros(B, 1, L, L)
        xc = torch.cat([zero, xcov.transpose(-1, -2)], dim=1)
        if xc.shape[1] != T:
            xc = xc[:, :T] if xc.shape[1] > T else torch.cat(
                [xc, xc.new_zeros(B, T - xc.shape[1], L, L)], dim=1)
        if is_first is not None:
            isf = is_first.reshape(B, T, 1, 1).to(xc.dtype)
            xc = torch.where(isf > 0.5, torch.zeros_like(xc), xc)
        return xc

    def z0_kl(self):
        s2 = self.z0.new_tensor(float(self._z0_prior_var))
        v0 = torch.exp(self.z0_logvar).clamp_min(1e-10)
        return 0.5 * ((v0 + self.z0 ** 2) / s2 - 1.0
                      + torch.log(s2) - torch.log(v0)).sum()

    @torch.no_grad()
    def get_extra_state(self):
        return dict(stat_store=(self.stat_store.state_dict()
                                if self.stat_store is not None else None),
                    async_version=int(getattr(self, '_async_version', 0)),
                    struct_gen=int(getattr(self, '_struct_gen', 0)),
                    gstep_count=int(getattr(self, '_gstep_count', 0)),
                    episode_cursor=int(getattr(self, '_episode_cursor', 0)),
                    stream_audit=dict(getattr(self, '_stream_audit', {}) or {}),
                    counts_initialised=bool(getattr(self, '_counts_initialised', False)),
                    stats_initialised=bool(getattr(self.regimes, '_stats_initialised', False)),
                    pg_init=bool(getattr(getattr(self, 'rstick', None), '_pg_init', False))
                    if getattr(self, 'rstick', None) is not None else False,
                    active_mask=(self.active_mask.tolist()
                                 if getattr(self, 'active_mask', None) is not None else None))

    def set_extra_state(self, state):
        _resumed = bool(state)
        self._async_version = int((state or {}).get("async_version", 0))
        self._struct_gen = int((state or {}).get("struct_gen", 0))
        self._gstep_count = int((state or {}).get("gstep_count", 0))
        self._episode_cursor = int((state or {}).get("episode_cursor", 0))
        self._stream_audit = dict((state or {}).get("stream_audit", {}) or {})
        st = state or {}
        self._counts_initialised = bool(st.get("counts_initialised", False))
        if getattr(self, 'regimes', None) is not None:
            self.regimes._stats_initialised = bool(st.get("stats_initialised", False))
        if getattr(self, 'rstick', None) is not None:
            self.rstick._pg_init = bool(st.get("pg_init", False))
        _am = st.get("active_mask", None)
        if _am is not None:
            self.active_mask = torch.tensor(_am, dtype=torch.bool, device=self.M_device())
        self._repr_frozen = False
        if _resumed and hasattr(self, "repr_version"):
            self.repr_version += 1
        payload = (state or {}).get("stat_store")
        if payload is not None and self.stat_store is not None:
            self.stat_store.load_state_dict(payload, device=self.ema_trans_counts.device)

    @torch.no_grad()
    def _structural_pre_load_hook(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
        resized = 0
        for name, buf in list(self.named_buffers(recurse=True)):
            key = prefix + name
            if key in state_dict and tuple(state_dict[key].shape) != tuple(buf.shape):
                owner = self
                parts = name.split(".")
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                owner._buffers[parts[-1]] = torch.empty_like(state_dict[key],
                                                            device=buf.device)
                resized += 1
        for name, par in list(self.named_parameters(recurse=True)):
            key = prefix + name
            if key in state_dict and tuple(state_dict[key].shape) != tuple(par.shape):
                owner = self
                parts = name.split(".")
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                setattr(owner, parts[-1],
                        torch.nn.Parameter(torch.empty_like(state_dict[key],
                                                            device=par.device),
                                           requires_grad=par.requires_grad))
                resized += 1
        key = prefix + "ema_trans_counts"
        if key in state_dict:
            k_new = int(state_dict[key].shape[0])
            if k_new != int(self.K):
                for obj in (self, self.regimes, self.hdp,
                            getattr(self, "rstick", None)):
                    if obj is not None and hasattr(obj, "K"):
                        obj.K = k_new

    def _post_load_refresh(self, module, incompatible_keys):
        try:
            lam = self.regimes.lam
            self.regimes._lam_chol = torch.linalg.cholesky(
                lam + 1e-8 * torch.eye(lam.shape[-1], device=lam.device,
                                       dtype=lam.dtype))
        except Exception:
            self.regimes._lam_chol = None

    def begin_full_batch_pass(self):
        if self.stat_store is not None:
            self.stat_store.begin_full_batch_pass()

    @torch.no_grad()
    def finalize_full_batch_pass(self):
        st = self.stat_store
        if st is None or st.mode != "full_batch":
            raise ValueError(
                "finalize_full_batch_pass() requires online_mode='full_batch' "
                f"(store mode is {None if st is None else st.mode!r}) ")
        if getattr(st, "_fb_finalized", False):
            raise ValueError(
                "full_batch pass already finalized; call begin_full_batch_pass() "
                "before accumulating and finalizing a new pass")
        if st._tot_stats is None:
            raise ValueError("cannot finalize an EMPTY full_batch pass -- no batches "
                             "have been accumulated")
        ok = self.global_step_from_totals()
        if ok:
            st._fb_finalized = True
        return ok

    @torch.no_grad()
    def stream_episode(self, stoch, deter, is_first=None, z_var=None, action=None,
                       valid=None, episode_id=None, audit_id=None):
        off = (int(episode_id) if episode_id is not None
               else int(getattr(self, "_episode_cursor", 0)))
        diag = self.absorb_episode(off, stoch, deter, is_first=is_first, z_var=z_var,
                                   action=action, valid=valid)
        self._episode_cursor = off + 1
        diag = dict(diag); diag["stream_offset"] = off
        _aid = audit_id if audit_id is not None else episode_id
        if _aid is not None:
            if not hasattr(self, "_stream_audit"):
                self._stream_audit = {}
            self._stream_audit[int(off)] = str(_aid)
            diag["audit_id"] = str(_aid)
        return diag

    @torch.no_grad()
    def stream_episode_chunks(self, chunks, episode_id=None):
        if not chunks:
            raise ValueError("stream_episode_chunks needs at least one chunk")

        def _cat(key):
            if chunks[0].get(key, None) is None:
                return None
            return torch.cat([c[key] for c in chunks], dim=1)

        z = _cat("stoch")
        d = _cat("deter")
        diag = self.stream_episode(z, d, is_first=_cat("is_first"),
                                   z_var=_cat("z_var"), action=_cat("action"),
                                   episode_id=episode_id)
        diag = dict(diag)
        diag["n_chunks"] = len(chunks)
        diag["n_steps"] = int(z.shape[0] * z.shape[1])
        return diag

    @torch.no_grad()
    def absorb_episode(self, episode_id, stoch, deter, is_first=None, z_var=None,
                       action=None, valid=None):
        st = self.stat_store
        if st is None or st.mode != "streaming":
            raise ValueError(
                "absorb_episode requires online_mode='episode_stream' (streaming); got "
                f"{None if st is None else st.mode!r}")
        _li = max(1, int(getattr(self, "stream_local_iters", 1)))
        gamma = tc = sc = g = None
        for _it in range(_li):
            gamma, tc, sc, g = self.regime_inference(stoch, deter, is_first=is_first,
                                                     cache_estep=True, z_var=z_var,
                                                     action=action, valid=valid)
            if _it < _li - 1:
                _zf = stoch.float()
                _zv = None if z_var is None else z_var.float()
                _stats = self.regimes.stats_from_batch(
                    gamma.to(_zf.dtype), _zf, g, z_var=_zv,
                    g_z_var=self._shift_var(_zv, is_first))
                if st._tot_stats is not None:
                    _stats = {k: st._tot_stats[k] + _stats[k].to(st._tot_stats[k].dtype)
                              for k in _stats}
                self.regimes.set_stats(_stats)
                self.regimes.m_step()
                _C = tc.double() + (st._tot_C.double() if st._tot_C is not None else 0.0)
                _s = sc.double() + (st._tot_s.double() if st._tot_s is not None else 0.0)
                self.hdp.update(_C, _s, n_global_iters=1)
        _lz = getattr(self, "_last_logZ", None)
        elbo = float(_lz.sum()) if torch.is_tensor(_lz) else float(_lz or 0.0)
        self.update_globals(stoch, deter, gamma, tc, sc, is_first=is_first,
                            z_var=z_var, batch_id=episode_id, action=action)
        occ = gamma.detach().sum(dim=tuple(range(gamma.dim() - 1)))
        return dict(episode_id=episode_id, elbo=elbo, K=int(self.K),
                    occupancy=occ.cpu(), n_steps=int(gamma.shape[-2] * gamma.shape[0]
                                                     if gamma.dim() == 3 else gamma.shape[0]),
                    stream_count=int(getattr(st, "_stream_count", 0)))

    def _lock(self):
        lk = getattr(self, "_commit_lock", None)
        if lk is None:
            lk = threading.RLock()
            self._commit_lock = lk
        return lk

    def bump_struct_gen(self):
        self._struct_gen = int(getattr(self, "_struct_gen", 0)) + 1

    def master_snapshot(self):
        with self._lock():
            return dict(
                version=int(getattr(self, "_async_version", 0)),
                struct_gen=int(getattr(self, "_struct_gen", 0)),
                repr_version=int(self.repr_version), K=int(self.K),
                recurrent=bool(self.recurrent),
                state={k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                       for k, v in self.state_dict().items()})

    def _snap_meta_of(self, snap):
        return dict(version=int(snap["version"]), struct_gen=int(snap["struct_gen"]),
                    repr_version=int(snap["repr_version"]), K=int(snap["K"]),
                    recurrent=bool(snap.get("recurrent", self.recurrent)))

    def load_snapshot(self, snap):
        self.load_state_dict(snap["state"])
        self.K = int(snap["K"])
        self.recurrent = bool(snap.get("recurrent", self.recurrent))
        self._async_version = int(snap["version"])
        self._struct_gen = int(snap["struct_gen"])
        self._snap_meta = self._snap_meta_of(snap)
        return self

    def make_worker(self, snapshot=None):
        snap = self.master_snapshot() if snapshot is None else snapshot
        w = RegimeHead(**self._ctor_kwargs)
        return w.load_snapshot(snap)

    @torch.no_grad()
    def async_worker_delta(self, stoch, deter, is_first=None, z_var=None,
                           snapshot=None, worker=None, data_repr_version=None,
                           local_iters: int = 1, action=None, valid=None):
        if worker is None:
            snap = self.master_snapshot() if snapshot is None else snapshot
            w = self.make_worker(snap)
        else:
            w = worker
            if not hasattr(w, "_snap_meta"):
                raise ValueError(
                    "reusable worker was not built by make_worker()/load_snapshot(); it "
                    "has no snapshot identity to tag deltas from")
            if snapshot is not None:
                w.load_snapshot(snapshot)
        meta = w._snap_meta
        if data_repr_version is None:
            if self.stat_store is not None and getattr(self.stat_store, "strict_stream", False):
                raise ValueError(
                    "strict streaming REQUIRES data_repr_version: declare which "
                    "representation encoded these tensors")
        elif int(data_repr_version) != int(meta["repr_version"]):
            raise ValueError(
                f"worker data declares repr_version {int(data_repr_version)} but the "
                f"replica's snapshot is repr_version {int(meta['repr_version'])}; "
                "re-encode the data or load the matching snapshot")
        gamma, tc, sc, _ = w.regime_inference(stoch, deter, is_first=is_first,
                                              cache_estep=True, z_var=z_var,
                                              action=action, valid=valid)
        prev = w._prev_stoch(stoch, is_first)
        g = w.build_g(prev, deter, w._shift_action(action, is_first))
        g_z_var = w._shift_var(None if z_var is None else z_var, is_first)
        stats = w.regimes.stats_from_batch(
            gamma, stoch, g, z_var=z_var, g_z_var=g_z_var)
        pg = None
        if w.recurrent and w._estep is not None:
            es = w._estep
            Dp = es["phi_steps"].shape[-1]
            pg = w.rstick.pg_stats_from_batch(
                es["phi_steps"].reshape(-1, Dp), es["r_mass"].reshape(-1, w.K),
                es["row_weight"].reshape(-1, w.K))
        em_backup = None
        if int(local_iters) > 1:
            em_backup = ({k: v.detach().clone()
                          for k, v in w.regimes.state_dict().items()},
                         bool(getattr(w.regimes, "_stats_initialised", False)))
        for _it in range(max(0, int(local_iters) - 1)):
            w.regimes.set_stats(stats)
            w.regimes.m_step()
            gamma, tc, sc, _ = w.regime_inference(stoch, deter, is_first=is_first,
                                                  cache_estep=True, z_var=z_var,
                                                  action=action, valid=valid)
            stats = w.regimes.stats_from_batch(
                gamma, stoch, g, z_var=z_var, g_z_var=g_z_var)
            if w.recurrent and w._estep is not None:
                es = w._estep
                Dp = es["phi_steps"].shape[-1]
                pg = w.rstick.pg_stats_from_batch(
                    es["phi_steps"].reshape(-1, Dp), es["r_mass"].reshape(-1, w.K),
                    es["row_weight"].reshape(-1, w.K))
        if em_backup is not None:
            w.regimes.load_state_dict(em_backup[0])
            w.regimes._stats_initialised = em_backup[1]
            if hasattr(w.regimes, "_lam_chol"):
                w.regimes._lam_chol = None
        w._estep = None
        return dict(s=stats, C=tc.detach().double(), v=sc.detach().double(), p=pg,
                    base_version=int(meta["version"]),
                    struct_gen=int(meta["struct_gen"]),
                    repr_version=int(meta["repr_version"]), K=int(meta["K"]),
                    recurrent=bool(meta["recurrent"]))

    @torch.no_grad()
    def async_commit(self, offset, delta, refresh: bool = True, tolerate_stale: bool = True,
                     max_stale: int | None = None):
        with self._lock():
            for key in ("base_version", "struct_gen", "repr_version", "K", "recurrent"):
                if key not in delta:
                    raise ValueError(
                        f"async delta is missing required metadata {key!r}; deltas must "
                        "come from async_worker_delta")
            if int(delta["K"]) != int(self.K):
                raise ValueError(
                    f"async delta was computed at K={int(delta['K'])} but the master is "
                    f"now K={int(self.K)}; a structural change invalidates it")
            if int(delta["struct_gen"]) != int(getattr(self, "_struct_gen", 0)):
                raise ValueError(
                    f"async delta was computed at structural generation "
                    f"{int(delta['struct_gen'])} but the master is at generation "
                    f"{int(getattr(self, '_struct_gen', 0))}; an accepted "
                    "birth/merge/delete re-identifies the states even at equal K")
            if int(delta["repr_version"]) != int(self.repr_version):
                if bool(getattr(self.stat_store, "strict_stream", False)):
                    raise ValueError(
                        f"async delta was computed at representation version "
                        f"{int(delta['repr_version'])} but the master is now "
                        f"{int(self.repr_version)}; recompute against the "
                        "current representation")
                self._async_repr_rejects = int(
                    getattr(self, "_async_repr_rejects", 0)) + 1
                if self._async_repr_rejects == 1:
                    import warnings as _w
                    _w.warn("async_commit: dropping delta from a stale "
                            f"representation (v{int(delta['repr_version'])} != "
                            f"v{int(self.repr_version)}); further drops counted "
                            "in _async_repr_rejects")
                return False
            if bool(delta["recurrent"]) != bool(self.recurrent):
                raise ValueError(
                    f"async delta was computed with recurrent={bool(delta['recurrent'])} "
                    f"but the master now runs recurrent={bool(self.recurrent)}; the "
                    "runtime mode changed (curriculum switch) so the delta's statistics "
                    "are structurally wrong")
            cur = int(getattr(self, "_async_version", 0))
            if max_stale is not None and cur - int(delta["base_version"]) > int(max_stale):
                raise ValueError(
                    f"async delta is {cur - int(delta['base_version'])} versions stale at "
                    f"COMMIT time (> max_stale={int(max_stale)}); recompute against a "
                    "fresher snapshot")
            if not tolerate_stale and int(delta["base_version"]) != cur:
                raise ValueError(
                    f"async delta is stale (base_version {int(delta['base_version'])} != "
                    f"master {cur}) and tolerate_stale=False")
            st = self.stat_store

            def _cl(x):
                if x is None:
                    return None
                if torch.is_tensor(x):
                    return x.clone()
                if isinstance(x, dict):
                    return {k: _cl(v) for k, v in x.items()}
                if isinstance(x, (tuple, list)):
                    return type(x)(_cl(v) for v in x)
                return x

            store_backup = dict(
                tot=(_cl(st._tot_stats), _cl(st._tot_C), _cl(st._tot_s), _cl(st._tot_pg)),
                lo=st._async_lo, ahead=set(st._async_ahead), cnt=st._stream_count,
                nup=st.n_updates, api=st._stream_api)
            _MISSING = object()
            head_backup = None
            attr_backup = None
            if refresh:
                head_backup = {k: (v.detach().clone() if torch.is_tensor(v)
                                   else copy.deepcopy(v))
                               for k, v in self.state_dict().items()}
                attr_sites = [(self, "_counts_initialised"),
                              (self.regimes, "_stats_initialised"),
                              (self.regimes, "_lam_chol")]
                if self.recurrent:
                    attr_sites += [(self.rstick, "_pg_init"),
                                   (self.rstick, "n_pg_guard_rejects")]
                attr_backup = [(obj, name, _cl(getattr(obj, name, _MISSING)))
                               for obj, name in attr_sites]
            st.async_commit(offset, dict(s=delta["s"], C=delta["C"], v=delta["v"],
                                         p=delta["p"]))
            if refresh:
                try:
                    self.global_step_from_totals()
                except Exception as e:
                    st._tot_stats, st._tot_C, st._tot_s, st._tot_pg = store_backup["tot"]
                    st._async_lo = store_backup["lo"]
                    st._async_ahead = store_backup["ahead"]
                    st._stream_count = store_backup["cnt"]
                    st.n_updates = store_backup["nup"]
                    st._stream_api = store_backup["api"]
                    self.load_state_dict(head_backup)
                    for obj, name, val in attr_backup:
                        if val is _MISSING:
                            if hasattr(obj, name):
                                delattr(obj, name)
                        else:
                            setattr(obj, name, val)
                    raise ValueError(
                        f"async commit ROLLED BACK: the global refresh failed ({e}); "
                        "the offset was not consumed and the master -- tensors, EMA "
                        "counts, caches and flags -- is unchanged")
            self._async_version = cur + 1
            return int(self._async_version)

    def _hdp_update_scheduled(self, C, s):
        self._gstep_count = int(getattr(self, "_gstep_count", 0)) + 1
        every = int(getattr(self, "hdp_every", 1))
        if every <= 1 or ((self._gstep_count - 1) % every) == 0:
            self.hdp.update(C, s, n_global_iters=self.hdp_iters)
            return True
        return False

    def _pg_set_scheduled(self, A, h):
        every = int(getattr(self, "pg_every", 1))
        if every <= 1 or ((int(getattr(self, "_gstep_count", 1)) - 1) % every) == 0:
            self.rstick.pg_set_totals(A, h)
            return True
        return False

    def global_step_from_totals(self):
        st = self.stat_store
        if st is None or st._tot_stats is None:
            return False
        self.regimes.set_stats(st._tot_stats)
        self.regimes.m_step()
        self.ema_trans_counts.copy_(st._tot_C)
        self.ema_start_counts.copy_(st._tot_s)
        self._counts_initialised = True
        self._hdp_update_scheduled(st._tot_C, st._tot_s)
        if self.recurrent and st._tot_pg is not None:
            self._pg_set_scheduled(st._tot_pg["A"], st._tot_pg["h"])
        return True

    def resync_store_from_buffer(self, buffer):
        store = getattr(self, "stat_store", None)
        if store is None or store.mode != "memoized" or buffer is None:
            return False
        batches = list(getattr(buffer, "batches", []))
        if not batches or any(b.batch_id is None for b in batches):
            return False
        store.reset()
        ids = []
        for b in batches:
            _bact = getattr(b, "action", None)
            gamma, counts, sc, _ = self.regime_inference(
                b.stoch, b.deter, b.is_first, cache_estep=True, z_var=b.z_var,
                action=_bact)
            self.update_globals(b.stoch, b.deter, gamma, counts, sc,
                                is_first=b.is_first, z_var=b.z_var, batch_id=b.batch_id,
                                stats_only=True, action=_bact)
            ids.append(b.batch_id)
        store.expected_ids = set(ids)
        store.expected_batches = len(ids)
        self.global_step_from_totals()
        return True

    def begin_repr_epoch(self):
        self._repr_frozen = True
        return int(self.repr_version)

    def end_repr_epoch(self):
        self._repr_frozen = False
        self.repr_version += 1
        return int(self.repr_version)

    def enable_fixed_kmax(self, n_active=None):
        n = int(self.K if n_active is None else n_active)
        m = torch.zeros(self.K, dtype=torch.bool, device=self.M_device())
        m[:max(0, min(n, self.K))] = True
        self.active_mask = m
        return self

    def M_device(self):
        try:
            return self.regimes.M.device
        except Exception:
            return torch.device("cpu")

    def n_active(self):
        return int(self.K if self.active_mask is None else int(self.active_mask.sum()))

    def activate_slot(self):
        if self.active_mask is None:
            return None
        idx = (~self.active_mask).nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None
        k = int(idx[0])
        self.active_mask[k] = True
        return k

    def deactivate_slot(self, k):
        if self.active_mask is not None and 0 <= int(k) < self.K:
            self.active_mask[int(k)] = False
        return self

    @torch.no_grad()
    def birth_into_slot(self):
        k = self.activate_slot()
        if k is not None and hasattr(self.regimes, "reset_slot"):
            self.regimes.reset_slot(k)
        return k

    @torch.no_grad()
    def delete_slot(self, k):
        self.deactivate_slot(k)
        if hasattr(self.regimes, "reset_slot"):
            self.regimes.reset_slot(int(k))
        return self

    def _mask_logpotentials(self, log_init, log_trans):
        if self.active_mask is None:
            return log_init, log_trans
        m = self.active_mask.to(log_init.device)
        neg = log_init.new_full((), -1e30)
        log_init = torch.where(m, log_init, neg)
        K = int(m.shape[0])
        row = m.view(*([1] * (log_trans.dim() - 2)), K, 1)
        col = m.view(*([1] * (log_trans.dim() - 2)), 1, K)
        log_trans = torch.where(row & col, log_trans, log_trans.new_full((), -1e30))
        return log_init, log_trans

    def _mask_weights(self, w):
        if self.active_mask is None:
            return w
        m = self.active_mask.to(w.device, w.dtype)
        w = w * m
        return w / w.sum(-1, keepdim=True).clamp_min(1e-12)

    def _apply_active_mask(self, ev):
        if self.active_mask is None:
            return ev
        m = self.active_mask.to(ev.device)
        neg = torch.full_like(ev, -1e30)
        return torch.where(m.view(*([1] * (ev.dim() - 1)), -1), ev, neg)

    def bump_repr_version(self):
        if getattr(self, "_repr_frozen", False):
            return
        self.repr_version += 1

    def update_globals(self, stoch, deter, gamma, trans_counts, start_counts,
                       is_first=None, z_var=None, batch_id=None, stats_only=False,
                       repr_version=None,
                       action=None, valid=None, z_cov=None, zg_xcov=None,
                       prev_stoch=None):
        valid = self._chunk_valid(valid, is_first, stoch.shape[0], stoch.shape[1],
                                  stoch.dtype, stoch.device)
        if valid is not None:
            gamma = gamma * valid.reshape(*gamma.shape[:-1], 1).to(gamma.dtype)
        _cond_prev = prev_stoch is not None
        prev = self._prev_stoch(prev_stoch if _cond_prev else stoch, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        if z_var is None and z_cov is not None:
            z_var = torch.diagonal(z_cov, dim1=-2, dim2=-1).clamp_min(0)
        g_z_var = (None if _cond_prev else
                   self._shift_var(None if z_var is None else z_var.detach(), is_first))
        g_zcov = (None if _cond_prev else
                  self._shift_cov(None if z_cov is None else z_cov.detach(), is_first))
        xc = self._align_xcov(None if zg_xcov is None else zg_xcov.detach(),
                              is_first=is_first, T=stoch.shape[1])
        stats = self.regimes.stats_from_batch(
            gamma.detach(), stoch.detach(), g.detach(),
            z_var=None if z_var is None else z_var.detach(), g_z_var=g_z_var,
            g_zcov=g_zcov, zg_xcov=xc,
            z_cov=None if z_cov is None else z_cov.detach())
        tc = trans_counts.detach().double()
        sc = start_counts.detach().double()
        if self.stat_store is not None:
            pg = None
            if self.recurrent and self._estep is not None:
                es = self._estep
                Dp = es["phi_steps"].shape[-1]
                pg = self.rstick.pg_stats_from_batch(
                    es["phi_steps"].reshape(-1, Dp),
                    es["r_mass"].reshape(-1, self.K),
                    es["row_weight"].reshape(-1, self.K))
            if batch_id is None and self.stat_store.mode == "memoized":
                raise ValueError(
                    "online_mode='memoized' requires a STABLE batch_id per corpus "
                    "partition (Hughes replace semantics). Pass update_globals(..., "
                    "batch_id=...) or SHSRSSM.set_batch_id(...); an auto-counter would "
                    "append every replay visit as a new batch and silently break "
                    "memoized/streaming guarantees.")
            if batch_id is not None:
                bid = batch_id
            elif self.stat_store.mode == "streaming":
                bid = int(self.stat_store.n_updates)
            else:
                bid = f"auto{self.stat_store.n_updates}"
            st_tot, C_tot, s_tot, pg_tot = self.stat_store.add_batch(
                bid, stats, tc, sc, pg,
                repr_version=int(self.repr_version if repr_version is None
                                 else repr_version))
            if (self.stat_store.mode == "full_batch" and not stats_only
                    and (self.stat_store.expected_batches is None
                         or self.stat_store.n_batches < self.stat_store.expected_batches)):
                self._estep = None
                return
            if stats_only:
                self._estep = None
                return
            self.regimes.set_stats(st_tot)
            self.regimes.m_step()
            self.ema_trans_counts.copy_(C_tot)
            self.ema_start_counts.copy_(s_tot)
            self._counts_initialised = True
            self._hdp_update_scheduled(C_tot, s_tot)
            if self.recurrent and pg_tot is not None:
                self._pg_set_scheduled(pg_tot["A"], pg_tot["h"])
            if self.stat_store.mode == "full_batch":
                self.stat_store._fb_finalized = True
            self._estep = None
            return
        self.regimes.ema_update_stats(stats, self.ema_tau)
        self.regimes.m_step()
        if not self._counts_initialised:
            self.ema_trans_counts.copy_(tc)
            self.ema_start_counts.copy_(sc)
            self._counts_initialised = True
        else:
            self.ema_trans_counts.mul_(1 - self.ema_tau).add_(self.ema_tau * tc)
            self.ema_start_counts.mul_(1 - self.ema_tau).add_(self.ema_tau * sc)
        self._hdp_update_scheduled(self.ema_trans_counts, self.ema_start_counts)

        if self.recurrent and self._estep is not None:
            es = self._estep
            phi = es["phi_steps"]
            r_mass = es["r_mass"]
            row_weight = es["row_weight"]
            B, S, K = r_mass.shape
            self.rstick.pg_update_statewise(
                phi.reshape(-1, phi.shape[-1]),
                r_mass.reshape(-1, K),
                row_weight.reshape(-1, K),
                lr=self.ema_tau,
            )
            self._estep = None

    def _score_potentials(self, deter, hdp, rstick, dtype=torch.float64, action=None, is_first=None):
        device = deter.device
        log_init = hdp.expected_log_init().to(dtype=dtype, device=device)
        base_elogpi = hdp.expected_log_trans().to(dtype=dtype, device=device)
        if rstick is not None:
            phi = self.build_stick_phi(deter, action, is_first=is_first).to(dtype)
            log_trans, aux = rstick.bound_log_trans(base_elogpi, phi[:, 1:])
            return log_init, log_trans, dict(aux, phi_steps=phi[:, 1:])
        return log_init, base_elogpi, None

    def _score_rstick(self, regimes=None, rstick=None):
        if rstick is not None:
            return rstick
        if self.recurrent and (regimes is None or regimes is self.regimes):
            return getattr(self, "rstick", None)
        return None

    @torch.no_grad()
    def bound_local(self, stoch, deter, is_first=None, regimes=None, hdp=None,
                    rstick=None, z_var=None, action=None, valid=None, prev_stoch=None):
        regimes = regimes if regimes is not None else self.regimes
        hdp = hdp if hdp is not None else self.hdp
        rstick = self._score_rstick(regimes=regimes, rstick=rstick)
        _cond_prev = prev_stoch is not None
        prev = self._prev_stoch(prev_stoch if _cond_prev else stoch, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        g_var = None if _cond_prev else self._g_var_from_z_var(z_var, g, is_first=is_first)
        ev = regimes.expected_loglik(stoch, g, z_var=z_var, g_var=g_var).double()
        log_init, log_trans, aux = self._score_potentials(deter, hdp, rstick, action=action, is_first=is_first)
        valid = self._chunk_valid(valid, is_first, stoch.shape[0], stoch.shape[1],
                                  stoch.dtype, stoch.device)
        gamma, xicount, logZ, xi = forward_backward(
            log_init, log_trans, ev, is_first=is_first, valid=valid,
            assume_start_at_t0=not getattr(self, 'chunk_boundary_mask', True),
            return_pairwise=True)
        if aux is not None:
            _, _, counts = rstick.attribute_bound(xi, aux)
        else:
            counts = xicount
        start_counts = start_counts_from(
            gamma, is_first, valid=valid,
            assume_start_at_t0=not getattr(self, 'chunk_boundary_mask', True))
        return float(logZ.sum()), counts.double(), start_counts.double()

    @torch.no_grad()
    def bound_global(self, regimes=None, hdp=None, rstick=None):
        regimes = regimes if regimes is not None else self.regimes
        hdp = hdp if hdp is not None else self.hdp
        rstick = self._score_rstick(regimes=regimes, rstick=rstick)
        param_kl = regimes.param_kl().sum().double()
        if rstick is not None:
            param_kl = param_kl + rstick.beta_kl().double()
        param_kl = param_kl + self.z0_kl().double()
        _hk = getattr(regimes, "hyper_kl", None)
        if callable(_hk):
            param_kl = param_kl + _hk().double()
        return float(hdp.exact_alloc_elbo() - param_kl)

    @torch.no_grad()
    def bound(self, stoch, deter, is_first=None, regimes=None, hdp=None, z_var=None,
              rstick=None):
        local, _, _ = self.bound_local(stoch, deter, is_first=is_first, regimes=regimes,
                                       hdp=hdp, rstick=rstick, z_var=z_var)
        return local + self.bound_global(regimes=regimes, hdp=hdp, rstick=rstick)

    def imagine_prior(self, prev_stoch, deter, resp_prev, sample=True, prev_var=None,
                      action=None, mode=None):
        g = self.build_g(prev_stoch, deter, action)
        if prev_var is not None:
            zeros_tail = g[..., self.L:] * 0.0
            g_var = torch.cat([prev_var.to(g.dtype), zeros_tail], dim=-1)
        else:
            g_var = None
        if self.recurrent:
            phi = self.build_stick_phi(deter, action)
            Pi = self._Epi().to(prev_stoch.dtype)
            sig = self.rstick.sigma(phi)
            eye = torch.eye(self.K, dtype=prev_stoch.dtype, device=prev_stoch.device)
            M = sig[..., :, None] * eye + (1.0 - sig[..., :, None]) * Pi
            w = torch.einsum("bk,bkl->bl", resp_prev, M).clamp_min(1e-8)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
            w = self._mask_weights(w)
        else:
            Epi = self._Epi().to(prev_stoch.dtype)
            w = mixture_weights(resp_prev, Epi).clamp_min(1e-8)
            w = w / w.sum(-1, keepdim=True)
            w = self._mask_weights(w)
        comp_mean, comp_var = self.regimes.predictive_moments(g, g_var=g_var)
        mean = (w.unsqueeze(-1) * comp_mean).sum(-2)
        dev = comp_mean - mean.unsqueeze(-2)
        var = (w.unsqueeze(-1) * (comp_var + dev.pow(2))).sum(-2)
        var = var.clamp_min(1e-8)
        std = var.sqrt()

        _mode = mode
        if _mode is None:
            _mode = "eval_sample" if getattr(self, "imag_sample_mixture", False) else "actor_moment"
        if _mode not in ("actor_moment", "eval_sample", "reinforce_sample"):
            raise ValueError(f"unknown imagination mode {_mode!r}")
        resp_out = w
        if (not sample) or _mode == "actor_moment":
            z = mean if not sample else (mean + std * torch.randn_like(mean))
        else:
            idx = torch.distributions.Categorical(probs=w).sample()
            L = comp_mean.shape[-1]
            sel = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, L)
            sel_mean = torch.gather(comp_mean, -2, sel).squeeze(-2)
            if self.regimes.q_rank > 0:
                _, comp_d, comp_U = self.regimes.predictive_cov_moments(g, g_var=g_var)
                r = comp_U.shape[-1]
                sel_d = torch.gather(comp_d, -2, sel).squeeze(-2)
                selU = idx.reshape(*idx.shape, 1, 1, 1).expand(*idx.shape, 1, L, r)
                sel_U = torch.gather(comp_U, -3, selU).squeeze(-3)
                eps_r = torch.randn(*sel_mean.shape[:-1], r,
                                    device=sel_mean.device, dtype=sel_mean.dtype)
                z = (sel_mean + sel_d.sqrt() * torch.randn_like(sel_mean)
                     + torch.einsum("...lr,...r->...l", sel_U, eps_r))
            else:
                sel_var = torch.gather(comp_var, -2, sel).squeeze(-2)
                z = sel_mean + sel_var.sqrt() * torch.randn_like(sel_mean)
            resp_out = torch.zeros_like(w).scatter_(-1, idx.unsqueeze(-1), 1.0)
            if _mode == "reinforce_sample":
                self._last_imag_logprob = torch.log(
                    w.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
                tr = getattr(self, "_imag_logprob_trace", None)
                if tr is not None:
                    tr.append(self._last_imag_logprob)
        zmax = 20.0
        std = std.clamp(max=5.0)
        mean = mean.clamp(-zmax, zmax)
        z = z.clamp(-zmax, zmax)
        return z, resp_out, mean, std