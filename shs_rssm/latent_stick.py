from __future__ import annotations

import torch

from .recurrent_stick import RecurrentStickiness


class LatentStickiness(RecurrentStickiness):

    def __init__(self, K: int, feat_dim: int, latent_dim: int, **kw):
        super().__init__(K=K, feat_dim=feat_dim, **kw)
        self.latent_dim = int(latent_dim)
        self.phi_dim = int(feat_dim) + 1

    @staticmethod
    def pack(phibar, s2):
        return torch.cat([phibar, s2], dim=-1)

    def _unpack(self, phi):
        D = self.phi_dim
        if phi.shape[-1] == 2 * D:
            return phi[..., :D], phi[..., D:].clamp_min(0.0)
        if phi.shape[-1] == D:
            return phi, torch.zeros_like(phi)
        raise ValueError(
            f"LatentStickiness expected phi of width {D} (mean only) or {2*D} "
            f"(mean+var packed); got {phi.shape[-1]}")

    @staticmethod
    def build_phi_moments(z_mean, z_var, action=None):
        parts = [z_mean]
        vparts = [z_var if z_var is not None else torch.zeros_like(z_mean)]
        if action is not None:
            parts.append(action.to(z_mean.dtype))
            vparts.append(torch.zeros_like(action, dtype=z_mean.dtype))
        ones = z_mean[..., :1] * 0.0 + 1.0
        parts.append(ones)
        vparts.append(torch.zeros_like(ones))
        return torch.cat(parts, -1), torch.cat(vparts, -1)

    def _psi_moments(self, phi):
        phibar, s2 = self._unpack(phi)
        mb = self.m_beta.to(phibar.dtype)
        Sb = self.Sigma_beta.to(phibar.dtype)
        EbbT = Sb + torch.einsum("kd,ke->kde", mb, mb)
        mu = torch.einsum("...d,kd->...k", phibar, mb)
        quad = torch.einsum("...d,kde,...e->...k", phibar, EbbT, phibar)
        diag = torch.einsum("kdd,...d->...k", EbbT, s2)
        return mu, (quad + diag - mu * mu).clamp_min(0.0)

    def pg_stats_from_batch(self, phi, r_mass, row_weight):
        wd = self.m_beta.dtype
        phibar, s2 = self._unpack(phi)
        phibar, s2 = phibar.to(wd), s2.to(wd)
        r = r_mass.to(wd)
        w = row_weight.to(wd).clamp_min(0.0)
        m, v = self._psi_moments(phi.to(wd))
        c = (m * m + v).clamp_min(1e-12).sqrt()
        Eom = (0.5 / c) * torch.tanh(0.5 * c) * w
        A = torch.einsum("nk,nd,ne->kde", Eom, phibar, phibar)
        A = A + torch.diag_embed(torch.einsum("nk,nd->kd", Eom, s2))
        h = torch.einsum("nk,nd->kd", r - 0.5 * w, phibar)
        return dict(A=A, h=h)

    @torch.no_grad()
    def pg_update_statewise(self, phi, r_mass, row_weight, lr=None, retain=None):
        st = self.pg_stats_from_batch(phi, r_mass, row_weight)
        if lr is None:
            for _ in range(max(1, int(self.pg_iters))):
                self.pg_set_totals(st["A"], st["h"])
                st = self.pg_stats_from_batch(phi, r_mass, row_weight)
        else:
            if torch.is_tensor(lr) and lr.dim() > 0:
                aA = lr.to(device=self.pg_A.device, dtype=self.pg_A.dtype).reshape(-1, 1, 1)
                ah = lr.to(device=self.pg_h.device, dtype=self.pg_h.dtype).reshape(-1, 1)
            else:
                aA = ah = float(lr)
            if retain is not None:
                rA = retain.to(device=self.pg_A.device, dtype=self.pg_A.dtype)
                kA, kh = (1 - aA) * rA.reshape(-1, 1, 1), (1 - ah) * rA.reshape(-1, 1)
            else:
                kA, kh = 1 - aA, 1 - ah
            self.pg_set_totals(kA * self.pg_A + aA * st["A"], kh * self.pg_h + ah * st["h"])

    def hessian_blocks(self, gamma, phi, row_weight):
        m, v = self._psi_moments(phi)
        c = (m * m + v).clamp_min(1e-12).sqrt()
        Eom = (0.5 / c) * torch.tanh(0.5 * c) * row_weight.clamp_min(0.0)
        L = self.latent_dim
        Ebb = (self.Sigma_beta[:, :L, :L]
               + torch.einsum("kd,ke->kde", self.m_beta[:, :L], self.m_beta[:, :L]))
        H = torch.einsum("nk,nk,kde->nde", gamma.to(Ebb.dtype),
                         Eom.to(Ebb.dtype), Ebb)
        return 0.5 * (H + H.transpose(-1, -2))

    def _as_latent(self, other):
        new = LatentStickiness(
            K=int(other.K), feat_dim=self.D, latent_dim=self.latent_dim,
            prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var,
            bias_prior_var=self.bias_prior_var, pg_iters=self.pg_iters,
            uncertainty_correction=self.uncorr,
            device=self.m0.device, dtype=self._dtype)
        new.load_state_dict(other.state_dict())
        return new

    def resized_like(self, new_K: int):
        return self._as_latent(super().resized_like(int(new_K)))

    def select_rows(self, keep_idx):
        return self._as_latent(super().select_rows(keep_idx))

    def merge_rows(self, i, j):
        return self._as_latent(super().merge_rows(i, j))
