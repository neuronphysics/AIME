"""LatentStickiness -- the persistence gate conditioned on z_{t-1}, not h_t.

The shipped RecurrentStickiness reads phi_t = [tanh(P h_t), a_{t-1}, 1], which
is CONSTANT with respect to the continuous latent.  That keeps the discrete and
continuous E-steps separable, but it severs the z -> s edge whenever the
deterministic carry is uninformative (the offline harness supplies h_t = 0, so
the rollout's discrete chain is an open-loop Markov chain; see fhn_benchmark).

This subclass replaces h_t with z_{t-1} and drops the tanh:

    phi_t = [ z_{t-1}, a_{t-1}, 1 ],      z_{t-1} ~ N(mu, diag(s2))

Dropping the nonlinearity is what preserves conjugacy: E[phi] and E[phi phi^T]
are then exact, so the Polya-Gamma update stays closed form.  With tanh(P z)
neither expectation has a closed form.

DROP-IN CONTRACT
----------------
Every method of RecurrentStickiness takes `phi` as its only data argument, so
this class keeps all signatures identical and instead PACKS both moments into
that one tensor:

    phi_packed = cat([E[phi], diag(Cov[phi])], dim=-1)     width 2 * feat_dim

`pack()` builds it; `_unpack()` splits it.  Only the three methods that
actually touch phi's second moment are overridden -- _psi_moments,
pg_stats_from_batch, pg_update_statewise.  bound_log_trans, sigma,
attribute_bound, beta_kl, resized_like, select_rows and merge_rows are
inherited unchanged, so moves.py needs no modification.

The one genuinely new method is hessian_blocks(), the transition potential's
contribution to the continuous-state precision -- identically zero for the
carry gate, non-zero here.

Verified against Monte Carlo and finite differences in
tests/test_latent_gate.py.
"""
from __future__ import annotations

import torch

from .recurrent_stick import RecurrentStickiness


class LatentStickiness(RecurrentStickiness):

    def __init__(self, K: int, feat_dim: int, latent_dim: int, **kw):
        # the parent stores beta over the REAL feature width; the packing is
        # only a transport convention for the two moments.
        # The parent appends its own bias slot: it stores beta over D + 1
        # columns for a feat_dim of D.  phi = [z, a, 1] therefore has width
        # latent_dim + action_dim + 1, so callers pass feat_dim = L + A and the
        # packed-tensor width is 2 * (feat_dim + 1).
        super().__init__(K=K, feat_dim=feat_dim, **kw)
        self.latent_dim = int(latent_dim)
        self.phi_dim = int(feat_dim) + 1

    # ---------------------------------------------------------------- packing
    @staticmethod
    def pack(phibar, s2):
        return torch.cat([phibar, s2], dim=-1)

    def _unpack(self, phi):
        """Accept either a packed (2D) tensor or a bare mean (D) tensor.

        The bare form is what the imagination path supplies when it has a point
        estimate of z and no variance; treating it as zero-variance is exact.
        """
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
        """phi_t = [z_{t-1}, a_{t-1}, 1]; returns (E[phi], diag Cov[phi]).

        Only the z block carries variance -- the action and the bias are
        deterministic given the conditioning information.
        """
        parts = [z_mean]
        vparts = [z_var if z_var is not None else torch.zeros_like(z_mean)]
        if action is not None:
            parts.append(action.to(z_mean.dtype))
            vparts.append(torch.zeros_like(action, dtype=z_mean.dtype))
        ones = z_mean[..., :1] * 0.0 + 1.0
        parts.append(ones)
        vparts.append(torch.zeros_like(ones))
        return torch.cat(parts, -1), torch.cat(vparts, -1)

    # ------------------------------------------------------------- overridden
    def _psi_moments(self, phi):
        """E[psi], Var[psi] with BOTH beta and phi random.

            E[psi]   = m_k^T phibar
            Var[psi] = tr(E[bb^T] E[phi phi^T]) - E[psi]^2
                     = phibar^T E[bb^T] phibar + sum_d E[bb^T]_dd s2_d - E[psi]^2

        The trace never materialises the (D x D) outer product.
        """
        phibar, s2 = self._unpack(phi)
        mb = self.m_beta.to(phibar.dtype)
        Sb = self.Sigma_beta.to(phibar.dtype)
        EbbT = Sb + torch.einsum("kd,ke->kde", mb, mb)
        mu = torch.einsum("...d,kd->...k", phibar, mb)
        quad = torch.einsum("...d,kde,...e->...k", phibar, EbbT, phibar)
        diag = torch.einsum("kdd,...d->...k", EbbT, s2)
        return mu, (quad + diag - mu * mu).clamp_min(0.0)

    def pg_stats_from_batch(self, phi, r_mass, row_weight):
        """Closed-form PG natural parameters under q(z).

            A_k = sum E[omega] (phibar phibar^T + diag(s2))
            h_k = sum (r - w/2) phibar
        """
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
    def pg_update_statewise(self, phi, r_mass, row_weight, lr=None):
        """Same coordinate ascent as the parent, but on the moment-corrected
        statistics.  Implemented via pg_stats_from_batch + pg_set_totals so the
        second-moment correction cannot be bypassed.

        E[omega] depends on q(beta) through c^2 = E[psi^2], so one pass is a
        single coordinate-ascent step; `pg_iters` passes re-evaluate the batch
        statistics under the refreshed q(beta).  In EMA mode the blend is
        always taken against the totals held BEFORE this batch, so the batch is
        not compounded into the memory once per iteration.
        """
        A_prev, h_prev = self.pg_A.clone(), self.pg_h.clone()
        # The first update has nothing to blend against: take the full batch, as
        # the regime/HDP statistics and the carry gate do, instead of scaling it
        # by tau and starting from an all-zero total.
        first = not bool(self._pg_init)
        for _ in range(max(1, int(self.pg_iters))):
            st = self.pg_stats_from_batch(phi, r_mass, row_weight)
            if lr is None or first:
                self.pg_set_totals(st["A"], st["h"])
            else:
                a = float(lr)
                self.pg_set_totals((1 - a) * A_prev + a * st["A"],
                                   (1 - a) * h_prev + a * st["h"])

    # ------------------------------------------------------------------- new
    def hessian_blocks(self, gamma, phi, row_weight):
        """Transition potential's contribution to the continuous-state precision.

        Under PG augmentation the potential is  kappa*psi - omega*psi^2/2  with
        psi = beta^T phi LINEAR in z_{t-1}.  Holding E[omega] fixed (it is the
        auxiliary variable), the m^2 terms cancel and

            -d2/dz2 = sum_k q(z_t=k) E[omega_tk] E[b_k b_k^T],
            b_k = beta_k restricted to its z block.

        psi depends on a SINGLE time index, so this lands only on the primary
        block diagonal: the block-tridiagonal structure of the smoother, and
        its O(T) scaling, are preserved.

        gamma, row_weight: (N, K).  Returns (N, L, L), symmetric PSD.
        """
        m, v = self._psi_moments(phi)
        c = (m * m + v).clamp_min(1e-12).sqrt()
        Eom = (0.5 / c) * torch.tanh(0.5 * c) * row_weight.clamp_min(0.0)
        L = self.latent_dim
        Ebb = (self.Sigma_beta[:, :L, :L]
               + torch.einsum("kd,ke->kde", self.m_beta[:, :L], self.m_beta[:, :L]))
        H = torch.einsum("nk,nk,kde->nde", gamma.to(Ebb.dtype),
                         Eom.to(Ebb.dtype), Ebb)
        return 0.5 * (H + H.transpose(-1, -2))          # kill round-off asymmetry

    # The parent hard-codes `RecurrentStickiness(...)` in resized_like /
    # select_rows / merge_rows, so a plain super() call would silently downgrade
    # the type and lose build_phi_moments.  Rebuild as LatentStickiness and copy
    # the parent's result across; this keeps moves.py working unmodified.
    def _as_latent(self, other):
        new = LatentStickiness(
            K=int(other.K), feat_dim=self.D, latent_dim=self.latent_dim,
            device=self.m0.device, dtype=self._dtype, **self._ctor_kwargs())
        new.load_state_dict(other.state_dict())
        new._pg_init = other._pg_init
        if new.hier_bias:
            new._hier_sync()
        return new

    def resized_like(self, new_K: int):
        return self._as_latent(super().resized_like(int(new_K)))

    def select_rows(self, keep_idx):
        return self._as_latent(super().select_rows(keep_idx))

    def merge_rows(self, i, j):
        return self._as_latent(super().merge_rows(i, j))
