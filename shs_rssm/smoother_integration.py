from __future__ import annotations

import torch

from .continuous_smoother import (chain_potentials, expected_potentials,
                                  build_blocks, smooth, smoothed_stats)


@torch.no_grad()
def smoothed_estep(head, z_enc, deter, is_first=None, valid=None, enc_prec=None,
                   prior_prec=1.0, n_iters=2, action=None):
    if enc_prec is None:
        enc_prec = 100.0
    res = head.smoothed_estep(z_enc, deter, is_first=is_first, valid=valid,
                              action=action, enc_prec=enc_prec,
                              prior_prec=prior_prec, n_iters=n_iters,
                              cache_estep=True)
    act = head._shift_action(action, is_first)
    g_full = head.build_g(head._prev_stoch(res["mean"], is_first), deter, act)
    res["stats"] = smoothed_stats(
        head.regimes, res["gamma"].to(res["mean"].dtype), res["mean"],
        res["cov"], res["xcov"], g_full[..., head.regimes.L:], valid=valid,
        is_first=is_first, z0_mean=head.z0.detach(),
        z0_var=torch.exp(head.z0_logvar.detach()))
    return res


@torch.no_grad()
def apply_smoothed_globals(head, res, batch_id=None, is_first=None, valid=None,
                           z=None, deter=None, action=None):
    if z is None:
        raise ValueError("apply_smoothed_globals requires z (the observations) as "
                         "target; passing the smoothed mean fits the model to itself.")
    head.update_globals(z, deter, res["gamma"], res["counts"], res["sc"],
                        is_first=is_first, valid=valid, action=action,
                        z_cov=res["cov"], zg_xcov=res["xcov"],
                        batch_id=batch_id, stats_only=True)
    head.global_step_from_totals()
    return head


@torch.no_grad()
def sweep_encoder_precision(head_factory, corpus, K0, precisions=(1.0, 4.0, 16.0, 100.0),
                            seed=0, n_iters=2):
    rows = []
    for p in precisions:
        torch.manual_seed(int(seed))
        head = head_factory(K0)
        e = corpus[0]
        z, dtr, isf = e[1], e[2], e[3]
        vm = e[5] if len(e) > 5 else None
        res = head.smoothed_estep(z, dtr, is_first=isf, valid=vm,
                                  enc_prec=float(p), n_iters=n_iters)
        drift = float((res['mean'] - z).abs().mean())
        sd = float(res['cov'].diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt().mean())
        xc = float(res['xcov'].abs().mean())
        rows.append(dict(enc_prec=p, logZ_z=float(res['logZ_z'].sum()),
                         drift=drift, post_sd=sd, xcov=xc))
    return rows
