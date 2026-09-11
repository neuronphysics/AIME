#!/usr/bin/env python3
"""Hughes protocol on ToyARK13, at real lap counts.

Reproduces the README benchmark: over-provision at K=25, random contiguous
block init, merge/delete(/seqcreate-birth) moves on a frozen corpus, and
watch K prune toward Ktrue=13 as the whole-corpus bound improves.  Settings
follow x-hdphmm-nips2015 (settings-bnpyHDPHMMdelmerge.txt /
createanddestroy.txt): K=25 start, initname randcontigblocks, hmmKappa 0,
alpha 0.5, gamma 10, startAlpha 10, threshold 0, no create bonus.

    NSEQ=12 LAPS=30 python run_hughes_protocol.py

For the cross-model comparison (rSLDS / TrSLDS / SHS on shared metrics and
figures) use ../compare/ instead; this script exists to reproduce the
protocol trace in the top-level README.
"""
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # AIME root -> shs_rssm

from shs_rssm.regime_head import RegimeHead       # noqa: E402
from shs_rssm.moves import MoveBuffer, sweep_moves  # noqa: E402


def hamming(zt, zp):
    zt, zp = np.asarray(zt).ravel(), np.asarray(zp).ravel()
    n = int(max(zt.max(), zp.max())) + 1
    o = np.zeros((n, n))
    np.add.at(o, (zt, zp), 1)
    r, c = linear_sum_assignment(-o)
    return 1.0 - o[r, c].sum() / len(zt)


def main():
    N = int(os.environ.get("NSEQ", "12"))
    LAPS = int(os.environ.get("LAPS", "30"))
    K0 = 25

    M = loadmat(str(HERE / "HMMdataset.mat"))
    dr = M["doc_range"].ravel()
    N = min(N, len(dr) - 1)
    T = int(dr[1] - dr[0])
    X = np.stack([M["X"][dr[i]:dr[i + 1]] for i in range(N)]).astype(np.float32)
    Z = np.stack([M["TrueZ"].ravel().astype(int)[dr[i]:dr[i + 1]]
                  for i in range(N)])
    Z -= Z.min()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    stoch = torch.tensor(X)
    deter = torch.zeros(N, T, 4)
    isf = torch.zeros(N, T)
    isf[:, 0] = 1.0

    h = RegimeHead(stoch=X.shape[-1], deter=4, K=K0, proj_dim=4,
                   recurrent=False, kappa=0.0, alpha=0.5, gamma=10.0,
                   start_alpha=10.0, ema_tau=1.0, device=torch.device("cpu"))
    h.regimes.calibrate_b0_from_data(stoch, sF=0.1)

    # randcontigblocks init: hard-assign contiguous 100-step blocks
    r0 = np.zeros((N, T, K0), dtype=np.float32)
    for n in range(N):
        for t0 in range(0, T, 100):
            r0[n, t0:t0 + 100, rng.integers(K0)] = 1.0
    r0 = r0 * 0.99 + 0.01 / K0
    r0 = r0 / r0.sum(-1, keepdims=True)
    rt = torch.tensor(r0)
    pv = torch.cat([rt[:, :1], rt[:, :-1]], 1)
    h.update_globals(stoch, deter, rt,
                     torch.einsum("btj,btk->jk", pv[:, 1:], rt[:, 1:]).double(),
                     rt[:, 0].sum(0).double(), is_first=isf)
    for _ in range(10):
        g, C, s0, _ = h.regime_inference(stoch, deter, isf)
        h.update_globals(stoch, deter, g, C, s0, is_first=isf)

    buf = MoveBuffer(max_batches=2, complete=True, expected_batches=1)
    buf.add(stoch, deter, isf, step=0, batch_id="toy", repr_version=0)

    t0 = time.time()
    print(f"ToyARK13 N={N} K0={K0} Ktrue={int(Z.max()) + 1} | "
          f"Hughes protocol, {LAPS} laps")
    for lap in range(LAPS):
        sweep_moves(h, buffer=buf, lap=float(lap), do_birth=(lap >= 2),
                    do_split=False, birth_style="seqcreate", threshold=0.0,
                    create_bonus=0.0)
        for _ in range(3):
            g, C, s0, _ = h.regime_inference(stoch, deter, isf)
            h.update_globals(stoch, deter, g, C, s0, is_first=isf)
        if lap % 5 == 0 or lap == LAPS - 1:
            gd = g.detach()
            occ = gd.sum((0, 1))
            fr = (occ / occ.sum()).numpy()
            print(f"  lap {lap:>3}: K={h.K:>2} occ={int((fr > 0.005).sum()):>2} "
                  f"ham={hamming(Z, gd.argmax(-1).numpy()):.3f} "
                  f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
