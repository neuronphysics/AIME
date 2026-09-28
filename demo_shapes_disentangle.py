import os

import numpy as np
import torch

from envs.shapes import generate_dataset
from shs_rssm.regime_head import RegimeHead
from shs_rssm.moves import MoveBuffer, sweep_moves, _refine, _apply
from shs_rssm import shs_disentangle as DZ

DT = torch.float64
OUT = os.environ.get("SHS_OUT", "./demo_outputs")
os.makedirs(OUT, exist_ok=True)


def pixel_pca_features(frames, out_dim=24, down=10, seed=0):
    import torch.nn.functional as Fnn
    x = torch.from_numpy(frames.astype(np.float32) / 255.0)
    B, T, H, W, C = x.shape
    x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    x = Fnn.adaptive_avg_pool2d(x, (down, down)).reshape(B * T, -1)
    F = x.double().numpy(); F = F - F.mean(0)
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    Z = F @ Vt[:out_dim].T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    return torch.from_numpy(Z.reshape(B, T, out_dim)).to(DT)


def random_conv_features(frames, out_dim=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(frames.astype(np.float32) / 255.0)
    B, T, H, W, C = x.shape
    x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    w1 = torch.randn(16, 3, 5, 5, generator=g) / np.sqrt(3 * 25)
    w2 = torch.randn(32, 16, 5, 5, generator=g) / np.sqrt(16 * 25)
    with torch.no_grad():
        h = torch.relu(torch.nn.functional.conv2d(x, w1, stride=2, padding=2))
        h = torch.relu(torch.nn.functional.conv2d(h, w2, stride=2, padding=2))
        feat = h.mean(dim=(2, 3))
    feat = feat.reshape(B, T, -1).double().numpy()
    F = feat.reshape(B * T, -1); F = F - F.mean(0)
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    Z = F @ Vt[:out_dim].T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    return torch.from_numpy(Z.reshape(B, T, out_dim)).to(DT)


def discover_regimes(Z, K_init=2, n_sweeps=6, refine_iters=4, seed=0):
    torch.manual_seed(seed)
    B, T, L = Z.shape
    H = 4
    deter = torch.zeros(B, T, H, dtype=DT)
    is_first = torch.zeros(B, T, dtype=DT); is_first[:, 0] = 1.0
    head = RegimeHead(stoch=L, deter=H, K=K_init, proj_dim=H, ard=False, kappa=40.0,
                      ema_tau=1.0, hdp_iters=3, dtype=DT)

    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=K_init, n_init=4, random_state=seed).fit_predict(
        Z.reshape(-1, L).numpy()).reshape(B, T)
    prev = head._prev_stoch(Z, is_first); g = head.build_g(prev, deter)
    resp = torch.zeros(B, T, K_init, dtype=DT).scatter_(
        -1, torch.from_numpy(lab).long().unsqueeze(-1), 1.0)
    head.regimes.fit_full_batch(resp, Z, g, n_iter=3)
    from shs_rssm.moves import _current_stats
    C = torch.zeros(K_init, K_init, dtype=DT); s = torch.zeros(K_init, dtype=DT)
    for b in range(B):
        s[lab[b, 0]] += 1
        for t in range(1, T):
            C[lab[b, t - 1], lab[b, t]] += 1
    head.ema_trans_counts.copy_(C); head.ema_start_counts.copy_(s)
    head._counts_initialised = True
    head.hdp.update(C, s, n_global_iters=3)

    buf = MoveBuffer(max_batches=B)
    buf.add(Z, deter, is_first)
    reg, hdp, Cc, sc = _refine(head, head.regimes, head.hdp, buf, iters=refine_iters)
    _apply(head, reg, hdp, Cc, sc)
    K_traj = [head.K]
    for _ in range(n_sweeps):
        sweep_moves(head, buffer=buf, refine_iters=refine_iters, confirm_top=8, merge_topm=25, merge_passes=12)
        K_traj.append(head.K)
    print(f"  regime discovery: K trajectory {K_traj} -> final K={head.K}")

    gamma, _, _, _ = head.regime_inference(Z, deter, is_first)
    return head, gamma


def main():
    print("[1/4] generating moving-shapes dataset (4 shapes, fade in/out) ...")
    frames, factors = generate_dataset(n_seq=6, T=150, n_shapes=4, size=(64, 64), seed=7)
    f64 = {k: torch.from_numpy(v).to(DT) for k, v in factors.items()}
    print(f"      frames {frames.shape}; mean #present="
          f"{factors['n_present'].mean():.2f}, range "
          f"[{int(factors['n_present'].min())},{int(factors['n_present'].max())}]")

    print("[2/4] extracting features (coarse-pixel PCA, fair encoder proxy) ...")
    Z = pixel_pca_features(frames, out_dim=24, down=10, seed=1)
    print(f"      latent features Z {tuple(Z.shape)}")

    print("[3/4] discovering dynamics regimes with the birth/merge/delete moves ...")
    head, gamma = discover_regimes(Z, K_init=2, n_sweeps=6, refine_iters=4, seed=0)

    print("[4/4] disentanglement diagnostics ...")
    m = DZ.plot_tsne_disentangle(
        Z, gamma, f64, f"{OUT}/shapes_tsne_disentangle.png",
        factor_key="n_present", trajectories=True,
        title="SHS-RSSM latents on moving shapes (random-encoder proxy)")
    M, names, mig = DZ.plot_mi_matrix(
        Z, f64, f"{OUT}/shapes_latent_factor_mi.png",
        keys=["n_present", "present", "x", "y"])
    align = DZ.regime_factor_alignment(gamma, f64)
    decode = DZ.factor_decodability(Z, f64, key="n_present")

    snaps = []
    for dim in [2, 6, 12, 24]:
        Zc = pixel_pca_features(frames, out_dim=dim, down=10, seed=1)
        snaps.append(dict(label=f"PCA dim {dim}", stoch=Zc, factors=f64))
    DZ.plot_tsne_evolution(
        snaps, f"{OUT}/shapes_tsne_evolution.png", color_by="factor",
        factor_key="n_present",
        title="Composition structure emerging in the latent as representation capacity "
              "grows (proxy for encoder training)")

    print("\n================ disentanglement metrics ================")
    print(f"  composition decodable from latent : {decode['accuracy']:.3f} "
          f"(balanced k-NN; chance {decode['baseline']:.3f})")
    print(f"  latent<->factor MIG               : {mig:.3f}  "
          f"(per-dim isolation; rises with a trained encoder)")
    print("  -- regimes (switching dynamics) --")
    print(f"  regimes used                      : {align['n_regimes_used']}")
    print(f"  regime<->composition NMI          : {align['regime_composition_nmi']:.3f}")
    print(f"  boundary alignment P/R/F1         : "
          f"{align['boundary_precision']:.2f} / {align['boundary_recall']:.2f} / "
          f"{align['boundary_f1']:.2f}  (switches vs object events)")
    print("  figures written to", OUT)
    print("=========================================================")


if __name__ == "__main__":
    main()
