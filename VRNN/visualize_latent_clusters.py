
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
from vdvae.vae_helpers import draw_gaussian_diag_samples 

try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
except Exception:  # pragma: no cover
    normalized_mutual_info_score = None
    adjusted_rand_score = None
    adjusted_mutual_info_score = None


# -----------------------------
# Utils
# -----------------------------
def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 [0..255] or float [0..1]/[-1..1] into float [-1..1]."""
    if x.dtype == torch.uint8:
        return x.float() / 127.5 - 1.0
    x = x.float()
    # Heuristic: if looks like [0,1], map to [-1,1]
    if x.numel() > 0:
        mx = float(x.max().item())
        mn = float(x.min().item())
        if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
            x = x * 2.0 - 1.0
    return x


def _to_0_1_for_plot(x: torch.Tensor) -> np.ndarray:
    """x: [C,H,W] or [H,W,C] in [-1,1] or [0,1] -> numpy [H,W,C] in [0,1]."""
    if x.dim() == 3 and x.shape[0] in (1, 3):  # CHW
        x = x.permute(1, 2, 0)
    x = x.detach().float().cpu()
    if x.numel() > 0:
        mx = float(x.max().item())
        mn = float(x.min().item())
        if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
            y = x
        else:
            y = (x + 1.0) / 2.0
    else:
        y = x
    y = y.clamp(0.0, 1.0).numpy()
    if y.shape[-1] == 1:
        y = np.repeat(y, 3, axis=-1)
    return y


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _effective_k(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # exp(H(p))
    return np.exp(_entropy(p, eps=eps))


def _make_grid(images: List[np.ndarray], ncols: int, pad: int = 2) -> np.ndarray:
    """
    images: list of [H,W,3] float in [0,1]
    returns: one big [H_grid, W_grid, 3]
    """
    if len(images) == 0:
        return np.zeros((64, 64, 3), dtype=np.float32)

    H, W = images[0].shape[:2]
    n = len(images)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    grid = np.ones((nrows * H + (nrows - 1) * pad, ncols * W + (ncols - 1) * pad, 3), dtype=np.float32)
    grid *= 0.15  # dark background

    for i, im in enumerate(images):
        r = i // ncols
        c = i % ncols
        y0 = r * (H + pad)
        x0 = c * (W + pad)
        grid[y0:y0 + H, x0:x0 + W] = im
    return grid


# -----------------------------
# DPGMM responsibilities
# -----------------------------
def compute_responsibilities(
    z_tokens: torch.Tensor,     # [N, D]
    pi: torch.Tensor,           # [N, K]
    means: torch.Tensor,        # [N, K, D]
    log_vars: torch.Tensor,     # [N, K, D]
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior responsibilities r_{nk} ∝ pi_{nk} * N(z_n | mu_{nk}, diag(sigma^2_{nk})).
    Returns:
        resp: [N, K] normalized
        k:    [N] argmax component index
    """
    # log N(z|mu, sigma)
    # = -0.5 * sum_d (log(2π) + log σ^2 + (z-mu)^2 / σ^2)
    var = torch.exp(log_vars).clamp_min(eps)
    diff2 = (z_tokens[:, None, :] - means) ** 2
    log_gauss = -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + diff2 / var).sum(dim=-1)  # [N,K]

    log_pi = torch.log(pi.clamp_min(eps))  # [N,K]
    logit = log_pi + log_gauss
    resp = torch.softmax(logit, dim=-1)
    k = torch.argmax(resp, dim=-1)
    return resp, k


def compute_tsne_embedding(
    latents: np.ndarray,
    max_samples: int = 10000,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    random_state: int = 42,
    n_components: int = 2,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust t-SNE embedding with safety checks for NaN/Inf, duplicates, and sklearn edge-cases.

    Returns:
        emb: [M, n_components]
        idx: [M] indices of original points used (downsampled + filtered if needed)
    """
    rng = np.random.default_rng(int(random_state))
    n = int(latents.shape[0])

    if n == 0:
        return np.zeros((0, int(n_components))), np.zeros((0,), dtype=np.int64)

    # Downsample first (keeps plotting cheap / stable)
    if n > int(max_samples):
        idx = rng.choice(n, size=int(max_samples), replace=False).astype(np.int64, copy=False)
    else:
        idx = np.arange(n, dtype=np.int64)

    # Use float64 here: avoids some rare neighbor-graph edge cases on float32
    X = np.asarray(latents[idx], dtype=np.float64)

    # Drop rows with NaN/Inf (should already be filtered upstream, but keep this defensive)
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        idx = idx[finite]
        X = X[finite]

    if X.shape[0] < 5:
        return np.zeros((0, int(n_components))), np.zeros((0,), dtype=np.int64)

    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        # avoid divide-by-zero on constant dims
        sd = np.where(sd < 1e-12, 1.0, sd)
        X = (X - mu) / sd

    # PCA pre-reduction
    if pca_dims is not None and int(pca_dims) > 0 and X.shape[1] > int(pca_dims):
        pca_dims_eff = int(min(int(pca_dims), X.shape[1], max(2, X.shape[0] - 1)))
        X = PCA(n_components=pca_dims_eff, random_state=int(random_state)).fit_transform(X)

    # Tiny jitter if there are (near-)duplicate rows; helps avoid rare kNN reshape issues
    if X.shape[0] >= 2:
        Xr = np.round(X, decimals=6)
        if np.unique(Xr, axis=0).shape[0] < Xr.shape[0]:
            X = X + (1e-6 * rng.standard_normal(X.shape))

    # Clamp perplexity to sklearn constraints: 2 <= perp < n_samples
    n_pts = int(X.shape[0])
    perp = float(perplexity)
    perp = min(perp, (n_pts - 1) / 3.0)  # common rule-of-thumb for stable kNN
    perp = max(2.0, perp)
    # must be strictly < n_samples
    perp = min(perp, float(n_pts - 1) - 1e-3)

    def _make_tsne(method: str, init: str):
        # sklearn changed `n_iter` -> `max_iter` around 1.2
        try:
            return TSNE(
                n_components=int(n_components),
                perplexity=perp,
                init=init,
                learning_rate="auto",
                random_state=int(random_state),
                method=method,
                max_iter=1000,
            )
        except TypeError:
            return TSNE(
                n_components=int(n_components),
                perplexity=perp,
                init=init,
                random_state=int(random_state),
                method=method,
                n_iter=1000,
            )

    # First try the fast default; if it hits the sklearn neighbor-graph reshape edge-case,
    # retry with a safer configuration.
    try:
        tsne = _make_tsne(method="barnes_hut", init="pca" if X.shape[0] > 10 else "random")
        emb = tsne.fit_transform(X)
    except ValueError as e:
        msg = str(e)
        # This specific failure can happen with some sklearn/numpy builds in kNN graph construction.
        if "cannot reshape array" in msg or "reshape" in msg:
            # Safer retry: slightly smaller perplexity + exact method (no kNN graph)
            perp2 = max(2.0, min(perp, 20.0, float(n_pts - 1) - 1e-3))
            perp = perp2  # update for the retry
            tsne = _make_tsne(method="exact", init="random")
            emb = tsne.fit_transform(X)
        else:
            raise

    return emb, idx


# -----------------------------
# Teacher-forced context (SpatioTemporalCore + temporal VDVAE)
 
@torch.no_grad()
def _get_vdvae_out_at_t(
    model,
    batch: Dict,
    device: torch.device,
    t_select: int = 0,
):
    obs = batch["observations"].to(device)
    obs = _to_minus1_1(obs)

    if obs.dim() == 4:
        # [B,C,H,W] -> [B,1,C,H,W]
        obs = obs[:, None, ...]
    if obs.dim() != 5:
        raise ValueError(f"Expected observations to be 4D or 5D, got shape={tuple(obs.shape)}")

    actions = batch.get("actions", None)
    if actions is not None:
        actions = actions.to(device)
        if actions.dim() == 2:
            actions = actions[:, None, ...]  # [B,1,A]
    dones = batch.get("done", batch.get("dones", None))
    if dones is not None:
        dones = dones.to(device)
        if dones.dim() == 1:
            dones = dones[:, None, ...]  # [B,1]v
            
    B, T, C, H, W = obs.shape
    t_select = int(max(0, min(int(t_select), T - 1)))

    # get the exact extra maps from the real forward path
    extra_maps_seq = None
    if getattr(model.rnn, "extra_channels", 0) > 0 and t_select > 0:
        obs_prefix = obs[:, :t_select + 1]
        act_prefix = actions[:, :t_select + 1] if actions is not None else None
        done_prefix = dones[:, :t_select + 1] if dones is not None else None

        prefix_out = model.forward_sequence(
            observations=obs_prefix,
            actions=act_prefix,
            dones=done_prefix,
        )
        extra_maps_seq = prefix_out.get("extra_maps_seq", None)

    core_state = model.rnn.init_state(B, device=device, dtype=obs.dtype)
    h_context_map = model.rnn.out_norm(core_state[0])
    temporal_state = model.vdvae.init_temporal_state(B, device=device, dtype=obs.dtype)

    for t in range(t_select + 1):
        x_t = obs[:, t]
        x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

        if actions is not None and actions.shape[1] > t:
            a_t = actions[:, t-1]
        else:
            a_t = torch.zeros(B, model.action_dim, device=device, dtype=obs.dtype)

        if dones is None or t == 0:
            mask_t = torch.ones(B, device=device, dtype=obs.dtype)
        else:
            mask_t = 1.0 - dones[:, t - 1].float()

        vdvae_out, temporal_state = model.vdvae.forward_temporal_step(
            x_t_nhwc,
            x_t_nhwc,
            h_context=h_context_map,
            a_t=a_t,
            mask_t=mask_t,
            edge_guide=None,
            temporal_state=temporal_state,
        )

        if t < t_select:
            z_mean = vdvae_out["top_q_mean_map"]
            z_logsigma = vdvae_out["top_q_logvar_map"]
            z_map = draw_gaussian_diag_samples(z_mean, z_logsigma)

            extra_maps_t = None
            if extra_maps_seq is not None and t < len(extra_maps_seq):
                saved = extra_maps_seq[t]
                if saved is not None:
                    extra_maps_t = [
                        em.to(device=device, dtype=z_map.dtype) for em in saved
                    ]

            h_context_map, core_state = model.rnn(
                z_map,
                a_t,
                state=core_state,
                mask_t=mask_t,
                extra_maps=extra_maps_t,
            )

    return vdvae_out, x_t, h_context_map



# Latent extraction (+ exemplars)
# -----------------------------
@torch.no_grad()
def extract_latents_and_assignments(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    t_select: int = 0,
    image_level: bool = True,
    max_points: int = 200000,
    chunk: int = 4096,
    use_rnn_context: bool = True,
    # New (appended) options; existing callers remain compatible:
    collect_exemplars: bool = True,
    n_exemplars_per_component: int = 6,
) -> Dict[str, Any]:
    """
    Extract latent representations and DPGMM component assignments.

    Returns dict with keys:
        latents:     [M, D] numpy array
        assignments: [M] numpy array
        pi:          [M, K] numpy array (image-level if image_level=True)
        labels:      [M] numpy array or None
        h_contexts:  [M, H] numpy array (if image_level=True)
        exemplar_images: [K, n, C, H, W] float32 (optional)
        exemplar_scores: [K, n] float32 (optional)
        exemplar_counts: [K] int64 (optional)
    """
    model.eval()

    z_all: List[torch.Tensor] = []
    k_all: List[torch.Tensor] = []
    pi_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    h_context_all: List[torch.Tensor] = []

    exemplar_images = None
    exemplar_scores = None
    exemplar_counts = None

    # Model dims (inferred on first batch)
    C_top = None
    res = None

    total_points = 0
    K_global = None

    # Temporary exemplar stores: list per component of (score, image_tensor)
    exemplar_store: List[List[Tuple[float, torch.Tensor]]] = []

    for bi, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
        if bi >= max_batches:
            break

        labels = None

        # Handle batch formats
        if isinstance(batch, dict) and "observations" in batch:
            obs = batch["observations"]
            if obs.dim() == 5:
                B, T = obs.shape[:2]
                t_sel = int(max(0, min(int(t_select), T - 1)))
                images = obs[:, t_sel]
            else:
                B = obs.shape[0]
                images = obs
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            B = images.shape[0]
            if len(batch) > 1:
                labels = batch[1]
        else:
            continue

        if labels is not None and torch.is_tensor(labels):
            labels = labels.detach().cpu()

        images = images.to(device)  # [B,C,H,W]
        images = _to_minus1_1(images)
        images_nhwc = images.permute(0, 2, 3, 1).contiguous()

                # --- teacher-forced VDVAE at t_select (matches SpatioTemporalCore + temporal decoding) ---
        if isinstance(batch, dict) and "observations" in batch:
            vdvae_out, images_tf, hctx_map = _get_vdvae_out_at_t(model, batch, device, t_select)
        else:
            # non-sequence batches: treat as single-step
            vdvae_out, images_tf, hctx_map = _get_vdvae_out_at_t(model, {"observations": images}, device, 0)

        # Use the teacher-forced frame for exemplars (still in [-1,1])
        images = images_tf

        # Top-level posterior mean map (used as a stable embedding for clustering)
        top_q_mean_map = vdvae_out["top_q_mean_map"]  # [B, C_top, res, res]

        # Prior params are already tokenized consistently inside forward_temporal_step
        prior_params = vdvae_out["prior_params"]
        pi_tok   = prior_params["pi"]        # [N, K]
        means    = prior_params["means"]     # [N, K, C_top]
        log_vars = prior_params["log_vars"]  # [N, K, C_top]

        B2, C_top, res_h, res_w = top_q_mean_map.shape
        assert B2 == B, f"Batch mismatch: expected B={B}, got {B2}"
        assert res_h == res_w, f"Expected square top map, got {res_h}x{res_w}"
        res = int(res_h)

        # tokens: [B*res*res, C_top]
        z_tokens = top_q_mean_map.permute(0, 2, 3, 1).reshape(B * res * res, C_top)

        # (optional) store a compact context summary for downstream analysis
        if hctx_map is not None:
            # [B,Hctx,Ht,Wt] -> [B,Hctx]
            h_context = hctx_map.mean(dim=(2, 3))
        else:
            h_context = None

        N, K = pi_tok.shape[:2]
        if K_global is None:
            K_global = int(K)
            if collect_exemplars and image_level:
                exemplar_store = [[] for _ in range(K_global)]
        else:
            K = int(K_global)

        # responsibilities -> assignments for tokens
        k_tok = torch.empty(N, dtype=torch.long, device=device)
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            resp, k = compute_responsibilities(z_tokens[s:e], pi_tok[s:e], means[s:e], log_vars[s:e])
            k_tok[s:e] = k

        if image_level:
            z_img = top_q_mean_map.mean(dim=(2, 3))                       # [B, C_top]
            k_img = k_tok.view(B, res * res)
            k_img = torch.mode(k_img, dim=1).values                       # [B]
            pi_img = pi_tok.view(B, res * res, -1).mean(dim=1)            # [B, K]

            z_all.append(z_img.detach().cpu())
            k_all.append(k_img.detach().cpu())
            pi_all.append(pi_img.detach().cpu())
            if labels is not None and torch.is_tensor(labels):
                labels_all.append(labels.cpu())
            h_context_all.append(h_context.detach().cpu())

            total_points += int(B)
            if collect_exemplars and exemplar_store and images is not None:
                # score = pi_img at assigned component (confidence)
                conf = pi_img[torch.arange(B), k_img].detach()
                for b in range(B):
                    kk = int(k_img[b].item())
                    sc = float(conf[b].item())
                    # keep top-n per component
                    lst = exemplar_store[kk]
                    if len(lst) < n_exemplars_per_component:
                        lst.append((sc, images[b].detach().cpu()))
                        lst.sort(key=lambda t: t[0], reverse=True)
                    else:
                        if sc > lst[-1][0]:
                            lst[-1] = (sc, images[b].detach().cpu())
                            lst.sort(key=lambda t: t[0], reverse=True)
        else:
            z_all.append(z_tokens.detach().cpu())
            k_all.append(k_tok.detach().cpu())
            pi_all.append(pi_tok.detach().cpu())
            total_points += int(N)

        if total_points >= max_points:
            break

    out: Dict[str, Any] = {}
    if len(z_all) == 0:
        out["latents"] = np.zeros((0, int(C_top or 0)), dtype=np.float32)
        out["assignments"] = np.zeros((0,), dtype=np.int64)
        out["pi"] = np.zeros((0, int(K_global or 0)), dtype=np.float32)
        out["labels"] = None
        return out

    out["latents"] = torch.cat(z_all, dim=0).numpy()
    out["assignments"] = torch.cat(k_all, dim=0).numpy()
    out["pi"] = torch.cat(pi_all, dim=0).numpy()

    out["labels"] = torch.cat(labels_all, dim=0).numpy() if len(labels_all) > 0 else None
    if len(h_context_all) > 0:
        out["h_contexts"] = torch.cat(h_context_all, dim=0).numpy()

    # Finalize exemplars into dense arrays for easier plotting
    if collect_exemplars and image_level and exemplar_store and K_global is not None:
        K = int(K_global)
        # infer C,H,W from first non-empty
        C = H = W = None
        for k in range(K):
            if len(exemplar_store[k]) > 0:
                im = exemplar_store[k][0][1]
                C, H, W = int(im.shape[0]), int(im.shape[1]), int(im.shape[2])
                break
        if C is not None:
            n = int(n_exemplars_per_component)
            ex_imgs = torch.zeros((K, n, C, H, W), dtype=torch.float32)
            ex_sco = torch.full((K, n), float("nan"), dtype=torch.float32)
            ex_cnt = torch.zeros((K,), dtype=torch.long)
            for k in range(K):
                lst = exemplar_store[k]
                ex_cnt[k] = len(lst)
                for j, (sc, im) in enumerate(lst[:n]):
                    ex_imgs[k, j] = im
                    ex_sco[k, j] = sc
            out["exemplar_images"] = ex_imgs.numpy()
            out["exemplar_scores"] = ex_sco.numpy()
            out["exemplar_counts"] = ex_cnt.numpy()

    return out


# Main visualization
def visualize_dpgmm_clustering(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    max_samples: int = 10000,
    perplexity: float = 30.0,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    image_level: bool = True,
    t_select: int = 0,
    use_rnn_context: bool = True,
    figsize: Tuple[int, int] = (18, 14),
    tsne_dims: int = 2,
) -> plt.Figure:
    """
    Create a 2x3 "best of" dashboard:
      (1) t-SNE by component (2D or 3D)
      (2) t-SNE by ground truth label (if available) OR confidence map
      (3) component utilization
      (4) assignment confidence + effective-K distribution
      (5) UMAP by component
      (6) metrics summary
    """
    model.eval()

    data = extract_latents_and_assignments(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        t_select=t_select,
        image_level=image_level,
        use_rnn_context=use_rnn_context,
        max_points=max_samples * 3,   # allow some headroom before t-SNE sampling
        collect_exemplars=True,
        n_exemplars_per_component=6,
    )

    # --- Load arrays ---
    latents = np.asarray(data["latents"], dtype=np.float32)
    assignments = np.asarray(data["assignments"], dtype=np.int64)
    pi = np.asarray(data["pi"], dtype=np.float32)
    labels = data.get("labels", None)
    if labels is not None:
        labels = np.asarray(labels)

    # --- Sanity checks: pi must align with latents ---
    if pi.ndim != 2:
        raise ValueError(f"[viz] Expected pi to be 2D [N,K], got pi.shape={pi.shape}")
    if latents.ndim != 2:
        raise ValueError(f"[viz] Expected latents to be 2D [N,D], got latents.shape={latents.shape}")
    if pi.shape[0] != latents.shape[0] or assignments.shape[0] != latents.shape[0]:
        raise ValueError(
            f"[viz] Shape mismatch: latents={latents.shape}, assignments={assignments.shape}, pi={pi.shape}"
        )

    N = latents.shape[0]
    K = pi.shape[1]

    # --- Mask out any bad rows (NaN/Inf) + invalid assignment indices ---
    finite_lat = np.isfinite(latents).all(axis=1)                 # [N]
    finite_pi  = np.isfinite(pi).all(axis=1)                      # [N]
    valid_asg  = (assignments >= 0) & (assignments < K)           # [N]

    mask = finite_lat & finite_pi & valid_asg

    if labels is not None:
        # support labels as [N] or [N, ...]
        if labels.shape[0] != N:
            raise ValueError(f"[viz] labels has wrong first dim: labels.shape={labels.shape}, expected {N}")
        finite_lab = np.isfinite(labels).all(axis=1) if labels.ndim > 1 else np.isfinite(labels)
        mask &= finite_lab

    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[viz] Dropping {dropped}/{N} points due to NaN/Inf or invalid assignment.")

        latents = latents[mask]
        assignments = assignments[mask]
        pi = pi[mask]
        if labels is not None:
            labels = labels[mask]

    latents = np.ascontiguousarray(latents, dtype=np.float32)
    pi = np.ascontiguousarray(pi, dtype=np.float32)

    # Not enough points?
    if latents.shape[0] < 5:
        raise ValueError("[viz] Not enough valid points for t-SNE after filtering. (Need >= 5)")

    # --- t-SNE dims ---
    tsne_dims = int(2 if tsne_dims is None else tsne_dims)
    tsne_dims = 3 if tsne_dims == 3 else 2

    # --- Clamp perplexity to safe range for current N ---
    # sklearn t-SNE requires perplexity < n_samples
    n_pts = latents.shape[0]
    max_perp = max(2.0, (n_pts - 1) / 3.0)
    if perplexity > max_perp:
        print(f"[viz] Clamping perplexity {perplexity:.2f} -> {max_perp:.2f} (n_points={n_pts}).")
        perplexity = float(max_perp)

    # --- t-SNE embedding ---
    emb, idx = compute_tsne_embedding(
        latents,
        max_samples=max_samples,
        perplexity=float(perplexity),
        pca_dims=50,
        random_state=42,
        n_components=tsne_dims,
        standardize=True,
    )

    # idx indexes into *filtered* arrays (latents/assignments/pi/labels)
    assign_s = assignments[idx]
    pi_s = pi[idx]
    labels_s = labels[idx] if labels is not None else None

    n_components = pi_s.shape[-1]  # should be K

    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, max(1, n_components))))
    if n_components > 20:
        colors = plt.cm.turbo(np.linspace(0, 1, n_components))
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=figsize)

    # (1) t-SNE by component
    if tsne_dims == 3:
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.view_init(elev=25, azim=40)
        sc1 = ax1.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=assign_s, cmap=cmap, s=6, alpha=0.75)
        ax1.set_zlabel("t-SNE 3")
    else:
        ax1 = fig.add_subplot(2, 3, 1)
        sc1 = ax1.scatter(emb[:, 0], emb[:, 1], c=assign_s, cmap=cmap, s=8, alpha=0.75)
    plt.colorbar(sc1, ax=ax1, label="DPGMM Component")
    ax1.set_title("t-SNE by DPGMM Component", fontsize=11)
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")

    # (2) t-SNE by ground truth OR confidence
    if tsne_dims == 3:
        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        ax2.view_init(elev=30, azim=120)
    else:
        ax2 = fig.add_subplot(2, 3, 2)

    if labels_s is not None:
        if tsne_dims == 3:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=labels_s, cmap="Set1", s=6, alpha=0.75)
            ax2.set_zlabel("t-SNE 3")
        else:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=labels_s, cmap="Set1", s=5, alpha=0.75)
        cbar = plt.colorbar(sc2, ax=ax2, label="Ground Truth Label")
        if class_names is not None:
            uniq = np.unique(labels_s)
            cbar.set_ticks(uniq)
            cbar.set_ticklabels([class_names[int(u)] for u in uniq])
        ax2.set_title("t-SNE by Ground Truth Label", fontsize=11)
    else:
        conf = pi_s[np.arange(pi_s.shape[0]), assign_s]
        if tsne_dims == 3:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=conf, cmap="viridis", s=6, alpha=0.75)
            ax2.set_zlabel("t-SNE 3")
        else:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=conf, cmap="viridis", s=8, alpha=0.75)
        plt.colorbar(sc2, ax=ax2, label="assignment confidence")
        ax2.set_title("t-SNE colored by assignment confidence", fontsize=11)

    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")

    # (3) Component utilization (use ALL filtered points, not just idx)
    ax3 = fig.add_subplot(2, 3, 3)
    counts = np.bincount(assignments, minlength=max(1, n_components)).astype(np.float64)
    frac = counts / max(1.0, counts.sum())
    order = np.argsort(frac)[::-1]
    ax3.bar(np.arange(len(frac)), frac[order])
    ax3.set_title("Component utilization (fraction)", fontsize=11)
    ax3.set_xlabel("component (sorted)")
    ax3.set_ylabel("fraction")
    ax3.grid(True, alpha=0.25)

    # (4) Confidence + effective-K distributions
    ax4 = fig.add_subplot(2, 3, 4)
    conf_all = pi[np.arange(pi.shape[0]), assignments]
    effk_all = _effective_k(pi)
    sns.histplot(conf_all, bins=30, stat='density', alpha=0.75, label="confidence (pi@assigned)", ax=ax4)
    sns.histplot(effk_all, bins=30, stat='density', alpha=0.75, label="effective-K = exp(H(pi))", ax=ax4)
    ax4.set_title("Mixture decisiveness diagnostics", fontsize=11)
    ax4.set_xlabel("value")
    ax4.set_ylabel("density")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    # (5) UMAP by component (on the same sampled points idx used for t-SNE)
    ax5 = fig.add_subplot(2, 3, 5)
    umap_emb = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(latents[idx])
    sc5 = ax5.scatter(umap_emb[:, 0], umap_emb[:, 1], c=assign_s, cmap=cmap, s=8, alpha=0.7)
    ax5.set_xlabel("UMAP 1"); ax5.set_ylabel("UMAP 2")
    ax5.set_title("UMAP by DPGMM Component", fontsize=11)
    plt.colorbar(sc5, ax=ax5, label="Component")

    # (6) Metrics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    active_thresh = 0.01  # 1% usage
    k_active = int((frac > active_thresh).sum())
    util_entropy = float(_entropy(frac[frac > 0])[()] / math.log(max(2, (frac > 0).sum()))) if (frac > 0).sum() > 1 else 0.0

    metrics_text = [
        f"Points (used for t-SNE): {emb.shape[0]} / total(valid): {latents.shape[0]}",
        f"K (seen): {n_components}",
        f"Active K (> {active_thresh*100:.0f}%): {k_active}",
        f"Mean confidence: {float(np.nanmean(conf_all)):.3f}",
        f"Mean effective-K: {float(np.nanmean(effk_all)):.2f}",
        f"Utilization entropy (normalized): {util_entropy:.3f}",
    ]

    nmi = ari = ami = None
    if labels is not None and normalized_mutual_info_score is not None:
        nmi = float(normalized_mutual_info_score(labels.astype(int), assignments.astype(int)))
        metrics_text.append(f"NMI(label, cluster): {nmi:.3f}")
    if labels is not None and adjusted_rand_score is not None:
        ari = float(adjusted_rand_score(labels.astype(int), assignments.astype(int)))
        metrics_text.append(f"ARI(label, cluster): {ari:.3f}")
    if labels is not None and adjusted_mutual_info_score is not None:
        ami = float(adjusted_mutual_info_score(labels.astype(int), assignments.astype(int)))
        metrics_text.append(f"AMI(label, cluster): {ami:.3f}")

    ax6.text(0.05, 0.95, "Clustering diagnostics", fontsize=14, fontweight="bold",
             va="top", transform=ax6.transAxes)
    ax6.text(0.05, 0.85, "\n".join(metrics_text), fontsize=10.5, family="monospace",
             va="top", transform=ax6.transAxes)

    title = f"DPGMM Latent Space (image_level={image_level}, t={t_select}, tsne_dims={tsne_dims})"
    if nmi is not None:
        title += f" | NMI={nmi:.3f}"
    fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    # Avoid leaking figures in long-running training jobs (common when using Agg).
    try:
        if "agg" in matplotlib.get_backend().lower():
            plt.close("all")
    except Exception:
        pass

    return fig


# -----------------------------
# Optional: temporal evolution (kept from original intent)
# -----------------------------
@torch.no_grad()
def visualize_component_evolution(
    model,
    dataloader,
    device: torch.device,
    n_timesteps: int = 10,
    max_batches: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how often each component is selected at different timesteps t=0..n_timesteps-1.
    Useful to see whether components map to temporal phases.
    """
    model.eval()

    counts_per_t: List[np.ndarray] = []
    for t in range(n_timesteps):
        data = extract_latents_and_assignments(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
            t_select=t,
            image_level=True,
            use_rnn_context=True,
            max_points=200000,
            collect_exemplars=False,
        )
        a = data["assignments"].astype(np.int64)
        K = int(a.max() + 1) if a.size > 0 else 0
        counts = np.bincount(a, minlength=max(1, K)).astype(np.float64)
        counts_per_t.append(counts)

    # Pad to same K
    Kmax = max(c.shape[0] for c in counts_per_t)
    mat = np.zeros((n_timesteps, Kmax), dtype=np.float64)
    for t, c in enumerate(counts_per_t):
        mat[t, :c.shape[0]] = c / max(1.0, c.sum())

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title("Component utilization over time (fraction)")
    ax.set_xlabel("component index")
    ax.set_ylabel("timestep")
    plt.colorbar(im, ax=ax, label="fraction")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig