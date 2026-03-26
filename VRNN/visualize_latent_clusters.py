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


def sample_prior_image_latent(
    prior_params: Dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one full image-level latent per image from the image-level DPGMM prior.

    Returns:
        z_img: [B, D] sampled latent
        idx:   [B] sampled component index
        probs: [B, K] sampling probabilities used
    """
    pi = prior_params["pi"]
    means = prior_params["means"]
    log_vars = prior_params["log_vars"]

    eps = torch.finfo(means.dtype).eps
    probs = pi.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    tau = max(float(temperature), 1e-4)
    logits = torch.log(probs)
    if temperature != 1.0:
        probs = torch.softmax(logits / tau, dim=-1)

    idx = torch.distributions.Categorical(probs=probs).sample()
    batch_idx = torch.arange(means.shape[0], device=means.device)
    mu_img = means[batch_idx, idx]
    std_img = torch.exp(0.5 * log_vars[batch_idx, idx]).clamp_min(eps)
    z_img = mu_img + torch.randn_like(mu_img) * std_img
    return z_img, idx, probs

# Teacher-forced context (SpatioTemporalCore + VDVAE)
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
        obs = obs[:, None, ...]
    if obs.dim() != 5:
        raise ValueError(f"Expected observations to be 4D or 5D, got shape={tuple(obs.shape)}")

    actions = batch.get("actions", None)
    if actions is not None:
        actions = actions.to(device)
        if actions.dim() == 2:
            actions = actions[:, None, ...]

    dones = batch.get("done", batch.get("dones", None))
    if dones is not None:
        dones = dones.to(device)
        if dones.dim() == 1:
            dones = dones[:, None]

    B, T, C, H, W = obs.shape
    t_select = int(max(0, min(int(t_select), T - 1)))

    edge_guide_t = None
    if t_select > 0:
        prefix_out = model.forward_sequence(
            observations=obs[:, :t_select + 1],
            actions=actions[:, :t_select + 1] if actions is not None else None,
            dones=dones[:, :t_select + 1] if dones is not None else None,
        )

        h_context_maps_seq = prefix_out["h_context_maps_seq"]
        if isinstance(h_context_maps_seq, torch.Tensor):
            h_context_map_t = h_context_maps_seq[:, t_select].to(device=device, dtype=obs.dtype)
        else:
            h_context_map_t = h_context_maps_seq[t_select].to(device=device, dtype=obs.dtype)

        edge_guide_seq = prefix_out.get("edge_guide_seq", None)
        if edge_guide_seq is not None and t_select < len(edge_guide_seq):
            saved_edge = edge_guide_seq[t_select]
            if saved_edge is not None:
                edge_guide_t = saved_edge.to(device=device, dtype=obs.dtype)
    else:
        core_state = model.rnn.init_state(B, device=device, dtype=obs.dtype)
        h_context_map_t = model.rnn.out_norm(core_state[0])

    x_t = obs[:, t_select]
    x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

    if dones is None or t_select == 0:
        mask_t = torch.ones(B, device=device, dtype=torch.float32)
    else:
        mask_t = 1.0 - dones[:, t_select - 1].float()

    vdvae_out = model.vdvae.forward(
        x_t_nhwc,
        x_t_nhwc,
        h_context=h_context_map_t,
        mask_t=mask_t,
        edge_guide=edge_guide_t,
        get_latents=True,
    )

    return vdvae_out, x_t, h_context_map_t
# Latent extraction (+ exemplars)
# -----------------------------
@torch.no_grad()
def extract_latents_and_assignments(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    t_select: int = 0,
    max_points: int = 200000,
    chunk: int = 4096,
    collect_exemplars: bool = True,
    n_exemplars_per_component: int = 6,
    store_h_context: bool = False,
) -> Dict[str, Any]:
    """
    Extract both posterior and true prior whole-image latents for the top image-level DPGMM.

    Returns dict with keys:
        posterior_latents: [M, D] numpy array, posterior encoder mean latents
        posterior_assignments: [M] numpy array, argmax responsibilities of posterior latents under the prior
        posterior_pi: [M, K] numpy array
        prior_latents: [M, D] numpy array, true samples from p(z_top | h)
        prior_assignments: [M] numpy array, sampled prior component indices
        prior_pi: [M, K] numpy array, sampling probabilities used for the prior samples
        labels: [M] numpy array or None
        h_contexts: [M, Hctx*Ht*Wt] numpy array (optional, only if store_h_context=True)
        exemplar_images: [K, n, C, H, W] float32 (optional)
        exemplar_scores: [K, n] float32 (optional)
        exemplar_counts: [K] int64 (optional)

        Backward-compatible aliases:
            latents -> posterior_latents
            assignments -> posterior_assignments
            pi -> posterior_pi
    """
    model.eval()

    posterior_z_all: List[torch.Tensor] = []
    posterior_k_all: List[torch.Tensor] = []
    posterior_pi_all: List[torch.Tensor] = []
    prior_z_all: List[torch.Tensor] = []
    prior_k_all: List[torch.Tensor] = []
    prior_pi_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    h_context_all: List[torch.Tensor] = []

    total_points = 0
    K_global: Optional[int] = None
    exemplar_store: List[List[Tuple[float, torch.Tensor]]] = []

    for bi, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
        if bi >= max_batches or total_points >= max_points:
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

        images = images.to(device)            # [B, C, H, W]
        images = _to_minus1_1(images)

        # Teacher-forced temporal forward to the selected timestep.
        if isinstance(batch, dict) and "observations" in batch:
            vdvae_out, images_tf, hctx_map = _get_vdvae_out_at_t(
                model, batch, device, t_select
            )
        else:
            vdvae_out, images_tf, hctx_map = _get_vdvae_out_at_t(
                model, {"observations": images}, device, 0
            )

        # Use the exact teacher-forced frame returned by the helper for exemplars.
        images = images_tf

        top_q_mean_map = vdvae_out["top_q_mean_map"]   # [B, C_top, Ht, Wt]
        prior_params = vdvae_out["prior_params"]

        if prior_params is None:
            continue

        B2, C_top, Ht, Wt = top_q_mean_map.shape
        if B2 != B:
            raise ValueError(f"Batch mismatch: expected B={B}, got B2={B2}")

        # This visualization is for the image-level DPGMM prior only.
        if (
            not isinstance(prior_params, dict)
            or "pi" not in prior_params
            or "means" not in prior_params
            or "log_vars" not in prior_params
            or prior_params["pi"].dim() != 2
            or prior_params["pi"].shape[0] != B
            or prior_params["means"].dim() != 3
            or prior_params["means"].shape[0] != B
            or prior_params["means"].shape[-1] != C_top * Ht * Wt
        ):
            raise ValueError(
                "extract_latents_and_assignments now expects an image-level DPGMM prior "
                "with pi:[B,K], means/log_vars:[B,K,C_top*Ht*Wt]."
            )

        # Posterior whole-image latent used by the actual image-level DPGMM code.
        top_q_logsigma_map = vdvae_out["top_q_logvar_map"]
        z_top_sample_map = draw_gaussian_diag_samples(top_q_mean_map, top_q_logsigma_map)
        z_img_full = z_top_sample_map.reshape(B, -1).contiguous()

        # Posterior responsibilities under the image-level DPGMM prior.
        resp_img = model.vdvae.prior.compute_responsibilities(
            z_img=z_img_full,
            prior_params=prior_params,
        )   # [B, K]
        k_img = torch.argmax(resp_img, dim=-1)   # [B]

        # True prior sample z ~ p(z_top | h) plus the sampled component index.
        z_prior_img, k_prior_img, pi_prior_img = sample_prior_image_latent(
            prior_params=prior_params,
            temperature=1.0,
        )

        K = int(resp_img.shape[1])
        if K_global is None:
            K_global = K
            if collect_exemplars:
                exemplar_store = [[] for _ in range(K_global)]
        elif K != K_global:
            raise ValueError(f"Inconsistent K across batches: got {K}, expected {K_global}")

        posterior_z_all.append(z_img_full.detach().cpu())
        posterior_k_all.append(k_img.detach().cpu())
        posterior_pi_all.append(resp_img.detach().cpu())
        prior_z_all.append(z_prior_img.detach().cpu())
        prior_k_all.append(k_prior_img.detach().cpu())
        prior_pi_all.append(pi_prior_img.detach().cpu())

        if labels is not None and torch.is_tensor(labels):
            labels_all.append(labels)

        if store_h_context and hctx_map is not None:
            # Keep the full context map, not a spatial mean.
            h_context_all.append(hctx_map.reshape(B, -1).detach().cpu())

        total_points += int(B)

        if collect_exemplars and images is not None:
            conf = resp_img[torch.arange(B, device=resp_img.device), k_img].detach()
            for b in range(B):
                kk = int(k_img[b].item())
                sc = float(conf[b].item())
                lst = exemplar_store[kk]
                if len(lst) < n_exemplars_per_component:
                    lst.append((sc, images[b].detach().cpu()))
                    lst.sort(key=lambda t: t[0], reverse=True)
                else:
                    if sc > lst[-1][0]:
                        lst[-1] = (sc, images[b].detach().cpu())
                        lst.sort(key=lambda t: t[0], reverse=True)

    out: Dict[str, Any] = {}

    if len(posterior_z_all) == 0:
        out["posterior_latents"] = np.zeros((0, 0), dtype=np.float32)
        out["posterior_assignments"] = np.zeros((0,), dtype=np.int64)
        out["posterior_pi"] = np.zeros((0, 0), dtype=np.float32)
        out["prior_latents"] = np.zeros((0, 0), dtype=np.float32)
        out["prior_assignments"] = np.zeros((0,), dtype=np.int64)
        out["prior_pi"] = np.zeros((0, 0), dtype=np.float32)
        out["latents"] = out["posterior_latents"]
        out["assignments"] = out["posterior_assignments"]
        out["pi"] = out["posterior_pi"]
        out["labels"] = None
        if store_h_context:
            out["h_contexts"] = np.zeros((0, 0), dtype=np.float32)
        return out

    out["posterior_latents"] = torch.cat(posterior_z_all, dim=0).numpy()
    out["posterior_assignments"] = torch.cat(posterior_k_all, dim=0).numpy()
    out["posterior_pi"] = torch.cat(posterior_pi_all, dim=0).numpy()
    out["prior_latents"] = torch.cat(prior_z_all, dim=0).numpy()
    out["prior_assignments"] = torch.cat(prior_k_all, dim=0).numpy()
    out["prior_pi"] = torch.cat(prior_pi_all, dim=0).numpy()
    out["latents"] = out["posterior_latents"]
    out["assignments"] = out["posterior_assignments"]
    out["pi"] = out["posterior_pi"]
    out["labels"] = torch.cat(labels_all, dim=0).numpy() if len(labels_all) > 0 else None

    if store_h_context:
        out["h_contexts"] = (
            torch.cat(h_context_all, dim=0).numpy()
            if len(h_context_all) > 0
            else np.zeros((out["latents"].shape[0], 0), dtype=np.float32)
        )

    # Finalize exemplars
    if collect_exemplars and K_global is not None and len(exemplar_store) > 0:
        K = int(K_global)
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
    t_select: int = 0,
    figsize: Tuple[int, int] = (20, 16),
    tsne_dims: int = 3,
) -> plt.Figure:
    """
    Create a 2x2 dashboard with true prior vs posterior embeddings:
      (1) 3D t-SNE of posterior latents, colored by component
      (2) 3D t-SNE of prior samples, colored by component
      (3) 2D UMAP of posterior latents, colored by component
      (4) 2D UMAP of prior samples, colored by component

    Posterior latents come from the encoder top posterior mean.
    Prior latents are true samples z ~ p(z_top | h) from the image-level DPGMM prior.
    """
    model.eval()

    data = extract_latents_and_assignments(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        t_select=t_select,
        max_points=max_samples * 3,
        collect_exemplars=True,
        n_exemplars_per_component=6,
    )

    posterior_latents = np.asarray(data["posterior_latents"], dtype=np.float32)
    posterior_assignments = np.asarray(data["posterior_assignments"], dtype=np.int64)
    posterior_pi = np.asarray(data["posterior_pi"], dtype=np.float32)

    prior_latents = np.asarray(data["prior_latents"], dtype=np.float32)
    prior_assignments = np.asarray(data["prior_assignments"], dtype=np.int64)
    prior_pi = np.asarray(data["prior_pi"], dtype=np.float32)

    if posterior_latents.ndim != 2 or prior_latents.ndim != 2:
        raise ValueError(
            f"[viz] Expected 2D latent arrays, got posterior={posterior_latents.shape}, prior={prior_latents.shape}"
        )
    if posterior_pi.ndim != 2 or prior_pi.ndim != 2:
        raise ValueError(
            f"[viz] Expected 2D pi arrays, got posterior={posterior_pi.shape}, prior={prior_pi.shape}"
        )

    if posterior_latents.shape[0] < 5 or prior_latents.shape[0] < 5:
        raise ValueError("[viz] Not enough valid points for visualization. Need at least 5 posterior and 5 prior points.")

    K = int(max(posterior_pi.shape[1], prior_pi.shape[1]))

    def _filter_valid(latents: np.ndarray, assignments: np.ndarray, pi: np.ndarray):
        finite_lat = np.isfinite(latents).all(axis=1)
        finite_pi = np.isfinite(pi).all(axis=1)
        valid_asg = (assignments >= 0) & (assignments < pi.shape[1])
        mask = finite_lat & finite_pi & valid_asg
        latents_f = np.ascontiguousarray(latents[mask], dtype=np.float32)
        assignments_f = np.asarray(assignments[mask], dtype=np.int64)
        pi_f = np.ascontiguousarray(pi[mask], dtype=np.float32)
        return latents_f, assignments_f, pi_f, int((~mask).sum())

    posterior_latents, posterior_assignments, posterior_pi, post_dropped = _filter_valid(
        posterior_latents, posterior_assignments, posterior_pi
    )
    prior_latents, prior_assignments, prior_pi, prior_dropped = _filter_valid(
        prior_latents, prior_assignments, prior_pi
    )

    if post_dropped > 0:
        print(f"[viz] Dropping {post_dropped} posterior points due to NaN/Inf or invalid assignment.")
    if prior_dropped > 0:
        print(f"[viz] Dropping {prior_dropped} prior points due to NaN/Inf or invalid assignment.")

    if posterior_latents.shape[0] < 5 or prior_latents.shape[0] < 5:
        raise ValueError("[viz] Not enough valid points after filtering. Need at least 5 posterior and 5 prior points.")

    colors = plt.cm.tab20b(np.linspace(0, 1, min(20, max(1, K))))
    if K > 20:
        colors = plt.cm.turbo(np.linspace(0, 1, K))
    cmap = ListedColormap(colors)

    def _embed_tsne_3d(latents: np.ndarray, seed: int):
        n_pts = latents.shape[0]
        max_perp = max(2.0, (n_pts - 1) / 3.0)
        perp = float(min(float(perplexity), max_perp))
        return compute_tsne_embedding(
            latents,
            max_samples=max_samples,
            perplexity=perp,
            pca_dims=50,
            random_state=seed,
            n_components=3,
            standardize=True,
        )

    def _embed_umap_2d(latents: np.ndarray, seed: int):
        n = int(latents.shape[0])
        rng = np.random.default_rng(seed)
        if n > int(max_samples):
            idx = rng.choice(n, size=int(max_samples), replace=False).astype(np.int64, copy=False)
        else:
            idx = np.arange(n, dtype=np.int64)
        X = np.asarray(latents[idx], dtype=np.float32)
        finite = np.isfinite(X).all(axis=1)
        idx = idx[finite]
        X = X[finite]
        if X.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        emb = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(X)
        return emb, idx

    post_tsne_3d, post_tsne_idx = _embed_tsne_3d(posterior_latents, seed=42)
    prior_tsne_3d, prior_tsne_idx = _embed_tsne_3d(prior_latents, seed=43)
    post_umap_2d, post_umap_idx = _embed_umap_2d(posterior_latents, seed=52)
    prior_umap_2d, prior_umap_idx = _embed_umap_2d(prior_latents, seed=53)

    if post_tsne_3d.shape[0] == 0 or prior_tsne_3d.shape[0] == 0:
        raise ValueError("[viz] t-SNE embedding failed to produce enough points.")
    if post_umap_2d.shape[0] == 0 or prior_umap_2d.shape[0] == 0:
        raise ValueError("[viz] UMAP embedding failed to produce enough points.")

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.view_init(elev=25, azim=40)
    sc1 = ax1.scatter(
        post_tsne_3d[:, 0],
        post_tsne_3d[:, 1],
        post_tsne_3d[:, 2],
        c=posterior_assignments[post_tsne_idx],
        cmap=cmap,
        s=6,
        alpha=0.75,
        vmin=0,
        vmax=max(K - 1, 0),
    )
    plt.colorbar(sc1, ax=ax1, label="DPGMM Component")
    ax1.set_title("Posterior Latent 3D t-SNE", fontsize=11)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.set_zlabel("t-SNE 3")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.view_init(elev=25, azim=40)
    sc2 = ax2.scatter(
        prior_tsne_3d[:, 0],
        prior_tsne_3d[:, 1],
        prior_tsne_3d[:, 2],
        c=prior_assignments[prior_tsne_idx],
        cmap=cmap,
        s=6,
        alpha=0.75,
        vmin=0,
        vmax=max(K - 1, 0),
    )
    plt.colorbar(sc2, ax=ax2, label="DPGMM Component")
    ax2.set_title("Prior Latent 3D t-SNE", fontsize=11)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_zlabel("t-SNE 3")

    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(
        post_umap_2d[:, 0],
        post_umap_2d[:, 1],
        c=posterior_assignments[post_umap_idx],
        cmap=cmap,
        s=8,
        alpha=0.75,
        vmin=0,
        vmax=max(K - 1, 0),
    )
    plt.colorbar(sc3, ax=ax3, label="DPGMM Component")
    ax3.set_title("Posterior Latent 2D UMAP", fontsize=11)
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")

    ax4 = fig.add_subplot(2, 2, 4)
    sc4 = ax4.scatter(
        prior_umap_2d[:, 0],
        prior_umap_2d[:, 1],
        c=prior_assignments[prior_umap_idx],
        cmap=cmap,
        s=8,
        alpha=0.75,
        vmin=0,
        vmax=max(K - 1, 0),
    )
    plt.colorbar(sc4, ax=ax4, label="DPGMM Component")
    ax4.set_title("Prior Latent 2D UMAP", fontsize=11)
    ax4.set_xlabel("UMAP 1")
    ax4.set_ylabel("UMAP 2")

    title = (
        f"DPGMM Prior vs Posterior Latent Embeddings (t={t_select}) | "
        f"posterior N={posterior_latents.shape[0]}, prior N={prior_latents.shape[0]}, K={K}"
    )
    fig.suptitle(title, fontsize=13, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    try:
        if "agg" in matplotlib.get_backend().lower():
            plt.close("all")
    except Exception:
        pass

    return fig


def visualize_component_evolution(
    model,
    dataloader,
    device: torch.device,
    n_timesteps: int = 10,
    max_batches: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    model.eval()

    post_list = []
    prior_list = []
    keff_post = []
    keff_prior = []

    Kmax = 0
    for t in range(n_timesteps):
        data = extract_latents_and_assignments(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
            t_select=t,
            max_points=200000,
            collect_exemplars=False,
        )

        post_pi = np.asarray(data["posterior_pi"], dtype=np.float64)   # [N,K]
        prior_pi = np.asarray(data["prior_pi"], dtype=np.float64)      # [N,K]

        if post_pi.ndim != 2 or prior_pi.ndim != 2 or post_pi.shape[0] == 0:
            post_mean = np.zeros((1,), dtype=np.float64)
            prior_mean = np.zeros((1,), dtype=np.float64)
            kpost = 0.0
            kprior = 0.0
        else:
            post_mean = post_pi.mean(axis=0)
            prior_mean = prior_pi.mean(axis=0)
            kpost = float(_effective_k(post_pi).mean())
            kprior = float(_effective_k(prior_pi).mean())

        Kmax = max(Kmax, post_mean.shape[0], prior_mean.shape[0])
        post_list.append(post_mean)
        prior_list.append(prior_mean)
        keff_post.append(kpost)
        keff_prior.append(kprior)

    post_mat = np.zeros((n_timesteps, Kmax), dtype=np.float64)
    prior_mat = np.zeros((n_timesteps, Kmax), dtype=np.float64)

    for t in range(n_timesteps):
        post_mat[t, :post_list[t].shape[0]] = post_list[t]
        prior_mat[t, :prior_list[t].shape[0]] = prior_list[t]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    im0 = axes[0].imshow(post_mat, aspect="auto")
    axes[0].set_title("Mean posterior soft responsibility over time")
    axes[0].set_xlabel("component index")
    axes[0].set_ylabel("timestep")
    plt.colorbar(im0, ax=axes[0], label="mean posterior responsibility")

    im1 = axes[1].imshow(prior_mat, aspect="auto")
    axes[1].set_title("Mean prior soft mixture weights over time")
    axes[1].set_xlabel("component index")
    axes[1].set_ylabel("timestep")
    plt.colorbar(im1, ax=axes[1], label="mean prior weight")

    axes[2].plot(np.arange(n_timesteps), keff_post, label="posterior K_eff")
    axes[2].plot(np.arange(n_timesteps), keff_prior, label="prior K_eff")
    axes[2].set_title("Effective number of active components over time")
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("exp(H)")
    axes[2].legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig