
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


# -----------------------------
# t-SNE
# -----------------------------
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
    Returns:
        emb: [M, n_components]
        idx: [M] indices of original points used (downsampled if needed)
    """
    rng = np.random.default_rng(random_state)
    n = latents.shape[0]

    if n == 0:
        return np.zeros((0, n_components)), np.zeros((0,), dtype=np.int64)

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)

    X = latents[idx].astype(np.float32, copy=False)

    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd

    # PCA pre-reduction
    if pca_dims is not None and pca_dims > 0 and X.shape[1] > pca_dims:
        pca_dims_eff = int(min(pca_dims, X.shape[1], max(2, X.shape[0] - 1)))
        X = PCA(n_components=pca_dims_eff, random_state=random_state).fit_transform(X)

    try:
        tsne = TSNE(
            n_components=int(n_components),
            perplexity=float(min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))),
            init="pca" if X.shape[0] > 10 else "random",
            learning_rate="auto",
            random_state=int(random_state),
            max_iter=1000,   # newer sklearn
        )
    except TypeError:
        # older sklearn
        tsne = TSNE(
            n_components=int(n_components),
            perplexity=float(min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))),
            init="pca" if X.shape[0] > 10 else "random",
            random_state=int(random_state),
            n_iter=1000,     # older sklearn
        )
    emb = tsne.fit_transform(X)
    return emb, idx


# -----------------------------
# RNN context helper (as in your original file)
# -----------------------------
@torch.no_grad()
def _compute_rnn_context(model, batch: Dict, device: torch.device, t_select: int) -> torch.Tensor:
    obs = batch["observations"].to(device)  # expect [B,T,C,H,W]
    if obs.dim() != 5:
        B = obs.shape[0]
        return model.h0[-1].expand(B, -1).to(device)

    actions = batch.get("actions", None)
    dones = batch.get("done", None)
    if actions is not None:
        actions = actions.to(device)
    if dones is not None:
        dones = dones.to(device)

    B, T = obs.shape[:2]
    t_select = int(max(0, min(int(t_select), T - 1)))

    h = model.h0.expand(model.number_lstm_layer, B, -1).contiguous()
    c = model.c0.expand(model.number_lstm_layer, B, -1).contiguous()

    for t in range(t_select):
        x_t = _to_minus1_1(obs[:, t])               # [B,C,H,W]
        x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

        h_context = h[-1]                           # [B,H]
        vdvae_out = model.vdvae(x_t_nhwc, x_t_nhwc, h_context)
        top_q_mean_map = vdvae_out["top_q_mean_map"]  # [B,C_top,res,res]
        z_t = top_q_mean_map.mean(dim=(2, 3))         # [B,C_top]

        if actions is not None and actions.dim() >= 2 and actions.shape[1] > t:
            a_t = actions[:, t]
        else:
            a_t = torch.zeros(B, model.action_dim, device=device)

        if dones is not None and dones.dim() >= 2 and t > 0:
            mask_t = 1.0 - dones[:, t - 1].float()
        else:
            mask_t = torch.ones(B, device=device)

        rnn_in = torch.cat([z_t, a_t], dim=-1)
        _, (h, c) = model._rnn(rnn_in, h, c, mask_t)

    return h[-1]  # [B,H]


# -----------------------------
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

    # Model dims
    top_block = model.vdvae.decoder.dec_blocks[0]
    C_top = int(top_block.zdim)
    res = int(top_block.base)

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

        # RNN context
        if use_rnn_context and isinstance(batch, dict) and "observations" in batch and batch["observations"].dim() == 5:
            h_context = _compute_rnn_context(model, batch, device, t_select)
        else:
            h_context = model.h0[-1].expand(B, -1).to(device)

        # VDVAE encoder (top layer)
        vdvae_out = model.vdvae(images_nhwc, images_nhwc, h_context)
        top_q_mean_map = vdvae_out["top_q_mean_map"]  # [B, C_top, res, res]

        # tokens: [B*res*res, C_top]
        z_tokens = top_q_mean_map.permute(0, 2, 3, 1).reshape(B * res * res, C_top)

        # h_tokens with coord encoding
        Hc = int(h_context.shape[1])
        h_map = h_context.view(B, Hc, 1, 1).expand(B, Hc, res, res).contiguous()
        h_map = model.vdvae.add_coord_no_proj(h_map, scale=0.05)
        h_tokens = h_map.permute(0, 2, 3, 1).reshape(B * res * res, Hc)

        # DPGMM prior params
        _, prior_params = model.prior(h_tokens)
        pi_tok = prior_params["pi"]                # [N,K]
        means = prior_params["means"]              # [N,K,C_top]
        log_vars = prior_params["log_vars"]        # [N,K,C_top]
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
        out["latents"] = np.zeros((0, C_top), dtype=np.float32)
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


# -----------------------------
# Main visualization
# -----------------------------
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
      (5) exemplar grid (most-used components)
      (6) metrics summary

    Note: `tsne_dims=3` gives 3D Matplotlib scatter; for nicer 3D, pass `save_html_path`.
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

    latents = data["latents"]
    assignments = data["assignments"].astype(np.int64)
    pi = data["pi"].astype(np.float32)
    labels = data.get("labels", None)

    if latents.shape[0] < 5:
        raise ValueError("Not enough points for t-SNE. Increase max_batches or ensure dataloader yields data.")

    # t-SNE (2D or 3D)
    tsne_dims = int(2 if tsne_dims is None else tsne_dims)
    tsne_dims = 3 if tsne_dims == 3 else 2
    emb, idx = compute_tsne_embedding(
        latents,
        max_samples=max_samples,
        perplexity=perplexity,
        pca_dims=50,
        random_state=42,
        n_components=tsne_dims,
        standardize=True,
    )

    assign_s = assignments[idx]
    pi_s = pi[idx]
    labels_s = labels[idx] if labels is not None else None

    n_components = pi_s.shape[-1]

    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, max(1, n_components))))
    if n_components > 20:
        colors = plt.cm.turbo(np.linspace(0, 1, n_components))
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=figsize)

    # (1) t-SNE by component
    if tsne_dims == 3:
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
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
    else:
        ax2 = fig.add_subplot(2, 3, 2)

    if labels_s is not None:
        if tsne_dims == 3:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=labels_s, cmap="Set1", s=6, alpha=0.75)
            ax2.set_zlabel("t-SNE 3")
        else:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=labels_s, cmap="Set1", s=8, alpha=0.75)
        cbar = plt.colorbar(sc2, ax=ax2, label="Ground Truth Label")
        if class_names is not None:
            uniq = np.unique(labels_s)
            cbar.set_ticks(uniq)
            cbar.set_ticklabels([class_names[int(u)] for u in uniq])
        ax2.set_title("t-SNE by Ground Truth Label", fontsize=11)
    else:
        # Confidence: pi at assigned component
        conf = pi_s[np.arange(pi_s.shape[0]), assign_s]
        if tsne_dims == 3:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=conf, cmap="viridis", s=6, alpha=0.75)
            ax2.set_zlabel("t-SNE 3")
        else:
            sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=conf, cmap="viridis", s=8, alpha=0.75)
        plt.colorbar(sc2, ax=ax2, label="assignment confidence")
        ax2.set_title("t-SNE colored by assignment confidence", fontsize=11)

    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")

    # (3) Component utilization
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
    ax4.hist(conf_all, bins=30, alpha=0.75, label="confidence (pi@assigned)")
    ax4.hist(effk_all, bins=30, alpha=0.75, label="effective-K = exp(H(pi))")
    ax4.set_title("Mixture decisiveness diagnostics", fontsize=11)
    ax4.set_xlabel("value")
    ax4.set_ylabel("count")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    # (5) Exemplars (top-used components)
    ax5 = fig.add_subplot(2, 3, 5)
    ex_imgs = data.get("exemplar_images", None)
    ex_sco = data.get("exemplar_scores", None)
    if ex_imgs is not None and ex_imgs.size > 0:
        # choose top components by utilization
        topk = min(8, len(order))
        show_components = order[:topk]
        images_for_grid: List[np.ndarray] = []
        captions: List[str] = []
        for kk in show_components:
            # add up to 6 exemplars
            for j in range(min(6, ex_imgs.shape[1])):
                im_t = torch.from_numpy(ex_imgs[kk, j])
                if torch.isnan(torch.tensor(ex_sco[kk, j])).item():
                    continue
                im = _to_0_1_for_plot(im_t)
                images_for_grid.append(im)
                captions.append(f"k={kk}")
        if len(images_for_grid) == 0:
            ax5.text(0.5, 0.5, "No exemplars stored", ha="center", va="center", transform=ax5.transAxes)
        else:
            grid = _make_grid(images_for_grid, ncols=6, pad=2)
            ax5.imshow(grid)
            ax5.set_title("Exemplar images (top-used components)", fontsize=11)
        ax5.axis("off")
    else:
        ax5.text(0.5, 0.5, "Exemplars not available", ha="center", va="center", transform=ax5.transAxes, fontsize=12)
        ax5.axis("off")
        ax5.set_title("Exemplars", fontsize=11)

    # (6) Metrics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    active_thresh = 0.01  # 1% usage
    k_active = int((frac > active_thresh).sum())
    util_entropy = float(_entropy(frac[frac > 0])[()] / math.log(max(2, (frac > 0).sum()))) if (frac > 0).sum() > 1 else 0.0

    metrics_text = [
        f"Points (used for t-SNE): {emb.shape[0]} / total: {latents.shape[0]}",
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

    ax6.text(0.05, 0.95, "Clustering diagnostics", fontsize=14, fontweight="bold", va="top", transform=ax6.transAxes)
    ax6.text(0.05, 0.85, "\n".join(metrics_text), fontsize=10.5, family="monospace", va="top", transform=ax6.transAxes)

    title = f"DPGMM Latent Space (image_level={image_level}, t={t_select}, tsne_dims={tsne_dims})"
    if nmi is not None:
        title += f" | NMI={nmi:.3f}"
    fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")



    # Avoid leaking figures in long-running training jobs (common when using Agg).
    try:
        if save_path is not None and "agg" in matplotlib.get_backend().lower():
            plt.close(fig)
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