from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP 
from vdvae.top_dpgmm_prior import compute_slot_kl_conditional_frozen,  sample_slots_conditional_frozen, sample_slot_vectors_conditional_frozen


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class VizConfig:
    t_select: int = 8
    max_batches: int = 5
    max_points: int = 3000          # total slot tokens extracted from loader
    max_embed_points: int = 2000    # total posterior+prior points embedded; comp means always kept
    n_example_frames: int = 3
    embed_method: str = "umap"      # "umap" | "tsne" | "pca"
    perplexity: float = 30.0
    random_state: int = 42
    point_size: float = 10.0
    dpi: int = 150
    eps: float = 1e-6
    active_component_threshold: float = 0.01
    show_kumaraswamy: bool = True


_PALETTE = plt.cm.tab10.colors


def _discrete_cmap(n: int):
    n = max(1, int(n))
    cols = [_PALETTE[i % len(_PALETTE)] for i in range(n)]
    return matplotlib.colors.ListedColormap(cols), BoundaryNorm(
        np.arange(-0.5, n + 0.5, 1.0), n
    )


# --------------------------------------------------------------------------- #
# Style helpers
# --------------------------------------------------------------------------- #
def _style_axis(ax, *, title: str = "", xlabel: str = "", ylabel: str = "", grid: bool = False) -> None:
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
    if grid:
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)


def _blank(ax, msg: str = "") -> None:
    ax.axis("off")
    if msg:
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8, color="0.35", transform=ax.transAxes)


# --------------------------------------------------------------------------- #
# Tensor/image helpers
# --------------------------------------------------------------------------- #
def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 [0,255], float [0,1], or already [-1,1] to float [-1,1]."""
    if x.dtype == torch.uint8:
        return x.float() / 127.5 - 1.0
    x = x.float()
    if x.numel() > 0:
        mx = float(x.max().detach().cpu().item())
        mn = float(x.min().detach().cpu().item())
        if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
            x = x * 2.0 - 1.0
    return x


def _chw_to_01(x: torch.Tensor) -> np.ndarray:
    """CHW tensor in [-1,1] or [0,1] -> HWC float array in [0,1]."""
    x = x.detach().float().cpu()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        # already HWC
        arr = x.clamp(0, 1).numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr
    if x.numel() and float(x.min()) < -0.01:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    arr = x.permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.float32, copy=False)


def _mask_to_01(mask: torch.Tensor) -> np.ndarray:
    m = mask.detach().float().cpu()
    if m.dim() == 4:
        m = m[0]
    if m.dim() == 3:
        m = m.squeeze(0)
    return m.clamp(0.0, 1.0).numpy().astype(np.float32, copy=False)


def _overlay(frame_chw: torch.Tensor, mask_1hw: torch.Tensor, alpha: float = 0.62) -> np.ndarray:
    """Show slot responsibility by dimming the image outside the mask."""
    img = _chw_to_01(frame_chw)
    m = _mask_to_01(mask_1hw)
    if m.shape != img.shape[:2]:
        mt = torch.from_numpy(m)[None, None].float()
        m = F.interpolate(mt, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    m = np.clip(m, 0.0, 1.0)[..., None]
    dimmed = img * (1.0 - alpha)
    highlighted = img * (0.25 + 0.75 * m)
    return np.clip(dimmed + alpha * highlighted, 0.0, 1.0)


def _apply_batch_mask(x: Optional[torch.Tensor], mask: torch.Tensor, B: int) -> Optional[torch.Tensor]:
    """Apply an image-level mask to tensors whose batch is B or B*S."""
    if x is None:
        return None
    if x.shape[0] == B:
        view = (B,) + (1,) * (x.dim() - 1)
        return x * mask.view(view).to(device=x.device, dtype=x.dtype)
    if x.shape[0] % B == 0:
        rep = x.shape[0] // B
        expanded = mask.repeat_interleave(rep, dim=0)
        view = (x.shape[0],) + (1,) * (x.dim() - 1)
        return x * expanded.view(view).to(device=x.device, dtype=x.dtype)
    raise ValueError(f"Cannot apply [B] mask with B={B} to tensor batch {x.shape[0]}")


# --------------------------------------------------------------------------- #
# Mask helpers (attention vs decoder alpha)
# --------------------------------------------------------------------------- #
def _attn_to_grid(top_slot_attn: torch.Tensor, top_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    """[B,S,L] slot-attention -> [B,S,h,w] using the TRUE top grid (top_H, top_W).

    Returns None if L != h*w so the caller can fall back gracefully instead of
    guessing a square grid.
    """
    B, S, L = top_slot_attn.shape
    h, w = int(top_hw[0]), int(top_hw[1])
    if h * w != L:
        return None
    return top_slot_attn.reshape(B, S, h, w).float()


def _peak_entropy_native(masks_bshw: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-image sharpness of a per-pixel distribution over the slot axis.

    masks_bshw: [B,S,H,W], summing to ~1 over dim=1.
    Returns (peak, ent), each [B]:
        peak = mean over pixels of max_slot p(slot|pixel)   (1/S collapsed -> 1 sharp)
        ent  = mean over pixels of -sum_slot p log p        (log S collapsed -> 0 sharp)

    IMPORTANT: pass NATIVE-resolution masks. Bilinear upsampling smooths the
    distribution and biases peak down / entropy up, and it penalizes the coarser
    attention grid more than the decoder grid, which corrupts the comparison.
    """
    p = masks_bshw.float()
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    peak = p.max(dim=1).values.mean(dim=(-1, -2))                                  # [B]
    ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1).mean(dim=(-1, -2))  # [B]
    return peak, ent


def _upsample_masks(masks_bshw: torch.Tensor, size: Tuple[int, int], renorm: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """[B,S,h,w] -> [B,S,1,H,W] bilinear, renormalized over slots for display only."""
    B, S, h, w = masks_bshw.shape
    up = F.interpolate(masks_bshw.reshape(B * S, 1, h, w), size=size, mode="bilinear", align_corners=False)
    up = up.reshape(B, S, 1, *size).clamp_min(0.0)
    if renorm:
        up = up / up.sum(dim=1, keepdim=True).clamp_min(eps)
    return up


def _argmax_overlay(frame_chw: torch.Tensor, masks_native_shw: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Hard slot segmentation overlay.

    frame_chw:        [C,H,W]
    masks_native_shw: [S,h,w] at the NATIVE grid. We argmax at native resolution
    and upsample the integer label map with nearest, so segment boundaries reflect
    the real assignment rather than interpolation artifacts.
    """
    img = _chw_to_01(frame_chw)
    H, W = img.shape[:2]
    seg_native = masks_native_shw.argmax(dim=0)[None, None].float()  # [1,1,h,w]
    seg = F.interpolate(seg_native, size=(H, W), mode="nearest")[0, 0].long().cpu().numpy()
    S = int(masks_native_shw.shape[0])
    colors = np.asarray([_PALETTE[i % len(_PALETTE)] for i in range(S)], dtype=np.float32)
    seg_rgb = colors[seg]
    return np.clip((1.0 - alpha) * img + alpha * seg_rgb, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Embedding / metrics
# --------------------------------------------------------------------------- #
def _standardize_pca(X: np.ndarray, dims: int = 50) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        X = X[finite]
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    d = min(int(dims), Xs.shape[1], max(2, Xs.shape[0] - 1))
    if Xs.shape[1] > d and Xs.shape[0] > 3:
        Xs = PCA(n_components=d, random_state=0).fit_transform(Xs)
    return Xs.astype(np.float32, copy=False)


def _embed_preselected(X: np.ndarray, cfg: VizConfig) -> Tuple[np.ndarray, str]:
    Xs = _standardize_pca(X)
    if Xs.shape[0] < 3:
        emb = np.zeros((Xs.shape[0], 2), dtype=np.float32)
        if Xs.shape[0] and Xs.shape[1] >= 1:
            emb[:, : min(2, Xs.shape[1])] = Xs[:, : min(2, Xs.shape[1])]
        return emb, "PCA"

    method = cfg.embed_method.lower()
    if method == "umap" and UMAP is not None and Xs.shape[0] >= 5:
        emb = UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, Xs.shape[0] - 1)),
            min_dist=0.1,
            random_state=cfg.random_state,
        ).fit_transform(Xs)
        return emb.astype(np.float32), "UMAP"

    if method in ("umap", "tsne") and Xs.shape[0] >= 5:
        perp = min(float(cfg.perplexity), float(Xs.shape[0] - 1) - 1e-3, max(2.0, (Xs.shape[0] - 1) / 3.0))
        perp = max(2.0, perp)
        kwargs = dict(n_components=2, perplexity=perp, random_state=cfg.random_state, init="pca")
        try:
            tsne = TSNE(**kwargs, learning_rate="auto", max_iter=1000)
        except TypeError:
            tsne = TSNE(**kwargs, n_iter=1000)
        return tsne.fit_transform(Xs).astype(np.float32), "t-SNE"

    n_comp = min(2, Xs.shape[1], Xs.shape[0] - 1)
    emb = PCA(n_components=n_comp, random_state=0).fit_transform(Xs)
    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])))
    return emb.astype(np.float32), "PCA"


def joint_embed_with_component_means(
    post_latents: np.ndarray,
    prior_latents: np.ndarray,
    comp_mean: np.ndarray,
    cfg: VizConfig,
) -> Dict[str, Any]:
    """One coordinate system for posterior, prior, and component means."""
    rng = np.random.default_rng(cfg.random_state)
    n_post = post_latents.shape[0]
    n_prior = prior_latents.shape[0]
    n_mean = comp_mean.shape[0]

    budget = max(10, int(cfg.max_embed_points))
    # Split budget roughly between posterior and prior, always keep component means.
    post_budget = min(n_post, max(1, budget // 2))
    prior_budget = min(n_prior, max(1, budget - post_budget))
    post_idx = rng.choice(n_post, post_budget, replace=False) if n_post > post_budget else np.arange(n_post)
    prior_idx = rng.choice(n_prior, prior_budget, replace=False) if n_prior > prior_budget else np.arange(n_prior)

    X = np.concatenate([post_latents[post_idx], prior_latents[prior_idx], comp_mean], axis=0).astype(np.float32)
    src = np.concatenate([
        np.zeros(len(post_idx), dtype=np.int64),
        np.ones(len(prior_idx), dtype=np.int64),
        np.full(n_mean, 2, dtype=np.int64),
    ])
    emb, method = _embed_preselected(X, cfg)
    return {"emb": emb, "src": src, "post_idx": post_idx, "prior_idx": prior_idx, "method": method}


def slot_component_contingency(slot_ids: np.ndarray, comp_ids: np.ndarray, n_slots: int, n_comp: int) -> np.ndarray:
    M = np.zeros((int(n_slots), int(n_comp)), dtype=np.float64)
    for s, k in zip(slot_ids.reshape(-1), comp_ids.reshape(-1)):
        if 0 <= s < n_slots and 0 <= k < n_comp:
            M[int(s), int(k)] += 1.0
    row = M.sum(1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def mapping_scores(slot_ids: np.ndarray, comp_ids: np.ndarray) -> Dict[str, float]:
    if slot_ids.size < 2 or np.unique(comp_ids).size < 2 or np.unique(slot_ids).size < 2:
        return {"ari": float("nan"), "nmi": float("nan")}
    return {
        "ari": float(adjusted_rand_score(slot_ids, comp_ids)),
        "nmi": float(normalized_mutual_info_score(slot_ids, comp_ids)),
    }


def _slot_centroid_cosine(latents: np.ndarray, slot_ids: np.ndarray, n_slots: int) -> np.ndarray:
    cent = np.zeros((int(n_slots), latents.shape[1]), dtype=np.float64)
    for s in range(int(n_slots)):
        m = slot_ids == s
        if np.any(m):
            cent[s] = latents[m].mean(0)
    norm = np.linalg.norm(cent, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (cent / norm) @ (cent / norm).T


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    return -(p * np.log(p)).sum(axis=-1)


def _effective_k(pi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.exp(_entropy(pi, eps=eps))


def compute_responsibilities(
    z_tokens: torch.Tensor,
    pi: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Responsibilities under pi_k(h) N(z | mu_k, diag_var_k)."""
    var = torch.exp(log_vars).clamp_min(eps)
    diff2 = (z_tokens[:, None, :] - means) ** 2
    log_gauss = -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + diff2 / var).sum(dim=-1)
    log_pi = torch.log(pi.clamp_min(eps))
    resp = torch.softmax(log_pi + log_gauss, dim=-1)
    return resp, resp.argmax(dim=-1)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass
class SlotDiagnostics:
    post_latents: np.ndarray
    post_comp: np.ndarray
    post_pi: np.ndarray
    post_slot_id: np.ndarray
    prior_latents: np.ndarray
    prior_comp: np.ndarray
    prior_pi: np.ndarray
    prior_slot_id: np.ndarray
    comp_mean: np.ndarray
    comp_var: np.ndarray
    comp_usage_post: np.ndarray
    comp_usage_prior: np.ndarray
    n_slots: int
    n_comp: int
    pi_per_image: np.ndarray
    kumar_a: np.ndarray
    kumar_b: np.ndarray
    slot_mask_mass: np.ndarray
    slot_mask_entropy: np.ndarray
    attn_peak: np.ndarray
    attn_entropy: np.ndarray
    dec_peak: np.ndarray
    dec_entropy: np.ndarray
    examples: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)


def _get_vdvae_out_at_t(model, batch: Dict[str, Any], device: torch.device, t_select: int):
    """Follow the real recurrent path up to t_select.

    This is the important part: h_t is produced from VMRNNCore state and passed
    as h_prior_top into VDVAE.forward. Do not replace this with h_t=None.
    """
    obs = batch["observations"].to(device)
    if obs.dim() == 6:  # [B,T,V,C,H,W] -> use first view
        obs = obs[:, :, 0]
    obs = _to_minus1_1(obs)
    if obs.dim() == 4:
        obs = obs[:, None]
    if obs.dim() != 5:
        raise ValueError(f"Expected observations [B,T,C,H,W] or [B,C,H,W], got {tuple(obs.shape)}")

    actions = batch.get("actions", None)
    if actions is not None:
        actions = actions.to(device)
        if actions.dim() == 2:
            actions = actions[:, None]

    dones = batch.get("done", batch.get("dones", None))
    if dones is not None:
        dones = dones.to(device)
        if dones.dim() == 1:
            dones = dones[:, None]

    B, T, _, _, _ = obs.shape
    t_select = int(max(0, min(int(t_select), T - 1)))
    param_dtype = next(model.parameters()).dtype
    if param_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        param_dtype = torch.float32

    def zero_action() -> torch.Tensor:
        return torch.zeros(B, int(model.action_dim), device=device, dtype=param_dtype)

    core_state = model.rnn.init_state(B, device=device, dtype=param_dtype)
    prev_latents = model._init_decoder_prev_latents(B, device, param_dtype)
    last = None

    for t in range(t_select + 1):
        x_t = obs[:, t].to(dtype=param_dtype)

        if dones is None or t == 0:
            mask_t = torch.ones(B, device=device, dtype=torch.float32)
        else:
            mask_t = (1.0 - dones[:, t - 1].float()).to(device=device, dtype=torch.float32)

        keep_tok = mask_t.view(B, 1, 1).to(device=device, dtype=param_dtype)
        h, c = core_state
        h0, c0 = model.rnn.init_state(B, device=device, dtype=param_dtype)
        core_state = (h * keep_tok + h0 * (1.0 - keep_tok), c * keep_tok + c0 * (1.0 - keep_tok))
        prev_latents = [_apply_batch_mask(pl, mask_t, B) for pl in prev_latents]

        h_tok = model.rnn.out_norm(core_state[0])
        B_, L_, D_ = h_tok.shape
        expected_L = int(model.top_H) * int(model.top_W)
        if L_ != expected_L:
            raise ValueError(f"VMRNN token state length {L_} != top_H*top_W={expected_L}")
        h_t = h_tok.transpose(1, 2).reshape(B_, D_, int(model.top_H), int(model.top_W)).contiguous()

        prev_latents_in = [None if pl is None else pl.detach().clone() for pl in prev_latents]
        x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

        vdvae_out = model.vdvae.forward(
            x_t_nhwc,
            x_t_nhwc,
            h_prior_top=h_t,
            mask_t=mask_t,
            prev_latents=prev_latents,
            get_latents=True,
        )

        z_top_state = vdvae_out.get("z_top_state", vdvae_out["current_latents"][0])
        a_cur = zero_action() if actions is None else actions[:, t].to(device=device, dtype=param_dtype)
        _, core_state = model.rnn(z_top_state, a_cur, state=core_state, mask_t=None, extra_maps=None)

        prev_latents = [None if idx == 0 else z.detach() for idx, z in enumerate(vdvae_out["current_latents"])]
        last = (vdvae_out, x_t.detach(), h_t.detach(), prev_latents_in)

    if last is None:
        raise RuntimeError("Could not extract a VDVAE output from the batch.")
    return last


@torch.no_grad()
def extract_slot_diagnostics(model, dataloader, device: torch.device, cfg: VizConfig) -> SlotDiagnostics:
    model.eval()
    snap = model.vdvae.top_prior_snapshot
    gate = model.vdvae.top_prior_gate
    if snap is None or gate is None:
        raise ValueError("Frozen top prior is not set. Run refresh_top_prior_from_buffer() or bootstrap it before visualizing.")

    K = int(snap.K)
    post_z, post_pi, post_k, post_sid = [], [], [], []
    prior_z, prior_pi, prior_k, prior_sid = [], [], [], []
    pi_img_all, kumar_a_all, kumar_b_all = [], [], []
    mask_mass_all, mask_entropy_all = [], []
    attn_peak_all, attn_ent_all, dec_peak_all, dec_ent_all = [], [], [], []
    examples: List[Dict[str, Any]] = []
    n_slots: Optional[int] = None
    total = 0

    for bi, batch in enumerate(tqdm(dataloader, desc="slot/DPGMM diagnostics", leave=False)):
        if bi >= int(cfg.max_batches) or total >= int(cfg.max_points):
            break
        if not isinstance(batch, dict) or "observations" not in batch:
            continue

        out, x_t, h_t, _prev = _get_vdvae_out_at_t(model, batch, device, cfg.t_select)
        slot_mu = out.get("top_slot_mu", None)
        slot_ls = out.get("top_slot_logsigma", None)
        if slot_mu is None or slot_ls is None:
            continue

        B, S, D = slot_mu.shape
        if n_slots is None:
            n_slots = int(S)
        elif int(S) != int(n_slots):
            raise ValueError(f"Number of slots changed across batches: first {n_slots}, now {S}")

        px_z = out["px_z"]
        N, Cpx, Hc, Wc = px_z.shape
        if N % B != 0:
            raise ValueError(f"Expected px_z batch B*S, got N={N}, B={B}")
        S_from_px = N // B
        if S_from_px != S:
            raise ValueError(f"top_slot_mu has S={S}, but px_z implies S={S_from_px}")

        width = int(model.vdvae.H.width)
        mask_logits = px_z[:, width:width + 1]
        masks = F.softmax(mask_logits.view(B, S, 1, Hc, Wc), dim=1)

        # posterior responsibilities under the frozen conditional DPGMM
        _, resp = compute_slot_kl_conditional_frozen(
            snapshot=snap,
            frozen_gate=gate,
            h_t=h_t,
            slot_mu=slot_mu,
            slot_logsigma=slot_ls,
        )
        k_post = resp.argmax(-1)
        sid = torch.arange(S, device=device, dtype=torch.long).view(1, S).expand(B, S).reshape(B * S)

        post_z.append(slot_mu.reshape(B * S, D).detach().cpu())
        post_pi.append(resp.reshape(B * S, K).detach().cpu())
        post_k.append(k_post.reshape(B * S).detach().cpu())
        post_sid.append(sid.detach().cpu())

        # conditional prior samples + full responsibility assignment
        pri,  prior_idx, log_pi_slot = sample_slot_vectors_conditional_frozen(snap, gate, h_t, num_slots=S, assignment_temperature=1.0, slot_temperature=1.0)
        pri_flat = pri.reshape(B * S, D)

        pi_img = gate.mean_pi(h_t)
        pi_img_all.append(log_pi_slot.exp().mean(dim=1).detach().cpu())
        pi_slot = log_pi_slot.exp().reshape(B * S, K)  # [B,S,K] -> [B,S,K] average over slot dim to get image-level pi TODO:check this??

        means = snap.comp_mean.to(device=device, dtype=pri_flat.dtype).view(K, -1)
        log_vars = torch.log(snap.comp_var.to(device=device, dtype=pri_flat.dtype).view(K, -1).clamp_min(1e-6))
        means_b = means[None].expand(B * S, K, D)
        log_vars_b = log_vars[None].expand(B * S, K, D)
        pri_resp, k_prior = compute_responsibilities(pri_flat, pi_slot, means_b, log_vars_b, eps=cfg.eps)

        prior_z.append(pri_flat.detach().cpu())
        prior_pi.append(pri_resp.detach().cpu())
        prior_k.append(prior_idx.reshape(B * S).detach().cpu())
        prior_sid.append(sid.detach().cpu())

        # Gate params for optional diagnostic.
        try:
            gate_out = gate(h_t)
            kumar_a_all.append(gate_out.get("a", torch.empty(B, 0, device=device)).detach().cpu())
            kumar_b_all.append(gate_out.get("b", torch.empty(B, 0, device=device)).detach().cpu())
        except Exception:
            pass

        # --- mask sharpness on NATIVE resolution (no upsampling bias) ---
        # decoder alpha masks: [B,S,1,Hc,Wc] -> [B,S,Hc,Wc]
        dec_native = masks.squeeze(2)
        dec_peak_b, dec_ent_b = _peak_entropy_native(dec_native, eps=cfg.eps)

        # slot-attention masks: [B,S,L] -> [B,S,top_H,top_W] using the TRUE grid
        top_attn = out.get("top_slot_attn", None)
        attn_native = _attn_to_grid(top_attn, (int(model.top_H), int(model.top_W))) if top_attn is not None else None
        if attn_native is not None:
            attn_peak_b, attn_ent_b = _peak_entropy_native(attn_native, eps=cfg.eps)
        else:
            attn_peak_b = torch.full((B,), float("nan"))
            attn_ent_b = torch.full((B,), float("nan"))

        attn_peak_all.append(attn_peak_b.detach().cpu())
        attn_ent_all.append(attn_ent_b.detach().cpu())
        dec_peak_all.append(dec_peak_b.detach().cpu())
        dec_ent_all.append(dec_ent_b.detach().cpu())

        # legacy fields kept for the external dict contract (not plotted)
        mass = dec_native.mean(dim=(2, 3))                       # [B,S]
        mask_mass_all.append(mass.detach().cpu())
        mask_entropy_all.append(dec_ent_b[:, None].expand(B, S).detach().cpu())

        if len(examples) < int(cfg.n_example_frames):
            take = min(B, int(cfg.n_example_frames) - len(examples))
            conf = resp.gather(-1, k_post[..., None]).squeeze(-1)
            for b in range(take):
                ex = {
                    "frame": x_t[b].detach().cpu(),
                    "decoder_masks_native": dec_native[b].detach().cpu(),   # [S,Hc,Wc]
                    "attn_masks_native": (attn_native[b].detach().cpu() if attn_native is not None else None),
                    "comp": k_post[b].detach().cpu(),
                    "conf": conf[b].detach().cpu(),
                    "decoder_peak": float(dec_peak_b[b]),
                    "decoder_entropy": float(dec_ent_b[b]),
                    "attn_peak": float(attn_peak_b[b]),
                    "attn_entropy": float(attn_ent_b[b]),
                    # legacy compat for any external reader of same_frame_examples:
                    "masks": _upsample_masks(dec_native[b:b + 1], x_t.shape[-2:])[0].detach().cpu(),
                    "mass": mass[b].detach().cpu(),
                }
                examples.append(ex)

        total += B * S

    if n_slots is None or not post_z or not prior_z:
        raise ValueError("No usable slot latents were extracted from the dataloader.")

    def cat(xs: List[torch.Tensor]) -> np.ndarray:
        return torch.cat(xs, dim=0).numpy() if xs else np.zeros((0,), dtype=np.float32)

    post_latents = cat(post_z).astype(np.float32)
    post_comp = cat(post_k).astype(np.int64)
    post_pi_arr = cat(post_pi).astype(np.float32)
    post_slot = cat(post_sid).astype(np.int64)
    prior_latents = cat(prior_z).astype(np.float32)
    prior_comp = cat(prior_k).astype(np.int64)
    prior_pi_arr = cat(prior_pi).astype(np.float32)
    prior_slot = cat(prior_sid).astype(np.int64)

    usage_post = np.bincount(post_comp.reshape(-1), minlength=K).astype(np.float64)
    usage_prior = np.bincount(prior_comp.reshape(-1), minlength=K).astype(np.float64)
    usage_post = usage_post / max(1.0, usage_post.sum())
    usage_prior = usage_prior / max(1.0, usage_prior.sum())

    pi_per_image = cat(pi_img_all).astype(np.float32) if pi_img_all else np.zeros((0, K), dtype=np.float32)
    kumar_a = cat(kumar_a_all).astype(np.float32) if kumar_a_all else np.zeros((0, max(0, K - 1)), dtype=np.float32)
    kumar_b = cat(kumar_b_all).astype(np.float32) if kumar_b_all else np.zeros((0, max(0, K - 1)), dtype=np.float32)
    mask_mass = cat(mask_mass_all).astype(np.float32) if mask_mass_all else np.zeros((0, int(n_slots)), dtype=np.float32)
    mask_entropy = cat(mask_entropy_all).astype(np.float32) if mask_entropy_all else np.zeros((0, int(n_slots)), dtype=np.float32)
    attn_peak = cat(attn_peak_all).astype(np.float32) if attn_peak_all else np.zeros((0,), dtype=np.float32)
    attn_entropy = cat(attn_ent_all).astype(np.float32) if attn_ent_all else np.zeros((0,), dtype=np.float32)
    dec_peak = cat(dec_peak_all).astype(np.float32) if dec_peak_all else np.zeros((0,), dtype=np.float32)
    dec_entropy = cat(dec_ent_all).astype(np.float32) if dec_ent_all else np.zeros((0,), dtype=np.float32)

    return SlotDiagnostics(
        post_latents=post_latents,
        post_comp=post_comp,
        post_pi=post_pi_arr,
        post_slot_id=post_slot,
        prior_latents=prior_latents,
        prior_comp=prior_comp,
        prior_pi=prior_pi_arr,
        prior_slot_id=prior_slot,
        comp_mean=snap.comp_mean.detach().cpu().numpy().reshape(K, -1).astype(np.float32),
        comp_var=snap.comp_var.detach().cpu().numpy().reshape(K, -1).astype(np.float32),
        comp_usage_post=usage_post,
        comp_usage_prior=usage_prior,
        n_slots=int(n_slots),
        n_comp=K,
        pi_per_image=pi_per_image,
        kumar_a=kumar_a,
        kumar_b=kumar_b,
        slot_mask_mass=mask_mass,
        slot_mask_entropy=mask_entropy,
        attn_peak=attn_peak,
        attn_entropy=attn_entropy,
        dec_peak=dec_peak,
        dec_entropy=dec_entropy,
        examples=examples,
        meta={"t_select": cfg.t_select, "points": int(post_latents.shape[0])},
    )


def _diag_to_legacy_dict(diag: SlotDiagnostics) -> Dict[str, Any]:
    return {
        "posterior_latents": diag.post_latents,
        "posterior_pi": diag.post_pi,
        "posterior_assignments": diag.post_comp,
        "posterior_slot_ids": diag.post_slot_id,
        "prior_latents": diag.prior_latents,
        "prior_pi": diag.prior_pi,
        "prior_assignments": diag.prior_comp,
        "prior_slot_ids": diag.prior_slot_id,
        "slot_mask_mass": diag.slot_mask_mass,
        "slot_mask_entropy": diag.slot_mask_entropy,
        "attn_peak": diag.attn_peak,
        "attn_entropy": diag.attn_entropy,
        "decoder_peak": diag.dec_peak,
        "decoder_entropy": diag.dec_entropy,
        "pi_per_image": diag.pi_per_image,
        "kumar_a": diag.kumar_a,
        "kumar_b": diag.kumar_b,
        "comp_mean": diag.comp_mean,
        "comp_var": diag.comp_var,
        "num_slots": diag.n_slots,
        "same_frame_examples": diag.examples,
    }


@torch.no_grad()
def extract_slot_latents_and_assignments(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 10,
    t_select: int = 8,
    max_points: int = 8000,
) -> Dict[str, Any]:
    cfg = VizConfig(t_select=t_select, max_batches=max_batches, max_points=max_points, max_embed_points=min(max_points, 4000))
    return _diag_to_legacy_dict(extract_slot_diagnostics(model, dataloader, device, cfg))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _scatter_discrete(ax, emb: np.ndarray, labels: np.ndarray, n_classes: int, title: str, legend_title: str, point_size: float) -> None:
    cmap, norm = _discrete_cmap(n_classes)
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap, norm=norm, s=point_size, linewidths=0, alpha=0.78)
    _style_axis(ax, title=title, xlabel="latent dim 1", ylabel="latent dim 2")
    if n_classes <= 12:
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=cmap(i), label=str(i)) for i in range(n_classes)]
        ax.legend(handles=handles, title=legend_title, fontsize=6.3, title_fontsize=7, loc="best", framealpha=0.82, borderpad=0.3, handletextpad=0.2, labelspacing=0.18)


def _plot_joint_prior_posterior(ax, emb: np.ndarray, src: np.ndarray, comp_for_means: np.ndarray, title: str, point_size: float) -> None:
    m_post = src == 0
    m_prior = src == 1
    m_mean = src == 2
    ax.scatter(emb[m_post, 0], emb[m_post, 1], s=point_size, alpha=0.42, linewidths=0, label="posterior slots")
    ax.scatter(emb[m_prior, 0], emb[m_prior, 1], s=point_size, alpha=0.42, linewidths=0, label="prior samples")
    if m_mean.any():
        # comp means are last K rows among src==2, label each with k
        ax.scatter(emb[m_mean, 0], emb[m_mean, 1], marker="X", s=95, edgecolor="black", linewidths=0.8, label="component means")
        coords = emb[m_mean]
        for k, (x, y) in zip(comp_for_means, coords):
            ax.text(x, y, str(int(k)), fontsize=7, ha="center", va="center", color="white")
    _style_axis(ax, title=title, xlabel="latent dim 1", ylabel="latent dim 2")
    ax.legend(fontsize=7, framealpha=0.86, loc="best")


def _plot_contingency(ax, M: np.ndarray, scores: Dict[str, float]) -> None:
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    _style_axis(ax, title="P(component | slot)\n(slot/component agreement)", xlabel="DPGMM component", ylabel="slot index")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    for s in range(M.shape[0]):
        for k in range(M.shape[1]):
            v = float(M[s, k])
            if v >= 0.04:
                ax.text(k, s, f"{v:.2f}", ha="center", va="center", fontsize=6.2, color="white" if v < 0.58 else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5)
    ari, nmi = scores["ari"], scores["nmi"]
    ax.text(0.0, 1.14, f"ARI={ari:.2f}   NMI={nmi:.2f}", transform=ax.transAxes, fontsize=8, color="0.2")


def _plot_slot_cosine(ax, C: np.ndarray) -> None:
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    _style_axis(ax, title="slot-centroid cosine\nhigh off-diagonal = slot collapse", xlabel="slot", ylabel="slot")
    ax.set_xticks(range(C.shape[0]))
    ax.set_yticks(range(C.shape[0]))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5)


def _plot_component_usage(ax, usage_post: np.ndarray, usage_prior: np.ndarray, threshold: float) -> None:
    k = np.arange(len(usage_post))
    w = 0.4
    ax.bar(k - w / 2, usage_post, width=w, label="posterior")
    ax.bar(k + w / 2, usage_prior, width=w, label="prior")
    ax.axhline(threshold, linestyle="--", linewidth=0.8, alpha=0.55)
    _style_axis(ax, title="component usage", xlabel="component", ylabel="fraction of slot tokens", grid=True)
    ax.set_xticks(k)
    ax.legend(fontsize=7, framealpha=0.85)


def _plot_entropy_hist(ax, diag: SlotDiagnostics) -> None:
    post_H = _entropy(diag.post_pi)
    prior_H = _entropy(diag.prior_pi)
    ax.hist(post_H, bins=30, alpha=0.6, label="posterior resp")
    ax.hist(prior_H, bins=30, alpha=0.45, label="prior resp")
    _style_axis(ax, title="responsibility entropy\nhigh = uncertain component", xlabel="entropy", ylabel="count", grid=True)
    ax.legend(fontsize=7, framealpha=0.85)


def _plot_gate_effective_k(ax, diag: SlotDiagnostics) -> None:
    if diag.pi_per_image.size == 0:
        _blank(ax, "no gate probabilities")
        return
    eff = _effective_k(diag.pi_per_image)
    ax.hist(eff, bins=20, alpha=0.75)
    _style_axis(ax, title="conditional gate effective K\nper image", xlabel="K_eff = exp(H(pi(h)))", ylabel="count", grid=True)
    ax.axvline(float(np.mean(eff)), linestyle="--", linewidth=1.0, alpha=0.8)


def _mask_means(diag: "SlotDiagnostics") -> Dict[str, float]:
    def _m(a):
        a = np.asarray(a, np.float32)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")
    return {
        "attn_peak": _m(diag.attn_peak), "attn_ent": _m(diag.attn_entropy),
        "dec_peak": _m(diag.dec_peak), "dec_ent": _m(diag.dec_entropy),
    }


def _mask_verdict(diag: "SlotDiagnostics") -> str:
    S = max(1, diag.n_slots)
    base = 1.0 / S
    thr = base + 0.5 * (1.0 - base)   # midpoint between uniform (1/S) and perfect (1)
    m = _mask_means(diag)
    ap, dp = m["attn_peak"], m["dec_peak"]
    if not np.isfinite(ap) and not np.isfinite(dp):
        return "masks: no mask signal available"
    if not np.isfinite(ap):
        return f"masks: attention unavailable; decoder peak={dp:.2f} (1/S={base:.2f})"
    attn_ok = ap > thr
    dec_ok = np.isfinite(dp) and dp > thr
    if attn_ok and dec_ok:
        return f"masks: decomposition works (attn peak={ap:.2f}, decoder peak={dp:.2f})"
    if attn_ok and not dec_ok:
        return (f"masks: attention segments (peak={ap:.2f}) but decoder alpha is ~uniform "
                f"(peak={dp:.2f} vs 1/S={base:.2f}) -> decoder ignores slot masks")
    if not attn_ok and not dec_ok:
        return (f"masks: both ~uniform (attn peak={ap:.2f}, decoder peak={dp:.2f} vs 1/S={base:.2f}) "
                f"-> slot attention itself collapsed")
    return f"masks: decoder peak={dp:.2f} above attention peak={ap:.2f} (unexpected)"


def _plot_mask_quality(ax, diag: "SlotDiagnostics") -> None:
    S = max(1, diag.n_slots)
    base_peak = 1.0 / S
    base_ent = math.log(S)
    m = _mask_means(diag)
    if not np.isfinite(m["attn_peak"]) and not np.isfinite(m["dec_peak"]):
        _blank(ax, "no mask statistics")
        return

    # both metrics on [0,1]: peak in [1/S,1]; entropy divided by log S in [0,1].
    def _en(x):
        return (x / base_ent) if (base_ent > 0 and np.isfinite(x)) else np.nan
    groups = ["peak\n(higher=sharper)", "entropy/logS\n(lower=sharper)"]
    attn_vals = [m["attn_peak"], _en(m["attn_ent"])]
    dec_vals = [m["dec_peak"], _en(m["dec_ent"])]
    base_vals = [base_peak, 1.0]

    x = np.arange(len(groups))
    w = 0.26
    ax.bar(x - w, attn_vals, width=w, label="slot attention")
    ax.bar(x, dec_vals, width=w, label="decoder alpha")
    ax.bar(x + w, base_vals, width=w, color="0.72", label="collapsed baseline")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7)
    _style_axis(ax, title="mask sharpness\nattention vs decoder vs collapse", ylabel="value", grid=True)
    ax.legend(fontsize=6.3, framealpha=0.85, loc="upper center", ncol=1)


def _plot_kumaraswamy(ax, diag: SlotDiagnostics) -> None:
    if diag.kumar_a.size == 0 or diag.kumar_b.size == 0 or diag.kumar_a.shape[1] == 0:
        _blank(ax, "no Kumaraswamy gate params")
        return
    m = min(8, diag.kumar_a.shape[1])
    vals = [diag.kumar_a[:, i] for i in range(m)] + [diag.kumar_b[:, i] for i in range(m)]
    labels = [f"a{i}" for i in range(m)] + [f"b{i}" for i in range(m)]
    ax.boxplot(vals, labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.5)
    _style_axis(ax, title="Kumaraswamy gate params", ylabel="value")


def _plot_mask_comparison_grid(fig: plt.Figure, gs_cell, examples: List[Dict[str, Any]], n_slots: int) -> None:
    """One row per example:
    input | attn argmax | decoder argmax | per-slot attention | per-slot decoder alpha.
    """
    n_rows = len(examples)
    if n_rows == 0:
        _blank(fig.add_subplot(gs_cell), "no example frames")
        return
    S = int(n_slots)
    cols = 3 + 2 * S
    sub = gs_cell.subgridspec(n_rows, cols, wspace=0.05, hspace=0.30)

    for r, ex in enumerate(examples):
        frame = ex["frame"]
        H, W = frame.shape[-2:]
        attn_n = ex.get("attn_masks_native", None)
        dec_n = ex.get("decoder_masks_native", None)
        attn_ok = attn_n is not None and torch.isfinite(attn_n).all()

        ax = fig.add_subplot(sub[r, 0])
        ax.imshow(_chw_to_01(frame))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"ex {r}", fontsize=8)
        if r == 0:
            ax.set_title("input", fontsize=8.5)

        ax = fig.add_subplot(sub[r, 1])
        ax.set_xticks([]); ax.set_yticks([])
        if attn_ok:
            ax.imshow(_argmax_overlay(frame, attn_n))
            ax.set_xlabel(f"peak={ex['attn_peak']:.2f} H={ex['attn_entropy']:.2f}", fontsize=5.5)
        else:
            _blank(ax, "no attn")
        if r == 0:
            ax.set_title("attn argmax", fontsize=8.5)

        ax = fig.add_subplot(sub[r, 2])
        ax.set_xticks([]); ax.set_yticks([])
        if dec_n is not None:
            ax.imshow(_argmax_overlay(frame, dec_n))
            ax.set_xlabel(f"peak={ex['decoder_peak']:.2f} H={ex['decoder_entropy']:.2f}", fontsize=5.5)
        else:
            _blank(ax, "no decoder")
        if r == 0:
            ax.set_title("decoder argmax", fontsize=8.5)

        attn_up = _upsample_masks(attn_n[None], (H, W))[0] if attn_ok else None
        for s in range(S):
            ax = fig.add_subplot(sub[r, 3 + s])
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(_chw_to_01(frame))
            if attn_up is not None:
                ax.imshow(attn_up[s, 0].numpy(), cmap="magma", alpha=0.6, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"attn s{s}", fontsize=6.5)

        dec_up = _upsample_masks(dec_n[None], (H, W))[0] if dec_n is not None else None
        for s in range(S):
            ax = fig.add_subplot(sub[r, 3 + S + s])
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(_chw_to_01(frame))
            if dec_up is not None:
                ax.imshow(dec_up[s, 0].numpy(), cmap="magma", alpha=0.6, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"dec s{s}", fontsize=6.5)


def _verdict_text(scores: Dict[str, float], C: np.ndarray, diag: SlotDiagnostics, cfg: VizConfig) -> str:
    off = C[~np.eye(diag.n_slots, dtype=bool)] if diag.n_slots > 1 else np.array([])
    max_off = float(np.nanmax(off)) if off.size else float("nan")
    active_post = int((diag.comp_usage_post > cfg.active_component_threshold).sum())
    active_prior = int((diag.comp_usage_prior > cfg.active_component_threshold).sum())
    k_eff = float(np.mean(_effective_k(diag.pi_per_image))) if diag.pi_per_image.size else float("nan")
    resp_H = float(np.mean(_entropy(diag.post_pi))) if diag.post_pi.size else float("nan")
    ari = scores["ari"]

    if np.isnan(ari):
        mapping = "slot-component mapping undefined (too few components/slots active)"
    elif ari > 0.50:
        mapping = f"strong slot-component mapping (ARI={ari:.2f})"
    elif ari > 0.15:
        mapping = f"weak/moderate slot-component mapping (ARI={ari:.2f})"
    else:
        mapping = f"little slot-component mapping (ARI={ari:.2f})"

    collapse = "possible slot collapse" if np.isfinite(max_off) and max_off > 0.90 else "no obvious centroid collapse"
    return (
        f"diagnosis: {mapping}; {collapse}; active posterior components \n"
        f"{active_post}/{diag.n_comp}, active prior components {active_prior}/{diag.n_comp};\n"
        f"mean gate K_eff={k_eff:.2f}; mean posterior responsibility entropy={resp_H:.2f};\n"
        f"slot tokens={diag.post_latents.shape[0]} at t={cfg.t_select}.\n"
        f"{_mask_verdict(diag)}"
    )


# --------------------------------------------------------------------------- #
# Top-level public API
# --------------------------------------------------------------------------- #
def visualize_dpgmm_clustering(
    model,
    dataloader,
    device: torch.device,
    cfg: Optional[VizConfig] = None,
    save_path: Optional[str] = None,
    # backwards-compatible trainer args:
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    perplexity: Optional[float] = None,
    t_select: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    cfg = cfg or VizConfig()
    if max_batches is not None:
        cfg.max_batches = int(max_batches)
    if max_samples is not None:
        cfg.max_points = int(max_samples) * 3
        cfg.max_embed_points = int(max_samples)
    if perplexity is not None:
        cfg.perplexity = float(perplexity)
    if t_select is not None:
        cfg.t_select = int(t_select)

    diag = extract_slot_diagnostics(model, dataloader, device, cfg)
    return render_figure(diag, cfg, save_path=save_path, figsize=figsize)


def render_figure(
    diag: SlotDiagnostics,
    cfg: VizConfig,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    joint = joint_embed_with_component_means(diag.post_latents, diag.prior_latents, diag.comp_mean, cfg)
    emb = joint["emb"]
    src = joint["src"]
    post_idx = joint["post_idx"]
    prior_idx = joint["prior_idx"]
    method = joint["method"]

    m_post = src == 0
    m_prior = src == 1
    m_mean = src == 2
    emb_post = emb[m_post]
    emb_prior = emb[m_prior]
    comp_post = diag.post_comp[post_idx]
    slot_post = diag.post_slot_id[post_idx]
    comp_prior = diag.prior_comp[prior_idx]

    M = slot_component_contingency(diag.post_slot_id, diag.post_comp, diag.n_slots, diag.n_comp)
    M_sampled = slot_component_contingency(slot_post, comp_post, diag.n_slots, diag.n_comp)
    scores = mapping_scores(diag.post_slot_id, diag.post_comp)
    C = _slot_centroid_cosine(diag.post_latents, diag.post_slot_id, diag.n_slots)

    fig = plt.figure(figsize=figsize or (18, 15), dpi=cfg.dpi)
    fig.suptitle(
        f"Slot / DPGMM posterior-prior diagnostics  (t={cfg.t_select}, embedding={method})",
        fontsize=14,
        y=0.996,
    )
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.0, 0.9, 1.0, 1.18],
        hspace=0.6,
        left=0.045,
        right=0.985,
        top=0.95,
        bottom=0.05,
    )

    # Row 1: core latent-space question.
    row1 = outer[0].subgridspec(1, 4, wspace=0.30)
    _scatter_discrete(fig.add_subplot(row1[0]), emb_post, comp_post, diag.n_comp,
                      "posterior slot latents\ncolored by DPGMM component", "comp", cfg.point_size)
    _scatter_discrete(fig.add_subplot(row1[1]), emb_post, slot_post, diag.n_slots,
                      "same posterior latents\ncolored by slot index", "slot", cfg.point_size)
    _plot_joint_prior_posterior(fig.add_subplot(row1[2]), emb, src, np.arange(diag.n_comp),
                                "posterior vs conditional prior\n(shared embedding + comp means)", cfg.point_size)
    _plot_contingency(fig.add_subplot(row1[3]), M, scores)

    # Row 2: usage/collapse/calibration.
    row2 = outer[1].subgridspec(1, 4, wspace=0.32)
    _scatter_discrete(fig.add_subplot(row2[0]), emb_prior, comp_prior, diag.n_comp,
                      "conditional prior samples\ncolored by component", "comp", cfg.point_size * 0.85)
    _plot_slot_cosine(fig.add_subplot(row2[1]), C)
    _plot_component_usage(fig.add_subplot(row2[2]), diag.comp_usage_post, diag.comp_usage_prior, cfg.active_component_threshold)
    _plot_entropy_hist(fig.add_subplot(row2[3]), diag)

    # Row 3: masks and gate.
    row3 = outer[2].subgridspec(1, 4, wspace=0.32)
    _plot_gate_effective_k(fig.add_subplot(row3[0]), diag)
    _plot_mask_quality(fig.add_subplot(row3[1]), diag)
    if cfg.show_kumaraswamy:
        _plot_kumaraswamy(fig.add_subplot(row3[2]), diag)
    else:
        _blank(fig.add_subplot(row3[2]), "Kumaraswamy hidden")
    ax_sum = fig.add_subplot(row3[3])
    ax_sum.axis("off")
    summary = _verdict_text(scores, C, diag, cfg)
    guide = (
        "Good signs:\n"
        "  • posterior and prior overlap in the joint plot\n"
        "  • more than one component is active\n"
        "  • attention masks are peaked (peak >> 1/S, entropy << log S)\n"
        "  • off-diagonal slot cosine is not near 1\n\n"
        "Warnings:\n"
        "  • one active component = DPGMM collapse\n"
        "  • attention peak ~ 1/S = slot attention collapsed\n"
        "  • attention sharp but decoder alpha ~ 1/S = decoder ignores masks\n"
        "  • high resp entropy = uncertain component assignment\n\n"
        + summary
    )
    ax_sum.text(0.0, 1.0, guide, fontsize=7.0, family="monospace", va="top", ha="left", linespacing=1.02, transform=ax_sum.transAxes)

    # Row 4: direct slot visual evidence (attention vs decoder alpha).
    _plot_mask_comparison_grid(fig, outer[3], diag.examples, diag.n_slots)

    if save_path:
        fig.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight")
    return fig


__all__ = [
    "VizConfig",
    "SlotDiagnostics",
    "extract_slot_diagnostics",
    "extract_slot_latents_and_assignments",
    "visualize_dpgmm_clustering",
    "render_figure",
]