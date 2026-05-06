from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vdvae.top_dpgmm_prior import ConditionalTopDPGMM, compute_slot_kl_conditional_frozen, sample_slots_conditional_frozen
from umap import UMAP  # type: ignore



def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 [0,255] or float [0,1]/[-1,1] into float [-1,1]."""
    if x.dtype == torch.uint8:
        return x.float() / 127.5 - 1.0
    x = x.float()
    if x.numel() > 0:
        mx = float(x.max().item())
        mn = float(x.min().item())
        if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
            x = x * 2.0 - 1.0
    return x


def _chw_to_01(x: torch.Tensor) -> np.ndarray:
    """Tensor [C,H,W] or [H,W,C] in [-1,1] or [0,1] -> numpy [H,W,3]."""
    x = x.detach().float().cpu()
    if x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if x.numel() == 0:
        arr = np.zeros((1, 1, 3), dtype=np.float32)
    else:
        mn, mx = float(x.min().item()), float(x.max().item())
        y = x if (mn >= -1e-6 and mx <= 1.0 + 1e-6) else (x + 1.0) * 0.5
        y = y.clamp(0.0, 1.0)
        arr = y.numpy()
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.float32, copy=False)


def _mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    """Mask tensor [1,H,W] or [H,W] -> RGB numpy [H,W,3]."""
    m = mask.detach().float().cpu()
    if m.dim() == 3:
        m = m.squeeze(0)
    if m.numel() == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)
    m = (m - m.min()) / (m.max() - m.min()).clamp_min(1e-6)
    arr = m.numpy().astype(np.float32, copy=False)
    return np.repeat(arr[..., None], 3, axis=-1)


def _slot_content(x_chw: torch.Tensor, mask_1hw: torch.Tensor) -> np.ndarray:
    """Original frame multiplied by a slot mask, shown as the slot content proxy."""
    x01 = torch.from_numpy(_chw_to_01(x_chw)).permute(2, 0, 1)
    m = mask_1hw.detach().float().cpu()
    if m.dim() == 2:
        m = m[None]
    m = m.clamp(0.0, 1.0)
    return (x01 * m + 0.15 * (1.0 - m)).permute(1, 2, 0).numpy().astype(np.float32, copy=False)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _effective_k(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.exp(_entropy(p, eps=eps))


def _standardize_and_pca(X: np.ndarray, pca_dims: int = 50) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    if X.shape[0] == 0:
        return X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X = (X - mu) / sd
    if pca_dims and X.shape[1] > pca_dims and X.shape[0] > 3:
        n_comp = min(int(pca_dims), X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=n_comp, random_state=42).fit_transform(X)
    return X.astype(np.float32, copy=False)


def compute_tsne_embedding(
    latents: np.ndarray,
    max_samples: int = 10000,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    random_state: int = 42,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Safe t-SNE with downsampling, PCA pre-reduction, and sklearn-version fallback."""
    rng = np.random.default_rng(int(random_state))
    n_total = int(latents.shape[0])
    if n_total == 0:
        return np.zeros((0, int(n_components)), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    idx = np.arange(n_total, dtype=np.int64)
    if n_total > int(max_samples):
        idx = rng.choice(idx, size=int(max_samples), replace=False).astype(np.int64, copy=False)

    X0 = np.asarray(latents[idx], dtype=np.float64)
    finite = np.isfinite(X0).all(axis=1)
    idx = idx[finite]
    X = _standardize_and_pca(X0[finite], pca_dims=pca_dims)

    if X.shape[0] < 5:
        # Not enough points for useful t-SNE. Return 2-D PCA/padded zeros.
        if X.shape[0] == 0:
            return np.zeros((0, int(n_components)), dtype=np.float32), idx
        n_comp = min(int(n_components), X.shape[1], max(1, X.shape[0] - 1))
        emb = PCA(n_components=n_comp, random_state=int(random_state)).fit_transform(X) if X.shape[0] > 1 else np.zeros((1, n_comp))
        if emb.shape[1] < int(n_components):
            emb = np.pad(emb, ((0, 0), (0, int(n_components) - emb.shape[1])))
        return emb.astype(np.float32, copy=False), idx

    # tiny jitter for duplicated rows, because duplicate slots can break neighbor graph assumptions
    if np.unique(np.round(X, 6), axis=0).shape[0] < X.shape[0]:
        X = X + (1e-6 * rng.standard_normal(X.shape)).astype(X.dtype)

    n_pts = int(X.shape[0])
    perp = min(float(perplexity), float(n_pts - 1) - 1e-3, max(2.0, (n_pts - 1) / 3.0))
    perp = max(2.0, perp)

    kwargs = dict(
        n_components=int(n_components),
        perplexity=perp,
        init="pca" if X.shape[0] > 10 else "random",
        learning_rate="auto",
        random_state=int(random_state),
        method="barnes_hut",
    )
    try:
        tsne = TSNE(**kwargs, max_iter=1000)
    except TypeError:  # older sklearn
        tsne = TSNE(**kwargs, n_iter=1000)

    try:
        emb = tsne.fit_transform(X)
    except ValueError:
        kwargs["method"] = "exact"
        kwargs["init"] = "random"
        kwargs["perplexity"] = max(2.0, min(perp, 20.0, float(n_pts - 1) - 1e-3))
        try:
            tsne = TSNE(**kwargs, max_iter=1000)
        except TypeError:
            tsne = TSNE(**kwargs, n_iter=1000)
        emb = tsne.fit_transform(X)

    return emb.astype(np.float32, copy=False), idx


def compute_umap_or_pca(
    latents: np.ndarray,
    max_samples: int = 10000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(int(random_state))
    n = int(latents.shape[0])
    idx = np.arange(n, dtype=np.int64)
    if n > int(max_samples):
        idx = rng.choice(idx, size=int(max_samples), replace=False).astype(np.int64, copy=False)
    X0 = np.asarray(latents[idx], dtype=np.float64)
    finite = np.isfinite(X0).all(axis=1)
    idx = idx[finite]
    X = _standardize_and_pca(X0[finite], pca_dims=50)
    if X.shape[0] < 3:
        return np.zeros((X.shape[0], 2), dtype=np.float32), idx, "PCA"
    if UMAP is not None:
        n_neighbors = max(2, min(15, X.shape[0] - 1))
        emb = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=int(random_state)).fit_transform(X)
        return emb.astype(np.float32, copy=False), idx, "UMAP"
    n_comp = min(2, X.shape[1], X.shape[0] - 1)
    emb = PCA(n_components=n_comp, random_state=int(random_state)).fit_transform(X)
    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])))
    return emb.astype(np.float32, copy=False), idx, "PCA fallback (install umap-learn for UMAP)"


def compute_responsibilities(
    z_tokens: torch.Tensor,
    pi: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    var = torch.exp(log_vars).clamp_min(eps)
    diff2 = (z_tokens[:, None, :] - means) ** 2
    log_gauss = -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + diff2 / var).sum(dim=-1)
    log_pi = torch.log(pi.clamp_min(eps))
    resp = torch.softmax(log_pi + log_gauss, dim=-1)
    return resp, resp.argmax(dim=-1)


@torch.no_grad()
def _get_vdvae_out_at_t(model, batch: Dict, device: torch.device, t_select: int = 0):
    obs = batch["observations"].to(device)
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
    dtype = obs.dtype

    def zero_action():
        return torch.zeros(B, model.action_dim, device=device, dtype=dtype)

    core_state = model.rnn.init_state(B, device=device, dtype=dtype)
    z_prev_top = torch.zeros(B, model.zdim, model.top_H, model.top_W, device=device, dtype=dtype)
    prev_latents = model._init_decoder_prev_latents(B, device, dtype)

    last = None
    for t in range(t_select + 1):
        x_t = obs[:, t]
        mask_t = torch.ones(B, device=device, dtype=torch.float32) if dones is None or t == 0 else (1.0 - dones[:, t - 1].float())
        keep = mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)

        h, c, m = core_state
        h0, c0, m0 = model.rnn.init_state(B, device=device, dtype=dtype)
        core_state = (h * keep + h0 * (1.0 - keep), c * keep + c0 * (1.0 - keep), m * keep + m0 * (1.0 - keep))
        prev_latents = [pl * keep if pl is not None else None for pl in prev_latents]

        h_t = model.rnn.out_norm(core_state[0])
        a_prev = zero_action() if t == 0 or actions is None else actions[:, t - 1].to(device=device, dtype=dtype) * mask_t.view(B, 1).to(dtype)
        tr = model.latent_transport(z_top_prev=z_prev_top * keep, h_t=h_t, action_prev=a_prev, dt=1.0)

        prev_latents_in = [None if pl is None else pl.detach().clone() for pl in prev_latents]
        h_decoder_top = tr["h_decoder_top"].detach().clone() if tr.get("h_decoder_top", None) is not None else None

        x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
        vdvae_out = model.vdvae.forward(
            x_t_nhwc,
            x_t_nhwc,
            cond_top=tr["cond_top"],
            h_prior_top=h_t,
            h_decoder_top=tr["h_decoder_top"],
            mask_t=mask_t,
            prev_latents=prev_latents,
            get_latents=True,
        )

        z_top_map = vdvae_out["current_latents"][0]
        a_cur = zero_action() if actions is None else actions[:, t].to(device=device, dtype=dtype)
        _, core_state = model.rnn(z_top_map, a_cur, state=core_state, mask_t=None, extra_maps=None)
        z_prev_top = z_top_map.detach()
        prev_latents = [None if idx == 0 else z.detach() for idx, z in enumerate(vdvae_out["current_latents"])]
        last = (vdvae_out, x_t, h_t, h_decoder_top, prev_latents_in)

    if last is None:
        raise RuntimeError("Could not extract a VDVAE output from the batch.")
    return last


@torch.no_grad()
def extract_slot_latents_and_assignments(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 10,
    t_select: int = 8,
    max_points: int = 8000,
    n_exemplars_per_component: int = 4,
) -> Dict[str, Any]:
    model.eval()
    snapshot = model.vdvae.top_prior_snapshot
    frozen_gate = model.vdvae.top_prior_gate
    if snapshot is None or frozen_gate is None:
        raise ValueError("Frozen top prior is not set. Run refresh_top_prior_from_buffer once or use the bootstrapped prior.")

    K = int(snapshot.K)
    post_z_all, post_pi_all, post_k_all, post_sid_all = [], [], [], []
    prior_z_all, prior_pi_all, prior_k_all, prior_sid_all = [], [], [], []
    slot_mask_mass_all, slot_attn_mass_all = [], []
    pi_per_image_all, kumar_a_all, kumar_b_all = [], [], []
    same_frame_examples: List[Dict[str, Any]] = []
    exemplars: List[List[Dict[str, Any]]] = [[] for _ in range(K)]
    slot_exemplars_by_index: Optional[List[List[Dict[str, Any]]]] = None
    num_slots_seen: Optional[int] = None
    total = 0

    for bi, batch in enumerate(tqdm(dataloader, desc="Extracting slot diagnostics", leave=False)):
        if bi >= max_batches or total >= max_points:
            break
        if not isinstance(batch, dict) or "observations" not in batch:
            continue

        vdvae_out, x_t, h_t, h_decoder_top, prev_latents_in = _get_vdvae_out_at_t(model, batch, device, t_select=t_select)
        slot_mu = vdvae_out.get("top_slot_mu", None)
        slot_logsigma = vdvae_out.get("top_slot_logsigma", None)
        slot_masks = vdvae_out.get("top_slot_masks", None)
        slot_attn = vdvae_out.get("top_slot_attn", None)
        slot_comp_maps = vdvae_out.get("top_slot_comp_maps", None)
        if slot_mu is None or slot_logsigma is None or slot_masks is None or slot_comp_maps is None:
            continue

        B, S, D = slot_mu.shape
        if slot_exemplars_by_index is None:
            slot_exemplars_by_index = [[] for _ in range(S)]
            num_slots_seen = S
        elif S != num_slots_seen:
            raise ValueError(f"Number of slots changed across batches: first S={num_slots_seen}, now S={S}")
        _, resp = compute_slot_kl_conditional_frozen(
            snapshot=snapshot,
            frozen_gate=frozen_gate,
            h_t=h_t,
            slot_mu=slot_mu,
            slot_logsigma=slot_logsigma,
        )
        k_slot = resp.argmax(dim=-1)

        slot_ids = torch.arange(S, device=device, dtype=torch.long).view(1, S).expand(B, S).reshape(B * S)
        post_z_all.append(slot_mu.reshape(B * S, D).detach().cpu())
        post_pi_all.append(resp.reshape(B * S, K).detach().cpu())
        post_k_all.append(k_slot.reshape(B * S).detach().cpu())
        post_sid_all.append(slot_ids.detach().cpu())

        prior_slots = sample_slots_conditional_frozen(snapshot, frozen_gate, h_t, num_slots=S, temperature=1.0)
        elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(snapshot, frozen_gate, h_t)
        pi_img = torch.softmax(elogpi, dim=-1)
        gate_out = frozen_gate(h_t)
        kumar_a = gate_out.get("a", torch.empty(B, 0, device=device, dtype=slot_mu.dtype))
        kumar_b = gate_out.get("b", torch.empty(B, 0, device=device, dtype=slot_mu.dtype))
        pi_per_image_all.append(pi_img.detach().cpu())
        kumar_a_all.append(kumar_a.detach().cpu())
        kumar_b_all.append(kumar_b.detach().cpu())
        pi_slot = pi_img[:, None, :].expand(B, S, K).reshape(B * S, K)
        means = snapshot.comp_mean.to(device=device, dtype=prior_slots.dtype).view(K, -1)
        log_vars = torch.log(snapshot.comp_var.to(device=device, dtype=prior_slots.dtype).view(K, -1).clamp_min(1e-6))
        means_b = means[None].expand(B * S, K, D)
        log_vars_b = log_vars[None].expand(B * S, K, D)
        prior_resp, prior_k = compute_responsibilities(prior_slots.reshape(B * S, D), pi_slot, means_b, log_vars_b)

        prior_z_all.append(prior_slots.reshape(B * S, D).detach().cpu())
        prior_pi_all.append(prior_resp.detach().cpu())
        prior_k_all.append(prior_k.detach().cpu())
        prior_sid_all.append(slot_ids.detach().cpu())

        H_img, W_img = x_t.shape[-2], x_t.shape[-1]
        comp_mask_up = F.interpolate(
            slot_masks.reshape(B * S, 1, slot_masks.shape[-2], slot_masks.shape[-1]),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, S, 1, H_img, W_img).clamp(0.0, 1.0)

        if slot_attn is not None and slot_attn.shape[-1] == slot_masks.shape[-2] * slot_masks.shape[-1]:
            attn_up = F.interpolate(
                slot_attn.reshape(B * S, 1, slot_masks.shape[-2], slot_masks.shape[-1]),
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, S, 1, H_img, W_img)
        else:
            attn_up = comp_mask_up

        if len(same_frame_examples) < 3:
            n_take = min(B, 3 - len(same_frame_examples))
            for b in range(n_take):
                same_frame_examples.append({
                    "frame": x_t[b].detach().cpu(),
                    "slot_attn": attn_up[b].detach().cpu(),
                    "slot_masks": comp_mask_up[b].detach().cpu(),
                    "coverage": comp_mask_up[b].sum(dim=0).detach().cpu(),
                    "resp": resp[b].detach().cpu(),
                    "pi": pi_img[b].detach().cpu(),
                    "h_decoder_top": None if h_decoder_top is None else h_decoder_top[b:b+1].detach().cpu(),
                    "prev_latents": [None if z is None else z[b:b+1].detach().cpu() for z in prev_latents_in],
                })

        slot_mask_mass_all.append(comp_mask_up.mean(dim=(2, 3, 4)).detach().cpu())
        slot_attn_mass_all.append(attn_up.mean(dim=(2, 3, 4)).detach().cpu())

        for b in range(B):
            prev_latents_b = [None if z is None else z[b:b+1].detach().cpu() for z in prev_latents_in]
            h_dec_b = None if h_decoder_top is None else h_decoder_top[b:b+1].detach().cpu()
            for s in range(S):
                kk = int(k_slot[b, s].item())
                conf = float(resp[b, s, kk].item())
                z_top_slot = (slot_comp_maps[b:b+1, s] * slot_masks[b:b+1, s]).detach().cpu()
                mask_mass = float(comp_mask_up[b, s].mean().item())
                item = {
                    "slot_id": int(s),
                    "component_id": kk,
                    "conf": conf,
                    "frame": x_t[b].detach().cpu(),
                    "attn": attn_up[b, s].detach().cpu(),
                    "comp_mask": comp_mask_up[b, s].detach().cpu(),
                    "z_top_slot": z_top_slot,
                    "h_decoder_top": h_dec_b,
                    "prev_latents": prev_latents_b,
                    "mask_mass": mask_mass,
                }
                bucket = exemplars[kk]
                if len(bucket) < n_exemplars_per_component:
                    bucket.append(item)
                elif conf > bucket[-1]["conf"]:
                    bucket[-1] = item
                bucket.sort(key=lambda z: z["conf"], reverse=True)
                slot_bucket = slot_exemplars_by_index[s]
                if len(slot_bucket) < n_exemplars_per_component:
                    slot_bucket.append(item)
                elif mask_mass > slot_bucket[-1]["mask_mass"]:
                    slot_bucket[-1] = item
                slot_bucket.sort(key=lambda z: z["mask_mass"], reverse=True)                

        total += B * S

    if not post_z_all or not prior_z_all:
        raise ValueError("No slot latents were extracted. Check that the dataloader returns dict batches with 'observations'.")

    return {
        "posterior_latents": torch.cat(post_z_all, dim=0).numpy(),
        "posterior_pi": torch.cat(post_pi_all, dim=0).numpy(),
        "posterior_assignments": torch.cat(post_k_all, dim=0).numpy(),
        "posterior_slot_ids": torch.cat(post_sid_all, dim=0).numpy(),
        "prior_latents": torch.cat(prior_z_all, dim=0).numpy(),
        "prior_pi": torch.cat(prior_pi_all, dim=0).numpy(),
        "prior_assignments": torch.cat(prior_k_all, dim=0).numpy(),
        "prior_slot_ids": torch.cat(prior_sid_all, dim=0).numpy(),
        "slot_mask_mass": torch.cat(slot_mask_mass_all, dim=0).numpy() if slot_mask_mass_all else np.zeros((0, int(num_slots_seen or 0)), dtype=np.float32),
        "slot_attn_mass": torch.cat(slot_attn_mass_all, dim=0).numpy() if slot_attn_mass_all else np.zeros((0, int(num_slots_seen or 0)), dtype=np.float32),
        "pi_per_image": torch.cat(pi_per_image_all, dim=0).numpy() if pi_per_image_all else np.zeros((0, K), dtype=np.float32),
        "kumar_a": torch.cat(kumar_a_all, dim=0).numpy() if kumar_a_all else np.zeros((0, max(0, K - 1)), dtype=np.float32),
        "kumar_b": torch.cat(kumar_b_all, dim=0).numpy() if kumar_b_all else np.zeros((0, max(0, K - 1)), dtype=np.float32),
        "comp_mean": snapshot.comp_mean.detach().cpu().numpy(),
        "comp_var": snapshot.comp_var.detach().cpu().numpy(),
        "same_frame_examples": same_frame_examples,
        "exemplars_with_masks": exemplars,
        "slot_exemplars_by_index": slot_exemplars_by_index if slot_exemplars_by_index is not None else [],
        "num_slots": int(num_slots_seen or 0),
    }


def _sanitize(latents: np.ndarray, assignments: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N,D], got {latents.shape}")
    if pi.ndim != 2:
        raise ValueError(f"Expected responsibilities [N,K], got {pi.shape}")
    assignments = assignments.reshape(-1)
    if latents.shape[0] != pi.shape[0] or latents.shape[0] != assignments.shape[0]:
        raise ValueError(f"Shape mismatch: latents={latents.shape}, assignments={assignments.shape}, pi={pi.shape}")
    K = pi.shape[1]
    mask = np.isfinite(latents).all(axis=1) & np.isfinite(pi).all(axis=1) & (assignments >= 0) & (assignments < K)
    return latents[mask].astype(np.float32, copy=False), assignments[mask].astype(np.int64, copy=False), pi[mask].astype(np.float32, copy=False), int((~mask).sum())



def _sanitize_with_slot_ids(
    latents: np.ndarray,
    assignments: np.ndarray,
    pi: np.ndarray,
    slot_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N,D], got {latents.shape}")
    if pi.ndim != 2:
        raise ValueError(f"Expected responsibilities [N,K], got {pi.shape}")
    assignments = assignments.reshape(-1)
    slot_ids = slot_ids.reshape(-1)
    if latents.shape[0] != pi.shape[0] or latents.shape[0] != assignments.shape[0] or latents.shape[0] != slot_ids.shape[0]:
        raise ValueError(
            f"Shape mismatch: latents={latents.shape}, assignments={assignments.shape}, "
            f"pi={pi.shape}, slot_ids={slot_ids.shape}"
        )
    K = pi.shape[1]
    mask = (
        np.isfinite(latents).all(axis=1)
        & np.isfinite(pi).all(axis=1)
        & np.isfinite(slot_ids)
        & (assignments >= 0)
        & (assignments < K)
        & (slot_ids >= 0)
    )
    return (
        latents[mask].astype(np.float32, copy=False),
        assignments[mask].astype(np.int64, copy=False),
        pi[mask].astype(np.float32, copy=False),
        slot_ids[mask].astype(np.int64, copy=False),
        int((~mask).sum()),
    )


def _slot_component_table(slot_ids: np.ndarray, assignments: np.ndarray, num_slots: int, num_components: int) -> np.ndarray:
    table = np.zeros((int(num_slots), int(num_components)), dtype=np.float64)
    if slot_ids.size == 0 or assignments.size == 0 or num_slots <= 0 or num_components <= 0:
        return table
    valid = (slot_ids >= 0) & (slot_ids < num_slots) & (assignments >= 0) & (assignments < num_components)
    np.add.at(table, (slot_ids[valid], assignments[valid]), 1.0)
    row_sum = table.sum(axis=1, keepdims=True)
    return table / np.maximum(row_sum, 1.0)



def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norm, 1e-12)
    return Xn @ Xn.T


def _pairwise_l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))


def _slot_means_by_index(latents: np.ndarray, slot_ids: np.ndarray, num_slots: int) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float64)
    slot_ids = np.asarray(slot_ids, dtype=np.int64).reshape(-1)
    if latents.ndim != 2 or latents.shape[0] == 0 or num_slots <= 0:
        return np.zeros((max(1, int(num_slots)), 1), dtype=np.float64)
    out = np.zeros((int(num_slots), latents.shape[1]), dtype=np.float64)
    for s in range(int(num_slots)):
        mask = slot_ids == s
        if np.any(mask):
            out[s] = latents[mask].mean(axis=0)
    return out


@torch.no_grad()
def _decode_component_mean_rgb(
    model,
    comp_mean_k: torch.Tensor,
    h_decoder_top: Optional[torch.Tensor],
    prev_latents: List[Optional[torch.Tensor]],
    device: torch.device,
) -> np.ndarray:
    """Decode one DPGMM component mean through the correct top-latent interface."""
    comp_mean_k = comp_mean_k.to(device=device, dtype=torch.float32)

    slot_bottleneck = None
    slot_to_map = None
    try:
        slot_bottleneck = model.vdvae.decoder.dec_blocks[0].slot_bottleneck
        slot_to_map = slot_bottleneck.slot_to_map
    except Exception:
        slot_bottleneck = None
        slot_to_map = None

    slot_dim = None
    if slot_to_map is not None:
        slot_dim = int(getattr(slot_to_map, "slot_dim", getattr(slot_bottleneck, "slot_dim", -1)))

    vdvae_h = getattr(getattr(model, "vdvae", None), "H", None)
    zdim = int(getattr(model, "zdim", getattr(vdvae_h, "zdim", comp_mean_k.shape[0])))
    top_h = int(getattr(model, "top_H", comp_mean_k.shape[-2]))
    top_w = int(getattr(model, "top_W", comp_mean_k.shape[-1]))

    # In the slot-DPGMM setup, component means live in slot space [D,1,1].
    # Prefer the slot-space path whenever the component has the learned slot dimensionality.
    if slot_to_map is not None and slot_dim is not None and int(comp_mean_k.numel()) == int(slot_dim):
        slot_vec = comp_mean_k.reshape(1, 1, int(slot_dim))
        z_top, _, _ = slot_to_map(slot_vec)
    elif int(comp_mean_k.numel()) == int(zdim * top_h * top_w):
        z_top = comp_mean_k.reshape(1, zdim, top_h, top_w)
    elif slot_to_map is not None:
        slot_vec = comp_mean_k.reshape(1, 1, -1)
        z_top, _, _ = slot_to_map(slot_vec)
    else:
        raise RuntimeError(
            "Component mean shape is not compatible with either slot-space or top-map decoding."
        )

    h_dec = None if h_decoder_top is None else h_decoder_top.to(device=device, dtype=torch.float32)
    prev = [None if z is None else z.to(device=device, dtype=torch.float32) for z in prev_latents]
    px_z, _ = model.vdvae.decode_from_top_latent(z_top, h_decoder_top=h_dec, prev_latents=prev)
    rgb = model.vdvae.decoder.out_net.sample(px_z)[0]
    return rgb.astype(np.float32) / 255.0

def _imshow_mask(ax, arr: np.ndarray, title: str = "", cmap: str = "magma") -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.size:
        arr = arr / max(float(np.nanmax(arr)), 1e-8)
    ax.imshow(arr, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def _plot_same_frame_examples(fig: plt.Figure, gs, examples: List[Dict[str, Any]], num_slots: int) -> None:
    n = min(len(examples), 3)
    S = int(max(1, num_slots))
    if n == 0:
        ax = fig.add_subplot(gs)
        ax.axis("off")
        ax.text(0.5, 0.5, "No same-frame examples collected", ha="center", va="center")
        return
    sub = gs.subgridspec(n, S + 2, wspace=0.04, hspace=0.14)
    for i in range(n):
        ex = examples[i]
        ax = fig.add_subplot(sub[i, 0])
        ax.imshow(_chw_to_01(ex["frame"]))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("frame", fontsize=8)
        ax.set_ylabel(f"example {i}", fontsize=8)
        for s in range(S):
            ax = fig.add_subplot(sub[i, s + 1])
            attn = ex["slot_attn"][s, 0].numpy() if s < ex["slot_attn"].shape[0] else np.zeros((1, 1), dtype=np.float32)
            mask = ex["slot_masks"][s, 0].numpy() if s < ex["slot_masks"].shape[0] else np.zeros((1, 1), dtype=np.float32)
            attn = attn / max(float(np.nanmax(attn)), 1e-8)
            mask = mask / max(float(np.nanmax(mask)), 1e-8)
            combined = np.concatenate([attn, mask], axis=1)
            _imshow_mask(ax, combined, title=f"slot {s}\nattn | mask" if i == 0 else "")
        ax = fig.add_subplot(sub[i, S + 1])
        _imshow_mask(ax, ex["coverage"].numpy(), title="mask coverage" if i == 0 else "", cmap="viridis")


def _plot_component_decodes(
    fig: plt.Figure,
    gs,
    model,
    device: torch.device,
    comp_mean: np.ndarray,
    pi_mean: np.ndarray,
    examples: List[Dict[str, Any]],
) -> None:
    K = int(comp_mean.shape[0]) if comp_mean.ndim >= 2 else 0
    if K <= 0:
        ax = fig.add_subplot(gs)
        ax.axis("off")
        ax.text(0.5, 0.5, "No component means available", ha="center", va="center")
        return
    sub = gs.subgridspec(1, K, wspace=0.05)
    comp_t = torch.from_numpy(np.asarray(comp_mean, dtype=np.float32))

    # Use one shared decoder context for all components. This makes differences
    # in the decoded row attributable to the component means rather than context.
    base_ex = examples[0] if examples else {}
    h_dec = base_ex.get("h_decoder_top", None)
    prev = base_ex.get("prev_latents", [])

    for k in range(K):
        ax = fig.add_subplot(sub[0, k])
        try:
            rgb = _decode_component_mean_rgb(model, comp_t[k], h_dec, prev, device=device)
            ax.imshow(np.clip(rgb, 0.0, 1.0))
            title = f"comp {k} | same ctx\nmean pi={pi_mean[k]:.2f}" if k < len(pi_mean) else f"comp {k} | same ctx"
            ax.set_title(title, fontsize=8)
        except Exception as exc:
            ax.text(0.5, 0.5, f"decode error\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])


def _select_components_by_coverage(frac: np.ndarray, coverage: float = 0.99) -> np.ndarray:
    frac = np.asarray(frac, dtype=np.float64)
    if frac.size == 0:
        return np.zeros((0,), dtype=np.int64)
    total = frac.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.arange(frac.shape[0], dtype=np.int64)
    frac = frac / total
    order = np.argsort(frac)[::-1]
    cum = np.cumsum(frac[order])
    n_keep = int(np.searchsorted(cum, float(coverage), side="left") + 1)
    n_keep = max(1, min(n_keep, order.shape[0]))
    return order[:n_keep].astype(np.int64, copy=False)


def _decode_approx_slot_rgb(model, exemplar: Dict[str, Any], device: torch.device) -> np.ndarray:
    z_top_slot = exemplar["z_top_slot"].to(device)
    h_decoder_top = None if exemplar.get("h_decoder_top", None) is None else exemplar["h_decoder_top"].to(device)
    prev_latents = [None if z is None else z.to(device) for z in exemplar.get("prev_latents", [])]
    px_z, _ = model.vdvae.decode_from_top_latent(
        z_top_slot,
        h_decoder_top=h_decoder_top,
        prev_latents=prev_latents,
    )
    rgb = model.vdvae.decoder.out_net.sample(px_z)[0]
    return rgb.astype(np.float32) / 255.0

def _plot_slot_index_exemplars(
    fig: plt.Figure,
    gs,
    slot_exemplars_by_index,
    num_slots: int,
    model,
    device: torch.device,
) -> None:
    """
    Plot one row per actual slot index.
    """
    S = int(num_slots)
    if S <= 0 or not slot_exemplars_by_index:
        ax = fig.add_subplot(gs)
        ax.axis("off")
        ax.text(0.5, 0.5, "No slot-index exemplars collected", ha="center", va="center")
        return

    sub = gs.subgridspec(S, 5, wspace=0.02, hspace=0.12)
    headers = ["frame", "attention", "slot content", "decoder mask", "approx RGB"]

    for s in range(S):
        if s >= len(slot_exemplars_by_index) or len(slot_exemplars_by_index[s]) == 0:
            for c in range(5):
                ax = fig.add_subplot(sub[s, c])
                ax.axis("off")
                if c == 0:
                    ax.text(0.5, 0.5, f"slot {s}\nno exemplar", ha="center", va="center")
            continue

        # Already sorted by mask_mass in extract_slot_latents_and_assignments
        ex = slot_exemplars_by_index[s][0]

        try:
            approx_rgb = _decode_approx_slot_rgb(model, ex, device=device)
        except Exception:
            approx_rgb = np.zeros_like(_chw_to_01(ex["frame"]))

        images = [
            _chw_to_01(ex["frame"]),
            _mask_to_rgb(ex["attn"]),
            _slot_content(ex["frame"], ex["comp_mask"]),
            _mask_to_rgb(ex["comp_mask"]),
            approx_rgb,
        ]

        for c, img in enumerate(images):
            ax = fig.add_subplot(sub[s, c])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if s == 0:
                ax.set_title(headers[c], fontsize=9)

            if c == 0:
                comp_id = ex.get("component_id", -1)
                conf = ex.get("conf", 0.0)
                mass = ex.get("mask_mass", 0.0)
                ax.set_ylabel(
                    f"slot={s}\nDPGMM k={comp_id}\nconf={conf:.2f}\nmass={mass:.3f}",
                    fontsize=8,
                )

def visualize_dpgmm_clustering(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    max_samples: int = 10000,
    perplexity: float = 30.0,
    t_select: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
    tsne_dims: int = 2,
    coverage_threshold: float = 0.99,
) -> plt.Figure:
    """
    Slot-focused diagnostic dashboard.

    """
    model.eval()
    data = extract_slot_latents_and_assignments(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        t_select=t_select,
        max_points=max_samples * 3,
        n_exemplars_per_component=4,
    )

    post_latents, post_assign, post_pi, post_slot_ids, dropped_post = _sanitize_with_slot_ids(
        np.asarray(data["posterior_latents"], dtype=np.float32),
        np.asarray(data["posterior_assignments"], dtype=np.int64),
        np.asarray(data["posterior_pi"], dtype=np.float32),
        np.asarray(data.get("posterior_slot_ids", np.zeros((len(data["posterior_latents"]),), dtype=np.int64)), dtype=np.int64),
    )
    prior_latents, prior_assign, prior_pi, prior_slot_ids, dropped_prior = _sanitize_with_slot_ids(
        np.asarray(data["prior_latents"], dtype=np.float32),
        np.asarray(data["prior_assignments"], dtype=np.int64),
        np.asarray(data["prior_pi"], dtype=np.float32),
        np.asarray(data.get("prior_slot_ids", np.zeros((len(data["prior_latents"]),), dtype=np.int64)), dtype=np.int64),
    )
    if dropped_post:
        print(f"[slot-viz] Dropped {dropped_post} invalid posterior points.")
    if dropped_prior:
        print(f"[slot-viz] Dropped {dropped_prior} invalid prior points.")
    if post_latents.shape[0] < 5 or prior_latents.shape[0] < 5:
        raise ValueError("Not enough valid posterior/prior slot points for visualization; need at least 5 each.")

    num_slots = int(data.get("num_slots", 0))
    num_slots = max(1, num_slots)
    tsne_dims = 3 if int(tsne_dims) == 3 else 2
    emb_post, idx_post = compute_tsne_embedding(
        post_latents,
        max_samples=max_samples,
        perplexity=perplexity,
        random_state=42,
        n_components=tsne_dims,
    )
    emb_prior, idx_prior = compute_tsne_embedding(
        prior_latents,
        max_samples=max_samples,
        perplexity=perplexity,
        random_state=43,
        n_components=tsne_dims,
    )

    n_joint = min(int(max_samples), post_latents.shape[0], prior_latents.shape[0])
    joint = np.concatenate([post_latents[:n_joint], prior_latents[:n_joint]], axis=0)
    joint_src = np.concatenate([np.zeros(n_joint, dtype=np.int64), np.ones(n_joint, dtype=np.int64)])
    joint_emb, _, joint_method = compute_umap_or_pca(joint, max_samples=2 * n_joint, random_state=44)

    K_seen = max(post_pi.shape[1], prior_pi.shape[1])
    if K_seen <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, max(1, K_seen))))
    else:
        colors = plt.cm.turbo(np.linspace(0, 1, K_seen))
    cmap = ListedColormap(colors)

    post_counts = np.bincount(post_assign, minlength=K_seen).astype(np.float64)
    prior_counts = np.bincount(prior_assign, minlength=K_seen).astype(np.float64)
    post_frac = post_counts / max(1.0, post_counts.sum())
    prior_frac = prior_counts / max(1.0, prior_counts.sum())
    combined_frac = post_frac + prior_frac
    combined_frac = combined_frac / max(1e-12, combined_frac.sum())
    order = np.argsort(combined_frac)[::-1]
    selected_components = _select_components_by_coverage(combined_frac, coverage=coverage_threshold)
    displayed_coverage = float(combined_frac[selected_components].sum()) if selected_components.size else 0.0

    slot_counts = np.bincount(post_slot_ids, minlength=num_slots).astype(np.float64)
    slot_frac = slot_counts / max(1.0, slot_counts.sum())
    slot_mask_mass = np.asarray(data.get("slot_mask_mass", np.zeros((0, num_slots))), dtype=np.float64)
    slot_attn_mass = np.asarray(data.get("slot_attn_mass", np.zeros((0, num_slots))), dtype=np.float64)
    mean_mask_mass = slot_mask_mass.mean(axis=0) if slot_mask_mass.ndim == 2 and slot_mask_mass.shape[0] else np.zeros((num_slots,), dtype=np.float64)
    mean_attn_mass = slot_attn_mass.mean(axis=0) if slot_attn_mass.ndim == 2 and slot_attn_mass.shape[0] else np.zeros((num_slots,), dtype=np.float64)
    slot_component = _slot_component_table(post_slot_ids, post_assign, num_slots, K_seen)

    comp_mean = np.asarray(data.get("comp_mean", np.zeros((K_seen, 1, 1, 1))), dtype=np.float32)
    comp_var = np.asarray(data.get("comp_var", np.ones_like(comp_mean)), dtype=np.float32)
    comp_mean_flat = comp_mean.reshape(comp_mean.shape[0], -1) if comp_mean.ndim >= 2 else np.zeros((K_seen, 1), dtype=np.float32)
    comp_var_flat = comp_var.reshape(comp_var.shape[0], -1) if comp_var.ndim >= 2 else np.ones((K_seen, 1), dtype=np.float32)
    comp_dist = _pairwise_l2(comp_mean_flat)
    comp_norm = np.linalg.norm(comp_mean_flat, axis=1)
    comp_var_mean = np.maximum(comp_var_flat.mean(axis=1), 1e-12)

    pi_per_image = np.asarray(data.get("pi_per_image", np.zeros((0, K_seen))), dtype=np.float64)
    if pi_per_image.ndim != 2 or pi_per_image.shape[0] == 0:
        pi_per_image = post_pi.reshape(-1, K_seen)
    pi_mean = pi_per_image.mean(axis=0)
    pi_eff = _effective_k(pi_per_image)
    kumar_a = np.asarray(data.get("kumar_a", np.zeros((0, max(0, K_seen - 1)))), dtype=np.float64)
    kumar_b = np.asarray(data.get("kumar_b", np.zeros((0, max(0, K_seen - 1)))), dtype=np.float64)

    slot_mean_by_idx = _slot_means_by_index(post_latents, post_slot_ids, num_slots)
    slot_cos = _pairwise_cosine(slot_mean_by_idx)

    n_same = min(3, len(data.get("same_frame_examples", [])))
    fig_h = max(float(figsize[1]), 13.5 + 1.15 * num_slots + 1.5 * max(1, n_same))
    fig = plt.figure(figsize=(max(float(figsize[0]), 26.0), fig_h), constrained_layout=True)
    outer = fig.add_gridspec(
        6,
        1,
        height_ratios=[max(1.4, 0.78 * max(1, n_same)), 1.10, 0.92, 1.00, 0.82, max(1.6, 0.62 * num_slots)],
    )

    _plot_same_frame_examples(fig, outer[0], data.get("same_frame_examples", []), num_slots=num_slots)

    row2 = outer[1].subgridspec(1, 5, wspace=0.25)
    if tsne_dims == 3:
        ax = fig.add_subplot(row2[0, 0], projection="3d")
        ax.scatter(emb_post[:, 0], emb_post[:, 1], emb_post[:, 2], c=post_assign[idx_post], cmap=cmap, s=8, alpha=0.75)
        ax.set_zlabel("t-SNE 3")
    else:
        ax = fig.add_subplot(row2[0, 0])
        sc = ax.scatter(emb_post[:, 0], emb_post[:, 1], c=post_assign[idx_post], cmap=cmap, s=9, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="DPGMM component")
    ax.set_title("Posterior slots by component")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if tsne_dims == 3:
        ax = fig.add_subplot(row2[0, 1], projection="3d")
        ax.scatter(emb_post[:, 0], emb_post[:, 1], emb_post[:, 2], c=post_slot_ids[idx_post], cmap=plt.cm.tab20, s=8, alpha=0.75)
        ax.set_zlabel("t-SNE 3")
    else:
        ax = fig.add_subplot(row2[0, 1])
        sc = ax.scatter(emb_post[:, 0], emb_post[:, 1], c=post_slot_ids[idx_post], cmap=plt.cm.tab20, s=9, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="slot index")
    ax.set_title("Posterior slots by slot index")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if tsne_dims == 3:
        ax = fig.add_subplot(row2[0, 2], projection="3d")
        ax.scatter(emb_prior[:, 0], emb_prior[:, 1], emb_prior[:, 2], c=prior_assign[idx_prior], cmap=cmap, s=8, alpha=0.75)
        ax.set_zlabel("t-SNE 3")
    else:
        ax = fig.add_subplot(row2[0, 2])
        sc = ax.scatter(emb_prior[:, 0], emb_prior[:, 1], c=prior_assign[idx_prior], cmap=cmap, s=9, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="DPGMM component")
    ax.set_title("Prior samples by component")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if tsne_dims == 3:
        ax = fig.add_subplot(row2[0, 3], projection="3d")
        ax.scatter(emb_prior[:, 0], emb_prior[:, 1], emb_prior[:, 2], c=prior_slot_ids[idx_prior], cmap=plt.cm.tab20, s=8, alpha=0.75)
        ax.set_zlabel("t-SNE 3")
    else:
        ax = fig.add_subplot(row2[0, 3])
        sc = ax.scatter(emb_prior[:, 0], emb_prior[:, 1], c=prior_slot_ids[idx_prior], cmap=plt.cm.tab20, s=9, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="slot index")
    ax.set_title("Prior samples by slot index")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    ax = fig.add_subplot(row2[0, 4])
    ax.scatter(joint_emb[joint_src == 0, 0], joint_emb[joint_src == 0, 1], s=10, alpha=0.6, label="posterior")
    ax.scatter(joint_emb[joint_src == 1, 0], joint_emb[joint_src == 1, 1], s=10, alpha=0.6, label="prior")
    ax.set_title(f"{joint_method}: posterior vs prior")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=8)

    row3 = outer[2].subgridspec(1, 4, wspace=0.25)
    ax = fig.add_subplot(row3[0, 0])
    comp_x = np.arange(len(order))
    width = 0.42
    ax.bar(comp_x - width / 2, post_frac[order], width=width, label="posterior")
    ax.bar(comp_x + width / 2, prior_frac[order], width=width, label="prior")
    ax.set_title("DPGMM component usage")
    ax.set_xlabel("component id, sorted")
    ax.set_ylabel("fraction of slot tokens")
    ax.set_xticks(comp_x)
    ax.set_xticklabels([str(int(k)) for k in order], rotation=0)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(row3[0, 1])
    slot_x = np.arange(num_slots)
    width2 = 0.30
    ax.bar(slot_x - width2, slot_frac[:num_slots], width=width2, label="token fraction")
    ax.bar(slot_x, mean_mask_mass[:num_slots], width=width2, label="decoder mask mass")
    ax.bar(slot_x + width2, mean_attn_mass[:num_slots], width=width2, label="attention mass")
    ax.set_title("Slot index usage")
    ax.set_xlabel("slot index")
    ax.set_ylabel("mean / fraction")
    ax.set_xticks(slot_x)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(row3[0, 2])
    im = ax.imshow(slot_component, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Posterior component assignment by slot")
    ax.set_xlabel("DPGMM component")
    ax.set_ylabel("slot index")
    ax.set_xticks(np.arange(K_seen))
    ax.set_yticks(np.arange(num_slots))
    fig.colorbar(im, ax=ax, label="fraction")

    ax = fig.add_subplot(row3[0, 3])
    im = ax.imshow(slot_cos, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_title("Slot-index mean cosine")
    ax.set_xlabel("slot")
    ax.set_ylabel("slot")
    ax.set_xticks(np.arange(num_slots))
    ax.set_yticks(np.arange(num_slots))
    fig.colorbar(im, ax=ax, label="cosine")

    row4 = outer[3].subgridspec(1, 5, wspace=0.25)
    ax = fig.add_subplot(row4[0, 0])
    im = ax.imshow(comp_dist, cmap="viridis")
    ax.set_title("Component mean pairwise L2")
    ax.set_xlabel("component")
    ax.set_ylabel("component")
    ax.set_xticks(np.arange(comp_mean.shape[0]))
    ax.set_yticks(np.arange(comp_mean.shape[0]))
    fig.colorbar(im, ax=ax, label="L2")

    ax = fig.add_subplot(row4[0, 1])
    ax.bar(np.arange(comp_mean.shape[0]), comp_norm, label="||mean||")
    ax2 = ax.twinx()
    ax2.plot(np.arange(comp_var_mean.shape[0]), comp_var_mean, marker="o", linewidth=1.0, label="mean var")
    ax2.set_yscale("log")
    ax.set_title("Component norm and variance")
    ax.set_xlabel("component")
    ax.set_ylabel("||component mean||")
    ax2.set_ylabel("mean variance")

    ax = fig.add_subplot(row4[0, 2])
    ax.bar(np.arange(K_seen), pi_mean[:K_seen])
    ax.set_title("Mean conditional gate pi(h)")
    ax.set_xlabel("component")
    ax.set_ylabel("mean probability")
    ax.set_xticks(np.arange(K_seen))
    ax.set_ylim(0.0, 1.05)

    ax = fig.add_subplot(row4[0, 3])
    if pi_eff.size:
        ax.hist(pi_eff, bins=min(20, max(4, int(np.sqrt(pi_eff.size)))), edgecolor="black")
        ax.axvline(1.0, linestyle="--", linewidth=1.0, label="K_eff=1")
        ax.axvline(float(K_seen), linestyle="--", linewidth=1.0, label=f"K_eff={K_seen}")
        ax.legend(fontsize=7)
    ax.set_title("Per-image effective K")
    ax.set_xlabel("exp(H[pi])")
    ax.set_ylabel("count")

    ax = fig.add_subplot(row4[0, 4])
    if kumar_a.size and kumar_b.size and kumar_a.shape[1] > 0:
        max_sticks_to_show = min(kumar_a.shape[1], 10)
        vals = [kumar_a[:, k] for k in range(max_sticks_to_show)] + [kumar_b[:, k] for k in range(max_sticks_to_show)]
        labels = [f"a{k}" for k in range(max_sticks_to_show)] + [f"b{k}" for k in range(max_sticks_to_show)]
        ax.boxplot(vals, labels=labels, showfliers=False)
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        if kumar_a.shape[1] > max_sticks_to_show:
            ax.text(0.5, 0.95, f"showing first {max_sticks_to_show} sticks", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No Kumaraswamy sticks", ha="center", va="center", fontsize=8)
    ax.set_title("Kumaraswamy gate parameters")
    ax.set_ylabel("value")

    row5 = outer[4].subgridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.20)
    _plot_component_decodes(
        fig,
        row5[0, 0],
        model=model,
        device=device,
        comp_mean=comp_mean,
        pi_mean=pi_mean,
        examples=data.get("same_frame_examples", []),
    )
    ax = fig.add_subplot(row5[0, 1])
    ax.axis("off")
    slot_cos_off = np.nan
    if num_slots >= 2 and slot_cos.size:
        c = slot_cos.copy()
        c[np.eye(num_slots, dtype=bool)] = np.nan
        slot_cos_off = float(np.nanmax(c))
    comp_min_off = np.nan
    if comp_dist.shape[0] >= 2:
        d = comp_dist.copy()
        d[np.eye(d.shape[0], dtype=bool)] = np.nan
        comp_min_off = float(np.nanmin(d))
    summary = [
        f"fixed slots per input:              {num_slots}",
        f"posterior slot tokens:             {post_latents.shape[0]}",
        f"prior slot tokens:                 {prior_latents.shape[0]}",
        f"DPGMM components in frozen prior:  {K_seen}",
        f"active posterior components >1%:   {int((post_frac > 0.01).sum())}",
        f"active prior components >1%:       {int((prior_frac > 0.01).sum())}",
        f"mean gate K_eff:                   {float(np.mean(pi_eff)) if pi_eff.size else 0.0:.2f}",
        f"max off-diagonal slot cosine:      {slot_cos_off:.3f}",
        f"min off-diagonal comp L2:          {comp_min_off:.3g}",
        f"coverage components at {coverage_threshold:.0%}:        {selected_components.tolist() if selected_components.size else []}",
        f"coverage of selected components:   {displayed_coverage:.1%}",
        #"",
        #"Reading guide:",
        #"same-frame slots should differ across a row;",
        #"one component color confirms DPGMM usage collapse;",
        #"high off-diagonal slot cosine indicates similar slot means;",
        #"low component L2 indicates duplicate component means;",
        #"decoded component means should differ if the bank is useful.",
    ]
    ax.text(0.01, 0.98, "Slot-DPGMM diagnostics", ha="left", va="top", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.86, "\n".join(summary), ha="left", va="top", fontsize=9, family="monospace")

    _plot_slot_index_exemplars(
        fig,
        outer[5],
        data["slot_exemplars_by_index"],
        num_slots=num_slots,
        model=model,
        device=device,
    )

    fig.suptitle(f"Slot posterior/prior diagnostics at t={t_select}", fontsize=14)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def visualize_component_evolution(
    model,
    dataloader,
    device: torch.device,
    n_timesteps: int = 10,
    max_batches: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot posterior/prior component usage over time using slot assignments."""
    model.eval()
    post_list, prior_list, keff_post, keff_prior = [], [], [], []
    Kmax = 0
    for t in range(n_timesteps):
        data = extract_slot_latents_and_assignments(model, dataloader, device, max_batches=max_batches, t_select=t, max_points=20000, n_exemplars_per_component=1)
        post_pi = np.asarray(data["posterior_pi"], dtype=np.float64)
        prior_pi = np.asarray(data["prior_pi"], dtype=np.float64)
        post_mean = post_pi.mean(axis=0) if post_pi.ndim == 2 and post_pi.shape[0] else np.zeros((1,), dtype=np.float64)
        prior_mean = prior_pi.mean(axis=0) if prior_pi.ndim == 2 and prior_pi.shape[0] else np.zeros((1,), dtype=np.float64)
        Kmax = max(Kmax, post_mean.shape[0], prior_mean.shape[0])
        post_list.append(post_mean)
        prior_list.append(prior_mean)
        keff_post.append(float(_effective_k(post_pi).mean()) if post_pi.ndim == 2 and post_pi.shape[0] else 0.0)
        keff_prior.append(float(_effective_k(prior_pi).mean()) if prior_pi.ndim == 2 and prior_pi.shape[0] else 0.0)

    post_mat = np.zeros((n_timesteps, Kmax), dtype=np.float64)
    prior_mat = np.zeros((n_timesteps, Kmax), dtype=np.float64)
    for t in range(n_timesteps):
        post_mat[t, :post_list[t].shape[0]] = post_list[t]
        prior_mat[t, :prior_list[t].shape[0]] = prior_list[t]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    im0 = axes[0].imshow(post_mat, aspect="auto")
    axes[0].set_title("Mean posterior slot responsibility over time")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("timestep")
    fig.colorbar(im0, ax=axes[0], label="mean responsibility")

    im1 = axes[1].imshow(prior_mat, aspect="auto")
    axes[1].set_title("Mean prior slot responsibility over time")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("timestep")
    fig.colorbar(im1, ax=axes[1], label="mean responsibility")

    axes[2].plot(np.arange(n_timesteps), keff_post, label="posterior K_eff")
    axes[2].plot(np.arange(n_timesteps), keff_prior, label="prior K_eff")
    axes[2].set_title("Effective active components over time")
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("exp(H)")
    axes[2].legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
