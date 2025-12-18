"""
DPGMM-VRNN Latent Space Visualization

Provides t-SNE visualization of the top-layer latent space to demonstrate 
unsupervised clustering induced by the Dirichlet Process Gaussian Mixture Model prior.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Dict, Optional, Tuple, List


@torch.no_grad()
def compute_responsibilities(
    z_tokens: torch.Tensor,
    pi: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior responsibilities for DPGMM component assignment.
    
    Args:
        z_tokens: [N, C] latent vectors
        pi:       [N, K] mixing weights per token
        means:    [N, K, C] component means
        log_vars: [N, K, C] component log-variances
        
    Returns:
        resp: [N, K] posterior responsibilities
        k:    [N] argmax component assignment
    """
    log_pi = torch.log(pi.clamp_min(eps))  # [N, K]
    
    # Log-likelihood: log N(z|mu, var) per component
    inv_var = torch.exp(-log_vars)  # [N, K, C]
    quad = -0.5 * ((z_tokens.unsqueeze(1) - means)**2 * inv_var).sum(dim=-1)  # [N, K]
    norm = -0.5 * (log_vars + math.log(2.0 * math.pi)).sum(dim=-1)  # [N, K]
    
    logits = log_pi + norm + quad  # [N, K]
    resp = F.softmax(logits, dim=-1)  # [N, K]
    k = resp.argmax(dim=-1)  # [N]
    
    return resp, k

def compute_tsne_embedding(
    latents: np.ndarray,
    max_samples: int = 10000,
    perplexity: float = 30.0,
    pca_dims: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = latents.shape[0]

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
    else:
        idx = np.arange(n)

    X = latents[idx].astype(np.float32, copy=False)

    # PCA pre-reduction
    if X.shape[1] > pca_dims:
        X = PCA(n_components=pca_dims, random_state=random_state).fit_transform(X)

    n_use = X.shape[0]
    if n_use < 3:
        raise ValueError(f"Need at least 3 samples, got {n_use}")

    # Choose perplexity safely
    hard_max = n_use - 1
    heuristic_max = (n_use - 1) / 3.0
    perp = min(float(perplexity), float(heuristic_max), float(hard_max))

    # Pick a floor that cannot violate hard_max
    floor = 5.0
    perp = max(min(floor, hard_max), perp)  # floor only if feasible
    perp = float(min(perp, hard_max))       # final hard clamp

    # If data is tiny, t-SNE isn't that meaningful; optional fallback
    if n_use < 10 and perp > hard_max:
        perp = float(hard_max)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )

    emb = tsne.fit_transform(X)
    return emb, idx

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
) -> Dict[str, np.ndarray]:
    """
    Extract latent representations and DPGMM component assignments.
    
    This function processes batches through the encoder, extracts top-layer
    latents, and computes DPGMM responsibilities to determine cluster assignments.
    
    Args:
        model: DPGMMVariationalRecurrentAutoencoder instance
        dataloader: DataLoader yielding batches
        device: torch device
        max_batches: maximum batches to process
        t_select: timestep to select from sequences
        image_level: if True, return one point per image (mean over spatial)
                    if False, return token-level points
        max_points: safety cap on total points
        chunk: chunk size for responsibility computation
        use_rnn_context: if True, run RNN forward pass to get meaningful h_context
                        if False, use zeros (less meaningful for visualization)
    
    Returns:
        dict with keys:
            latents:     [M, D] numpy array of latent vectors
            assignments: [M] numpy array of component indices
            pi:          [M, K] numpy array of mixing weights
            labels:      [M] numpy array of ground-truth labels (if available)
            h_contexts:  [M, H] numpy array of RNN hidden states (if image_level)
    """
    model.eval()
    
    latents_all = []
    assign_all = []
    pi_all = []
    labels_all = []
    h_context_all = []
    
    # Get model dimensions
    hidden_dim = model.hidden_dim
    top_block = model.vdvae.decoder.dec_blocks[0]
    C_top = top_block.zdim
    res = top_block.base
    
    for bi, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
        if bi >= max_batches:
            break
        
        labels = None
        
        # Handle various dataloader formats
        if isinstance(batch, dict) and "observations" in batch:
            obs = batch["observations"]  # [B, T, C, H, W] or [B, C, H, W]
            
            if obs.dim() == 5:
                B, T = obs.shape[:2]
                images = obs[:, t_select]  # [B, C, H, W]
            else:
                B = obs.shape[0]
                T = 1
                images = obs
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            B = images.shape[0]
            T = 1
            if len(batch) > 1:
                labels = batch[1]
        else:
            images = batch
            B = images.shape[0]
            T = 1
        
        images = images.to(device)
        
        # Normalize images to [-1, 1]
        images = _to_minus1_1(images)  # [B, C, H, W]
        images_nhwc = images.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Compute h_context from RNN
        if use_rnn_context and isinstance(batch, dict) and "observations" in batch and batch["observations"].dim() == 5:
            h_context = _compute_rnn_context(model, batch, device, t_select)
        else:
            h_context = model.h0[-1].expand(B, -1).to(device)
        
        # Forward through VDVAE encoder
        vdvae_out = model.vdvae(images_nhwc, images_nhwc, h_context)
        top_q_mean_map = vdvae_out["top_q_mean_map"]  # [B, C_top, res, res]
        
        B_out, C_out, res_out, _ = top_q_mean_map.shape
        assert C_out == C_top and res_out == res
        
        # Flatten to tokens: [B*res*res, C_top]
        z_tokens = top_q_mean_map.permute(0, 2, 3, 1).reshape(B * res * res, C_top)
        
        # Build h_tokens with coordinate encoding (matching model.vdvae.sample)
        Hc = h_context.shape[1]
        h_map = h_context.view(B, Hc, 1, 1).expand(B, Hc, res, res).contiguous()
        h_map = model.vdvae.add_coord_no_proj(h_map, scale=0.05)
        h_tokens = h_map.permute(0, 2, 3, 1).reshape(B * res * res, Hc)
        
        # Get DPGMM prior parameters
        _, prior_params = model.prior(h_tokens)
        pi_tok = prior_params["pi"]         # [N, K]
        means = prior_params["means"]       # [N, K, C_top]
        log_vars = prior_params["log_vars"] # [N, K, C_top]
        
        # Compute responsibilities in chunks
        N = z_tokens.shape[0]
        k_tok = torch.empty(N, dtype=torch.long, device=device)
        
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            _, k = compute_responsibilities(
                z_tokens[s:e], pi_tok[s:e], means[s:e], log_vars[s:e]
            )
            k_tok[s:e] = k
        
        if image_level:
            # Aggregate to image level: mean latent, mode assignment
            z_img = top_q_mean_map.mean(dim=(2, 3))  # [B, C_top]
            k_img = k_tok.view(B, res * res)
            k_img = torch.mode(k_img, dim=1).values  # [B]
            pi_img = pi_tok.view(B, res * res, -1).mean(dim=1)  # [B, K]
            
            latents_all.append(z_img.cpu())
            assign_all.append(k_img.cpu())
            pi_all.append(pi_img.cpu())
            h_context_all.append(h_context.cpu())
            
            if labels is not None:
                if torch.is_tensor(labels):
                    labels_all.append(labels.cpu())
                else:
                    labels_all.append(torch.as_tensor(labels))
        else:
            # Token-level points
            latents_all.append(z_tokens.cpu())
            assign_all.append(k_tok.cpu())
            pi_all.append(pi_tok.cpu())
        
        # Safety cap
        total = sum(x.shape[0] for x in latents_all)
        if total >= max_points:
            break
    
    out = {
        "latents": torch.cat(latents_all, dim=0).numpy(),
        "assignments": torch.cat(assign_all, dim=0).numpy(),
        "pi": torch.cat(pi_all, dim=0).numpy(),
    }
    
    if len(labels_all) > 0:
        out["labels"] = torch.cat(labels_all, dim=0).numpy()
    
    if len(h_context_all) > 0:
        out["h_contexts"] = torch.cat(h_context_all, dim=0).numpy()
    
    return out


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W]
    if x.dtype == torch.uint8:
        x = x.float() / 127.5 - 1.0
    else:
        x = x.float()
        if x.numel() > 0 and (x.min() >= 0.0) and (x.max() <= 1.0):
            x = x * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)

@torch.no_grad()
def _compute_rnn_context(model, batch: Dict, device: torch.device, t_select: int) -> torch.Tensor:
    obs = batch["observations"].to(device)  # expect [B,T,C,H,W]
    if obs.dim() != 5:
        B = obs.shape[0]
        return model.h0[-1].expand(B, -1).to(device)

    actions = batch["actions"]
    dones   = batch["done"] if "done" in batch else None
    if actions is not None: actions = actions.to(device)
    if dones   is not None: dones   = dones.to(device)

    B, T = obs.shape[:2]
    t_select = int(min(t_select, T - 1))

    h = model.h0.expand(model.number_lstm_layer, B, -1).contiguous()
    c = model.c0.expand(model.number_lstm_layer, B, -1).contiguous()

    # IMPORTANT: advance ONLY through frames < t_select
    for t in range(t_select):
        x_t = _to_minus1_1(obs[:, t])
        x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

        h_context = h[-1]  # this is the context used for frame t

        vdvae_out = model.vdvae(x_t_nhwc, x_t_nhwc, h_context)

        # match training: sample z_map from q (not just mean)
        eps  = torch.randn_like(vdvae_out["top_q_logvar_map"])
        z_map = vdvae_out["top_q_mean_map"] + eps * torch.exp(0.5 * vdvae_out["top_q_logvar_map"])
        z_t = z_map.contiguous().view(B, -1)

        if actions is not None and actions.shape[1] > t:
            a_t = actions[:, t]
        else:
            a_t = torch.zeros(B, model.action_dim, device=device)

        if dones is not None and t > 0:
            mask_t = 1.0 - dones[:, t - 1].float()
        else:
            mask_t = torch.ones(B, device=device)

        rnn_in = torch.cat([z_t, a_t], dim=-1)
        _, (h, c) = model._rnn(rnn_in, h, c, mask_t)

    # Now h[-1] is exactly the pre-update context for frame t_select
    return h[-1]


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
) -> plt.Figure:
    """
    Create comprehensive visualization of DPGMM latent space clustering.
    
    Generates a 2x3 figure with:
    - t-SNE colored by DPGMM component
    - t-SNE colored by ground truth (if available)
    - Component utilization histogram
    - Top mixing weights bar chart
    - h_context t-SNE (if available)
    - Metrics summary
    
    Args:
        model: DPGMMVariationalRecurrentAutoencoder
        dataloader: data source
        device: torch device
        max_batches: batches to process
        max_samples: samples for t-SNE
        perplexity: t-SNE perplexity
        save_path: optional path to save figure
        class_names: optional list of class names for labels
        image_level: if True, aggregate to image level
        t_select: timestep to visualize
        use_rnn_context: if True, use RNN forward pass for h_context
        figsize: figure size
        
    Returns:
        matplotlib Figure object
    """
    # Extract data
    data = extract_latents_and_assignments(
        model, dataloader, device,
        max_batches=max_batches,
        image_level=image_level,
        t_select=t_select,
        use_rnn_context=use_rnn_context,
    )
    
    # Compute t-SNE embedding
    emb, idx = compute_tsne_embedding(
        data["latents"],
        max_samples=max_samples,
        perplexity=perplexity,
    )
    
    assignments = data["assignments"][idx]
    labels = data.get("labels")
    if labels is not None:
        labels = labels[idx]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Custom colormap for components
    n_components = int(assignments.max()) + 1
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_components)))
    if n_components > 20:
        colors = plt.cm.turbo(np.linspace(0, 1, n_components))
    
    # 1. t-SNE colored by DPGMM component
    cmap = ListedColormap(colors)
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(
        emb[:, 0], emb[:, 1],
        c=assignments, cmap=cmap, s=8, alpha=0.7
    )
    plt.colorbar(scatter1, ax=ax1, label="DPGMM Component")
    ax1.set_title("t-SNE by DPGMM Component Assignment", fontsize=11)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    
    # 2. t-SNE colored by ground truth
    ax2 = fig.add_subplot(2, 3, 2)
    if labels is not None:
        scatter2 = ax2.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap='Set1', s=8, alpha=0.7
        )
        cbar = plt.colorbar(scatter2, ax=ax2, label="Ground Truth Label")
        if class_names is not None:
            uniq = np.unique(labels)
            cbar.set_ticks(uniq)
            cbar.set_ticklabels([class_names[int(u)] for u in uniq])
        ax2.set_title("t-SNE by Ground Truth", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "No ground truth labels\navailable",
                ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Ground Truth (unavailable)", fontsize=11)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    
    # 3. Component utilization histogram
    ax3 = fig.add_subplot(2, 3, 3)
    all_assignments = data["assignments"]
    uniq, counts = np.unique(all_assignments, return_counts=True)
    order = np.argsort(-counts)
    sorted_counts = counts[order] / counts.sum()
    sorted_uniq = uniq[order]
    
    bars = ax3.bar(np.arange(len(sorted_counts)), sorted_counts, color='steelblue')
    
    # Highlight active components (>1% usage)
    active_mask = sorted_counts > 0.01
    for i, (bar, is_active) in enumerate(zip(bars, active_mask)):
        if is_active:
            bar.set_color('darkblue')
    
    ax3.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    ax3.set_title(f"Component Utilization (K_active={active_mask.sum()})", fontsize=11)
    ax3.set_xlabel("Component (sorted by usage)")
    ax3.set_ylabel("Proportion")
    ax3.legend()
    
    # 4. Top mixing weights
    ax4 = fig.add_subplot(2, 3, 4)
    mean_pi = data["pi"].mean(axis=0)
    top_k = min(15, len(mean_pi))
    top_indices = np.argsort(-mean_pi)[:top_k]
    
    ax4.bar(np.arange(top_k), mean_pi[top_indices], color='coral')
    ax4.set_xticks(np.arange(top_k))
    ax4.set_xticklabels([f"K{i}" for i in top_indices], rotation=45)
    ax4.set_title("Top Mean Mixing Weights (π)", fontsize=11)
    ax4.set_xlabel("Component")
    ax4.set_ylabel("Mean π")
    
    # 5. h_context visualization (if available)
    ax5 = fig.add_subplot(2, 3, 5)
    if "h_contexts" in data and data["h_contexts"].shape[0] > 10:
        h_contexts = data["h_contexts"][idx]
        h_emb, _ = compute_tsne_embedding(h_contexts, max_samples=len(h_contexts))
        scatter5 = ax5.scatter(
            h_emb[:, 0], h_emb[:, 1],
            c=assignments, cmap='tab20', s=8, alpha=0.7
        )
        ax5.set_title("h_context t-SNE (colored by DPGMM)", fontsize=11)
        ax5.set_xlabel("t-SNE 1")
        ax5.set_ylabel("t-SNE 2")
    else:
        ax5.text(0.5, 0.5, "h_context\nnot available",
                ha="center", va="center", transform=ax5.transAxes, fontsize=12)
        ax5.set_title("RNN Hidden State Visualization", fontsize=11)
    
    # 6. Metrics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = []
    metrics_text.append(f"Total samples: {len(data['latents']):,}")
    metrics_text.append(f"t-SNE samples: {len(emb):,}")
    metrics_text.append(f"Total components (K): {model.max_K}")
    metrics_text.append(f"Active components (>1%): {active_mask.sum()}")
    metrics_text.append(f"Top-1 component usage: {sorted_counts[0]:.1%}")
    metrics_text.append(f"Top-5 component coverage: {sorted_counts[:5].sum():.1%}")
    
    # Entropy of component distribution
    pi_mean = mean_pi[mean_pi > 1e-8]
    entropy = -np.sum(pi_mean * np.log(pi_mean + 1e-10))
    max_entropy = np.log(len(pi_mean))
    metrics_text.append(f"π entropy: {entropy:.2f} / {max_entropy:.2f}")
    
    if labels is not None:
        nmi = normalized_mutual_info_score(labels, assignments)
        ari = adjusted_rand_score(labels, assignments)
        metrics_text.append(f"NMI: {nmi:.3f}")
        metrics_text.append(f"ARI: {ari:.3f}")
    
    metrics_str = "\n".join(metrics_text)
    ax6.text(0.1, 0.9, "Clustering Metrics", fontsize=14, fontweight='bold',
             transform=ax6.transAxes, va='top')
    ax6.text(0.1, 0.75, metrics_str, fontsize=11, transform=ax6.transAxes,
             va='top', family='monospace')
    
    # Title
    title = f"DPGMM Latent Space Analysis (image_level={image_level}, t={t_select})"
    if labels is not None:
        title += f"  |  NMI={nmi:.3f}"
    fig.suptitle(title, fontsize=13, y=1.02)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    plt.close(fig)
    return fig


def visualize_component_evolution(
    model,
    dataloader,
    device: torch.device,
    n_timesteps: int = 10,
    max_batches: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize how DPGMM component assignments evolve over time in sequences.
    
    This shows whether the model uses different components for different
    temporal dynamics (e.g., motion phases).
    """
    model.eval()
    
    component_counts_per_t = {t: [] for t in range(n_timesteps)}
    
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(dataloader, desc="Analyzing temporal evolution")):
            if bi >= max_batches:
                break
            
            if not isinstance(batch, dict) or "observations" not in batch:
                continue
            
            obs = batch["observations"].to(device)  # [B, T, C, H, W]
            B, T = obs.shape[:2]
            
            if T < n_timesteps:
                continue
            
            for t in range(min(n_timesteps, T)):
                data_t = extract_latents_and_assignments(
                    model,
                    [{"observations": obs[:, :t+1], 
                      "actions": batch.get("actions", torch.zeros(B, t+1, model.action_dim)).to(device)[:, :t+1],
                      "done": batch.get("done", torch.zeros(B, t+1)).to(device)[:, :t+1]}],
                    device,
                    max_batches=1,
                    t_select=t,
                    image_level=True,
                    use_rnn_context=True,
                )
                component_counts_per_t[t].extend(data_t["assignments"].tolist())
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Heatmap of component usage over time
    K = model.max_K
    usage_matrix = np.zeros((n_timesteps, K))
    
    for t in range(n_timesteps):
        if len(component_counts_per_t[t]) > 0:
            counts = np.bincount(component_counts_per_t[t], minlength=K)
            usage_matrix[t] = counts / counts.sum()
    
    im = axes[0].imshow(usage_matrix.T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Component")
    axes[0].set_title("DPGMM Component Usage Over Time")
    plt.colorbar(im, ax=axes[0], label="Usage proportion")
    
    # Entropy over time
    entropies = []
    for t in range(n_timesteps):
        p = usage_matrix[t]
        p = p[p > 1e-8]
        if len(p) > 0:
            entropies.append(-np.sum(p * np.log(p + 1e-10)))
        else:
            entropies.append(0)
    
    axes[1].plot(range(n_timesteps), entropies, 'o-', color='steelblue', linewidth=2)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Component Distribution Entropy Over Time")
    axes[1].axhline(y=np.log(K), color='red', linestyle='--', alpha=0.5, label=f'Max entropy (log {K})')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved temporal evolution to: {save_path}")
    plt.close(fig)
    return fig