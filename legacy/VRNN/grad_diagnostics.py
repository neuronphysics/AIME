import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
def compute_param_cosines(named_params, task_losses, patterns=None):
    """
    Build unit gradient directions for each task w.r.t. a selected parameter set.
    named_params: list(model.named_parameters())
    patterns: optional list[str] of substrings to filter parameters by name
    Returns:
        G: [T, D] (unit vectors), cos: [T, T]
    """
    if patterns:
        params = [p for (n, p) in named_params if any(k in n for k in patterns) and p.requires_grad]
    else:
        params = [p for (_, p) in named_params if p.requires_grad]

    grads = []
    for obj in task_losses:
        if not torch.is_tensor(obj):
            continue
        g_list = torch.autograd.grad(obj, params, retain_graph=True, allow_unused=True)
        flat = []
        for p, g in zip(params, g_list):
            flat.append((torch.zeros_like(p) if g is None else g).contiguous().view(-1))
        v = torch.cat(flat)
        n = v.norm()
        grads.append(v / (n + 1e-12) if float(n) > 0 else torch.zeros_like(v))

    if len(grads) == 0:
        raise RuntimeError("No task loss produced grads for selected parameters.")

    G = torch.stack(grads, dim=0).cpu()
    cos = (G @ G.t()).clamp(-1, 1)
    return G, cos

def compute_shared_cosines( shared_repr, task_losses):
    z = shared_repr
    grads = []
    for obj in task_losses:
        if not torch.is_tensor(obj):
            continue
        g = torch.autograd.grad(obj, z, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        if g is None:
            # loss does not depend on z; skip
            g= torch.zeros_like(z)
        g = g.detach().view(-1)
        n = g.norm()
        grads.append(g / (n + 1e-12) if float(n) > 0 else torch.zeros_like(g))

    if len(grads) == 0:
        raise RuntimeError("No task loss depends on shared_repr (z).")

    G = torch.stack(grads, dim=0).cpu()
    cos = (G @ G.t()).clamp(-1, 1)
    return G, cos

def component_param_mask(named_params, module_patterns):
    return [p for (name,p) in named_params if any(k in name for k in module_patterns) and p.requires_grad]

@torch.no_grad()
def zero_grads(params): 
    for p in params:
        if p.grad is not None: 
            p.grad.zero_()

def per_task_component_grad_norms(model, task_losses, component_groups):
    """
    component_groups
    Returns: dict task_name -> dict component -> grad_norm
    """
    named = list(model.named_parameters())
    results = {}
    for t_name, loss in task_losses.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        task_dict = {}
        for comp, patterns in component_groups.items():
            params = component_param_mask(named, patterns)
            if not params: 
                task_dict[comp] = 0.0
                continue
            s = 0.0
            n = 0
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach()
                    s += float(g.norm(2).item()**2)
                    n += 1
            task_dict[comp] = (s ** 0.5) / max(n,1)
        results[t_name] = task_dict
        model.zero_grad(set_to_none=True)  # clear for next task
    return results

def _grad_symbol(mode):
    return r'$\nabla_z$' if mode == "shared" else r'$\nabla_{\theta}$'

def _subset_label(param_patterns):
    if not param_patterns:
        return ""
    return " [" + ", ".join(param_patterns) + "]"

def title_for_dirs(mode, param_patterns=None):
    return f"Directions of {_grad_symbol(mode)}{_subset_label(param_patterns)} (PCA-2D)"

def title_for_cos(mode, param_patterns=None):
    return f"Mean cosine({_grad_symbol(mode)}) across tasks{_subset_label(param_patterns)}"

def quiver_shared_dirs(G, task_names, ax=None, scale=1.0, mode="shared", param_patterns=None, 
                       title=None, label_frac=0.5, offset_frac=0.06, fontsize=9, task_colors=None, cmap_name='tab10', flip_upside_down=True):
    """
    G: [T, D] unit-norm grads on z (mode='shared') or params (mode='params')
    """
    # PCA to 2D
    X = G.cpu().numpy()
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P2 = X @ Vt[:2].T

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    else:
        fig = ax.figure
    ax.axhline(0, ls='--', lw=0.5)
    ax.axvline(0, ls='--', lw=0.5)
    if task_colors is None:
        cmap = plt.cm.get_cmap(cmap_name, len(task_names))
        colors = [cmap(i) for i in range(len(task_names))]
        ax.quiver(np.zeros(len(P2)), np.zeros(len(P2)), P2[:,0], P2[:,1],
              angles='xy', scale_units='xy', scale=scale, color=colors)
    else:
        colors = [task_colors[name] for name in task_names]
        ax.quiver(np.zeros(len(P2)), np.zeros(len(P2)), P2[:,0], P2[:,1],
              angles='xy', scale_units='xy', scale=scale, color=colors)

    for i,((dx, dy), name) in enumerate(zip(P2, task_names)):
        # skip zero-length vectors
        length = float(np.hypot(dx, dy))
        if length < 1e-9:
            continue

        # position along the ray
        xm = dx * float(label_frac)
        ym = dy * float(label_frac)

        # unit normal (perpendicular) for "above the line" offset
        nx, ny = -dy, dx
        nlen = float(np.hypot(nx, ny))
        if nlen > 0:
            nx, ny = nx / nlen, ny / nlen
        # offset magnitude proportional to ray length
        off = float(offset_frac) * length
        xlab = xm + nx * off
        ylab = ym + ny * off

        # rotation to match ray angle (degrees)
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if flip_upside_down and (angle > 90 or angle < -90):
            angle = angle - 180 
        ax.text(
            xlab, ylab, name,
            rotation=angle,
            rotation_mode='anchor',
            ha='center', va='bottom',  # Changed from 'center' to 'bottom'
            fontsize=fontsize, color=colors[i],
            zorder=3,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')  # Optional: add background
        )
    ax.set_title(title or title_for_dirs(mode, param_patterns))
    fig.tight_layout()
    return fig

def make_cosine_heatmap(cos: torch.Tensor, task_names, title=None, mode="shared", param_patterns=None):
    fig, ax = plt.subplots(figsize=(4.5,4.0))
    im = ax.imshow(cos.cpu().numpy(), vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_title(title or title_for_cos(mode, param_patterns), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def _unit_grads_z(shared_repr, task_losses):
    """Return unit ∇_z for each task (T x D) and cosine matrix (T x T)."""
    G, cos = compute_shared_cosines( shared_repr=shared_repr, task_losses=task_losses)
    return G, cos

class GradDiagnosticsAggregator:

    def __init__(self, task_names, component_groups=None, average_component_norms=False):
        self.task_names = list(task_names)
        self.T = len(task_names)
        self.sum_unit_grads = None    # [T, D]
        self.sum_cos = torch.zeros(self.T, self.T)
        self.count = 0

        # shared-z amplitude (RMS over batches)
        self._sum_sq_shared = torch.zeros(self.T)

        # optional per-component (very compute-heavy)
        self.component_groups = component_groups or {}
        self.average_component_norms = bool(average_component_norms)
        self._comp_sum_sq = {t: defaultdict(float) for t in self.task_names}

    @torch.no_grad()
    def _accumulate_param_amp(self, named_params, task_losses, patterns=None):
        # Same semantics as _accumulate_shared_amp but over a parameter vector
        if patterns:
            params = [p for (n, p) in named_params if any(k in n for k in patterns) and p.requires_grad]
        else:
            params = [p for (_, p) in named_params if p.requires_grad]

        norms = []
        for L in task_losses:
            if not torch.is_tensor(L):
                continue
            g_list = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            flat = []
            for p, g in zip(params, g_list):
                flat.append((torch.zeros_like(p) if g is None else g).contiguous().view(-1))
            v = torch.cat(flat)
            norms.append(v.norm().detach().cpu())
        if norms:
            norms = torch.stack(norms, dim=0)
            self._sum_sq_shared += norms**2

    @torch.no_grad()
    def _accumulate_shared_amp(self, shared_repr, task_losses):
        norms = []
        for L in task_losses:
            if not torch.is_tensor(L):
                continue
            g = torch.autograd.grad(L, shared_repr, retain_graph=True, only_inputs=True, allow_unused=True)[0]
            if g is None:
                norms.append(torch.tensor(0.0))
            else:
                norms.append(g.view(-1).norm().detach().cpu())
        if norms:
            norms = torch.stack(norms, dim=0)
            self._sum_sq_shared += norms**2

    def update(self, model, shared_repr, task_losses, named_params=None, param_patterns=None):
        """
        If shared_repr is not None: compute directions and amplitudes w.r.t. z (original behavior).
        If shared_repr is None: compute them w.r.t. selected parameters (named_params / param_patterns).
        """
        if shared_repr is not None:
            # ----- original ∇z path -----
            assert shared_repr.requires_grad, "shared_repr must require grad during eval diagnostics"
            G, cos = _unit_grads_z(shared_repr, task_losses)  # unit ∇z
            if self.sum_unit_grads is None:
                self.sum_unit_grads = torch.zeros_like(G)
            self.sum_unit_grads += G.detach()
            self.sum_cos += cos.detach()
            self.count += 1
            self._accumulate_shared_amp(shared_repr, task_losses)
        else:
            if named_params is None:
                named_params = list(model.named_parameters())
            G, cos = compute_param_cosines(named_params, task_losses, patterns=param_patterns)
            if self.sum_unit_grads is None:
                self.sum_unit_grads = torch.zeros_like(G)
            self.sum_unit_grads += G.detach()
            self.sum_cos += cos.detach()
            self.count += 1
            self._accumulate_param_amp(named_params, task_losses, patterns=param_patterns)

        if self.average_component_norms and len(self.component_groups) > 0:
            if named_params is None:
                named_params = list(model.named_parameters())
            tdict = {t: L for t, L in zip(self.task_names, task_losses)}
            comp = per_task_component_grad_norms(model, tdict, self.component_groups)
            for t_name, d in comp.items():
                for comp_name, val in d.items():
                    self._comp_sum_sq[t_name][comp_name] += float(val) ** 2


    def finalize(self):
        # mean of unit vectors then renormalize (directional average)
        mean_unit = self.sum_unit_grads / max(self.count, 1)
        mean_dirs = mean_unit / (mean_unit.norm(dim=1, keepdim=True) + 1e-12)  # [T,D]

        # arithmetic mean of cosine matrices (bounded, interpretable)
        cos_mean = self.sum_cos / max(self.count, 1)

        # RMS amplitude across batches
        grad_amp_rms = torch.sqrt(self._sum_sq_shared / max(self.count, 1))

        # Optional per-component RMS
        comp_amp_rms = None
        if self.average_component_norms and len(self.component_groups) > 0:
            comp_amp_rms = {t: {c: np.sqrt(v / max(self.count, 1))
                                for c, v in dd.items()} for t, dd in self._comp_sum_sq.items()}
        return mean_dirs, cos_mean, grad_amp_rms, comp_amp_rms

    def tensorboard_log(self, writer, tag_prefix, global_step, quiver_scale=1.0, mode="params", param_patterns=None):
        mean_dirs, cos_mean, grad_amp_rms, comp_amp_rms = self.finalize()
        task_colors = {
            "elbo":           "#1f77b4",  # blue
            "perceiver":      "#ff7f0e",  # orange
            "predictive":     "#2ca02c",  # green
            "adversarial":     "#d62728",  # red
        }
        # Quiver (2D PCA of mean dirs) — ensure we pass a Figure to TB
        fig_q = quiver_shared_dirs(mean_dirs, self.task_names, scale=quiver_scale, mode=mode, param_patterns=param_patterns, task_colors=task_colors)
        writer.add_figure(f"{tag_prefix}/quiver_mean_dirs", fig_q, global_step=global_step)
        plt.close(fig_q)

        # Cosine heatmap
        fig_c = make_cosine_heatmap(cos_mean, self.task_names, title=f"Mean cosine({_grad_symbol(mode)}) across eval", mode=mode, param_patterns=param_patterns)
        writer.add_figure(f"{tag_prefix}/cosine_matrix", fig_c, global_step=global_step)
        plt.close(fig_c)

        # ∇ RMS amplitude (one scalar per task)
        for i, t in enumerate(self.task_names):
            writer.add_scalar(f"{tag_prefix}/amp_rms/{t}", float(grad_amp_rms[i].item()), global_step)

        if comp_amp_rms:
            for t in self.task_names:
                comp_dict = comp_amp_rms.get(t) or {}
                if not comp_dict:
                    continue
                # Sort to keep the plot readable
                keys, vals = zip(*sorted(comp_dict.items(), key=lambda kv: -kv[1]))
                fig_w = max(4.0, 0.6 * len(keys))
                fig, ax = plt.subplots(figsize=(fig_w, 2.8))
                ax.bar(range(len(keys)), vals)
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels(keys, rotation=45, ha='right')
                ax.set_ylabel("RMS ||grad||")
                ax.set_title(f"{t}: per-component RMS grad")
                fig.tight_layout()
                writer.add_figure(f"{tag_prefix}/components/{t}", fig, global_step)
                plt.close(fig)

