import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies."""
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        """Define and initialize trainable parameters required by specific weighting methods."""
        pass

    def _compute_grad_dim(self):
        """Compute and cache the total gradient dimension across all shared parameters."""
        if hasattr(self, "grad_index"):
            return
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        """Flatten all shared parameter gradients into a single vector."""
        params = list(self.get_share_params())
        dev = params[0].device if len(params) else torch.device("cpu")
        grad = torch.zeros(self.grad_dim, device=dev)
        count = 0
        for p in params:
            if p.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = p.grad.view(-1).detach()
            count += 1
        return grad
    
    def zero_grad_share_params(self):
        """Set gradients of the shared parameters to zero."""
        for p in self.get_share_params():
            if p.grad is not None:
                p.grad.zero_()

    def _compute_grad(self, losses, mode, rep_grad=False):
        """
        Compute per-task gradients.
        
        Args:
            losses: List of per-task losses
            mode: 'backward' or 'autograd'
            rep_grad: Whether to compute representation gradients
        """
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True)
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    params = list(self.get_share_params())
                    grad_list = torch.autograd.grad(
                        losses[tn], params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    flat = []
                    for g, p in zip(grad_list, params):
                        if g is None:
                            flat.append(torch.zeros_like(p, device=p.device).view(-1))
                        else:
                            flat.append(g.contiguous().view(-1))
                    grads[tn] = torch.cat(flat)
                else:
                    raise ValueError(f'No support {mode} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads


    def _reset_grad(self, new_grads):
        """Reset shared parameter gradients to new values."""
        offset = 0
        for p, n in zip(self.get_share_params(), self.grad_index):
            slice_ = new_grads[offset: offset + n].contiguous()
            offset += n

            slice_ = slice_.view_as(p).to(p.device, p.dtype)

            if p.grad is None:
                p.grad = slice_.detach().clone()
            else:
                p.grad.detach_()
                p.grad.copy_(slice_)
            
    def _get_grads(self, losses, mode='backward'):
        """Returns gradients of representations or shared parameters."""
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        """Reset gradients and make a backward-like accumulation."""
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        """
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass


# ==============================================
# RGB (Rotation-Based Gradient Balancing)
# ==============================================
class RGB(AbsWeighting):
    r"""
    Rotation-Based Gradient Balancing (RGB).
    
    Paper: "Preserving Gradient Harmony: A Rotation-Based Gradient Balancing 
           for Multi-Task Conflict Remedy" (ICLR 2026 submission)
    
    Key features:
    - Rotates normalized task gradients toward consensus direction
    - Minimizes conflicts while preserving task specificity
    """

    def __init__(self):
        super().__init__()
        # Core hyperparameters (align with paper defaults)
        self.mu = 0.9               # EMA momentum for consensus direction
        self.lambd = 1.0            # Proximity weight λ (balance term)
        self.alpha_steps = 3        # Inner optimization steps (paper uses 3)
        self.lr_inner = 0.2         # Learning rate for angle updates
        self.update_interval = 1    # Solve rotation every N steps (1=always)
        self.eps = 1e-8            # Numerical stability constant

        # Adaptive step size parameters (Equation 2 from paper)
        self.use_adaptive_steps = False
        self.alpha_min = 1
        self.alpha_max = 10
        self.k_std = 1.0
        self.align_threshold = 0.999
        # Persistent state
        self.ema_direction = None   # Consensus direction d_t (EMA of mean gradients)
        self.alpha_state = None     # Previous rotation angles (warm-start)
        self.prev_losses = None     # For adaptive step computation
        self.step_count = 0         # Global step counter

    # ========== Core Helper Methods ==========
    
    def _normalize_gradients(self, grads):
        """
        Normalize gradients to unit vectors.
        
        Args:
            grads: [T, D] tensor of task gradients
        Returns:
            gbar: [T, D] normalized gradients
            norms: [T, 1] original norms
        """
        norms = grads.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return grads / norms, norms

    def _orthogonal_helpers(self, gbar, d_t):
        """
        Compute orthogonal correction vectors w_i.
        
        Paper: w_i = (d_t - (ḡ_i^T d_t)ḡ_i) / ||d_t - (ḡ_i^T d_t)ḡ_i||
        
        This is the component of d_t orthogonal to ḡ_i, forming the basis
        for rotation along with ḡ_i.
        
        Args:
            gbar: [T, D] normalized gradients
            d_t: [D] consensus EMA direction
        Returns:
            w: [T, D] orthogonal unit vectors
            proj: [T] projection magnitudes (ḡ_i^T d_t) 
        """
        proj = (gbar * d_t.unsqueeze(0)).sum(dim=1)  # [T]
        w_raw = d_t.unsqueeze(0) - proj.unsqueeze(1) * gbar                      # [T, D]
        w = w_raw / w_raw.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return w, proj

    def _rotate(self, gbar, w, alpha):
        """
        Apply rotation operator: r_i(α_i) = cos(α_i)ḡ_i + sin(α_i)w_i
        
        This smoothly interpolates between ḡ_i (α=0) and w_i (α=π/2),
        where w_i is aligned with the consensus direction d_t.
        
        Args:
            gbar: [T, D] normalized gradients
            w: [T, D] orthogonal helper vectors
            alpha: [T] rotation angles
        Returns:
            r: [T, D] rotated unit vectors
        """
        a = alpha.unsqueeze(1)                                    # [T, 1]
        r = torch.cos(a) * gbar + torch.sin(a) * w                # [T, D]
        # Normalize for numerical stability (theoretically already unit norm)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return r

    def _conflict_proximity_loss_fast(self, r, gbar):
        """
        O(TD) objective function (paper Equation 1).
        
        Conflict term (normalized to [0,1]):
            (1/(T(T-1))) Σ_{i<j} (1 - r_i^T r_j) 
          = (1/(T(T-1))) [T(T-1)/2 - Σ_{i<j} r_i^T r_j]
          
        Proximity term (normalized to [0,1]):
            (1/4T) Σ_i ||r_i - ḡ_i||²
            
        Total: conflict + λ * proximity
        
        **Efficiency trick**: Transform pairwise O(T²) conflict computation
        into O(TD) using: Σ_{i<j} r_i^T r_j = (1/2)[||Σ_i r_i||² - T]
        
        Args:
            r: [T, D] rotated gradients
            gbar: [T, D] original normalized gradients
        Returns:
            loss: scalar
        """
        T = r.shape[0]

        # Edge case: single task
        if T == 1:
            proximity = ((r - gbar).pow(2).sum()) / 4.0
            return proximity

        # Conflict term (O(D) sum, then O(D) dot product)
        s = r.sum(dim=0)                              # [D] - sum over tasks
        sum_dot_pairs = 0.5 * (s @ s - T)             # Σ_{i<j} r_i·r_j
        pair_count = T * (T - 1) / 2                  # Number of pairs
        conflict = (pair_count - sum_dot_pairs) / (T * (T - 1))  # Normalized to [0,1]

        # Proximity term (squared distance, normalized to [0,1])
        proximity = ((r - gbar).pow(2).sum()) / (4.0 * T)
        
        return conflict + self.lambd * proximity

    def _optimize_angles(self, gbar, w, alpha_init, steps, lr, flag=None):
        """
        Inner optimization: minimize rotation objective w.r.t. angles.
        
        Uses manual gradient descent (no optimizer overhead) with:
        - Warm-start from alpha_init
        - Small number of steps (3-10)
        - Clamping to [0, π/2) to maintain valid rotation
        
        Args:
            gbar: [T, D] normalized gradients
            w: [T, D] orthogonal helpers
            alpha_init: [T] initial angles (warm-start)
            steps: number of GD iterations
            lr: learning rate for angle updates
        Returns:
            alpha: [T] optimized angles
        """
        alpha = alpha_init.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            # Forward: compute loss
            r = self._rotate(gbar, w, alpha)
            loss = self._conflict_proximity_loss_fast(r, gbar)
            
            # Backward: compute gradients w.r.t. alpha
            if alpha.grad is not None:
                alpha.grad.zero_()
            loss.backward()
            
            # Manual GD step with clamping
            with torch.no_grad():
                if flag is not None:
                    alpha.grad[flag] = 0.0  # No update if already aligned
                alpha -= lr * alpha.grad
                alpha.clamp_(0.0, math.pi/2 - 1e-6)  # Keep in valid range
                if flag is not None:
                    alpha[flag] = 0.0  # Ensure aligned tasks remain at 0 angle
        return alpha.detach()

    def _compute_adaptive_alpha_steps(self, losses):
        """
        Adaptive step size based on loss variability (paper Equation 2).
        
        Intuition: When task losses change erratically (high std of relative
        change), we need more steps to find accurate rotation angles.
        
        Formula: α_steps = α_min + (α_max - α_min) * (s_t / (s_t + k_std))
        where s_t = std(δ_i^t) and δ_i^t = (L_i^t - L_i^{t-1}) / L_i^{t-1}
        
        Args:
            losses: List of current task losses
        Returns:
            steps: int in [alpha_min, alpha_max]
        """
        dev = self.ema_direction.device if self.ema_direction is not None else losses[0].device
        cur = torch.tensor([l.item() for l in losses], device=dev, dtype=torch.float32)
        
        # Initialize on first call
        if self.prev_losses is None or len(self.prev_losses) != len(cur):
            self.prev_losses = cur
            return self.alpha_steps
        
        # Compute relative loss changes
        delta = (cur - self.prev_losses) / (self.prev_losses.abs() + self.eps)
        s_t = delta.std()
        
        # Adaptive formula (Equation 2)
        steps = self.alpha_min + (self.alpha_max - self.alpha_min) * \
                (s_t / (s_t + self.k_std + self.eps))
        
        self.prev_losses = cur
        return int(steps.clamp(self.alpha_min, self.alpha_max).item())

    # ========== Main API ==========
    
    def backward(self, losses, **kwargs):
        """
        Main RGB algorithm (matches paper Algorithm 1).
        
        Steps:
        1. Compute and normalize task gradients
        2. Update EMA consensus direction d_t
        3. Optionally solve rotation angles α_i (every update_interval steps)
        4. Apply rotations and aggregate equally
        5. Reset gradients to aggregated direction
        
        Args:
            losses: List[Tensor] of per-task losses
            **kwargs: Runtime hyperparameter overrides
                - mu: EMA momentum (default 0.9)
                - lambd: proximity weight (default 1.0)
                - alpha_steps: inner steps (default 3)
                - lr_inner: inner LR (default 0.2)
                - update_interval: solve frequency (default 1)
                - use_adaptive_steps: enable adaptive (default False)
                - alpha_min, alpha_max, k_std: adaptive params
        
        Returns:
            batch_weight: np.array of shape [T] (always equal weights for RGB)
        """
        # === Hyperparameter override (runtime) ===
        self.mu = kwargs.get('mu', self.mu)
        self.lambd = kwargs.get('lambd', self.lambd)
        self.alpha_steps = int(kwargs.get('alpha_steps', self.alpha_steps))
        self.lr_inner = kwargs.get('lr_inner', self.lr_inner)
        self.update_interval = int(kwargs.get('update_interval', self.update_interval))
        self.use_adaptive_steps = kwargs.get('use_adaptive_steps', self.use_adaptive_steps)
        self.alpha_min = kwargs.get('alpha_min', self.alpha_min)
        self.alpha_max = kwargs.get('alpha_max', self.alpha_max)
        self.k_std = kwargs.get('k_std', self.k_std)

        # RGB doesn't support rep_grad mode
        if self.rep_grad:
            raise ValueError('RGB does not support representation gradients (rep_grad=True)')

        # === Step 1: Collect and normalize task gradients ===
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')      # [T, D]
        T, D = grads.shape
        device = grads.device

        gbar, norms = self._normalize_gradients(grads)           # [T, D], [T, 1]

        # === Step 2: Update EMA consensus direction ===
        y = gbar.mean(dim=0)                                     # [D] mean direction
        y = y / y.norm().clamp_min(self.eps)                     # Normalize
        
        if (self.ema_direction is None) or (self.ema_direction.shape[0] != D):
            self.ema_direction = y.detach().clone()
        else:
            # EMA update: d_t = μ*d_{t-1} + (1-μ)*y
            self.ema_direction = (self.mu * self.ema_direction + (1.0 - self.mu) * y)
            self.ema_direction = self.ema_direction / self.ema_direction.norm().clamp_min(self.eps)

        # === Step 3: Decide whether to solve rotation ===
        self.step_count += 1
        do_rotation = (self.step_count % self.update_interval == 0)

        # Fast path: skip rotation solving (equal-weight mean of normalized grads)
        if (not do_rotation) or T <= 1:
            new_grads = gbar.mean(dim=0)                         # [D]
            self._reset_grad(new_grads)
            return np.ones(T)

        # Optional: adaptive step size based on loss variability
        current_alpha_steps = (self._compute_adaptive_alpha_steps(losses) 
                               if self.use_adaptive_steps else self.alpha_steps)

        # === Step 4: Build rotation operator components ===
        w, proj = self._orthogonal_helpers(gbar, self.ema_direction)   # [T, D]
        flag = proj.abs() > self.align_threshold

        # Warm-start angles from previous iteration
        if (self.alpha_state is None) or (self.alpha_state.shape[0] != T):
            alpha0 = torch.zeros(T, device=device)
        else:
            alpha0 = self.alpha_state.to(device).clamp(0.0, math.pi/2 - 1e-6)
        alpha0[flag] = 0.0  # No rotation if already aligned
        # === Step 5: Optimize rotation angles ===
        alpha_star = self._optimize_angles(
            gbar=gbar, w=w, alpha_init=alpha0,
            steps=current_alpha_steps, lr=self.lr_inner, flag=flag
        )
        self.alpha_state = alpha_star.detach().clone()

        # === Step 6: Apply final rotation and aggregate ===
        r_final = self._rotate(gbar, w, alpha_star)              # [T, D]
        new_grads = r_final.mean(dim=0)                          # [D] equal-weight mean
        self._reset_grad(new_grads)
        #cos_to_agg = torch.nn.functional.cosine_similarity(gbar, new_grads.unsqueeze(0), dim=1)   # [T]
        # Project each rotated grad on the aggregate to get contribution
        #contrib = (r_final @ new_grads) / (new_grads.norm()**2 + self.eps)                         # [T]

        # Stash for logging
        #self.last_metrics = {
        #    "cos_to_agg":  cos_to_agg.detach().cpu(),
        #    "contrib":     contrib.detach().cpu(),
        #    "norms":       norms.squeeze(1).detach().cpu(),   # from _normalize_gradients
        #}

        # Return equal weights (RGB doesn't use adaptive task weighting)
        return np.ones(T)



class GradNorm(AbsWeighting):
    r"""Gradient Normalization (GradNorm).

    Chen et al., ICML 2018: 
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks".
    """

    def __init__(self):
        super().__init__()

    def init_param(self):
        # one scale per task
        self.loss_scale = nn.Parameter(
            torch.ones(self.task_num, device=self.device)
        )

    def _canonicalize_losses(self, losses):
        """Convert different loss containers to a list of tensors [L_0, ..., L_{T-1}]."""
        if isinstance(losses, (list, tuple)):
            return list(losses)
        if isinstance(losses, dict):
            # assumes integer keys 0..task_num-1
            return [losses[i] for i in range(self.task_num)]
        if isinstance(losses, torch.Tensor):
            # e.g., shape [task_num]
            if losses.ndim == 0:
                # single scalar loss; treat as 1 task
                return [losses]
            return [losses[i] for i in range(self.task_num)]
        raise TypeError(f"Unsupported type for losses: {type(losses)}")

    def backward(self, losses, **kwargs):
        alpha = kwargs["alpha"]
        log_grads = kwargs.get("log_grads", False)

        # Canonicalize
        losses_list = self._canonicalize_losses(losses)

        if self.epoch >= 1:
            # ----- Main GradNorm logic -----
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)

            grads = self._get_grads(losses_list, mode="backward")
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]

            # grads: [task_num, grad_dim]
            G_per_loss = torch.norm(
                loss_scale.unsqueeze(1) * grads, p=2, dim=-1
            )  # [task_num]
            G = G_per_loss.mean(0)  # scalar

            L_i = torch.tensor(
                [
                    losses_list[tn].item() / self.train_loss_buffer[tn, 0]
                    for tn in range(self.task_num)
                ],
                device=self.device,
                dtype=self.loss_scale.dtype,
            )

            r_i = L_i / L_i.mean()
            constant_term = (G * (r_i ** alpha)).detach()  # [task_num]

            L_grad = (G_per_loss - constant_term).abs().sum(0)  # scalar
            L_grad.backward()

            loss_weight = loss_scale.detach().clone()

            if self.rep_grad:
                new_grads = self._backward_new_grads(
                    loss_weight, per_grads=per_grads
                )
            else:
                new_grads = self._backward_new_grads(
                    loss_weight, grads=grads
                )

            return loss_weight.cpu().numpy(), new_grads
        else:
            # ----- Warmup: epoch < 1 -----
            if log_grads:
                # log gradients without applying GradNorm yet
                self._compute_grad_dim()
                grads = self._compute_grad(losses_list, mode="backward")
                self._reset_grad(grads.sum(0))
                return np.ones(self.task_num), grads

            # Just sum all task losses and backprop once
            total_loss = sum(losses_list)
            total_loss.backward()
            return np.ones(self.task_num), None
