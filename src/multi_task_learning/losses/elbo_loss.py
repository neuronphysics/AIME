"""
ELBO Loss Computation

Implements the Evidence Lower Bound (ELBO) loss components:
- Reconstruction loss: -E_q[log p(x|z)]
- KL divergences: KL[q || p] for latent z, hierarchical prior, attention
- Cluster entropy: Encourages diverse cluster usage

Tensor Flow:
    Input: outputs from model.forward_sequence()
    Output: Scalar loss terms
"""

import torch
import torch.nn as nn


class ELBOLoss(nn.Module):
    """
    Computes ELBO task loss components.

    The ELBO task optimizes the generative model:
    L_ELBO = E_q[log p(x|z)] - KL[q(z|x) || p(z|h,c)] - KL[q(π) || p(π)] - ...

    We maximize ELBO = minimize -ELBO.
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                outputs: dict,
                beta: float = 1.0,
                lambda_recon: float = 1.0,
                lambda_gram: float = 0.05,
                entropy_weight: float = 0.1) -> dict:
        """
        Compute ELBO loss components.

        Args:
            outputs: Dictionary from model.forward_sequence() containing:
                - reconstruction_losses: List[Tensor] per-timestep losses
                - kl_latents: List[Tensor] KL[q(z|x) || p(z|h,c)]
                - kumaraswamy_kl_losses: List[Tensor] hierarchical prior KL
                - attention_losses: List[Tensor] attention KL
                - cluster_entropies: List[Tensor] cluster entropy H(π)
                - gram_enc: List[Tensor] encoder Gram matrix loss
            beta: KL annealing factor (default: 1.0)
            lambda_recon: Reconstruction weight (default: 1.0)
            lambda_gram: Gram loss weight (default: 0.05)
            entropy_weight: Cluster entropy weight (default: 0.1)

        Returns:
            losses: Dictionary with keys:
                - recon_loss: Reconstruction term
                - kl_z: Latent KL term
                - hierarchical_kl: DPGMM prior KL term
                - attention_kl: Attention KL term
                - cluster_entropy: Cluster entropy term
                - gram_enc_loss: Encoder Gram loss term
                - total_elbo: Weighted sum (main ELBO task loss)

        Tensor Shapes:
            All outputs are scalars: torch.Size([])
        """
        device = outputs.get('device', 'cpu')
        losses = {}

        # === Reconstruction Term ===
        # Negative log-likelihood: -E_q[log p(x|z)]
        # Higher is worse (more reconstruction error)
        losses['recon_loss'] = (
            torch.stack(outputs['reconstruction_losses']).mean()
            if outputs.get('reconstruction_losses')
            else torch.tensor(0.0, device=device)
        )

        # === KL Divergence Terms ===

        # 1. Latent KL: KL[q(z|x) || p(z|h,c)]
        # Measures how far posterior deviates from prior
        losses['kl_z'] = (
            torch.stack(outputs['kl_latents']).mean()
            if outputs.get('kl_latents')
            else torch.tensor(0.0, device=device)
        )

        # 2. Hierarchical KL: Includes both stick-breaking AND alpha prior
        # KL[q(β) || p(β)] + KL[q(α) || p(α)]
        losses['hierarchical_kl'] = (
            torch.stack(outputs['kumaraswamy_kl_losses']).mean()
            if outputs.get('kumaraswamy_kl_losses')
            else torch.tensor(0.0, device=device)
        )

        # 3. Attention KL: KL between attention posterior and prior
        losses['attention_kl'] = (
            torch.stack(outputs['attention_losses']).mean()
            if outputs.get('attention_losses')
            else torch.tensor(0.0, device=device)
        )

        # === Entropy Term ===
        # Encourages diverse cluster usage: H(π) = -Σ π_k log π_k
        # Negative in loss (we maximize entropy = minimize -H)
        losses['cluster_entropy'] = (
            torch.stack(outputs['cluster_entropies']).mean()
            if outputs.get('cluster_entropies')
            else torch.tensor(0.0, device=device)
        )

        # === Encoder Gram Loss ===
        # Student-teacher consistency for encoder features
        losses['gram_enc_loss'] = (
            torch.stack(outputs['gram_enc']).mean()
            if outputs.get('gram_enc')
            else torch.tensor(0.0, device=device)
        )

        # === Total ELBO Loss ===
        # Minimize: reconstruction error + KL terms - entropy
        losses['total_elbo'] = (
            lambda_recon * losses['recon_loss'] +
            beta * losses['kl_z'] +
            beta * losses['hierarchical_kl'] +
            beta * losses['attention_kl'] -
            entropy_weight * losses['cluster_entropy'] +
            lambda_gram * losses['gram_enc_loss']
        )

        return losses

    def compute_detailed_kl_breakdown(self, outputs: dict) -> dict:
        """
        Compute detailed KL breakdown for debugging.

        Useful for understanding which KL terms are dominant.

        Args:
            outputs: Dictionary from model.forward_sequence()

        Returns:
            breakdown: Dictionary with per-component KL values
        """
        device = outputs.get('device', 'cpu')

        breakdown = {
            'kl_z_mean': torch.stack(outputs['kl_latents']).mean().item()
                         if outputs.get('kl_latents') else 0.0,
            'kl_z_std': torch.stack(outputs['kl_latents']).std().item()
                        if outputs.get('kl_latents') else 0.0,
            'hierarchical_kl_mean': torch.stack(outputs['kumaraswamy_kl_losses']).mean().item()
                                   if outputs.get('kumaraswamy_kl_losses') else 0.0,
            'attention_kl_mean': torch.stack(outputs['attention_losses']).mean().item()
                                if outputs.get('attention_losses') else 0.0,
        }

        return breakdown
