"""
Loss Aggregator

Coordinates all loss computations and prepares them for multi-task optimization.

This module:
1. Computes individual loss components using task-specific modules
2. Aggregates losses by task (ELBO, Perceiver, Predictive, Adversarial)
3. Provides clean interface for RGB optimizer

Tensor Flow:
    Input: outputs from model.forward_sequence()
    Output: losses_dict (all components) + task_losses (for RGB)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .losses import ELBOLoss, PerceiverLoss, PredictiveLoss


class LossAggregator(nn.Module):
    """
    Aggregates all loss computations for multi-task learning.

    Usage:
        >>> loss_agg = LossAggregator()
        >>> losses_dict, task_losses = loss_agg.compute_losses(
        ...     outputs=model_outputs,
        ...     beta=1.0,
        ...     lambda_recon=1.0,
        ...     # ... other hyperparameters
        ... )
        >>> # losses_dict contains all individual components
        >>> # task_losses is List[Tensor] ready for RGB optimizer
    """

    def __init__(self):
        super().__init__()

        # Initialize task-specific loss modules
        self.elbo_loss = ELBOLoss()
        self.perceiver_loss = PerceiverLoss()
        self.predictive_loss = PredictiveLoss()

    def compute_losses(self,
                      outputs: Dict,
                      beta: float = 1.0,
                      entropy_weight: float = 0.1,
                      lambda_recon: float = 1.0,
                      lambda_att_dyn: float = 0.1,
                      lambda_gram: float = 0.05) -> Tuple[Dict, List[torch.Tensor]]:
        """
        Compute all losses and prepare for multi-task optimization.

        Args:
            outputs: Dictionary from model.forward_sequence() containing:
                - reconstruction_losses: List[Tensor]
                - kl_latents: List[Tensor]
                - kumaraswamy_kl_losses: List[Tensor]
                - attention_losses: List[Tensor]
                - cluster_entropies: List[Tensor]
                - gram_enc: List[Tensor]
                - perceiver_total_loss: Tensor
                - attention_dynamics_loss: Tensor
                - attention_diversity_losses: List[Tensor]
            beta: KL annealing factor (default: 1.0)
            entropy_weight: Cluster entropy weight (default: 0.1)
            lambda_recon: Reconstruction weight (default: 1.0)
            lambda_att_dyn: Attention dynamics weight (default: 0.1)
            lambda_gram: Gram loss weight (default: 0.05)

        Returns:
            losses_dict: Dictionary containing all individual loss components
            task_losses: List[Tensor] of per-task losses for RGB optimizer
                [total_elbo, perceiver_loss, total_predictive]

        Tensor Shapes:
            All losses are scalars: torch.Size([])
        """
        # === Task 1: ELBO Loss ===
        elbo_losses = self.elbo_loss(
            outputs=outputs,
            beta=beta,
            lambda_recon=lambda_recon,
            lambda_gram=lambda_gram,
            entropy_weight=entropy_weight
        )

        # === Task 2: Perceiver Loss ===
        perceiver_losses = self.perceiver_loss(outputs=outputs)

        # === Task 3: Predictive Loss ===
        predictive_losses = self.predictive_loss(
            outputs=outputs,
            beta=beta,
            lambda_att_dyn=lambda_att_dyn
        )

        # === Combine all losses into single dictionary ===
        losses_dict = {
            # ELBO task components
            'recon_loss': elbo_losses['recon_loss'],
            'kl_z': elbo_losses['kl_z'],
            'hierarchical_kl': elbo_losses['hierarchical_kl'],
            'attention_kl': elbo_losses['attention_kl'],
            'cluster_entropy': elbo_losses['cluster_entropy'],
            'gram_enc_loss': elbo_losses['gram_enc_loss'],
            'total_elbo': elbo_losses['total_elbo'],

            # Perceiver task components
            'perceiver_loss': perceiver_losses['perceiver_loss'],

            # Predictive task components
            'attention_dynamics_loss': predictive_losses['attention_dynamics_loss'],
            'attention_diversity': predictive_losses['attention_diversity'],
            'total_predictive': predictive_losses['total_predictive'],
        }

        # === Prepare task losses for RGB optimizer ===
        # Note: The original compute_total_loss splits losses differently:
        # - total_vae_loss includes: recon + kl_z + hierarchical_kl - entropy + gram
        # - attention_loss includes: attention_kl + dynamics + diversity
        #
        # For consistency with the original implementation:
        losses_dict['total_vae_loss'] = (
            lambda_recon * losses_dict['recon_loss'] +
            beta * losses_dict['kl_z'] +
            beta * losses_dict['hierarchical_kl'] -
            entropy_weight * losses_dict['cluster_entropy'] +
            lambda_gram * losses_dict['gram_enc_loss']
        )

        losses_dict['attention_loss'] = (
            beta * losses_dict['attention_kl'] +
            lambda_att_dyn * losses_dict['attention_dynamics_loss'] +
            losses_dict['attention_diversity']
        )

        # Task losses for RGB optimizer
        task_losses = [
            losses_dict['total_vae_loss'],    # Task 1: VAE/ELBO
            losses_dict['perceiver_loss'],    # Task 2: Perceiver
            losses_dict['attention_loss'],    # Task 3: Attention/Predictive
            # Future: Add adversarial loss as Task 4
        ]

        return losses_dict, task_losses

    def get_all_metrics(self, outputs: Dict) -> Dict:
        """
        Extract all monitoring metrics from outputs.

        Args:
            outputs: Dictionary from model.forward_sequence()

        Returns:
            metrics: Dictionary with all monitoring metrics

        Note:
            These metrics should be logged separately from losses
            for monitoring training dynamics and model health.
        """
        metrics = {}

        # Perceiver metrics
        perceiver_metrics = self.perceiver_loss.get_perceiver_metrics(outputs)
        metrics.update(perceiver_metrics)

        # Attention metrics
        attention_metrics = self.predictive_loss.get_attention_metrics(outputs)
        metrics.update(attention_metrics)

        # ELBO metrics (detailed KL breakdown)
        kl_breakdown = self.elbo_loss.compute_detailed_kl_breakdown(outputs)
        metrics.update(kl_breakdown)

        return metrics

    def compute_legacy_total_loss(self,
                                 observations: torch.Tensor,
                                 actions: torch.Tensor = None,
                                 beta: float = 1.0,
                                 entropy_weight: float = 0.1,
                                 lambda_recon: float = 1.0,
                                 lambda_att_dyn: float = 0.1,
                                 lambda_gram: float = 0.05) -> Tuple[Dict, Dict]:
        """
        Legacy interface matching original compute_total_loss signature.

        This method is provided for backward compatibility.
        For new code, use compute_losses() instead.

        Args:
            observations: [B, T, C, H, W] video tensor
            actions: [B, T, action_dim] action tensor (optional)
            beta: KL annealing factor
            entropy_weight: Cluster entropy weight
            lambda_recon: Reconstruction weight
            lambda_att_dyn: Attention dynamics weight
            lambda_gram: Gram loss weight

        Returns:
            losses: Dictionary matching original compute_total_loss output
            outputs: Dictionary from forward_sequence

        Note:
            This requires access to the model, so it's typically called
            from within the model class, not standalone.
        """
        raise NotImplementedError(
            "Legacy interface requires model reference. "
            "Call model.compute_total_loss() instead."
        )
