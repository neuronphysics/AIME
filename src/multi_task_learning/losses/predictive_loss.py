"""
Predictive Loss Computation

Implements losses for attention dynamics and diversity:
- Attention dynamics loss: Measures temporal consistency of attention
- Attention diversity loss: Encourages diverse slot usage

Tensor Flow:
    Input: outputs from model.forward_sequence()
    Output: Scalar loss term
"""

import torch
import torch.nn as nn


class PredictiveLoss(nn.Module):
    """
    Computes Predictive task loss components.

    The Predictive task optimizes attention schema dynamics:
    L_Predictive = attention_dynamics + diversity_penalty
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                outputs: dict,
                beta: float = 1.0,
                lambda_att_dyn: float = 0.1) -> dict:
        """
        Compute Predictive loss components.

        Args:
            outputs: Dictionary from model.forward_sequence() containing:
                - attention_losses: List[Tensor] attention KL terms
                - attention_dynamics_loss: Tensor, temporal dynamics loss
                - attention_diversity_losses: List[Tensor] diversity penalties
            beta: KL annealing factor (default: 1.0)
            lambda_att_dyn: Attention dynamics weight (default: 0.1)

        Returns:
            losses: Dictionary with keys:
                - attention_kl: Attention KL divergence
                - attention_dynamics_loss: Temporal consistency loss
                - attention_diversity: Slot diversity loss
                - total_predictive: Weighted sum (main Predictive task loss)

        Tensor Shapes:
            All outputs are scalars: torch.Size([])
        """
        device = outputs.get('device', 'cpu')
        losses = {}

        # === Attention KL ===
        # KL divergence between attention posterior and prior
        losses['attention_kl'] = (
            torch.stack(outputs['attention_losses']).mean()
            if outputs.get('attention_losses')
            else torch.tensor(0.0, device=device)
        )

        # === Attention Dynamics Loss ===
        # Measures temporal consistency of attention predictions
        losses['attention_dynamics_loss'] = outputs.get(
            'attention_dynamics_loss',
            torch.tensor(0.0, device=device)
        )

        # === Attention Diversity Loss ===
        # Encourages slots to attend to different regions
        # Computed as negative entropy or similarity penalty
        losses['attention_diversity'] = (
            torch.stack(outputs['attention_diversity_losses']).mean()
            if outputs.get('attention_diversity_losses')
            else torch.tensor(0.0, device=device)
        )

        # === Total Predictive Loss ===
        # Weighted combination of attention-related terms
        losses['total_predictive'] = (
            beta * losses['attention_kl'] +
            lambda_att_dyn * losses['attention_dynamics_loss'] +
            losses['attention_diversity']
        )

        return losses

    def get_attention_metrics(self, outputs: dict) -> dict:
        """
        Extract attention-related metrics for logging.

        Args:
            outputs: Dictionary from model.forward_sequence()

        Returns:
            metrics: Dictionary with monitoring metrics:
                - mean_attention_entropy: Average slot entropy
                - slot_usage: Number of active slots
                - attention_concentration: Max attention per slot

        Note:
            These metrics help monitor attention schema health
            and detect potential collapse (all slots attending to same region).
        """
        metrics = {}

        # Extract attention statistics if available
        if 'attention_entropy' in outputs:
            metrics['mean_attention_entropy'] = torch.stack(
                outputs['attention_entropy']
            ).mean().item() if outputs['attention_entropy'] else 0.0

        if 'slot_usage' in outputs:
            metrics['slot_usage'] = outputs['slot_usage']

        if 'attention_concentration' in outputs:
            metrics['attention_concentration'] = outputs['attention_concentration']

        return metrics
