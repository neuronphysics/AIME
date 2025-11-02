"""
Perceiver Loss Computation

Implements losses for the Perceiver IO video tokenization:
- VQ commitment loss: Encourages encoder outputs to commit to codebook entries
- Perceiver reconstruction loss: Measures token prediction quality
- Codebook perplexity: Monitors codebook usage diversity

Tensor Flow:
    Input: outputs from model.forward_sequence()
    Output: Scalar loss term
"""

import torch
import torch.nn as nn


class PerceiverLoss(nn.Module):
    """
    Computes Perceiver task loss components.

    The Perceiver task optimizes video tokenization:
    L_Perceiver = VQ_commitment + VQ_reconstruction + perplexity_penalty
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs: dict) -> dict:
        """
        Compute Perceiver loss components.

        Args:
            outputs: Dictionary from model.forward_sequence() containing:
                - perceiver_total_loss: Tensor, aggregated Perceiver loss

        Returns:
            losses: Dictionary with keys:
                - perceiver_loss: Total Perceiver task loss

        Tensor Shapes:
            All outputs are scalars: torch.Size([])

        Note:
            The perceiver_total_loss is already computed inside the
            CausalPerceiverIO module during forward_sequence().
            This wrapper exists for consistency with other loss modules
            and to enable future decomposition if needed.
        """
        device = outputs.get('device', 'cpu')
        losses = {}

        # Perceiver loss (already aggregated in model)
        losses['perceiver_loss'] = outputs.get(
            'perceiver_total_loss',
            torch.tensor(0.0, device=device)
        )

        return losses

    def get_perceiver_metrics(self, outputs: dict) -> dict:
        """
        Extract Perceiver-related metrics for logging.

        Args:
            outputs: Dictionary from model.forward_sequence()

        Returns:
            metrics: Dictionary with monitoring metrics:
                - vq_perplexity: Codebook usage diversity
                - vq_commitment_loss: Commitment term
                - vq_reconstruction_loss: Reconstruction term

        Note:
            These metrics should be logged separately from the loss
            for monitoring codebook health and training dynamics.
        """
        metrics = {}

        # Extract metrics if available in outputs
        if 'vq_perplexity' in outputs:
            metrics['vq_perplexity'] = outputs['vq_perplexity']
        if 'vq_commitment_loss' in outputs:
            metrics['vq_commitment_loss'] = outputs['vq_commitment_loss']
        if 'vq_reconstruction_loss' in outputs:
            metrics['vq_reconstruction_loss'] = outputs['vq_reconstruction_loss']

        return metrics
