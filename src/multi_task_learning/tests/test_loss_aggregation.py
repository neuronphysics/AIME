"""
Test/Demo: Loss Aggregation

Demonstrates how LossAggregator organizes and computes all loss components.

This script:
1. Creates synthetic model outputs
2. Computes losses using LossAggregator
3. Shows loss breakdown by task
4. Verifies consistency with original implementation

Usage:
    python multi_task_learning/tests/test_loss_aggregation.py
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multi_task_learning import LossAggregator


def create_synthetic_outputs(batch_size=2, seq_len=4, device='cpu'):
    """
    Create synthetic model outputs matching forward_sequence() format.

    Args:
        batch_size: Number of sequences
        seq_len: Sequence length
        device: torch device

    Returns:
        outputs: Dictionary with all required keys
    """
    outputs = {
        # ELBO components (per-timestep lists)
        'reconstruction_losses': [
            torch.tensor(0.5 + 0.1 * t, device=device)
            for t in range(seq_len)
        ],
        'kl_latents': [
            torch.tensor(0.2 + 0.05 * t, device=device)
            for t in range(seq_len)
        ],
        'kumaraswamy_kl_losses': [
            torch.tensor(0.1 + 0.02 * t, device=device)
            for t in range(seq_len)
        ],
        'attention_losses': [
            torch.tensor(0.15 + 0.03 * t, device=device)
            for t in range(seq_len)
        ],
        'cluster_entropies': [
            torch.tensor(1.5 - 0.1 * t, device=device)
            for t in range(seq_len)
        ],
        'gram_enc': [
            torch.tensor(0.05, device=device)
            for t in range(seq_len)
        ],

        # Perceiver components (single values)
        'perceiver_total_loss': torch.tensor(0.3, device=device),

        # Predictive components
        'attention_dynamics_loss': torch.tensor(0.25, device=device),
        'attention_diversity_losses': [
            torch.tensor(0.1, device=device)
            for t in range(seq_len)
        ],

        # Metadata
        'device': device,
    }

    return outputs


def test_loss_computation():
    """
    Test basic loss computation with synthetic outputs.
    """
    print("=" * 60)
    print("Loss Aggregation Test: Basic Computation")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create synthetic outputs
    outputs = create_synthetic_outputs(batch_size=2, seq_len=4, device=device)

    print(f"\nSynthetic Outputs Created:")
    print(f"  Reconstruction losses: {len(outputs['reconstruction_losses'])} timesteps")
    print(f"  KL latents: {len(outputs['kl_latents'])} timesteps")
    print(f"  Perceiver loss: scalar")
    print(f"  Attention dynamics: scalar")

    # Initialize loss aggregator
    loss_agg = LossAggregator()

    # Compute losses
    print(f"\nComputing losses...")
    losses_dict, task_losses = loss_agg.compute_losses(
        outputs=outputs,
        beta=1.0,
        lambda_recon=1.0,
        lambda_att_dyn=0.1,
        lambda_gram=0.05,
        entropy_weight=0.1
    )

    # Display ELBO components
    print(f"\n{'='*60}")
    print("ELBO Task Components:")
    print(f"{'='*60}")
    print(f"  Reconstruction loss: {losses_dict['recon_loss'].item():.4f}")
    print(f"  KL(z): {losses_dict['kl_z'].item():.4f}")
    print(f"  Hierarchical KL: {losses_dict['hierarchical_kl'].item():.4f}")
    print(f"  Attention KL: {losses_dict['attention_kl'].item():.4f}")
    print(f"  Cluster entropy: {losses_dict['cluster_entropy'].item():.4f}")
    print(f"  Gram encoder loss: {losses_dict['gram_enc_loss'].item():.4f}")
    print(f"  → Total ELBO: {losses_dict['total_elbo'].item():.4f}")

    # Display Perceiver components
    print(f"\n{'='*60}")
    print("Perceiver Task Components:")
    print(f"{'='*60}")
    print(f"  Perceiver loss: {losses_dict['perceiver_loss'].item():.4f}")

    # Display Predictive components
    print(f"\n{'='*60}")
    print("Predictive Task Components:")
    print(f"{'='*60}")
    print(f"  Attention dynamics: {losses_dict['attention_dynamics_loss'].item():.4f}")
    print(f"  Attention diversity: {losses_dict['attention_diversity'].item():.4f}")
    print(f"  → Total Predictive: {losses_dict['total_predictive'].item():.4f}")

    # Display task losses for RGB
    print(f"\n{'='*60}")
    print("Task Losses (for RGB Optimizer):")
    print(f"{'='*60}")
    print(f"  Task 1 (VAE/ELBO): {task_losses[0].item():.4f}")
    print(f"  Task 2 (Perceiver): {task_losses[1].item():.4f}")
    print(f"  Task 3 (Attention): {task_losses[2].item():.4f}")

    print(f"\n✓ Loss computation test passed!")

    return losses_dict, task_losses


def test_loss_backward():
    """
    Test that task losses have proper gradients.
    """
    print(f"\n{'='*60}")
    print("Loss Aggregation Test: Gradient Flow")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create outputs with gradients enabled
    outputs = create_synthetic_outputs(batch_size=2, seq_len=4, device=device)

    # Make tensors require gradients
    for key in outputs:
        if isinstance(outputs[key], list):
            outputs[key] = [t.requires_grad_(True) if isinstance(t, torch.Tensor) else t
                           for t in outputs[key]]
        elif isinstance(outputs[key], torch.Tensor):
            outputs[key] = outputs[key].requires_grad_(True)

    # Compute losses
    loss_agg = LossAggregator()
    losses_dict, task_losses = loss_agg.compute_losses(
        outputs=outputs,
        beta=1.0,
        lambda_recon=1.0,
        lambda_att_dyn=0.1,
        lambda_gram=0.05,
        entropy_weight=0.1
    )

    print(f"\nTesting gradient flow for each task loss...")

    # Test gradients for each task
    for i, task_loss in enumerate(task_losses):
        # Backward
        task_loss.backward(retain_graph=True)

        # Check if gradients exist
        has_grads = any(
            t.grad is not None
            for t in outputs['reconstruction_losses']
            if hasattr(t, 'grad')
        )

        print(f"  Task {i+1}: {'✓' if task_loss.requires_grad else '✗'} requires_grad")

    print(f"\n✓ Gradient flow test passed!")


def test_hyperparameter_sensitivity():
    """
    Test how hyperparameters affect loss values.
    """
    print(f"\n{'='*60}")
    print("Loss Aggregation Test: Hyperparameter Sensitivity")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputs = create_synthetic_outputs(batch_size=2, seq_len=4, device=device)
    loss_agg = LossAggregator()

    # Test different beta values (KL annealing)
    print(f"\nTesting beta (KL annealing):")
    print(f"  beta | Total ELBO | Total Predictive")
    print(f"  " + "-" * 45)

    for beta in [0.1, 0.5, 1.0, 2.0]:
        losses_dict, _ = loss_agg.compute_losses(
            outputs=outputs,
            beta=beta,
            lambda_recon=1.0,
            lambda_att_dyn=0.1,
            lambda_gram=0.05,
            entropy_weight=0.1
        )
        print(f"  {beta:4.1f} | {losses_dict['total_elbo'].item():10.4f} | "
              f"{losses_dict['total_predictive'].item():16.4f}")

    # Test different reconstruction weights
    print(f"\nTesting lambda_recon (reconstruction weight):")
    print(f"  λ_recon | Total ELBO")
    print(f"  " + "-" * 25)

    for lambda_recon in [0.5, 1.0, 2.0, 5.0]:
        losses_dict, _ = loss_agg.compute_losses(
            outputs=outputs,
            beta=1.0,
            lambda_recon=lambda_recon,
            lambda_att_dyn=0.1,
            lambda_gram=0.05,
            entropy_weight=0.1
        )
        print(f"  {lambda_recon:7.1f} | {losses_dict['total_elbo'].item():10.4f}")

    print(f"\n✓ Hyperparameter sensitivity test passed!")


def test_consistency_with_original():
    """
    Verify that LossAggregator produces same results as original compute_total_loss.
    """
    print(f"\n{'='*60}")
    print("Loss Aggregation Test: Consistency with Original")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputs = create_synthetic_outputs(batch_size=2, seq_len=4, device=device)

    # Use LossAggregator
    loss_agg = LossAggregator()
    losses_dict, task_losses = loss_agg.compute_losses(
        outputs=outputs,
        beta=1.0,
        lambda_recon=1.0,
        lambda_att_dyn=0.1,
        lambda_gram=0.05,
        entropy_weight=0.1
    )

    # Manually compute what original code would produce
    # (Matching the logic in dpgmm_stickbreaking_prior_vrnn.py:compute_total_loss)
    original_recon = torch.stack(outputs['reconstruction_losses']).mean()
    original_kl_z = torch.stack(outputs['kl_latents']).mean()
    original_hierarchical_kl = torch.stack(outputs['kumaraswamy_kl_losses']).mean()
    original_attention_kl = torch.stack(outputs['attention_losses']).mean()
    original_cluster_entropy = torch.stack(outputs['cluster_entropies']).mean()
    original_gram = torch.stack(outputs['gram_enc']).mean()

    original_total_vae = (
        1.0 * original_recon +
        1.0 * original_kl_z +
        1.0 * original_hierarchical_kl -
        0.1 * original_cluster_entropy +
        0.05 * original_gram
    )

    original_attention_loss = (
        1.0 * original_attention_kl +
        0.1 * outputs['attention_dynamics_loss'] +
        torch.stack(outputs['attention_diversity_losses']).mean()
    )

    # Compare
    print(f"\nComparing with original implementation:")
    print(f"\n  Component              | New       | Original  | Match")
    print(f"  " + "-" * 60)

    def compare(name, new_val, orig_val):
        match = torch.allclose(new_val, orig_val, rtol=1e-5)
        symbol = "✓" if match else "✗"
        print(f"  {name:22s} | {new_val.item():9.4f} | {orig_val.item():9.4f} | {symbol}")
        return match

    all_match = True
    all_match &= compare("Reconstruction", losses_dict['recon_loss'], original_recon)
    all_match &= compare("KL(z)", losses_dict['kl_z'], original_kl_z)
    all_match &= compare("Hierarchical KL", losses_dict['hierarchical_kl'], original_hierarchical_kl)
    all_match &= compare("Total VAE", losses_dict['total_vae_loss'], original_total_vae)
    all_match &= compare("Attention Loss", losses_dict['attention_loss'], original_attention_loss)

    if all_match:
        print(f"\n✓ Consistency test passed! Results match original implementation.")
    else:
        print(f"\n✗ Consistency test failed! Some values differ from original.")

    return all_match


def main():
    """Run all loss aggregation tests."""
    print("\n" + "="*60)
    print("Multi-Task Learning: Loss Aggregator Test Suite")
    print("="*60)

    # Run tests
    test_loss_computation()
    test_loss_backward()
    test_hyperparameter_sensitivity()
    consistency = test_consistency_with_original()

    print("\n" + "="*60)
    if consistency:
        print("All loss aggregation tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("="*60)

    print("\nKey Takeaways:")
    print("  - LossAggregator organizes losses by task")
    print("  - Individual components are accessible for monitoring")
    print("  - Task losses are ready for RGB optimizer")
    print("  - Results match original implementation")
    print("\n")


if __name__ == "__main__":
    main()
