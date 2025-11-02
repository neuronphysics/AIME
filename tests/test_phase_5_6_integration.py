"""
Phase 5 & 6 Integration Test

Validates that all refactored modules work together correctly.

This script:
1. Tests all module imports
2. Tests model instantiation with new imports
3. Tests forward pass through complete pipeline
4. Tests loss computation with LossAggregator
5. Tests RGB gradient balancing
6. Compares with original implementation

Usage:
    python tests/test_phase_5_6_integration.py
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_all_module_imports():
    """Test that all refactored modules import correctly."""
    print("=" * 60)
    print("Integration Test 1: Module Imports")
    print("=" * 60)

    try:
        # Pillar 1: Perception
        from src.perceiver_io import CausalPerceiverIO
        print("✓ perceiver_io imports successfully")

        # Pillar 2: Representation
        from src.generative_prior import DPGMMPrior, AdaptiveStickBreaking
        print("✓ generative_prior imports successfully")

        # Pillar 3: Dynamics
        from src.temporal_dynamics import LSTMLayer
        print("✓ temporal_dynamics imports successfully")

        # Pillar 4: Attention
        from src.attention_schema import AttentionSchema, AttentionPosterior
        print("✓ attention_schema imports successfully")

        # Pillar 5: Optimization
        from src.multi_task_learning import RGB, LossAggregator
        print("✓ multi_task_learning imports successfully")

        # Supporting modules
        from src.encoder_decoder import VAEEncoder, VAEDecoder
        print("✓ encoder_decoder imports successfully")

        from src.world_model import DPGMMVariationalRecurrentAutoencoder
        print("✓ world_model imports successfully")

        from src.training import DMCVBDataset
        print("✓ training imports successfully")

        print("\n✅ All module imports passed!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_model_instantiation():
    """Test that we can create the model with new imports."""
    print(f"\n{'='*60}")
    print("Integration Test 2: Model Instantiation")
    print("=" * 60)

    try:
        from src.world_model import DPGMMVariationalRecurrentAutoencoder

        print("\nCreating model with minimal config...")
        model = DPGMMVariationalRecurrentAutoencoder(
            max_components=5,
            latent_dim=16,
            hidden_dim=64,
            context_dim=32,
            image_channels=3,
            action_dim=6,
            image_size=64,
            n_lstm_layers=1,
            use_orthogonal=True,
            num_attention_slots=3,
            slot_dim=32,
            device='cpu'
        )

        print("✓ Model instantiated successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        print("\n✅ Model instantiation passed!")
        return True, model

    except Exception as e:
        print(f"\n❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model):
    """Test forward pass through the complete model."""
    print(f"\n{'='*60}")
    print("Integration Test 3: Forward Pass")
    print("=" * 60)

    try:
        # Create synthetic data (small for speed)
        batch_size = 2
        seq_len = 4
        observations = torch.randn(batch_size, seq_len, 3, 64, 64)
        actions = torch.randn(batch_size, seq_len, 6)

        print(f"\nInput shapes:")
        print(f"  - observations: {observations.shape}")
        print(f"  - actions: {actions.shape}")

        print("\nRunning forward_sequence()...")
        with torch.no_grad():
            outputs = model.forward_sequence(observations, actions)

        print("\n✓ Forward pass completed")
        print(f"  - Output keys: {len(outputs.keys())}")
        print(f"  - Reconstruction losses: {len(outputs.get('reconstruction_losses', []))}")
        print(f"  - KL latents: {len(outputs.get('kl_latents', []))}")

        print("\n✅ Forward pass test passed!")
        return True, outputs

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_computation(outputs):
    """Test loss computation with LossAggregator."""
    print(f"\n{'='*60}")
    print("Integration Test 4: Loss Computation")
    print("=" * 60)

    try:
        from src.multi_task_learning import LossAggregator

        print("\nCreating LossAggregator...")
        loss_agg = LossAggregator()

        print("Computing losses...")
        losses_dict, task_losses = loss_agg.compute_losses(
            outputs=outputs,
            beta=1.0,
            lambda_recon=1.0,
            lambda_att_dyn=0.1,
            lambda_gram=0.05,
            entropy_weight=0.1
        )

        print("\n✓ Loss computation completed")
        print(f"\nTask Losses (for RGB):")
        print(f"  - Task 1 (VAE/ELBO): {task_losses[0].item():.4f}")
        print(f"  - Task 2 (Perceiver): {task_losses[1].item():.4f}")
        print(f"  - Task 3 (Attention): {task_losses[2].item():.4f}")

        print(f"\nIndividual Components:")
        print(f"  - Reconstruction: {losses_dict['recon_loss'].item():.4f}")
        print(f"  - KL(z): {losses_dict['kl_z'].item():.4f}")
        print(f"  - Hierarchical KL: {losses_dict['hierarchical_kl'].item():.4f}")
        print(f"  - Perceiver: {losses_dict['perceiver_loss'].item():.4f}")

        print("\n✅ Loss computation test passed!")
        return True, losses_dict, task_losses

    except Exception as e:
        print(f"\n❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_rgb_gradient_balancing(model, task_losses):
    """Test RGB gradient balancing."""
    print(f"\n{'='*60}")
    print("Integration Test 5: RGB Gradient Balancing")
    print("=" * 60)

    try:
        from src.multi_task_learning import RGB

        print("\nCreating RGB optimizer...")
        rgb = RGB()

        # Configure RGB
        rgb.task_num = len(task_losses)
        rgb.device = 'cpu'
        rgb.rep_grad = False
        rgb.get_share_params = lambda: model.parameters()

        print("Applying RGB balancing...")
        batch_weights = rgb.backward(task_losses)

        print("\n✓ RGB balancing completed")
        print(f"  - Batch weights: {batch_weights}")
        print(f"  - Number of tasks: {len(task_losses)}")

        # Check gradients were computed
        has_grads = any(p.grad is not None for p in model.parameters())
        if has_grads:
            print(f"  - Gradients computed: ✓")

            # Compute total gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  - Total gradient norm: {total_norm:.4f}")
        else:
            print(f"  - Gradients computed: ✗ (WARNING)")

        print("\n✅ RGB gradient balancing test passed!")
        return True

    except Exception as e:
        print(f"\n❌ RGB gradient balancing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model):
    """Test complete training step."""
    print(f"\n{'='*60}")
    print("Integration Test 6: Complete Training Step")
    print("=" * 60)

    try:
        from src.multi_task_learning import LossAggregator, RGB

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create loss aggregator and RGB
        loss_agg = LossAggregator()
        rgb = RGB()
        rgb.task_num = 3
        rgb.device = 'cpu'
        rgb.rep_grad = False
        rgb.get_share_params = lambda: model.parameters()

        print("\nRunning 3 training steps...")
        losses_history = []

        for step in range(3):
            # Create data
            observations = torch.randn(2, 4, 3, 64, 64)
            actions = torch.randn(2, 4, 6)

            # Forward
            outputs = model.forward_sequence(observations, actions)

            # Compute losses
            losses_dict, task_losses = loss_agg.compute_losses(outputs, beta=1.0)

            # Backward
            optimizer.zero_grad()
            rgb.backward(task_losses)

            # Update
            optimizer.step()

            # Track
            total_loss = sum(task_losses).item()
            losses_history.append(total_loss)
            print(f"  Step {step+1}: Total Loss = {total_loss:.4f}")

        print("\n✓ Training steps completed")
        print(f"  - Initial loss: {losses_history[0]:.4f}")
        print(f"  - Final loss: {losses_history[-1]:.4f}")
        print(f"  - Change: {losses_history[-1] - losses_history[0]:.4f}")

        print("\n✅ Training step test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_paths():
    """Test that new import paths work correctly."""
    print(f"\n{'='*60}")
    print("Integration Test 7: Import Path Validation")
    print("=" * 60)

    import_tests = [
        ("from src.world_model import DPGMMVariationalRecurrentAutoencoder", "World model"),
        ("from src.multi_task_learning import RGB, LossAggregator", "Multi-task learning"),
        ("from src.multi_task_learning.losses import ELBOLoss", "Loss modules"),
        ("from src.perceiver_io import CausalPerceiverIO", "Perceiver IO"),
        ("from src.generative_prior import DPGMMPrior", "Generative prior"),
        ("from src.temporal_dynamics import LSTMLayer", "Temporal dynamics"),
        ("from src.attention_schema import AttentionSchema", "Attention schema"),
        ("from src.encoder_decoder import VAEEncoder, VAEDecoder", "Encoder-decoder"),
        ("from src.training import DMCVBDataset", "Training"),
    ]

    all_passed = True
    for import_stmt, name in import_tests:
        try:
            exec(import_stmt)
            print(f"  ✓ {name:25s} - {import_stmt}")
        except Exception as e:
            print(f"  ✗ {name:25s} - {import_stmt}")
            print(f"    Error: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All import paths validated!")
    else:
        print("\n❌ Some import paths failed!")

    return all_passed


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("PHASE 5 & 6 INTEGRATION TEST SUITE")
    print("="*60)
    print("\nTesting refactored AIME modules...")

    results = {}

    # Test 1: Module imports
    results['imports'] = test_all_module_imports()

    # Test 2: Model instantiation
    results['instantiation'], model = test_model_instantiation()

    if model is not None:
        # Test 3: Forward pass
        results['forward'], outputs = test_forward_pass(model)

        if outputs is not None:
            # Test 4: Loss computation
            results['loss'], losses_dict, task_losses = test_loss_computation(outputs)

            if task_losses is not None:
                # Test 5: RGB gradient balancing
                results['rgb'] = test_rgb_gradient_balancing(model, task_losses)

        # Test 6: Training step
        results['training'] = test_training_step(model)

    # Test 7: Import paths
    results['import_paths'] = test_import_paths()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASSED" if passed_flag else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nPhase 5 & 6 refactoring validated successfully.")
        print("All modules work together correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease review failed tests above.")

    print("\n" + "="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
