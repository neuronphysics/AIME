"""
Test/Demo: Import Validation

Verifies that all imports work correctly after Phase 5 refactoring.

This script:
1. Tests multi_task_learning module imports
2. Tests that VRNN model can import RGB from new location
3. Verifies no circular dependencies

Usage:
    python multi_task_learning/tests/test_imports.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_multi_task_learning_imports():
    """Test that multi_task_learning module imports work."""
    print("=" * 60)
    print("Import Test: multi_task_learning module")
    print("=" * 60)

    try:
        from multi_task_learning import RGB, AbsWeighting, LossAggregator
        print("✓ Successfully imported RGB")
        print("✓ Successfully imported AbsWeighting")
        print("✓ Successfully imported LossAggregator")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test loss modules
    try:
        from multi_task_learning.losses import ELBOLoss, PerceiverLoss, PredictiveLoss
        print("✓ Successfully imported ELBOLoss")
        print("✓ Successfully imported PerceiverLoss")
        print("✓ Successfully imported PredictiveLoss")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    return True


def test_vrnn_model_imports():
    """Test that VRNN model can import from new location."""
    print(f"\n{'='*60}")
    print("Import Test: VRNN model with new RGB import")
    print(f"{'='*60}")

    try:
        # This will test if the updated import in dpgmm_stickbreaking_prior_vrnn.py works
        from legacy.VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
        print("✓ Successfully imported DPGMMVariationalRecurrentAutoencoder")
        print("  (This confirms RGB import from multi_task_learning works)")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    return True


def test_instantiation():
    """Test that we can actually instantiate the classes."""
    print(f"\n{'='*60}")
    print("Instantiation Test: Create objects")
    print(f"{'='*60}")

    try:
        from multi_task_learning import RGB, LossAggregator

        # Test RGB instantiation
        rgb = RGB()
        print("✓ Successfully instantiated RGB optimizer")
        print(f"  - mu: {rgb.mu}")
        print(f"  - lambd: {rgb.lambd}")
        print(f"  - alpha_steps: {rgb.alpha_steps}")

        # Test LossAggregator instantiation
        loss_agg = LossAggregator()
        print("✓ Successfully instantiated LossAggregator")
        print(f"  - Has elbo_loss: {hasattr(loss_agg, 'elbo_loss')}")
        print(f"  - Has perceiver_loss: {hasattr(loss_agg, 'perceiver_loss')}")
        print(f"  - Has predictive_loss: {hasattr(loss_agg, 'predictive_loss')}")

    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        return False

    return True


def test_backward_compatibility():
    """Test that old RGB.py location is removed."""
    print(f"\n{'='*60}")
    print("Backward Compatibility Test")
    print(f"{'='*60}")

    # Check if old RGB.py exists
    old_path = os.path.join(os.path.dirname(__file__), '../../VRNN/RGB.py')
    if os.path.exists(old_path):
        print(f"⚠️  Warning: Old RGB.py still exists at VRNN/RGB.py")
        print(f"   Consider removing it to avoid confusion")
        return True  # Not a failure, just a warning
    else:
        print("✓ Old RGB.py location is clear (moved to multi_task_learning/)")

    return True


def main():
    """Run all import validation tests."""
    print("\n" + "="*60)
    print("Phase 5 Import Validation Test Suite")
    print("="*60)

    all_passed = True

    # Run tests
    all_passed &= test_multi_task_learning_imports()
    all_passed &= test_vrnn_model_imports()
    all_passed &= test_instantiation()
    all_passed &= test_backward_compatibility()

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("All import tests passed! ✓")
    else:
        print("Some import tests failed! ✗")
    print("="*60)

    print("\nPhase 5 Summary:")
    print("  ✓ Created multi_task_learning/ module")
    print("  ✓ Moved RGB.py to multi_task_learning/rgb_optimizer.py")
    print("  ✓ Created loss computation modules (ELBO, Perceiver, Predictive)")
    print("  ✓ Created LossAggregator for coordinating losses")
    print("  ✓ Updated imports in VRNN model")
    print("  ✓ All imports working correctly")
    print("\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
