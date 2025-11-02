# AIME Refactoring Migration Guide

**Date:** November 2, 2025
**Phases Completed:** 5 & 6
**Status:** Ready for Use

This guide helps you migrate to the new modular structure of AIME.

---

## What Changed?

### Before (Old Structure)

```python
# Old imports
from VRNN.RGB import RGB
from VRNN.lstm import LSTMLayer
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
```

### After (New Structure)

```python
# New imports
from multi_task_learning import RGB
from temporal_dynamics import LSTMLayer
from world_model import DPGMMVariationalRecurrentAutoencoder
```

---

## Module Migration Map

| Old Location | New Location | Module |
|-------------|--------------|--------|
| `VRNN.RGB` | `multi_task_learning` | RGB optimizer |
| `VRNN.lstm` | `temporal_dynamics` | LSTM layer |
| `VRNN.dpgmm_stickbreaking_prior_vrnn` | `world_model` | Main model |
| `VRNN.perceiver.*` | `perceiver_io` | Perceiver components |
| (various) | `generative_prior` | DPGMM components |
| (various) | `attention_schema` | Attention components |
| `nvae_architecture` | `encoder_decoder` | VAE components |
| `VRNN.dmc_vb_transition_dynamics_trainer` | `training` | Training utils |

---

## Step-by-Step Migration

### 1. Update Imports

**Old code:**
```python
from VRNN.RGB import RGB
from VRNN.lstm import LSTMLayer
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
```

**New code:**
```python
from multi_task_learning import RGB
from temporal_dynamics import LSTMLayer
from world_model import DPGMMVariationalRecurrentAutoencoder
```

### 2. Update Loss Computation (Optional but Recommended)

**Old approach:**
```python
# Manual loss computation
losses, outputs = model.compute_total_loss(observations, actions, beta=1.0)
```

**New approach (recommended):**
```python
# Using LossAggregator
from multi_task_learning import LossAggregator

loss_agg = LossAggregator()
outputs = model.forward_sequence(observations, actions)
losses_dict, task_losses = loss_agg.compute_losses(outputs, beta=1.0)
```

**Benefits:**
- Modular loss computation
- Easy to add/remove loss terms
- Better monitoring of individual components

### 3. Update RGB Optimizer Usage

**Old approach:**
```python
# RGB integrated in model
rgb = RGB()
# ... manual setup
```

**New approach (clearer):**
```python
from multi_task_learning import RGB

rgb = RGB()
rgb.task_num = 3
rgb.device = 'cuda'
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()

# Use in training loop
optimizer.zero_grad()
rgb.backward(task_losses)
optimizer.step()
```

---

## Complete Training Loop Example

### Old Style

```python
# Old training loop
model = DPGMMVariationalRecurrentAutoencoder(...)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

for batch in dataloader:
    obs, actions = batch

    # Forward and loss
    losses, outputs = model.compute_total_loss(obs, actions)

    # Backward
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()
```

### New Style (Recommended)

```python
# New training loop with modular components
from world_model import DPGMMVariationalRecurrentAutoencoder
from multi_task_learning import LossAggregator, RGB

# Setup
model = DPGMMVariationalRecurrentAutoencoder(...)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_agg = LossAggregator()
rgb = RGB()

# Configure RGB
rgb.task_num = 3
rgb.device = 'cuda'
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()

for batch in dataloader:
    obs, actions = batch

    # Forward
    outputs = model.forward_sequence(obs, actions)

    # Compute losses
    losses_dict, task_losses = loss_agg.compute_losses(
        outputs,
        beta=1.0,
        lambda_recon=1.0,
        lambda_att_dyn=0.1
    )

    # Backward with RGB
    optimizer.zero_grad()
    rgb.backward(task_losses)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    # Log individual losses
    print(f"ELBO: {losses_dict['total_elbo'].item():.4f}")
    print(f"Perceiver: {losses_dict['perceiver_loss'].item():.4f}")
    print(f"Predictive: {losses_dict['total_predictive'].item():.4f}")
```

---

## Breaking Changes

### ⚠️ None!

All changes are backward compatible:
- Original files remain in `VRNN/` directory
- Old imports still work (import from original locations)
- New modules are **wrappers** around existing code
- No logic changes to any algorithms

### What This Means

✅ **Your existing code will continue to work**
✅ **No need to update immediately**
✅ **Can migrate gradually**
✅ **Can mix old and new imports**

---

## Updated Import Cheat Sheet

### Core Model

```python
# Main world model
from world_model import DPGMMVariationalRecurrentAutoencoder
```

### Individual Pillars

```python
# Pillar 1: Perception
from perceiver_io import CausalPerceiverIO, VQPTTokenizer, PerceiverTokenPredictor

# Pillar 2: Representation
from generative_prior import DPGMMPrior, AdaptiveStickBreaking
from generative_prior.distributions import KumaraswamyStable

# Pillar 3: Dynamics
from temporal_dynamics import LSTMLayer

# Pillar 4: Attention
from attention_schema import AttentionSchema, AttentionPosterior, AttentionPrior

# Pillar 5: Optimization
from multi_task_learning import RGB, LossAggregator
from multi_task_learning.losses import ELBOLoss, PerceiverLoss, PredictiveLoss
```

### Supporting Components

```python
# Encoder/Decoder
from encoder_decoder import VAEEncoder, VAEDecoder, GramLoss

# Training
from training import DMCVBTrainer, DMCVBDataset
```

---

## Benefits of New Structure

### For Development

✅ **Clear Module Organization**: Each pillar has dedicated directory
✅ **Better Documentation**: Each module has comprehensive README
✅ **Easier Navigation**: Know exactly where to find components
✅ **Modular Testing**: Test each component independently

### For Understanding

✅ **Theoretical Alignment**: Structure matches Five Pillars philosophy
✅ **Tensor Shape Docs**: Clear documentation in each module
✅ **Usage Examples**: Complete examples in READMEs
✅ **Integration Guide**: `world_model/README.md` shows complete flow

### For Extension

✅ **Easy to Add**: Add new components to appropriate modules
✅ **Easy to Swap**: Replace components (e.g., different optimizer)
✅ **Easy to Ablate**: Remove modules to test importance
✅ **Easy to Share**: Each module is self-contained

---

## Validation

### Integration Tests

Run comprehensive integration tests:

```bash
python tests/test_phase_5_6_integration.py
```

This validates:
- All module imports work
- Model instantiation succeeds
- Forward pass completes
- Loss computation matches original
- RGB gradient balancing works
- Training step executes correctly

### Individual Module Tests

```bash
# Multi-task learning
python multi_task_learning/tests/test_rgb_balancing.py
python multi_task_learning/tests/test_loss_aggregation.py

# Import validation
python multi_task_learning/tests/test_imports.py
```

---

## Troubleshooting

### Issue: Import Error

**Problem:**
```python
ModuleNotFoundError: No module named 'multi_task_learning'
```

**Solution:**
Ensure you're running from the AIME root directory:
```bash
cd /path/to/AIME
python your_script.py
```

### Issue: Model Fails to Load

**Problem:**
Model state dict doesn't match after refactoring.

**Solution:**
State dict keys are unchanged. Load checkpoints normally:
```python
model.load_state_dict(torch.load('checkpoint.pth'))
```

### Issue: Different Loss Values

**Problem:**
Loss values differ from before refactoring.

**Solution:**
This shouldn't happen. If it does:
1. Check you're using same hyperparameters
2. Verify random seed is set
3. Run comparison test (see tests/)

---

## FAQs

### Q: Do I need to retrain my models?

**A:** No! Checkpoints are fully compatible. Model parameters haven't changed.

### Q: Can I use old and new imports together?

**A:** Yes! Mix as needed:
```python
from multi_task_learning import RGB  # New
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder  # Old
```

### Q: When should I migrate?

**A:** Migrate when convenient:
- New projects: Use new structure from start
- Existing projects: Migrate gradually
- Production code: Test thoroughly before switching

### Q: What if I find a bug?

**A:** Report it! Include:
- Which module
- Error message
- Minimal reproduction code
- Whether old imports work

### Q: Where is the actual implementation code?

**A:** Current locations:
- Main model: `VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- VAE: `nvae_architecture.py`
- Other models: `models.py`

New modules are **organizational wrappers** with documentation.
Future phases may move actual implementations.

---

## Module Documentation

For detailed documentation on each module:

- `perceiver_io/README.md` - Video tokenization
- `generative_prior/README.md` - DPGMM prior
- `temporal_dynamics/README.md` - LSTM dynamics
- `attention_schema/README.md` - Attention mechanisms
- `multi_task_learning/README.md` - RGB optimizer and losses
- `encoder_decoder/README.md` - VAE components
- `world_model/README.md` - Integration overview
- `training/README.md` - Training pipeline

---

## Timeline

**Phase 1-4:** Completed previously (Perceiver IO, Generative Prior, Attention Schema)
**Phase 5:** Multi-Task Learning ✓ Complete (Nov 2, 2025)
**Phase 6:** Remaining Components ✓ Complete (Nov 2, 2025)
**Phase 7:** Integration & Validation ✓ Complete (Nov 2, 2025)

**Status:** Ready for use! 🎉

---

## Questions?

- Documentation: See `docs/` directory
- Architecture: See `docs/ARCHITECTURE_OVERVIEW.md`
- Theory: See `docs/THEORY_AND_PHILOSOPHY.md`
- Refactoring plan: See `docs/REORGANIZATION_PLAN.md`

---

**Last Updated:** November 2, 2025
