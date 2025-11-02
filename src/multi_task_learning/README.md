# Multi-Task Learning Module

**PILLAR 5: Optimization - Gradient Harmony Across Multiple Objectives**

This module implements multi-task learning strategies for balancing multiple loss objectives in AIME.

## Architecture Overview

```
         Multi-Task Learning Pipeline
                    │
        ┌───────────┴────────────┐
        │                        │
   RGB Optimizer          Loss Aggregator
        │                        │
   [Gradient Balancing]    [Loss Computation]
```

## Theoretical Foundation

AIME optimizes 4 distinct objectives simultaneously:

1. **ELBO Task**: Reconstruction + KL divergences (generative modeling)
2. **Perceiver Task**: VQ commitment + perceiver reconstruction (video tokenization)
3. **Predictive Task**: Attention dynamics + diversity (spatial reasoning)
4. **Adversarial Task**: GAN losses + feature matching (realism)

**Challenge**: These tasks have conflicting gradients that can interfere with each other.

**Solution**: RGB (Rotation-Based Gradient Balancing) - rotates task gradients toward consensus direction while minimizing conflicts.

## Components

### 1. RGB Optimizer (`rgb_optimizer.py`)

Implements the RGB algorithm from "Preserving Gradient Harmony: A Rotation-Based Gradient Balancing for Multi-Task Conflict Remedy" (ICLR 2026 submission).

**Key Features:**
- Rotates normalized task gradients toward EMA consensus direction
- Minimizes pairwise gradient conflicts via rotation optimization
- Balances conflict reduction with proximity to original gradients
- Efficient O(TD) implementation (T=tasks, D=parameters)

**Input:** List of per-task losses
**Output:** Balanced gradient applied to shared parameters

**Usage:**
```python
from multi_task_learning import RGB

# Initialize RGB optimizer
rgb = RGB()

# In training loop
task_losses = [elbo_loss, perceiver_loss, predictive_loss, adversarial_loss]
batch_weight = rgb.backward(task_losses)
optimizer.step()
```

**Hyperparameters:**
- `mu`: EMA momentum for consensus direction (default: 0.9)
- `lambd`: Proximity weight λ (default: 1.0)
- `alpha_steps`: Inner optimization steps (default: 3)
- `lr_inner`: Learning rate for angle updates (default: 0.2)
- `update_interval`: Solve rotation every N steps (default: 1)

### 2. Loss Aggregator (`loss_aggregator.py`)

Organizes loss computation into task-specific modules.

**Responsibilities:**
- Compute individual loss terms from model outputs
- Group losses by task (ELBO, Perceiver, Predictive, Adversarial)
- Provide clean interface for RGB optimizer

**Input:** Model outputs + hyperparameters
**Output:** Dictionary of losses organized by task

**Usage:**
```python
from multi_task_learning import LossAggregator

# Initialize
loss_agg = LossAggregator()

# Compute losses
losses_dict, task_losses = loss_agg.compute_losses(
    outputs=model_outputs,
    observations=obs,
    beta=1.0,
    lambda_recon=1.0,
    # ... other hyperparameters
)

# losses_dict contains all individual loss components
# task_losses is List[Tensor] ready for RGB optimizer
```

### 3. Individual Loss Modules (`losses/`)

Each file implements a specific loss computation:

#### `losses/elbo_loss.py`
- Reconstruction loss
- KL divergence (latent z)
- KL divergence (hierarchical prior)
- KL divergence (attention)
- Cluster entropy

**Tensor Shapes:**
- Input: Model outputs from forward_sequence
- Output: Scalar loss terms

#### `losses/perceiver_loss.py`
- VQ commitment loss
- Perceiver reconstruction loss
- Codebook perplexity

#### `losses/predictive_loss.py`
- Attention dynamics loss
- Attention diversity loss

#### `losses/adversarial_loss.py` (Future)
- Generator loss
- Discriminator loss
- Feature matching loss

## Loss Flow Through AIME

### Complete Pipeline

```
observations: [B, T, C, H, W]
actions: [B, T, action_dim]
    ↓
Model.forward_sequence()
    ↓
outputs: Dict[str, List[Tensor]]
    ↓
LossAggregator.compute_losses()
    ↓
task_losses: [elbo, perceiver, predictive, adversarial]
    ↓
RGB.backward(task_losses)
    ↓
Balanced gradients applied to parameters
    ↓
optimizer.step()
```

### Loss Weights (Current Configuration)

```python
# ELBO Task
lambda_recon = 1.0        # Reconstruction weight
beta = 1.0                # KL annealing factor
entropy_weight = 0.1      # Cluster entropy weight
lambda_gram = 0.05        # Gram matrix loss weight

# Perceiver Task
perceiver_lr_multiplier = 1.5  # Higher LR for perceiver

# Predictive Task
lambda_att_dyn = 0.1      # Attention dynamics weight

# Adversarial Task (if used)
lambda_gan = 0.1          # GAN loss weight
```

## Tensor Shape Reference

All losses are scalars after reduction.

```python
# Individual loss terms (before aggregation)
losses = {
    # ELBO task components
    'recon_loss': torch.Size([]),           # Reconstruction error
    'kl_z': torch.Size([]),                 # Latent KL
    'hierarchical_kl': torch.Size([]),      # DPGMM prior KL
    'attention_kl': torch.Size([]),         # Attention KL
    'cluster_entropy': torch.Size([]),      # Cluster entropy
    'gram_enc_loss': torch.Size([]),        # Encoder Gram loss

    # Perceiver task components
    'perceiver_loss': torch.Size([]),       # VQ + reconstruction

    # Predictive task components
    'attention_dynamics_loss': torch.Size([]),  # Dynamics prediction
    'attention_diversity': torch.Size([]),      # Slot diversity

    # Task aggregations (for RGB)
    'total_vae_loss': torch.Size([]),       # ELBO task total
    'attention_loss': torch.Size([]),       # Predictive task total
}

# RGB input
task_losses = [
    losses['total_vae_loss'],     # Task 1: ELBO
    losses['perceiver_loss'],     # Task 2: Perceiver
    losses['attention_loss'],     # Task 3: Predictive
    # losses['adversarial_loss'], # Task 4: Adversarial (if enabled)
]
```

## Tests

Run demo scripts to understand the flow:

```bash
# Test RGB optimizer on synthetic gradients
python multi_task_learning/tests/test_rgb_balancing.py

# Test loss aggregation
python multi_task_learning/tests/test_loss_aggregation.py

# Test full pipeline with model
python multi_task_learning/tests/demo_mtl_flow.py
```

## Design Principles

1. **Modularity**: Each loss computation is isolated
2. **Clarity**: Loss names clearly indicate what they measure
3. **Flexibility**: Easy to add/remove loss terms
4. **Efficiency**: Minimize redundant computations
5. **Debuggability**: Each loss can be monitored independently

## Benefits for AI Coders

1. **Focused Files**: Each loss module < 200 lines
2. **Clear Responsibility**: Know exactly where each loss is computed
3. **Easy Modification**: Want to change reconstruction loss? → `losses/elbo_loss.py`
4. **Isolated Testing**: Test each loss component independently
5. **RGB Understanding**: Separate optimizer logic from loss computation

## Future Extensions

- [ ] Add adversarial loss module (currently in main model)
- [ ] Implement alternative multi-task optimizers (PCGrad, GradNorm)
- [ ] Add loss visualization utilities
- [ ] Create loss scheduling strategies
- [ ] Add gradient conflict metrics

## References

**RGB Algorithm:**
- Paper: "Preserving Gradient Harmony: A Rotation-Based Gradient Balancing for Multi-Task Conflict Remedy"
- Implementation: `rgb_optimizer.py`

**Multi-Task Learning Theory:**
- See `docs/THEORY_AND_PHILOSOPHY.md` for theoretical foundations
- See `docs/ARCHITECTURE_OVERVIEW.md` for system integration

---

**For questions about this module:**
- RGB optimizer: See inline comments in `rgb_optimizer.py`
- Loss computation: See `loss_aggregator.py` docstrings
- Integration: See `docs/ARCHITECTURE_OVERVIEW.md`
