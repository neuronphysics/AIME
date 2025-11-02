# World Model Module

**High-Level Integration of All Five Pillars**

This module provides access to the complete AIME world model that integrates all components.

## Architecture Overview

```
              AIME World Model
                     │
     ┌───────────────┼───────────────┐
     │               │               │
  Pillar 1      Pillar 2-3      Pillar 4-5
     │               │               │
 Perceiver      Generative      Multi-Task
    IO          + Dynamics       Learning
     │               │               │
     └───────────────┴───────────────┘
                     │
        DPGMMVariationalRecurrentAutoencoder
```

## What is the World Model?

The **DPGMM Variational Recurrent Autoencoder** is the main integration point for AIME.
It combines all five pillars into a unified world model for visual reinforcement learning.

### Five Pillars Integration

| Pillar | Module | Role in World Model |
|--------|--------|---------------------|
| 1: Perception | `perceiver_io/` | Video tokenization & context extraction |
| 2: Representation | `generative_prior/` | Adaptive DPGMM prior for latent beliefs |
| 3: Dynamics | `temporal_dynamics/` | LSTM-based temporal prediction |
| 4: Attention | `attention_schema/` | Spatial attention & precision estimation |
| 5: Optimization | `multi_task_learning/` | RGB gradient balancing across tasks |

## Main Model: DPGMMVariationalRecurrentAutoencoder

Located in: `VRNN/dpgmm_stickbreaking_prior_vrnn.py` (2353 lines)

### Key Methods

#### `forward_sequence(observations, actions)`

Processes a sequence of observations and actions through the complete model.

**Input:**
- observations: [B, T, C, H, W] - Video sequence
- actions: [B, T, action_dim] - Action sequence

**Output:**
- Dictionary with:
  - Reconstructions
  - Latent representations
  - Prior/posterior distributions
  - Attention maps
  - All loss components

**Flow:**
```python
observations → Perceiver (context) → Encoder (z_posterior)
                                            ↓
actions → LSTM (h_t) ← DPGMM Prior (z_prior) ← Stick-breaking
              ↓
         Attention Schema
              ↓
         Decoder → reconstructions
```

#### `compute_total_loss(observations, actions, ...)`

Computes all loss terms using the multi-task learning framework.

**Output:**
- losses_dict: All individual loss components
- outputs: Model forward pass outputs

**Note:** This method can be replaced with `LossAggregator` from `multi_task_learning/`.

### Initialization

```python
from world_model import DPGMMVariationalRecurrentAutoencoder

model = DPGMMVariationalRecurrentAutoencoder(
    # Perceiver IO
    perceiver_config={
        'codebook_size': 256,
        'context_dim': 256,
        # ... see perceiver_io/README.md
    },

    # Generative Prior (DPGMM)
    max_components=15,
    latent_dim=36,

    # Encoder/Decoder
    encoder_config={'base_channels': 32, ...},
    decoder_config={'base_channels': 32, ...},

    # Temporal Dynamics
    hidden_dim=512,
    n_lstm_layers=1,
    use_orthogonal=True,

    # Attention Schema
    attention_config={
        'num_slots': 5,
        'slot_dim': 64,
        # ... see attention_schema/README.md
    },

    # Multi-Task Learning
    use_rgb=True,
    rgb_config={
        'mu': 0.9,
        'lambd': 1.0,
        # ... see multi_task_learning/README.md
    }
)
```

### Training

```python
# Forward pass
outputs = model.forward_sequence(observations, actions)

# Compute losses
losses_dict, task_losses = loss_aggregator.compute_losses(outputs)

# Backward with RGB
rgb_optimizer.backward(task_losses)

# Update
optimizer.step()
```

## Complete Data Flow

```
Step 1: Perception
  observations [B,T,3,64,64] → CausalPerceiverIO → context [B,T,256]

Step 2: Encoding
  observations, context → VAEEncoder → z_mean, z_logvar [B,T,36]
                                    → z_posterior [B,T,36]

Step 3: Prior
  DPGMMPrior + StickBreaking → pi [B,T,15], mu_k, sigma_k [15,36]
                             → z_prior [B,T,36]

Step 4: Dynamics
  concat(z_posterior, context, actions) [B,T,298] → LSTM → h_t [B,T,512]

Step 5: Attention
  observations → AttentionSchema(h_t) → attention_maps [B,T,5,8,8]

Step 6: Decoding
  z_posterior, attention_maps → VAEDecoder → reconstructions [B,T,3,64,64]

Step 7: Multi-Task Loss
  outputs → LossAggregator → [ELBO, Perceiver, Predictive] losses
         → RGB → balanced gradients
```

## Tensor Shape Summary

| Variable | Shape | Description |
|----------|-------|-------------|
| observations | [B, T, 3, 64, 64] | Input video |
| actions | [B, T, 6] | Control signals |
| context | [B, T, 256] | Perceiver context |
| z_posterior | [B, T, 36] | Encoded latent |
| z_prior | [B, T, 36] | DPGMM prior sample |
| h_t | [B, T, 512] | LSTM hidden states |
| attention_maps | [B, T, 5, 8, 8] | Spatial attention (5 slots) |
| reconstructions | [B, T, 3, 64, 64] | Decoded video |

## Hyperparameters (Current)

```python
# Architecture
latent_dim = 36
hidden_dim = 512
context_dim = 256
max_components = 15
num_attention_slots = 5

# Training
batch_size = 8
sequence_length = 16
learning_rate = 0.0002
beta = 1.0  # KL annealing
grad_clip = 5.0

# Loss weights
lambda_recon = 1.0
lambda_att_dyn = 0.1
lambda_gram = 0.05
entropy_weight = 0.1

# RGB Optimizer
mu = 0.9
lambd = 1.0
alpha_steps = 3
```

## Usage Example

```python
# Setup
from world_model import DPGMMVariationalRecurrentAutoencoder
from multi_task_learning import LossAggregator, RGB
import torch

# Create model
model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15,
    latent_dim=36,
    hidden_dim=512,
    # ... see full config above
)

# Create loss aggregator and optimizer
loss_agg = LossAggregator()
rgb = RGB()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Training step
observations = torch.randn(8, 16, 3, 64, 64)  # Batch of sequences
actions = torch.randn(8, 16, 6)

# Forward
outputs = model.forward_sequence(observations, actions)

# Compute losses
losses_dict, task_losses = loss_agg.compute_losses(
    outputs, beta=1.0, lambda_recon=1.0, ...
)

# Backward with RGB
optimizer.zero_grad()
rgb.task_num = len(task_losses)
rgb.device = observations.device
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()
rgb.backward(task_losses)

# Update
optimizer.step()

# Log
print(f"ELBO: {losses_dict['total_elbo'].item():.4f}")
print(f"Perceiver: {losses_dict['perceiver_loss'].item():.4f}")
print(f"Predictive: {losses_dict['total_predictive'].item():.4f}")
```

## Design Principles

1. **Modular Integration**: Each pillar is independent but coordinated
2. **Clean Interfaces**: Clear inputs/outputs between components
3. **Flexible Configuration**: Easy to modify hyperparameters
4. **Monitoring**: All intermediate values accessible for debugging

## Benefits for AI Coders

1. **High-Level View**: Understand complete system from one place
2. **Module References**: Links to detailed docs for each component
3. **Data Flow Clarity**: See how tensors transform through pipeline
4. **Integration Example**: Complete training loop provided

## Future Extensions

- [ ] Add hierarchical latent variables
- [ ] Add multi-resolution processing
- [ ] Add action-conditioned prior
- [ ] Add world model rollouts for planning

---

**Note:** The actual implementation remains in `VRNN/dpgmm_stickbreaking_prior_vrnn.py`.
This module provides organized access and high-level documentation.

**For questions about this module:**
- Implementation: See `VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- Individual components: See respective module READMEs
- Theory: See `docs/THEORY_AND_PHILOSOPHY.md`
- Architecture: See `docs/ARCHITECTURE_OVERVIEW.md`
