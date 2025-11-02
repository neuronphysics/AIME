# AI Coder Quick Start Guide

**Get productive with AIME in 5 minutes**

---

## What is AIME? (30 seconds)

**AIME** = **A**daptive **I**nfinite **M**ixture **E**ngine

A world model for embodied AI that:
- **Perceives**: Compresses video into discrete tokens (Perceiver IO)
- **Believes**: Models latent space as infinite Gaussian mixture (DPGMM)
- **Predicts**: Forecasts future states via recurrent dynamics (VRNN)
- **Attends**: Computes spatial precision weights (Attention Schema)
- **Learns**: Balances 4 objectives via gradient rotation (RGB)

**In one sentence**: AIME learns a video prediction model grounded in active inference theory, using adaptive mixture of beliefs and attention-weighted reconstruction.

---

## Where is Everything? (1 minute)

### Current File Locations (Pre-Refactoring)

```
AIME/
├── VRNN/
│   ├── dpgmm_stickbreaking_prior_vrnn.py  # Main model (2353 lines)
│   ├── dmc_vb_transition_dynamics_trainer.py  # Training loop (1710 lines)
│   ├── perceiver/
│   │   └── video_prediction_perceiverIO.py  # Perceiver IO (1232 lines)
│   ├── RGB.py  # Multi-task optimizer (462 lines)
│   └── Kumaraswamy.py  # Stable distributions
├── models.py  # Attention components (2265 lines)
└── nvae_architecture.py  # VAE encoder/decoder (924 lines)
```

### Key Entry Points

| Task | File | Function |
|------|------|----------|
| **Train model** | `VRNN/dmc_vb_transition_dynamics_trainer.py` | `main()` at line ~1600 |
| **Model forward pass** | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | `forward_sequence()` at line ~1200 |
| **Compute loss** | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | `compute_total_loss()` at line ~1465 |
| **Perceiver context** | `VRNN/perceiver/video_prediction_perceiverIO.py` | `extract_context()` at line ~1240 |

---

## How Do I Run It? (2 minutes)

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - torch >= 2.0
# - einops, einx
# - geoopt (for hyperbolic geometry)
# - wandb (optional, for logging)
```

### Quick Training Test

```bash
cd /home/g/zahra-dir/AIME

# Run training for 100 steps (quick test)
python VRNN/dmc_vb_transition_dynamics_trainer.py \
    --env humanoid-walk \
    --batch-size 4 \
    --sequence-length 16 \
    --max-steps 100 \
    --device cuda

# Expected output:
# - Checkpoints saved to VRNN/checkpoints/
# - Logs to console (or wandb if enabled)
# - Should take ~5-10 min on GPU
```

### Common Commands

```bash
# Load pre-trained checkpoint
python VRNN/dmc_vb_transition_dynamics_trainer.py \
    --load-checkpoint path/to/checkpoint.pt \
    --eval-only

# Change hyperparameters
python VRNN/dmc_vb_transition_dynamics_trainer.py \
    --latent-dim 48 \
    --max-components 20 \
    --learning-rate 0.0001

# Visualize attention maps
python VRNN/dmc_vb_transition_dynamics_trainer.py \
    --visualize-attention \
    --save-visualizations
```

---

## Quick Debugging Checklist (1 minute)

### Shape Errors?
→ Read [TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md) section "Common Shape Errors"

### Loss is NaN?
→ Check:
1. Gradient clipping is enabled (`grad_clip=5.0`)
2. Beta warmup is working (KL weight starts small)
3. No division by zero in KL computation

### Model not learning?
→ Check:
1. RGB optimizer is balancing gradients (see grad_diagnostics output)
2. All 4 tasks have non-zero losses
3. Learning rate schedule is working
4. Discriminator not dominating (n_critic >= 1)

### Out of memory?
→ Reduce:
1. `batch_size` (default 8 → try 4 or 2)
2. `sequence_length` (default 16 → try 8)
3. `num_codebook_perceiver` (default 1024 → try 512)

---

## Where Do I Go Next? (Decision Tree)

```
START: What do you want to do?
│
├─► "Understand the theory and vision"
│   └─► Read: THEORY_AND_PHILOSOPHY.md
│       - Five Pillars section (10 min)
│       - Connection to Active Inference (5 min)
│
├─► "Find specific code quickly"
│   └─► Use: NAVIGATION_GUIDE.md
│       - Task-Based Navigation section
│       - Class Locations table
│
├─► "See how components fit together"
│   └─► Read: ARCHITECTURE_OVERVIEW.md
│       - Look at diagrams (5 min)
│       - Data Flow section (10 min)
│
├─► "Debug shape mismatches"
│   └─► Read: TENSOR_SHAPE_REFERENCE.md
│       - Quick Reference Tables (2 min)
│       - Per-Component Shape Documentation (as needed)
│
├─► "Help refactor the codebase"
│   └─► Read: REORGANIZATION_PLAN.md
│       - Theory-Driven Refactoring section (10 min)
│       - Session implementation plan (5 min)
│
└─► "Add a new feature"
    └─► 1. Read: THEORY_AND_PHILOSOPHY.md → Design Principles
        2. Use: NAVIGATION_GUIDE.md → Find similar code
        3. Check: REORGANIZATION_PLAN.md → Will it affect structure?
        4. Write code + tests
```

---

## Quick Reference Cards

### Card 1: Model Architecture (Copy-Paste Friendly)

```python
# Instantiate full model
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder

model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15,        # DPGMM mixture components
    input_dim=64,             # Image height/width
    latent_dim=36,            # Latent variable dimension
    hidden_dim=512,           # LSTM hidden size
    attention_dim=64,         # Slot feature dimension
    action_dim=6,             # Action space (DMC humanoid)
    sequence_length=16,       # Frames per sequence
    img_perceiver_channels=64,
    img_disc_layers=4,
    disc_num_heads=4,
    device=torch.device('cuda'),
    # Perceiver config
    num_latent_perceiver=128,
    num_latent_channels_perceiver=256,
    num_codebook_perceiver=1024,
    perceiver_code_dim=256,
).to('cuda')

# Forward pass
outputs = model.forward_sequence(
    observations,  # [B, T, 3, 64, 64]
    actions        # [B, T, 6]
)

# Compute loss
losses, _ = model.compute_total_loss(
    observations,
    actions,
    beta=1.0,              # KL weight
    entropy_weight=0.6,    # Exploration bonus
    lambda_recon=1.0,      # Reconstruction weight
    lambda_att_dyn=0.1,    # Attention dynamics weight
    lambda_gram=0.05       # Gram loss weight
)
```

### Card 2: Training Loop (Minimal Example)

```python
import torch
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from VRNN.dmc_vb_transition_dynamics_trainer import DMCVBDataset

# Setup
device = torch.device('cuda')
model = DPGMMVariationalRecurrentAutoencoder(...).to(device)
dataset = DMCVBDataset(root='data/dmc_humanoid', sequence_length=16)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Training step
for batch in loader:
    observations = batch['observations'].to(device)  # [8, 16, 3, 64, 64]
    actions = batch['actions'].to(device)            # [8, 16, 6]

    # Forward + loss
    losses, outputs = model.compute_total_loss(observations, actions)

    # RGB backward (handles gradient balancing)
    model.rgb_optimizer.backward([
        losses['total_vae_loss'],
        losses['perceiver_loss'],
        losses['attention_loss'],
        losses.get('adversarial_loss', torch.tensor(0.0))
    ])

    # Update
    model.optimizer.step()
    model.optimizer.zero_grad()

    print(f"Loss: {losses['total_vae_loss'].item():.4f}")
```

### Card 3: Common Tensor Shapes

| Variable | Shape | Meaning |
|----------|-------|---------|
| `observations` | `[8, 16, 3, 64, 64]` | 8 sequences, 16 frames, RGB, 64×64 |
| `actions` | `[8, 16, 6]` | 8 sequences, 16 timesteps, 6-dim actions |
| `context` | `[8, 16, 256]` | Perceiver context per frame |
| `latents` | `[8, 16, 36]` | VAE latent states |
| `h_t` | `[8, 512]` | LSTM hidden state at time t |
| `attention_map` | `[8, 21, 21]` | Spatial attention (21×21 grid) |
| `dpgmm_weights` | `[8, 15]` | Mixture component weights |
| `reconstructions` | `[8, 16, 3, 64, 64]` | Decoded video |

### Card 4: Key Hyperparameters

```python
# Model capacity
max_components = 15        # DPGMM: how many mixture components
latent_dim = 36           # VAE: dimensionality of z
hidden_dim = 512          # LSTM: working memory size
num_codebook_perceiver = 1024  # Perceiver: VQ codebook size

# Training
learning_rate = 0.0002    # Base LR (Adam)
batch_size = 8            # Sequences per batch
sequence_length = 16      # Frames per sequence
grad_clip = 5.0           # Gradient clipping norm
warmup_epochs = 25        # Beta warmup for KL

# Loss weights
beta = 1.0                # KL divergence weight
entropy_weight = 0.6      # Exploration bonus
lambda_recon = 1.0        # Reconstruction importance
lambda_att_dyn = 0.1      # Attention dynamics importance

# RGB optimizer
mu = 0.9                  # EMA momentum for consensus
lambd = 1.0               # Proximity weight
alpha_steps = 3           # Inner optimization steps
```

---

## Common Pitfalls (What NOT to Do)

### ❌ Don't modify code without reading theory
**Why**: Design decisions have theoretical justifications. Random changes may break subtle properties.

**Do instead**: Read relevant section in THEORY_AND_PHILOSOPHY.md first.

---

### ❌ Don't assume standard VAE
**Why**: AIME uses DPGMM prior, not standard Gaussian. KL computation is different.

**Do instead**: See `DPGMMPrior.compute_kl_divergence_mc()` for Monte Carlo KL estimation.

---

### ❌ Don't train without RGB optimizer
**Why**: The 4 task objectives naturally conflict. Standard multi-task weighted sum will fail.

**Do instead**: Always use `model.rgb_optimizer.backward()` for gradient balancing.

---

### ❌ Don't ignore attention diversity loss
**Why**: Without diversity regularization, all attention slots collapse to same location.

**Do instead**: Keep `enforce_diversity=True` in AttentionPosterior config.

---

### ❌ Don't skip beta warmup
**Why**: Starting with full KL weight causes posterior collapse (latents become deterministic).

**Do instead**: Use `warmup_epochs=25` to gradually increase beta from 0 to 1.

---

## Key Contacts & Resources

### Documentation
- **Theory**: [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md)
- **Navigation**: [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)
- **Architecture**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)
- **Shapes**: [TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md)
- **Refactoring**: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md)

### Code Locations (Critical Classes)
| Class | File | Line |
|-------|------|------|
| Main model | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 693 |
| DPGMM Prior | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 379 |
| Attention Schema | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 536 |
| Perceiver IO | `VRNN/perceiver/video_prediction_perceiverIO.py` | 1117 |
| RGB Optimizer | `VRNN/RGB.py` | 150 |
| Trainer | `VRNN/dmc_vb_transition_dynamics_trainer.py` | 800 |

### References
- Free Energy Principle: Friston (2010)
- Slot Attention: Locatello et al. (2020)
- Perceiver IO: Jaegle et al. (2021)
- VQ-VAE: van den Oord et al. (2017)
- Dirichlet Process: Ferguson (1973)

---

## TL;DR - Absolute Minimum

**What**: World model with adaptive infinite mixture beliefs and attention

**Where**: Main code in `VRNN/dpgmm_stickbreaking_prior_vrnn.py`

**Run**: `python VRNN/dmc_vb_transition_dynamics_trainer.py --env humanoid-walk`

**Read**: [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) to find things, [TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md) to debug shapes

**Understand**: 5 Pillars = Perception (Perceiver) + Representation (DPGMM) + Dynamics (VRNN) + Attention (Slots) + Optimization (RGB)

**Help**: Check docs/ directory, all questions answered there

---

*Welcome to AIME! You're now ready to contribute. 🚀*

*For deeper understanding, continue to [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md)*
