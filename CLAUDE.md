# AIME - Claude AI Assistant Guide

**Context for AI assistants working on AIME (Adaptive Infinite Mixture for Embodied Intelligence)**

---

## 🎯 What is AIME?

A **hierarchical variational world model** for visual RL combining:
- **Perceiver IO** for video tokenization
- **DPGMM (Dirichlet Process GMM)** for adaptive latent priors
- **Attention Schema** for spatial reasoning
- **RGB Optimizer** for multi-task gradient balancing

**Trained on:** DeepMind Control Suite (DMC)

---

## 📂 Repository Structure

```
AIME/
├── src/                     # ALL SOURCE CODE (use this!)
│   ├── perceiver_io/        # Pillar 1: Video tokenization
│   ├── generative_prior/    # Pillar 2: DPGMM prior
│   ├── encoder_decoder/     # Pillar 2: Hierarchical VAE
│   ├── temporal_dynamics/   # Pillar 3: LSTM dynamics
│   ├── attention_schema/    # Pillar 4: Attention mechanisms
│   ├── multi_task_learning/ # Pillar 5: RGB optimizer + losses
│   ├── world_model/         # Main model integration
│   ├── training/            # Training infrastructure
│   ├── models.py            # Shared utilities (discriminators, etc.)
│   └── nvae_architecture.py # VAE encoder/decoder implementation
│
├── scripts/                 # SLURM training scripts
├── tests/                   # Integration tests
├── docs/                    # Documentation
└── legacy/                  # OLD CODE - archived, use for reference only
    ├── VRNN/                # Original implementation
    ├── utils/               # Original utilities
    ├── agac_torch/          # AGAC RL algorithm (used by legacy)
    └── scripts/
```

**IMPORTANT:** Always import from `src.*` for new code!

---

## 🚀 Quick Start - Common Patterns

### Standard Imports

```python
# Main model
from src.world_model import DPGMMVariationalRecurrentAutoencoder

# Training & optimization
from src.multi_task_learning import RGB, LossAggregator
from src.training import DMCVBTrainer, DMCVBDataset

# Individual components (as needed)
from src.perceiver_io import CausalPerceiverIO
from src.generative_prior import DPGMMPrior, AdaptiveStickBreaking
from src.temporal_dynamics import LSTMLayer
from src.attention_schema import AttentionSchema, SlotAttention
from src.encoder_decoder import VAEEncoder, VAEDecoder
from src.models import check_tensor, TemporalDiscriminator
```

### Typical Training Loop

```python
# Create model
model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15, latent_dim=36, hidden_dim=512, device='cuda'
).to('cuda')

# Setup optimization
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_agg = LossAggregator()
rgb = RGB()
rgb.task_num = 3
rgb.device = 'cuda'
rgb.get_share_params = lambda: model.parameters()

# Training step
outputs = model.forward_sequence(observations, actions)
losses_dict, task_losses = loss_agg.compute_losses(outputs, beta=1.0)
optimizer.zero_grad()
rgb.backward(task_losses)  # Multi-task gradient balancing
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
```

---

## 🧩 Five Pillars Architecture

| Pillar | Module | What It Does |
|--------|--------|--------------|
| **1: Perception** | `src/perceiver_io/` | VQ-VAE tokenization of video → discrete tokens |
| **2: Representation** | `src/generative_prior/` | DPGMM adaptive prior (context-dependent mixtures) |
| | `src/encoder_decoder/` | Hierarchical VAE encoder/decoder (NVAE-style) |
| **3: Dynamics** | `src/temporal_dynamics/` | Orthogonal LSTM for temporal evolution |
| **4: Attention** | `src/attention_schema/` | Slot attention + precision weighting |
| **5: Optimization** | `src/multi_task_learning/` | RGB gradient balancing + loss aggregation |

**Integration:** `src/world_model/` provides `DPGMMVariationalRecurrentAutoencoder` (2353 lines in `legacy/VRNN/dpgmm_stickbreaking_prior_vrnn.py`)

---

## 🔧 Key Classes & Methods

### Main Model
- **Class:** `DPGMMVariationalRecurrentAutoencoder`
- **Location:** `legacy.VRNN.dpgmm_stickbreaking_prior_vrnn` (imported via `src.world_model`)
- **Key method:** `forward_sequence(observations, actions)` → outputs dict with losses

### RGB Multi-Task Optimizer
- **Class:** `RGB` from `src.multi_task_learning`
- **Purpose:** Balances conflicting gradients across 3+ tasks
- **Usage:** `rgb.backward(task_losses)` instead of `loss.backward()`
- **Key params:** `mu`, `lambd`, `alpha_steps`

### Loss Aggregation
- **Class:** `LossAggregator` from `src.multi_task_learning`
- **Purpose:** Coordinates ELBO, Perceiver, Predictive, Adversarial losses
- **Method:** `compute_losses(outputs, beta, lambda_recon, ...)` → (losses_dict, task_losses)

### DPGMM Prior
- **Class:** `DPGMMPrior` from `src.generative_prior`
- **Purpose:** Adaptive infinite mixture prior for latent space
- **Key:** Context-dependent, uses stick-breaking with Kumaraswamy distributions

---

## 📏 Tensor Shapes (Critical!)

```python
# Inputs
observations: [B, T, C, H, W]  # B=batch, T=sequence, C=3, H=W=84
actions:      [B, T, action_dim] # action_dim varies by env (e.g., 21 for humanoid)

# Latents
z_t:          [B, latent_dim]    # latent_dim=36 typical
h_t:          [B, hidden_dim]    # hidden_dim=512 typical
c_t:          [B, context_dim]   # context_dim=256 typical

# Perceiver tokens
tokens:       [B, T', H_t, W_t]  # Compressed spatiotemporal tokens

# DPGMM
mixture_weights: [B, K]          # K=max_components (15 typical)
component_means: [B, K, latent_dim]
component_vars:  [B, K, latent_dim]
```

---

## 🎓 Key Concepts for AI Assistants

### 1. DPGMM Prior (Pillar 2)
**Problem:** Standard VAE uses N(0,I) prior - too simple for complex beliefs
**Solution:** Adaptive infinite mixture prior: `p(z|h) = Σ π_k(h) N(z | μ_k(h), σ²_k(h))`
- All parameters generated from hidden state h_t via neural networks
- Stick-breaking construction for mixture weights
- KL computed via Monte Carlo sampling

### 2. RGB Optimizer (Pillar 5)
**Problem:** Multi-task learning has conflicting gradients
**Solution:** Rotation-Based Gradient balancing
- Ensures positive projection onto all task gradients
- Maintains convergence for all tasks simultaneously
- Used for: ELBO + Perceiver + Predictive + Adversarial losses

### 3. Attention Schema (Pillar 4)
**Problem:** Where to focus in visual scenes?
**Solution:** Dual attention system
- **AttentionPosterior:** Bottom-up from observations (SlotAttention)
- **AttentionPrior:** Top-down predictive attention
- Provides precision weighting for inference

### 4. Perceiver IO (Pillar 1)
**Problem:** Raw video is high-dimensional
**Solution:** Efficient tokenization
- VQ-VAE compresses video to discrete tokens
- Cross-attention for efficient processing
- Provides compact context for dynamics model

---

## 💡 Common Tasks for AI Assistants

### Task: Add a new loss term
1. Create loss class in `src/multi_task_learning/losses/`
2. Register in `LossAggregator.compute_losses()`
3. Update `rgb.task_num` count
4. Document shape contracts

### Task: Modify DPGMM prior
- Edit: `src/generative_prior/dpgmm_prior.py` or `stick_breaking.py`
- Note: Main integration in `legacy/VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- Test: `src/generative_prior/tests/test_dpgmm_sampling.py`

### Task: Train on new environment
1. Update `action_dim` in model initialization
2. Modify data loader in `src/training/`
3. Use SLURM script: `export AIME_DOMAIN=new_domain; sbatch scripts/train_world_model.sh`

### Task: Debug NaN/Inf issues
- Use `from src.models import check_tensor`
- Check: gradient clipping (5.0 default), KL weights (β), learning rate
- DPGMM: Check concentration parameter bounds in `src/generative_prior/distributions.py`

---

## 🔍 Where to Find Things

| Need | Location |
|------|----------|
| Main model definition | `legacy/VRNN/dpgmm_stickbreaking_prior_vrnn.py` (2353 lines) |
| Imports/wrapper | `src/world_model/__init__.py` |
| Training loop | `legacy/VRNN/dmc_vb_transition_dynamics_trainer.py` |
| Loss computation | `src/multi_task_learning/loss_aggregator.py` |
| RGB implementation | `src/multi_task_learning/rgb_optimizer.py` |
| DPGMM prior logic | `src/generative_prior/dpgmm_prior.py` + `stick_breaking.py` |
| VAE encoder/decoder | `src/nvae_architecture.py` |
| Shared utilities | `src/models.py` (discriminators, transformers, utils) |
| SLURM scripts | `scripts/train_world_model.sh`, `evaluate_model.sh` |
| Integration tests | `tests/test_phase_5_6_integration.py` |

---

## ⚠️ Important Notes for AI Assistants

1. **Always use `src.*` imports** - Never import from `legacy/` unless working on legacy code
2. **Tensor shapes matter** - AIME is shape-sensitive, always verify [B, T, ...] dimensions
3. **RGB is critical** - Multi-task training requires RGB optimizer for stability
4. **DPGMM is complex** - Stick-breaking, Kumaraswamy distributions, Monte Carlo KL
5. **Legacy code is reference** - The main model (2353 lines) is in `legacy/VRNN/` but imported via wrappers
6. **No logic changes** - Refactoring preserved all original logic, just reorganized
7. **Dependencies** - Requires: geoopt, opencv-python, h5py, wandb, einops

---

## 🏃 Running AIME

### Local Training
```bash
python -m src.world_model.train --domain walker --task walk --seed 1
```

### SLURM Cluster
```bash
# Configure
export AIME_ENV_NAME="dm_control"
export AIME_DOMAIN="walker"
export AIME_TASK="walk"
export AIME_SEED=1

# Submit
sbatch scripts/train_world_model.sh
```

### Evaluation
```bash
sbatch scripts/evaluate_model.sh checkpoints/walker-walk-seed-1/model_epoch_500.pt
```

---

## 📚 Documentation Map

- **This file (CLAUDE.md):** Quick context for AI assistants
- **README.md:** Project overview and quick start
- **docs/MIGRATION_GUIDE.md:** How to use new `src/` structure
- **docs/QUICK_REFERENCE.md:** Code snippets and common tasks
- **docs/REORGANIZATION_PLAN.md:** Refactoring details (Phases 1-10)
- **src/*/README.md:** Module-specific documentation
- **scripts/README.md:** SLURM job details

---

## 🎯 TL;DR for Claude

**What:** World model for visual RL with DPGMM prior + Perceiver IO
**Structure:** All code in `src/`, organized by Five Pillars
**Main model:** 2353 lines in `legacy/VRNN/`, imported via `src.world_model`
**Critical:** Use RGB optimizer, handle tensor shapes carefully, import from `src.*`
**Training:** SLURM scripts in `scripts/`, or `python -m src.world_model.train`

✨ **Zero breaking changes** - All original logic preserved, just better organized!
