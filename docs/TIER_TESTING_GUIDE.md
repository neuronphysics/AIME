# AIME 3-Tier Testing Guide

Quick iteration strategy for testing AIME on faster environments.

---

## Overview

**Problem:** DMC Humanoid is slow (~30min/epoch). Hard to debug and iterate quickly.

**Solution:** 3-tier testing strategy with progressively complex environments.

| Tier | Environment | Speed vs Humanoid | Use Case |
|------|-------------|-------------------|----------|
| **Tier 1** | Cartpole-swingup | **10x faster** | Bug finding, gradient diagnostics, feature testing |
| **Tier 2** | Reacher-easy | **8x faster** | Validating fixes at scale, hyperparameter tuning |
| **Tier 3** | Humanoid-walk | **1x (baseline)** | Final validation before publication |

---

## Quick Start

### Step 1: Collect Data

Each tier needs data collected first:

```bash
# Tier 1: Cartpole (50 episodes, ~10 seconds)
bash scripts/collect_tier1_data.sh

# Tier 2: Reacher (200 episodes, ~1 minute)
bash scripts/collect_tier2_data.sh

# Tier 3: Humanoid (1000 episodes, ~10 minutes)
bash scripts/collect_tier3_data.sh
```

**Custom environments:**
```bash
# Different domain/task
AIME_DOMAIN=pendulum AIME_TASK=swingup bash scripts/collect_tier1_data.sh

# More/fewer episodes
AIME_NUM_EPISODES=100 bash scripts/collect_tier1_data.sh
```

### Step 2: Train

### Tier 1: Ultra-Fast Debug (< 1 min/epoch)

```bash
# Basic usage (requires data from Step 1)
bash scripts/tier1_ultrafast_debug.sh

# With debug mode (CUDA error checking)
AIME_DEBUG=1 bash scripts/tier1_ultrafast_debug.sh

# Custom environment (must match collected data)
AIME_DOMAIN=pendulum AIME_TASK=swingup bash scripts/tier1_ultrafast_debug.sh

# Custom data path
AIME_DATA_PATH=/path/to/custom.hdf5 bash scripts/tier1_ultrafast_debug.sh

# Custom seed
AIME_SEED=42 bash scripts/tier1_ultrafast_debug.sh
```

**Settings:**
- Video: 10 frames × 64×64
- Model: latent_dim=18, hidden_dim=256, max_components=8
- Training: 50 epochs, batch_size=16

**Best for:**
- Finding NaN/Inf bugs
- Testing new loss terms
- Gradient flow diagnostics
- Quick sanity checks

---

### Tier 2: Medium Validation (5-10 min/epoch)

```bash
# Basic usage
bash scripts/tier2_medium_validation.sh

# Custom environment
AIME_DOMAIN=finger AIME_TASK=spin bash scripts/tier2_medium_validation.sh
```

**Settings:**
- Video: 20 frames × 84×84
- Model: latent_dim=36, hidden_dim=512, max_components=15 (full model)
- Training: 200 epochs, batch_size=32, AMP enabled

**Best for:**
- Confirming bug fixes work at scale
- Hyperparameter tuning
- Comparing model variants
- Pre-publication checks

---

### Tier 3: Full Benchmark (production)

```bash
# Basic usage (will take hours/days!)
bash scripts/tier3_full_benchmark.sh

# Multi-seed for final results
for seed in 1 2 3 4 5; do
    AIME_SEED=$seed bash scripts/tier3_full_benchmark.sh
done
```

**Settings:**
- Video: 50 frames × 84×84 (as designed)
- Model: Full production config
- Training: 500 epochs, batch_size=32, gradient accumulation

**Best for:**
- Final validation before publication
- Benchmark comparisons
- Camera-ready results

---

## Recommended Tier 1 Environments

Fast DMC tasks for debugging (ranked by speed):

```bash
# Fastest
AIME_DOMAIN=pendulum AIME_TASK=swingup      # ~12x faster, 1 actuator
AIME_DOMAIN=cartpole AIME_TASK=swingup      # ~10x faster, 1 actuator
AIME_DOMAIN=cartpole AIME_TASK=balance      # ~10x faster, simpler

# Fast
AIME_DOMAIN=reacher AIME_TASK=easy          # ~8x faster, 2 actuators
AIME_DOMAIN=ball_in_cup AIME_TASK=catch     # ~7x faster, 2 actuators
AIME_DOMAIN=finger AIME_TASK=spin           # ~6x faster, 2 actuators

# Medium (good for Tier 2)
AIME_DOMAIN=cheetah AIME_TASK=run           # ~4x faster, 6 actuators
AIME_DOMAIN=walker AIME_TASK=walk           # ~3x faster, 6 actuators
```

---

## Development Workflow

### Workflow 1: Finding Bugs

1. **Collect Tier 1 data** - `bash scripts/collect_tier1_data.sh`
2. **Reproduce bug** - Run Tier 1 training
3. **Fix code** - Edit source files
4. **Verify fix** - Re-run Tier 1 training
5. **Scale test** - Collect Tier 2 data and confirm fix
6. **(Optional)** Final check on Tier 3

### Workflow 2: New Feature

1. **Test on synthetic data** - `python scripts/test_training_synthetic.py`
2. **Collect Tier 1 data** - Quick real-world test
3. **Implement feature** - Edit source files
4. **Tune hyperparameters** - Multiple Tier 1 runs
5. **Validate at scale** - Tier 2 training
6. **Benchmark** - Tier 3 comparison

### Workflow 3: Hyperparameter Search

1. **Tier 1** - Coarse grid search (many configs, fast)
2. **Tier 2** - Fine-tune top 3-5 configs
3. **Tier 3** - Multi-seed runs on best config

### Workflow 4: First-Time Setup

```bash
# 1. Install dependencies
pip install dm_control h5py wandb

# 2. Test synthetic data (no DMC needed)
python scripts/test_training_synthetic.py

# 3. Test Perceiver IO fix
python scripts/test_perceiver_fix.py

# 4. Collect Tier 1 data
bash scripts/collect_tier1_data.sh

# 5. Run Tier 1 training
bash scripts/tier1_ultrafast_debug.sh
```

---

## Customization

All scripts support environment variables for easy customization:

```bash
# Environment
export AIME_DOMAIN=reacher
export AIME_TASK=easy
export AIME_SEED=42

# Conda environment
export AIME_ENV_NAME=aime_env

# Logging
export WANDB_PROJECT=my-project

# Debug mode
export AIME_DEBUG=1  # Enables CUDA_LAUNCH_BLOCKING

# Run
bash scripts/tier1_ultrafast_debug.sh
```

Or pass additional args directly:

```bash
bash scripts/tier1_ultrafast_debug.sh --use_3d_conv --temporal_downsample
```

---

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in project root
cd /path/to/AIME

# Activate environment
conda activate aime_env

# Run from root
bash scripts/tier1_ultrafast_debug.sh
```

### OOM (Out of Memory)

Reduce batch size or model dimensions:

```bash
# For Tier 1
bash scripts/tier1_ultrafast_debug.sh --batch_size 8

# For Tier 2
bash scripts/tier2_medium_validation.sh --batch_size 16 --gradient_accumulation_steps 2
```

### NaN/Inf during training

Enable debug mode:

```bash
AIME_DEBUG=1 bash scripts/tier1_ultrafast_debug.sh
```

This enables CUDA_LAUNCH_BLOCKING for better error messages.

---

## Performance Comparison

Approximate epoch times on RTX 3090 (batch_size=32):

| Environment | Tier 1 (64×64, T=10) | Tier 2 (84×84, T=20) | Tier 3 (84×84, T=50) |
|-------------|----------------------|----------------------|----------------------|
| Cartpole | 5s | 45s | 3min |
| Reacher | 6s | 50s | 3.5min |
| Cheetah | 10s | 1.5min | 7min |
| Walker | 15s | 2min | 10min |
| Humanoid | 30s | 4min | 30min |

**Speedup from Tier 1 cartpole to Tier 3 humanoid: ~360x faster per epoch!**

---

## When to Use Each Tier

### Use Tier 1 when:
- ✅ Finding bugs in new code
- ✅ Testing gradient flow
- ✅ Rapid prototyping
- ✅ Checking if model trains at all
- ✅ You need results in minutes, not hours

### Use Tier 2 when:
- ✅ Validating bug fixes
- ✅ Hyperparameter tuning
- ✅ Comparing model architectures
- ✅ Pre-publication sanity checks
- ✅ You have 1-2 hours for a run

### Use Tier 3 when:
- ✅ Final validation
- ✅ Benchmark comparisons
- ✅ Paper results
- ✅ Multi-seed runs for statistics
- ✅ You're ready to wait hours/days

---

## Additional Tips

1. **Start small:** Always test on Tier 1 first, even if you think it'll work
2. **Save checkpoints:** Tier 2/3 runs are expensive, checkpoint frequently
3. **Use wandb:** Track experiments across tiers with consistent naming
4. **Parallel runs:** Use Tier 1 for hyperparameter sweeps (many parallel jobs)
5. **Debug with CUDA_LAUNCH_BLOCKING:** Slower but much better error messages

---

## See Also

- `scripts/test_perceiver_fix.py` - Test the autoregressive generation fix
- `docs/QUICK_REFERENCE.md` - Code snippets and common patterns
- `docs/MIGRATION_GUIDE.md` - Using new `src/` structure
