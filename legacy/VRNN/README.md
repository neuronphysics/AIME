# VRNN Directory - Original Implementation

**Status:** Functional - Still in use by new modules
**Date:** Original implementation
**Note:** This code is actively used by the new modular structure

---

## ⚠️ Important Note

This directory contains the **original implementation** of the AIME model. It is **NOT deprecated** - it's still the actual implementation that the new modules import from!

### Why It's Still Here

The new modular structure (Phases 5-8) creates **organizational wrappers** around this code:
- `world_model/` imports from `VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- `training/` imports from `VRNN/dmc_vb_transition_dynamics_trainer.py`
- These are the canonical implementations

### For New Users

**Use the new modules:**
```python
from world_model import DPGMMVariationalRecurrentAutoencoder
from training import DMCVBTrainer
```

**Instead of direct imports:**
```python
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from VRNN.dmc_vb_transition_dynamics_trainer import DMCVBTrainer
```

Both work, but new modules provide better documentation and organization.

---

## Directory Contents

### Main Model
- **`dpgmm_stickbreaking_prior_vrnn.py`** (2353 lines)
  - Complete DPGMM Variational RNN implementation
  - Still the canonical implementation
  - Imported by `world_model/`

### Trainers
- **`dmc_vb_transition_dynamics_trainer.py`** (1710 lines)
  - DMC training loop
  - Dataset loader
  - Imported by `training/`

- **`training_vrnn.py`**
  - Alternative training script
  - Direct execution script

### Perceiver IO (Original)
- **`perceiver/`** directory
  - Original Perceiver implementation
  - Code reorganized into `perceiver_io/` module
  - This version still works but new version is documented

### Utilities
- **`grad_diagnostics.py`** - Gradient monitoring
- **`vrnn_utilities.py`** - Helper functions
- **`advanced_metrics.py`** - Evaluation metrics
- Various other utilities

### Scripts
- **`run_perceiver_io_dmc_vb.py`** - Run Perceiver training
- **`main.py`** - Main training entry point
- **`run_VRNN_minimum.sh`** - Shell script for cluster

---

## Relationship to New Modules

| This Directory | New Module | Relationship |
|---------------|------------|--------------|
| `dpgmm_stickbreaking_prior_vrnn.py` | `world_model/` | Imported as-is |
| `dmc_vb_transition_dynamics_trainer.py` | `training/` | Imported as-is |
| `perceiver/` | `perceiver_io/` | Code reorganized |
| Various utils | Multiple modules | Functionality extracted |

---

## Files Moved to New Modules

- ✅ `RGB.py` → `multi_task_learning/rgb_optimizer.py`
- ✅ `lstm.py` → `temporal_dynamics/lstm.py`

These files were moved because they are standalone components that fit better in the new modular structure.

---

## Documentation

### For This Code
- Read inline comments and docstrings
- See class and method documentation

### For New Modules
- `docs/ARCHITECTURE_OVERVIEW.md` - System overview
- `docs/MIGRATION_GUIDE.md` - How to use new structure
- `world_model/README.md` - Integration guide
- `training/README.md` - Training guide

---

## Running Code from This Directory

### Direct Training
```bash
# Original training script
python VRNN/training_vrnn.py --env humanoid_walk --batch-size 8

# Perceiver training
python VRNN/run_perceiver_io_dmc_vb.py

# Main training
python VRNN/main.py --env cheetah-run
```

### Importing in Python
```python
# Direct import (works)
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder

# Recommended: Use new modules (better docs)
from world_model import DPGMMVariationalRecurrentAutoencoder
```

---

## Future Plans

### Short Term (Current)
- Code stays here
- Imported by new modules
- Fully functional

### Optional Future Phase
- Could move implementations into new modules
- Would consolidate all code
- Not urgent - current structure works well

---

## Summary

✅ **This code is active and used**
✅ **Not deprecated or legacy**
✅ **Canonical implementation**
✅ **Imported by new modules**
✅ **You can use it directly or via new modules**

---

**Last Updated:** November 2, 2025
**Status:** Active Implementation
