# Legacy Code Archive

**Date Archived:** November 2, 2025
**Reason:** Code reorganization into modular structure (Phases 5-8)

This directory contains the original AIME implementation before the modular refactoring. The code here is **functional and complete** but has been superseded by the new modular structure.

---

## What's in This Archive

### `VRNN/` - Original VRNN Implementation

The complete original implementation of the DPGMM Variational RNN model.

**Key Files:**
- `dpgmm_stickbreaking_prior_vrnn.py` (2353 lines) - Main model
- `dmc_vb_transition_dynamics_trainer.py` (1710 lines) - Trainer
- `training_vrnn.py` - Alternative training script
- `perceiver/` - Original Perceiver IO implementation
- Various utility files

**Status:** Still functional, imports work from legacy path

**Why Archived:**
- Code has been reorganized into focused modules
- New modular structure is easier to navigate
- All functionality preserved in new modules

### `utils/` - Root-level Utility Files

Various utility files that were in the root directory.

**Categories:**
- Environment wrappers (DMC, Gym, ALF)
- Data collection utilities
- Tensor utilities
- Dataset loaders

**Status:** Some still used by legacy scripts

### `scripts/` - Training and Experiment Scripts

Original training scripts and shell scripts.

**Files:**
- `run_WorldModel_D2E_minimum.sh` - Original SLURM training script
- Various experiment scripts

**Status:** Reference implementations, superseded by `scripts/` in root

### `agac_torch/` - AGAC RL Algorithm

PyTorch implementation of Adversarially Guided Actor Critic (AGAC).

**Purpose:** Provides RL agent (PPO) for model-based control with world models

**Used by:**
- `utils/WorldModel_D2E.py` - World model planning and control
- `utils/WorldModel_D2E_Utils.py` - RL utilities
- `utils/WorldModel_D2E_Structures.py` - Agent structures

**Status:** Self-contained module, only used by legacy utilities

**Note:** This is a complete, independent RL implementation with:
- MPI parallelization support
- Neptune logging integration
- MiniGrid environment experiments
- Own README, configs, requirements

---

## Relationship to New Structure

### New Modules Replace Legacy Code

| Legacy Location | New Location | Status |
|----------------|--------------|--------|
| `VRNN/RGB.py` | `multi_task_learning/rgb_optimizer.py` | Moved ✓ |
| `VRNN/lstm.py` | `temporal_dynamics/lstm.py` | Moved ✓ |
| `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | `world_model/` (wrapper) | Referenced |
| `VRNN/perceiver/` | `perceiver_io/` | Reorganized |
| Various DPGMM code | `generative_prior/` | Reorganized |
| Various attention code | `attention_schema/` | Reorganized |

### Files Still Referenced

Some files in `VRNN/` are still the canonical implementation:
- `dpgmm_stickbreaking_prior_vrnn.py` - Main model (imported by world_model)
- `dmc_vb_transition_dynamics_trainer.py` - Trainer (imported by training)

These work perfectly fine and are imported by the new module wrappers.

---

## Should You Use This Code?

### Use New Modules If:
- ✅ Starting a new project
- ✅ Want clean modular structure
- ✅ Need better documentation
- ✅ Building on specific components

### Use Legacy Code If:
- ⚠️ Maintaining old projects
- ⚠️ Have existing checkpoints
- ⚠️ Need exact original behavior
- ⚠️ Running old experiments

**Note:** Both work! The new modules import from legacy where needed.

---

## Migration Path

If you're using legacy code and want to migrate:

1. **Read the Migration Guide:**
   ```
   docs/MIGRATION_GUIDE.md
   ```

2. **Update Imports:**
   ```python
   # Old
   from VRNN.RGB import RGB
   from VRNN.lstm import LSTMLayer

   # New
   from multi_task_learning import RGB
   from temporal_dynamics import LSTMLayer
   ```

3. **Test Your Code:**
   ```bash
   python tests/test_phase_5_6_integration.py
   ```

4. **Gradual Migration:**
   You can mix old and new imports! Migrate one component at a time.

---

## Documentation

### Original Documentation
- See files in `VRNN/` for inline comments
- Original README at repository root (now updated)

### New Documentation
- `docs/MIGRATION_GUIDE.md` - How to migrate
- `docs/QUICK_REFERENCE.md` - Quick access to new API
- Module READMEs in each new module directory

---

## Key Differences

### Structure
- **Old:** Monolithic files (2000+ lines)
- **New:** Focused modules (< 500 lines)

### Organization
- **Old:** Files grouped by training pipeline
- **New:** Components grouped by theoretical function (Five Pillars)

### Documentation
- **Old:** Inline comments
- **New:** Comprehensive READMEs with tensor shapes, examples, theory

### Imports
- **Old:** `from VRNN.module import Class`
- **New:** `from pillar_module import Class`

---

## What Was NOT Changed

✅ **Model Logic:** Zero changes to algorithms
✅ **Model Weights:** Checkpoints compatible
✅ **Hyperparameters:** Default values unchanged
✅ **Training Procedure:** Same training loop
✅ **Performance:** Identical results

---

## Future of Legacy Code

### Short Term (Phases 5-8)
- ✅ Archived but accessible
- ✅ Still imported by new modules where needed
- ✅ Fully functional

### Medium Term (Optional Future Phase)
- Could move actual implementations into new modules
- Would become pure reference
- Still kept for comparison

### Long Term
- Reference implementation
- Historical record
- Comparison baseline

---

## File Inventory

### VRNN/ Directory (~15 Python files)
```
VRNN/
├── dpgmm_stickbreaking_prior_vrnn.py    # Main model (2353 lines)
├── dmc_vb_transition_dynamics_trainer.py # Trainer (1710 lines)
├── training_vrnn.py                      # Alt trainer
├── perceiver/                            # Perceiver implementation
│   ├── video_prediction_perceiverIO.py  # Original (1232 lines)
│   ├── modules.py
│   ├── vector_quantize.py
│   └── ...
├── grad_diagnostics.py                   # Gradient monitoring
├── vrnn_utilities.py                     # Helper functions
└── ...
```

### Root Utilities (~24 Python files)
```
Root utilities:
├── Environment wrappers (6 files)
│   ├── alf_environment.py
│   ├── dmc_gym_wrapper.py
│   └── ...
├── Data utilities (4 files)
│   ├── dataset.py
│   ├── DataCollectionD2E.py
│   └── ...
├── Tensor utilities (3 files)
├── Model utilities (4 files)
│   ├── models.py (2265 lines)
│   ├── nvae_architecture.py (920 lines)
│   └── ...
└── Others (7 files)
```

---

## Accessing Legacy Code

### From New Modules
```python
# New modules import from legacy where needed
# Example: world_model/__init__.py
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
```

### Direct Import
```python
# You can still import directly if needed
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from VRNN.dmc_vb_transition_dynamics_trainer import DMCVBTrainer
```

### Running Legacy Scripts
```bash
# Original training script still works
python VRNN/training_vrnn.py --env humanoid_walk

# Original Perceiver script still works
python VRNN/run_perceiver_io_dmc_vb.py
```

---

## Questions?

- **New structure:** See `docs/` directory
- **Migration:** See `docs/MIGRATION_GUIDE.md`
- **Quick start:** See `docs/QUICK_REFERENCE.md`
- **Theory:** See `docs/THEORY_AND_PHILOSOPHY.md`

---

## Summary

✅ **Legacy code is preserved and functional**
✅ **New modular structure is recommended**
✅ **Both can coexist peacefully**
✅ **Migration is optional and gradual**
✅ **Zero functionality lost**

The refactoring improved organization and documentation without changing any core functionality!

---

**Last Updated:** November 2, 2025
**Archive Created During:** Phases 5-8 Refactoring
