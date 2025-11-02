# AIME: Adaptive Infinite Mixture for Embodied Intelligence

**A world model for visual reinforcement learning combining Perceiver IO, DPGMM priors, and multi-task learning.**

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

---

## Overview

AIME is a hierarchical variational world model that integrates:
- **Perceiver IO** for efficient video tokenization
- **DPGMM (Dirichlet Process GMM)** for adaptive latent representations
- **Attention Schema** for spatial reasoning
- **RGB Optimizer** for multi-task gradient balancing

Trained on DeepMind Control Suite (DMC) for learning visual dynamics and control.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/AIME.git
cd AIME

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt  # If available

# Required packages
pip install geoopt opencv-python h5py wandb
```

### Basic Usage

```python
from src.world_model import DPGMMVariationalRecurrentAutoencoder
from src.multi_task_learning import LossAggregator, RGB
from src.training import DMCVBDataset

# Create model
model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15,
    latent_dim=36,
    hidden_dim=512,
    context_dim=256
)

# Training setup
loss_agg = LossAggregator()
rgb = RGB()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Training loop - see docs/MIGRATION_GUIDE.md for complete example
```

### Training

```bash
# Train on DMC using new structure
sbatch scripts/train_world_model.sh

# Or run locally
python -m src.world_model.train --domain walker --task walk
```

---

## Architecture

AIME follows a **Five Pillars** philosophy:

```
                    AIME World Model
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
  Perception          Dynamics          Optimization
      │                   │                   │
  Perceiver IO     VRNN + DPGMM          RGB MTL
      │                   │                   │
      └───────────────────┴───────────────────┘
                          │
                   [Trained Model]
```

### Five Pillars

| Pillar | Module | Purpose |
|--------|--------|---------|
| **1. Perception** | `perceiver_io/` | Video tokenization & context extraction |
| **2. Representation** | `generative_prior/` | Adaptive DPGMM prior for beliefs |
| **3. Dynamics** | `temporal_dynamics/` | LSTM-based temporal prediction |
| **4. Attention** | `attention_schema/` | Spatial attention & precision |
| **5. Optimization** | `multi_task_learning/` | RGB gradient balancing |

---

## Repository Structure

```
AIME/
├── src/                       # Source code (organized by Five Pillars)
│   ├── perceiver_io/          # Pillar 1: Perception
│   │   ├── tokenizer.py       # VQ-VAE video tokenizer
│   │   ├── predictor.py       # Token prediction
│   │   └── README.md
│   │
│   ├── generative_prior/      # Pillar 2: Representation
│   │   ├── dpgmm_prior.py     # DPGMM prior
│   │   ├── stick_breaking.py  # Stick-breaking process
│   │   └── README.md
│   │
│   ├── temporal_dynamics/     # Pillar 3: Dynamics
│   │   ├── lstm.py            # Orthogonal LSTM
│   │   └── README.md
│   │
│   ├── attention_schema/      # Pillar 4: Attention
│   │   ├── attention_schema.py    # Attention wrapper
│   │   ├── attention_posterior.py # Slot attention
│   │   └── README.md
│   │
│   ├── multi_task_learning/   # Pillar 5: Optimization
│   │   ├── rgb_optimizer.py   # RGB gradient balancing
│   │   ├── loss_aggregator.py # Loss coordination
│   │   └── README.md
│   │
│   ├── encoder_decoder/       # VAE components
│   ├── world_model/           # Complete integration
│   ├── training/              # Training infrastructure
│   ├── models.py              # Shared utilities
│   ├── nvae_architecture.py   # VAE implementation
│   └── parallel_environment.cpp   # C++ extensions
│
├── scripts/                   # Training & evaluation scripts
│   ├── train_world_model.sh   # SLURM training script
│   ├── evaluate_model.sh      # SLURM evaluation script
│   └── README.md
│
├── tests/                     # Integration tests
│   └── test_phase_5_6_integration.py
│
├── docs/                      # Documentation
│   ├── MIGRATION_GUIDE.md     # How to use new structure
│   ├── REORGANIZATION_PLAN.md # Refactoring details
│   ├── THEORY_AND_PHILOSOPHY.md
│   └── ARCHITECTURE_OVERVIEW.md
│
├── tests/                     # Integration tests
│   └── test_phase_5_6_integration.py
│
└── legacy/                    # Original implementation (archived)
    ├── VRNN/                  # Original DPGMM-VRNN code
    ├── utils/                 # Original utility scripts
    ├── scripts/               # Old SLURM scripts
    ├── agac_torch/            # AGAC RL algorithm (used by legacy utils)
    └── README.md              # Archive documentation
```

---

## Key Features

### 🎯 Adaptive Latent Space
- DPGMM prior automatically adjusts number of components
- Stick-breaking process for infinite mixtures
- Handles multi-modal distributions

### 🔄 Multi-Task Learning
- RGB optimizer balances conflicting gradients
- 4 task objectives: ELBO, Perceiver, Predictive, Adversarial
- Maintains positive projection to all tasks

### 👁️ Attention Schema
- Slot-based spatial attention
- Precision-weighted inference
- Diversity constraints

### 📹 Perceiver IO Integration
- Efficient video tokenization
- VQ-VAE with multi-head codebooks
- Context extraction for dynamics

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Migration Guide](docs/MIGRATION_GUIDE.md) | How to use the refactored structure |
| [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md) | System design and data flow |
| [Theory & Philosophy](docs/THEORY_AND_PHILOSOPHY.md) | Theoretical foundations |
| [Tensor Shape Reference](docs/TENSOR_SHAPE_REFERENCE.md) | Complete tensor flow documentation |

### Module Documentation

Each module has a comprehensive README in `src/`:
- `src/perceiver_io/README.md` - Video tokenization details
- `src/generative_prior/README.md` - DPGMM prior explanation
- `src/temporal_dynamics/README.md` - LSTM dynamics
- `src/attention_schema/README.md` - Attention mechanisms
- `src/multi_task_learning/README.md` - RGB optimization & losses
- `src/world_model/README.md` - Complete integration
- `src/training/README.md` - Training infrastructure
- `scripts/README.md` - SLURM job scripts

---

## Testing

```bash
# Run integration tests
python tests/test_phase_5_6_integration.py

# Test RGB optimizer
python multi_task_learning/tests/test_rgb_balancing.py

# Test loss aggregation
python multi_task_learning/tests/test_loss_aggregation.py

# Test imports
python multi_task_learning/tests/test_imports.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aime2024,
  title={AIME: Adaptive Infinite Mixture for Embodied Intelligence},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

### References

This work builds on:
1. [Perceiver IO](https://arxiv.org/abs/2107.14795) - Attention-based architecture
2. [NVAE](https://arxiv.org/abs/2007.03898) - Hierarchical VAE
3. [RGB Optimizer](https://openreview.net) - Multi-task learning
4. [DMC](https://arxiv.org/abs/1801.00690) - DeepMind Control Suite

---

## Project Status

✅ **Phases 5-8 Complete** (Nov 2, 2025)
- Modular structure implemented
- Comprehensive documentation
- Integration tests passing
- Migration guide available
- Final cleanup complete

**Version:** 1.0 (Refactored) - Production Ready

---

## Code Organization

### New Modular Structure (Recommended)
The codebase has been reorganized into focused modules following the Five Pillars philosophy:
- Use imports like: `from multi_task_learning import RGB`
- Each module has comprehensive documentation
- See `docs/MIGRATION_GUIDE.md` for details

### Original Implementation
The `VRNN/` directory contains the original implementation:
- Still functional and actively used
- New modules import from here
- Can be used directly if preferred
- See `VRNN/README.md` for details

### Legacy Archive
The `legacy/` directory documents the refactoring process:
- Historical record
- Migration information
- See `legacy/README.md` for details

**Both old and new imports work!** Choose based on your needs.

---

## Contributing

Contributions welcome! Please:
1. Read `docs/REORGANIZATION_PLAN.md` for structure
2. Follow the Five Pillars organization
3. Add tests for new features
4. Update relevant READMEs

---

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- geoopt (for Riemannian optimization)

### Optional Dependencies
- wandb (for logging)
- tensorboard (for visualization)
- opencv-python (for video processing)
- h5py (for data loading)

See `requirements.txt` for complete list.

---

## License

[Add your license here]

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/your-org/AIME/issues)
- **Documentation:** See `docs/` directory
- **Email:** [your-email@domain.com]

---

## Acknowledgments

- Based on research in world models and visual RL
- Built on top of PyTorch and DeepMind Control Suite
- Inspired by Perceiver, NVAE, and multi-task learning research

---

**Last Updated:** November 2, 2025
**Version:** 1.0 (Refactored)
