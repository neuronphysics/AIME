# AIME Training Scripts

Modern SLURM scripts for training and evaluating the AIME world model using the new `src/` structure.

## Available Scripts

### 1. `train_world_model.sh`
Train the complete DPGMM-VRNN world model on DMC tasks.

**Usage:**
```bash
sbatch scripts/train_world_model.sh
```

**Configuration via environment variables:**
```bash
export AIME_ENV_NAME="my_env"      # Python environment name
export AIME_DOMAIN="walker"         # DMC domain
export AIME_TASK="walk"             # DMC task
export AIME_SEED=42                 # Random seed
sbatch scripts/train_world_model.sh
```

**Resource requirements:**
- 2x V100 GPUs
- 30GB RAM
- ~14 hours walltime

### 2. `evaluate_model.sh`
Evaluate a trained model checkpoint.

**Usage:**
```bash
sbatch scripts/evaluate_model.sh checkpoints/walker-walk-seed-1/model_epoch_500.pt
```

**Configuration:**
```bash
export AIME_NUM_EPISODES=100
sbatch scripts/evaluate_model.sh path/to/checkpoint.pt
```

**Resource requirements:**
- 1x V100 GPU
- 16GB RAM
- ~2 hours walltime

## Module Structure

All training code uses the new `src/` module structure:

```python
# Import world model
from src.world_model import DPGMMVariationalRecurrentAutoencoder

# Import training infrastructure
from src.training import DMCVBTrainer, DMCVBDataset

# Import individual components
from src.perceiver_io import CausalPerceiverIO
from src.generative_prior import DPGMMPrior
from src.multi_task_learning import RGB, LossAggregator
```

## Logs and Checkpoints

Scripts automatically create:
- `logs/` - Training logs and SLURM output
- `checkpoints/{domain}-{task}-seed-{seed}/` - Model checkpoints
- `videos/eval-{checkpoint}/` - Evaluation videos

## Cluster-Specific Configuration

The scripts are configured for Compute Canada clusters. To adapt for your cluster:

1. **Module loading:**
   ```bash
   # Edit line ~28 in each script
   module load StdEnv/2020 python/3.8.10 scipy-stack
   ```

2. **GPU specification:**
   ```bash
   #SBATCH --gres=gpu:v100:2  # Change gpu type/count
   ```

3. **Environment activation:**
   ```bash
   # Edit line ~31
   source ~/envs/${AIME_ENV_NAME}/bin/activate
   ```

## Legacy Scripts

Old scripts have been moved to `legacy/scripts/` for reference:
- `legacy/scripts/run_WorldModel_D2E_minimum.sh` - Original training script

These scripts use the old structure and are maintained for compatibility only.

## Quick Start

1. **Setup environment:**
   ```bash
   conda create -n dm_control python=3.8
   conda activate dm_control
   pip install -r requirements.txt
   ```

2. **Create necessary directories:**
   ```bash
   mkdir -p logs checkpoints videos
   ```

3. **Submit training job:**
   ```bash
   sbatch scripts/train_world_model.sh
   ```

4. **Monitor progress:**
   ```bash
   tail -f logs/world-model-*.out
   tensorboard --logdir checkpoints/
   ```

## See Also

- `docs/QUICK_REFERENCE.md` - Common tasks and code snippets
- `docs/MIGRATION_GUIDE.md` - Migrating from old to new structure
- `src/training/README.md` - Training infrastructure details
- `src/world_model/README.md` - World model architecture
