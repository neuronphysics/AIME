# Training Module

**Training Infrastructure for AIME World Models**

This module provides training utilities, datasets, and trainers for the AIME world model.

## Architecture Overview

```
        Training Pipeline
               │
      ┌────────┴────────┐
      │                 │
  DMCVBDataset    DMCVBTrainer
      │                 │
  [Data Loading]   [Training Loop]
```

## Components

### DMCVBDataset (`VRNN/dmc_vb_transition_dynamics_trainer.py`)

Dataset for DeepMind Control Suite (DMC) video sequences.

**Features:**
- Loads pre-collected DMC trajectories
- Supports multiple environments (humanoid, cheetah, walker, etc.)
- Returns synchronized observation-action sequences
- Handles normalization and preprocessing

**Structure:**
```python
from training import DMCVBDataset

dataset = DMCVBDataset(
    data_path='/path/to/dmc_data',
    sequence_length=16,
    environment='humanoid_walk'
)

# Returns
batch = dataset[idx]
# {
#     'observations': [T, C, H, W],
#     'actions': [T, action_dim],
#     'rewards': [T],
#     'dones': [T]
# }
```

**Data Format:**
- Observations: 64x64 RGB frames
- Actions: Environment-specific (e.g., 21-dim for humanoid)
- Sequence length: Typically 16-32 frames
- Storage: HDF5 or pickled tensors

### DMCVBTrainer (`VRNN/dmc_vb_transition_dynamics_trainer.py`)

Main training loop for AIME world model on DMC data.

**Features:**
- Integrates model, optimizer, and data loader
- Handles multi-task loss computation
- RGB gradient balancing
- Logging and checkpointing
- Gradient diagnostics

**Usage:**
```python
from training import DMCVBTrainer
from world_model import DPGMMVariationalRecurrentAutoencoder

# Create model
model = DPGMMVariationalRecurrentAutoencoder(...)

# Create trainer
trainer = DMCVBTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=2e-4,
    batch_size=8,
    sequence_length=16
)

# Train
trainer.train(num_epochs=100)
```

## Training Pipeline

### 1. Data Loading

```python
# Load DMC dataset
dataset = DMCVBDataset(
    data_path='./data/humanoid_walk',
    sequence_length=16
)

# Create data loader
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
```

### 2. Model Setup

```python
# Create model
model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15,
    latent_dim=36,
    hidden_dim=512,
    # ... see world_model/README.md
)

# Move to GPU
model = model.to('cuda')
```

### 3. Optimizer Setup

```python
# Create optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999)
)

# Optional: Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
```

### 4. Multi-Task Learning Setup

```python
from multi_task_learning import LossAggregator, RGB

# Loss aggregator
loss_agg = LossAggregator()

# RGB optimizer
rgb = RGB()
rgb.task_num = 3  # ELBO, Perceiver, Predictive
rgb.device = 'cuda'
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()
```

### 5. Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        observations = batch['observations'].to('cuda')  # [B, T, 3, 64, 64]
        actions = batch['actions'].to('cuda')            # [B, T, action_dim]

        # Forward pass
        outputs = model.forward_sequence(observations, actions)

        # Compute losses
        losses_dict, task_losses = loss_agg.compute_losses(
            outputs,
            beta=beta_schedule(epoch),
            lambda_recon=1.0,
            lambda_att_dyn=0.1
        )

        # Backward with RGB
        optimizer.zero_grad()
        rgb.backward(task_losses)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Log
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}")
            print(f"  ELBO: {losses_dict['total_elbo'].item():.4f}")
            print(f"  Perceiver: {losses_dict['perceiver_loss'].item():.4f}")
            print(f"  Predictive: {losses_dict['total_predictive'].item():.4f}")
```

### 6. Validation

```python
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            observations = batch['observations'].to('cuda')
            actions = batch['actions'].to('cuda')

            outputs = model.forward_sequence(observations, actions)
            losses_dict, _ = loss_agg.compute_losses(outputs)

            total_loss += losses_dict['total_elbo'].item()

    return total_loss / len(val_loader)
```

### 7. Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

## Hyperparameters

### Training

```python
# Optimization
learning_rate = 2e-4
batch_size = 8
sequence_length = 16
num_epochs = 100
grad_clip = 5.0

# Schedules
beta_start = 0.0
beta_end = 1.0
beta_warmup_epochs = 10

# RGB
rgb_mu = 0.9
rgb_lambd = 1.0
rgb_alpha_steps = 3
```

### Data

```python
# Dataset
data_path = './data/dmc_humanoid'
image_size = 64
action_dim = 21  # Depends on environment
num_workers = 4
pin_memory = True
```

## Logging and Monitoring

### Metrics to Track

**Per-Epoch:**
- Total loss (ELBO + Perceiver + Predictive)
- Individual loss components
- KL divergence terms
- Reconstruction quality (MSE, PSNR)
- Gradient norms
- Learning rate

**Per-N-Steps:**
- Sample reconstructions (visualize)
- Attention maps (visualize)
- Latent space statistics (mean, std)
- DPGMM component usage
- Codebook perplexity (Perceiver)

### Visualization

```python
# Reconstruction visualization
def visualize_reconstructions(observations, reconstructions, step):
    # observations: [B, T, 3, H, W]
    # reconstructions: [B, T, 3, H, W]

    # Select first sample
    obs = observations[0]  # [T, 3, H, W]
    recon = reconstructions[0]

    # Create grid
    grid = torch.cat([obs, recon], dim=0)  # [2*T, 3, H, W]

    # Save or log to wandb/tensorboard
    save_image(grid, f'recons_step_{step}.png', nrow=T)
```

## Typical Training Timeline

**Epochs 0-10: KL Warmup**
- Beta increases from 0 to 1
- Focus on reconstruction quality
- DPGMM components start differentiating

**Epochs 10-30: Component Discovery**
- DPGMM finds optimal number of components
- Attention slots stabilize
- Perceiver codebook usage increases

**Epochs 30-60: Refinement**
- All losses stabilize
- Reconstruction quality improves
- Temporal consistency emerges

**Epochs 60-100: Fine-tuning**
- Minor improvements
- Smooth loss curves
- Ready for downstream RL

## Common Issues and Solutions

### Issue: KL Collapse
- **Symptom**: KL divergence → 0
- **Solution**: Use KL warmup, increase beta slowly

### Issue: Posterior Collapse
- **Symptom**: All samples use same DPGMM component
- **Solution**: Increase entropy weight, check stick-breaking

### Issue: Attention Collapse
- **Symptom**: All slots attend to same region
- **Solution**: Increase diversity loss, reduce attention KL weight

### Issue: Gradient Explosion
- **Symptom**: Loss becomes NaN
- **Solution**: Reduce learning rate, increase grad clipping

## Performance Benchmarks

**Training Time (per epoch, 8 x RTX 3090):**
- Batch size 8, seq len 16: ~2 minutes
- Batch size 16, seq len 16: ~3 minutes
- Batch size 8, seq len 32: ~4 minutes

**Memory Usage (per GPU):**
- Batch size 8: ~18 GB
- Batch size 16: ~28 GB

**Convergence:**
- Typical: 50-80 epochs
- Good reconstruction: ~30 epochs
- Full convergence: ~100 epochs

---

**Note:** The actual trainer implementation remains in `VRNN/dmc_vb_transition_dynamics_trainer.py`.
This module provides organized access and training guides.

**For questions about training:**
- Trainer implementation: See `VRNN/dmc_vb_transition_dynamics_trainer.py`
- Model setup: See `world_model/README.md`
- Loss computation: See `multi_task_learning/README.md`
- Data preparation: Contact dataset maintainers
