# AIME Quick Reference Guide

**Fast access to common tasks and code snippets**

---

## Common Import Patterns

### Essential Imports

```python
# Core model
from world_model import DPGMMVariationalRecurrentAutoencoder

# Multi-task learning
from multi_task_learning import RGB, LossAggregator

# Training
from training import DMCVBDataset

# Individual pillars (as needed)
from perceiver_io import CausalPerceiverIO
from generative_prior import DPGMMPrior
from temporal_dynamics import LSTMLayer
from attention_schema import AttentionSchema
from encoder_decoder import VAEEncoder, VAEDecoder
```

---

## Common Tasks

### 1. Train a Model

```python
import torch
from world_model import DPGMMVariationalRecurrentAutoencoder
from multi_task_learning import LossAggregator, RGB
from training import DMCVBDataset

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
model = DPGMMVariationalRecurrentAutoencoder(
    max_components=15,
    latent_dim=36,
    hidden_dim=512,
    context_dim=256,
    image_channels=3,
    action_dim=21,  # Depends on environment
    device=device
).to(device)

# Create optimizer and helpers
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_agg = LossAggregator()
rgb = RGB()

# Configure RGB
rgb.task_num = 3
rgb.device = device
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()

# Load data
dataset = DMCVBDataset('path/to/data', sequence_length=16)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        obs = batch['observations'].to(device)
        actions = batch['actions'].to(device)

        # Forward
        outputs = model.forward_sequence(obs, actions)

        # Compute losses
        losses_dict, task_losses = loss_agg.compute_losses(
            outputs, beta=1.0, lambda_recon=1.0, lambda_att_dyn=0.1
        )

        # Backward with RGB
        optimizer.zero_grad()
        rgb.backward(task_losses)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Log
        print(f"Epoch {epoch}, ELBO: {losses_dict['total_elbo'].item():.4f}")
```

### 2. Load a Checkpoint

```python
# Load model
checkpoint = torch.load('checkpoint.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# Continue training
print(f"Resumed from epoch {epoch}")
```

### 3. Generate Predictions

```python
model.eval()
with torch.no_grad():
    # Get context
    obs = torch.randn(1, 16, 3, 64, 64).to(device)
    actions = torch.randn(1, 16, 6).to(device)

    # Forward pass
    outputs = model.forward_sequence(obs, actions)

    # Get reconstructions
    recons = outputs['reconstructions']  # List of [B, C, H, W]

    # Visualize
    import torchvision
    grid = torchvision.utils.make_grid(torch.stack(recons), nrow=8)
    torchvision.utils.save_image(grid, 'predictions.png')
```

### 4. Inspect Loss Components

```python
# After forward pass
outputs = model.forward_sequence(obs, actions)
losses_dict, task_losses = loss_agg.compute_losses(outputs)

# Print all components
print("ELBO Components:")
print(f"  Reconstruction: {losses_dict['recon_loss'].item():.4f}")
print(f"  KL(z): {losses_dict['kl_z'].item():.4f}")
print(f"  Hierarchical KL: {losses_dict['hierarchical_kl'].item():.4f}")
print(f"  Attention KL: {losses_dict['attention_kl'].item():.4f}")

print("\nTask Losses:")
print(f"  ELBO: {task_losses[0].item():.4f}")
print(f"  Perceiver: {task_losses[1].item():.4f}")
print(f"  Predictive: {task_losses[2].item():.4f}")
```

### 5. Use Individual Components

#### Perceiver IO Only

```python
from perceiver_io import CausalPerceiverIO

perceiver = CausalPerceiverIO(
    in_channels=3,
    codebook_size=256,
    context_dim=256
)

# Extract context
context = perceiver.extract_context(observations)  # [B, T, 256]

# Predict future tokens
future_tokens = perceiver.predict_future(observations, num_steps=10)
```

#### DPGMM Prior Only

```python
from generative_prior import DPGMMPrior

dpgmm = DPGMMPrior(
    latent_dim=36,
    max_components=15,
    hidden_dim=512
)

# Sample from prior
z_prior, pi = dpgmm.sample(batch_size=8, seq_len=16)
# z_prior: [8, 16, 36]
# pi: [8, 16, 15]
```

#### RGB Optimizer Only

```python
from multi_task_learning import RGB

rgb = RGB()
rgb.task_num = 4
rgb.device = 'cuda'
rgb.rep_grad = False
rgb.get_share_params = lambda: model.parameters()

# Compute losses manually
task_losses = [loss1, loss2, loss3, loss4]

# Balance gradients
batch_weights = rgb.backward(task_losses)
```

---

## Debugging Tips

### Check Tensor Shapes

```python
# Enable shape printing
def print_shapes(outputs):
    for key, value in outputs.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                print(f"{key}: List of {len(value)} x {value[0].shape}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

outputs = model.forward_sequence(obs, actions)
print_shapes(outputs)
```

### Monitor Gradient Flow

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT")
```

### Check NaN/Inf

```python
def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")

# After forward
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        check_nan_inf(value, key)
```

---

## Configuration Templates

### Minimal Config (Fast Training)

```python
config = {
    'max_components': 5,
    'latent_dim': 16,
    'hidden_dim': 64,
    'context_dim': 32,
    'num_attention_slots': 3,
    'batch_size': 4,
    'sequence_length': 8,
}
```

### Standard Config (Recommended)

```python
config = {
    'max_components': 15,
    'latent_dim': 36,
    'hidden_dim': 512,
    'context_dim': 256,
    'num_attention_slots': 5,
    'batch_size': 8,
    'sequence_length': 16,
}
```

### Large Config (Best Quality)

```python
config = {
    'max_components': 20,
    'latent_dim': 64,
    'hidden_dim': 1024,
    'context_dim': 512,
    'num_attention_slots': 7,
    'batch_size': 16,
    'sequence_length': 32,
}
```

---

## Common Hyperparameters

### Learning Rates

```python
# Standard
lr = 2e-4

# With multipliers
base_lr = 2e-4
perceiver_lr = base_lr * 1.5  # Perceiver learns faster

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)
```

### Loss Weights

```python
# ELBO task
beta = 1.0               # KL annealing (0.0 → 1.0 over first 10 epochs)
lambda_recon = 1.0       # Reconstruction weight
entropy_weight = 0.1     # Cluster entropy

# Other tasks
lambda_att_dyn = 0.1     # Attention dynamics
lambda_gram = 0.05       # Gram matrix loss
```

### RGB Hyperparameters

```python
rgb_config = {
    'mu': 0.9,              # EMA momentum
    'lambd': 1.0,           # Proximity weight
    'alpha_steps': 3,       # Inner optimization steps
    'lr_inner': 0.2,        # Inner learning rate
    'update_interval': 1,   # Solve rotation every N steps
}
```

---

## File Locations

### Quick Find

```bash
# Find a specific class
grep -r "class ClassName" --include="*.py"

# Find imports of a module
grep -r "from module_name import" --include="*.py"

# Find where a function is defined
grep -r "def function_name" --include="*.py"
```

### Common Files

```
Main Model:
  - VRNN/dpgmm_stickbreaking_prior_vrnn.py (line ~750)

Losses:
  - multi_task_learning/losses/elbo_loss.py
  - multi_task_learning/losses/perceiver_loss.py
  - multi_task_learning/losses/predictive_loss.py

RGB Optimizer:
  - multi_task_learning/rgb_optimizer.py (line ~150)

VAE Components:
  - nvae_architecture.py (VAEEncoder: line ~183, VAEDecoder: line ~621)

Perceiver:
  - perceiver_io/causal_perceiver.py
  - perceiver_io/tokenizer.py
  - perceiver_io/predictor.py
```

---

## Testing Commands

```bash
# Full integration test
python tests/test_phase_5_6_integration.py

# RGB optimizer test
python multi_task_learning/tests/test_rgb_balancing.py

# Loss aggregation test
python multi_task_learning/tests/test_loss_aggregation.py

# Import validation
python multi_task_learning/tests/test_imports.py

# Quick syntax check
python -m py_compile path/to/file.py
```

---

## Common Errors & Solutions

### Error: "No module named 'geoopt'"

```bash
pip install geoopt
```

### Error: "CUDA out of memory"

```python
# Reduce batch size
batch_size = 4

# Reduce sequence length
sequence_length = 8

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Error: "NaN in loss"

```python
# Check inputs
assert not torch.isnan(observations).any()
assert not torch.isnan(actions).any()

# Reduce learning rate
lr = 1e-4

# Increase gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## Documentation Links

- **Full Documentation:** `docs/`
- **Migration Guide:** `docs/MIGRATION_GUIDE.md`
- **Architecture:** `docs/ARCHITECTURE_OVERVIEW.md`
- **Theory:** `docs/THEORY_AND_PHILOSOPHY.md`

### Module READMEs

- `perceiver_io/README.md`
- `generative_prior/README.md`
- `temporal_dynamics/README.md`
- `attention_schema/README.md`
- `multi_task_learning/README.md`
- `world_model/README.md`
- `training/README.md`

---

**Last Updated:** November 2, 2025
