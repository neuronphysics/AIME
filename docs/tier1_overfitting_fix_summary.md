# Tier 1 Perceiver IO Training - Complete Status & Critical Issue

**Last Updated**: November 2, 2025
**Status**: ⚠️ CRITICAL ISSUE DISCOVERED - Exposure Bias

---

## 📊 Current State Summary

### What's Complete ✅

#### 1. Data Collection
- **Small Dataset**: `data/tier1/cartpole_swingup.hdf5` (6.9MB, 50 episodes)
- **Large Dataset**: `data/tier1/cartpole_swingup_large.hdf5` (227MB, 500 episodes)
- **Tier 2 Dataset**: `data/tier2/reacher_easy.hdf5` (41MB)

#### 2. Training Runs Completed
All training completed with multiple configurations:

| Run | Checkpoint Dir | Best Epoch | Val Loss | Val Acc | Notes |
|-----|---------------|------------|----------|---------|-------|
| Simple | `perceiver_simple` | 50 | - | - | Initial baseline (50 epochs) |
| Long | `perceiver_tier1_long` | - | - | - | Extended training |
| Final | `perceiver_tier1_final` | 21 | 2.79 | - | Best by val loss, early stop |
| Balanced | `perceiver_tier1_balanced` | 40 | 1.54 | 91.68% | Best by val accuracy |

**Checkpoints**: ~596MB in `perceiver_simple`, ~162MB each for `tier1_balanced` and `tier1_final`

#### 3. Evaluation Infrastructure
- **Script**: `scripts/evaluate_perceiver_progression.py` ✅
- **Output**: `evaluation_results/tier1_cartpole/` with visualizations
- **Metrics**: Sample PNGs and GIFs generated for 4 test samples

---

## 🚨 CRITICAL PROBLEM: Exposure Bias

### The Issue
The model exhibits **catastrophic failure** in autoregressive generation despite good training metrics:

| Evaluation Mode | Performance | Explanation |
|-----------------|-------------|-------------|
| **Training accuracy** | 77-91% | Teacher forcing - model sees real frames at each step |
| **Autoregressive generation** | 0.01-0.37% | Model must use its own predictions - collapses to noise |

### Visual Evidence
Files in `evaluation_results/tier1_cartpole/`:
- ✅ **Context frames**: Clear, recognizable cartpole images
- ✅ **Ground truth**: Clear sequences showing cartpole dynamics
- ❌ **Generated frames**: Random colorful noise (no structure or coherence)
- ❌ **GIFs**: Essentially blank/broken (924 bytes - minimal data)

### Root Cause: Train/Test Mismatch

**During Training (Teacher Forcing)**:
```
Context: [real, real, real, real, real]
Predict: [real] -> next frame

Model learns: Given perfect context → predict next frame
```

**During Generation (Autoregressive)**:
```
Context: [real, real, real, real, real]
Generate: [predicted1, predicted2, predicted3, ...]

Model fails: Never trained to handle imperfect predictions
Errors compound → collapse to noise
```

### Why This Happened
1. **Only trained with teacher forcing** - model always sees ground truth frames
2. **Never exposed to its own mistakes** during training
3. **No autoregressive training objective** - didn't learn to correct errors
4. **High temperature sampling** (1.0) - adds randomness that model can't recover from

---

## 🔧 What Needs Fixing

### Core Training Loop Changes Required

The problem is **not** the architecture - it's the training procedure. Need to implement:

### 1. Scheduled Sampling
Gradually expose model to its own predictions during training:

```python
# Pseudocode for scheduled sampling
for epoch in range(num_epochs):
    # Start with teacher forcing, gradually use own predictions
    teacher_forcing_prob = 1.0 - (epoch / num_epochs) * 0.5  # Decay from 1.0 to 0.5

    for step in sequence:
        if random() < teacher_forcing_prob:
            input_frame = ground_truth[step]  # Teacher forcing
        else:
            input_frame = model_prediction[step]  # Use own prediction

        next_prediction = model(input_frame)
```

### 2. Autoregressive Training Loss
Add explicit multi-step generation during training:

```python
# Train on autoregressive sequences
context = sequence[:context_frames]
generated = model.generate_autoregressive(context, num_frames=10)
ground_truth = sequence[context_frames:context_frames+10]

# Loss on entire generated sequence
loss = criterion(generated, ground_truth)
```

### 3. Lower Temperature Sampling
During generation, reduce randomness:

```python
# Current: temperature=1.0 (full randomness)
# Fix: temperature=0.5 or 0.3 (more deterministic)
generated = model.generate_autoregressive(context, temperature=0.5)
```

### 4. Longer Context Windows
Give model more stability:

```python
# Current: 5 context frames
# Better: 8-10 context frames
context_frames = 10  # More history to condition on
```

---

## 📂 File Locations

### Key Files to Modify
```
scripts/train_perceiver_regularized.py   # Main training loop - add scheduled sampling
src/perceiver_io/causal_perceiver.py     # Model class - verify generate_autoregressive()
```

### Datasets
```
data/tier1/cartpole_swingup.hdf5         # 6.9MB - 50 episodes
data/tier1/cartpole_swingup_large.hdf5   # 227MB - 500 episodes ✅ Use this!
data/tier2/reacher_easy.hdf5             # 41MB - Tier 2 ready
```

### Checkpoints (Can be reused as initialization)
```
checkpoints/perceiver_tier1_balanced/best_model.pt    # Best val acc (91.68%)
checkpoints/perceiver_tier1_final/best_model.pt       # Best val loss (2.79)
```

### Evaluation
```
scripts/evaluate_perceiver_progression.py       # Evaluation script ✅
evaluation_results/tier1_cartpole/              # Output directory
```

---

## 🎯 Next Steps for Implementation

### Priority 1: Implement Scheduled Sampling
1. Modify `scripts/train_perceiver_regularized.py`
2. Add `teacher_forcing_ratio` parameter that decays during training
3. At each training step, randomly choose between ground truth or model prediction

### Priority 2: Add Autoregressive Training Objective
1. Every N epochs, compute loss on autoregressive generation
2. Backprop through entire generated sequence
3. Weight this loss alongside the standard next-frame prediction loss

### Priority 3: Adjust Generation Parameters
1. Lower temperature to 0.5 or 0.3
2. Increase context frames from 5 to 8-10
3. Re-evaluate existing checkpoints with new generation settings

### Priority 4: Retrain
```bash
# New training command with fixes
python scripts/train_perceiver_regularized.py \
    --data data/tier1/cartpole_swingup_large.hdf5 \
    --sequence_length 15 \
    --context_frames 10 \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 3e-4 \
    --dropout 0.3 \
    --val_split 0.1 \
    --teacher_forcing_decay 0.5 \  # NEW: Scheduled sampling
    --autoregressive_loss_weight 0.3 \  # NEW: AR training
    --generation_temperature 0.5 \  # NEW: Lower temperature
    --early_stopping_patience 20 \
    --out_dir checkpoints/perceiver_tier1_exposure_bias_fix \
    --seed 42
```

---

## 📈 Expected Outcome After Fix

| Metric | Before (Current) | After (Target) |
|--------|------------------|----------------|
| Training Accuracy | 77-91% | 75-85% (may decrease slightly) |
| Autoregressive Accuracy | 0.01-0.37% | 60-80% ✅ |
| Generated Video Quality | Random noise | Coherent cartpole motion ✅ |
| GIF File Size | 924 bytes | 50-100KB (actual content) ✅ |

---

## 💡 Key Insights for Next Coding Session

1. **Architecture is fine** - 6.27M parameters, Perceiver IO design is sound
2. **Training loop is broken** - teacher forcing only, no autoregressive training
3. **Quick fix exists** - scheduled sampling is a well-known solution
4. **Data is ready** - 227MB large dataset is perfect for retraining
5. **Can reuse checkpoints** - initialize from `best_model.pt` to save time

---

## 🔗 References

### Key Concepts
- **Teacher Forcing**: Always feeding ground truth during training (what we do now)
- **Scheduled Sampling**: Gradually mixing ground truth with model predictions during training
- **Exposure Bias**: Train/test mismatch when model never sees its own errors during training

### Related Papers
- Bengio et al. (2015) - "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
- Goyal et al. (2017) - "Professor Forcing: A New Algorithm for Training Recurrent Networks"

### Training Logs to Review
```
logs/tier1_balanced_training.log    # 6.0MB - most recent successful run
logs/tier1_final_training.log       # 4.1MB - early stopped at epoch 21
logs/training_extended.log          # 499KB - extended training baseline
```

---

## Summary for Future Agent

**TL;DR**: Your Perceiver IO model has good architecture and trains well (91% accuracy), but completely fails at autoregressive generation (0.37% accuracy) because it was only trained with teacher forcing. The model never learned to handle its own imperfect predictions. You need to implement scheduled sampling and autoregressive training objectives in the training loop. The 227MB dataset is ready, the evaluation script works, and you can initialize from existing checkpoints. This is a fixable training procedure issue, not an architecture problem.
