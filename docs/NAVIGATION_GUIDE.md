# AIME Module Map: Cognitive Functions → Files

**Quick reference for AI coders: "I want to understand/modify X" → "Read these files"**

---

## The Five Pillars (Cognitive Functions)

| Pillar | Cognitive Function | Module Directory | Key Concept |
|--------|-------------------|------------------|-------------|
| 1 | **Perception** | `perceiver_io/` | Sensory compression via VQ tokenization |
| 2 | **Representation** | `generative_prior/` | Adaptive infinite mixture of beliefs |
| 3 | **Dynamics** | `temporal_dynamics/` | Temporal belief propagation |
| 4 | **Attention** | `attention_schema/` | Precision-weighted inference |
| 5 | **Optimization** | `multi_task_learning/` | Multi-objective gradient harmony |

---

## Task-Based Navigation

### "I want to understand how observations are processed"

**Path**: Raw video → Tokens → Context

1. Start: `perceiver_io/README.md` (theory)
2. Code: `perceiver_io/tokenizer.py` (VQPTTokenizer class)
3. Detail: `perceiver_io/modules/vector_quantize.py` (VQ layer)
4. Test: `perceiver_io/tests/test_tokenizer.py` (example usage)

**Key classes**: `VQPTTokenizer`, `VectorQuantize`, `UNetEncoder3D`

---

### "I want to understand the latent space representation"

**Path**: Context → DPGMM Prior → Latent z

1. Start: `generative_prior/README.md` (theory: adaptive infinite mixtures)
2. Code: `generative_prior/dpgmm_prior.py` (DPGMMPrior class)
3. Detail: `generative_prior/stick_breaking.py` (Kumaraswamy stick-breaking)
4. Math: `docs/PHILOSOPHY_AND_THEORY.md` (section: "DPGMM Prior")

**Key classes**: `DPGMMPrior`, `AdaptiveStickBreaking`, `KumaraswamyStable`

**Key equations**:
```
π_k = v_k ∏_{j<k}(1 - v_j)  # Mixture weights from sticks
p(z|h,c) = ∑_k π_k N(μ_k(h,c), σ_k²(h,c))  # Mixture of Gaussians
```

---

### "I want to understand temporal dynamics"

**Path**: History h_{t-1} + Current evidence → New belief h_t

1. Start: `temporal_dynamics/README.md` (theory: belief propagation)
2. Code: `temporal_dynamics/vrnn_core.py` (forward_sequence method)
3. Detail: `temporal_dynamics/lstm.py` (LSTMLayer with orthogonal init)
4. Current location: `VRNN/dpgmm_stickbreaking_prior_vrnn.py:929` (LSTM), `:1200-1350` (forward_sequence)

**Key classes**: `LSTMLayer`

**Information flow**:
```
[h_{t-1}, c_{t-1}] + [z_t, context_t, action_t] → LSTM → [h_t, c_t]
```

---

### "I want to understand attention mechanisms"

**Path**: Bottom-up stimulus + Top-down expectations → Attention map

1. Start: `attention_schema/README.md` (theory: precision estimation)
2. Code: `attention_schema/attention_schema.py` (AttentionSchema wrapper)
3. Bottom-up: `attention_schema/attention_posterior.py` (stimulus-driven, slot attention)
4. Top-down: `attention_schema/attention_prior.py` (expectation-driven, motion fields)
5. Mechanism: `attention_schema/slot_attention.py` (iterative competitive binding)
6. Current location: `models.py:400-620` (AttentionSchema, AttentionPosterior, AttentionPrior)

**Key classes**: `AttentionSchema`, `AttentionPosterior`, `AttentionPrior`, `SlotAttention`

**Attention flow**:
```
Image φ(x) → SlotAttention → K slots [B,K,D], attention maps [B,K,H,W]
Hidden h_t → AttentionPrior → predicted attention [B,H,W]
Fusion → Final attention A_t [B,H,W]
```

---

### "I want to understand the loss function"

**Path**: Outputs → Individual losses → Balanced total loss

1. Start: `multi_task_learning/README.md` (theory: gradient harmony)
2. Main: `multi_task_learning/loss_aggregator.py` (combines 4 task objectives)
3. Task 1: `multi_task_learning/losses/elbo_loss.py` (reconstruction + KL terms)
4. Task 2: `multi_task_learning/losses/perceiver_loss.py` (VQ commitment)
5. Task 3: `multi_task_learning/losses/predictive_loss.py` (attention dynamics)
6. Task 4: `multi_task_learning/losses/adversarial_loss.py` (GAN)
7. Balancing: `multi_task_learning/rgb_optimizer.py` (RGB gradient balancing)
8. Current location: `VRNN/dpgmm_stickbreaking_prior_vrnn.py:1465-1560` (compute_total_loss)

**Key classes**: `RGB` (optimizer), `LossAggregator`

**Loss breakdown**:
```
L_total = L_ELBO + L_perceiver + L_predictive + L_adversarial

where gradients g_i are balanced via rotation:
  r_i = cos(α_i)ḡ_i + sin(α_i)w_i  # Rotate toward consensus
```

---

### "I want to understand the training loop"

**Path**: Dataset → Batch → Forward → Loss → Backward → Update

1. Start: `training/README.md`
2. Trainer: `training/trainer.py` (DMCVBTrainer class)
3. Dataset: `training/dataset.py` (DMCVBDataset, loads DMC sequences)
4. Diagnostics: `training/utils/grad_diagnostics.py` (gradient monitoring)
5. Current location: `VRNN/dmc_vb_transition_dynamics_trainer.py:800-1400`

**Training flow**:
```
DataLoader → [B,T,C,H,W] batch
          → model.forward_sequence()
          → compute_total_loss()
          → RGB.backward() [gradient balancing]
          → optimizer.step()
```

---

### "I want to understand the full model architecture"

**Path**: All components integrated

1. Start: `docs/PHILOSOPHY_AND_THEORY.md` (complete theoretical framework)
2. Start: `docs/ARCHITECTURE.md` (system diagram - TO BE CREATED)
3. Code: `world_model/aime_model.py` (DPGMMVariationalRecurrentAutoencoder)
4. Initialization: `world_model/initialization.py` (all _init_* methods)
5. Forward: `world_model/forward_pass.py` (forward_sequence logic)
6. Training: `world_model/training_step.py` (training_step_sequence)
7. Current location: `VRNN/dpgmm_stickbreaking_prior_vrnn.py:693-end`

**Complete information flow**:
```
Observations x_{1:T}, Actions a_{1:T}
  ↓
Perceiver: x → tokens → context c_t
  ↓
Encoder: (x_t, c_t) → q(z_t|x_t,c_t) → sample z_t
  ↓
Prior: (h_t, c_t) → DPGMM → p(z_t|h_t,c_t)
  ↓
LSTM: (h_{t-1}, z_t, c_t, a_t) → h_t
  ↓
Attention: (x_t, h_t) → posterior + prior → A_t
  ↓
Decoder: (z_t, A_t) → reconstruction x̂_t
  ↓
Losses: ELBO + Perceiver + Predictive + Adversarial
  ↓
RGB: Balance gradients → Update parameters
```

---

## File-to-Component Mapping (Current State)

### Currently Implemented Files

| File | Lines | Contains | Future Module |
|------|-------|----------|---------------|
| `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 2353 | DPGMMVariationalRecurrentAutoencoder, DPGMMPrior, AttentionSchema | Split into `world_model/`, `generative_prior/`, `temporal_dynamics/` |
| `models.py` | 2265 | AttentionPosterior, AttentionPrior, SlotAttention, TemporalDiscriminator | Split into `attention_schema/`, `encoder_decoder/` |
| `nvae_architecture.py` | 924 | VAEEncoder, VAEDecoder, GramLoss | Move to `encoder_decoder/` |
| `VRNN/perceiver/video_prediction_perceiverIO.py` | 1232 | VQPTTokenizer, PerceiverTokenPredictor, CausalPerceiverIO | Move to `perceiver_io/` |
| `VRNN/perceiver/vector_quantize.py` | 1414 | VectorQuantize | Move to `perceiver_io/modules/` |
| `VRNN/RGB.py` | 462 | RGB optimizer | Move to `multi_task_learning/` |
| `VRNN/lstm.py` | ~200 | LSTMLayer | Move to `temporal_dynamics/` |
| `VRNN/Kumaraswamy.py` | 443 | KumaraswamyStable distribution | Move to `generative_prior/` |
| `VRNN/dmc_vb_transition_dynamics_trainer.py` | 1710 | DMCVBTrainer, DMCVBDataset | Split into `training/trainer.py`, `training/dataset.py` |
| `VRNN/grad_diagnostics.py` | ~14K | Gradient monitoring tools | Move to `training/utils/` |

---

## Component Dependencies

**Dependency graph** (A → B means A uses B):

```
world_model/aime_model.py
  → perceiver_io/causal_perceiver.py
  → generative_prior/dpgmm_prior.py
  → temporal_dynamics/vrnn_core.py (which uses lstm.py)
  → attention_schema/attention_schema.py
  → encoder_decoder/vae_encoder.py
  → encoder_decoder/vae_decoder.py
  → encoder_decoder/discriminators.py

training/trainer.py
  → world_model/aime_model.py
  → training/dataset.py
  → multi_task_learning/rgb_optimizer.py
  → multi_task_learning/loss_aggregator.py

multi_task_learning/loss_aggregator.py
  → multi_task_learning/losses/elbo_loss.py
  → multi_task_learning/losses/perceiver_loss.py
  → multi_task_learning/losses/predictive_loss.py
  → multi_task_learning/losses/adversarial_loss.py
```

**Layered architecture**:
```
Layer 4 (Application): training/trainer.py
Layer 3 (Integration): world_model/aime_model.py, multi_task_learning/loss_aggregator.py
Layer 2 (Components): perceiver_io/, generative_prior/, temporal_dynamics/, attention_schema/, encoder_decoder/
Layer 1 (Primitives): Kumaraswamy, VectorQuantize, SlotAttention, etc.
```

---

## Quick Reference: Class Locations

| Class | Current File | Line | Future Module |
|-------|-------------|------|---------------|
| `DPGMMVariationalRecurrentAutoencoder` | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 693 | `world_model/aime_model.py` |
| `DPGMMPrior` | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 379 | `generative_prior/dpgmm_prior.py` |
| `AdaptiveStickBreaking` | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | ~200 | `generative_prior/stick_breaking.py` |
| `KumaraswamyStable` | `VRNN/Kumaraswamy.py` | 247 | `generative_prior/kumaraswamy.py` |
| `AttentionSchema` | `VRNN/dpgmm_stickbreaking_prior_vrnn.py` | 536 | `attention_schema/attention_schema.py` |
| `AttentionPosterior` | `models.py` | ~200 | `attention_schema/attention_posterior.py` |
| `AttentionPrior` | `models.py` | ~150 | `attention_schema/attention_prior.py` |
| `SlotAttention` | `models.py` | 59 | `attention_schema/slot_attention.py` |
| `CausalPerceiverIO` | `VRNN/perceiver/video_prediction_perceiverIO.py` | 1117 | `perceiver_io/causal_perceiver.py` |
| `VQPTTokenizer` | `VRNN/perceiver/video_prediction_perceiverIO.py` | 361 | `perceiver_io/tokenizer.py` |
| `PerceiverTokenPredictor` | `VRNN/perceiver/video_prediction_perceiverIO.py` | 529 | `perceiver_io/predictor.py` |
| `VectorQuantize` | `VRNN/perceiver/vector_quantize.py` | ~200 | `perceiver_io/modules/vector_quantize.py` |
| `VAEEncoder` | `nvae_architecture.py` | ~200 | `encoder_decoder/vae_encoder.py` |
| `VAEDecoder` | `nvae_architecture.py` | ~400 | `encoder_decoder/vae_decoder.py` |
| `TemporalDiscriminator` | `models.py` | ~600 | `encoder_decoder/discriminators.py` |
| `LSTMLayer` | `VRNN/lstm.py` | ~50 | `temporal_dynamics/lstm.py` |
| `RGB` | `VRNN/RGB.py` | 150 | `multi_task_learning/rgb_optimizer.py` |
| `DMCVBTrainer` | `VRNN/dmc_vb_transition_dynamics_trainer.py` | ~800 | `training/trainer.py` |
| `DMCVBDataset` | `VRNN/dmc_vb_transition_dynamics_trainer.py` | ~200 | `training/dataset.py` |

---

## AI Coder Workflow Examples

### Example 1: "Fix a bug in attention diversity loss"

1. Understand theory: `docs/PHILOSOPHY_AND_THEORY.md` → Section "Attention Schema"
2. Find code: This MODULE_MAP → "attention mechanisms" → `models.py:400-620`
3. Read diversity loss: `models.py:468-474` (`_compute_diversity_losses` method)
4. Test fix: Create `attention_schema/tests/test_diversity_loss.py`
5. Update docs: `attention_schema/README.md`

### Example 2: "Experiment with different stick-breaking priors"

1. Understand theory: `docs/PHILOSOPHY_AND_THEORY.md` → Section "DPGMM Prior"
2. Find code: This MODULE_MAP → "latent space representation" → `generative_prior/stick_breaking.py`
3. Current location: `VRNN/dpgmm_stickbreaking_prior_vrnn.py:~200` (AdaptiveStickBreaking)
4. Create alternative: `generative_prior/truncated_stick_breaking.py`
5. Swap in model: `world_model/aime_model.py` → change import

### Example 3: "Visualize what Perceiver tokens represent"

1. Understand theory: `docs/PHILOSOPHY_AND_THEORY.md` → Section "Perceiver IO"
2. Find code: This MODULE_MAP → "observations are processed" → `perceiver_io/tokenizer.py`
3. Load pretrained: `perceiver_io/tests/demo_perceiver_flow.py` (example)
4. Decode tokens: Use `VQPTTokenizer.decode()` method
5. Visualize codebook: Query `VQPTTokenizer.vq.codebook` embeddings

---

## Summary: Navigation Rules

**Rule 1**: Theoretical question → `docs/PHILOSOPHY_AND_THEORY.md`

**Rule 2**: "Where is class X?" → This MODULE_MAP → Quick Reference table

**Rule 3**: "How does Y work?" → This MODULE_MAP → Task-Based Navigation

**Rule 4**: "I want to modify Z" → Find module → Read module README → Edit relevant file → Write test

**Rule 5**: Lost in codebase → `docs/ARCHITECTURE.md` (system diagram)

---

*This map will be updated as refactoring progresses. Current status: PLANNING PHASE*
