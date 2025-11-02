# AIME Architecture Overview

**Visual guide to system components and their interactions**

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The Five Pillars Diagram](#the-five-pillars-diagram)
3. [Data Flow: Forward Pass](#data-flow-forward-pass)
4. [Component Interaction Patterns](#component-interaction-patterns)
5. [Training Loop](#training-loop)
6. [Module Dependency Graph](#module-dependency-graph)
7. [Inference vs Training Modes](#inference-vs-training-modes)

---

## High-Level Architecture

### The 30-Second Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AIME WORLD MODEL                              │
│                 (Adaptive Infinite Mixture Engine)                    │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
        ┌────────────────┐  ┌────────────┐  ┌────────────┐
        │   PERCEPTION   │  │ PREDICTION │  │  CONTROL   │
        │ (Perceiver IO) │  │   (VRNN)   │  │  (Policy)  │
        └────────────────┘  └────────────┘  └────────────┘
                │                  │              │
                │      ┌───────────┴──────┐       │
                │      ▼                  ▼       │
                │  ┌──────────┐      ┌────────┐  │
                │  │ BELIEFS  │      │ ACTION │  │
                │  │ (DPGMM)  │      │ SCHEMA │  │
                │  └──────────┘      └────────┘  │
                │                                 │
                └────────────┬────────────────────┘
                             ▼
                     ┌──────────────┐
                     │  ATTENTION   │
                     │   (Slots)    │
                     └──────────────┘
```

**Key insight**: AIME is a **perception-prediction loop** with attention modulating both.

---

## The Five Pillars Diagram

### Cognitive Functions → Neural Modules

```
╔═══════════════════════════════════════════════════════════════════════╗
║                      THE FIVE PILLARS OF AIME                         ║
╚═══════════════════════════════════════════════════════════════════════╝

PILLAR 1: PERCEPTION                    PILLAR 2: REPRESENTATION
┌─────────────────────────┐            ┌──────────────────────────┐
│   Perceiver IO          │            │   DPGMM Prior            │
│   ┌───────────────┐     │            │   ┌────────────────┐    │
│   │ Video Frames  │     │            │   │ Infinite GMM   │    │
│   │   [B,T,3,H,W] │     │            │   │ p(z|h,c)       │    │
│   └───────┬───────┘     │            │   │                │    │
│           │ VQ-VAE      │            │   │ π₁ N(μ₁,σ₁²) + │    │
│   ┌───────▼───────┐     │            │   │ π₂ N(μ₂,σ₂²) + │    │
│   │ Discrete      │     │            │   │ ...            │    │
│   │ Tokens        │     │            │   │ π_K N(μ_K,σ_K²)│    │
│   │ [B,T,Ht,Wt]   │     │            │   └────────────────┘    │
│   └───────┬───────┘     │            │   Stick-Breaking:       │
│           │ Compress    │            │   v_k ~ Kumaraswamy     │
│   ┌───────▼───────┐     │            │   π_k = v_k∏(1-v_j)     │
│   │ Context       │     │            └──────────────────────────┘
│   │ [B,T,256]     │     │                      │
│   └───────────────┘     │                      │
└─────────────────────────┘                      │
           │                                     │
           └────────────┬────────────────────────┘
                        │
                        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                          PILLAR 3: DYNAMICS                           ║
║   ┌─────────────────────────────────────────────────────────────┐    ║
║   │                     VRNN (Temporal Loop)                     │    ║
║   │                                                              │    ║
║   │   h_{t-1}, c_{t-1}                                          │    ║
║   │        │                                                     │    ║
║   │        │    ┌──────────┐                                    │    ║
║   │        └───►│   LSTM   │───────► h_t, c_t                  │    ║
║   │             └─────▲────┘                                    │    ║
║   │                   │                                         │    ║
║   │           [z_t, c_t, a_t]                                   │    ║
║   │                   │                                         │    ║
║   │          ┌────────┴────────┐                                │    ║
║   │          │                 │                                │    ║
║   │     z_t: latent      c_t: context      a_t: action         │    ║
║   │     from VAE         from Perceiver    from policy         │    ║
║   └─────────────────────────────────────────────────────────────┘    ║
╚═══════════════════════════════════════════════════════════════════════╝
                        │
                        │
                        ▼
PILLAR 4: ATTENTION                    PILLAR 5: OPTIMIZATION
┌─────────────────────────┐            ┌──────────────────────────┐
│  Attention Schema       │            │   RGB Multi-Task         │
│  ┌──────────────────┐   │            │   ┌────────────────┐    │
│  │ Bottom-Up        │   │            │   │  4 Tasks:      │    │
│  │ (Slot Attention) │   │            │   │                │    │
│  │ φ(x) → K slots   │   │            │   │  g₁: ELBO      │    │
│  └────────┬─────────┘   │            │   │  g₂: Perceiver │    │
│           │             │            │   │  g₃: Predictive│    │
│  ┌────────▼─────────┐   │            │   │  g₄: Adversary │    │
│  │ Top-Down         │   │            │   └────────────────┘    │
│  │ (Prior from h_t) │   │            │   Rotation-Based:       │
│  │ h_t → A_prior    │   │            │   r_i = cos(α)g_i +     │
│  └────────┬─────────┘   │            │         sin(α)w_i       │
│           │             │            │                         │
│  ┌────────▼─────────┐   │            │   Harmonized gradient:  │
│  │ Fusion           │   │            │   ∇θ = Σ||g_i||·r_i     │
│  │ A_t [B,H,W]      │   │            └──────────────────────────┘
│  └──────────────────┘   │
└─────────────────────────┘
```

---

## Data Flow: Forward Pass

### Complete Sequence Processing

```
INPUT: observations [B, T, 3, 64, 64], actions [B, T, 6]
│
├─► PILLAR 1: PERCEPTION (Perceiver IO)
│   │
│   ├─► Tokenize video with VQ-VAE
│   │   ├─ 3D UNet Encoder: [B,T,3,64,64] → [B,C,T',H',W']
│   │   ├─ Vector Quantize: → [B,T',H',W'] discrete indices
│   │   └─ 3D UNet Decoder: → [B,T,3,64,64] reconstruction
│   │
│   └─► Extract context
│       ├─ Perceiver Encoder: tokens → [B,T,N_latent,D_latent]
│       └─ Pool: → context [B,T,256]
│
├─► PILLAR 2 + 3: REPRESENTATION + DYNAMICS (VAE + VRNN Loop)
│   │
│   │   FOR each timestep t in 1..T:
│   │   │
│   │   ├─► VAE Encoder: q(z_t | x_t, c_t)
│   │   │   ├─ input: observation x_t [B,3,64,64], context c_t [B,256]
│   │   │   ├─ hierarchical encoding (NVAE-style)
│   │   │   └─ output: z_t ~ N(μ_post, σ²_post) [B,36]
│   │   │
│   │   ├─► DPGMM Prior: p(z_t | h_t, c_t)
│   │   │   ├─ input: hidden h_t [B,512], context c_t [B,256]
│   │   │   ├─ stick-breaking: π [B,15] (mixture weights)
│   │   │   ├─ component params: μ_k [15,36], σ²_k [15,36]
│   │   │   └─ mixture: Σ_k π_k N(μ_k, σ²_k)
│   │   │
│   │   ├─► LSTM Dynamics: h_t = f(h_{t-1}, z_t, c_t, a_t)
│   │   │   ├─ concat: [z_t, c_t, a_t] → [B,36+256+6]
│   │   │   ├─ LSTM forward: (h_{t-1}, input) → h_t
│   │   │   └─ output: h_t [B,512], c_t [B,512] (cell state)
│   │   │
│   │   ├─► Attention Schema: A_t = g(x_t, h_t)
│   │   │   ├─ Bottom-up: φ(x_t) → SlotAttention → slots [B,K,D]
│   │   │   ├─ Top-down: h_t → AttentionPrior → A_prior [B,H,W]
│   │   │   └─ Fusion: → A_t [B,H,W] (final attention map)
│   │   │
│   │   └─► VAE Decoder: p(x_t | z_t, A_t)
│   │       ├─ input: z_t [B,36], attention A_t [B,H,W]
│   │       ├─ hierarchical decoding (NVAE-style with attention modulation)
│   │       └─ output: x̂_t [B,3,64,64] (reconstructed frame)
│   │
│   └─► END FOR
│
├─► PILLAR 5: OPTIMIZATION (RGB + Multi-Task Learning)
│   │
│   ├─► Compute 4 Task Losses:
│   │   │
│   │   ├─ L_ELBO = reconstruction + KL_z + KL_hierarchical + KL_attention - entropy
│   │   ├─ L_perceiver = VQ_commitment + perceiver_recon
│   │   ├─ L_predictive = attention_dynamics + diversity
│   │   └─ L_adversarial = GAN + feature_matching
│   │
│   ├─► Compute Task Gradients:
│   │   ├─ g₁ = ∇θ L_ELBO
│   │   ├─ g₂ = ∇θ L_perceiver
│   │   ├─ g₃ = ∇θ L_predictive
│   │   └─ g₄ = ∇θ L_adversarial
│   │
│   ├─► RGB Gradient Balancing:
│   │   ├─ Normalize: ḡ_i = g_i / ||g_i||
│   │   ├─ Update consensus: d_t = 0.9·d_{t-1} + 0.1·mean(ḡ_i)
│   │   ├─ Compute orthogonal helpers: w_i = (d_t - <ḡ_i,d_t>ḡ_i) / ||...||
│   │   ├─ Solve rotation angles: α_i = argmin [conflict + λ·proximity]
│   │   ├─ Rotate gradients: r_i = cos(α_i)ḡ_i + sin(α_i)w_i
│   │   └─ Aggregate: g_final = Σ ||g_i||·r_i
│   │
│   └─► Update Parameters:
│       └─ θ ← θ - lr · g_final
│
OUTPUT: reconstructions [B,T,3,64,64], latents [B,T,36], attention [B,T,H,W], losses
```

---

## Component Interaction Patterns

### Pattern 1: Encoder-Prior-Decoder (VAE Structure)

```
        ┌──────────────────────────────────────┐
        │         VAE at timestep t            │
        │                                      │
        │   x_t [B,3,64,64]                   │
        │        │                             │
        │        │                             │
        │   ┌────▼────────┐                   │
        │   │ VAE Encoder │                   │
        │   │ q(z|x,c)    │◄─── c_t [B,256]  │
        │   └────┬────────┘                   │
        │        │                             │
        │        │ z_t [B,36]                 │
        │        │                             │
        │   ┌────┼────────────────────┐       │
        │   │    │                    │       │
        │   │    ▼                    ▼       │
        │   │  Sample              KL(q||p)   │
        │   │                        ▲        │
        │   └────┬───────────────────┼────────┘
        │        │                   │
        │        │              ┌────┴────────┐
        │        │              │ DPGMM Prior │
        │        │              │ p(z|h,c)    │◄─── h_t, c_t
        │        │              └─────────────┘
        │        │
        │   ┌────▼────────┐
        │   │ VAE Decoder │
        │   │ p(x|z,A)    │◄─── A_t [B,H,W]
        │   └────┬────────┘
        │        │
        │        ▼
        │   x̂_t [B,3,64,64]
        │
        └──────────────────────────────────────┘
```

**Key insight**: The prior p(z|h,c) is **context-dependent** - it adapts based on:
- h_t: What happened before (temporal context)
- c_t: What I see now (perceptual context)

---

### Pattern 2: Recurrent State Update (VRNN Dynamics)

```
timestep t-1                    timestep t
┌─────────────┐                ┌─────────────┐
│   h_{t-1}   │                │     h_t     │
│   c_{t-1}   │                │     c_t     │
│ (LSTM state)│                │ (LSTM state)│
└──────┬──────┘                └──────▲──────┘
       │                              │
       │                              │
       │         ┌──────────┐         │
       └────────►│   LSTM   │─────────┘
                 └─────▲────┘
                       │
                       │
              ┌────────┴────────┐
              │                 │
         z_t [B,36]        c_t [B,256]        a_t [B,6]
              │                 │                 │
              │                 │                 │
         from VAE          from Perceiver    from policy
         Encoder           context           or dataset

Concat: [z_t, c_t, a_t] → [B, 36+256+6] = [B, 298]
```

**Key insight**: The LSTM integrates three information sources:
1. **z_t**: What I inferred (latent state)
2. **c_t**: What I perceived (sensory summary)
3. **a_t**: What I did (action context)

---

### Pattern 3: Attention Fusion (Bottom-Up + Top-Down)

```
                    Attention Schema at timestep t

BOTTOM-UP (Stimulus-Driven)          TOP-DOWN (Prediction-Driven)
┌──────────────────────┐            ┌───────────────────────┐
│  Observation x_t     │            │   Hidden State h_t    │
│  [B,3,64,64]         │            │   [B,512]             │
└──────────┬───────────┘            └──────────┬────────────┘
           │                                   │
           │ Feature Pyramid                   │ Predict Motion
           ▼                                   ▼
    ┌────────────┐                      ┌────────────┐
    │ FPN        │                      │ Attention  │
    │ φ(x)       │                      │ Prior      │
    └──────┬─────┘                      └──────┬─────┘
           │ [B,C,H,W]                         │ [B,H,W]
           │                                   │
           ▼                                   │
    ┌────────────┐                             │
    │ Slot       │                             │
    │ Attention  │                             │
    └──────┬─────┘                             │
           │                                   │
           │ K slots [B,K,D]                   │
           │ attention maps [B,K,H,W]          │
           │                                   │
           └───────────┬───────────────────────┘
                       │
                       ▼
                ┌────────────┐
                │  Fusion    │
                │  (weighted │
                │  or max)   │
                └──────┬─────┘
                       │
                       ▼
                 A_t [B,H,W]
                 (final attention map)
```

**Key insight**: Attention emerges from balancing:
- **Bottom-up**: "What stands out in the image?" (salient features)
- **Top-down**: "What do I expect to attend to?" (predictive model)

---

## Training Loop

### Full Training Iteration

```
┌───────────────────────────────────────────────────────────────┐
│                      TRAINING STEP N                          │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │  DataLoader        │
                    │  - Load batch      │
                    │  - Augment (zoom)  │
                    └─────────┬──────────┘
                              │
                    Batch: [B,T,C,H,W], [B,T,A]
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Generator Training (Main Model)    │
        └─────────────────────────────────────────┘
                              │
                              ├─► Forward Pass
                              │   (see "Data Flow" above)
                              │
                              ├─► Compute Losses
                              │   ├─ L_ELBO
                              │   ├─ L_perceiver
                              │   ├─ L_predictive
                              │   └─ L_adversarial (generator part)
                              │
                              ├─► RGB Backward
                              │   ├─ Compute task gradients g₁,g₂,g₃,g₄
                              │   ├─ Rotate + aggregate
                              │   └─ Set parameter gradients
                              │
                              └─► Optimizer Step
                                  └─ Update all model parameters
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   Discriminator Training (if step % n)  │
        └─────────────────────────────────────────┘
                              │
                              ├─► Forward Pass (discriminators only)
                              │   ├─ Real data: D(x) → score_real
                              │   └─ Fake data: D(x̂.detach()) → score_fake
                              │
                              ├─► Compute Discriminator Loss
                              │   └─ L_D = - E[log D(x)] - E[log(1 - D(x̂))]
                              │
                              ├─► Backward
                              │   └─ Standard gradient descent
                              │
                              └─► Optimizer Step
                                  └─ Update discriminator parameters only
                              │
                              ▼
                    ┌────────────────────┐
                    │  Logging           │
                    │  - Losses          │
                    │  - Gradients       │
                    │  - Visualizations  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Checkpointing     │
                    │  (every N steps)   │
                    └─────────┬──────────┘
                              │
                              ▼
                         Next batch
```

**Training schedule**:
- Generator: Every step
- Discriminator: Every `n_critic` steps (typically 1-5)
- EMA model: Updated every step with decay 0.995
- Checkpoints: Every 1000 steps

---

## Module Dependency Graph

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: APPLICATION                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  training/trainer.py (DMCVBTrainer)                      │  │
│  │  - Orchestrates training loop                            │  │
│  │  - Manages data loading, checkpointing, logging          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: INTEGRATION                                           │
│  ┌──────────────────────┐       ┌────────────────────────────┐ │
│  │ world_model/         │       │ multi_task_learning/       │ │
│  │ aime_model.py        │       │ loss_aggregator.py         │ │
│  │ (full VRNN)          │       │ rgb_optimizer.py           │ │
│  └──────────────────────┘       └────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: COMPONENTS (The Five Pillars)                        │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │ perceiver_  │ │ generative_  │ │ temporal_dynamics/      │ │
│  │ io/         │ │ prior/       │ │ vrnn_core.py            │ │
│  │ causal_     │ │ dpgmm_prior  │ │ lstm.py                 │ │
│  │ perceiver   │ │              │ └─────────────────────────┘ │
│  └─────────────┘ └──────────────┘                             │
│  ┌──────────────────┐  ┌───────────────────────────────────┐  │
│  │ attention_       │  │ encoder_decoder/                  │  │
│  │ schema/          │  │ vae_encoder.py                    │  │
│  │ attention_schema │  │ vae_decoder.py                    │  │
│  │                  │  │ discriminators.py                 │  │
│  └──────────────────┘  └───────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: PRIMITIVES                                            │
│  ┌────────────┐ ┌────────────────┐ ┌────────────────────────┐  │
│  │ Kumaraswamy│ │ VectorQuantize │ │ SlotAttention          │  │
│  │ Stable     │ │                │ │                        │  │
│  └────────────┘ └────────────────┘ └────────────────────────┘  │
│  ┌────────────┐ ┌────────────────┐ ┌────────────────────────┐  │
│  │ FPN/UNet   │ │ RoPE Positional│ │ GramLoss               │  │
│  │ blocks     │ │ Embeddings     │ │                        │  │
│  └────────────┘ └────────────────┘ └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Component Data Flow

```
perceiver_io  ────► context c_t ────► temporal_dynamics (LSTM input)
                                   └──► generative_prior (DPGMM input)

generative_prior ──► prior params ──► world_model (KL divergence)

encoder_decoder ───► z_t ──────────► temporal_dynamics (LSTM input)
                 └──► x̂_t ──────────► multi_task_learning (losses)

attention_schema ──► A_t ──────────► encoder_decoder (decoder modulation)

temporal_dynamics ─► h_t ──────────► generative_prior (prior conditioning)
                              └─────► attention_schema (top-down attention)
```

---

## Inference vs Training Modes

### Training Mode

```
┌──────────────────────────────────────────┐
│  TRAINING (model.train())                │
│                                          │
│  ✓ Dropout enabled                       │
│  ✓ All losses computed                   │
│  ✓ Discriminators active                 │
│  ✓ Gradient computation                  │
│  ✓ Diversity regularization              │
│  ✓ Attention dropout                     │
│                                          │
│  Forward pass:                           │
│    x_{1:T} → forward_sequence() → {      │
│      reconstructions,                    │
│      latents,                            │
│      all intermediate outputs            │
│    }                                     │
│                                          │
│  Backward pass:                          │
│    RGB.backward(losses) → harmonized ∇θ  │
│    optimizer.step()                      │
└──────────────────────────────────────────┘
```

### Inference Mode

```
┌──────────────────────────────────────────┐
│  INFERENCE (model.eval())                │
│                                          │
│  ✓ Dropout disabled                      │
│  ✓ Only reconstruction computed          │
│  ✓ Discriminators bypassed               │
│  ✗ No gradients                          │
│  ✓ Deterministic sampling (mean)         │
│                                          │
│  Options:                                │
│  1. Reconstruction:                      │
│     x_{1:T} → forward_sequence()         │
│             → x̂_{1:T}                    │
│                                          │
│  2. Prediction (future frames):          │
│     x_{1:T_ctx} → predict_future()       │
│                 → x̂_{T_ctx+1:T}          │
│                                          │
│  3. Latent encoding:                     │
│     x_{1:T} → encode() → z_{1:T}         │
│                                          │
│  4. Context extraction:                  │
│     x_{1:T} → extract_context()          │
│             → c_{1:T}                    │
└──────────────────────────────────────────┘
```

---

## Summary: Key Architectural Insights

### 1. **Hierarchical Abstraction**
```
Raw pixels → VQ tokens → Context vectors → Latent variables → Mixture components
(3×64×64)    (8×8)       (256-dim)         (36-dim)          (15 components)
```
Each level serves different cognitive functions.

### 2. **Context-Dependent Everything**
- Prior: p(z|h,c) not p(z)
- Attention: A(x, h) not just A(x)
- Stick-breaking: π(h) not fixed π

The model adapts to context at every level.

### 3. **Multi-Objective Harmony**
Four objectives (ELBO, Perceiver, Predictive, Adversarial) are balanced via RGB, not simple weighted sum.

### 4. **Recurrent Inference**
The VRNN loop maintains beliefs h_t across time, enabling temporal coherence and prediction.

### 5. **Attention as Precision**
Attention maps A_t modulate decoder reconstruction, implementing precision-weighted inference from active inference theory.

---

*For implementation details, see [TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md)*

*For theoretical foundations, see [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md)*

*For finding specific code, see [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)*
