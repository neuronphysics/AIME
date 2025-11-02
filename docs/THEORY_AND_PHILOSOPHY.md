# AIME: Philosophy, Vision & Theoretical Foundations

**Adaptive Infinite Mixture Engine**

*A world model for embodied AI grounded in active inference principles*

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architectural Components as Inference Mechanisms](#architectural-components-as-inference-mechanisms)
4. [The Five Pillars of AIME](#the-five-pillars-of-aime)
5. [Mathematical Framework](#mathematical-framework)
6. [Design Principles](#design-principles)
7. [Connection to Active Inference](#connection-to-active-inference)
8. [Future Vision](#future-vision)

---

## Core Philosophy

**AIME embodies the principle that intelligent behavior emerges from the interplay of:**

1. **Adaptive Representation** - The world is not a fixed set of categories, but an infinite mixture of dynamically discovered concepts
2. **Hierarchical Abstraction** - From pixels to discrete tokens to continuous latents to semantic slots
3. **Predictive Dynamics** - The future is predicted by integrating bottom-up sensory evidence with top-down expectations
4. **Attention as Selection** - Cognition is fundamentally about what to attend to, not just what is there
5. **Multi-Objective Harmony** - Multiple learning objectives naturally conflict; intelligence requires their harmonious balance

> *"The mind constructs the world it perceives through an infinite mixture of adaptive beliefs, continuously refined through prediction error minimization."*

---

## Theoretical Foundation

### Active Inference & The Free Energy Principle

AIME is inspired by **active inference** theory from cognitive neuroscience, which posits that the brain is a prediction machine that:

1. **Maintains generative models** of the world (DPGMM Prior + VRNN Dynamics)
2. **Minimizes variational free energy** (ELBO objective)
3. **Uses attention to select salient features** (Attention Schema)
4. **Represents uncertainty hierarchically** (Stick-breaking prior, VQ tokens, latent spaces)

While not a strict implementation of active inference, AIME incorporates its key principles:

- **Perception = Inference**: The encoder q(z|x) infers latent states from observations
- **Action = Belief Propagation**: The LSTM h_t propagates beliefs forward in time
- **Attention = Precision Weighting**: Attention maps weight sensory evidence by estimated precision
- **Learning = Model Optimization**: Training minimizes surprise (reconstruction error + KL divergence)

### Why "Adaptive Infinite Mixture"?

**Adaptive**: The model adapts both its representation (VQ codebook) and its generative prior (DPGMM components) during learning.

**Infinite**: The Dirichlet Process allows the model to discover an unbounded number of latent mixture components, growing complexity as needed.

**Mixture**: The latent space is modeled as a mixture of Gaussian components, each capturing a different mode of the data distribution.

**Engine**: AIME is designed to be a core component driving embodied AI agents in complex environments.

---

## Architectural Components as Inference Mechanisms

### 1. **Perceiver IO: Sensory Compression & Abstract Tokenization**

**Theoretical Role**: Bottom-up sensory processing

**What it does**:
- Compresses raw video (continuous, high-dimensional) into discrete tokens (categorical, compact)
- Learns a codebook of visual primitives through vector quantization
- Extracts context vectors that summarize spatiotemporal patterns

**Why discrete tokens?**
- **Compositionality**: Complex scenes are compositions of reusable visual primitives
- **Sparsity**: Most information can be represented with a small active set of tokens
- **Robustness**: Discrete representations are more robust to noise than raw pixels
- **Scalability**: Attention over tokens is O(T·H_t·W_t) vs O(T·H·W) over pixels

**Connection to cognition**:
The brain doesn't process raw retinal input - it chunks it into objects, features, and concepts. VQ tokenization is AIME's analog of **perceptual chunking**.

```
Raw Video [B,T,3,64,64]  →  VQ Tokens [B,T,8,8]  →  Context [B,T,256]
  (continuous)               (discrete primitives)     (abstract summary)
```

---

### 2. **DPGMM Prior: Adaptive Infinite Mixture of Beliefs**

**Theoretical Role**: Top-down generative model with adaptive complexity

**What it does**:
- Models the latent space p(z|h,c) as an infinite Gaussian mixture
- Uses **stick-breaking** construction to dynamically discover the number of components K
- Conditions component parameters on hidden state h and context c (context-dependent prior)

**Why Dirichlet Process?**
- **Bayesian non-parametrics**: Automatically determines model complexity from data
- **Flexibility**: Can represent multimodal distributions (e.g., multiple walking gaits)
- **Regularization**: Encourages sparse mixture weights through Beta(1,α) prior
- **Hierarchical uncertainty**: α itself is learned, adapting concentration of the prior

**Mathematics**:
```
Stick-breaking construction:
  v_k ~ Kumaraswamy(a_k(h), b_k(h))         # Posterior stick proportions
  π_k = v_k ∏_{j<k}(1 - v_j)                # Mixture weights

Prior over sticks:
  v_k ~ Beta(1, α)                          # Prior favors small K
  α ~ Gamma(α₀, β₀)                         # Hierarchical prior on concentration

Component parameters:
  μ_k(h,c), σ_k²(h,c) = ComponentNN(h,c)   # Context-dependent Gaussians

Generative process:
  z ~ ∑_k π_k N(μ_k, σ_k²)                 # Sample from mixture
```

**Connection to cognition**:
The brain maintains multiple hypotheses about the world state. DPGMM is AIME's analog of **probabilistic inference over a repertoire of world models**.

---

### 3. **VRNN Dynamics: Temporal Belief Propagation**

**Theoretical Role**: Recurrent state estimation and temporal prediction

**What it does**:
- Maintains a hidden state h_t that integrates:
  - Past beliefs (h_{t-1})
  - Current sensory evidence (context c_t from Perceiver)
  - Actions taken (a_t)
  - Current latent state (z_t)
- Predicts future states by rolling forward the LSTM

**Why LSTM with orthogonal initialization?**
- **Long-term dependencies**: Humanoid locomotion requires remembering balance over many timesteps
- **Stability**: Orthogonal weights prevent gradient explosion/vanishing
- **Gating**: LSTM gates act as precision estimators (forget gate ≈ precision over past, input gate ≈ precision over present)

**Information flow**:
```
h_{t-1}, c_{t-1} → LSTM → h_t
         ↑
    [z_t, c_t, a_t]
```

**Connection to cognition**:
Working memory in the brain maintains a compressed representation of task-relevant history. LSTM h_t is AIME's **working memory**.

---

### 4. **Attention Schema: Spatial Precision Estimation**

**Theoretical Role**: Estimating where to look (precision-weighted inference)

**What it does**:
- **Posterior (bottom-up)**: Slot attention discovers K salient regions from stimulus
- **Prior (top-down)**: Predicts attention based on beliefs (h_t) and goals
- **Fusion**: Combines bottom-up and top-down to compute final attention map A_t

**Why slot attention?**
- **Object-centric**: Each slot corresponds to an entity in the scene (limbs, obstacles, etc.)
- **Iterative refinement**: Attention evolves through competitive binding iterations
- **Compositional**: The scene is a composition of K attended objects

**Mathematics**:
```
Bottom-up (stimulus-driven):
  Features φ_att(x) → SlotAttention → slots S [B,K,D], attention A_post [B,K,H,W]

Top-down (expectation-driven):
  h_t → AttentionPrior → motion field Δ(x,y), saliency prior A_prior [B,H,W]

Fusion:
  A_t = Fusion(A_post, A_prior; h_t, c_t)  # Modulated combination

Diversity regularization:
  L_div = -∑_k H(A_k)  # Encourage each slot to be localized (low entropy)
  L_ortho = ∑_{k≠j} <A_k, A_j>  # Encourage slot separation (low overlap)
```

**Connection to cognition (Active Inference)**:
Attention in active inference is **precision weighting** - the brain allocates more processing to sensory inputs it expects to be reliable. Slot attention computes these precisions for different spatial regions.

---

### 5. **RGB Optimizer: Multi-Objective Harmony**

**Theoretical Role**: Resolving conflicts between learning objectives

**What it does**:
- Balances 4 task objectives:
  1. **ELBO (VAE loss)**: Reconstruction + KL divergences + entropy
  2. **Perceiver loss**: VQ commitment + token reconstruction
  3. **Predictive loss**: Attention dynamics + diversity regularization
  4. **Adversarial loss**: GAN discriminator + feature matching

- Uses **rotation-based gradient balancing** to minimize gradient conflicts while preserving task specificity

**Why multi-task learning?**
Each objective captures a different aspect of good world models:
- ELBO → statistical fidelity
- Perceiver → efficient representation
- Predictive → temporal coherence
- Adversarial → perceptual realism

**Mathematics (RGB Algorithm)**:
```
1. Normalize task gradients: ḡ_i = g_i / ||g_i||

2. Update consensus direction (EMA):
   d_t = μ·d_{t-1} + (1-μ)·mean(ḡ_i)

3. Compute orthogonal helpers:
   w_i = (d_t - <ḡ_i, d_t>ḡ_i) / ||d_t - <ḡ_i, d_t>ḡ_i||

4. Solve for rotation angles α_i minimizing:
   L(α) = (1/(T(T-1))) ∑_{i<j} (1 - r_i·r_j) + λ(1/4T) ∑_i ||r_i - ḡ_i||²
   where r_i = cos(α_i)ḡ_i + sin(α_i)w_i

5. Aggregate: g = ∑_i ||g_i||·r_i
```

**Connection to cognition**:
The brain simultaneously optimizes for perception, action, planning, and memory - multiple objectives with inherent conflicts. RGB is AIME's mechanism for **cognitive equilibrium**.

---

## The Five Pillars of AIME

| Pillar | Component | Theoretical Role | Key Innovation |
|--------|-----------|------------------|----------------|
| **Perception** | Perceiver IO | Sensory compression | 3D spatiotemporal VQ-VAE with multi-head quantization |
| **Representation** | DPGMM Prior | Adaptive beliefs | Stick-breaking prior conditioned on context |
| **Dynamics** | VRNN + LSTM | Temporal prediction | Orthogonal LSTM with multi-scale context integration |
| **Attention** | Attention Schema | Precision estimation | Slot attention with top-down/bottom-up fusion |
| **Optimization** | RGB | Multi-objective harmony | Rotation-based gradient balancing |

---

## Mathematical Framework

### Complete Generative Model

**Joint distribution**:
```
p(x_{1:T}, z_{1:T}, A_{1:T}, c_{1:T}, π, α | a_{1:T}) =
  p(α) ∏_t [
    p(x_t | z_t, A_t) ·              # Decoder (likelihood)
    p(z_t | h_t, c_t, π_t) ·         # DPGMM prior
    p(A_t | h_t, φ(x_t)) ·           # Attention prior/posterior
    p(c_t | x_{1:t}) ·               # Perceiver context
    p(π_t | h_t, α) ·                # Stick-breaking weights
    p(h_t | h_{t-1}, z_t, c_t, a_t)  # LSTM dynamics
  ]
```

**Variational posterior (inference network)**:
```
q(z_{1:T}, A_{1:T}, π, α | x_{1:T}, a_{1:T}) =
  q(α | h_{1:T}) · ∏_t [
    q(z_t | x_t, c_t) ·              # VAE encoder
    q(A_t | x_t, h_t) ·              # Attention posterior
    q(π_t | h_t) ·                   # Stick-breaking posterior
  ]
```

### ELBO Decomposition

**Evidence Lower Bound**:
```
log p(x_{1:T} | a_{1:T}) ≥ ELBO =
  𝔼_q[log p(x_{1:T} | z_{1:T}, A_{1:T})]           # Reconstruction
  - KL[q(z_{1:T} | x_{1:T}) || p(z_{1:T} | h_{1:T}, c_{1:T}, π)]  # Latent KL
  - KL[q(A_{1:T} | x_{1:T}) || p(A_{1:T} | h_{1:T})]              # Attention KL
  - KL[q(π | h) || p(π | α)]                                        # Mixture KL
  - KL[q(α | h) || p(α)]                                            # Concentration KL
  + H[q(π)]                                                          # Entropy (exploration)
```

**Actual loss optimized** (minimizing negative ELBO + auxiliary objectives):
```
L_total = L_ELBO + L_perceiver + L_predictive + L_adversarial

where:
  L_ELBO = λ_recon·L_recon + β·KL_z + β·KL_hierarchical + β·KL_attention - ω·H(π)
  L_perceiver = L_vq_commitment + L_perceiver_recon
  L_predictive = λ_dyn·L_attention_dynamics + L_diversity
  L_adversarial = L_GAN + λ_fm·L_feature_match
```

These are balanced via RGB gradient harmonization.

---

## Design Principles

### 1. **Separation of Concerns**

Each module has a single, well-defined purpose:
- **Perceiver**: Compress sensory data
- **Encoder**: Infer latents from observations
- **Prior**: Generate beliefs from context
- **Decoder**: Reconstruct observations from latents + attention
- **LSTM**: Maintain temporal state
- **Attention**: Estimate precision weights
- **RGB**: Balance objectives

This modularity enables:
- Independent debugging
- Ablation studies (turn off components)
- Future extensions (swap out modules)

### 2. **Hierarchical Abstraction**

Information flows through abstraction levels:
```
Pixels [H×W×3]
  ↓ Perceiver Encoder
Tokens [H_t×W_t discrete]
  ↓ Perceiver Context
Context [C_dim continuous]
  ↓ VAE Encoder
Latents [Z_dim continuous]
  ↓ DPGMM Prior
Mixture Beliefs [K components]
  ↓ Attention Schema
Spatial Precision [H_att×W_att map]
```

Each level is appropriate for different cognitive tasks.

### 3. **Probabilistic Everything**

All representations include uncertainty:
- **Perceiver tokens**: Softmax probabilities over codebook
- **Latent z**: Gaussian q(z|x) with mean and variance
- **DPGMM components**: Mixture weights π with stick-breaking uncertainty
- **Attention maps**: Soft spatial distributions summing to 1

This enables:
- Calibrated predictions
- Exploration through sampling
- Uncertainty-aware planning

### 4. **Context-Dependent Priors**

The prior p(z) is not static - it adapts based on context:
```
p(z) = p(z | h_t, c_t, a_t)
```

Where:
- h_t = temporal context (what happened before)
- c_t = perceptual context (what I see now)
- a_t = action context (what I'm doing)

This implements **empirical Bayes** - the prior itself is learned from data.

### 5. **End-to-End Differentiability**

Despite discrete tokens and mixture selection, the entire model is differentiable:
- VQ uses straight-through estimators
- Kumaraswamy distribution is reparameterizable
- Attention maps are soft (differentiable)
- Gumbel-softmax for sampling when needed

This allows joint optimization of all components.

---

## Connection to Active Inference

### The Free Energy Principle

Active inference states that agents minimize **variational free energy**:
```
F = 𝔼_q[log q(z|x) - log p(x,z)] = -ELBO
```

AIME directly minimizes this via VAE training!

### Perception as Inference

In active inference, perception is inverting a generative model:
```
Observations x → Infer hidden states z
```

AIME encoder q(z|x,c) implements this inversion.

### Attention as Precision

Active inference posits that attention modulates precision (inverse variance):
```
p(x | z, precision) where precision = attention
```

AIME attention schema computes spatial precision weights for reconstruction:
```
p(x_t | z_t, A_t) where A_t = attention map
```

### Action as Inference (Future Extension)

Active inference treats action selection as inference over policies. While AIME currently trains with offline data, the natural extension is:

```
a_t = argmax_a 𝔼[future reward | π_policy, current beliefs h_t]
```

This would close the loop, making AIME a full active inference agent.

### The Attention Schema as a "Where" Model

Attention Schema Theory (Graziano) proposes the brain builds a model of its own attention process. AIME's AttentionPrior is exactly this - a predictive model of where attention *should* be, which is compared against bottom-up AttentionPosterior.

---

## Future Vision

### Short-term Extensions

1. **Action-Conditional World Model**
   - Currently: Actions are inputs to LSTM
   - Goal: Make actions influence DPGMM prior directly: p(z|h,c,**a**)
   - Benefit: Clearer counterfactual reasoning ("what if I had turned left?")

2. **Hierarchical Temporal Scales**
   - Currently: Single LSTM timescale
   - Goal: Multi-scale LSTMs (fast, medium, slow) à la Hierarchical Perceiver
   - Benefit: Capture both reactive (balance) and strategic (navigation) dynamics

3. **Object-Centric Slots**
   - Currently: Attention slots are generic
   - Goal: Each slot has object properties (position, velocity, type)
   - Benefit: Structured representations for reasoning and planning

### Long-term Vision

**AIME as a Foundation Model for Embodied AI**

Imagine AIME pre-trained on:
- Millions of hours of robot interaction data
- Diverse embodiments (quadrupeds, bipeds, manipulators)
- Multi-modal sensors (vision, proprioception, touch, audio)

Then fine-tuned for specific tasks:
- Locomotion in new terrains
- Manipulation of novel objects
- Social interaction with humans

The DPGMM prior would capture a **universal library of sensorimotor primitives**, adaptively mixing them for each new context.

**Towards Cognitive Architecture**

Future AIME could incorporate:
- **Episodic memory**: Store and retrieve past experiences (beyond LSTM)
- **Compositional reasoning**: Combine learned primitives in new ways
- **Language grounding**: Map instructions to latent goals in z-space
- **Curiosity-driven exploration**: Maximize H(π) to discover new mixture components

---

## Summary: Why AIME?

**AIME is not just a world model - it's a philosophy of intelligence.**

It embodies the idea that:
1. **The world is compositional** (VQ tokens, DPGMM components, attention slots)
2. **Representations should adapt** (infinite mixture, learned codebook)
3. **Prediction requires context** (context-dependent priors)
4. **Attention is fundamental** (precision-weighted inference)
5. **Objectives naturally conflict** (multi-task harmony via RGB)

By grounding these principles in a rigorous probabilistic framework (ELBO + active inference), AIME provides a **unified architecture for perception, prediction, and attention** in embodied agents.

---

**References & Inspiration**

- **Free Energy Principle**: Friston, K. (2010). The free-energy principle: a unified brain theory?
- **Active Inference**: Friston et al. (2017). Active inference: a process theory.
- **Dirichlet Process**: Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems.
- **Stick-Breaking**: Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
- **Slot Attention**: Locatello et al. (2020). Object-centric learning with slot attention.
- **Perceiver IO**: Jaegle et al. (2021). Perceiver IO: A general architecture for structured inputs & outputs.
- **VQ-VAE**: van den Oord et al. (2017). Neural discrete representation learning.
- **Attention Schema Theory**: Graziano, M. S. A. (2013). Consciousness and the social brain.
- **RGB**: (ICLR 2026 submission) Rotation-based gradient balancing for multi-task learning.

---

*"An infinite mixture of adaptive beliefs, harmoniously balanced, attending to what matters."* — The AIME Philosophy
