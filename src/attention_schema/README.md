# Attention Schema Module

**PILLAR 4: ATTENTION - Precision-Weighted Inference**

---

## Overview

The Attention Schema module implements a sophisticated dual-stream attention mechanism that combines **bottom-up** (stimulus-driven) and **top-down** (predictive) attention processing. This architecture enables the AIME system to dynamically allocate computational resources to relevant parts of the visual field while maintaining predictive models of where attention should move next.

### Core Principle

Following **Attention Schema Theory** (Graziano & Kastner, 2011), attention itself is modeled as an internal state that can be:
1. **Inferred** from sensory observations (bottom-up posterior)
2. **Predicted** based on current beliefs and dynamics (top-down prior)
3. **Tracked** over time to learn spatiotemporal attention patterns

This dual representation allows the model to balance reactive (stimulus-driven) and proactive (goal-driven) attention mechanisms.

---

## Theory

### Attention Schema Theory (AST)

**Graziano & Kastner (2011)** propose that the brain constructs an internal model of attention itself - a "schema" that represents:
- Where attention is currently focused
- How attention moves over time
- What features are being attended to

This meta-representational approach suggests that attention is not just a passive filter, but an actively modeled cognitive state.

### Connection to Active Inference

In the Active Inference framework (Friston et al., 2017), attention is interpreted as **precision weighting** - the inverse variance of prediction errors. The attention schema implements this through:

```
Precision-weighted prediction error = attention_map ⊙ (observation - prediction)
```

Where:
- **High attention** → High precision → Prediction errors weighted strongly
- **Low attention** → Low precision → Prediction errors downweighted

This enables the model to:
1. Focus processing on informative regions (reduce computational cost)
2. Weight evidence appropriately during inference (improve accuracy)
3. Predict where to attend next (enable proactive control)

### Information Flow

```
                    Attention Schema Architecture
                              |
              ┌───────────────┴───────────────┐
              ↓                               ↓
    POSTERIOR (Bottom-up)            PRIOR (Top-down)
    Stimulus-driven                  Predictive
              |                               |
    ┌─────────┴─────────┐         ┌──────────┴──────────┐
    ↓                   ↓         ↓                     ↓
  FPN              SlotAttention  ConvGRU          Self-Attention
(Multi-scale)      (Object-centric) (Spatial dynamics) (Global context)
    |                   |           |                   |
    └─────────┬─────────┘           └─────────┬─────────┘
              ↓                               ↓
    attention_map_t                  predicted_attention_{t+1}
       [B, H, W]                           [B, H, W]
```

---

## Architecture

The Attention Schema module consists of **five key components**:

### 1. SlotAttention (`slot_attention.py`)

**Purpose**: Object-centric representation learning through iterative attention routing.

**Reference**: Locatello et al. (2020) - "Object-centric learning with slot attention"

**Mechanism**:
- Iterative competitive attention where K slots compete to explain input features
- Attention weights sum to 1 over slots (not over inputs) → "explaining away"
- Each slot is updated via GRU + MLP based on attended features

**Mathematical Formulation**:
```
For iteration i:
  q_k = Linear_q(slot_k)                    # Query from slot
  k_n, v_n = Linear_k(x_n), Linear_v(x_n)  # Keys, values from inputs

  attn_{k,n} = softmax_k(q_k · k_n / √d)   # Attention (softmax over K)

  update_k = Σ_n attn_{k,n} · v_n          # Weighted sum of values

  slot_k = GRU(update_k, slot_k)           # Update slot
  slot_k = slot_k + MLP(slot_k)            # Residual MLP
```

**Tensor Flow**:
```python
Input:  x [B, N, D_in]           # N = H*W spatial positions
Init:   slots ~ N(μ, σ)          # [B, K, D_slot] - Learnable Gaussian init
Iterate 3 times:
   q = Linear_q(slots)           # [B, K, D]
   k, v = Linear_k(x), Linear_v(x)  # [B, N, D]
   attn = softmax_k(q @ k^T)     # [B, K, N] - Responsibilities
   updates = attn @ v            # [B, K, D]
   slots = GRU(updates, slots)   # [B, K, D]
   slots = slots + MLP(slots)    # [B, K, D]
Output: slots [B, K, D_slot]     # K object-centric representations
        attn [B, K, N]           # Attention weights per slot
```

**Key Properties**:
- **Permutation invariant**: Slot order doesn't matter
- **Competitive**: Slots compete for explaining inputs
- **Iterative refinement**: Multiple rounds improve separation
- **Differentiable**: End-to-end trainable

---

### 2. AttentionPosterior (`attention_posterior.py`)

**Purpose**: Bottom-up multi-object attention mechanism that processes visual observations to extract spatial attention maps.

**Architecture Components**:

1. **Feature Pyramid Network (FPN)**: Multi-scale feature extraction
   ```
   84x84 → [Conv+GN+SiLU] → 42x42 (32 channels)
          → [Conv+GN+SiLU] → 21x21 (48 channels)
          → [Conv+GN+SiLU] → 21x21 (64 channels)
          → [1x1 Conv] → 21x21 (64 channels fused)
   ```

2. **SlotAttention**: GroupViT-style semantic grouping with K=4 slots

3. **Top-down Modulation**: Context-aware slot conditioning
   ```python
   modulation = MLP(concat(h_t, c_t))  # [B, hidden_dim + context_dim]
                                        → [B, K * d] → reshape → [B, K, d]
   ```

4. **Multi-Head Fusion**: Combine per-slot attention maps
   - **Weighted**: Learnable per-slot weights (softmax normalized)
   - **Max**: Max-pooling over slots (hard selection)
   - **Gated**: Sigmoid gates per slot (soft selection)

5. **Spatial Regularizer**: Combine K slot centers into unified coordinates
   ```python
   coords = MLP(concat(center_1, ..., center_K)) → [B, 2]  # Range: [-1, 1]
   ```

**Tensor Flow**:
```python
Input:  observation [B, 3, 84, 84]
        hidden_state [B, hidden_dim]   # From VRNN
        context [B, context_dim]        # From Perceiver

# 1. Feature Pyramid
features = FPN(observation)             # [B, 64, 21, 21]

# 2. Flatten + Positional Encoding
feat_seq = features.flatten(2).transpose(1,2)  # [B, N, 64] where N=441
pos_2d = row_embed ⊕ col_embed          # [1, N, 64]
feat_seq = feat_seq + pos_2d            # [B, N, 64]

# 3. Top-down modulation (seed slots)
modulation = slot_modulator(concat(h, c))  # [B, K*64] → [B, K, 64]

# 4. Slot Attention
slots, attn = SlotAttention(feat_seq, seed_slots=modulation)
# slots: [B, 4, 64]
# attn: [B, 4, 441] - Responsibilities

# 5. Reshape to spatial
slot_maps = attn.view(B, 4, 21, 21)     # [B, K, H, W]

# 6. Fusion (weighted example)
fusion_weights = fusion_head(slots.flatten(1))  # [B, K]
fused_map = (slot_maps * fusion_weights[..., None, None]).sum(1)  # [B, 21, 21]

# 7. Compute slot centers
attention_probs = softmax(fused_map.flatten(1)).view(B, 21, 21)  # [B, H, W]
slot_centers = weighted_centroid(slot_maps)  # [B, K, 2]
coords = spatial_regularizer(slot_centers.flatten(1))  # [B, 2]

Output: attention_probs [B, 21, 21]     # Unified attention map
        coords [B, 2]                   # Attention center in [-1,1]

Side Attributes:
  .slot_attention_maps [B, K, 21, 21]   # Per-slot attention
  .slot_centers [B, K, 2]               # Per-slot centers
  .slot_features [B, K, 64]             # Per-slot representations
  .bottleneck_features [B, hidden_dim//2]  # Compressed features
  .diversity_loss: scalar               # Encourages slot separation
```

**Design Highlights**:
- **Linear complexity**: O(N·K) instead of O(N²) for full attention
- **Object-centric**: Explicit multi-object decomposition via slots
- **Context-aware**: Hidden state h and context c guide slot initialization
- **Spatial regularization**: Gaussian noise injection + regularization network

---

### 3. AttentionPrior (`attention_prior.py`)

**Purpose**: Top-down prediction of attention dynamics using spatial recurrence and self-attention.

**Architecture Components**:

1. **ConvGRU**: Spatial recurrent dynamics
   ```python
   h_t = ConvGRU(attention_{t-1})  # [B, 1, 21, 21] → [B, 32, 21, 21]
   ```

2. **Motion Kernels**: Learnable filters for motion detection
   ```python
   motion_features = [Conv2d(attention_{t-1}, kernel_i) for i in 1..K]
   # K=8 learnable 5x5 kernels
   ```

3. **Efficient Self-Attention**: Global attention over spatial positions
   ```python
   Q, K, V = Linear(spatial_seq)
   attn = MultiheadAttention(Q, K, V, num_heads=4)
   ```

4. **Cross-Attention with Context**: Incorporate h_t and z_t
   ```python
   context_features = MLP(concat(h_t, z_t))  # [B, feature_dim]
   attended = CrossAttention(spatial_seq, context_features)
   ```

5. **Movement Predictor**: Predict attention shift
   ```python
   movement = Conv2d(attention_{t-1}, out_channels=2)  # [B, 2, 21, 21]
   dx, dy = movement[:, 0], movement[:, 1]
   ```

**Tensor Flow**:
```python
Input:  prev_attention [B, 21, 21]      # Attention at t-1
        hidden_state [B, 256]           # From VRNN
        latent_state [B, 32]            # From VAE posterior

# 1. Spatial dynamics
spatial_feat = ConvGRU(prev_attention.unsqueeze(1))  # [B, 32, 21, 21]

# 2. Motion features
motion_feat = concat([Conv2d(prev_attention, k_i) for i in range(8)])
# [B, 8, 21, 21]

# 3. Project to attention dimension
spatial_proj = Conv2d(spatial_feat)     # [B, 64, 21, 21]
motion_proj = Conv2d(motion_feat)       # [B, 32, 21, 21]
combined = spatial_proj + pad(motion_proj)  # [B, 64, 21, 21]

# 4. Convert to sequence + positional encoding
seq = combined.flatten(2).transpose(1,2)  # [B, 441, 64]
seq = seq + pos_embed                   # [B, 441, 64]

# 5. Context features
context = MLP(concat(hidden_state, latent_state))  # [B, 64]

# 6. Self-attention (GLOBAL)
seq_norm = LayerNorm(seq)
seq = seq + MultiheadAttention(seq_norm, seq_norm, seq_norm)  # [B, 441, 64]

# 7. Cross-attention with context
seq_norm = LayerNorm(seq)
context_expanded = context.unsqueeze(1).expand(-1, 441, -1)  # [B, 441, 64]
seq = seq + CrossAttention(seq_norm, context_expanded, context_expanded)

# 8. FFN
seq = seq + FFN(LayerNorm(seq))         # [B, 441, 64]

# 9. Output projection
attention_logits = Linear(seq)          # [B, 441, 1] → [B, 441]
attention_probs = softmax(attention_logits).view(B, 21, 21)

# 10. Movement prediction
movement = Conv2d(prev_attention.unsqueeze(1), out=2)  # [B, 2, 21, 21]
dx, dy = movement[:, 0], movement[:, 1]

Output: attention_probs [B, 21, 21]     # Predicted attention at t
        info['predicted_movement']: (dx, dy)  # Predicted shift
        info['spatial_features']: [B, 32, 21, 21]
        info['motion_features']: [B, 8, 21, 21]
```

**Key Features**:
- **Gradient checkpointing**: Memory-efficient training (enabled by default)
- **Relative position bias**: Learnable 2D position relationships
- **Temporal coherence**: ConvGRU maintains spatial state across time
- **Motion-aware**: Explicit motion kernel responses

---

### 4. AttentionSchema (`attention_schema.py`)

**Purpose**: Integration wrapper that combines AttentionPosterior and AttentionPrior.

**Role**: Provides unified interface for VRNN integration and attention dynamics tracking.

**Key Methods**:

```python
class AttentionSchema:
    def __init__(self, image_size=84, attention_resolution=21, ...):
        self.posterior_net = AttentionPosterior(...)
        self.prior_net = AttentionPrior(...)

    def compute_attention_dynamics_loss(
        self,
        attention_sequence: List[Tensor],     # [B, H, W] over time
        predicted_movements: List[Tensor]     # [(dx, dy)] over time
    ) -> Tensor:
        """
        Measure consistency between predicted and actual attention movement.

        Computes center-of-mass for each attention map and compares
        actual movement (COM_{t+1} - COM_t) with predicted (dx, dy).

        Returns: smooth_l1_loss(predicted, actual)
        """
```

**Tensor Flow (Full Pipeline)**:
```python
# At timestep t:

# 1. Bottom-up attention (posterior)
attn_posterior, coords = attention_schema.posterior_net(
    observation=obs_t,     # [B, 3, 84, 84]
    hidden_state=h_t,      # [B, 256]
    context=c_t            # [B, 128]
)
# attn_posterior: [B, 21, 21]
# coords: [B, 2]

# 2. Top-down attention (prior) - predict t+1
attn_prior, info = attention_schema.prior_net(
    prev_attention=attn_posterior,  # [B, 21, 21]
    hidden_state=h_t,               # [B, 256]
    latent_state=z_t                # [B, 32]
)
# attn_prior: [B, 21, 21] - prediction for t+1
# info['predicted_movement']: (dx, dy)

# 3. At end of sequence: compute dynamics loss
dynamics_loss = attention_schema.compute_attention_dynamics_loss(
    attention_sequence=[attn_0, attn_1, ..., attn_T],
    predicted_movements=[(dx_0, dy_0), ..., (dx_{T-1}, dy_{T-1})]
)
```

---

### 5. ConvGRUCell (`spatial_utils.py`)

**Purpose**: Convolutional GRU cell for spatial-temporal dynamics modeling.

**Architecture**: Standard GRU extended to spatial dimensions using convolutions.

**Formulation**:
```
Reset gate:     r_t = σ(Conv(concat(x_t, h_{t-1})))
Update gate:    u_t = σ(Conv(concat(x_t, h_{t-1})))
Candidate:      c_t = tanh(Conv(concat(x_t, r_t ⊙ h_{t-1})))
Hidden state:   h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t
```

**Tensor Flow**:
```python
Input:  x [B, in_channels, H, W]        # Current input
        h [B, hidden_channels, H, W]    # Previous hidden (or zeros)

# Gate computation
gates = Conv2d(concat(x, h), out=2*hidden_channels)  # [B, 2*C, H, W]
gates = GroupNorm(gates)
r, u = split(sigmoid(gates))            # Each [B, C, H, W]

# Candidate
gated_h = r ⊙ h
candidate = Conv2d(concat(x, gated_h), out=hidden_channels)
candidate = GroupNorm(candidate)
c = tanh(candidate)                     # [B, C, H, W]

# Update
h_next = u ⊙ h + (1 - u) ⊙ c           # [B, C, H, W]

Output: h_next [B, hidden_channels, H, W]
```

**Usage**:
```python
conv_gru = ConvGRUCell(
    input_size=1,        # Attention map is 1 channel
    hidden_size=32,
    kernel_size=5,
    cuda_flag=True
)

h = None  # Initialize to zeros
for t in range(T):
    h = conv_gru(attention_maps[t].unsqueeze(1), h)
    # h: [B, 32, 21, 21]
```

---

## Tensor Shape Reference

### Complete Shape Flow

| Component | Input | Output | Description |
|-----------|-------|--------|-------------|
| **SlotAttention** | | | |
| `forward()` | x: `[B, N, D_in]` | slots: `[B, K, D_slot]` | Iterative routing to K slots |
|  | seed_slots: `[B, K, D_slot]` (optional) | attn: `[B, K, N]` | Attention responsibilities |
| **AttentionPosterior** | | | |
| `forward()` | observation: `[B, 3, 84, 84]` | attention_probs: `[B, 21, 21]` | Bottom-up attention map |
|  | hidden_state: `[B, 256]` | coords: `[B, 2]` | Attention center |
|  | context: `[B, 128]` | | |
| Side attributes | | slot_attention_maps: `[B, K, 21, 21]` | Per-slot attention |
|  | | slot_centers: `[B, K, 2]` | Per-slot centers |
|  | | slot_features: `[B, K, 64]` | Slot representations |
|  | | bottleneck_features: `[B, 128]` | Compressed features |
|  | | diversity_loss: `scalar` | Slot separation loss |
| **AttentionPrior** | | | |
| `forward()` | prev_attention: `[B, 21, 21]` | attention_probs: `[B, 21, 21]` | Predicted attention |
|  | hidden_state: `[B, 256]` | info: `Dict` | Auxiliary outputs |
|  | latent_state: `[B, 32]` | | |
| info dict | | `'spatial_features'`: `[B, 32, 21, 21]` | ConvGRU features |
|  | | `'motion_features'`: `[B, 8, 21, 21]` | Motion kernel responses |
|  | | `'predicted_movement'`: `(dx, dy)` | Movement prediction |
|  | | `'attention_logits'`: `[B, 21, 21]` | Pre-softmax logits |
| **AttentionSchema** | | | |
| `posterior_net()` | Same as AttentionPosterior | Same as AttentionPosterior | Bottom-up attention |
| `prior_net()` | Same as AttentionPrior | Same as AttentionPrior | Top-down attention |
| `compute_attention_dynamics_loss()` | attention_seq: `List[[B, H, W]]` | loss: `scalar` | Dynamics consistency |
|  | predicted_mov: `List[(dx, dy)]` | | |
| **ConvGRUCell** | | | |
| `forward()` | input: `[B, in_channels, H, W]` | hidden: `[B, hidden_channels, H, W]` | Spatial recurrence |
|  | hidden: `[B, hidden_channels, H, W]` (optional) | | |

**Legend**: B=batch, K=num_slots, N=spatial_positions (H*W), H=height, W=width

---

## Usage Examples

### 1. Standalone AttentionPosterior

```python
from attention_schema import AttentionPosterior

# Initialize
posterior = AttentionPosterior(
    image_size=84,
    attention_resolution=21,
    hidden_dim=256,
    context_dim=128,
    num_semantic_slots=4,
    attention_fusion_mode="weighted",
    enforce_diversity=True
)

# Forward pass
obs = torch.randn(2, 3, 84, 84)      # Batch of 2 images
h = torch.randn(2, 256)              # Hidden state from VRNN
c = torch.randn(2, 128)              # Context from Perceiver

attn_probs, coords = posterior(obs, h, c)

print(f"Attention map: {attn_probs.shape}")  # [2, 21, 21]
print(f"Attention center: {coords.shape}")   # [2, 2]

# Access per-slot information
print(f"Slot attention maps: {posterior.slot_attention_maps.shape}")  # [2, 4, 21, 21]
print(f"Slot centers: {posterior.slot_centers.shape}")                # [2, 4, 2]
print(f"Diversity loss: {posterior.diversity_loss}")                  # scalar
```

### 2. Standalone AttentionPrior

```python
from attention_schema import AttentionPrior

# Initialize
prior = AttentionPrior(
    attention_resolution=21,
    hidden_dim=256,
    latent_dim=32,
    motion_kernels=8,
    use_checkpoint=True  # Enable gradient checkpointing
)

# Forward pass
prev_attn = torch.randn(2, 21, 21)   # Attention at t-1
h = torch.randn(2, 256)              # Hidden state
z = torch.randn(2, 32)               # Latent state

pred_attn, info = prior(prev_attn, h, z)

print(f"Predicted attention: {pred_attn.shape}")  # [2, 21, 21]
print(f"Predicted dx: {info['predicted_movement'][0].shape}")  # [2, 21, 21]
print(f"Predicted dy: {info['predicted_movement'][1].shape}")  # [2, 21, 21]
```

### 3. Full AttentionSchema Pipeline

```python
from attention_schema import AttentionSchema

# Initialize
attention_schema = AttentionSchema(
    image_size=84,
    attention_resolution=21,
    hidden_dim=256,
    latent_dim=32,
    context_dim=128
)

# Simulate sequence
attention_sequence = []
predicted_movements = []

for t in range(10):
    obs_t = torch.randn(2, 3, 84, 84)
    h_t = torch.randn(2, 256)
    c_t = torch.randn(2, 128)
    z_t = torch.randn(2, 32)

    # Bottom-up attention
    attn_posterior, coords = attention_schema.posterior_net(obs_t, h_t, c_t)
    attention_sequence.append(attn_posterior)

    # Top-down prediction (if not first frame)
    if t > 0:
        attn_prior, info = attention_schema.prior_net(
            attention_sequence[t-1], h_t, z_t
        )
        predicted_movements.append(info['predicted_movement'])

# Compute dynamics loss
dynamics_loss = attention_schema.compute_attention_dynamics_loss(
    attention_sequence,
    predicted_movements
)

print(f"Dynamics loss: {dynamics_loss.item()}")
```

### 4. Visualization

```python
# Visualize multi-object attention
vis_outputs = attention_schema.posterior_net.visualize_multi_object_attention(
    observation=obs,                             # [B, 3, 84, 84]
    slot_attention_maps=posterior.slot_attention_maps,  # [B, K, 21, 21]
    slot_centers=posterior.slot_centers,         # [B, K, 2]
    group_assignments=posterior.group_assignments,  # [B, 21, 21, K]
    return_mode="all"
)

# vis_outputs contains:
# - 'slot_overlays': [B*K, 3, 84, 84] - Per-slot colored overlays
# - 'semantic_segmentation': [B, 3, 84, 84] - Colored by dominant slot
# - 'combined_with_contours': [B, 3, 84, 84] - Edges + centers marked
```

---

## Integration with VRNN

The attention schema integrates with the VRNN world model at multiple levels:

### 1. Initialization (in `dpgmm_stickbreaking_prior_vrnn.py`)

```python
class VRNN(nn.Module):
    def __init__(self, ...):
        # ... other components ...

        # Initialize attention schema
        self.attention_prior_posterior = AttentionSchema(
            image_size=self.image_size,           # 84
            attention_resolution=21,              # 84/4
            hidden_dim=self.hidden_dim,           # 256
            latent_dim=self.latent_dim,           # 32
            context_dim=self.context_dim,         # 128
            attention_dim=self.attention_dim,     # 64
            input_channels=self.input_channels    # 3
        )

        # Feature extractor for attention
        self.phi_attention = LinearResidual(
            self.hidden_dim//2 + 2,  # bottleneck_features + coords
            2*self.hidden_dim
        )
```

### 2. Forward Pass Integration

```python
def forward_sequence(self, observations, actions):
    """
    observations: [B, T, 3, 84, 84]
    actions: [B, T, action_dim]
    """
    B, T = observations.shape[:2]

    # Extract context from Perceiver (once)
    c = self.perceiver_model.extract_context(observations)  # [B, T, 128]

    # Initialize hidden states
    h_t = self.init_hidden(B)                    # [B, 256]
    attention_t = torch.zeros(B, 21, 21)         # [B, 21, 21]

    attention_sequence = []
    predicted_movements = []

    for t in range(T):
        obs_t = observations[:, t]               # [B, 3, 84, 84]
        c_t = c[:, t]                            # [B, 128]
        a_t = actions[:, t]                      # [B, action_dim]

        # === ATTENTION POSTERIOR (Bottom-up) ===
        attention_posterior, coords = self.attention_prior_posterior.posterior_net(
            observation=obs_t,
            hidden_state=h_t,
            context=c_t
        )
        # attention_posterior: [B, 21, 21]
        # coords: [B, 2]

        attention_sequence.append(attention_posterior)

        # === ATTENTION PRIOR (Top-down) ===
        if t > 0:
            attention_prior, prior_info = self.attention_prior_posterior.prior_net(
                prev_attention=attention_t,
                hidden_state=h_t,
                latent_state=z_t  # From previous step
            )
            predicted_movements.append(prior_info['predicted_movement'])

        # === USE ATTENTION FOR VAE ===
        # 1. Extract attention features
        bottleneck = self.attention_prior_posterior.posterior_net.bottleneck_features
        # bottleneck: [B, 128]

        attention_features = self.phi_attention(
            torch.cat([bottleneck, coords], dim=-1)
        )  # [B, 512]

        # 2. Use attention in VAE encoder (precision weighting)
        z_params = self.encoder(
            obs_t,
            c_t,
            attention_map=attention_posterior  # Weight encoder by attention
        )

        # 3. Sample latent
        z_t = reparameterize(z_params['mean'], z_params['logvar'])

        # 4. Use attention in VAE decoder (reconstruction guidance)
        reconstruction = self.decoder(
            z_t,
            c_t,
            attention_map=attention_posterior  # Focus reconstruction on attended regions
        )

        # === UPDATE DYNAMICS ===
        # Incorporate attention features into LSTM input
        lstm_input = torch.cat([z_t, c_t, a_t, attention_features], dim=-1)
        h_t, cell_t = self.rnn(lstm_input, (h_t, cell_t))

        # Update attention for next step
        attention_t = attention_posterior

    # === COMPUTE ATTENTION DYNAMICS LOSS ===
    dynamics_loss = self.attention_prior_posterior.compute_attention_dynamics_loss(
        attention_sequence,
        predicted_movements
    )

    # === COMPUTE DIVERSITY LOSS ===
    diversity_loss = sum([
        self.attention_prior_posterior.posterior_net.diversity_loss
        for _ in range(T)
    ]) / T

    return {
        'reconstruction': reconstructions,
        'attention_maps': attention_sequence,
        'dynamics_loss': dynamics_loss,
        'diversity_loss': diversity_loss,
        ...
    }
```

### 3. Loss Function

```python
def compute_loss(self, outputs, observations):
    # Standard ELBO terms
    reconstruction_loss = ...
    kl_divergence = ...

    # Attention-specific losses
    attention_dynamics_loss = outputs['dynamics_loss']
    diversity_loss = outputs['diversity_loss']

    # Total loss
    total_loss = (
        reconstruction_loss +
        β * kl_divergence +
        λ_dynamics * attention_dynamics_loss +
        λ_diversity * diversity_loss
    )

    return total_loss
```

### 4. Benefits of Integration

1. **Precision weighting**: Attention guides VAE encoder/decoder to focus on informative regions
2. **Computational efficiency**: Sparse attention reduces effective input size
3. **Object-centric representations**: Slot attention provides multi-object decomposition
4. **Temporal coherence**: Attention dynamics loss enforces smooth attention movement
5. **Predictive attention**: Prior network enables proactive attention control

---

## Design Decisions

### Why Slot Attention for Object Decomposition?

**Advantages**:
- **Unsupervised**: No object annotations required
- **Permutation invariant**: Robust to slot ordering
- **Competitive binding**: Slots naturally segment scene into parts
- **Efficient**: O(K·N) complexity instead of O(N²)

**Alternatives considered**:
- **MONet/IODINE**: More complex, require spatial mixture models
- **SPACE**: Requires learned proposal network
- **Transformer**: O(N²) complexity too expensive for video

### Why Dual-Stream (Posterior + Prior)?

**Inspiration**: Predictive coding and active inference frameworks suggest attention should be both:
1. **Reactive**: Driven by bottom-up sensory signals (posterior)
2. **Proactive**: Guided by top-down predictions (prior)

**Benefits**:
- **Robustness**: Posterior handles unexpected stimuli; prior handles predictable patterns
- **Efficiency**: Prior can pre-allocate attention before stimulus arrives
- **Learning**: Dynamics loss teaches attention movement patterns

**Alternative**: Single stream with attention loss
- **Drawback**: Cannot learn predictive attention patterns

### Why ConvGRU for Spatial Dynamics?

**Advantages**:
- **Spatial inductive bias**: Convolutions respect spatial locality
- **Parameter efficiency**: Shared weights across positions
- **Temporal coherence**: GRU tracks state over time

**Alternatives considered**:
- **ConvLSTM**: More complex (separate cell state), marginal benefits
- **3D Conv**: No explicit recurrence, harder to track long-term dynamics
- **Attention-only**: More expensive, less spatially biased

### Why Feature Pyramid Network?

**Multi-scale processing** captures:
- **Low-level**: Edges, textures (high resolution)
- **Mid-level**: Parts, shapes (medium resolution)
- **High-level**: Objects, context (low resolution)

Fusing all scales provides rich representation for slot attention.

---

## Hyperparameters

### AttentionPosterior

```python
image_size = 84                    # Input image size
attention_resolution = 21          # Attention map resolution (image_size/4)
hidden_dim = 256                   # Hidden state dimension
context_dim = 128                  # Context dimension from Perceiver
feature_channels = 64              # Internal feature dimension
num_semantic_slots = 4             # Number of slots (K)
num_heads = 4                      # Multi-head attention (not used currently)
attention_fusion_mode = "weighted" # How to combine slots: 'weighted'|'max'|'gated'
enforce_diversity = True           # Whether to compute diversity loss
dropout_p = 0.2                    # Dropout on slot features
coord_noise_std = 0.0              # Gaussian noise on coordinates (data augmentation)

# SlotAttention hyperparameters
slot_iters = 3                     # Number of iterative refinements
slot_mlp_hidden = 128              # MLP hidden dimension for slot update
```

### AttentionPrior

```python
attention_resolution = 21          # Must match posterior
hidden_dim = 256                   # Hidden state dimension
latent_dim = 32                    # Latent state dimension
motion_kernels = 8                 # Number of learnable motion filters
num_heads = 4                      # Self-attention heads
feature_dim = 64                   # Internal feature dimension
use_relative_position_bias = True  # Use 2D relative position bias
use_checkpoint = True              # Gradient checkpointing (memory efficient)
dropout = 0.1                      # Attention dropout

# ConvGRU hyperparameters
convgru_hidden_size = 32           # ConvGRU hidden channels
convgru_kernel_size = 5            # Spatial kernel size
```

### Loss Weights (in VRNN)

```python
λ_dynamics = 0.1                   # Weight for attention dynamics loss
λ_diversity = 0.05                 # Weight for slot diversity loss
```

---

## Loss Functions

### 1. Attention Dynamics Loss

**Purpose**: Encourage consistency between predicted and actual attention movement.

**Formulation**:
```python
def compute_attention_dynamics_loss(attention_sequence, predicted_movements):
    # Compute center-of-mass for each attention map
    centers = [center_of_mass(attn) for attn in attention_sequence]  # [T, B, 2]

    # Actual movement
    actual_delta = centers[1:] - centers[:-1]  # [T-1, B, 2]

    # Predicted movement (from AttentionPrior)
    pred_delta = predicted_movements  # [T-1, B, 2] or [T-1, B, 2, H, W]

    # If spatial field, average to get single vector
    if pred_delta.ndim == 5:
        pred_delta = pred_delta.mean(dim=(3, 4))  # [T-1, B, 2]

    # Smooth L1 loss
    loss = smooth_l1(pred_delta[..., 0], actual_delta[..., 0]) + \
           smooth_l1(pred_delta[..., 1], actual_delta[..., 1])

    return loss
```

**Interpretation**: This loss teaches the AttentionPrior to predict where attention will move next, enabling proactive attention control.

### 2. Diversity Loss

**Purpose**: Encourage slots to specialize on different objects/regions.

**Formulation**:
```python
def compute_diversity_loss(slot_attention_maps):
    # slot_attention_maps: [B, K, H, W]

    # Convert to probabilities
    p = slot_attention_maps.view(B, K, -1).softmax(dim=-1)  # [B, K, N]

    # L2 normalize
    p = F.normalize(p, p=2, dim=-1)  # [B, K, N]

    # Correlation matrix
    corr = p @ p.transpose(-2, -1)  # [B, K, K]

    # Penalize off-diagonal correlations
    I = torch.eye(K).unsqueeze(0)   # [1, K, K]
    loss = ((corr - I) ** 2).sum() / (B * K * (K - 1))

    return loss
```

**Interpretation**: Minimizing off-diagonal correlations encourages slots to attend to different regions, promoting object decomposition.

### 3. Integration with VRNN Loss

```python
total_loss = (
    reconstruction_loss +              # E_q[(x - μ_θ(z))²]
    β * kl_divergence +                # KL[q(z|x) || p(z|h)]
    λ_dynamics * dynamics_loss +       # Attention movement consistency
    λ_diversity * diversity_loss       # Slot separation
)
```

---

## References

### Theoretical Foundations

1. **Graziano, M. S., & Kastner, S. (2011)**
   "Human consciousness and its relationship to social neuroscience: A novel hypothesis"
   *Cognitive Neuroscience*, 2(2), 98-113.
   - Original Attention Schema Theory paper

2. **Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017)**
   "Active inference: A process theory"
   *Neural Computation*, 29(1), 1-49.
   - Attention as precision weighting in active inference

3. **Feldman, H., & Friston, K. (2010)**
   "Attention, uncertainty, and free-energy"
   *Frontiers in Human Neuroscience*, 4, 215.
   - Connection between attention and Bayesian inference

### Architectural References

4. **Locatello, F., Weissenborn, D., Unterthiner, T., et al. (2020)**
   "Object-centric learning with slot attention"
   *Advances in Neural Information Processing Systems*, 33, 11525-11538.
   - Slot Attention mechanism

5. **Rao, R. P., & Ballard, D. H. (1999)**
   "Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects"
   *Nature Neuroscience*, 2(1), 79-87.
   - Predictive coding theory (inspiration for attention prior)

6. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)**
   "Attention is all you need"
   *Advances in Neural Information Processing Systems*, 30.
   - Transformer self-attention mechanism

7. **Shi, X., Chen, Z., Wang, H., et al. (2015)**
   "Convolutional LSTM network: A machine learning approach for precipitation nowcasting"
   *Advances in Neural Information Processing Systems*, 28.
   - ConvLSTM (inspiration for ConvGRU)

8. **Lin, T. Y., Dollár, P., Girshick, R., et al. (2017)**
   "Feature pyramid networks for object detection"
   *IEEE Conference on Computer Vision and Pattern Recognition*.
   - Feature Pyramid Network (FPN)

### Related Work

9. **Burgess, C. P., Matthey, L., Watters, N., et al. (2019)**
   "MONet: Unsupervised scene decomposition and representation"
   *arXiv preprint arXiv:1901.11390*.
   - Alternative object-centric approach

10. **Greff, K., Kaufman, R. L., Kabra, R., et al. (2019)**
    "Multi-Object Representation Learning with Iterative Variational Inference"
    *International Conference on Machine Learning*.
    - IODINE (another object-centric method)

---

## Future Extensions

### Potential Improvements

1. **Hierarchical Slot Attention**
   Multi-level slots for part-whole hierarchies:
   ```python
   coarse_slots = SlotAttention(features, num_slots=4)     # Object level
   fine_slots = SlotAttention(features, num_slots=16)      # Part level
   ```

2. **Temporal Slot Persistence**
   Track slot identities across time using Hungarian matching:
   ```python
   slot_assignments = hungarian_match(slots_t, slots_{t-1})
   identity_loss = || slots_t[assignments] - slots_{t-1} ||²
   ```

3. **Goal-Conditioned Attention**
   Condition attention prior on task goals:
   ```python
   attention_prior = AttentionPrior(h, z, goal_embedding)
   ```

4. **Attention-Based Action Selection**
   Use slot features for object-centric policy:
   ```python
   slot_features = [f_1, ..., f_K]  # Per-object representations
   object_values = [Q(f_k, a) for f_k in slot_features]
   action = argmax_a(max_k Q(f_k, a))  # Act on most relevant object
   ```

5. **Differentiable Slot Retrieval**
   Query slots by feature similarity:
   ```python
   query = encoder(target_object)
   similarities = [cosine_sim(query, slot_k) for slot_k in slots]
   attended_slot = sum(softmax(similarities) * slots)
   ```

---

## File Structure

```
attention_schema/
├── __init__.py                  # Public API exports
├── README.md                    # This file
├── slot_attention.py            # SlotAttention implementation
├── attention_posterior.py       # AttentionPosterior (bottom-up)
├── attention_prior.py           # AttentionPrior (top-down)
├── attention_schema.py          # AttentionSchema integration wrapper
└── spatial_utils.py             # ConvGRUCell utility
```

---

## Testing

### Basic Functionality Test

```python
import torch
from attention_schema import AttentionSchema

# Initialize
attention = AttentionSchema(
    image_size=84,
    attention_resolution=21,
    hidden_dim=256,
    latent_dim=32,
    context_dim=128
)

# Test posterior
obs = torch.randn(2, 3, 84, 84)
h = torch.randn(2, 256)
c = torch.randn(2, 128)

attn_post, coords = attention.posterior_net(obs, h, c)
assert attn_post.shape == (2, 21, 21)
assert coords.shape == (2, 2)
print("✓ Posterior test passed")

# Test prior
z = torch.randn(2, 32)
attn_prior, info = attention.prior_net(attn_post, h, z)
assert attn_prior.shape == (2, 21, 21)
print("✓ Prior test passed")

# Test dynamics loss
attention_seq = [torch.randn(2, 21, 21) for _ in range(10)]
pred_mov = [torch.randn(2, 2) for _ in range(9)]
loss = attention.compute_attention_dynamics_loss(attention_seq, pred_mov)
assert loss.ndim == 0  # Scalar
print("✓ Dynamics loss test passed")

print("\nAll tests passed!")
```

### Shape Verification Script

```python
def test_attention_schema_shapes():
    """Comprehensive shape testing"""
    B, T = 4, 16  # Batch size, sequence length

    attention = AttentionSchema(
        image_size=84,
        attention_resolution=21,
        hidden_dim=256,
        latent_dim=32,
        context_dim=128
    )

    print("Testing AttentionPosterior shapes...")
    obs = torch.randn(B, 3, 84, 84)
    h = torch.randn(B, 256)
    c = torch.randn(B, 128)

    attn, coords = attention.posterior_net(obs, h, c)

    assert attn.shape == (B, 21, 21), f"Expected {(B, 21, 21)}, got {attn.shape}"
    assert coords.shape == (B, 2), f"Expected {(B, 2)}, got {coords.shape}"

    posterior = attention.posterior_net
    assert posterior.slot_attention_maps.shape == (B, 4, 21, 21)
    assert posterior.slot_centers.shape == (B, 4, 2)
    assert posterior.slot_features.shape == (B, 4, 64)
    assert posterior.bottleneck_features.shape == (B, 128)

    print("✓ All posterior shapes correct")

    print("\nTesting AttentionPrior shapes...")
    z = torch.randn(B, 32)
    prev_attn = torch.randn(B, 21, 21)

    pred_attn, info = attention.prior_net(prev_attn, h, z)

    assert pred_attn.shape == (B, 21, 21)
    assert info['spatial_features'].shape == (B, 32, 21, 21)
    assert info['motion_features'].shape == (B, 8, 21, 21)
    assert len(info['predicted_movement']) == 2  # (dx, dy)
    assert info['predicted_movement'][0].shape == (B, 21, 21)

    print("✓ All prior shapes correct")

    print("\nAll shape tests passed!")

if __name__ == "__main__":
    test_attention_schema_shapes()
```

---

**PILLAR 4: ATTENTION complete - Precision-weighted inference operational** ✓
