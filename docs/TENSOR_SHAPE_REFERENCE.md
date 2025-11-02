# AIME Tensor Shape Reference

**Complete tensor transformations for debugging and development**

---

## Table of Contents

1. [Quick Reference Tables](#quick-reference-tables)
2. [Configuration & Hyperparameters](#configuration--hyperparameters)
3. [Forward Pass: Complete Shape Trace](#forward-pass-complete-shape-trace)
4. [Per-Component Shape Documentation](#per-component-shape-documentation)
5. [Common Shape Errors & Solutions](#common-shape-errors--solutions)
6. [Shape Validation Checklist](#shape-validation-checklist)

---

## Quick Reference Tables

### Standard Dimensions

| Symbol | Meaning | Typical Value | Description |
|--------|---------|---------------|-------------|
| `B` | Batch size | 8 | Number of sequences per batch |
| `T` | Sequence length | 16 | Number of frames per sequence |
| `C` | Image channels | 3 | RGB channels |
| `H` | Image height | 64 | Original image height |
| `W` | Image width | 64 | Original image width |
| `D` | Latent dim | 36 | Dimensionality of latent variable z |
| `K` | Max DPGMM components | 15 | Maximum mixture components |
| `K_slots` | Attention slots | 4-5 | Number of attention slots |
| `H_att` | Attention resolution | 21 | Spatial resolution of attention maps |
| `W_att` | Attention resolution | 21 | Spatial resolution of attention maps |
| `H_t` | Token height | 8 | VQ token grid height (H / downsample) |
| `W_t` | Token width | 8 | VQ token grid width (W / downsample) |
| `A` | Action dim | 6 | Action space dimensionality (DMC humanoid) |
| `hidden_dim` | LSTM hidden | 512 | LSTM hidden state size |
| `context_dim` | Perceiver context | 256 | Context vector size |

### Key Tensor Shapes at a Glance

| Tensor | Shape | Location | Description |
|--------|-------|----------|-------------|
| Observations | `[B, T, 3, 64, 64]` | Input | Raw video frames |
| Actions | `[B, T, 6]` | Input | Action sequences |
| VQ Tokens | `[B, T, 8, 8]` | Perceiver | Discrete token indices |
| Context | `[B, T, 256]` | Perceiver | Per-frame context vectors |
| Latent z | `[B, T, 36]` | VAE Encoder | Latent state variables |
| LSTM hidden | `[B, T, 512]` | VRNN | Recurrent hidden states |
| DPGMM weights | `[B, T, 15]` | Prior | Mixture component weights |
| Attention map | `[B, T, 21, 21]` | Attention | Spatial attention probabilities |
| Slot features | `[B, T, K_slots, 64]` | Attention | Per-slot feature vectors |
| Reconstructions | `[B, T, 3, 64, 64]` | Output | Decoded video frames |

---

## Configuration & Hyperparameters

### Default Configuration

```python
# From VRNN/dpgmm_stickbreaking_prior_vrnn.py:698-732

config = {
    # Core dimensions
    'max_components': 15,          # K in equations
    'input_dim': 64,               # H, W (square images)
    'latent_dim': 36,              # D
    'hidden_dim': 512,             # LSTM hidden size
    'attention_dim': 64,           # Slot feature dimension
    'action_dim': 6,               # A (DMC humanoid)
    'sequence_length': 16,         # T

    # Perceiver IO
    'num_latent_perceiver': 128,   # Number of latent queries
    'num_latent_channels_perceiver': 256,  # context_dim
    'num_codebook_perceiver': 1024,  # VQ codebook size
    'perceiver_code_dim': 256,     # VQ embedding dimension
    'downsample_perceiver': 4,     # Spatial downsample factor

    # Attention
    'attention_resolution': 21,    # H_att, W_att
    'num_semantic_slots': 4,       # K_slots

    # Training
    'batch_size': 8,               # B
    'learning_rate': 0.0002,
    'grad_clip': 5.0,
    'warmup_epochs': 25,
}
```

### Derived Shapes

```python
# Token grid dimensions (from downsample factor)
H_t = H // downsample_perceiver  # 64 // 4 = 16 (but may be 8 after 3D conv)
W_t = W // downsample_perceiver  # 64 // 4 = 16 (but may be 8 after 3D conv)

# Actual values depend on use_3d_conv and temporal_downsample flags
# Typical: H_t = W_t = 8 for 3D convolution mode
```

---

## Forward Pass: Complete Shape Trace

### Example: Single Batch Forward Pass

**Input Configuration:**
- Batch size: B = 8
- Sequence length: T = 16
- Image size: 64×64 RGB
- DMC Humanoid actions: 6-dim

### Step-by-Step Shapes

```python
# ============================================================
# INPUT
# ============================================================
observations = torch.randn(8, 16, 3, 64, 64)  # [B, T, C, H, W]
actions = torch.randn(8, 16, 6)               # [B, T, A]

# ============================================================
# PILLAR 1: PERCEIVER IO (Context Extraction)
# ============================================================

# --- VQ Tokenizer ---
# Input video reshaped for 3D processing
video_input = observations  # [8, 16, 3, 64, 64]

# 3D UNet Encoder
encoder_input = video_input.view(8*16, 3, 64, 64)  # [128, 3, 64, 64] (process frames)
# OR keep temporal: [8, 3, 16, 64, 64] for true 3D conv
# After 3D convolutions + downsampling
encoded = ...  # [8, 256, 2, 8, 8] (C=256, T'=2 if temporal_downsample, H'=W'=8)

# Vector Quantization
# Flatten spatial+temporal for VQ
vq_input = encoded.view(8, 256, 2*8*8)  # [8, 256, 128]
vq_input = vq_input.permute(0, 2, 1)     # [8, 128, 256] for VQ layer

quantized, vq_loss, perplexity = vq_layer(vq_input)
# quantized: [8, 128, 256] (same shape, but from codebook)
# vq_loss: scalar
# perplexity: scalar

# Get token indices
token_indices = vq_layer.get_indices(vq_input)  # [8, 128] discrete in [0, 1023]
# Reshape to spatial grid
tokens = token_indices.view(8, 2, 8, 8)  # [B, T', H_t, W_t]
# After temporal upsampling back to T=16
tokens = upsample_temporal(tokens)  # [8, 16, 8, 8]

# 3D UNet Decoder
decoded = decoder(quantized)  # [8, 16, 3, 64, 64]

# --- Perceiver Context Extraction ---
# Flatten tokens to sequence
token_seq = tokens.view(8*16, 8*8)  # [128, 64] tokens per frame
token_embeddings = embed(token_seq)  # [128, 256] embed each token

# Perceiver Encoder (cross-attention)
latent_queries = repeat(learned_queries, '1 n d -> b n d', b=128)  # [128, 128, 256]
attended = perceiver_encoder(latent_queries, token_embeddings)  # [128, 128, 256]

# Pool to context vector
context_per_frame = attended.mean(dim=1)  # [128, 256]
context = context_per_frame.view(8, 16, 256)  # [B, T, context_dim]

# ============================================================
# PILLAR 2 + 3: VAE + VRNN (Per-Timestep Loop)
# ============================================================

# Initialize LSTM states
h_0 = torch.zeros(8, 512)  # [B, hidden_dim]
c_0 = torch.zeros(8, 512)  # [B, hidden_dim]

outputs = {
    'latents': [],
    'reconstructions': [],
    'attention_maps': [],
    # ... other outputs
}

for t in range(16):  # T = 16
    # --- Extract current timestep ---
    x_t = observations[:, t]  # [8, 3, 64, 64]
    c_t = context[:, t]       # [8, 256]
    a_t = actions[:, t]       # [8, 6]

    # --- VAE Encoder: q(z_t | x_t, c_t) ---
    encoder_input = torch.cat([x_t, c_t.view(8, 256, 1, 1).expand(8, 256, 64, 64)], dim=1)
    # After concatenation: [8, 3+256, 64, 64] (channel-wise)
    # Actually, encoder takes x_t and c_t separately, not concatenated

    z_params = vae_encoder(x_t, c_t)
    # z_params = {'mu': [8, 36], 'logvar': [8, 36]}

    # Reparameterization trick
    std = torch.exp(0.5 * z_params['logvar'])  # [8, 36]
    eps = torch.randn_like(std)                 # [8, 36]
    z_t = z_params['mu'] + eps * std            # [8, 36]

    # --- DPGMM Prior: p(z_t | h_t, c_t) ---
    h_c = torch.cat([h_t, c_t], dim=-1)  # [8, 512+256] = [8, 768]

    # Stick-breaking
    stick_params = stick_breaking_net(h_c)
    # kumar_a: [8, 14] (K-1 sticks)
    # kumar_b: [8, 14]
    v = kumaraswamy_sample(stick_params['a'], stick_params['b'])  # [8, 14]

    # Convert sticks to mixture weights
    pi = stick_to_weights(v)  # [8, 15] (K components)

    # Component parameters
    component_params = component_nn(h_c)  # [8, 2*36*15] = [8, 1080]
    component_params = component_params.view(8, 15, 2, 36)  # [B, K, 2, D]
    mu_k = component_params[:, :, 0, :]  # [8, 15, 36]
    logvar_k = component_params[:, :, 1, :]  # [8, 15, 36]

    # --- LSTM Dynamics: h_t = f(h_{t-1}, z_t, c_t, a_t) ---
    lstm_input = torch.cat([z_t, c_t, a_t], dim=-1)  # [8, 36+256+6] = [8, 298]

    h_t, c_t_cell = lstm(lstm_input, (h_t, c_t))
    # h_t: [8, 512]
    # c_t_cell: [8, 512] (LSTM cell state, not context!)

    # --- Attention Schema: A_t = g(x_t, h_t) ---
    # Bottom-up (stimulus-driven)
    phi_att = feature_extractor(x_t)  # [8, 64, 21, 21]
    phi_seq = phi_att.flatten(2).transpose(1, 2)  # [8, 441, 64] (21*21=441)

    # Slot attention
    slots, attn = slot_attention(phi_seq)
    # slots: [8, 4, 64] (K_slots=4)
    # attn: [8, 4, 441] (responsibilities)

    slot_maps = attn.view(8, 4, 21, 21)  # [B, K_slots, H_att, W_att]

    # Top-down (prediction-driven)
    h_c_att = torch.cat([h_t, c_t], dim=-1)  # [8, 768]
    motion_field = attention_prior(h_c_att)  # [8, 2, 21, 21] (dx, dy)

    # Fusion
    A_t = fusion(slot_maps, motion_field, h_c_att)  # [8, 21, 21]

    # --- VAE Decoder: p(x_t | z_t, A_t) ---
    # Decoder takes z and attention
    x_hat_t = vae_decoder(z_t, A_t)  # [8, 3, 64, 64]

    # --- Store outputs ---
    outputs['latents'].append(z_t)           # [8, 36]
    outputs['reconstructions'].append(x_hat_t)  # [8, 3, 64, 64]
    outputs['attention_maps'].append(A_t)    # [8, 21, 21]
    # ... other outputs

    # Update for next timestep
    h_t = h_t  # [8, 512] (already updated by LSTM)
    c_t = c_t_cell  # [8, 512]

# END FOR

# Stack outputs across time
latents = torch.stack(outputs['latents'], dim=1)  # [8, 16, 36]
reconstructions = torch.stack(outputs['reconstructions'], dim=1)  # [8, 16, 3, 64, 64]
attention_maps = torch.stack(outputs['attention_maps'], dim=1)  # [8, 16, 21, 21]

# ============================================================
# PILLAR 5: LOSS COMPUTATION
# ============================================================

# Task 1: ELBO
recon_loss = F.mse_loss(reconstructions, observations)  # scalar
kl_z = kl_divergence(z_params, prior_params)  # scalar
# ... other KL terms

# Task 2: Perceiver
vq_loss = vq_loss  # scalar (from earlier)
perceiver_recon_loss = F.mse_loss(decoded, observations)  # scalar

# Task 3: Predictive
attention_dynamics_loss = attention_schema.compute_dynamics_loss(...)  # scalar
diversity_loss = attention_schema.diversity_loss  # scalar

# Task 4: Adversarial
real_score = discriminator(observations)  # [8, 1]
fake_score = discriminator(reconstructions.detach())  # [8, 1]
gan_loss = -torch.log(fake_score).mean()  # scalar

# Total losses per task
L_ELBO = recon_loss + kl_z + ...  # scalar
L_perceiver = vq_loss + perceiver_recon_loss  # scalar
L_predictive = attention_dynamics_loss + diversity_loss  # scalar
L_adversarial = gan_loss + ...  # scalar

losses = [L_ELBO, L_perceiver, L_predictive, L_adversarial]

# ============================================================
# PILLAR 5: RGB GRADIENT BALANCING
# ============================================================

# Compute task gradients
grads = []  # List of [D] tensors, where D = total parameter count
for loss in losses:
    loss.backward(retain_graph=True)
    grad_vec = flatten_gradients()  # [D] (e.g., D ~ millions)
    grads.append(grad_vec)
    zero_grad()

grads = torch.stack(grads)  # [4, D] (4 tasks)

# RGB algorithm
balanced_grad = rgb.balance(grads)  # [D]

# Set gradients and update
unflatten_gradients(balanced_grad)
optimizer.step()
```

---

## Per-Component Shape Documentation

### 1. Perceiver IO

#### VQPTTokenizer

```python
class VQPTTokenizer:
    def encode(self, video):
        """
        Args:
            video: [B, T, C, H, W]

        Returns:
            tokens: [B, T, H_t, W_t] discrete indices in [0, num_codes-1]
            vq_loss: scalar
            perplexity: scalar
        """

    def decode(self, tokens):
        """
        Args:
            tokens: [B, T, H_t, W_t] discrete indices

        Returns:
            reconstructed: [B, T, C, H, W]
        """
```

**Shape flow:**
```
Input:  [B, T, 3, 64, 64]
   ↓ 3D UNet Encoder
[B, C_enc, T', H', W']  # e.g., [B, 256, 2, 8, 8]
   ↓ Flatten spatial+temporal
[B, C_enc, T'*H'*W']  # [B, 256, 128]
   ↓ VQ Layer
[B, T'*H'*W', C_vq]  # [B, 128, 256]
   ↓ Quantize
tokens: [B, T'*H'*W'] discrete  # [B, 128]
   ↓ Reshape
[B, T', H', W']  # [B, 2, 8, 8]
   ↓ Upsample temporal
[B, T, H_t, W_t]  # [B, 16, 8, 8]
```

#### CausalPerceiverIO

```python
class CausalPerceiverIO:
    def extract_context(self, videos):
        """
        Args:
            videos: [B, T, C, H, W]

        Returns:
            context: [B, T, context_dim]
        """
```

---

### 2. DPGMM Prior

#### DPGMMPrior

```python
class DPGMMPrior:
    def forward(self, h, n_samples=10):
        """
        Args:
            h: [B, hidden_dim + context_dim]  # Concatenated
               Typical: [B, 512+256] = [B, 768]

        Returns:
            prior_params: Dict with keys:
                'pi': [B, K] mixture weights (sum to 1)
                'means': [B, K, D] component means
                'log_vars': [B, K, D] component log-variances
                'kumar_a': [B, K-1] Kumaraswamy alpha params
                'kumar_b': [B, K-1] Kumaraswamy beta params
                'alpha': [B, n_samples] concentration params
        """
```

**Stick-breaking shape flow:**
```
Input h: [B, 768]
   ↓ Kumaraswamy network
kumar_a: [B, K-1]  # e.g., [B, 14]
kumar_b: [B, K-1]
   ↓ Sample sticks
v: [B, K-1]  # in (0, 1)
   ↓ Convert to weights
pi: [B, K]  # mixture weights, sum to 1
```

---

### 3. Temporal Dynamics

#### LSTMLayer

```python
class LSTMLayer(nn.LSTM):
    def forward(self, input, hx):
        """
        Args:
            input: [B, input_dim] or [B, T, input_dim]
            hx: Tuple of ([num_layers, B, hidden_dim], [num_layers, B, hidden_dim])

        Returns:
            output: [B, hidden_dim] or [B, T, hidden_dim]
            (h_n, c_n): Same shape as hx
        """
```

**Typical usage in VRNN:**
```python
# Per-timestep
lstm_input = torch.cat([z_t, c_t, a_t], dim=-1)  # [B, 36+256+6] = [B, 298]
h_t, (h_new, c_new) = lstm(lstm_input, (h_t, c_t))
# h_t: [B, 512]
# h_new: [1, B, 512] (for single-layer LSTM)
# c_new: [1, B, 512]
```

---

### 4. Attention Schema

#### SlotAttention

```python
class SlotAttention:
    def forward(self, x, seed_slots=None):
        """
        Args:
            x: [B, N, D] input features (e.g., flattened spatial features)
               Typical: [B, 441, 64] for 21×21 attention resolution
            seed_slots: [B, K, D] optional initialization

        Returns:
            slots: [B, K, D] slot feature vectors
            attn: [B, K, N] attention weights (sum over K = 1 for each position)
        """
```

**Shape flow:**
```
Input features: [B, 441, 64]  # 21×21 spatial positions
   ↓ Initialize slots
slots: [B, 4, 64]  # K_slots = 4
   ↓ Iterative attention (3 iterations)
For iter in range(3):
    Q = to_q(slots)  # [B, 4, 64]
    K = to_k(x)      # [B, 441, 64]
    V = to_v(x)      # [B, 441, 64]

    attn = softmax(Q @ K^T / sqrt(64), dim=2)  # [B, 4, 441]
    updates = attn @ V  # [B, 4, 64]

    slots = GRU(updates, slots)  # [B, 4, 64]
   ↓ Final output
slots: [B, 4, 64]
attn: [B, 4, 441]
```

#### AttentionSchema

```python
class AttentionSchema:
    def forward(self, observation, hidden_state, context, fused_feat=None):
        """
        Args:
            observation: [B, 3, H, W]
            hidden_state: [B, hidden_dim]
            context: [B, context_dim]
            fused_feat: [B, C, H_att, W_att] optional pre-computed features

        Returns:
            attention_probs_2d: [B, H_att, W_att]
            regularized_coords: [B, 2] center of attention in [-1, 1]
        """
```

---

### 5. VAE Encoder/Decoder

#### VAEEncoder

```python
class VAEEncoder:
    def forward(self, x, context):
        """
        Args:
            x: [B, C, H, W] image
            context: [B, context_dim] from Perceiver

        Returns:
            z_params: Dict with keys:
                'mu': [B, latent_dim]
                'logvar': [B, latent_dim]
        """
```

**Hierarchical encoding:**
```
Input: [B, 3, 64, 64]
   ↓ Downsample blocks
[B, 64, 32, 32]   # ResBlock + Down
[B, 128, 16, 16]  # ResBlock + Down
[B, 256, 8, 8]    # ResBlock + Down
[B, 512, 4, 4]    # ResBlock + Down
   ↓ Flatten
[B, 512*4*4] = [B, 8192]
   ↓ Concat context
[B, 8192+256]
   ↓ Linear layers
mu: [B, 36]
logvar: [B, 36]
```

#### VAEDecoder

```python
class VAEDecoder:
    def forward(self, z, attention_map):
        """
        Args:
            z: [B, latent_dim]
            attention_map: [B, H_att, W_att]

        Returns:
            reconstruction: [B, C, H, W]
        """
```

**Hierarchical decoding with attention:**
```
z: [B, 36]
   ↓ Linear + Reshape
[B, 512, 4, 4]
   ↓ ResBlock + Up
[B, 256, 8, 8]
   ↓ Attention modulation (interpolate attention to 8×8)
[B, 256, 8, 8] ⊙ attention [B, 1, 8, 8]
   ↓ ResBlock + Up
[B, 128, 16, 16]
   ↓ Attention modulation (16×16)
[B, 128, 16, 16] ⊙ attention [B, 1, 16, 16]
   ↓ ResBlock + Up
[B, 64, 32, 32]
   ↓ ResBlock + Up
[B, 3, 64, 64]
```

---

## Common Shape Errors & Solutions

### Error 1: Dimension Mismatch in LSTM Input

**Error:**
```
RuntimeError: Expected input batch_size (128) to match target batch_size (8)
```

**Cause:** Passing [B*T, D] instead of [B, D] to LSTM in per-timestep mode.

**Solution:**
```python
# WRONG
lstm_input = torch.cat([z, c, a], dim=-1)  # [B*T, D] if z,c,a not sliced
h, c = lstm(lstm_input, (h_prev, c_prev))

# RIGHT
for t in range(T):
    z_t = z[:, t]  # [B, D]
    c_t = c[:, t]  # [B, context_dim]
    a_t = a[:, t]  # [B, A]
    lstm_input = torch.cat([z_t, c_t, a_t], dim=-1)  # [B, 36+256+6]
    h_t, c_t = lstm(lstm_input, (h_prev, c_prev))
```

---

### Error 2: Attention Map Size Mismatch

**Error:**
```
RuntimeError: The size of tensor a (21) must match the size of tensor b (64) at non-singleton dimension 2
```

**Cause:** Trying to modulate decoder feature map [B, C, 64, 64] with attention [B, 21, 21].

**Solution:**
```python
# Interpolate attention to match feature map size
attention_upsampled = F.interpolate(
    attention_map.unsqueeze(1),  # [B, 1, 21, 21]
    size=(64, 64),
    mode='bilinear',
    align_corners=False
)  # [B, 1, 64, 64]

modulated_features = features * attention_upsampled  # [B, C, 64, 64]
```

---

### Error 3: VQ Token Shape After 3D Convolution

**Error:**
```
RuntimeError: shape '[8, 16, 8, 8]' is invalid for input of size 1024
```

**Cause:** Temporal dimension changed after 3D convolution with temporal downsampling.

**Solution:**
```python
# Check actual temporal dimension after encoding
print(f"Encoded shape: {encoded.shape}")  # e.g., [8, 256, 2, 8, 8] not [8, 256, 16, 8, 8]

# Compute actual T'
T_enc = encoded.shape[2]  # 2

# Reshape accordingly
tokens = vq_indices.view(B, T_enc, H_t, W_t)  # [8, 2, 8, 8]

# Upsample temporal back to T=16
tokens = F.interpolate(
    tokens.float().unsqueeze(1),  # [8, 1, 2, 8, 8]
    size=(T, H_t, W_t),
    mode='nearest'
).squeeze(1).long()  # [8, 16, 8, 8]
```

---

### Error 4: DPGMM Component Parameter Shape

**Error:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x768 and 512x1080)
```

**Cause:** Component network expects `hidden_dim` but receives `hidden_dim + context_dim`.

**Solution:**
```python
# WRONG
prior_input = h_t  # [B, 512]
prior_params = dpgmm_prior(prior_input)  # Network expects [B, 768]

# RIGHT
prior_input = torch.cat([h_t, c_t], dim=-1)  # [B, 512+256] = [B, 768]
prior_params = dpgmm_prior(prior_input)
```

---

### Error 5: Slot Attention Input Not Flattened

**Error:**
```
RuntimeError: Expected 3D tensor [B, N, D], got 4D [B, C, H, W]
```

**Cause:** Feeding spatial feature map directly to slot attention without flattening.

**Solution:**
```python
# WRONG
slots, attn = slot_attention(feature_map)  # [B, 64, 21, 21]

# RIGHT
B, C, H, W = feature_map.shape
feature_seq = feature_map.flatten(2).transpose(1, 2)  # [B, H*W, C] = [B, 441, 64]
slots, attn = slot_attention(feature_seq)
# slots: [B, K, 64]
# attn: [B, K, 441]

# Reshape attention back to spatial if needed
attn_spatial = attn.view(B, K, H, W)  # [B, K, 21, 21]
```

---

## Shape Validation Checklist

Use this checklist when adding new components or debugging shape issues:

### Input Validation

- [ ] Observations are `[B, T, 3, H, W]`
- [ ] Actions are `[B, T, A]`
- [ ] Batch size B is consistent across all inputs
- [ ] Sequence length T is consistent
- [ ] Image dimensions H, W match config

### Perceiver IO

- [ ] VQ tokens are `[B, T, H_t, W_t]` where `H_t = H // downsample`
- [ ] Context vectors are `[B, T, context_dim]`
- [ ] Codebook embeddings are `[num_codes, code_dim]`
- [ ] VQ loss and perplexity are scalars

### VAE Encoder/Decoder

- [ ] Encoder input is `[B, C, H, W]` (single frame, not sequence)
- [ ] Encoder output mu, logvar are `[B, latent_dim]`
- [ ] Decoder input z is `[B, latent_dim]`
- [ ] Decoder output is `[B, C, H, W]`

### DPGMM Prior

- [ ] Prior input is `[B, hidden_dim + context_dim]` (concatenated)
- [ ] Mixture weights pi are `[B, K]` and sum to 1 along dim 1
- [ ] Component means are `[B, K, latent_dim]`
- [ ] Component log_vars are `[B, K, latent_dim]`
- [ ] Stick parameters kumar_a, kumar_b are `[B, K-1]`

### LSTM Dynamics

- [ ] LSTM input is `[B, latent_dim + context_dim + action_dim]` per timestep
- [ ] Hidden state h is `[B, hidden_dim]` or `[num_layers, B, hidden_dim]`
- [ ] Cell state c has same shape as h
- [ ] LSTM output matches hidden_dim

### Attention Schema

- [ ] Input features are `[B, N, D]` for slot attention (flattened spatial)
- [ ] Slot features are `[B, K_slots, slot_dim]`
- [ ] Attention weights are `[B, K_slots, N]` and sum to 1 along dim 1
- [ ] Final attention map is `[B, H_att, W_att]`
- [ ] Attention is in [0, 1] and sums to 1

### Loss Computation

- [ ] All losses are scalars (0-dimensional tensors)
- [ ] Reconstruction loss is non-negative
- [ ] KL divergences are non-negative
- [ ] No NaN or Inf in any loss

### Gradient Shapes

- [ ] Task gradients are 1D vectors `[D]` where D = total parameters
- [ ] RGB receives gradient matrix `[num_tasks, D]`
- [ ] Balanced gradient is `[D]`

---

## Debugging Tips

### 1. Add Shape Assertions

```python
def forward_sequence(self, observations, actions):
    B, T, C, H, W = observations.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    assert H == W == 64, f"Expected 64x64 images, got {H}x{W}"

    context = self.perceiver_model.extract_context(observations)
    assert context.shape == (B, T, self.context_dim), \
        f"Context shape mismatch: {context.shape}"

    # ... rest of forward pass
```

### 2. Log Intermediate Shapes

```python
def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    x = self.layer2(x)
    print(f"After layer2: {x.shape}")
    return x
```

### 3. Use Shape Comments

```python
# [B, T, 3, 64, 64]
observations = ...

# [B, T, 256]
context = self.perceiver.extract_context(observations)

# [B, 36] per timestep
z_t = self.encoder(observations[:, t], context[:, t])
```

### 4. Validate Shapes at Runtime

```python
from typing import Tuple

def validate_shape(tensor: torch.Tensor, expected: Tuple[int, ...], name: str):
    """Validate tensor shape, allowing -1 for any dimension."""
    if len(tensor.shape) != len(expected):
        raise ValueError(f"{name}: Expected {len(expected)} dims, got {len(tensor.shape)}")

    for i, (actual, exp) in enumerate(zip(tensor.shape, expected)):
        if exp != -1 and actual != exp:
            raise ValueError(f"{name}: Dim {i} expected {exp}, got {actual}")

# Usage
validate_shape(observations, (8, 16, 3, 64, 64), "observations")
validate_shape(context, (-1, -1, 256), "context")  # Allow any B, T
```

---

*For architectural context, see [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)*

*For code locations, see [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)*

*For theoretical understanding, see [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md)*
