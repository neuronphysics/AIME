# Perceiver IO Module

**PILLAR 1: PERCEPTION - Sensory Compression & Abstract Tokenization**

---

## Theory

### Cognitive Role

The Perceiver IO module implements **perceptual chunking** - the process of converting raw, high-dimensional sensory data (video frames) into compact, structured representations.

**Biological Inspiration**: The human brain doesn't process raw retinal input. Instead, it chunks visual information into objects, features, and concepts. Perceiver IO is AIME's analog of this perceptual organization.

### Why Discrete Tokens?

1. **Compositionality**: Complex scenes = compositions of reusable primitives
2. **Sparsity**: Most information can be represented with small active set
3. **Robustness**: Discrete representations resist noise better than continuous
4. **Scalability**: Attention over tokens is O(T·H_t·W_t) vs O(T·H·W) over pixels

### Information Flow

```
Raw Video [B, T, 3, 64, 64]
    ↓ (VQ-VAE Tokenization)
Discrete Tokens [B, T, 8, 8]
    ↓ (Perceiver Encoding)
Context Vectors [B, T, 256]
```

---

## Components

### 1. VQPTTokenizer (`tokenizer.py`)

**Purpose**: Compress video frames into discrete token sequences using Vector Quantization.

**Architecture**: 3D UNet Encoder + VQ Layer + 3D UNet Decoder

**Key Features**:
- Supports 2D (per-frame) and 3D (spatiotemporal) convolutions
- Multi-head vector quantization (4 codebooks)
- Configurable spatial downsampling (typically 4x or 8x)
- Optional temporal downsampling

**Tensor Flow**:
```python
Input:  [B, T, 3, 64, 64]  # Raw RGB video
   ↓ 3D UNet Encoder
[B, 256, T', 8, 8]  # Encoded features (T' may be smaller if temporal_downsample)
   ↓ Vector Quantize
[B, T, 8, 8]  # Discrete token indices in [0, 1023]
   ↓ 3D UNet Decoder
[B, T, 3, 64, 64]  # Reconstructed video
```

**Usage**:
```python
from perceiver_io import VQPTTokenizer

tokenizer = VQPTTokenizer(
    in_channels=3,
    code_dim=256,
    num_codes=1024,
    downsample=4,
    use_3d_conv=True,
    temporal_downsample=False
)

# Encode video to tokens
tokens, vq_loss, perplexity = tokenizer.encode(video)
# tokens: [B, T, 8, 8] discrete indices

# Decode tokens back to video
reconstructed = tokenizer.decode(tokens)
# reconstructed: [B, T, 3, 64, 64]
```

---

### 2. PerceiverTokenPredictor (`predictor.py`)

**Purpose**: Predict future video tokens using Perceiver IO architecture.

**Architecture**:
- Cross-attention encoder (tokens → latent bottleneck)
- Self-attention processor (refine latents)
- Cross-attention decoder (latents → future tokens)

**Key Features**:
- Autoregressive prediction (predict t+1 from 1:t)
- MaskGIT-style iterative generation
- Perceptual loss for better visual quality

**Tensor Flow**:
```python
Context Tokens [B, T_ctx, H_t, W_t]
   ↓ Flatten & Embed
[B, T_ctx * H_t * W_t, D]  # Token embeddings
   ↓ Perceiver Encoder (cross-attention)
[B, N_latent, D_latent]  # Latent bottleneck (e.g., [B, 128, 256])
   ↓ Self-Attention Layers
[B, N_latent, D_latent]  # Refined latents
   ↓ Perceiver Decoder (cross-attention)
[B, T_pred, H_t, W_t, vocab_size]  # Logits for future tokens
```

**Usage**:
```python
from perceiver_io import PerceiverTokenPredictor

predictor = PerceiverTokenPredictor(
    tokenizer=tokenizer,
    num_latents=128,
    num_latent_channels=256,
    num_self_attention_layers=6,
    sequence_length=16
)

# Predict future frames
outputs = predictor.forward(
    videos,
    num_context_frames=10  # Use first 10 frames to predict next 6
)

# outputs['reconstructed']: [B, 16, 3, 64, 64]
# outputs['predicted_tokens']: [B, 6, 8, 8]
```

---

### 3. CausalPerceiverIO (`causal_perceiver.py`)

**Purpose**: Full Perceiver IO pipeline - the main interface for VRNN integration.

**Combines**: VQPTTokenizer + PerceiverTokenPredictor

**Key Method**: `extract_context()` - used by VRNN to get per-frame context vectors

**Tensor Flow**:
```python
Video [B, T, 3, 64, 64]
   ↓ Tokenize
Tokens [B, T, 8, 8]
   ↓ Embed
Token Embeddings [B, T, 64, 256]  # 64 = 8*8 tokens per frame
   ↓ Perceiver Encoder
Latents [B, T, N_latent, D_latent]  # e.g., [B, T, 128, 256]
   ↓ Pool (mean over latents)
Context [B, T, 256]  # One context vector per frame
```

**Usage** (VRNN Integration):
```python
from perceiver_io import CausalPerceiverIO

# Initialize
perceiver = CausalPerceiverIO(
    video_shape=(16, 3, 64, 64),
    num_latents=128,
    num_latent_channels=256,
    num_codes=1024,
    downsample=4
)

# Extract context for VRNN
context = perceiver.extract_context(observations)
# context: [B, T, 256] - one vector per frame

# This context is then used by:
# 1. VAE Encoder: q(z | x, context)
# 2. DPGMM Prior: p(z | h, context)
# 3. LSTM Dynamics: h_t = f(h_{t-1}, z_t, context_t, a_t)
```

---

## Modules (`modules/` directory)

### Core Perceiver Components

- **`encoder.py`**: PerceiverEncoder (cross-attention: inputs → latents)
- **`decoder.py`**: PerceiverDecoder (cross-attention: latents → outputs)
- **`vector_quantize.py`**: VectorQuantize layer with multi-head support
- **`position.py`**: RoPE positional embeddings
- **`adapter.py`**: Adapter layers for fine-tuning
- **`utilities.py`**: Helper functions

---

## Tests (`tests/` directory)

### Standalone Test Scripts

Each test uses small synthetic data and prints shapes:

**`test_tokenizer.py`**:
```bash
python perceiver_io/tests/test_tokenizer.py
```
Demonstrates: Video → Tokens → Reconstructed Video

**`test_predictor.py`**:
```bash
python perceiver_io/tests/test_predictor.py
```
Demonstrates: Context tokens → Future prediction

**`demo_perceiver_flow.py`**:
```bash
python perceiver_io/tests/demo_perceiver_flow.py
```
Demonstrates: Full pipeline (end-to-end)

---

## Design Decisions

### Why 3D Convolutions?

**2D (Per-Frame)**:
- Process each frame independently
- Faster, less memory
- Good for static scenes

**3D (Spatiotemporal)**:
- Capture motion directly
- Better for dynamic scenes
- Used in AIME by default

### Why Multi-Head VQ?

Multiple codebooks (4 heads) learn complementary representations:
- Head 1: Low-level textures
- Head 2: Mid-level shapes
- Head 3: High-level structures
- Head 4: Dynamics/motion

Increases codebook capacity from 1024 → 4096 effective tokens.

### Why Perceiver Architecture?

**Advantages over Transformers**:
- O(N_latent · N_inputs) vs O(N_inputs²)
- Fixed compute regardless of input size
- Naturally handles spatiotemporal inputs

**Latent bottleneck** (128 or 512 latents) acts as working memory.

---

## Hyperparameters

### VQPTTokenizer

```python
in_channels = 3              # RGB
code_dim = 256               # VQ embedding dimension
num_codes = 1024             # Codebook size per head
downsample = 4               # Spatial downsampling (64 → 16 or 8)
base_channels = 64           # UNet base channels
use_3d_conv = True           # Spatiotemporal convolutions
temporal_downsample = False  # Keep temporal dimension
num_quantizers = 4           # Multi-head VQ
commitment_weight = 0.05     # VQ commitment loss weight
```

### PerceiverTokenPredictor

```python
num_latents = 128            # Number of latent queries
num_latent_channels = 256    # Context dimension
num_cross_attention_heads = 8
num_self_attention_layers = 6
widening_factor = 4          # MLP hidden dim multiplier
dropout = 0.1
```

### CausalPerceiverIO (Combined)

Uses above hyperparameters, typically:
- 128-512 latents
- 256-512 context dimension
- 1024-4096 codebook size (with multi-head)

---

## Loss Functions

### VQ-VAE Losses

```python
L_vq = L_reconstruction + β_commit · L_commitment + L_diversity

where:
  L_reconstruction = ||x - decode(quantize(encode(x)))||²
  L_commitment = ||sg(z_e) - z_q||²  # sg = stop gradient
  L_diversity = entropy penalty on codebook usage
```

### Perceiver Losses

```python
L_perceiver = L_token_prediction + λ_perceptual · L_perceptual

where:
  L_token_prediction = CrossEntropy(predicted_tokens, target_tokens)
  L_perceptual = LPIPS(predicted_frames, target_frames)
```

---

## Integration with VRNN

The Perceiver IO module is used by the main VRNN model as follows:

**In `VRNN/dpgmm_stickbreaking_prior_vrnn.py`**:

```python
# Initialization (line ~764)
self._init_perceiver_context():
    self.perceiver_model = CausalPerceiverIO(...)

# Forward pass (line ~1200)
def forward_sequence(self, observations, actions):
    # Extract context once for entire sequence
    c = self.perceiver_model.extract_context(observations)
    # c: [B, T, 256]

    for t in range(T):
        c_t = c[:, t]  # [B, 256]

        # Use context in:
        # 1. VAE encoder
        z_params = self.encoder(observations[:, t], c_t)

        # 2. DPGMM prior
        h_c = torch.cat([h_t, c_t], dim=-1)
        prior_params = self.prior(h_c)

        # 3. LSTM dynamics
        lstm_input = torch.cat([z_t, c_t, actions[:, t]], dim=-1)
        h_t, c_t_cell = self._rnn(lstm_input, (h_t, c_t_cell))
```

---

## Future Extensions

### Hierarchical Perceiver

Multi-scale latents for different temporal abstractions:
```python
latents_fast = perceiver.encode(video, timescale='fast')    # Frame-level
latents_medium = perceiver.encode(video, timescale='medium') # ~4 frames
latents_slow = perceiver.encode(video, timescale='slow')    # ~16 frames
```

### Conditional Generation

Condition on goals or instructions:
```python
context = perceiver.extract_context(video, condition=goal_embedding)
```

### Online Adaptation

Update codebook during deployment:
```python
perceiver.tokenizer.adapt_codebook(new_domain_videos)
```

---

## References

- **Perceiver IO**: Jaegle et al. (2021) - General architecture for structured inputs
- **VQ-VAE**: van den Oord et al. (2017) - Discrete representation learning
- **RoPE**: Su et al. (2021) - Rotary positional embeddings
- **MaskGIT**: Chang et al. (2022) - Masked generative image transformer

---

For detailed tensor shapes, see: `docs/TENSOR_SHAPE_REFERENCE.md`

For architectural context, see: `docs/ARCHITECTURE_OVERVIEW.md`

For theory, see: `docs/THEORY_AND_PHILOSOPHY.md` - Section "Perceiver IO: Sensory Compression"
