# Encoder-Decoder Module

**VAE Components for Hierarchical Image Encoding/Decoding**

This module provides access to the Variational Autoencoder (VAE) encoder and decoder used in AIME.

## Architecture Overview

```
                VAE Pipeline
                     │
        ┌────────────┴────────────┐
        │                         │
    VAEEncoder              VAEDecoder
        │                         │
  [Encode to z]           [Decode from z]

  observation → z_mean, z_logvar → z → reconstruction
```

## Theoretical Foundation

The VAE implements the probabilistic encoding and decoding:

**Encoder**: q(z|x, c) - Approximate posterior
- Input: observations x, context c
- Output: z_mean, z_logvar (parameters of Gaussian posterior)

**Decoder**: p(x|z, A) - Conditional likelihood
- Input: latent z, attention map A
- Output: reconstructed observation

## Components

### VAEEncoder (`nvae_architecture.py`)

Hierarchical encoder with U-Net architecture and skip connections.

**Features:**
- Multi-scale feature extraction (C2, C3, C4, C5, C6 levels)
- Squeeze-and-excitation blocks for channel attention
- Residual connections for gradient flow
- Skip connection caching for decoder

**Input:**
- observations: [B, T, C, H, W] or [B, C, H, W]
- context: [B, T, context_dim] or [B, context_dim] (optional)

**Output:**
- z_mean: [B, latent_dim]
- z_logvar: [B, latent_dim]
- Additional encoder features (cached for skip connections)

**Usage:**
```python
from encoder_decoder import VAEEncoder

encoder = VAEEncoder(
    latent_dim=36,
    context_dim=256,
    image_channels=3,
    base_channels=32
)

z_mean, z_logvar = encoder(observations, context)
z = encoder.reparameterize(z_mean, z_logvar)
```

### VAEDecoder (`nvae_architecture.py`)

Hierarchical decoder with attention-modulated generation.

**Features:**
- Multi-scale upsampling with residual blocks
- Attention map integration for spatial control
- Mixture of Discretized Logistics (MDL) output head
- Skip connection integration from encoder

**Input:**
- z: [B, latent_dim] - Latent representation
- attention_map: [B, K, H_attn, W_attn] (optional) - Spatial attention
- skips: Dict of encoder features (from encoder.get_unet_skips())

**Output:**
- logit_probs: [B, C, num_mix, H, W] - Mixture component probabilities
- means: [B, C, num_mix, H, W] - Component means
- log_scales: [B, C, num_mix, H, W] - Component log scales
- coeffs: [B, C, num_mix, H, W] - RGB coefficients (for color channels)

**Usage:**
```python
from encoder_decoder import VAEDecoder

decoder = VAEDecoder(
    latent_dim=36,
    image_channels=3,
    base_channels=32,
    num_mixtures=10
)

# Get skip connections from encoder
skips = encoder.get_unet_skips(observations, levels=("C2","C3","C4","C5","C6"))

# Decode
logit_probs, means, log_scales, coeffs = decoder(z, skips=skips, attention_map=A)

# Sample reconstruction
reconstruction = decoder.mdl_head.sample(logit_probs, means, log_scales, coeffs)
```

### GramLoss (`nvae_architecture.py`)

Gram matrix loss for encoder feature consistency (student-teacher).

**Usage:**
```python
from encoder_decoder import GramLoss

gram_loss = GramLoss()
loss = gram_loss(student_features, teacher_features, img_level=True)
```

## Tensor Shape Reference

### Encoder Flow

```python
# Input
observations: [B, 3, 64, 64]       # RGB images
context: [B, 256]                   # Optional context vector

# Intermediate features (cached for skip connections)
C2: [B, 32, 32, 32]                # Early features
C3: [B, 64, 16, 16]                # Mid features
C4: [B, 128, 8, 8]                 # Deep features
C5: [B, 256, 4, 4]                 # Very deep features
C6: [B, 512, 2, 2]                 # Bottleneck features

# Output
z_mean: [B, 36]                    # Latent mean
z_logvar: [B, 36]                  # Latent log variance
```

### Decoder Flow

```python
# Input
z: [B, 36]                         # Latent code
attention_map: [B, K, 8, 8]        # Optional attention (K slots)
skips: Dict with encoder features

# Intermediate (progressive upsampling)
[B, 512, 2, 2] → [B, 256, 4, 4] → [B, 128, 8, 8] → [B, 64, 16, 16] → [B, 32, 32, 32]

# Output (Mixture of Discretized Logistics)
logit_probs: [B, 3, 10, 64, 64]   # Mixture probabilities
means: [B, 3, 10, 64, 64]          # Component means
log_scales: [B, 3, 10, 64, 64]    # Component scales
coeffs: [B, 3, 10, 64, 64]         # RGB coefficients

# Final reconstruction
reconstruction: [B, 3, 64, 64]     # Sampled RGB image
```

## Integration with AIME

Used in main VRNN model:

```python
# Encode observations
z_posterior_params = self.encoder(observations, c_t)
z_posterior = self._reparameterize(z_posterior_params)

# Get skip connections
skips = self.encoder.get_unet_skips(observations, levels=("C2","C3","C4","C5","C6"))

# Decode with attention modulation
logit_probs, means, log_scales, coeffs = self.decoder(
    z_posterior,
    skips=skips,
    attention_map=attention_output['attention_map']
)

# Sample reconstruction
reconstructions = self.decoder.mdl_head.sample(logit_probs, means, log_scales, coeffs)
```

## Design Principles

1. **Hierarchical Processing**: Multi-scale features for rich representations
2. **Skip Connections**: Preserve spatial information during decoding
3. **Attention Modulation**: Spatial control via attention schema
4. **Flexible Output**: MDL head for expressive likelihood modeling

## Benefits for AI Coders

1. **Modular Access**: Import encoder/decoder independently
2. **Clear Documentation**: Tensor shapes and usage examples provided
3. **Skip Connection System**: Well-documented feature caching mechanism
4. **Integration Points**: Clear how components connect to main model

## Future Extensions

- [ ] Add variational encoder layers (hierarchical VAE)
- [ ] Add VQ-VAE alternative
- [ ] Add flow-based decoder option
- [ ] Add diffusion decoder option

## References

**NVAE Architecture:**
- Vahdat & Kautz "NVAE: A Deep Hierarchical Variational Autoencoder" (NeurIPS 2020)

**Mixture of Discretized Logistics:**
- Salimans et al. "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood" (2017)

---

**Note:** The actual implementation remains in `nvae_architecture.py` (920 lines).
This module provides organized access and documentation.

**For questions about this module:**
- Implementation details: See `nvae_architecture.py`
- Integration: See `VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- Theory: See `docs/THEORY_AND_PHILOSOPHY.md`
