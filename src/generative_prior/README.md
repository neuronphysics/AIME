## Generative Prior Module

**PILLAR 2: Representation** - Adaptive infinite mixture priors for latent space

---

## Overview

The generative prior module implements **Dirichlet Process Gaussian Mixture Models (DPGMM)** as a flexible, context-dependent prior for VAE latent variables.

### Why DPGMM?

Standard VAEs use a simple Gaussian prior `p(z) = N(0, I)`, which assumes:
- Unimodal latent space (single "average" representation)
- Fixed prior (doesn't adapt to context)
- Predetermined capacity (dimensionality is fixed)

DPGMM addresses these limitations by providing:
1. **Multi-modal representations**: Multiple Gaussian components capture diverse hypotheses
2. **Context-dependent priors**: Prior adapts based on current hidden state `h_t`
3. **Automatic capacity**: Dirichlet Process learns the number of components

---

## Architecture

```
                    DPGMM Prior p(z|h)
                           ↓
        ┌──────────────────┴──────────────────┐
        │                                     │
    Stick-Breaking                    Component Parameters
        π(h)                              μ_k(h), σ²_k(h)
        │                                     │
    ┌───┴───┐                            ┌────┴────┐
    │       │                            │         │
 Kumara-  Gamma                       Neural     Neural
  swamy  Posterior                    Network    Network
Network   (α)                         (means)  (log-vars)
```

### Mathematical Formulation

The DPGMM prior is a mixture of Gaussians:

```
p(z|h) = Σ_{k=1}^K π_k(h) · N(z | μ_k(h), σ²_k(h))
```

where:
- **π(h)**: Mixing weights from stick-breaking
- **μ_k(h)**: Component means from neural network
- **σ²_k(h)**: Component variances from neural network
- **h**: Hidden state providing context

#### Stick-Breaking Construction

Mixing weights are generated via stick-breaking:

```
π_k = v_k ∏_{j<k} (1 - v_j)    where v_k ~ Kumaraswamy(a_k(h), b_k(h))
```

This ensures:
- π_k ≥ 0 for all k
- Σ_k π_k = 1
- Adaptive allocation (larger π for important components)

---

## Components

### 1. `DPGMMPrior` (`dpgmm_prior.py`)

Main class implementing the DPGMM prior distribution.

**Key Methods**:
- `forward(h)`: Generate mixture distribution from hidden state
- `compute_kl_divergence_mc(...)`: Monte Carlo KL estimation
- `compute_kl_loss(...)`: KL divergence for stick-breaking variables
- `get_effective_components(pi)`: Count active components

**Usage**:
```python
from generative_prior import DPGMMPrior

prior = DPGMMPrior(
    max_components=15,
    latent_dim=36,
    hidden_dim=512,
    device=torch.device('cuda')
)

# Generate prior from hidden state
mixture_dist, params = prior(h_t)

# Sample from prior
z_prior = mixture_dist.sample()  # [batch_size, latent_dim]

# Compute KL divergence with posterior
kl = prior.compute_kl_divergence_mc(
    posterior_mean,
    posterior_logvar,
    params,
    n_samples=10
)
```

### 2. `AdaptiveStickBreaking` (`stick_breaking.py`)

Implements stick-breaking construction for generating mixture weights.

**Features**:
- Kumaraswamy-based stick-breaking (numerically stable)
- Random permutation for permutation invariance
- Adaptive truncation (prunes small components)
- Variational posterior over concentration parameter α

**Usage**:
```python
from generative_prior import AdaptiveStickBreaking

stick_breaking = AdaptiveStickBreaking(
    max_components=15,
    hidden_dim=512,
    device=torch.device('cuda')
)

# Generate mixing weights
pi, info = stick_breaking(h_t)

print(f"Weights: {pi.shape}")  # [batch_size, 15]
print(f"Active components: {info['active_components']}")
```

### 3. `GammaPosterior` (`distributions/gamma_posterior.py`)

Variational posterior for Gamma-distributed concentration parameter.

**Usage**:
```python
from generative_prior.distributions import GammaPosterior

gamma_post = GammaPosterior(hidden_dim=512, device=torch.device('cuda'))

# Generate Gamma parameters
concentration, rate = gamma_post(h_t)

# Sample concentration parameter
alpha = gamma_post.sample(h_t, n_samples=10)

# Compute KL with prior
kl = gamma_post.kl_divergence(h_t, prior_concentration=1.0, prior_rate=1.0)
```

### 4. `KumaraswamyStable` (`distributions/Kumaraswamy.py`)

Numerically stable implementation of Kumaraswamy distribution.

**Properties**:
- Reparameterization trick for gradient estimation
- Log-space computations for stability
- Support on [0, 1] (ideal for stick-breaking)

---

## Tensor Shape Reference

| Component | Input | Output | Description |
|-----------|-------|--------|-------------|
| **DPGMMPrior** | | | |
| `forward()` | h: `[B, hidden_dim]` | mixture: `MixtureSameFamily` | Generate prior distribution |
|  |  | params['pi']: `[B, K]` | Mixing weights |
|  |  | params['means']: `[B, K, D]` | Component means |
|  |  | params['log_vars']: `[B, K, D]` | Component log-variances |
| `compute_kl_divergence_mc()` | μ_q: `[B, D]` | kl: `scalar` | Monte Carlo KL estimation |
|  | log σ²_q: `[B, D]` |  |  |
| **AdaptiveStickBreaking** | | | |
| `forward()` | h: `[B, hidden_dim]` | π: `[B, K]` | Mixing weights |
|  |  | info: `Dict` | Auxiliary information |
| **GammaPosterior** | | | |
| `forward()` | h: `[B, hidden_dim]` | α: `[B]`, β: `[B]` | Concentration, rate |
| `sample()` | h: `[B, hidden_dim]` | α: `[B, n_samples]` | Sampled concentration |

**Legend**: B=batch, K=max_components, D=latent_dim

---

## Integration with VRNN

The DPGMM prior integrates with the VRNN model as follows:

```python
# In VRNN forward pass (dpgmm_stickbreaking_prior_vrnn.py)

# 1. Generate prior from hidden state
prior_dist, prior_params = self.dpgmm_prior(h_t)

# 2. Generate posterior from observations
posterior_mean, posterior_logvar = self.encoder(x_t, context_t)

# 3. Sample latent variable
z_t = posterior_mean + torch.exp(0.5 * posterior_logvar) * eps

# 4. Compute KL divergence
kl_div = self.dpgmm_prior.compute_kl_divergence_mc(
    posterior_mean,
    posterior_logvar,
    prior_params,
    n_samples=10
)

# 5. Add stick-breaking KL
kl_stickbreaking = self.dpgmm_prior.compute_kl_loss(
    prior_params,
    prior_params['alpha'],
    h_t
)

total_kl = kl_div + kl_stickbreaking
```

---

## Testing

### Test Scripts

Run the test scripts to verify functionality:

```bash
# Test DPGMM sampling
python generative_prior/tests/test_dpgmm_sampling.py

# Test stick-breaking construction
python generative_prior/tests/test_stick_breaking.py
```

### Expected Outputs

```
✓ DPGMM generates valid mixture distributions
✓ Mixing weights sum to 1.0
✓ KL divergence is non-negative
✓ Effective components adapt to context
✓ Stick-breaking produces valid proportions
```

---

## Design Principles

### 1. Context-Dependent Beliefs

Everything adapts based on hidden state `h_t`:
- Mixing weights: `π(h_t)` via neural networks
- Component means: `μ_k(h_t)` via neural networks
- Component variances: `σ²_k(h_t)` via neural networks

This allows the prior to:
- Focus on relevant modes given current context
- Allocate capacity to important components
- Adapt belief structure over time

### 2. Numerical Stability

The implementation prioritizes numerical stability:
- Log-space computations for small probabilities
- Clamping of log-variances to prevent overflow
- Epsilon additions to prevent division by zero
- NaN/Inf detection and handling

### 3. Flexible Capacity

The Dirichlet Process automatically allocates capacity:
- Active components determined by data
- Small mixing weights → pruned components
- `get_effective_components()` counts active modes
- Adaptive truncation reduces computational cost

---

## Theory: Why DPGMM for Active Inference?

### Connection to Active Inference

Active inference posits that the brain maintains **beliefs** over latent states of the world:

```
p(z|o, h) ∝ p(o|z) · p(z|h)
           ↑        ↑
      likelihood  prior
```

The DPGMM prior `p(z|h)` implements:

1. **Context-dependent beliefs**: Prior adapts based on current state `h_t`
2. **Multi-hypothesis tracking**: Multiple Gaussian components = multiple beliefs
3. **Precision weighting**: Mixture weights act as confidence in each hypothesis
4. **Hierarchical inference**: DP concentration parameter controls model complexity

### Comparison to Standard Gaussian Prior

| Property | Gaussian N(0,I) | DPGMM p(z\|h) |
|----------|----------------|---------------|
| **Modality** | Unimodal | Multi-modal |
| **Adaptation** | Fixed | Context-dependent |
| **Capacity** | Fixed dimensionality | Adaptive components |
| **Expressiveness** | Limited | Rich |
| **Cognitive Interpretation** | Single belief | Multiple hypotheses |

### Benefits for World Modeling

1. **Ambiguity resolution**: Multiple modes capture alternative interpretations
2. **Temporal consistency**: Prior adapts smoothly based on hidden state
3. **Automatic regularization**: DP prunes unnecessary complexity
4. **Interpretability**: Component weights indicate confidence in beliefs

---

## File Structure

```
generative_prior/
├── __init__.py              # Public API exports
├── README.md                # This file
├── dpgmm_prior.py           # Main DPGMM implementation
├── stick_breaking.py        # Adaptive stick-breaking
├── distributions/           # Distribution utilities
│   ├── __init__.py
│   ├── gamma_posterior.py   # Gamma variational posterior
│   └── Kumaraswamy.py       # Stable Kumaraswamy distribution
└── tests/                   # Test scripts
    ├── test_dpgmm_sampling.py
    └── test_stick_breaking.py
```

---

## References

### Theoretical Foundations

1. **Ferguson, T. S. (1973)**: "A Bayesian analysis of some nonparametric problems"
   - Original Dirichlet Process paper

2. **Sethuraman, J. (1994)**: "A constructive definition of Dirichlet priors"
   - Stick-breaking construction

3. **Nalisnick, E., & Smyth, P. (2017)**: "Stick-breaking variational autoencoders"
   - DPGMM-VAE methodology

4. **Jones, M. C. (2009)**: "Kumaraswamy's distribution: A beta-type distribution"
   - Kumaraswamy distribution properties

### Implementation References

5. **Nalisnick et al. (2016)**: "Infinite variational autoencoder for semi-supervised learning"
   - Practical implementation details

6. **PyTorch VAE**: https://github.com/threewisemonkeys-as/PyTorch-VAE
   - Gamma posterior implementation

---

## Future Extensions

### Potential Improvements

1. **Hierarchical Dirichlet Process**: Multi-level belief hierarchies
2. **Dependent Dirichlet Process**: Temporal smoothness in mixture evolution
3. **Student-t components**: Heavy-tailed distributions for robustness
4. **Hyperbolic embeddings**: Hierarchical latent structures

### Research Directions

1. **Interpretability**: Visualize component activations over time
2. **Ablation studies**: Compare DPGMM vs. Gaussian prior
3. **Component analysis**: What do different mixture components represent?
4. **Capacity analysis**: How many components are actually used?

---

*PILLAR 2: REPRESENTATION complete - Adaptive infinite mixture priors operational* ✓
