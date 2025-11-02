"""
Generative Prior Module (PILLAR 2: Representation)

Implements adaptive infinite mixture priors for VAE latent space using
Dirichlet Process Gaussian Mixture Models (DPGMM).

Main Components:
- DPGMMPrior: Context-dependent mixture of Gaussians prior
- AdaptiveStickBreaking: Stick-breaking construction for mixture weights
- GammaPosterior: Variational posterior for concentration parameter
- KumaraswamyStable: Numerically stable Kumaraswamy distribution

Usage:
    from src.generative_prior import DPGMMPrior

    prior = DPGMMPrior(
        max_components=15,
        latent_dim=36,
        hidden_dim=512,
        device=torch.device('cuda')
    )

    # Generate prior distribution from hidden state
    mixture_dist, params = prior(hidden_state)

    # Compute KL divergence with posterior
    kl = prior.compute_kl_divergence_mc(
        posterior_mean, posterior_logvar, params, n_samples=10
    )

Theory:
    The DPGMM replaces standard N(0,I) prior with adaptive mixtures:

    p(z|h) = Σ_k π_k(h) N(z | μ_k(h), σ²_k(h))

    where all parameters are generated from hidden state h_t via:
    - π: Stick-breaking with Kumaraswamy distributions
    - μ, σ²: Neural networks conditioned on h

    This provides:
    1. Context-dependent beliefs (prior adapts to current state)
    2. Multi-modal representations (supports multiple hypotheses)
    3. Automatic capacity allocation (DP learns number of components)

References:
- Dirichlet Process: Ferguson (1973)
- Stick-breaking: Sethuraman (1994)
- DPGMM-VAE: Nalisnick & Smyth (2017)
"""

from .dpgmm_prior import DPGMMPrior
from .stick_breaking import AdaptiveStickBreaking, KumaraswamyNetwork
from .distributions import GammaPosterior, KumaraswamyStable

__all__ = [
    'DPGMMPrior',
    'AdaptiveStickBreaking',
    'KumaraswamyNetwork',
    'GammaPosterior',
    'KumaraswamyStable',
]

__version__ = '0.1.0'
