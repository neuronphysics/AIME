"""SHS-RSSM: a finite-truncation recurrent-persistence HDP switching state-space prior
for DreamerV3.

Generative form (shared-carry, recurrent mode):
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])            # Dreamer deterministic state
    beta ~ GEM(gamma);  pi_i ~ Dir(alpha*beta + kappa*e_i)
    recurrent mode sets kappa = 0 and gets stickiness from a learned, input-dependent
    per-regime persistence  rho_{t,i} = sigmoid(w_i . phi(h_t) + b_i), so
        p(s_t = j | s_{t-1} = i, h_t) = rho_{t,i} 1[j=i] + (1 - rho_{t,i}) pi_ij
    z_t = M_{s_t} r_t + C P h_t + U_{s_t} f_t + eps,  r_t = [z_{t-1}, a_{t-1}, 1]
    eps ~ N(0, diag(1/tau_{s_t}));  q_rank>0 adds the low-rank U U^T term (q_rank=0 omits it)

Notes: this is a FINITE truncation of the HDP (Kmax slots + an active mask), not an
infinite DP; the E-step is analytic (forward-backward with a Polya-Gamma / Jaakkola-Jordan
bound for the logistic persistence) -- no Monte Carlo in the switching head.

Modules:
  regimes.py / regimes_shared.py - per-regime Bayesian linear-Gaussian dynamics
                                   (Normal-Gamma; optional low-rank-plus-diagonal noise)
  sticky_hdp.py       - HDP-HMM transition q(pi) and stick-breaking root q(beta), kappa branch
  recurrent_stick.py  - input-dependent Bernoulli persistence with the analytic PG/JJ bound
  forward_backward.py - HMM forward-backward: responsibilities, counts, pairwise, messages
  mixture_prior.py    - mixture prior + Gaussian-to-mixture KL for the Dreamer loss
  moves.py            - birth / merge / delete / split structural moves (adaptive K)
  online_vb.py        - EMA / streaming / memoized sufficient-statistic stores
  shs_rssm.py         - RSSM subclass wiring the above into dreamerv3-torch
"""

__version__ = "0.1.0"