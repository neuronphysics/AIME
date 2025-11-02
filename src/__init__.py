"""
AIME Source Code

Core implementation of the AIME world model organized by the Five Pillars:

1. Perception (perceiver_io/) - Video tokenization and context extraction
2. Representation (encoder_decoder/, generative_prior/) - Hierarchical VAE and DPGMM prior
3. Dynamics (temporal_dynamics/) - Temporal state evolution
4. Attention (attention_schema/) - Slot-based attention mechanisms
5. Optimization (multi_task_learning/) - RGB gradient balancing and loss aggregation

Additional modules:
- world_model/ - Complete DPGMM-VRNN integration
- training/ - Training infrastructure and pipelines
- models.py - Shared architecture components (discriminators, transformers, utilities)
- nvae_architecture.py - Hierarchical VAE encoder/decoder implementation
"""
