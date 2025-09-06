
import torch, pytest
from models import TemporalDiscriminator

def test_temporal_discriminator_forward(device):
    B,T,C,H,W = 2, 4, 3, 64, 64
    x = torch.rand(B,T,C,H,W, device=device)
    z = torch.randn(B,T,32, device=device)
    disc = TemporalDiscriminator(input_channels=C, image_size=H, hidden_dim=128, n_layers=2, n_heads=2, max_sequence_length=T, patch_size=8, z_dim=32, device=device)
    y = disc(x, z=z)  # logits per patch per frame
    assert y.ndim in (2,3), "Output should be [B,T] or [B,T,patches] logits"
