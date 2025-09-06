
import torch, pytest

from dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder

def test_vrnn_forward_and_losses(fake_video_batch, device):
    x, a = fake_video_batch  # x: [B,T,C,H,W], a: [B,T,A]
    model = DPGMMVariationalRecurrentAutoencoder(
        image_size=64, input_channels=3, latent_dim=16, hidden_dim=64, max_K=6,
        action_dim=a.shape[-1], sequence_length=x.shape[1], device=device
    )
    # Compute total loss
    loss, out = model.compute_total_loss(observations=x, actions=a)
    # Basic keys present
    for k in ["recon_loss", "hierarchical_kl", "kl_z", "total_loss"]:
        assert k in out, f"Missing {k} in outputs"
    assert torch.isfinite(loss), "Loss should be finite"
    # Backprop smoke test
    loss.backward()
    for n,p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradients in {n}"
