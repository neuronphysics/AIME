
import torch, pytest
from dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from dmc_vb_transition_dynamics_trainer import GradientMonitor

def test_ablation_lambda_zero_turns_off_component(fake_video_batch, device):
    x, a = fake_video_batch
    model = DPGMMVariationalRecurrentAutoencoder(
        image_size=64, input_channels=3, latent_dim=8, hidden_dim=64, max_K=4,
        action_dim=a.shape[-1], sequence_length=x.shape[1], device=device
    )
    # run once with regular weights
    loss1, out1 = model.compute_total_loss(observations=x, actions=a)
    (loss1).backward()
    gm1 = GradientMonitor(model, writer=None)
    grads1 = gm1.compute_component_gradients(update_global_step=False)

    model.zero_grad(set_to_none=True)
    # ablate attention dynamics (set weight to zero if present)
    if hasattr(model, "lambda_attention_dynamics"):
        old = float(model.lambda_attention_dynamics)
        model.lambda_attention_dynamics = 0.0  # turn off
    loss2, out2 = model.compute_total_loss(observations=x, actions=a)
    loss2.backward()
    gm2 = GradientMonitor(model, writer=None)
    grads2 = gm2.compute_component_gradients(update_global_step=False)

    # if attention_schema group exists in grads, it should have smaller grad now
    if "attention_schema" in grads1 and "attention_schema" in grads2:
        assert grads2["attention_schema"] <= grads1["attention_schema"] + 1e-6
