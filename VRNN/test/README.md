
# Tests for DPGMM-VRNN + Perceiver stack

## How to run

```bash
# (optional) activate your env
export PYTHONPATH=/mnt/data:$PYTHONPATH
pytest -q /mnt/data/tests -k "not integration"
# To include real dataset tests (requires local DMC-VB data):
export DMC_VB_DIR=/path/to/dmc_vb   # folder with TFRecords or converted npz
pytest -q /mnt/data/tests -m integration
```

## What’s covered

- `test_kumaraswamy.py`: correctness & gradients for KumaraswamyStable (rsample/log_prob).
- `test_stickbreaking.py`: sums-to-one, differentiability, effective components for AdaptiveStickBreaking.
- `test_temporal_discriminator.py`: shape/forward pass for latent-conditioned TemporalDiscriminator.
- `test_vrnn_forward.py`: end-to-end forward+loss+backprop smoke test for DPGMMVariationalRecurrentAutoencoder.
- `test_ablation_losses.py`: ablation-style check that turning a loss weight to 0 reduces grads in the targeted component group.
- `test_real_dataset_loader.py` (integration): exercises dataset loader using a real DMC-VB directory if available.

Each test is tiny/fast and will fail loudly on NaNs, wrong shapes, or broken imports.
