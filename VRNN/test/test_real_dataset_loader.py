
import os, pytest, torch
from dmc_vb_transition_dynamics_trainer import DMCVBTrainer, DMCVBInfo

@pytest.mark.integration
def test_real_dataset_loader_smoke(dmc_vb_dir, device):
    if dmc_vb_dir is None:
        pytest.skip("Set DMC_VB_DIR to run dataset tests.")
    # minimal trainer init with dummy model that only queries dataset
    class Dummy(torch.nn.Module):
        def __init__(self): super().__init__()
    cfg = dict(
        domain_name="humanoid",
        task_name=DMCVBInfo.DMC_INFO["humanoid"]["task_name"],
        image_size=64,
        sequence_length=4,
        frame_stack=3,
        batch_size=2,
        num_workers=0,
    )
    trainer = DMCVBTrainer(model=Dummy(), data_dir=dmc_vb_dir, config=cfg, device=device)
    batch = next(iter(trainer.train_loader))
    assert "observations" in batch and "actions" in batch
    x = batch["observations"]
    a = batch["actions"]
    assert x.ndim == 5 and a.ndim in (2,3)
