
import os, sys, types, importlib.util, pathlib, torch, random
import numpy as np
import pytest

# Add repo root to path
REPO_ROOT = pathlib.Path("/media/zsheikhb/29cd0dc6-0ccb-4a96-8e75-5aa530301a7e/home/zahra/Work/prgress")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Create VRNN.* alias modules so dpgmm_stickbreaking_prior_vrnn imports resolve ---
alias_map = {
    "VRNN.Kumaraswamy": REPO_ROOT / "Kumaraswamy.py",
    "VRNN.mtl_optimizers": REPO_ROOT / "mtl_optimizers.py",
    "VRNN.perceiver.Utils": REPO_ROOT / "Utils.py",
    "VRNN.perceiver.perceiver": REPO_ROOT / "perceiver.py",
    "VRNN.perceiver.perceiver_helpers": REPO_ROOT / "perceiver_helpers.py",
}

def load_as_module(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

for mname, fpath in alias_map.items():
    try:
        if mname not in sys.modules:
            load_as_module(mname, fpath)
    except FileNotFoundError:
        pass  # Some aliases may not be needed in certain tests

# Determinism
def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.fixture(autouse=True)
def _seed():
    set_seed()

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small synthetic batch of image sequences: [B, T, C, H, W]
@pytest.fixture
def fake_video_batch(device):
    B, T, C, H, W = 2, 4, 3, 64, 64
    x = torch.rand(B, T, C, H, W, device=device)
    a = torch.randn(B, T, 8, device=device)  # small dummy action space
    return x, a

# Path to "real" DMC-VB data if available (user-provided). Set env DMC_VB_DIR to the folder containing TFRecords or processed npz.
@pytest.fixture(scope="session")
def dmc_vb_dir():
    p = os.environ.get("DMC_VB_DIR")
    if p and os.path.isdir(p):
        return p
    return None

def require_data_available():
    if not os.environ.get("DMC_VB_DIR"):
        pytest.skip("Set DMC_VB_DIR to run dataset-based tests on the real DMC-VB dataset.")
