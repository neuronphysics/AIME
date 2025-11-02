# Session 2: Perceiver IO Extraction - Implementation Plan

**Status**: Ready to Execute
**Estimated Time**: 2-3 hours
**Risk Level**: Low (code movement only, no logic changes)

---

## Progress So Far ✅

### Completed
- [x] Created `perceiver_io/` directory structure
- [x] Created `perceiver_io/__init__.py` with clean exports
- [x] Created `perceiver_io/README.md` with comprehensive theory documentation
- [x] Created `perceiver_io/modules/` subdirectory
- [x] Created `perceiver_io/tests/` subdirectory

---

## Next Steps (Ready to Execute)

### Step 1: Copy Utility Modules (No Modification)

Move existing utility files from `VRNN/perceiver/` to `perceiver_io/modules/`:

```bash
# These files are already complete, just copy them
cp VRNN/perceiver/vector_quantize.py perceiver_io/modules/
cp VRNN/perceiver/position.py perceiver_io/modules/
cp VRNN/perceiver/adapter.py perceiver_io/modules/
cp VRNN/perceiver/utilities.py perceiver_io/modules/
cp VRNN/perceiver/modules.py perceiver_io/modules/  # Will split later
```

Create `perceiver_io/modules/__init__.py`:
```python
from .vector_quantize import VectorQuantize
from .position import RopePositionEmbedding
# etc.
```

---

### Step 2: Extract VQPTTokenizer

**From**: `VRNN/perceiver/video_prediction_perceiverIO.py` lines 1-360

**To**: `perceiver_io/tokenizer.py`

**Classes to extract**:
- `ResBlock3D` (lines ~1-50)
- `Down3D`, `Up3D` (lines ~50-150)
- `UNetEncoder3D`, `UNetDecoder3D` (lines ~150-360)
- `VQPTTokenizer` (lines ~361-528)

**Imports needed**:
```python
import torch
import torch.nn as nn
from einops import rearrange
from .modules.vector_quantize import VectorQuantize
```

**New file structure**:
```python
# perceiver_io/tokenizer.py

"""
VQ-VAE Tokenizer for Video Prediction

Converts video sequences to discrete token sequences using 3D UNet + VQ.
"""

# ... copy ResBlock3D, Down3D, Up3D classes
# ... copy UNetEncoder3D, UNetDecoder3D classes
# ... copy VQPTTokenizer class
```

---

### Step 3: Extract PerceiverTokenPredictor

**From**: `VRNN/perceiver/video_prediction_perceiverIO.py` lines 529-1116

**To**: `perceiver_io/predictor.py`

**Classes to extract**:
- `PerceiverTokenPredictor` (entire class)

**Imports needed**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import VQPTTokenizer
from .modules import PerceiverEncoder, PerceiverDecoder
```

---

### Step 4: Extract CausalPerceiverIO

**From**: `VRNN/perceiver/video_prediction_perceiverIO.py` lines 1117-1232

**To**: `perceiver_io/causal_perceiver.py`

**Classes to extract**:
- `CausalPerceiverIO` (entire class)

**Imports needed**:
```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .tokenizer import VQPTTokenizer
from .predictor import PerceiverTokenPredictor
```

**Key method** (for VRNN integration):
```python
@torch.no_grad()
def extract_context(self, videos: torch.Tensor) -> torch.Tensor:
    """
    Extract per-frame context vectors for VRNN.

    Args:
        videos: [B, T, C, H, W]

    Returns:
        context: [B, T, context_dim]
    """
    # Implementation stays the same
```

---

### Step 5: Update Imports in Main Model

**File**: `VRNN/dpgmm_stickbreaking_prior_vrnn.py`

**Change** (line ~26):
```python
# OLD
from VRNN.perceiver.video_prediction_perceiverIO import CausalPerceiverIO

# NEW
from perceiver_io import CausalPerceiverIO
```

**Verify** no other changes needed - the API stays the same!

---

### Step 6: Update Imports in Trainer

**File**: `VRNN/dmc_vb_transition_dynamics_trainer.py`

Check if it imports anything from perceiver - if so, update to:
```python
from perceiver_io import CausalPerceiverIO
```

---

### Step 7: Create Simple Test Scripts

**File**: `perceiver_io/tests/test_tokenizer.py`

```python
"""
Test: VQPTTokenizer

Demonstrates video → tokens → reconstructed video
"""
import torch
from perceiver_io import VQPTTokenizer

def test_tokenizer():
    # Small synthetic video
    B, T, C, H, W = 2, 4, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    print(f"Input video: {video.shape}")

    # Initialize tokenizer
    tokenizer = VQPTTokenizer(
        in_channels=3,
        code_dim=256,
        num_codes=256,  # Small for test
        downsample=4,
        base_channels=32,  # Small for test
        use_3d_conv=True,
        num_quantizers=1  # Single head for test
    )

    # Encode
    tokens, vq_loss, perplexity = tokenizer.encode(video)
    print(f"Tokens: {tokens.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")

    # Decode
    reconstructed = tokenizer.decode(tokens)
    print(f"Reconstructed: {reconstructed.shape}")

    # Verify shapes match
    assert reconstructed.shape == video.shape, "Shape mismatch!"
    print("✓ Test passed!")

if __name__ == "__main__":
    test_tokenizer()
```

**File**: `perceiver_io/tests/demo_perceiver_flow.py`

```python
"""
Demo: Full Perceiver IO Pipeline

Shows complete flow: video → context → prediction
"""
import torch
from perceiver_io import CausalPerceiverIO

def demo_perceiver():
    # Synthetic video
    B, T, C, H, W = 2, 8, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    print(f"Input video: {video.shape}")

    # Initialize Perceiver
    perceiver = CausalPerceiverIO(
        video_shape=(T, C, H, W),
        num_latents=64,  # Small for test
        num_latent_channels=128,  # Small for test
        num_codes=256,
        downsample=4
    )

    # Extract context (main use case for VRNN)
    context = perceiver.extract_context(video)
    print(f"Context: {context.shape}")
    assert context.shape == (B, T, 128), f"Expected [2, 8, 128], got {context.shape}"

    print("✓ Demo passed!")

if __name__ == "__main__":
    demo_perceiver()
```

---

### Step 8: Validation Checklist

Run these commands to verify everything works:

```bash
# 1. Test imports
python -c "from perceiver_io import CausalPerceiverIO, VQPTTokenizer; print('✓ Imports work')"

# 2. Run tokenizer test
python perceiver_io/tests/test_tokenizer.py

# 3. Run full demo
python perceiver_io/tests/demo_perceiver_flow.py

# 4. Check main model still imports correctly
python -c "from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder; print('✓ Main model imports work')"

# 5. (Optional) Run quick training test if feeling confident
python VRNN/dmc_vb_transition_dynamics_trainer.py --max-steps 5
```

---

## File Movement Summary

### New Files Created
```
perceiver_io/
├── __init__.py               ✓ DONE
├── README.md                 ✓ DONE
├── tokenizer.py              ← Extract from video_prediction_perceiverIO.py
├── predictor.py              ← Extract from video_prediction_perceiverIO.py
├── causal_perceiver.py       ← Extract from video_prediction_perceiverIO.py
├── modules/
│   ├── __init__.py           ← Create
│   ├── vector_quantize.py    ← Copy from VRNN/perceiver/
│   ├── position.py           ← Copy from VRNN/perceiver/
│   ├── adapter.py            ← Copy from VRNN/perceiver/
│   ├── utilities.py          ← Copy from VRNN/perceiver/
│   ├── encoder.py            ← Extract from VRNN/perceiver/modules.py
│   └── decoder.py            ← Extract from VRNN/perceiver/modules.py
└── tests/
    ├── test_tokenizer.py     ← Create
    ├── test_predictor.py     ← Create (optional)
    └── demo_perceiver_flow.py ← Create
```

### Files Modified
```
VRNN/dpgmm_stickbreaking_prior_vrnn.py  # Change line ~26 import
VRNN/dmc_vb_transition_dynamics_trainer.py  # Update imports if needed
```

### Files to Deprecate (Later)
```
VRNN/perceiver/  # Keep for now, delete after validation in Session 7
```

---

## Git Commit Strategy

After completing all steps and validation:

```bash
git add perceiver_io/
git add VRNN/dpgmm_stickbreaking_prior_vrnn.py
git add VRNN/dmc_vb_transition_dynamics_trainer.py
git commit -m "refactor(perceiver): Extract Perceiver IO to standalone module

- Create perceiver_io/ module with clean API
- Split video_prediction_perceiverIO.py (1232 lines) into:
  - tokenizer.py (VQPTTokenizer + UNet components)
  - predictor.py (PerceiverTokenPredictor)
  - causal_perceiver.py (CausalPerceiverIO wrapper)
- Move utility modules to perceiver_io/modules/
- Add comprehensive README with theory documentation
- Add test scripts for validation
- Update imports in VRNN main model

PILLAR 1: PERCEPTION - Sensory compression module complete
No logic changes, pure code reorganization
All tests pass ✓"
```

---

## Rollback Plan

If anything breaks:

```bash
# Single commit, easy to revert
git revert HEAD

# Or reset if not pushed
git reset --hard HEAD~1
```

---

## Time Estimates

| Task | Estimated Time |
|------|----------------|
| Copy utility modules | 10 min |
| Extract tokenizer.py | 20 min |
| Extract predictor.py | 20 min |
| Extract causal_perceiver.py | 15 min |
| Update imports | 10 min |
| Create test scripts | 20 min |
| Run validation | 15 min |
| Git commit | 5 min |
| **Total** | **~2 hours** |

Add 30-60 min buffer for unexpected issues.

---

## Success Criteria

- [ ] All files in `perceiver_io/` are < 500 lines
- [ ] `python -c "from perceiver_io import CausalPerceiverIO"` works
- [ ] Test scripts run without errors
- [ ] Main model imports successfully
- [ ] (Optional) Training runs for 5 steps without errors
- [ ] Git commit is clean and atomic

---

## Next Session Preview

**Session 3** will extract **PILLAR 2: Representation** (generative_prior/)
- Extract `DPGMMPrior` class
- Extract `AdaptiveStickBreaking` class
- Move `Kumaraswamy.py`
- Create sampling tests

---

**Ready to execute! Let me know when you want to proceed with the actual file extraction.**
