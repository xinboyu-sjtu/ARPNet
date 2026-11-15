# Model weights

The trained Stage I and Stage II checkpoints are distributed as assets of the
[`v1.0.0` release](https://github.com/xinboyu-sjtu/ARPNet/releases/tag/v1.0.0).
They are release assets rather than regular Git files because each checkpoint
is larger than GitHub's 100 MB per-file limit.

| File | Description | Size | SHA-256 |
| --- | --- | --- | --- |
| `ARPNetStageI.pth` | Stage I morphology-prediction network (`ARPNetStageI`) | 260.3 MiB | `e9aafd58e5b4e337140e34f2c77fce6ae782fb15933cd038c769fc7c04461450` |
| `ARPNetStageII.pth` | Stage II NBA classifier (`ARPNetStageII`) | 254.2 MiB | `01c2c88d622afb81d29e60441654c71fe16529609f02ff6f0ba48a7e17e748b8` |

## Download

```bash
mkdir -p checkpoints

curl -L -o checkpoints/ARPNetStageI.pth \
  https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.0.0/ARPNetStageI.pth

curl -L -o checkpoints/ARPNetStageII.pth \
  https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.0.0/ARPNetStageII.pth
```

Verify the downloads with:

```bash
sha256sum checkpoints/ARPNetStageI.pth checkpoints/ARPNetStageII.pth
```

## Loading

Both files are raw PyTorch state dicts. They load strictly into the public
module definitions:

```python
import torch
from arpnet.models.stage1 import ARPNetStageI
from arpnet.models.stage2 import ARPNetStageII

stage1 = ARPNetStageI(image_size=192, down_scale=3)
stage1.load_state_dict(torch.load("checkpoints/ARPNetStageI.pth", map_location="cpu"))

stage2 = ARPNetStageII(num_classes=3, in_channels=1, depth=18)
stage2.load_state_dict(torch.load("checkpoints/ARPNetStageII.pth", map_location="cpu"))
```

The scripts also accept checkpoints written by `train_stage1.py` and
`train_stage2.py` (which wrap the state dict in a `{"state_dict": ...}`
payload); `arpnet.utils.load_checkpoint` handles both formats.
