# Model weights

The trained checkpoints are distributed as GitHub Release assets because each
file is larger than GitHub's 100 MB per-file limit.

| File | Description | Size | SHA-256 |
| --- | --- | --- | --- |
| `ARPNetStageI.pth` | Stage I morphology-prediction network (`ARPNetStageI`) | 260.3 MiB | `e9aafd58e5b4e337140e34f2c77fce6ae782fb15933cd038c769fc7c04461450` |
| `ARPNetStageII_multimodal.pth` | Multimodal Stage II classifier (`ARPNetStageII`) | 259.4 MiB | `9b1eefdb5084f701c73d6fed77ba4ac74d3107350c428fd4070cd3c7641560d3` |
| `ARPNetStageII.pth` (legacy) | Image-only Stage II model without clinical variables | 254.2 MiB | `01c2c88d622afb81d29e60441654c71fe16529609f02ff6f0ba48a7e17e748b8` |

`ARPNetStageII_multimodal.pth` is the current multimodal model. The legacy
`ARPNetStageII.pth` is kept for the earlier image-only release and is not used
by the multimodal code path.

## Download

```bash
mkdir -p checkpoints

# Stage I
curl -L -o checkpoints/ARPNetStageI.pth \
  https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.0.0/ARPNetStageI.pth

# Multimodal Stage II
curl -L -o checkpoints/ARPNetStageII_multimodal.pth \
  https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.1.0/ARPNetStageII_multimodal.pth
```

Verify the downloads with:

```bash
sha256sum checkpoints/ARPNetStageI.pth checkpoints/ARPNetStageII_multimodal.pth
```

## Loading

The multimodal Stage II file is a payload containing the state dict and the
clinical-encoding metadata:

```python
import torch
from arpnet.models.stage1 import ARPNetStageI
from arpnet.models.stage2 import ARPNetStageII

stage1 = ARPNetStageI(image_size=192, down_scale=3)
stage1.load_state_dict(torch.load("checkpoints/ARPNetStageI.pth", map_location="cpu"))

payload = torch.load(
    "checkpoints/ARPNetStageII_multimodal.pth", map_location="cpu"
)
stage2 = ARPNetStageII(clinical_dim=payload["clinical_dim"])
stage2.load_state_dict(payload["state_dict"])

print(payload["clinical_age_mean"], payload["clinical_age_std"])
```

`ARPNetStageI.pth` is a raw state dict. The scripts accept both raw state dicts
and payloads containing a `state_dict` key; `arpnet.utils.load_checkpoint`
handles both formats.
