# Model weights

Trained weights are not included in this repository. The following files are
expected by the training and inference scripts when pretrained components are
used:

| File | Description |
| --- | --- |
| `stage1_best.pth` | Stage I morphology-prediction network (written as `runs/stage1/best.pth` by `train_stage1.py`) |
| `stage2_best.pth` | Stage II NBA classifier (written as `runs/stage2/best.pth` by `train_stage2.py`) |
| `resnet3d_pre.pth` | Optional pretrained 3D ResNet-18 encoder for the preoperative volume |
| `resnet3d_post.pth` | Optional pretrained 3D ResNet-18 encoder for the generated postoperative mask |

Place the files in this directory (or pass their paths explicitly with
`--stage1_ckpt`, `--stage2_ckpt`, `--encoder1_ckpt` and `--encoder2_ckpt`).

The checkpoints produced by `scripts/train_stage1.py` and
`scripts/train_stage2.py` use the same module names as
`arpnet.models.stage1.ARPNetStageI` and `arpnet.models.stage2.ARPNetStageII`,
so they can be loaded with `model.load_state_dict(torch.load(path)["state_dict"])`
or with `arpnet.utils.load_checkpoint`.
