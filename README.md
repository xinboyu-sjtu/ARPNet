# ARPNet

Official PyTorch implementation of **ARPNet**, a two-stage deep learning model for
predicting alveolar ridge morphology and the need for ancillary bone augmentation
(NBA) after alveolar ridge preservation (ARP) from preoperative CBCT.

> **Deep Learning-Based Outcome Prediction after Alveolar Ridge Preservation:
> A Multicenter Development and Validation Study**
>
> Xinbo Yu, Haitao Li, Ziang Wu, Hanqi Gao, Qi Yan, Dongning He, Zhen Fan,
> Dayong Zhang, Feng Wang, Xiaojun Chen, Yiqun Wu
>
> *These authors contributed equally: Xinbo Yu, Haitao Li.*
>
> Corresponding authors: Yiqun Wu, Xiaojun Chen, Feng Wang.

This repository contains the model, training, inference and evaluation code for
the paper. It deliberately excludes the ARPFlow generative-model experiments,
baseline comparisons and other exploratory scripts. The complete two-stage
network is named **ARPNet** throughout the code:

- `ARPNetStageI` — alveolar ridge morphology prediction.
- `ARPNetStageII` — three-class NBA classification.
- `ARPNet` — end-to-end wrapper that runs Stage I and Stage II together.

## Model overview

ARPNet is trained sequentially. Stage I is trained first; its parameters are
then frozen and Stage II is trained on the Stage I predictions.

### Stage I: alveolar ridge morphology prediction

**Input:** a cropped preoperative CBCT mask containing the extraction socket
and 1–2 mm of surrounding bone.
**Output:** the predicted healed post-ARP mask at the same location.

Stage I uses three parallel branches operating at full, 2/3 and 1/3 scale.
Each branch is a U-Mamba-style encoder–decoder: residual convolutional stages
are interleaved with Mamba state-space layers, and skip connections recover the
fine spatial detail. The branch outputs are fused coarse-to-fine with 1×1
convolutions, added to a global residual, and passed through a final 1×1
refiner. The paper uses mean squared error (MSE) as the Stage I loss.

The axial dimension of the volume is used as the channel dimension of the 2D
convolutional backbone. `ARPNetStageI` accepts a standard
`[B, 1, D, H, W]` tensor and returns `[B, 1, D, H, W]`. The cubic size must be
divisible by `down_scale × 16` so that all scale branches have consistent
feature-map sizes; all experiments in the paper use `192³` with
`down_scale=3`.

### Stage II: need for ancillary bone augmentation

**Inputs:** the preoperative CBCT mask and the Stage I predicted post-ARP mask.
**Output:** three NBA classes:

| Class | Definition |
| --- | --- |
| 0 | Implant placement without NBA |
| 1 | Implant placement with simultaneous bone augmentation and/or osteotome sinus floor elevation (OSFE) |
| 2 | Implant placement after complex bone augmentation (staged GBR or lateral sinus floor elevation) |

Two 3D ResNet-18 encoders extract a 512-d feature vector from each volume.
The vectors are concatenated (1024-d) and passed through a fully connected
head (`1024 → 256 → 3`). The paper uses MSE on the softmax probabilities
(the multi-class Brier score) as the Stage II loss; weighted cross-entropy is
also available with `--loss ce` for the variant used in some internal runs.

## Repository layout

```
ARPNet/
├── arpnet/
│   ├── models/
│   │   ├── blocks.py       # Mamba layer, residual blocks, scale encoder/decoder
│   │   ├── stage1.py       # ARPNetStageI
│   │   ├── stage2.py       # ARPNetStageII and 3D ResNet encoder
│   │   └── arpnet.py       # end-to-end ARPNet wrapper
│   ├── data/
│   │   ├── dataset.py      # paired preoperative/postoperative dataset
│   │   └── preprocessing.py # NIfTI resampling, centre cropping, largest component
│   ├── losses.py           # MSE, Dice, BCE, Brier, weighted CE
│   ├── metrics.py          # segmentation and classification metrics
│   └── utils.py            # seeding, logging, checkpointing
├── scripts/
│   ├── prepare_crops.py    # crop registered mask pairs to 192^3
│   ├── train_stage1.py     # train Stage I
│   ├── train_stage2.py     # train Stage II with frozen Stage I
│   ├── predict.py          # end-to-end inference
│   └── evaluate.py         # aggregate metrics and bootstrap CIs
├── examples/
│   └── manifest_example.csv
├── checkpoints/            # place trained weights here (not included)
├── requirements.txt
└── README.md
```

## Installation

The code was developed and tested with:

- Python 3.10
- PyTorch 2.1/2.2 with CUDA 11.8
- MONAI 1.3
- `mamba-ssm` 1.1.x and `causal-conv1d` 1.1.x
- A 24 GB GPU (e.g. RTX 3090) for 192³ volumes

```bash
conda create -n arpnet python=3.10 -y
conda activate arpnet

# Install a CUDA-enabled PyTorch build first.
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` compile CUDA extensions. If the build fails,
install the prebuilt wheels that match your PyTorch/CUDA ABI:

```bash
pip install causal_conv1d-*.whl --no-deps
pip install mamba_ssm-*.whl --no-deps
```

A common failure mode is an ABI mismatch between PyTorch 2.1 and 2.2
(`cxx11abiTRUE` vs `cxx11abiFALSE`). Check the wheel filename against
`torch._C._GLIBCXX_USE_CXX11_ABI` before installing.

## Data preparation

### Expected input

Each case consists of a registered preoperative/postoperative mask pair. The
masks must be binary and have the same orientation, spacing and field of view.
The final volumes are cubic with size `192 × 192 × 192` in `(D, H, W)` order,
matching `SimpleITK.GetArrayFromImage`.

The paper's preprocessing uses the following steps in 3D Slicer:

1. Jaw and tooth segmentation with the DentalSegmentator module (nnU-Net).
2. Virtual extraction of the segmented teeth from the pre- and postoperative
   CBCTs.
3. Surface registration of the pre- and postoperative jaws.
4. Cropping of the extraction socket with 1–2 mm of surrounding bone, using
   the same bounding box for both time points.
5. Resampling to an isotropic spacing and padding/cropping to `192³`.

`scripts/prepare_crops.py` performs step 5 (and optional isotropic resampling)
once the registered mask pairs are available.

### Manifest format

All scripts read a CSV manifest:

```csv
case_id,pre_path,post_path,label,split
case_001,data/case_001/pre.nii.gz,data/case_001/post.nii.gz,0,train
case_002,data/case_002/pre.nii.gz,data/case_002/post.nii.gz,1,train
case_003,data/case_003/pre.nii.gz,data/case_003/post.nii.gz,2,val
case_004,data/case_004/pre.nii.gz,data/case_004/post.nii.gz,0,test
```

- `case_id`: unique case identifier.
- `pre_path`, `post_path`: paths to the preoperative and postoperative masks.
  Relative paths are resolved relative to the manifest file.
- `label`: NBA class (`0`, `1`, `2`). Required for training; optional for
  inference.
- `split`: `train`, `val` or `test`.

A template is provided in `examples/manifest_example.csv`.

### Cropping helper

```bash
python scripts/prepare_crops.py \
    --manifest data/raw_manifest.csv \
    --out_dir data/crops \
    --size 192 \
    --spacing 0.2
```

The script writes `pre_<case_id>.nii.gz`, `post_<case_id>.nii.gz` and a new
`data/crops/manifest.csv`. Omit `--spacing` if the images are already
resampled.

## Training

### Stage I

```bash
python scripts/train_stage1.py \
    --manifest data/crops/manifest.csv \
    --out_dir runs/stage1 \
    --epochs 200 \
    --batch_size 1 \
    --lr 1e-4 \
    --loss mse
```

The best checkpoint (selected by validation Dice) is written to
`runs/stage1/best.pth`. The training history and resolved arguments are saved
alongside it.

For the loss-function comparison reported in the revision:

```bash
python scripts/train_stage1.py ... --loss dice --out_dir runs/stage1_dice
python scripts/train_stage1.py ... --loss bce  --out_dir runs/stage1_bce
```

MSE uses the raw network output and thresholds at 0.5. Dice and BCE use a
sigmoid before thresholding.

### Stage II

```bash
python scripts/train_stage2.py \
    --manifest data/crops/manifest.csv \
    --stage1_ckpt runs/stage1/best.pth \
    --out_dir runs/stage2 \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-5 \
    --loss mse
```

Stage I is frozen and runs in evaluation mode. The best checkpoint is selected
by validation balanced accuracy and written to `runs/stage2/best.pth`.

Optional pretrained encoders (for example, encoders pretrained on a related
segmentation task) can be supplied with:

```bash
python scripts/train_stage2.py ... \
    --encoder1_ckpt checkpoints/resnet3d_post.pth \
    --encoder2_ckpt checkpoints/resnet3d_pre.pth \
    --freeze_encoders
```

`encoder1` processes the generated postoperative mask; `encoder2` processes
the preoperative volume. This order matches the original ARPNet checkpoints.

## Inference

```bash
python scripts/predict.py \
    --manifest data/crops/manifest.csv \
    --split test \
    --stage1_ckpt runs/stage1/best.pth \
    --stage2_ckpt runs/stage2/best.pth \
    --out_dir predictions \
    --eval_stage1 \
    --save_masks
```

Outputs:

- `predictions/predictions.csv`: case-level Stage II probabilities, predicted
  labels, and (with `--eval_stage1`) Stage I Dice/IoU/precision/recall and
  voxel counts.
- `predictions/masks/<case_id>_pred.nii.gz`: predicted postoperative masks
  (with `--save_masks`).

Use `--keep_largest` to retain only the largest connected component of each
predicted mask. Use `--stage1_loss dice` or `--stage1_loss bce` if Stage I was
trained with a sigmoid output; the script also reads the loss name from the
Stage I checkpoint metadata when available.

## Evaluation

```bash
python scripts/evaluate.py \
    --predictions predictions/predictions.csv \
    --out predictions/metrics.json \
    --bootstrap 1000
```

The script reports:

**Stage I:** per-case Dice, mIoU, precision and recall (mean ± SD), pooled
Dice/IoU, and a percentile bootstrap 95% CI for Dice.

**Stage II:** accuracy, balanced accuracy, macro precision/recall/F1, macro
one-vs-rest AUROC, and the multi-class Brier score. Bootstrap 95% CIs are
computed for accuracy, balanced accuracy, macro F1 and AUROC. If a class is
absent from a resampled cohort, AUROC is computed over the classes present.

## Reproducibility notes

- Splits are patient-level and are supplied through the `split` column of the
  manifest; the code does not create splits automatically.
- The default seed is 42. `set_seed` controls Python, NumPy and PyTorch.
- Class-balanced oversampling used in the original training pipeline can be
  enabled with `--balance`. Validation is never balanced.
- Stage I input is mapped to `{-1, +1}`; Stage II input is the same
  preoperative volume and a binary Stage I mask thresholded at 0.5.
- The paper uses MSE for both stages. `--loss ce` in Stage II enables the
  weighted cross-entropy variant (`1.0`, `2.0`, `4.0` for classes 0–2).
- Training Stage I on 192³ volumes with batch size 1 requires a GPU with at
  least 24 GB of memory. Reduce the volume size only if the preprocessing is
  adjusted consistently.

## Pretrained weights

Trained weights are not distributed with this repository. See
`checkpoints/README.md` for the expected filenames and loading instructions.
The public module definitions are checkpoint-compatible with the models used
for the paper: the Stage I and Stage II state dicts load strictly into
`ARPNetStageI` and `ARPNetStageII`, respectively.

## Citation

```bibtex
@article{yu2026arpnet,
  title   = {Deep Learning-Based Outcome Prediction after Alveolar Ridge
             Preservation: A Multicenter Development and Validation Study},
  author  = {Yu, Xinbo and Li, Haitao and Wu, Ziang and Gao, Hanqi and
             Yan, Qi and He, Dongning and Fan, Zhen and Zhang, Dayong and
             Wang, Feng and Chen, Xiaojun and Wu, Yiqun},
  year    = {2026}
}
```

## Ethics and data availability

The study protocol was approved by the Institutional Review Boards of Shanghai
Ninth People's Hospital (SH9H-2025-T473-1) and the University of Hong
Kong/Hospital Authority Hong Kong West Cluster (UW 25-577). Patient data
cannot be shared publicly; requests should be addressed to the corresponding
authors.

## Funding

This study was supported by the Health and Medical Research Fund, the Shanghai
Oriental Talent Leadership Program (LJ2024091), the AI + Oral and
Craniomaxillofacial Implantology Micro-Specialty Program (DGD-2025-2), and the
AI Interdisciplinary Research Special Program of Shanghai Ninth People's
Hospital Affiliated to Shanghai Jiao Tong University School of Medicine
(JYJC2025006).

## Contact

- Yiqun Wu — yiqunwu@hotmail.com
- Xiaojun Chen — xiaojunchen@sjtu.edu.cn
- Feng Wang — diawang@hku.hk

## License

This code is released for academic research use. Commercial use requires
permission from the corresponding authors.
