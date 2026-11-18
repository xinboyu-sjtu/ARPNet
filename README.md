# ARPNet

Official PyTorch implementation of **ARPNet**, a two-stage deep learning model for
predicting alveolar ridge morphology and the need for ancillary bone augmentation
(NBA) after alveolar ridge preservation (ARP) from pre-operative cone-beam computed tomography (CBCT) scans.

> **Deep Learning-Based Preoperative Prediction of Alveolar Ridge Morphology and Need for Ancillary Bone Augmentation after Alveolar Ridge Preservation:
> A Multicenter Development and Validation Study**
>
> Xinbo Yu, Haitao Li, Ziang Wu, Feng Wang, Hanqi Gao, Qi Yan, Dongning He, Zhen Fan,
> Dayong Zhang, Xiaojun Chen, Yiqun Wu
>
> *These authors contributed equally: Xinbo Yu, Haitao Li.*
>
> Corresponding authors: Yiqun Wu, Xiaojun Chen.

This repository contains the model, training, inference, and evaluation code for
the paper. It deliberately excludes the ARPFlow generative-model experiments,
baseline comparisons and other exploratory scripts. The complete two-stage
network is named **ARPNet** throughout the code:

- `ARPNetStageI` — Alveolar Ridge Morphology Prediction.
- `ARPNetStageII` — Multimodal Three-class NBA Classification Prediction.
- `ARPNet` — End-to-end wrapper that runs Stage I and Stage II together.

## Model overview

ARPNet is trained sequentially. Stage I is trained first; its parameters are
then frozen and Stage II is trained on the Stage I predictions.

## Model design and architecture

**Stage I (Morphology Prediction).** 

**Input:** 3D binary mask of the pre-extraction 
socket, 192×192×192 voxels. 

**Output:** The predicted healed post-ARP mask at the same location.

Three parallel branches process the input at full (192³), two-thirds (128³) 
and one-third (64³) resolution, each with an independent U-Mamba-based encoder. 
An attention mechanism within each branch enhances features of the centered 
socket and suppresses less relevant peripheral regions. 
Each branch downsamples four times, halving the resolution each time (full: 
192→96→48→24→12; two-thirds: 128→64→32→16→8; one-third: 64→32→16→8→4), with 
feature channels increasing 64→128→192→256. 
The branches are upsampled back to 192³ voxels and merged by 1×1 convolutions; 
a final 1×1 convolution outputs the predicted 3D mask of the post-ARP healed site.
The paper uses mean squared error (MSE) as the Stage I loss.

The axial dimension of the volume is used as the channel dimension of the 2D
convolutional backbone. `ARPNetStageI` accepts a standard
`[B, 1, D, H, W]` tensor and returns `[B, 1, D, H, W]`. The cubic size must be
divisible by `down_scale × 16` so that all scale branches have consistent
feature-map sizes; all experiments in the paper use `192³` with
`down_scale=3`.

**Stage II (NBA Classification).** 

**Inputs:** The pre-extraction socket mask and the 
Stage I-predicted post-ARP mask. Each is encoded by a 3D ResNet-18 (downsampling 
and global average pooling) into a 512-dimensional vector. The class with the highest 
softmax probability (Class I, II or III) is the prediction.

**Output:** Three NBA classes:

| Class | Definition |
| --- | --- |
| 0 | Implant placement without NBA |
| 1 | Implant placement with simultaneous bone augmentation and/or osteotome sinus floor elevation (OSFE) |
| 2 | Implant placement after complex bone augmentation (staged GBR or lateral sinus floor elevation) |

**Multimodal Stage II head (Representative Model).** Tooth type (molar, premolar, 
incisor, canine; 4 one-hot values), jaw (maxilla or mandible; 2 one-hot values), 
gender (1 value), and standardized age (1 value) are one-hot encoded and age is 
z-score normalized, giving an 8-dimensional clinical vector. 
A clinical branch (LayerNorm, Linear 8→64, ReLU) produces a 64-dimensional vector, 
which is concatenated with the 1024-dimensional imaging vector (1088 dimensions). 
A three-layer classifier (1088→1024→512→3, dropout 0.3) outputs the class scores. 
The final residual block of each image encoder is unfrozen during training; 
all other settings match the image-only model.

The released multimodal checkpoint was trained with weighted cross-entropy
(class weights 1:2:4 for classes 0–2).  `--loss mse` is also available and uses
the multi-class Brier score.

**Image-only Stage II head (Comparison Model).** The two 512-dimensional vectors are 
concatenated into a 1024-dimensional vector and passed through Linear (1024→256) + 
ReLU + dropout (0.1), then Linear (256→3), giving one score per NBA class.

**Stage II Class Balancing.** Random oversampling was applied to the Stage II training 
set only. Let N_c be the number of original training cases in class c (c = 0, 1, 2 for 
Class I, II, III), N_max = max(N_0, N_1, N_2), and m_c = N_max − N_c. All original cases 
were kept; for each class, m_c cases were drawn uniformly at random, independently and 
with replacement from that class's original cases and added to it. The resulting training 
set had N_max cases per class (3 × N_max in total; ratio 1:1:1). A given case in class c 
appeared 1 + K times, with K ~ Binomial(m_c, 1/N_c), so its expected number of appearances 
was N_max/N_c. In this study, N = 140, 62 and 22 for Classes I–III, so 78 Class II and 
118 Class III cases were added, giving 420 training cases. The balanced set was built once 
before training; each epoch reshuffled this fixed set into mini-batches, with no new 
oversampling per epoch. This is dataset-level duplication, not a class-weighted loss. 
Validation and test sets were not oversampled.

**Calibration.** True labels are one-hot encoded and predicted probabilities flattened 
across the three classes (micro-averaging); probabilities are binned into five equal-width 
bins and the observed fraction of positives is computed per bin (scikit-learn).

## Training settings

| Setting | Stage I | Stage II |
|---|---|---|
| Input | Cropped 3D mask of the pre-extraction socket | Same mask, Stage I-predicted post-ARP mask, and four clinical variables (tooth type, jaw, gender, age) |
| Output | 3D mask of the healed post-ARP site | Probabilities for the three NBA classes |
| Optimizer | Adam | Adam |
| Learning rate | 1×10⁻⁵ | 1×10⁻⁵ |
| Loss | Mean squared error | Mean squared error |
| Batch size | 1 | 4 |
| Epochs | Up to 200; early stopping (patience 50, validation loss) | Up to 200; early stopping (patience 50, validation loss) |
| Checkpoint | Lowest validation loss | Lowest validation loss |

Split: 6:2:2 (224:75:75 patients). Hardware: 2× NVIDIA RTX 3090. Environment: Python 3.10; 
PyTorch 2.1.1 with CUDA 11.8; MONAI 1.4.0; mamba-ssm 1.1.1; causal-conv1d 1.1.1; 
dynamic-network-architectures ≥0.2; NumPy ≥1.24; pandas ≥2.0; SciPy ≥1.10; scikit-learn ≥1.3; 
tqdm ≥4.66; and SimpleITK ≥2.3.

## Size distribution of the post-extraction socket (cropped training images, n = 224)

The socket was defined as the postoperative bone region absent from the preoperative mask. 
All values are percentages of the cropped image.

| Measure | Mean ± SD (%) | Median (%) | IQR (%) | 5th–95th percentile (%) |
|---|:---:|:---:|:---:|:---:|
| Socket 3D bounding-box volume / image volume | 36.22 ± 17.84 | 34.43 | 20.94–49.22 | 12.01–67.71 |
| Socket maximal cross-sectional area / slice area | 30.66 ± 17.28 | 25.93 | 17.07–40.98 | 8.55–64.08 |
| Socket cross-sectional height / image height | 47.53 ± 13.82 | 46.88 | 37.50–57.29 | 26.20–71.72 |
| Socket cross-sectional width / image width | 61.16 ± 22.47 | 60.94 | 41.67–76.30 | 29.17–100.00 |
| Preoperative bone (x ≠ 0) 3D bounding-box volume / image volume | 41.90 ± 18.36 | 40.45 | 25.50–57.03 | 16.87–71.88 |

IQR, interquartile range; SD, standard deviation.

## Data availability

Clinical and radiographic data are not publicly available due to hospital and patient privacy regulations. See the Data Availability Statement of the article.

## Repository layout

```
ARPNet/
├── arpnet/
│   ├── models/
│   │   ├── blocks.py       # Mamba layer, residual blocks, scale encoder/decoder
│   │   ├── stage1.py       # ARPNetStageI
│   │   ├── stage2.py       # multimodal ARPNetStageII and 3D ResNet encoder
│   │   └── arpnet.py       # end-to-end ARPNet wrapper
│   ├── data/
│   │   ├── dataset.py      # paired preoperative/postoperative dataset
│   │   ├── clinical.py     # clinical-variable encoding and multimodal dataset
│   │   └── preprocessing.py # NIfTI resampling, centre cropping, largest component
│   ├── losses.py           # MSE, Dice, BCE, Brier, weighted CE
│   ├── metrics.py          # segmentation and classification metrics
│   └── utils.py            # seeding, logging, checkpointing
├── scripts/
│   ├── prepare_crops.py    # crop registered mask pairs to 192^3
│   ├── train_stage1.py     # train Stage I
│   ├── train_stage2.py     # train multimodal Stage II with frozen Stage I
│   ├── predict.py          # end-to-end inference
│   └── evaluate.py         # aggregate metrics and bootstrap CIs
├── examples/
│   ├── manifest_example.csv
│   └── clinical_example.csv
├── checkpoints/            # trained weights (release assets; see checkpoints/README.md)
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

### Clinical data

Stage II also requires a clinical CSV with the following columns:

```csv
case_id,tooth_type,jaw,age,gender
case_001,molar,mand,52,0
case_002,premolar,max,47,1
```

- `case_id`: must match the `case_id` in the imaging manifest.
- `tooth_type`: `molar`, `premolar`, `incisor`, or `canine`.
- `jaw`: `max` or `mand`.
- `age`: age in years. It is standardized using the mean and standard deviation
  of the training split.
- `gender`: `0` or `1`.

A template is provided in `examples/clinical_example.csv`. Cases without a
clinical record receive an all-zero vector; the prediction CSV records
`clinical_available` so these cases can be identified.

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
    --clinical_csv data/clinical.csv \
    --stage1_ckpt runs/stage1/best.pth \
    --out_dir runs/stage2 \
    --epochs 200 \
    --batch_size 2 \
    --lr 3e-4 \
    --loss ce \
    --encoder1_ckpt checkpoints/resnet3d_post.pth \
    --encoder2_ckpt checkpoints/resnet3d_pre.pth \
    --freeze_encoders
```

Stage I is frozen and runs in evaluation mode. The released multimodal
checkpoint was trained with frozen pretrained encoders and weighted
cross-entropy; only the clinical projection and fusion head were trainable.
The best checkpoint is selected by validation balanced accuracy and written to
`runs/stage2/best.pth`.

`encoder1` processes the generated postoperative mask; `encoder2` processes
the preoperative volume. This order matches the original ARPNet checkpoints.
The checkpoint stores the age mean and standard deviation so that inference
uses exactly the same clinical encoding as training.

## Inference

```bash
python scripts/predict.py \
    --manifest data/crops/manifest.csv \
    --clinical_csv data/clinical.csv \
    --split test \
    --stage1_ckpt runs/stage1/best.pth \
    --stage2_ckpt runs/stage2/best.pth \
    --out_dir predictions \
    --eval_stage1 \
    --save_masks
```

Outputs:

- `predictions/predictions.csv`: case-level Stage II probabilities, predicted
  labels, `clinical_available` (1 if a clinical record was found, 0 if the
  zero-vector fallback was used), and (with `--eval_stage1`) Stage I
  Dice/IoU/precision/recall and voxel counts.
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
one-vs-rest AUROC, and the multi-class Brier score. The Brier score is averaged
over both cases and classes,
`(1 / (N * C)) * sum_i sum_c (p_ic - y_ic)^2`, matching the paper. Bootstrap
95% CIs are computed for accuracy, balanced accuracy, macro F1 and AUROC. If a
class is absent from a resampled cohort, AUROC is computed over the classes
present.

## Reproducibility notes

- Splits are patient-level and are supplied through the `split` column of the
  manifest; the code does not create splits automatically.
- The default seed is 42. `set_seed` controls Python, NumPy and PyTorch.
- Class-balanced oversampling used in the original training pipeline can be
  enabled with `--balance`. Validation is never balanced.
- Stage I input is mapped to `{-1, +1}`; Stage II input is the same
  preoperative volume, a binary Stage I mask thresholded at 0.5, and an
  8-dimensional clinical vector.
- Stage I uses MSE. The released multimodal Stage II checkpoint was trained
  with weighted cross-entropy (`1.0`, `2.0`, `4.0` for classes 0–2);
  `--loss mse` switches to the multi-class Brier score.
- The clinical vector is built from tooth type (4 one-hot), jaw (2 one-hot),
  gender (1), and age standardized with training-set statistics (1). The age
  mean and standard deviation are stored in the Stage II checkpoint.
- The released multimodal checkpoint used frozen pretrained ResNet-3D encoders;
  only the clinical projection and fusion head were trained.
- Training Stage I on 192³ volumes with batch size 1 requires a GPU with at
  least 24 GB of memory. Reduce the volume size only if the preprocessing is
  adjusted consistently.

## Pretrained weights

The trained checkpoints are distributed as GitHub Release assets:

- Stage I: [ARPNetStageI.pth](https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.0.0/ARPNetStageI.pth)
- Multimodal Stage II: [ARPNetStageII_multimodal.pth](https://github.com/xinboyu-sjtu/ARPNet/releases/download/v1.1.0/ARPNetStageII_multimodal.pth)

Place them in `checkpoints/` and pass them with `--stage1_ckpt` and
`--stage2_ckpt`. See `checkpoints/README.md` for file sizes, checksums and
loading examples. The multimodal Stage II checkpoint stores the clinical
encoding metadata (`clinical_dim`, `clinical_age_mean`, `clinical_age_std`) and
loads strictly into `ARPNetStageII`.

The earlier `ARPNetStageII.pth` asset in the `v1.0.0` release corresponds to the
image-only Stage II model without clinical variables. Use
`ARPNetStageII_multimodal.pth` for the multimodal model described above.

## Citation

```bibtex
@article{yu2026arpnet,
  title   = {Deep Learning-Based Preoperative Prediction of Alveolar Ridge Morphology and Need for Ancillary Bone Augmentation after Alveolar Ridge Preservation: A Multicenter Development and Validation Study},
  author  = {Yu, Xinbo and Li, Haitao and Wu, Ziang and Wang, Feng and Gao, Hanqi and
             Yan, Qi and He, Dongning and Fan, Zhen and Zhang, Dayong and Chen, Xiaojun and Wu, Yiqun},
  year    = {2026}
}
```

## Ethics and data availability

The study protocol was approved by the Institutional Review Board of Shanghai
Ninth People's Hospital (SH9H-2025-T473-1). Patient data
cannot be shared publicly; requests should be addressed to the corresponding
authors.

## Funding

This study was supported by the AI + Oral and
Craniomaxillofacial Implantology Micro-Specialty Program (DGD-2025-2).

## Contact

- Yiqun Wu — yiqunwu@hotmail.com
- Xiaojun Chen — xiaojunchen@sjtu.edu.cn

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
