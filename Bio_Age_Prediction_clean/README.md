# FaRL-ViT facial phenotype prediction

This repository contains the minimal code needed to reproduce the 2D ViT-FaRL
classifier and the last-layer attention Grad-CAM analysis used in the study.
It includes image preprocessing, model loading, binary prediction,
per-image Grad-CAM, and cohort-level averaging. Raw images, labels, trained
downstream checkpoints, and the public FaRL checkpoint are intentionally kept
outside the repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Download the FaRL-Base-Patch16-LAIONFace20M-ep64 checkpoint from the official
[FaRL repository](https://github.com/FacePerceiver/FaRL). Supply the checkpoint
path explicitly; no personal or machine-specific paths are used by this repo.

## Fine-tune one outcome

The compact training entry point uses the study configuration by default:
20 epochs, batch size 64, AdamW with learning rate 5 × 10⁻⁵ and weight decay
0.01, cosine learning-rate decay, seed 0, and weighted cross-entropy.

```bash
python scripts/train_classifier.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --image-dir /path/to/images \
  --task MetS \
  --farl-checkpoint /path/to/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --output-dir checkpoints/MetS
```

## Export features and logits

```bash
python scripts/predict.py \
  --input-csv /path/to/fold_5_test.csv \
  --image-dir /path/to/images \
  --farl-checkpoint /path/to/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --model-checkpoint checkpoints/MetS/best.pt \
  --output-csv results/MetS.csv
```

The output contains `id`, 512 encoder features, `logit_0`, `logit_1`, and the
argmax prediction.

## Single-image Grad-CAM

The downstream checkpoint should contain the FaRL encoder and the two-logit
classification head for the selected outcome.

```bash
python scripts/run_gradcam.py \
  --task MetS \
  --image-path /path/to/image.png \
  --farl-checkpoint /path/to/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --model-checkpoint /path/to/task_MetS_fold_5.pt \
  --output-dir results/gradcam
```

The implementation targets `logit_1` by default. Each sample produces a 14 ×
14 CAM, a heatmap, an overlay, and metadata containing both logits and the
checkpoint path.

## Cohort-level mean Grad-CAM

Prepare a CSV with an `ID` column and one binary column for each outcome. Put
aligned face images in one directory, named `<ID>.<extension>`. Provide a JSON
mapping from outcome names to downstream checkpoints:

```json
{
  "MetS": "/path/to/task_MetS_fold_5.pt",
  "obesity": "/path/to/task_obesity_fold_5.pt"
}
```

Then run:

```bash
python scripts/average_gradcam.py \
  --split-csv /path/to/fold_5_test.csv \
  --image-dir /path/to/images \
  --farl-checkpoint /path/to/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --model-checkpoint-map /path/to/checkpoints.json \
  --output-dir results/cohort_mean
```

The script averages equally weighted, per-person CAMs after their individual
normalization to [0, 1]. It writes an overall mean and true-label-0/true-label-1
means for each outcome, together with sample counts and a provenance manifest.

## Method

Images are converted to RGB, resized to 224 × 224 pixels, converted to tensors,
and normalized with channel means `(0.48145466, 0.45782750, 0.40821073)` and
standard deviations `(0.26862954, 0.26130258, 0.27577711)`. During fine-tuning,
training augmentation is random horizontal flipping (p = 0.5), color jitter
(0.3 for brightness, contrast, saturation, and hue), and random rotation within
±25°. Validation and test images use deterministic preprocessing.

The ViT-Base encoder is initialized from FaRL, which was pretrained on
LAION-Face with visual-linguistic and masked-image objectives. The downstream
classifier uses a two-layer head and is fine-tuned with the study-specific
checkpoint. Scheme 1 uses the final Transformer block's CLS-to-patch attention,
weights each head by the mean gradient of the target logit, averages heads,
clips negative values, and reshapes the result to a 14 × 14 patch grid.

## Citation

Zheng Y, Yang H, Zhang T, et al. *General Facial Representation Learning in a
Visual-Linguistic Manner*. CVPR, 2022. DOI:
[10.1109/CVPR52688.2022.01814](https://doi.org/10.1109/CVPR52688.2022.01814).

The attention-map implementation follows the design described in
[Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
