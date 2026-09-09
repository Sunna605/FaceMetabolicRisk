#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farl_vit_gradcam.gradcam import attention_gradcam, save_cam_bundle
from farl_vit_gradcam.model import load_farl_classifier
from farl_vit_gradcam.preprocess import eval_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image-path", type=Path, nargs="+", required=True)
    parser.add_argument("--farl-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--target-class", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_farl_classifier(args.farl_checkpoint, args.model_checkpoint, device=device)
    model.eval()
    transform = eval_transform()
    for image_path in args.image_path:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            cams, logits, valid, error = attention_gradcam(model, tensor, args.target_class)
        logit = logits[0]
        metadata = {
            "method": "last_layer_cls_attention_gradcam",
            "task": args.task,
            "fold": args.fold,
            "target_class": args.target_class,
            "target_logit": f"logit_{args.target_class}",
            "image_path": str(image_path.resolve()),
            "checkpoint": str(args.model_checkpoint.resolve()),
            "logit_0": float(logit[0]),
            "logit_1": float(logit[1]),
            "pred": int(np.argmax(logit)),
            "valid_map": bool(valid[0]),
            "patch_grid": [14, 14],
            "max_capture_path_logit_error": error,
        }
        sample_dir = args.output_dir / args.task / image_path.stem
        save_cam_bundle(sample_dir, image, cams[0], metadata)
        print(sample_dir)


if __name__ == "__main__":
    main()
