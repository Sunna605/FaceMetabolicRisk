#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farl_vit_gradcam.aggregate import group_means
from farl_vit_gradcam.gradcam import attention_gradcam, render_heatmap
from farl_vit_gradcam.model import load_farl_classifier
from farl_vit_gradcam.preprocess import eval_transform


TASKS = ("MetS", "obesity", "COE", "dyslipidemia", "hypertension", "hyperglycemia")


class FaceDataset(Dataset):
    def __init__(self, ids, image_paths, transform):
        self.ids, self.image_paths, self.transform = ids, image_paths, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        with Image.open(self.image_paths[sample_id]) as image:
            return sample_id, self.transform(image.convert("RGB"))


def save_group(path, task, group, mean_cam, mean_face, metadata):
    directory = path / task / group
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "mean_cam_14x14.npy", mean_cam.astype(np.float32))
    heatmap = render_heatmap(mean_cam)
    Image.fromarray(np.uint8(np.clip(mean_face, 0, 1) * 255)).save(directory / "mean_face.png")
    plt.imsave(directory / "mean_heatmap.png", heatmap)
    plt.imsave(directory / "overlay_on_mean_face.png", np.clip(0.5 * mean_face + 0.5 * heatmap, 0, 1))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--farl-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--target-class", type=int, choices=(0, 1), default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    split = pd.read_csv(args.split_csv, dtype={"ID": str})
    image_paths = {p.stem: p for p in args.image_dir.iterdir() if p.is_file()}
    ids = [sample_id for sample_id in split["ID"].astype(str) if sample_id in image_paths]
    missing = [sample_id for sample_id in split["ID"].astype(str) if sample_id not in image_paths]
    checkpoints = json.loads(args.model_checkpoint_map.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True)
    pd.DataFrame({"id": ids}).to_csv(args.output_dir / "included_ids.csv", index=False)
    pd.DataFrame({"id": missing, "reason": "matching face image not found"}).to_csv(
        args.output_dir / "missing_face_ids.csv", index=False
    )
    mean_face = np.mean(
        [np.asarray(Image.open(image_paths[s]).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0 for s in ids], axis=0
    )
    Image.fromarray(np.uint8(np.clip(mean_face, 0, 1) * 255)).save(args.output_dir / "cohort_mean_face.png")

    device = torch.device(args.device)
    transform = eval_transform()
    summary = []
    for task in TASKS:
        if task not in checkpoints:
            continue
        labels = pd.to_numeric(split.set_index("ID").loc[ids, task], errors="coerce")
        if labels.max() > 1:
            labels = labels - 1
        labels = labels.fillna(labels.mode().iloc[0]).astype(int).to_numpy()
        model = load_farl_classifier(args.farl_checkpoint, checkpoints[task], device=device)
        model.eval()
        loader = DataLoader(FaceDataset(ids, image_paths, transform), batch_size=args.batch_size, shuffle=False)
        cams = []
        valid = []
        max_error = 0.0
        for _, batch in loader:
            batch = batch.to(device)
            batch_cams, _, batch_valid, error = attention_gradcam(model, batch, args.target_class)
            cams.append(batch_cams)
            valid.append(batch_valid)
            max_error = max(max_error, error)
        cams = np.concatenate(cams)
        valid = np.concatenate(valid)
        for group, result in group_means(cams, labels).items():
            metadata = {
                "method": "mean_of_per_person_last_layer_cls_attention_gradcam",
                "aggregation": "pixelwise arithmetic mean after per-person 0-1 normalization",
                "cohort": "user-provided",
                "fold": args.fold,
                "task": task,
                "group": group,
                "target_class": args.target_class,
                "target_logit": f"logit_{args.target_class}",
                "n": result["n"],
                "zero_map_count": result["zero_map_count"],
                "patch_grid": [14, 14],
                "checkpoint": str(Path(checkpoints[task]).resolve()),
                "max_capture_path_logit_error": max_error,
            }
            save_group(args.output_dir, task, group, result["mean"], mean_face, metadata)
            summary.append(metadata)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "target_class": args.target_class,
                "included_face_count": len(ids),
                "missing_face_count": len(missing),
                "tasks": [row["task"] for row in summary if row["group"] == "overall"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
