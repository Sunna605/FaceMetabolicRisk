#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farl_vit_gradcam.model import load_farl_classifier
from farl_vit_gradcam.preprocess import eval_transform


class FaceDataset(Dataset):
    def __init__(self, ids, image_paths, transform):
        self.ids, self.image_paths, self.transform = ids, image_paths, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        with Image.open(self.image_paths[sample_id]) as image:
            return sample_id, self.transform(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--farl-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, dtype={args.id_column: str})
    image_paths = {path.stem: path for path in args.image_dir.iterdir() if path.is_file()}
    ids = [sample_id for sample_id in frame[args.id_column].astype(str) if sample_id in image_paths]
    if not ids:
        raise RuntimeError("No input row has a matching image")
    device = torch.device(args.device)
    model = load_farl_classifier(args.farl_checkpoint, args.model_checkpoint, device=device).eval()
    loader = DataLoader(FaceDataset(ids, image_paths, eval_transform()), batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for batch_ids, images in loader:
            result, features = model(images.to(device), return_features=True)
            logits = result["cls_logits"].cpu().numpy()
            features = features.cpu().numpy()
            for sample_id, feature, logit in zip(batch_ids, features, logits):
                row = {"id": sample_id}
                row.update({f"feature_{i}": float(value) for i, value in enumerate(feature)})
                row.update(
                    {
                        "logit_0": float(logit[0]),
                        "logit_1": float(logit[1]),
                        "pred": int(np.argmax(logit)),
                    }
                )
                rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
