#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farl_vit_gradcam.losses import weighted_cross_entropy
from farl_vit_gradcam.model import load_farl_classifier
from farl_vit_gradcam.preprocess import eval_transform, train_transform


class LabelledFaces(Dataset):
    def __init__(self, frame, image_paths, task, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.image_paths[str(row["ID"])]) as image:
            image = self.transform(image.convert("RGB"))
        label = int(row[self.task])
        if label > 1:
            label -= 1
        return image, label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, labels, scores = [], [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))["cls_logits"]
            batch_labels = batch_labels.to(device)
            losses.append(float(criterion(logits, batch_labels).item()))
            labels.extend(batch_labels.cpu().numpy().tolist())
            scores.extend(logits[:, 1].cpu().numpy().tolist())
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    return float(np.mean(losses)), auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--farl-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {path.stem: path for path in args.image_dir.iterdir() if path.is_file()}
    train_frame = pd.read_csv(args.train_csv, dtype={"ID": str})
    val_frame = pd.read_csv(args.val_csv, dtype={"ID": str})
    train_frame = train_frame[train_frame["ID"].isin(paths)]
    val_frame = val_frame[val_frame["ID"].isin(paths)]
    device = torch.device(args.device)
    model = load_farl_classifier(args.farl_checkpoint, device=device)
    train_loader = DataLoader(LabelledFaces(train_frame, paths, args.task, train_transform()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LabelledFaces(val_frame, paths, args.task, eval_transform()), batch_size=args.batch_size, shuffle=False)
    criterion = weighted_cross_entropy(args.task, device=device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or "bias" in name or "bn" in name or "ln" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": args.weight_decay}],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_auc = -float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))["cls_logits"]
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch + 1, "val_loss": val_loss, "val_auc": val_auc})
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "val_auc": val_auc}, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
