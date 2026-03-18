"""
Task 3 — Computer Vision (NorgesGruppen Data)
Fine-tune an EfficientNet image classifier.

Expects either:
  a) data/raw/task3/train/<class_name>/<image.jpg>  (ImageFolder layout)
  b) data/raw/task3_train.csv with columns: id, filepath, label

TODO: confirm layout once task is revealed.

Usage:
  python -m src.task3.train
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from src.common.config import TASK3
from src.common.io import ensure_dir, read_csv


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int, augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


class CSVImageDataset(Dataset):
    """Dataset backed by a CSV with filepath + label columns."""
    def __init__(self, df: pd.DataFrame, label_col: str, filepath_col: str,
                 transform=None, label_encoder=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.filepath_col = filepath_col
        self.transform = transform
        if label_encoder is None:
            self.classes = sorted(df[label_col].unique())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.classes, self.class_to_idx = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.filepath_col]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row[self.label_col]]
        return img, label


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    cfg = TASK3
    img_size  = cfg["img_size"]
    batch_sz  = cfg["batch_size"]
    n_epochs  = cfg["epochs"]
    model_path = cfg["model_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ load data
    train_dir = Path(cfg["train_dir"])
    train_csv = Path(cfg["train_csv"])

    if train_csv.exists():
        train_df = read_csv(train_csv)
        dataset = CSVImageDataset(
            train_df,
            label_col=cfg["target_column"],
            filepath_col="filepath",     # TODO: update if different column name
            transform=get_transforms(img_size, augment=True),
        )
        classes = dataset.classes
        le = (dataset.classes, dataset.class_to_idx)
    elif train_dir.exists():
        full_dataset = ImageFolder(train_dir, transform=get_transforms(img_size, augment=True))
        classes = full_dataset.classes
        le = (classes, full_dataset.class_to_idx)
        dataset = full_dataset
    else:
        raise FileNotFoundError(
            f"No training data found at {train_csv} or {train_dir}\n"
            "Download the data first."
        )

    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Train/val split
    val_size = max(1, int(0.1 * len(dataset)))
    tr_size  = len(dataset) - val_size
    tr_dataset, val_dataset = random_split(
        dataset, [tr_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Val uses no-augment transforms
    val_dataset.dataset.transform = get_transforms(img_size, augment=False)

    tr_loader  = DataLoader(tr_dataset,  batch_size=batch_sz, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------ model
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_acc = 0.0
    ensure_dir("models")

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(imgs)
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{n_epochs}  loss: {tr_loss/tr_size:.4f}  val_acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (acc={best_acc:.4f})")

    # Save label encoder alongside model
    with open(Path(model_path).with_suffix(".pkl"), "wb") as f:
        pickle.dump({"classes": classes, "class_to_idx": le[1]}, f)

    print(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
