"""
Task 3 — Computer Vision (NorgesGruppen): Generate predictions.

Usage:
  python -m src.task3.predict
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.common.config import TASK3
from src.common.io import read_csv, write_csv
from src.task3.train import build_model, get_transforms


class TestImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def main() -> None:
    cfg = TASK3

    meta_path = Path(cfg["model_path"]).with_suffix(".pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    classes = meta["classes"]
    idx_to_class = {v: k for k, v in meta["class_to_idx"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    transform = get_transforms(cfg["img_size"], augment=False)

    test_csv = Path(cfg["test_csv"])
    test_dir = Path(cfg["test_dir"])

    if test_csv.exists():
        test_df = read_csv(test_csv)
        image_paths = test_df["filepath"].tolist()   # TODO: update column name if different
        ids = test_df[cfg["id_column"]].tolist()
    else:
        image_paths = sorted(test_dir.rglob("*.jpg")) + sorted(test_dir.rglob("*.png"))
        ids = [p.stem for p in image_paths]

    dataset = TestImageDataset(image_paths, transform=transform)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=2)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            preds = model(batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    labels = [idx_to_class[p] for p in all_preds]
    submission = pd.DataFrame({cfg["id_column"]: ids, cfg["target_column"]: labels})
    write_csv(submission, cfg["submission_path"])
    print(f"Submission written to {cfg['submission_path']}")


if __name__ == "__main__":
    main()
