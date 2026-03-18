"""
Task 2 — NLP / Language Model (Astar): Generate predictions.

Usage:
  python -m src.task2.predict
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import TASK2
from src.common.io import read_csv, write_csv


def predict_texts(texts, model, tokenizer, batch_size=32, device="cpu"):
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=256, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.array(all_preds)


def main() -> None:
    cfg = TASK2
    model_dir = cfg["model_path"]

    with open(Path(model_dir) / "label_encoder.pkl", "rb") as f:
        saved = pickle.load(f)
    le, id_col, target_col, text_col = (
        saved["le"], saved["id_col"], saved["target_col"], saved["text_col"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    test_df = read_csv(cfg["test_path"])
    texts = test_df[text_col].fillna("").tolist()

    preds_enc = predict_texts(texts, model, tokenizer, device=device)
    preds = le.inverse_transform(preds_enc)

    submission = pd.DataFrame({id_col: test_df[id_col], target_col: preds})
    write_csv(submission, cfg["submission_path"])
    print(f"Submission written to {cfg['submission_path']}")


if __name__ == "__main__":
    main()
