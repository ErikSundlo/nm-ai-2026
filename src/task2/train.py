"""
Task 2 — NLP / Language Model (Astar)
Fine-tune a Norwegian BERT model for text classification.

TODO: Update text_column, target_column, and num_labels in config once task is revealed.

Usage:
  python -m src.task2.train
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.common.config import TASK2
from src.common.io import ensure_dir, read_csv


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, average="macro")}


def main() -> None:
    cfg = TASK2
    train_df = read_csv(cfg["train_path"])

    id_col = cfg["id_column"]
    target_col = cfg["target_column"]
    text_col = cfg["text_column"]

    le = LabelEncoder()
    labels = le.fit_transform(train_df[target_col])
    num_labels = len(le.classes_)
    print(f"Classes ({num_labels}): {le.classes_}")

    texts = train_df[text_col].fillna("").tolist()
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    tr_dataset  = TextDataset(tr_texts,  tr_labels,  tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    model_dir = cfg["model_path"]
    ensure_dir(model_dir)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save label encoder
    import pickle
    with open(Path(model_dir) / "label_encoder.pkl", "wb") as f:
        pickle.dump({"le": le, "id_col": id_col, "target_col": target_col, "text_col": text_col}, f)

    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
