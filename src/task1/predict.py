"""
Task 1 — Tabular ML (Tripletex): Generate predictions.

Usage:
  python -m src.task1.predict
"""
import pickle

import pandas as pd

from src.common.config import TASK1
from src.common.io import read_csv, write_csv


def main() -> None:
    cfg = TASK1

    with open(cfg["model_path"], "rb") as f:
        saved = pickle.load(f)

    preprocessor = saved["preprocessor"]
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    le = saved["label_encoder"]
    id_col = saved["id_col"]
    target_col = saved["target_col"]

    test_df = read_csv(cfg["test_path"])
    X_test = preprocessor.transform(test_df[feature_cols])

    preds = model.predict(X_test)
    if le is not None:
        preds = le.inverse_transform(preds)

    submission = pd.DataFrame({id_col: test_df[id_col], target_col: preds})
    write_csv(submission, cfg["submission_path"])
    print(f"Submission written to {cfg['submission_path']}")


if __name__ == "__main__":
    main()
