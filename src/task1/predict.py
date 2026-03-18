import pickle

import pandas as pd

from src.common.config import TASK1
from src.common.io import read_csv, write_csv


def main() -> None:
    cfg = TASK1

    with open("models/task1_model.pkl", "rb") as f:
        saved = pickle.load(f)

    pipeline = saved["pipeline"]
    feature_cols = saved["feature_cols"]
    id_col = saved["id_col"]

    test_df = read_csv(cfg["test_path"])
    X_test = test_df[feature_cols]

    preds = pipeline.predict(X_test)

    submission = pd.DataFrame({
        id_col: test_df[id_col],
        cfg["target_column"]: preds,
    })

    write_csv(submission, cfg["submission_path"])
    print(f"Submission written to {cfg['submission_path']}")


if __name__ == "__main__":
    main()