from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.config import TASK1
from src.common.io import read_csv, ensure_dir


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def main() -> None:
    cfg = TASK1
    train_df = read_csv(cfg["train_path"])

    target_col = cfg["target_column"]
    id_col = cfg["id_column"]

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X = train_df[feature_cols]
    y = train_df[target_col]

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    ensure_dir("models")
    with open("models/task1_model.pkl", "wb") as f:
        pickle.dump(
            {
                "pipeline": pipeline,
                "feature_cols": feature_cols,
                "id_col": id_col,
            },
            f,
        )

    print("Model trained and saved to models/task1_model.pkl")


if __name__ == "__main__":
    main()