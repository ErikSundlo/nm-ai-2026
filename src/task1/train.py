"""
Task 1 — Tabular ML (Tripletex)
Train a LightGBM model with cross-validation.

Usage:
  python -m src.task1.train
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.common.config import TASK1
from src.common.io import ensure_dir, read_csv


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = [
        ("num", SimpleImputer(strategy="median"), numeric_cols),
    ]
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def main() -> None:
    cfg = TASK1
    train_df = read_csv(cfg["train_path"])

    id_col = cfg["id_column"]
    target_col = cfg["target_column"]
    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

    X = train_df[feature_cols]
    y = train_df[target_col]

    # Encode target if categorical
    le = None
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    preprocessor = build_preprocessor(X)
    X_enc = preprocessor.fit_transform(X)

    # Cross-validation to gauge performance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = []

    lgb_params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_enc, y)):
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_enc[tr_idx], y[tr_idx],
            eval_set=[(X_enc[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_enc[val_idx])
        score = roc_auc_score(y[val_idx], preds[:, 1] if preds.shape[1] == 2 else preds, multi_class="ovr")
        oof_scores.append(score)
        print(f"  Fold {fold + 1}  AUC: {score:.4f}")

    print(f"CV AUC: {np.mean(oof_scores):.4f} ± {np.std(oof_scores):.4f}")

    # Retrain on full data
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X_enc, y, callbacks=[lgb.log_evaluation(0)])

    ensure_dir("models")
    with open(cfg["model_path"], "wb") as f:
        pickle.dump({
            "preprocessor": preprocessor,
            "model": final_model,
            "feature_cols": feature_cols,
            "label_encoder": le,
            "id_col": id_col,
            "target_col": target_col,
        }, f)

    print(f"Model saved to {cfg['model_path']}")


if __name__ == "__main__":
    main()
