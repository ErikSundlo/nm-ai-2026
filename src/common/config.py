import os

API_BASE = "https://api.ainm.no"
API_TOKEN = os.environ.get("NMIAI_TOKEN", "")  # set via: export NMIAI_TOKEN=<your_token>

TASK1 = {
    # Sponsor: Tripletex (accounting SaaS) — expected: tabular ML
    "name": "task1",
    "train_path": "data/raw/task1_train.csv",
    "test_path": "data/raw/task1_test.csv",
    "submission_path": "data/submissions/task1_submission.csv",
    "model_path": "models/task1_model.pkl",
    "target_column": "target",
    "id_column": "id",
}

TASK2 = {
    # Sponsor: Astar (tech) — expected: language model / NLP
    "name": "task2",
    "train_path": "data/raw/task2_train.csv",
    "test_path": "data/raw/task2_test.csv",
    "submission_path": "data/submissions/task2_submission.csv",
    "model_path": "models/task2_model",
    "target_column": "target",
    "id_column": "id",
    "text_column": "text",          # TODO: update once task is revealed
    "model_name": "ltg/norbert3-base",  # good Norwegian BERT baseline
}

TASK3 = {
    # Sponsor: NorgesGruppen Data (grocery) — expected: computer vision
    "name": "task3",
    "train_dir": "data/raw/task3/train",
    "test_dir": "data/raw/task3/test",
    "train_csv": "data/raw/task3_train.csv",   # may have labels CSV instead
    "test_csv": "data/raw/task3_test.csv",
    "submission_path": "data/submissions/task3_submission.csv",
    "model_path": "models/task3_model.pt",
    "id_column": "id",
    "target_column": "label",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 10,
}