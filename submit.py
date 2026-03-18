"""
Submit predictions to api.ainm.no.
Set token first: export NMIAI_TOKEN=<your_token>

Usage:
  python submit.py task1
  python submit.py task2
  python submit.py task3
  python submit.py all
"""
import sys
from src.common.api import submit_csv
from src.common.config import TASK1, TASK2, TASK3

TASKS = {
    "task1": TASK1,
    "task2": TASK2,
    "task3": TASK3,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(TASKS.keys()) if target == "all" else [target]

    for key in keys:
        cfg = TASKS[key]
        print(f"\nSubmitting {key}...")
        try:
            submit_csv(key, cfg["submission_path"])
        except FileNotFoundError:
            print(f"  No submission file found at {cfg['submission_path']} — run predict first.")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
