"""
Shared API client for api.ainm.no.
Set your token: export NMIAI_TOKEN=<your_token>
"""
from pathlib import Path

import requests

from src.common.config import API_BASE, API_TOKEN


def _headers():
    if not API_TOKEN:
        raise ValueError("NMIAI_TOKEN env var not set")
    return {"Authorization": f"Bearer {API_TOKEN}"}


def get_me():
    """Fetch current user info."""
    r = requests.get(f"{API_BASE}/users/me", headers=_headers())
    r.raise_for_status()
    return r.json()


def submit_csv(task: str, csv_path: str | Path):
    """
    Submit a CSV file for a given task.
    Endpoint and exact format TBD — update once competition launches.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # TODO: update endpoint once revealed on March 19
    url = f"{API_BASE}/tasks/{task}/submit"

    with open(csv_path, "rb") as f:
        r = requests.post(
            url,
            headers=_headers(),
            files={"file": (csv_path.name, f, "text/csv")},
        )

    r.raise_for_status()
    result = r.json()
    print(f"[{task}] Submission accepted — score: {result.get('score', '?')}")
    return result
