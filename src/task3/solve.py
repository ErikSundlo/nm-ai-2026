"""
Task 3 — Astar Island Norse World Prediction
Strategy:
  - Tile the 40x40 map with 15x15 viewports (8 queries = full coverage per seed)
  - 5 seeds x 8 = 40 queries; remaining 10 re-sample high-entropy areas
  - Build W×H×6 probability tensor per seed from observed final states
  - For unobserved cells, fall back to initial-state prior

Usage:
    NMIAI_TOKEN=xxx python -m src.task3.solve
"""
import json
import os
import sys
import time
from collections import defaultdict

import httpx
import numpy as np

API = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("NMIAI_TOKEN", "")

# Terrain values present in the maps -> sorted order -> class indices 0-5
# Discovered from initial_states: {1, 2, 4, 5, 10, 11}
TERRAIN_VALUES = sorted([1, 2, 4, 5, 10, 11])
VAL_TO_CLASS   = {v: i for i, v in enumerate(TERRAIN_VALUES)}
NUM_CLASSES    = len(TERRAIN_VALUES)  # 6


def headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def get_active_round():
    resp = httpx.get(f"{API}/rounds", timeout=10)
    rounds = resp.json()
    for r in rounds:
        if r["status"] == "active":
            return r
    return None


def simulate(round_id: str, seed_index: int, x: int, y: int, w: int, h: int) -> list[list[int]] | None:
    """Call simulate API, return the observed viewport grid (or None on error)."""
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport": {"x": x, "y": y, "width": w, "height": h},
    }
    try:
        resp = httpx.post(f"{API}/simulate", json=payload, headers=headers(), timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Response format: {"grid": [[...], ...], ...} or just the grid
            if isinstance(data, list):
                return data
            return data.get("grid") or data.get("cells") or data.get("viewport")
        else:
            print(f"  simulate {seed_index} ({x},{y}) → {resp.status_code}: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"  simulate error: {e}")
        return None


def submit_predictions(round_id: str, predictions: list[dict]) -> dict:
    """Submit W×H×6 tensors for all seeds."""
    # Try different endpoint formats
    for endpoint in [f"{API}/submit", f"{API}/rounds/{round_id}/predictions",
                     f"{API}/rounds/{round_id}/submit", f"{API}/predictions"]:
        try:
            payload = {"round_id": round_id, "predictions": predictions}
            resp = httpx.post(endpoint, json=payload, headers=headers(), timeout=30)
            if resp.status_code not in (404,):
                print(f"  submit → {resp.status_code} via {endpoint}: {resp.text[:200]}")
                return resp.json() if resp.status_code < 300 else {"error": resp.text}
        except Exception as e:
            print(f"  submit error ({endpoint}): {e}")
    return {"error": "no valid endpoint found"}


def build_tile_queries(map_w: int, map_h: int, vp_size: int = 15) -> list[tuple[int, int, int, int]]:
    """Generate non-overlapping viewport tiles covering the full map."""
    tiles = []
    for y in range(0, map_h, vp_size):
        for x in range(0, map_w, vp_size):
            w = min(vp_size, map_w - x)
            h = min(vp_size, map_h - y)
            tiles.append((x, y, w, h))
    return tiles


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    """Convert integer grid to one-hot class probabilities (H, W, 6)."""
    h, w = len(grid), len(grid[0])
    tensor = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for row_i, row in enumerate(grid):
        for col_i, val in enumerate(row):
            cls = VAL_TO_CLASS.get(val, VAL_TO_CLASS.get(11, NUM_CLASSES - 1))
            tensor[row_i, col_i, cls] = 1.0
    return tensor


def main():
    if not TOKEN:
        print("ERROR: Set NMIAI_TOKEN env var")
        sys.exit(1)

    round_info = get_active_round()
    if not round_info:
        print("No active round found")
        sys.exit(1)

    round_id  = round_info["id"]
    map_w     = round_info["map_width"]
    map_h     = round_info["map_height"]
    n_seeds   = round_info["seeds_count"]
    closes_at = round_info["closes_at"]
    initial_states = round_info["initial_states"]

    print(f"Round {round_info['round_number']}: {map_w}x{map_h}, {n_seeds} seeds, closes {closes_at}")

    # Build tiling plan
    tiles = build_tile_queries(map_w, map_h)
    queries_per_seed = len(tiles)  # ~8 for 40x40 with 15x15 tiles
    total_needed = queries_per_seed * n_seeds
    QUERY_BUDGET = 50
    print(f"Tiles per seed: {queries_per_seed}, total needed: {total_needed}, budget: {QUERY_BUDGET}")

    # Accumulator: for each (seed, row, col) track class counts across multiple observations
    # Shape: (n_seeds, map_h, map_w, NUM_CLASSES)
    counts = np.zeros((n_seeds, map_h, map_w, NUM_CLASSES), dtype=np.float32)

    # Seed prior from initial states
    for seed_idx, state in enumerate(initial_states):
        if state and "grid" in state:
            prior = grid_to_onehot(state["grid"])
            counts[seed_idx] += prior * 0.3  # soft prior weight

    queries_used = 0

    # Phase 1: full coverage for each seed
    for seed_idx in range(n_seeds):
        if queries_used >= QUERY_BUDGET:
            break
        for (x, y, w, h) in tiles:
            if queries_used >= QUERY_BUDGET:
                break
            print(f"  Seed {seed_idx} tile ({x},{y},{w},{h}) [query {queries_used+1}/{QUERY_BUDGET}]")
            grid = simulate(round_id, seed_idx, x, y, w, h)
            queries_used += 1
            time.sleep(0.2)  # be nice to the API
            if grid is None:
                continue
            # Accumulate observations
            for row_i, row in enumerate(grid):
                for col_i, val in enumerate(row):
                    cls = VAL_TO_CLASS.get(val, VAL_TO_CLASS.get(11, NUM_CLASSES - 1))
                    counts[seed_idx, y + row_i, x + col_i, cls] += 1.0

    # Phase 2: use remaining queries on high-entropy areas
    remaining = QUERY_BUDGET - queries_used
    if remaining > 0:
        print(f"\nPhase 2: {remaining} queries on high-entropy areas")
        # Find cells with most uncertainty (close to uniform distribution)
        # Normalise counts to get current probabilities
        prob_tmp = counts / (counts.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -(prob_tmp * np.log(prob_tmp + 1e-8)).sum(axis=-1)  # (n_seeds, H, W)

        for _ in range(remaining):
            if queries_used >= QUERY_BUDGET:
                break
            # Pick the seed and position with highest entropy
            flat_idx = np.argmax(entropy)
            seed_idx, best_y, best_x = np.unravel_index(flat_idx, entropy.shape)
            x = max(0, min(int(best_x) - 7, map_w - 15))
            y = max(0, min(int(best_y) - 7, map_h - 15))
            w, h = min(15, map_w - x), min(15, map_h - y)

            print(f"  High-entropy seed {seed_idx} at ({x},{y}) [query {queries_used+1}/{QUERY_BUDGET}]")
            grid = simulate(round_id, int(seed_idx), x, y, w, h)
            queries_used += 1
            time.sleep(0.2)
            if grid:
                for row_i, row in enumerate(grid):
                    for col_i, val in enumerate(row):
                        cls = VAL_TO_CLASS.get(val, NUM_CLASSES - 1)
                        counts[int(seed_idx), y + row_i, x + col_i, cls] += 1.0
            # Zero out entropy in observed area to avoid re-selecting
            entropy[seed_idx, y:y+h, x:x+w] = 0.0

    print(f"\nTotal queries used: {queries_used}")

    # Build final probability tensors
    predictions = []
    for seed_idx in range(n_seeds):
        raw = counts[seed_idx]  # (H, W, 6)
        row_sums = raw.sum(axis=-1, keepdims=True)
        # Where we have no observations, counts are all 0 → uniform
        mask_zero = (row_sums == 0)
        probs = np.where(mask_zero, 1.0 / NUM_CLASSES, raw / (row_sums + 1e-8))
        probs = probs.astype(np.float32)

        predictions.append({
            "seed_index": seed_idx,
            "grid": probs.tolist(),
        })
        print(f"  Seed {seed_idx}: pred entropy mean={float(-(probs*np.log(probs+1e-8)).sum(-1).mean()):.3f}")

    # Save locally
    out_path = f"submissions/task3_round{round_info['round_number']}.json"
    os.makedirs("submissions", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"round_id": round_id, "predictions": predictions}, f)
    print(f"Saved predictions to {out_path}")

    # Submit
    print("\nSubmitting...")
    result = submit_predictions(round_id, predictions)
    print("Result:", result)


if __name__ == "__main__":
    main()
