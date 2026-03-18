"""
NM i AI 2026 - Grocery Bot
WebSocket bot for the warm-up grocery store challenge.

Usage:
  1. Log in at https://app.ainm.no/challenge
  2. Click "Play" on a map to get a token
  3. Run: python grocery_bot.py <token> [--map easy|medium|hard|expert|nightmare]

Game mechanics:
  - Bots navigate a grocery store grid
  - Pick up items adjacent to shelves (walls)
  - Deliver matching items to drop-off zones
  - +1 per item delivered, +5 per order completed
"""

import asyncio
import json
import sys
import argparse
from collections import deque, Counter

import websockets

WS_BASE = "wss://game.ainm.no/ws"


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def bfs(width, height, walls, start, goals, avoid=None):
    """
    BFS from `start` to any position in `goals`.
    Returns list of action strings, or None if unreachable.
    Returns [] if start is already a goal.
    """
    avoid = set(map(tuple, avoid)) if avoid else set()
    goal_set = set(map(tuple, goals))
    start = tuple(start)

    if start in goal_set:
        return []

    queue = deque([(start, [])])
    visited = {start}

    MOVES = [
        (0, -1, "move_up"),
        (0,  1, "move_down"),
        (-1, 0, "move_left"),
        (1,  0, "move_right"),
    ]

    while queue:
        pos, path = queue.popleft()
        for dx, dy, action in MOVES:
            npos = (pos[0] + dx, pos[1] + dy)
            if (
                0 <= npos[0] < width
                and 0 <= npos[1] < height
                and npos not in walls
                and npos not in visited
                and npos not in avoid
            ):
                new_path = path + [action]
                if npos in goal_set:
                    return new_path
                visited.add(npos)
                queue.append((npos, new_path))

    return None


def adjacent_walkable(pos, width, height, walls):
    """Return walkable cells adjacent to pos."""
    x, y = pos
    result = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ap = (x + dx, y + dy)
        if 0 <= ap[0] < width and 0 <= ap[1] < height and ap not in walls:
            result.append(ap)
    return result


def next_pos_for_action(pos, action):
    DELTAS = {
        "move_up":    (0, -1),
        "move_down":  (0,  1),
        "move_left":  (-1, 0),
        "move_right": (1,  0),
    }
    dx, dy = DELTAS[action]
    return (pos[0] + dx, pos[1] + dy)


# ---------------------------------------------------------------------------
# Bot logic
# ---------------------------------------------------------------------------

def decide_actions(state):
    grid = state["grid"]
    width, height = grid["width"], grid["height"]
    walls = frozenset(tuple(w) for w in grid["walls"])

    bots = state["bots"]
    map_items = state["items"]  # [{id, type, position}, ...]

    drop_zones = [tuple(z) for z in state.get("drop_off_zones", [state["drop_off"]])]

    # Active order
    active_order = next((o for o in state["orders"] if o["status"] == "active"), None)
    if not active_order:
        return [{"bot": b["id"], "action": "wait"} for b in bots]

    required = Counter(active_order["items_required"])
    delivered = Counter(active_order["items_delivered"])
    still_needed = required - delivered  # types still needed overall

    # Items on map that are relevant
    needed_map_items = [i for i in map_items if i["type"] in still_needed]

    # Assign each bot a task ------------------------------------------------
    # We process bots in order; track which map items are "claimed"
    claimed_item_ids = set()
    actions = []
    reserved_cells = {}  # bot_id -> cell it will occupy next round

    # First, figure out useful inventory per bot
    def useful_inventory(bot):
        return Counter(t for t in bot["inventory"] if t in still_needed)

    # Sort: bots with full inventories of useful items go first (drop-off priority)
    sorted_bots = sorted(
        bots,
        key=lambda b: (-sum(useful_inventory(b).values()), b["id"])
    )

    for bot in sorted_bots:
        bot_id = bot["id"]
        bot_pos = tuple(bot["position"])
        inventory = bot["inventory"]
        inv_counter = Counter(inventory)
        useful = useful_inventory(bot)

        # Other bots' current positions (for collision avoidance)
        other_current = {
            tuple(b["position"]) for b in bots if b["id"] != bot_id
        }
        # Cells reserved by bots already processed this round
        reserved = set(reserved_cells.values())
        avoid = other_current | reserved

        has_useful = sum(useful.values()) > 0
        inv_full = len(inventory) >= 3

        # Decide: drop-off or pick-up?
        go_drop = False
        if has_useful:
            # Check if all remaining needed items are accounted for
            # (either in inventories or claimed from map)
            unclaimed_on_map = [
                i for i in needed_map_items if i["id"] not in claimed_item_ids
            ]
            if inv_full or not unclaimed_on_map:
                go_drop = True
            elif sum(still_needed.values()) <= sum(useful.values()):
                go_drop = True

        if go_drop:
            if bot_pos in drop_zones:
                actions.append({"bot": bot_id, "action": "drop_off"})
                reserved_cells[bot_id] = bot_pos
            else:
                # Avoid reserved cells but allow other bots' current cells
                # (they'll move away)
                path = bfs(width, height, walls, bot_pos, drop_zones, reserved)
                if path:
                    action = path[0]
                    npos = next_pos_for_action(bot_pos, action)
                    actions.append({"bot": bot_id, "action": action})
                    reserved_cells[bot_id] = npos
                else:
                    actions.append({"bot": bot_id, "action": "wait"})
                    reserved_cells[bot_id] = bot_pos
            continue

        # Pick-up logic
        if inv_full:
            # Full but no useful items — can't do much, wait
            actions.append({"bot": bot_id, "action": "wait"})
            reserved_cells[bot_id] = bot_pos
            continue

        # Find best unclaimed item to target
        candidates = [
            i for i in needed_map_items if i["id"] not in claimed_item_ids
        ]

        if not candidates:
            # Nothing to pick up; if we have anything useful, go drop off
            if has_useful:
                if bot_pos in drop_zones:
                    actions.append({"bot": bot_id, "action": "drop_off"})
                    reserved_cells[bot_id] = bot_pos
                else:
                    path = bfs(width, height, walls, bot_pos, drop_zones, reserved)
                    if path:
                        action = path[0]
                        npos = next_pos_for_action(bot_pos, action)
                        actions.append({"bot": bot_id, "action": action})
                        reserved_cells[bot_id] = npos
                    else:
                        actions.append({"bot": bot_id, "action": "wait"})
                        reserved_cells[bot_id] = bot_pos
            else:
                actions.append({"bot": bot_id, "action": "wait"})
                reserved_cells[bot_id] = bot_pos
            continue

        # Find nearest candidate (by path length to an adjacent cell)
        best_item = None
        best_path = None
        best_dist = float("inf")

        for item in candidates:
            item_pos = tuple(item["position"])
            adj = adjacent_walkable(item_pos, width, height, walls)
            if not adj:
                continue

            if bot_pos in adj:
                path = []
                dist = 0
            else:
                path = bfs(width, height, walls, bot_pos, adj, reserved)
                dist = len(path) if path is not None else float("inf")

            if dist < best_dist:
                best_dist = dist
                best_item = item
                best_path = path

        if best_item is None:
            actions.append({"bot": bot_id, "action": "wait"})
            reserved_cells[bot_id] = bot_pos
            continue

        claimed_item_ids.add(best_item["id"])

        if best_dist == 0:
            # Already adjacent — pick up
            actions.append({
                "bot": bot_id,
                "action": "pick_up",
                "item_id": best_item["id"],
            })
            reserved_cells[bot_id] = bot_pos
        else:
            action = best_path[0]
            npos = next_pos_for_action(bot_pos, action)
            actions.append({"bot": bot_id, "action": action})
            reserved_cells[bot_id] = npos

    return actions


# ---------------------------------------------------------------------------
# WebSocket runner
# ---------------------------------------------------------------------------

async def run_bot(token: str):
    url = f"{WS_BASE}?token={token}"
    print(f"Connecting to {url}")

    async with websockets.connect(url) as ws:
        print("Connected! Waiting for game state...")
        while True:
            try:
                raw = await ws.recv()
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection closed: {e}")
                break

            msg = json.loads(raw)

            if msg["type"] == "game_over":
                print(f"Game over! Final score: {msg.get('score', '?')}")
                break

            if msg["type"] == "game_state":
                rnd = msg["round"]
                score = msg["score"]
                n_orders = len(msg["orders"])
                active = next((o for o in msg["orders"] if o["status"] == "active"), None)
                needed = []
                if active:
                    req = Counter(active["items_required"])
                    dlv = Counter(active["items_delivered"])
                    needed = list((req - dlv).elements())
                print(
                    f"Round {rnd:>3}/{msg['max_rounds']}  "
                    f"Score: {score:>4}  "
                    f"Orders: {n_orders}  "
                    f"Need: {needed}"
                )

                acts = decide_actions(msg)
                await ws.send(json.dumps({"actions": acts}))

            # Unknown message types — ignore
            else:
                print(f"Unknown message type: {msg.get('type')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 Grocery Bot")
    parser.add_argument("token", help="JWT token from app.ainm.no/challenge")
    args = parser.parse_args()

    asyncio.run(run_bot(args.token))


if __name__ == "__main__":
    main()
