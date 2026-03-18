# -*- coding: utf-8 -*-
"""
NM i AI 2026 - Grocery Bot GUI
Visual interface: watch your bot play in real time.

Usage:
  python grocery_gui.py                  # prompts for token
  python grocery_gui.py <token>          # start directly
"""

import asyncio
import json
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from collections import Counter

import websockets

# Import bot logic from grocery_bot.py
from grocery_bot import decide_actions

WS_BASE = "wss://game.ainm.no/ws"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_BG        = "#1a1a2e"   # dark navy background
C_FLOOR     = "#16213e"   # dark cell
C_WALL      = "#5c4033"   # brown shelf
C_DROPOFF   = "#1b4332"   # dark green drop-off zone
C_DROPOFF_B = "#40916c"   # border of drop-off
C_ITEM_BG   = "#f4a261"   # item background
C_ITEM_TXT  = "#1a1a2e"
C_BOT_COLS  = [
    "#e63946", "#457b9d", "#f4d35e", "#06d6a0",
    "#9b5de5", "#f15bb5", "#00bbf9", "#fee440",
    "#fb5607", "#8338ec", "#3a86ff", "#ffbe0b",
    "#ff006e", "#06d6a0", "#118ab2", "#073b4c",
    "#ffd166", "#06d6a0", "#ef476f", "#118ab2",
]
C_PANEL_BG  = "#0f3460"
C_TEXT      = "#e0e0e0"
C_ACCENT    = "#e94560"


CELL = 36          # pixels per cell
PADDING = 12       # outer padding


def item_emoji(item_type: str) -> str:
    """Return a short label for an item type."""
    table = {
        "milk": "🥛", "bread": "🍞", "eggs": "🥚", "cheese": "🧀",
        "butter": "🧈", "apple": "🍎", "banana": "🍌", "orange": "🍊",
        "tomato": "🍅", "potato": "🥔", "carrot": "🥕", "onion": "🧅",
        "chicken": "🍗", "beef": "🥩", "fish": "🐟", "yogurt": "🍶",
        "juice": "🧃", "water": "💧", "soda": "🥤", "coffee": "☕",
        "tea": "🍵", "cereal": "🌾",
    }
    return table.get(item_type.lower(), item_type[:3].upper())


# ---------------------------------------------------------------------------
# Game renderer (Canvas)
# ---------------------------------------------------------------------------

class GameCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C_BG, highlightthickness=0, **kwargs)
        self.state = None

    def render(self, state):
        self.state = state
        self.delete("all")
        if not state:
            return

        grid = state["grid"]
        W, H = grid["width"], grid["height"]
        walls = set(tuple(w) for w in grid["walls"])

        cw = W * CELL + 2 * PADDING
        ch = H * CELL + 2 * PADDING
        self.config(width=cw, height=ch)

        drop_zones = set(
            tuple(z) for z in state.get("drop_off_zones", [state["drop_off"]])
        )

        # Draw cells
        for y in range(H):
            for x in range(W):
                pos = (x, y)
                px = PADDING + x * CELL
                py = PADDING + y * CELL

                if pos in walls:
                    color = C_WALL
                    self.create_rectangle(
                        px, py, px + CELL, py + CELL,
                        fill=color, outline="#3d2b1f", width=1
                    )
                elif pos in drop_zones:
                    self.create_rectangle(
                        px, py, px + CELL, py + CELL,
                        fill=C_DROPOFF, outline=C_DROPOFF_B, width=2
                    )
                    self.create_text(
                        px + CELL // 2, py + CELL // 2,
                        text="📦", font=("", int(CELL * 0.45))
                    )
                else:
                    self.create_rectangle(
                        px, py, px + CELL, py + CELL,
                        fill=C_FLOOR, outline="#0d1b2a", width=1
                    )

        # Draw items (on shelves — just draw on the wall cell)
        items_on_map = {item["id"]: item for item in state["items"]}
        for item in state["items"]:
            x, y = item["position"]
            px = PADDING + x * CELL
            py = PADDING + y * CELL
            margin = 4
            self.create_rectangle(
                px + margin, py + margin,
                px + CELL - margin, py + CELL - margin,
                fill=C_ITEM_BG, outline="#c77c30", width=1,
            )
            label = item_emoji(item["type"])
            self.create_text(
                px + CELL // 2, py + CELL // 2,
                text=label,
                font=("", int(CELL * 0.38)),
                fill=C_ITEM_TXT,
            )

        # Draw bots
        for bot in state["bots"]:
            x, y = bot["position"]
            px = PADDING + x * CELL
            py = PADDING + y * CELL
            color = C_BOT_COLS[bot["id"] % len(C_BOT_COLS)]
            r = CELL // 2 - 3
            cx_ = px + CELL // 2
            cy_ = py + CELL // 2
            self.create_oval(
                cx_ - r, cy_ - r, cx_ + r, cy_ + r,
                fill=color, outline="white", width=2,
            )
            self.create_text(
                cx_, cy_,
                text=str(bot["id"]),
                font=("Consolas", int(CELL * 0.32), "bold"),
                fill="white",
            )
            # Inventory dots above bot
            inv = bot["inventory"]
            dot_r = 4
            for i, item_type in enumerate(inv[:3]):
                dot_x = cx_ - (len(inv) - 1) * 6 + i * 12
                dot_y = cy_ - r - 7
                self.create_oval(
                    dot_x - dot_r, dot_y - dot_r,
                    dot_x + dot_r, dot_y + dot_r,
                    fill=C_ITEM_BG, outline="#c77c30", width=1,
                )


# ---------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------

class SidePanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C_PANEL_BG, **kwargs)
        self._build()

    def _label(self, text, size=11, bold=False, color=C_TEXT):
        weight = "bold" if bold else "normal"
        return tk.Label(
            self, text=text, bg=C_PANEL_BG, fg=color,
            font=("Consolas", size, weight), anchor="w"
        )

    def _build(self):
        pad = dict(padx=12, pady=3, sticky="w")

        self._label("NM i AI 2026", 14, bold=True, color=C_ACCENT).grid(row=0, **pad)
        self._label("Grocery Bot", 11, color="#aaaaaa").grid(row=1, **pad)

        ttk.Separator(self, orient="horizontal").grid(
            row=2, sticky="ew", padx=8, pady=6
        )

        self.var_round  = tk.StringVar(value="Round: —")
        self.var_score  = tk.StringVar(value="Score: —")
        self.var_status = tk.StringVar(value="Status: connecting…")

        self._dyn_label(self.var_round,  3, 13, bold=True)
        self._dyn_label(self.var_score,  4, 13, bold=True, color="#f4d35e")
        self._dyn_label(self.var_status, 5, 10, color="#aaaaaa")

        ttk.Separator(self, orient="horizontal").grid(
            row=6, sticky="ew", padx=8, pady=6
        )

        self._label("Active order:", 10, bold=True).grid(row=7, **pad)
        self.order_frame = tk.Frame(self, bg=C_PANEL_BG)
        self.order_frame.grid(row=8, padx=12, pady=2, sticky="w")

        ttk.Separator(self, orient="horizontal").grid(
            row=9, sticky="ew", padx=8, pady=6
        )

        self._label("Bots:", 10, bold=True).grid(row=10, **pad)
        self.bot_frame = tk.Frame(self, bg=C_PANEL_BG)
        self.bot_frame.grid(row=11, padx=12, pady=2, sticky="w")

        ttk.Separator(self, orient="horizontal").grid(
            row=12, sticky="ew", padx=8, pady=6
        )

        self.var_log = tk.StringVar(value="")
        self._dyn_label(self.var_log, 13, 9, color="#888888")

    def _dyn_label(self, var, row, size=11, bold=False, color=C_TEXT):
        weight = "bold" if bold else "normal"
        lbl = tk.Label(
            self, textvariable=var, bg=C_PANEL_BG, fg=color,
            font=("Consolas", size, weight), anchor="w"
        )
        lbl.grid(row=row, padx=12, pady=2, sticky="w")
        return lbl

    def update(self, state):
        if state is None:
            return

        rnd = state["round"]
        max_rnd = state["max_rounds"]
        score = state["score"]

        self.var_round.set(f"Round: {rnd} / {max_rnd}")
        self.var_score.set(f"Score: {score}")
        self.var_status.set(f"Status: playing")

        # Active order
        for w in self.order_frame.winfo_children():
            w.destroy()

        active = next((o for o in state["orders"] if o["status"] == "active"), None)
        if active:
            req = Counter(active["items_required"])
            dlv = Counter(active["items_delivered"])
            needed = req - dlv
            row = 0
            for item_type, count in req.items():
                have = dlv.get(item_type, 0)
                color = "#40916c" if have >= count else C_TEXT
                txt = f"{item_emoji(item_type)} {item_type}  {have}/{count}"
                tk.Label(
                    self.order_frame, text=txt, bg=C_PANEL_BG, fg=color,
                    font=("Consolas", 10), anchor="w"
                ).grid(row=row, sticky="w")
                row += 1

        # Bots
        for w in self.bot_frame.winfo_children():
            w.destroy()

        for i, bot in enumerate(state["bots"]):
            color = C_BOT_COLS[bot["id"] % len(C_BOT_COLS)]
            inv_str = ", ".join(
                item_emoji(t) for t in bot["inventory"]
            ) if bot["inventory"] else "empty"
            txt = f"Bot {bot['id']}  [{inv_str}]"
            tk.Label(
                self.bot_frame, text=txt, bg=C_PANEL_BG, fg=color,
                font=("Consolas", 10), anchor="w"
            ).grid(row=i, sticky="w")

    def set_log(self, text):
        self.var_log.set(text)

    def set_status(self, text):
        self.var_status.set(f"Status: {text}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, token=None):
        super().__init__()
        self.title("NM i AI 2026 — Grocery Bot")
        self.configure(bg=C_BG)
        self.resizable(True, True)

        self._state_queue = queue.Queue()
        self._token = token
        self._ws_thread = None
        self._running = False

        self._build_ui()

        if token:
            self._start_game(token)
        else:
            self.after(200, self._prompt_token)

    def _build_ui(self):
        # Side panel (left)
        self.panel = SidePanel(self)
        self.panel.grid(row=0, column=0, sticky="ns", padx=(8, 0), pady=8)

        # Scrollable canvas area
        canvas_outer = tk.Frame(self, bg=C_BG)
        canvas_outer.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        scroll_y = ttk.Scrollbar(canvas_outer, orient="vertical")
        scroll_x = ttk.Scrollbar(canvas_outer, orient="horizontal")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

        self.game_canvas = GameCanvas(
            canvas_outer,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
        )
        self.game_canvas.grid(row=0, column=0, sticky="nsew")
        canvas_outer.rowconfigure(0, weight=1)
        canvas_outer.columnconfigure(0, weight=1)

        scroll_y.config(command=self.game_canvas.yview)
        scroll_x.config(command=self.game_canvas.xview)

        self.game_canvas.bind("<Configure>", self._on_canvas_resize)

        # Bottom toolbar
        bar = tk.Frame(self, bg=C_PANEL_BG)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        tk.Button(
            bar, text="New Game", bg=C_ACCENT, fg="white",
            font=("Consolas", 10, "bold"), relief="flat",
            command=self._prompt_token, padx=10,
        ).pack(side="left", padx=8, pady=4)

        self.btn_stop = tk.Button(
            bar, text="Stop", bg="#555", fg="white",
            font=("Consolas", 10), relief="flat",
            command=self._stop_game, padx=10,
        )
        self.btn_stop.pack(side="left", padx=4, pady=4)

        self.status_bar = tk.Label(
            bar, text="Ready", bg=C_PANEL_BG, fg="#888",
            font=("Consolas", 9), anchor="e"
        )
        self.status_bar.pack(side="right", padx=8)

    def _on_canvas_resize(self, event=None):
        self.game_canvas.configure(scrollregion=self.game_canvas.bbox("all"))

    def _prompt_token(self):
        token = simpledialog.askstring(
            "Enter Token",
            "Paste your JWT token from app.ainm.no/challenge\n(click Play on a map to get one):",
            parent=self,
        )
        if token and token.strip():
            self._start_game(token.strip())

    def _start_game(self, token):
        self._stop_game()
        self._token = token
        self._running = True
        self.panel.set_status("connecting…")
        self.status_bar.config(text=f"Token: {token[:20]}…")

        self._ws_thread = threading.Thread(
            target=self._ws_worker, args=(token,), daemon=True
        )
        self._ws_thread.start()
        self.after(50, self._poll_queue)

    def _stop_game(self):
        self._running = False

    def _ws_worker(self, token):
        """Run asyncio WebSocket loop in background thread."""
        asyncio.run(self._ws_loop(token))

    async def _ws_loop(self, token):
        url = f"{WS_BASE}?token={token}"
        try:
            async with websockets.connect(url) as ws:
                self._state_queue.put(("status", "connected"))
                while self._running:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self._state_queue.put(("status", "disconnected"))
                        break

                    msg = json.loads(raw)

                    if msg["type"] == "game_over":
                        self._state_queue.put(("game_over", msg))
                        break

                    if msg["type"] == "game_state":
                        self._state_queue.put(("state", msg))
                        actions = decide_actions(msg)
                        await ws.send(json.dumps({"actions": actions}))

        except Exception as e:
            self._state_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, data = self._state_queue.get_nowait()

                if kind == "state":
                    self.game_canvas.render(data)
                    self.game_canvas.configure(
                        scrollregion=self.game_canvas.bbox("all")
                    )
                    self.panel.update(data)

                elif kind == "game_over":
                    score = data.get("score", "?")
                    self.panel.set_status(f"game over — score {score}")
                    self.status_bar.config(text=f"Final score: {score}")
                    messagebox.showinfo(
                        "Game Over",
                        f"Final score: {score}\n\nClick 'New Game' to play again.",
                        parent=self,
                    )
                    self._running = False

                elif kind == "status":
                    self.panel.set_status(data)
                    self.status_bar.config(text=data)

                elif kind == "error":
                    self.panel.set_status(f"error: {data}")
                    self.status_bar.config(text=f"Error: {data}")
                    messagebox.showerror("Connection Error", data, parent=self)
                    self._running = False

        except queue.Empty:
            pass

        if self._running:
            self.after(50, self._poll_queue)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else None
    app = App(token=token)
    app.mainloop()


if __name__ == "__main__":
    main()
