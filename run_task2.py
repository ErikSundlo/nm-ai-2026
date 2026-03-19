"""
Start the Task 2 Tripletex agent server + expose it via serveo.net (SSH tunnel).
No account or token needed — just SSH.

Usage:
    python run_task2.py

Environment variables:
    ANTHROPIC_API_KEY   — required for Claude
    PORT                — optional, defaults to 8000
"""
import os
import subprocess
import threading
import time
import re
import uvicorn

PORT = int(os.environ.get("PORT", 8000))

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ERROR: Set ANTHROPIC_API_KEY env var first.")


def start_tunnel():
    """Open SSH tunnel via localhost.run, auto-restart on disconnect."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", f"80:localhost:{PORT}",
        "nokey@localhost.run"
    ]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            # localhost.run format: "tunneled with tls termination, https://xxx.lhr.life"
            m = re.search(r"(https://\S+\.lhr\.life)", line)
            if m:
                public_url = m.group(1)
                print(f"\n{'='*60}")
                print(f"  BradskiBeat - Tripletex Agent")
                print(f"  Public URL : {public_url}")
                print(f"  Solve URL  : {public_url}/solve  <- submit this on ainm.no")
                print(f"  Health     : {public_url}/health")
                print(f"{'='*60}\n", flush=True)
        proc.wait()
        print("[tunnel] disconnected, reconnecting in 5s...", flush=True)
        time.sleep(5)


# Start tunnel in background thread
t = threading.Thread(target=start_tunnel, daemon=True)
t.start()
time.sleep(5)  # Give tunnel a moment to connect

# Run server
from src.task2.server import app
uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
