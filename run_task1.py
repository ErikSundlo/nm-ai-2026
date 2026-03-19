"""
Start the Task 1 NorgesGruppen detector server + expose it via serveo.net (SSH tunnel).

Usage:
    python run_task1.py

Environment variables:
    ANTHROPIC_API_KEY   — not required for this task
    PORT                — optional, defaults to 8001
"""
import os
import subprocess
import threading
import time
import re
import uvicorn

PORT = int(os.environ.get("PORT", 8001))


def start_serveo():
    """Open SSH tunnel to serveo.net and print the public URL."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-R", f"80:localhost:{PORT}",
        "serveo.net"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"Forwarding HTTP traffic from (https://\S+)", line)
        if m:
            public_url = m.group(1)
            print(f"\n{'='*60}")
            print(f"  BradskiBeat - NorgesGruppen Detector")
            print(f"  Public URL : {public_url}")
            print(f"  Solve URL  : {public_url}/solve  <- submit this on ainm.no")
            print(f"  Health     : {public_url}/health")
            print(f"{'='*60}\n")


t = threading.Thread(target=start_serveo, daemon=True)
t.start()
time.sleep(3)

from src.task1.server import app
uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
