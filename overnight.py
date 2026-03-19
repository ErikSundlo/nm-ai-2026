"""
Overnight watchdog: keeps the localhost.run tunnel alive and monitors training.
Run with: python overnight.py
"""
import subprocess
import threading
import time
import re
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

LOG = Path("overnight.log")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

CURRENT_URL = {"url": None}

def run_tunnel():
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", "80:localhost:8000",
        "nokey@localhost.run"
    ]
    while True:
        log("Starting localhost.run tunnel...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                m = re.search(r"(https://\S+\.lhr\.life)", line)
                if m:
                    url = m.group(1)
                    CURRENT_URL["url"] = url
                    log(f"TUNNEL UP: {url}/solve  <-- update ainm.no with this URL")
        proc.wait()
        CURRENT_URL["url"] = None
        log("Tunnel dropped, reconnecting in 10s...")
        time.sleep(10)


def monitor_training():
    best_pt = Path("runs/detect/runs/task1/groceries_nano/weights/best.pt")
    results_csv = Path("runs/detect/runs/task1/groceries_nano/results.csv")
    packaged_epoch = -1

    while True:
        time.sleep(60)
        if not results_csv.exists():
            continue
        lines = results_csv.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        last = lines[-1].split(",")
        epoch = int(float(last[0]))
        map50 = float(last[6]) if len(last) > 6 else 0

        log(f"Training epoch {epoch}/50  mAP@50={map50:.4f}")

        # Repackage every 5 epochs and at completion
        if epoch > packaged_epoch and (epoch % 5 == 0 or epoch >= 50):
            if best_pt.exists():
                run_py = Path("src/task1/run.py")
                out = Path(f"submissions/task1_epoch{epoch}_map{map50:.3f}.zip")
                out.parent.mkdir(exist_ok=True)
                # Always also update the main submission file
                main_out = Path("submissions/task1_submission.zip")
                for dst in [out, main_out]:
                    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                        z.write(run_py, "run.py")
                        z.write(best_pt, "best.pt")
                log(f"Packaged model → {out.name}  ({main_out.stat().st_size/1e6:.1f}MB)")
                packaged_epoch = epoch

        if epoch >= 50:
            log("Training complete!")
            break


# Start tunnel thread
t = threading.Thread(target=run_tunnel, daemon=True)
t.start()
time.sleep(5)

# Start training monitor thread
m = threading.Thread(target=monitor_training, daemon=True)
m.start()

log("Overnight watchdog running. Press Ctrl+C to stop.")
log(f"Tunnel URL will appear above. Check overnight.log for updates.")

try:
    while True:
        time.sleep(300)
        url = CURRENT_URL.get("url")
        log(f"Heartbeat — tunnel: {url or 'DOWN'}")
except KeyboardInterrupt:
    log("Stopped.")
