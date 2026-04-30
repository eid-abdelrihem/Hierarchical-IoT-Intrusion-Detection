"""
start.py
========
Launches the IoT IDS Stream Simulator + Streamlit Dashboard in one command.

Usage:
    python Inference/start.py
"""

import subprocess, sys, time, os, webbrowser, signal, threading

BASE       = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable
DASHBOARD  = os.path.join(BASE, 'dashboard.py')
SIMULATOR  = os.path.join(BASE, 'stream_simulator.py')
URL        = "http://localhost:8501"

procs = []

def kill_all(sig=None, frame=None):
    print("\n\nShutting down…")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, kill_all)

print("="*55)
print("  🛡️  IoT IDS — Starting All Services")
print("="*55)

# 1. Stream Simulator
print("\n[1/2] Starting stream simulator…")
sim = subprocess.Popen(
    [PYTHON, SIMULATOR],
    cwd=BASE,
    env={**os.environ, 'PYTHONUTF8': '1'}
)
procs.append(sim)
time.sleep(1)

# 2. Streamlit Dashboard
print("[2/2] Starting Streamlit dashboard…")
dash = subprocess.Popen(
    [PYTHON, '-m', 'streamlit', 'run', DASHBOARD,
     '--server.port', '8501',
     '--server.address', '127.0.0.1',
     '--server.headless', 'true'],
    cwd=BASE,
    env={**os.environ, 'PYTHONUTF8': '1'}
)
procs.append(dash)

# 3. Open browser after 3 seconds
def open_browser():
    time.sleep(3)
    print(f"\n✅ Dashboard ready → {URL}\n")
    webbrowser.open(URL)

threading.Thread(target=open_browser, daemon=True).start()

print("\n[Press Ctrl+C to stop everything]\n")

# Keep alive — Streamlit restarts itself on error, so only kill if simulator dies
while True:
    time.sleep(2)
    # Only shut down if the stream simulator dies (critical process)
    if sim.poll() is not None:
        print(f"\n⚠️  Stream simulator stopped. Shutting down.")
        kill_all()
    # Streamlit auto-restarts itself — don't kill it

