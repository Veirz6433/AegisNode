import subprocess
import time
import sys
import os
import signal

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_WAIT = 4

CLIENTS = [
    {"id": 0, "script": "client.py",           "label": "Honest  ✅"},
    {"id": 1, "script": "client.py",           "label": "Honest  ✅"},
    {"id": 2, "script": "malicious_client.py", "label": "ROGUE   ☠️"},
    {"id": 3, "script": "client.py",           "label": "Honest  ✅"},
    {"id": 4, "script": "client.py",           "label": "Honest  ✅"},
]

processes = []


def banner():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              AegisNode — Automated Launch Pad                ║")
    print("║   Federation: 5 Clients (4 Honest + 1 Rogue)                 ║")
    print("║   Strategy  : ByzantineShield (Coordinate-wise Median)       ║")
    print("║   Dataset   : EllipticBitcoin — all 5 shards in use          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


def open_terminal(title: str, command: str) -> subprocess.Popen:
    """Open a new xterm window for the given command."""
    keep_open = f"{command}; echo '\n--- Done. Press Enter to close ---'; read"
    full_cmd  = [
        "xterm",
        "-title", title,
        "-fa", "Monospace",   # clean font
        "-fs", "10",          # font size
        "-bg", "black",
        "-fg", "white",
        "-e", f"bash -c '{keep_open}'"
    ]
    proc = subprocess.Popen(full_cmd, cwd=BASE_DIR)
    return proc


def shutdown(sig=None, frame=None):
    print("\n\n🛑  Shutdown signal received — exiting launcher...")
    print("    Close individual xterm windows manually if needed.")
    sys.exit(0)


def launch_server():
    print("🚀  [1/2] Opening Server terminal...")
    cmd  = f"{PYTHON} {os.path.join(BASE_DIR, 'server.py')}"
    proc = open_terminal("AegisNode - SERVER", cmd)
    processes.append(("Server", proc))
    print(f"    Server xterm opened | PID: {proc.pid}")
    print(f"    Waiting {SERVER_WAIT}s for gRPC to bind to port 8080...\n")
    time.sleep(SERVER_WAIT)


def launch_clients():
    print("🚀  [2/2] Opening Client terminals...\n")
    for client in CLIENTS:
        script = os.path.join(BASE_DIR, client["script"])
        cid    = client["id"]
        label  = client["label"]
        title  = f"AegisNode - Client {cid} {label}"
        cmd    = f"{PYTHON} {script} --client-id {cid}"

        proc = open_terminal(title, cmd)
        processes.append((title, proc))
        print(f"    ✔ Opened: {title} | PID: {proc.pid}")
        time.sleep(0.5)

    print(f"\n✅  All {len(CLIENTS)} client terminals launched.")
    print("📡  Federation is live!\n")
    print("ℹ️   Each xterm window shows its own process logs.")
    print("ℹ️   Press Ctrl+C here to exit the launcher.\n")


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    banner()
    launch_server()
    launch_clients()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        shutdown()
