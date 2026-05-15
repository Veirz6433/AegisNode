import subprocess
import time
import os
import pandas as pd
import signal

# --- Config ---
PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_experiment(exp_id, robust_flag, num_honest, include_rogue):
    print(f"\n🚀 STARTING EXPERIMENT: {exp_id} (Robust={robust_flag})")
    processes = []

    # 1. Launch Server
    srv_cmd = [PYTHON, "server.py", "--id", exp_id, "--robust", str(robust_flag)]
    srv_proc = subprocess.Popen(srv_cmd)
    processes.append(srv_proc)
    time.sleep(5)

    # 2. Launch Honest Clients
    for i in range(num_honest):
        c_cmd = [PYTHON, "client.py", "--client-id", str(i)]
        processes.append(subprocess.Popen(c_cmd, stdout=subprocess.DEVNULL))
    
    # 3. Launch Rogue Client if needed
    if include_rogue:
        r_cmd = [PYTHON, "malicious_client.py", "--client-id", "99"]
        processes.append(subprocess.Popen(r_cmd, stdout=subprocess.DEVNULL))

    print(f"📡 All {len(processes)-1} clients live. Waiting for 50 rounds...")
    
    # Wait for server to finish (it exits after 50 rounds)
    srv_proc.wait()

    # Cleanup clients
    for p in processes:
        p.terminate()
    print(f"✅ Experiment {exp_id} Complete. Cleanup finished.")

if __name__ == "__main__":
    # RUN THE SUITE
    run_experiment("baseline", robust_flag=0, num_honest=4, include_rogue=False)
    run_experiment("attacked", robust_flag=0, num_honest=4, include_rogue=True)
    run_experiment("aegisnode", robust_flag=1, num_honest=4, include_rogue=True)

    # PRINT FINAL TABLE
    print("\n\n" + "="*60)
    print("      FINAL RESEARCH METRICS FOR AEGISNODE PAPER")
    print("="*60)
    print(f"{'Experiment':<15} | {'Robust?':<8} | {'Rogue?':<8} | {'Final Accuracy'}")
    print("-" * 60)

    for eid in ["baseline", "attacked", "aegisnode"]:
        try:
            df = pd.read_csv(f"results_{eid}.csv")
            final_acc = df["accuracy"].iloc[-1]
            robust = "Yes" if eid == "aegisnode" else "No"
            rogue = "No" if eid == "baseline" else "Yes"
            print(f"{eid:<15} | {robust:<8} | {rogue:<8} | {final_acc:.2%}")
        except:
            print(f"{eid:<15} | Error reading results.")

    print("="*60)