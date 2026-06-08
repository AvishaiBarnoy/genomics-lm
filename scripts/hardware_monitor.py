import time
import os
import subprocess
from pathlib import Path

CURVES_PATH = "outputs/scores/2026-06-04_stage2.5_6L4H_d256_e10/curves.csv"

def get_ram_stats():
    cmd = "vm_stat | perl -ne '/page\\s+size\\s+of\\s+(\\d+)/ and $s=$1; /Pages\\s+free:\\s+(\\d+)/ and printf \"%.2f\", $1*$s/1024**3;'"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return float(res.stdout.strip())

def monitor():
    print("[*] Hardware Monitor Active.")
    last_size = os.path.getsize(CURVES_PATH) if os.path.exists(CURVES_PATH) else 0
    start_time = time.time()
    
    while True:
        current_ram = get_ram_stats()
        current_size = os.path.getsize(CURVES_PATH) if os.path.exists(CURVES_PATH) else 0
        
        # Check for progress
        if current_size > last_size:
            print(f"[Heartbeat] Step completed. Current RAM: {current_ram} GB. Time since last: {time.time() - start_time:.2f}s")
            last_size = current_size
            start_time = time.time()
        
        # Alert if RAM is critical and speed is dropping
        if current_ram < 0.05:
            print(f"[ALERT] RAM Critical: {current_ram} GB. Monitoring for swap lag...")
            
        time.sleep(60) # Check every minute

if __name__ == "__main__":
    monitor()
