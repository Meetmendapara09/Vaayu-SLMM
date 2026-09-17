"""
Local orchestrator to push, monitor, and retrieve Vaayu-Large (492M) training from Kaggle online GPUs.
"""

import subprocess
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent.parent / "kaggle_run_large"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

def run_cmd(cmd: list):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print(result.stdout)
    return result

def main():
    print("==================================================")
    print(" Vaayu-Large SLMM (492M): Kaggle GPU Orchestrator")
    print("==================================================")
    
    kernel_id = "mendaparameet/vaayu-large-slmm-training"
    
    print("\n1. Pushing training kernel to Kaggle...")
    run_cmd(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)])

    print(f"\n2. Monitoring kernel status for {kernel_id}...")
    run_cmd(["kaggle", "kernels", "status", kernel_id])

    print(f"\nTo check status anytime, run:\n  kaggle kernels status {kernel_id}")
    print(f"\nTo fetch trained weights after completion, run:\n  kaggle kernels output {kernel_id} -p checkpoints/")

if __name__ == "__main__":
    main()