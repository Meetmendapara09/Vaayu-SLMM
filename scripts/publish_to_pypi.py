"""
Automated PyPI publisher for Vaayu SLMM.
Validates build artifacts and uploads them to PyPI using Twine.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload Vaayu to PyPI")
    parser.add_argument("--token", type=str, default=None, help="PyPI API Token (starts with pypi-...)")
    parser.add_argument("--repository", type=str, default="pypi", choices=["pypi", "testpypi"], help="Target repository")
    args = parser.parse_args()

    token = args.token or os.environ.get("TWINE_PASSWORD") or os.environ.get("PYPI_TOKEN")
    
    dist_dir = Path(__file__).resolve().parent.parent / "dist"
    files = list(dist_dir.glob("*"))
    if not files:
        print("[ERROR] No distribution artifacts found in dist/. Run: python -m build")
        sys.exit(1)

    print("Artifacts ready for upload:")
    for f in files:
        print(f" - {f.name} ({f.stat().st_size} bytes)")

    # Validate packages
    print("\nRunning twine check...")
    check_res = subprocess.run([sys.executable, "-m", "twine", "check"] + [str(f) for f in files])
    if check_res.returncode != 0:
        print("[ERROR] Twine validation failed.")
        sys.exit(1)

    if not token:
        print("\n" + "="*60)
        print(" PyPI Authentication Required")
        print("="*60)
        print("To complete publication to https://pypi.org/project/vaayu/:")
        print("1. Get your PyPI API token from: https://pypi.org/manage/account/token/")
        print("2. Run:")
        print("   python scripts/publish_to_pypi.py --token <pypi-token>")
        print("   OR")
        print("   $env:TWINE_USERNAME='__token__'; $env:TWINE_PASSWORD='<pypi-token>'; twine upload dist/*")
        print("="*60 + "\n")
        return

    print(f"\nUploading to {args.repository}...")
    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--username", "__token__",
        "--password", token,
    ]
    if args.repository == "testpypi":
        cmd.extend(["--repository", "testpypi"])
    cmd.extend([str(f) for f in files])

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n[OK] Vaayu successfully published to PyPI!")
        print("  https://pypi.org/project/vaayu/")
        print("\nInstall anywhere with:\n  pip install vaayu\n")
    else:
        print("\n[ERROR] Upload failed. Please verify token permissions.")

if __name__ == "__main__":
    main()