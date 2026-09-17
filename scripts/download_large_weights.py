"""
Direct, high-speed streaming downloader for Vaayu-Large final weights from Kaggle.
Streams chunks directly to disk with live progress logging.
"""

import os
import sys
import time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def download_file_stream(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to download stream for {target_path.name}...", flush=True)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    total_mb = total_size / (1024 * 1024) if total_size else 0
    print(f"Target: {target_path.name} ({total_mb:.1f} MB)", flush=True)
    
    downloaded = 0
    start_time = time.time()
    last_print = start_time
    
    # Write to a temporary file first then atomic rename
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=16 * 1024 * 1024): # 16MB chunks for maximum throughput
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_size:
                    elapsed = now - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / (elapsed + 1e-5)
                    pct = (downloaded / total_size * 100) if total_size else 0
                    print(f"Progress: {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.1f} MB/s", flush=True)
                    last_print = now
    
    if tmp_path.exists():
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)
                    
    print(f"[OK] Download completed: {target_path.name} ({target_path.stat().st_size / (1024*1024):.1f} MB in {time.time() - start_time:.1f}s)\n", flush=True)

def main():
    api = KaggleApi()
    api.authenticate()
    kernel = "mendaparameet/vaayu-large-slmm-training"
    target_dir = Path("checkpoints/vaayu_large")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel)
    with api.build_kaggle_client() as kaggle:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner_slug
        request.kernel_slug = kernel_slug
        api._set_paging(request, 20, None)
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
        
    print(f"Found {len(resp.files or [])} output files on Kaggle kernel {kernel}:", flush=True)
    for item in resp.files or []:
        fname = item.file_name.split("/")[-1]
        print(f" - {fname}", flush=True)
        if "final" in fname or "config.json" in fname:
            out_file = target_dir / fname
            download_file_stream(item.url, out_file)

    print("[SUCCESS] All required final weights and configuration downloaded!", flush=True)

if __name__ == "__main__":
    main()
