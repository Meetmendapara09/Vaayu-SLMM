"""
Kaggle CLI & Python API automated dataset downloader for Vaayu SLMM.
Downloads Glaive Function Calling and MCP Registry datasets.
"""

import os
import sys
import zipfile
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

DATASETS = [
    {
        "dataset": "thedevastator/ai-chatbot-conversational-data",
        "name": "glaive_function_calling",
        "description": "Glaive Function Calling v2 dataset with diverse multi-turn tool calling traces"
    },
    {
        "dataset": "renato2marinho/vinkius-mcp-registry",
        "name": "mcp_registry",
        "description": "Global Model Context Protocol Registry of active MCP tools and servers"
    }
]

def authenticate_kaggle() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    print("[OK] Successfully authenticated with Kaggle API.")
    return api

def download_dataset(api: KaggleApi, dataset_id: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_id} into {dest_folder}...")
    try:
        api.dataset_download_files(dataset_id, path=str(dest_folder), unzip=True, quiet=False)
        print(f"[OK] Downloaded and unzipped {dataset_id}")
    except Exception as e:
        print(f"[WARN] Failed to download {dataset_id}: {e}")

def main():
    print("==================================================")
    print(" Vaayu SLMM: Kaggle Dataset Ingestion Engine")
    print("==================================================")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = authenticate_kaggle()
    
    for item in DATASETS:
        dest = DATA_DIR / item["name"]
        print(f"\nProcessing: {item['description']}")
        download_dataset(api, item["dataset"], dest)
    
    print("\n[OK] Dataset acquisition complete. Files available at:", DATA_DIR)

if __name__ == "__main__":
    main()
