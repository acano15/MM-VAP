# -*- coding: utf-8 -*-
import urllib.request
from pathlib import Path
import sys

urls = [
    "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_small.pt",
    "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt"
]

base_path = Path(__file__).resolve().parent.parent
target_dir = base_path / "pretrained_models" / "whisper_flamingo"
target_dir.mkdir(parents=True, exist_ok=True)


def show_progress(block_num, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading... {percent:6.2f}%")
    sys.stdout.flush()


for url in urls:
    file_name = url.split("/")[-1]
    dest_path = target_dir / file_name

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Skipping {file_name} (already exists: {dest_path})")
        continue

    print(f"Downloading {file_name} into {target_dir} ...")
    try:
        urllib.request.urlretrieve(url, dest_path, show_progress)
        print(f"\nSaved to: {dest_path}\n")
    except Exception as e:
        print(f"\nFailed to download {url}: {e}\n")
