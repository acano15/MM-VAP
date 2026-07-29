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


progress_state = {"last_percent": -1}


def show_progress(block_num, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve, throttled for CI logs."""
    downloaded = block_num * block_size
    percent = min(100, int(downloaded * 100 / total_size)) if total_size > 0 else 0
    if block_num == 0:
        progress_state["last_percent"] = -1
    if percent == progress_state["last_percent"]:
        return
    progress_state["last_percent"] = percent
    sys.stdout.write(f"\rDownloading... {percent:3d}%")
    sys.stdout.flush()


failures = []
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
        # Do not mistake an incomplete download for a valid checkpoint on the
        # next run.
        dest_path.unlink(missing_ok=True)
        failures.append(url)

if failures:
    print(f"Failed to download {len(failures)} checkpoint(s).", file=sys.stderr)
    sys.exit(1)
