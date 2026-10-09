"""
Step 1: Download the source datasets from Hugging Face and save them to disk.

Run:
    python scripts/download_datasets.py

Output:
    data/raw/mathvista/   (English + Chinese subset)
    data/raw/xmmmu/       (English, Hindi, Japanese splits)
"""

import os
import re
import time
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, list_repo_files

RAW_DIR = os.path.join("data", "raw")
XMMMU_REPO = "neulab/PangeaBench-xmmmu"

# Filenames in this repo have changed a few times upstream (e.g. "en.parquet"
# vs "en-00000-of-00001.parquet"), so instead of hardcoding one, we look up
# the current file list and match by pattern each time the script runs.
XMMMU_PATTERNS = {
    "en": re.compile(r"^data/en[._-]"),
    "hi": re.compile(r"Hindi\.parquet$"),
    "ja": re.compile(r"Japanese\.parquet$"),
}


def download_with_retry(fn, *args, retries=4, delay=5, **kwargs):
    """Retry wrapper — connection resets are usually transient (antivirus,
    flaky wifi, VPN). Retries with a short backoff before giving up."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt}/{retries} failed ({e.__class__.__name__}: {e}); retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def resolve_xmmmu_filenames():
    """Look up the repo's current file list and match each language to its
    actual filename (avoids hardcoding names that the maintainer renames)."""
    files = download_with_retry(list_repo_files, XMMMU_REPO, repo_type="dataset")
    resolved = {}
    for lang, pattern in XMMMU_PATTERNS.items():
        matches = [f for f in files if pattern.search(f)]
        if not matches:
            print(f"  WARNING: no file matched for '{lang}' — repo file list may have changed again.")
            print(f"  Full file list: {files}")
            continue
        resolved[lang] = matches[0]
    return resolved


def download_mathvista():
    out_dir = os.path.join(RAW_DIR, "mathvista")
    if os.path.exists(out_dir):
        print("MathVista already downloaded, skipping.")
        return
    print("Downloading MathVista (testmini split)...")
    ds = download_with_retry(load_dataset, "AI4Math/MathVista", split="testmini")
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"Saved {len(ds)} examples to {out_dir}")
    if "language" in ds.column_names:
        langs = {}
        for row in ds["language"]:
            langs[row] = langs.get(row, 0) + 1
        print("Language counts:", langs)


def download_xmmmu():
    print("Downloading xMMMU (en, hi, ja splits)...")
    out_dir = os.path.join(RAW_DIR, "xmmmu")
    os.makedirs(out_dir, exist_ok=True)
    filenames = resolve_xmmmu_filenames()
    for split, filename in filenames.items():
        split_dir = os.path.join(out_dir, split)
        if os.path.exists(split_dir):
            print(f"  {split}: already downloaded, skipping.")
            continue
        print(f"  {split}: fetching {filename} ...")
        local_path = download_with_retry(
            hf_hub_download,
            repo_id=XMMMU_REPO,
            repo_type="dataset",
            filename=filename,
        )
        ds = Dataset.from_parquet(local_path)
        os.makedirs(split_dir, exist_ok=True)
        ds.save_to_disk(split_dir)
        print(f"  {split}: saved {len(ds)} examples to {split_dir}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    download_mathvista()
    download_xmmmu()
    print("\nDone. Raw data is under data/raw/.")
    print("Next: python scripts/unify_schema.py")