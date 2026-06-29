"""
download_benchmark.py
---------------------
Downloads GSE194122 NeurIPS 2021 benchmark h5ad files from GEO.
Safe to re-run — skips files that already exist.

Usage:
    python scripts/download_benchmark.py
    python scripts/download_benchmark.py --skip-cite
    python scripts/download_benchmark.py --no-unzip
"""
import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR  = REPO_ROOT / "data" / "benchmark"

BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn"
    "/GSE194122/suppl"
)

FILES = {
    "cite": {
        "gz":   "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
        "h5ad": "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad",
        "desc": "CITE-seq (RNA + protein ADT, ~70k cells)",
    },
    "multiome": {
        "gz":   "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz",
        "h5ad": "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
        "desc": "Multiome (RNA + ATAC, ~69k cells)",
    },
}


def _progress(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {done/1e6:.1f}/{total/1e6:.1f} MB",
              end="", flush=True)


def download(url, dest):
    if dest.exists():
        print(f"  Exists ({dest.stat().st_size/1e6:.0f} MB) — skipping.")
        return
    print(f"  {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n  Saved → {dest.name}")
    except Exception as e:
        print(f"\n  Failed: {e}")
        dest.unlink(missing_ok=True)
        sys.exit(1)


def unzip(gz_path, out_path):
    if out_path.exists():
        print(f"  Already decompressed — skipping.")
        return
    print(f"  Decompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  Done → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cite",     action="store_true")
    parser.add_argument("--skip-multiome", action="store_true")
    parser.add_argument("--no-unzip",      action="store_true")
    parser.add_argument("--keep-gz",       action="store_true")
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    keys = []
    if not args.skip_cite:     keys.append("cite")
    if not args.skip_multiome: keys.append("multiome")

    for key in keys:
        info = FILES[key]
        gz_path = DEST_DIR / info["gz"]
        h5_path = DEST_DIR / info["h5ad"]
        print(f"\n── {info['desc']} ──")
        download(f"{BASE_URL}/{info['gz']}", gz_path)
        if not args.no_unzip:
            unzip(gz_path, h5_path)
            if not args.keep_gz and gz_path.exists():
                gz_path.unlink()
                print("  Removed .gz to save space.")

    print("\nDone.")


if __name__ == "__main__":
    main()
