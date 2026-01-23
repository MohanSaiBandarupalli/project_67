from __future__ import annotations
import argparse
from pathlib import Path
import urllib.request
import zipfile

ML32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ml-32m.zip"
    if not zip_path.exists():
        print(f"[download] {ML32M_URL} -> {zip_path}")
        urllib.request.urlretrieve(ML32M_URL, zip_path)
    else:
        print(f"[download] exists: {zip_path}")

    extract_dir = out_dir / "ml-32m"
    if not extract_dir.exists():
        print(f"[extract] -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    else:
        print(f"[extract] exists: {extract_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/external/movielens"))
    args = ap.parse_args()
    main(args.out)
    
