"""Download official ImageNet weights without starting training (standard library only)."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = [("alexnet", "alexnet-owt-7be5be79.pth", "7be5be79"),
         ("resnet34", "resnet34-b627a593.pth", "b627a593")]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    folder = ROOT / "weights"
    folder.mkdir(exist_ok=True)
    manifest = []
    for name, filename, prefix in ITEMS:
        url = "https://download.pytorch.org/models/" + filename
        path = folder / filename
        if not path.exists():
            temporary = path.with_suffix(".part")
            print(f"Downloading {name}: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as src, temporary.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            if not sha256(temporary).startswith(prefix):
                raise RuntimeError(f"Checksum mismatch: {temporary}; not installed")
            os.replace(temporary, path)
        digest = sha256(path)
        if not digest.startswith(prefix):
            raise RuntimeError(f"Existing weight file checksum mismatch: {path}")
        manifest.append({"model": name, "weights": "IMAGENET1K_V1", "file": filename,
                         "url": url, "sha256": digest, "bytes": path.stat().st_size})
        print(f"Verified {filename}: {digest}", flush=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
