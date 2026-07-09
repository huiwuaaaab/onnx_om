#!/usr/bin/env python3
"""
Dump vision_bin/pixel_values.bin for Qwen3-VL (448_512).

Steps: load image -> resize 448 -> norm -> patchify -> write fp16 [784,1536]

Usage:
  python dump_vision_om_inputs.py --image path/img.jpg
  python dump_vision_om_inputs.py --image-dir path/images
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OM = Path(__file__).resolve().parent
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pixel_values(image: Path, size: int) -> np.ndarray:
    img = Image.open(image).convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img, np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x - MEAN) / STD
    g = size // 16
    x = x.reshape(1, 3, g // 2, 2, 16, g // 2, 2, 16)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(1, g * g, 1536)
    return x[0].numpy().astype(np.float16)


def dump_one(image: Path, out_dir: Path, size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = pixel_values(image, size)
    out = out_dir / "pixel_values.bin"
    arr.tofile(out)
    print(f"{image} -> {out}  shape={arr.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("256_256", "448_512"), default=os.environ.get("QWEN3_EXPORT_PROFILE", "448_512"))
    p.add_argument("--image", type=Path)
    p.add_argument("--image-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=OM / "vision_bin")
    p.add_argument("--batch-root", type=Path, default=OM / "batch")
    p.add_argument("--skip-exist", action="store_true")
    args = p.parse_args()

    size = 256 if args.profile == "256_256" else 448

    if args.image_dir:
        for img in sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXT):
            out = args.batch_root / img.stem / "vision_bin"
            if args.skip_exist and (out / "pixel_values.bin").is_file():
                print(f"skip {img.name}")
                continue
            dump_one(img, out, size)
        return

    image = args.image or (OM.parent.parent / "imgs" / "20260616-111801.jpg")
    dump_one(image, args.out_dir, size)


if __name__ == "__main__":
    main()
