#!/usr/bin/env python3
"""
Dump vision_bin/pixel_values.bin for InternVL3_5-1B.

Steps: load image -> resize 448 -> /255 -> ImageNet norm -> write fp16 [1,3,448,448]

Usage:
  python dump_vision_om_inputs.py --image path/img.jpg
  python dump_vision_om_inputs.py --image-dir path/images
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OM = Path(__file__).resolve().parent
SIZE = 448
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pixel_values(image: Path) -> np.ndarray:
    img = Image.open(image).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    x = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    return ((x - MEAN) / STD)[None].astype(np.float16)


def dump_one(image: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = pixel_values(image)
    out = out_dir / "pixel_values.bin"
    arr.tofile(out)
    print(f"{image} -> {out}  shape={arr.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path)
    p.add_argument("--image-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=OM / "vision_bin")
    p.add_argument("--batch-root", type=Path, default=OM / "batch")
    p.add_argument("--skip-exist", action="store_true")
    args = p.parse_args()

    if args.image_dir:
        images = sorted(
            p for p in args.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXT
        )
        for img in images:
            out = args.batch_root / img.stem / "vision_bin"
            if args.skip_exist and (out / "pixel_values.bin").is_file():
                print(f"skip {img.name}")
                continue
            dump_one(img, out)
        return

    image = args.image or (OM.parent / "InternVL3_5-1B-HF/examples/image1.jpg")
    dump_one(image, args.out_dir)


if __name__ == "__main__":
    main()
