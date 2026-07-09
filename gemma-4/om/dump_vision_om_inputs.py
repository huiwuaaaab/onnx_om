#!/usr/bin/env python3
"""
Dump vision OM input bins for Gemma-4.

Steps: load image -> resize 768 -> patchify -> pad -> write 2 bins

Single: vision_bin/pixel_values.bin + image_position_ids.bin
Batch:  batch/<stem>/vision_bin/*.bin   (<stem> = image filename stem, e.g. 20260616-111801)

Preblock: shared prompt_bin/ (dump_llm_preblock_inputs.py)

Usage:
  python dump_vision_om_inputs.py --image path/img.jpg
  python dump_vision_om_inputs.py --image-dir ../../imgs
  python dump_vision_om_inputs.py                    # batch dump ../../imgs -> batch/
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

OM = Path(__file__).resolve().parent
DEFAULT_OUT = OM / "vision_bin"
DEFAULT_BATCH = OM / "batch"
DEFAULT_IMGS = OM.parent.parent / "imgs"
DEFAULT_IMAGE = DEFAULT_IMGS / "20260616-111801.jpg"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed gemma-4-E2B-it vision preprocess
IMAGE_SIZE = (768, 768)
PATCH_SIZE = 16
POOL_SIZE = 3
MAX_SOFT_TOKENS = 280
MAX_PATCHES = MAX_SOFT_TOKENS * POOL_SIZE**2  # 2520
RESCALE = 1.0 / 255.0


def _target_size(h: int, w: int) -> tuple[int, int]:
    target_px = MAX_PATCHES * PATCH_SIZE * PATCH_SIZE
    factor = math.sqrt(target_px / (h * w))
    side = POOL_SIZE * PATCH_SIZE
    th = int(math.floor(factor * h / side)) * side
    tw = int(math.floor(factor * w / side)) * side
    max_side = (MAX_PATCHES // POOL_SIZE**2) * side
    if th == 0 and tw == 0:
        raise ValueError("resize to 0x0")
    if th == 0:
        th, tw = side, min(int(math.floor(w / h)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(h / w)) * side, max_side)
    if th * tw > target_px:
        raise ValueError(f"resize [{h}x{w}] -> [{th}x{tw}] exceeds {MAX_PATCHES} patches")
    return th, tw


def vision_bins(image: Path) -> dict[str, np.ndarray | int]:
    img = Image.open(image).convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
    arr = np.array(img, np.float32).transpose(2, 0, 1) * RESCALE
    h, w = arr.shape[1], arr.shape[2]

    th, tw = _target_size(h, w)
    if th != h or tw != w:
        arr = np.array(img.resize((tw, th), Image.BICUBIC), np.float32).transpose(2, 0, 1) * RESCALE

    gh, gw = arr.shape[1] // PATCH_SIZE, arr.shape[2] // PATCH_SIZE
    patches = arr.reshape(3, gh, PATCH_SIZE, gw, PATCH_SIZE).transpose(1, 3, 2, 4, 0).reshape(gh * gw, -1)
    num_soft = patches.shape[0] // POOL_SIZE**2

    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh), indexing="xy")
    pos = np.stack([gx, gy], axis=-1).reshape(patches.shape[0], 2)
    pad_n = MAX_PATCHES - patches.shape[0]
    if pad_n > 0:
        patches = np.pad(patches, [(0, pad_n), (0, 0)])
        pos = np.pad(pos, [(0, pad_n), (0, 0)], constant_values=-1)

    return {
        "pixel_values": patches.astype(np.float16)[None],
        "image_position_ids": pos.astype(np.int32)[None],
        "num_soft_tokens": int(num_soft),
    }


def dump_one(image: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = vision_bins(image)
    for name in ("pixel_values", "image_position_ids"):
        arr = data[name]
        path = out_dir / f"{name}.bin"
        np.ascontiguousarray(arr).tofile(path)
        print(f"  {path}  shape={arr.shape}")
    print(f"num_soft_tokens={data['num_soft_tokens']}")
    return int(data["num_soft_tokens"])


def dump_batch(image_dir: Path, batch_root: Path, *, skip_exist: bool, stem: str = "") -> None:
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXT)
    if stem:
        images = [p for p in images if p.stem == stem]
    if not images:
        raise SystemExit(f"no images under {image_dir}" + (f" stem={stem}" if stem else ""))

    print(f"BATCH_ROOT={batch_root}  images={len(images)}")
    for img in images:
        out = batch_root / img.stem / "vision_bin"
        if skip_exist and (out / "pixel_values.bin").is_file():
            print(f"skip {img.name}")
            continue
        print(f"image={img}")
        dump_one(img, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, help="single image -> --out-dir")
    p.add_argument("--image-dir", type=Path, help=f"batch images dir (default: {DEFAULT_IMGS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    p.add_argument("--stem", default="", help="batch: only this image stem")
    p.add_argument("--skip-exist", action="store_true")
    args = p.parse_args()

    if args.image:
        print(f"image={args.image}")
        dump_one(args.image, args.out_dir)
        return

    image_dir = args.image_dir or DEFAULT_IMGS
    dump_batch(image_dir, args.batch_root, skip_exist=args.skip_exist, stem=args.stem)


if __name__ == "__main__":
    main()
