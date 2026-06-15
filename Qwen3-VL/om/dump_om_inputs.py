#!/usr/bin/env python3
"""
[本地机] Dump static OM input bins for Qwen3-VL.

No transformers dependency: image + prompt preprocessing are implemented manually.
Uses tokenizers + jinja2 chat template + torch (mrope position_ids via llm.py).

Pipeline (default profile 448_512):
  vision_448:     pixel_values [784,1536]
  llm_preblock:   input_ids, attention_mask, position_ids (+ image_embeds from vision)
  llm_block1..3:  (chained on board)
  lm_head:        slice b3 hidden[:, cur_len-1]

Modes:
  image-only (default): vision → out-dir/vision/; copy llm_preblock from prompt_bin/ to out-dir/
  full:               image + prompt bins → dump/

Usage:
  python dump_om_inputs.py --image path/img.jpg
  python dump_om_inputs.py --mode full --prompt "..." --image path/img.jpg
  python dump_om_inputs.py --image-dir path/images
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jinja2 import Environment
from PIL import Image
from tokenizers import Tokenizer

# Default OM layout: 448px vision + 512 seq (override via --profile or QWEN3_EXPORT_PROFILE).
os.environ.setdefault("QWEN3_EXPORT_PROFILE", "448_512")

OM_DIR = Path(__file__).resolve().parent
REPO_ROOT = OM_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_config import ExportProfile, get_export_profile  # noqa: E402
from llm import apply_export_profile, compute_static_position_ids  # noqa: E402

PROFILE: ExportProfile = get_export_profile()
apply_export_profile(PROFILE)
MAX_SEQ_LEN = PROFILE.max_seq_len

DEFAULT_MODEL = REPO_ROOT / "Qwen3-VL-2B-Instruct"
DEFAULT_DUMP = OM_DIR / "dump"
DEFAULT_BATCH = OM_DIR  / "batch"
DEFAULT_PROMPT_DIR = OM_DIR / "prompt_bin"
DEFAULT_PROMPT = "What is shown in this image?"
DEFAULT_IMAGE = (
    "path/to/image.jpg"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = PROFILE.image_size
VISION_OM_LABEL = PROFILE.vision_onnx_name.replace(".onnx", "")
PAD_TOKEN_ID = 151643
IMAGE_PAD_ID = 151655
IMAGE_PAD_COUNT = PROFILE.num_image_tokens

PIPELINE_NOTES = {
    "llm_preblock": {
        "image_embeds_from": "vision OM output merged_hidden_states",
        "shape": [PROFILE.num_image_tokens, 2048],
        "dtype": "float16",
    },
}


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_bin(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)


def write_meta(out_dir: Path, description: str, tensors: dict[str, np.ndarray]) -> None:
    meta = {
        "description": description,
        "source": "dump_om_inputs.py (manual preprocess, no transformers)",
        "tensors": {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in tensors.items()
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_model_config(model_dir: Path) -> tuple[dict, dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    pre_cfg = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    return cfg, tok_cfg, pre_cfg


def _load_tokenizer(model_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def preprocess_image(image_path: Path, model_dir: Path) -> np.ndarray:
    """Vision OM input → pixel_values [num_patches, 1536] fp16."""
    _, _, pre_cfg = _load_model_config(model_dir)
    patch_size = int(pre_cfg["patch_size"])
    temporal_patch_size = int(pre_cfg["temporal_patch_size"])
    merge_size = int(pre_cfg["merge_size"])
    mean = torch.tensor(pre_cfg["image_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(pre_cfg["image_std"], dtype=torch.float32).view(3, 1, 1)

    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    patches = ((arr - mean) / std).unsqueeze(0)

    _, channel, rh, rw = patches.shape
    grid_h, grid_w = rh // patch_size, rw // patch_size
    patches = patches.reshape(
        1,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flat = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(1, grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    )
    return flat[0].numpy().astype(np.float16)


def _render_chat_text(
    prompt_text: str,
    model_dir: Path,
    *,
    add_generation_prompt: bool = True,
) -> str:
    template_obj = json.loads((model_dir / "chat_template.json").read_text(encoding="utf-8"))
    template_src = template_obj["chat_template"]
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return env.from_string(template_src).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        add_vision_id=False,
    )


def _expand_image_pad(token_ids: list[int]) -> list[int]:
    out: list[int] = []
    for tid in token_ids:
        if tid == IMAGE_PAD_ID:
            out.extend([IMAGE_PAD_ID] * IMAGE_PAD_COUNT)
        else:
            out.append(tid)
    return out


def build_prompt_inputs(
    image_path: Path,
    prompt_text: str,
    model_dir: Path,
) -> dict[str, np.ndarray]:
    text = _render_chat_text(prompt_text, model_dir)
    print(f"prompt_text: {prompt_text!r}")
    print(f"chat_template:\n{text}")

    tokenizer = _load_tokenizer(model_dir)
    encoded = _expand_image_pad(tokenizer.encode(text).ids)
    seq_len = len(encoded)
    print(f"seq_len (prefill) = {seq_len}")
    if seq_len > MAX_SEQ_LEN:
        raise ValueError(f"seq_len {seq_len} > {MAX_SEQ_LEN}")

    input_ids = np.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, dtype=np.int32)
    attention_mask = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids[0, :seq_len] = encoded
    attention_mask[0, :seq_len] = 1

    ids_t = torch.from_numpy(input_ids)
    mask_t = torch.from_numpy(attention_mask)
    position_ids = compute_static_position_ids(ids_t, mask_t, "cpu").numpy().astype(np.int32)

    pixel_values = preprocess_image(image_path, model_dir)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def load_preblock_bins(prompt_dir: Path) -> dict[str, np.ndarray]:
    pre = prompt_dir / "llm_preblock"
    pos_shape = (3, 1, MAX_SEQ_LEN)
    out: dict[str, np.ndarray] = {}
    for name, shape, dtype in (
        ("input_ids", (1, MAX_SEQ_LEN), np.int32),
        ("attention_mask", (1, MAX_SEQ_LEN), np.int32),
        ("position_ids", pos_shape, np.int32),
    ):
        path = pre / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = np.fromfile(path, dtype=dtype).reshape(shape)
        out[name] = arr
    return out


def print_loaded_prompt(prompt_dir: Path, model_dir: Path) -> None:
    pre = load_preblock_bins(prompt_dir)
    seq_len = int(pre["attention_mask"].sum())
    print(f"prompt_source: {(prompt_dir / 'llm_preblock').resolve()}")
    print(f"seq_len (prefill) = {seq_len}")
    tokenizer = _load_tokenizer(model_dir)
    ids = pre["input_ids"][0, :seq_len].tolist()
    try:
        decoded = tokenizer.decode(ids)
    except TypeError:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"decode(input_ids[:seq_len]):\n{decoded}")


def copy_preblock(src_root: Path, dst_root: Path) -> None:
    src = src_root / "llm_preblock"
    dst = dst_root / "llm_preblock"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask", "position_ids"):
        shutil.copy2(src / f"{name}.bin", dst / f"{name}.bin")
        print(f"  {dst / f'{name}.bin'}  (from {src / f'{name}.bin'})")
    meta = src / "meta.json"
    if meta.is_file():
        shutil.copy2(meta, dst / "meta.json")
        print(f"  {dst / 'meta.json'}  (from {meta})")
    print()


def dump_preblock(data: dict[str, np.ndarray], prompt_root: Path) -> None:
    pre_dir = prompt_root / "llm_preblock"
    pre_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask", "position_ids"):
        save_bin(data[name], pre_dir / f"{name}.bin")
        print(f"  {pre_dir / f'{name}.bin'}  shape={data[name].shape}")

    seq_len = int(data["attention_mask"].sum())
    meta = {
        "description": "llm_preblock.onnx OM inputs (image_embeds from vision chain)",
        "source": "dump_om_inputs.py (manual preprocess, no transformers)",
        "seq_len": seq_len,
        "tensors": {
            name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
            for name in ("input_ids", "attention_mask", "position_ids")
        },
        "upstream": PIPELINE_NOTES["llm_preblock"],
    }
    (pre_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dump_all(data: dict[str, np.ndarray], out_dir: Path, *, vision_only: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vision_dir = out_dir / "vision"
    save_bin(data["pixel_values"], vision_dir / "pixel_values.bin")
    write_meta(
        vision_dir,
        f"{VISION_OM_LABEL}.onnx OM input",
        {"pixel_values": data["pixel_values"]},
    )
    print(f"  {vision_dir / 'pixel_values.bin'}  shape={data['pixel_values'].shape}")

    if vision_only:
        return

    pre_dir = out_dir / "llm_preblock"
    for name in ("input_ids", "attention_mask", "position_ids"):
        save_bin(data[name], pre_dir / f"{name}.bin")
        print(f"  {pre_dir / f'{name}.bin'}  shape={data[name].shape}")

    seq_len = int(data["attention_mask"].sum())
    meta = {
        "description": "llm_preblock.onnx OM inputs (image_embeds from vision chain)",
        "source": "dump_om_inputs.py (manual preprocess, no transformers)",
        "seq_len": seq_len,
        "tensors": {
            name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
            for name in ("input_ids", "attention_mask", "position_ids")
        },
        "upstream": PIPELINE_NOTES["llm_preblock"],
    }
    (pre_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "pipeline.json").write_text(
        json.dumps({"seq_len": seq_len, "notes": PIPELINE_NOTES}, indent=2),
        encoding="utf-8",
    )


def safe_stem(path: Path | str) -> str:
    base = Path(path).name.rsplit(".", 1)[0]
    s = re.sub(r"[ /:]", "___", base)
    return "".join(c for c in s if c.isalnum() or c in "_.-")


def find_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    return sorted(
        (
            p.resolve()
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )


def configure_profile(profile_name: str | None) -> ExportProfile:
    global PROFILE, MAX_SEQ_LEN, IMAGE_SIZE, IMAGE_PAD_COUNT, VISION_OM_LABEL, PIPELINE_NOTES

    profile = get_export_profile(profile_name)
    apply_export_profile(profile)
    PROFILE = profile
    MAX_SEQ_LEN = profile.max_seq_len
    IMAGE_SIZE = profile.image_size
    IMAGE_PAD_COUNT = profile.num_image_tokens
    VISION_OM_LABEL = profile.vision_onnx_name.replace(".onnx", "")
    PIPELINE_NOTES = {
        "llm_preblock": {
            "image_embeds_from": "vision OM output merged_hidden_states",
            "shape": [profile.num_image_tokens, 2048],
            "dtype": "float16",
        },
    }
    return profile


def dump_one(
    image: Path,
    out_dir: Path,
    *,
    mode: str,
    prompt_dir: Path | None,
    prompt_text: str,
    model_dir: Path,
) -> None:
    print(
        f"profile={PROFILE.name}  image_size={IMAGE_SIZE}  max_seq_len={MAX_SEQ_LEN}  "
        f"image_tokens={IMAGE_PAD_COUNT}"
    )
    print(f"mode={mode}  image={image}")
    if mode == "image-only":
        assert prompt_dir is not None
        print_loaded_prompt(prompt_dir, model_dir)
        pixel_values = preprocess_image(image, model_dir)
        dump_all({"pixel_values": pixel_values}, out_dir, vision_only=True)
        print(f"copy preblock: {prompt_dir.resolve()} -> {out_dir.resolve()}\n")
        copy_preblock(prompt_dir, out_dir)
    else:
        print("prompt: generate from --prompt (default if omitted)")
        data = build_prompt_inputs(image, prompt_text, model_dir)
        dump_all(data, out_dir)
        dump_preblock(data, DEFAULT_PROMPT_DIR)


def run_batch(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir).resolve()
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    images = find_images(image_dir)
    if not images:
        raise SystemExit(f"ERROR: no images under {image_dir}")

    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    if args.mode == "image-only" and prompt_dir is None:
        prompt_dir = DEFAULT_PROMPT_DIR

    _log(f"BATCH_ROOT={batch_root}  images={len(images)}  mode={args.mode}")
    summary = batch_root / "summary_dump.tsv"
    summary.write_text("image\tstem\tstatus\tdump_dir\n", encoding="utf-8")

    for idx, img in enumerate(images, start=1):
        stem = safe_stem(img)
        dump_dir = batch_root / stem / "dump"
        _log(f"========== [{idx}/{len(images)}] {img} -> {dump_dir} ==========")

        if args.skip_exist and (dump_dir / "vision" / "pixel_values.bin").is_file():
            _log("SKIP_EXIST")
            with summary.open("a", encoding="utf-8") as f:
                f.write(f"{img}\t{stem}\tskip_exist\t{dump_dir}\n")
            continue

        dump_one(
            img,
            dump_dir,
            mode=args.mode,
            prompt_dir=prompt_dir,
            prompt_text=args.prompt,
            model_dir=Path(args.model_dir),
        )
        with summary.open("a", encoding="utf-8") as f:
            f.write(f"{img}\t{stem}\tok\t{dump_dir}\n")

    _log(f"Dump done. summary: {summary}")
    _log(f"MDC: RUN_MSAME=1 bash run_om_pipeline.sh --batch-root {batch_root}")


def main() -> None:
    p = argparse.ArgumentParser(description="Dump Qwen3-VL OM static input bins")
    p.add_argument(
        "--profile",
        choices=("256_256", "448_512"),
        default=os.environ.get("QWEN3_EXPORT_PROFILE", "448_512"),
        help="export layout profile (default: 448_512)",
    )
    p.add_argument("--mode", choices=("full", "image-only"), default="image-only")
    p.add_argument("--image", type=Path, help="single image path")
    p.add_argument("--image-dir", type=Path, help="batch: directory of images")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DUMP)
    p.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    p.add_argument("--prompt-dir", type=Path, help="image-only: source dump root with llm_preblock/")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--skip-exist", action="store_true")
    args = p.parse_args()
    configure_profile(args.profile)

    if args.image_dir:
        run_batch(args)
        return

    if not args.image:
        args.image = Path(DEFAULT_IMAGE)

    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    if args.mode == "image-only" and prompt_dir is None:
        prompt_dir = DEFAULT_PROMPT_DIR

    dump_one(
        Path(args.image),
        Path(args.out_dir),
        mode=args.mode,
        prompt_dir=prompt_dir,
        prompt_text=args.prompt,
        model_dir=Path(args.model_dir),
    )
    _log(f"Done. output: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
