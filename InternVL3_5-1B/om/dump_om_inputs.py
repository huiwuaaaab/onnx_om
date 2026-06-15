#!/usr/bin/env python3
"""
[本地机] Dump static OM input bins for InternVL3_5-1B.

No transformers dependency: image + prompt preprocessing are implemented manually.
Uses tokenizers + jinja2 chat template + PIL resize/normalize.

Pipeline:
  vision_448:     pixel_values [1,3,448,448]
  mm_proj:        (chained) vision last_hidden_state [1,1025,1024]
  llm_preblock:   input_ids, attention_mask, position_ids (+ image_embeds from mm_proj)
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
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment
from PIL import Image
from tokenizers import Tokenizer

OM_DIR = Path(__file__).resolve().parent
REPO_ROOT = OM_DIR.parent
DEFAULT_MODEL = REPO_ROOT / "InternVL3_5-1B-HF"
DEFAULT_DUMP = OM_DIR / "dump"
# Batch dump / MDC scp 根目录：om/batch/
DEFAULT_BATCH = OM_DIR / "batch"
DEFAULT_PROMPT_DIR = OM_DIR / "prompt_bin"
DEFAULT_PROMPT = "What is shown in this image?"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SEQ_LEN = 512
PAD_TOKEN_ID = 151643
IMAGE_SEQ_LENGTH = 256

PIPELINE_NOTES = {
    "mm_proj": {
        "inputs_from": "vision OM output last_hidden_state",
        "shape": [1, 1025, 1024],
        "dtype": "float16",
    },
    "llm_preblock": {
        "image_embeds_from": "mm_proj OM output",
        "shape": [1, 256, 1024],
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


def _load_model_config(model_dir: Path) -> tuple[dict, dict, dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    pre_cfg = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    proc_cfg = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    return cfg, tok_cfg, pre_cfg, proc_cfg


def _load_tokenizer(model_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def _image_seq_length(model_dir: Path) -> int:
    _, _, _, proc_cfg = _load_model_config(model_dir)
    return int(proc_cfg.get("image_seq_length", IMAGE_SEQ_LENGTH))


def _image_tokens(model_dir: Path) -> tuple[str, str, str]:
    tok_cfg = _load_model_config(model_dir)[1]
    start = tok_cfg["start_image_token"]
    end = tok_cfg["end_image_token"]
    ctx = tok_cfg["context_image_token"]
    return start, end, ctx


def preprocess_image(image_path: Path, model_dir: Path) -> np.ndarray:
    """Vision-only preprocess → pixel_values [1,3,448,448] fp16."""
    _, _, pre_cfg, _ = _load_model_config(model_dir)
    size = pre_cfg["size"]
    height = int(size["height"])
    width = int(size["width"])
    mean = np.array(pre_cfg["image_mean"], dtype=np.float32)[:, None, None]
    std = np.array(pre_cfg["image_std"], dtype=np.float32)[:, None, None]
    rescale = float(pre_cfg.get("rescale_factor", 1.0 / 255.0))

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    if pre_cfg.get("do_rescale", True):
        arr = arr * rescale
    if pre_cfg.get("do_normalize", True):
        arr = (arr - mean) / std
    return arr[None].astype(np.float16)


def _render_chat_text(
    prompt_text: str,
    model_dir: Path,
    *,
    add_generation_prompt: bool = True,
) -> str:
    _, tok_cfg, _, _ = _load_model_config(model_dir)
    template_src = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = env.from_string(template_src).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=tok_cfg.get("bos_token"),
        eos_token=tok_cfg.get("eos_token"),
        pad_token=tok_cfg.get("pad_token"),
    )
    start, end, ctx = _image_tokens(model_dir)
    seq_len = _image_seq_length(model_dir)
    replacement = f"{start}{ctx * seq_len}{end}"
    return text.replace(ctx, replacement, 1)


def build_prompt_inputs(
    image_path: Path,
    prompt_text: str,
    model_dir: Path,
) -> dict[str, np.ndarray]:
    text = _render_chat_text(prompt_text, model_dir)
    print(f"prompt_text: {prompt_text!r}")
    print(f"chat_template:\n{text}")

    tokenizer = _load_tokenizer(model_dir)
    encoded = tokenizer.encode(text)
    seq_len = len(encoded.ids)
    print(f"seq_len (prefill) = {seq_len}")
    if seq_len > MAX_SEQ_LEN:
        raise ValueError(f"seq_len {seq_len} > {MAX_SEQ_LEN}")

    input_ids = np.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, dtype=np.int32)
    attention_mask = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids[0, :seq_len] = encoded.ids
    attention_mask[0, :seq_len] = 1
    position_ids = np.arange(MAX_SEQ_LEN, dtype=np.int32).reshape(1, MAX_SEQ_LEN)

    pixel_values = preprocess_image(image_path, model_dir)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def load_preblock_bins(prompt_dir: Path) -> dict[str, np.ndarray]:
    pre = prompt_dir / "llm_preblock"
    out: dict[str, np.ndarray] = {}
    for name, shape, dtype in (
        ("input_ids", (1, MAX_SEQ_LEN), np.int32),
        ("attention_mask", (1, MAX_SEQ_LEN), np.int32),
        ("position_ids", (1, MAX_SEQ_LEN), np.int32),
    ):
        path = pre / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = np.fromfile(path, dtype=dtype).reshape(shape)
        out[name] = arr
    return out


def print_loaded_prompt(prompt_dir: Path, model_dir: Path) -> None:
    """image-only: show preblock source and decoded prompt."""
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

    meta = {
        "description": "llm_preblock.onnx OM inputs (image_embeds from mm_proj chain)",
        "source": "dump_om_inputs.py (manual preprocess, no transformers)",
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
        "vision_448.onnx OM input",
        {"pixel_values": data["pixel_values"]},
    )
    print(f"  {vision_dir / 'pixel_values.bin'}  shape={data['pixel_values'].shape}")

    if vision_only:
        return

    pre_dir = out_dir / "llm_preblock"
    for name in ("input_ids", "attention_mask", "position_ids"):
        save_bin(data[name], pre_dir / f"{name}.bin")
        print(f"  {pre_dir / f'{name}.bin'}  shape={data[name].shape}")

    meta = {
        "description": "llm_preblock.onnx OM inputs (image_embeds from mm_proj chain)",
        "source": "dump_om_inputs.py (manual preprocess, no transformers)",
        "tensors": {
            name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
            for name in ("input_ids", "attention_mask", "position_ids")
        },
        "upstream": PIPELINE_NOTES["llm_preblock"],
    }
    (pre_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (out_dir / "pipeline.json").write_text(
        json.dumps({"notes": PIPELINE_NOTES}, indent=2), encoding="utf-8"
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


def dump_one(
    image: Path,
    out_dir: Path,
    *,
    mode: str,
    prompt_dir: Path | None,
    prompt_text: str,
    model_dir: Path,
) -> None:
    print(f"mode={mode}  image={image}")
    if mode == "image-only":
        assert prompt_dir is not None
        print_loaded_prompt(prompt_dir, model_dir)
        pixel_values = preprocess_image(image, model_dir)
        dump_all({"pixel_values": pixel_values}, out_dir, vision_only=True)
        print(f"copy preblock: {prompt_dir.resolve()} -> {out_dir.resolve()}\n")
        copy_preblock(prompt_dir, out_dir)
    else:
        print(f"prompt: generate from --prompt (default if omitted)")
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
    p = argparse.ArgumentParser(description="Dump InternVL3_5 OM static input bins")
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

    if args.image_dir:
        run_batch(args)
        return

    if not args.image:
        p.error("need --image or --image-dir")

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
