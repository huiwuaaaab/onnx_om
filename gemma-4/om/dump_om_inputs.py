#!/usr/bin/env python3
"""
Dump frontend input .bin files for OM model testing.

[本地机] 图像/prompt 预处理 → 静态 bin；不在 MDC 上跑。
推理与中间 OM I/O 在 MDC 上由 run_om_pipeline.sh 完成；
结果解析在本地机用 parse_state.py 读取 om_output/。

No transformers dependency: image + prompt preprocessing are implemented manually.
Uses tokenizers + jinja2 chat template + PLE lookup table.

Pipeline (OM 链式喂数):
  vision:     pixel_values, image_position_ids
  mm_proj:    (无独立 input bin) ← 使用 vision OM 输出 hidden_states
  llm_preblock: input_ids, attention_mask, per_layer_inputs, position_ids
                (无 image_embeds bin) ← 使用 mm_proj OM 输出 hidden_states
  llm_block_*:  (无 input bin) ← preblock + 上游 block 在板端串联
  lm_head:      切 b7 hidden[:, cur_len-1] → logits
  assistant:    不在此 dump；板端由 om_bin_utils_it_assistant.py prepare-assistant-input-chain
                从主链 b7/b3 输出 + llm_preblock state 拼装

Modes:
  image-only (default): vision → out-dir/vision/; copy llm_preblock from prompt_bin/ to out-dir/
  full:               preprocess_image() + build_prompt_bins() → vision + llm_preblock

Single image:
  python om/dump_om_inputs.py --image /path/to.jpg
  python om/dump_om_inputs.py --mode full --prompt-text "..." --image img.jpg

Batch (directory of images -> om/batch/<stem>/dump/):
  python om/dump_om_inputs.py --image-dir /path/to/images
  python om/dump_om_inputs.py --image-dir /path/to/images --batch-root batch
  python om/dump_om_inputs.py --image-dir imgs --skip-exist

Default single output (same as run_om_pipeline.sh DUMP_ROOT):
  om/dump/vision/
  om/dump/llm_preblock/          (image-only: copied from prompt_bin/)
  om/prompt_bin/llm_preblock/    (default shared prompt)
  om/ple_table/embed_tokens_per_layer.bin  (与 dump/ 同级，缺失时自动生成一次)

PLE only:
  python om/dump_om_inputs.py --ple-only
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_DUMP_ROOT = OM_DIR / "dump"
MODEL_DIR = REPO_ROOT / "gemma-4-E2B-it"
DEFAULT_IMAGE = (
    "/e-vepfs-01/perception/wuhui/InternVL3_5-1B/InternVL3_5-1B-HF/examples/image1.jpg"
)
DEFAULT_PLE_DIR = OM_DIR / "ple_table"
DEFAULT_PLE_TABLE = DEFAULT_PLE_DIR / "embed_tokens_per_layer.bin"
DEFAULT_SAFETENSORS = MODEL_DIR / "model.safetensors"
DEFAULT_CONFIG = MODEL_DIR / "config.json"
DEFAULT_PROMPT_DIR = OM_DIR / "prompt_bin"
# Batch dump / MDC scp 根目录：om/batch/
DEFAULT_BATCH_ROOT = OM_DIR / "batch"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_SEQ_LEN = 512
NUM_LAYERS = 35
PLE_DIM = 256

def save_bin(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)


def load_bin(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    want = int(np.prod(shape))
    if arr.size != want:
        raise ValueError(f"{path}: size {arr.size} != expected {want} for shape {shape}")
    return arr.reshape(shape)


def _load_model_config(model_dir: Path) -> tuple[dict, dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    proc_cfg = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    return cfg, tok_cfg, proc_cfg


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to 0 x 0 image; check patch_size / max_soft_tokens."
        )
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(
            int(math.floor(width / height)) * side_mult,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult,
            max_side_length,
        )
    if target_height * target_width > target_px:
        raise ValueError(
            f"Resize [{height}x{width}] -> [{target_height}x{target_width}] "
            f"exceeds {max_patches} patches"
        )
    return target_height, target_width


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


def pad_along_first_dim(
    image: np.ndarray, positions: np.ndarray, target_length: int
) -> tuple[np.ndarray, np.ndarray]:
    padding_length = target_length - image.shape[0]
    if padding_length > 0:
        image = np.pad(
            image,
            [(0, padding_length)] + [(0, 0)] * (image.ndim - 1),
            mode="constant",
            constant_values=0,
        )
        positions = np.pad(positions, [(0, padding_length), (0, 0)], constant_values=-1)
    return image, positions


def preprocess_image(
    image_path: str | Path,
    model_dir: Path = MODEL_DIR,
    *,
    force_resize: tuple[int, int] | None = (768, 768),
) -> dict[str, np.ndarray | int]:
    """Vision OM inputs: pixel_values [1, max_patches, patch_pixels], image_position_ids."""
    _, _, proc_root = _load_model_config(model_dir)
    img_proc = proc_root["image_processor"]
    patch_size = int(img_proc["patch_size"])
    max_soft_tokens = int(img_proc["max_soft_tokens"])
    pooling_kernel_size = int(img_proc["pooling_kernel_size"])
    rescale_factor = float(img_proc["rescale_factor"])
    max_patches = max_soft_tokens * pooling_kernel_size**2

    image = Image.open(image_path).convert("RGB")
    if force_resize is not None:
        image = image.resize(force_resize, Image.BICUBIC)

    arr = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    if img_proc.get("do_rescale", True):
        arr = arr * rescale_factor

    height, width = arr.shape[1], arr.shape[2]
    if img_proc.get("do_resize", True):
        target_height, target_width = get_aspect_ratio_preserving_size(
            height, width, patch_size, max_patches, pooling_kernel_size
        )
        if target_height != height or target_width != width:
            resized = image.resize((target_width, target_height), Image.BICUBIC)
            arr = np.array(resized, dtype=np.float32).transpose(2, 0, 1) * rescale_factor

    patches = convert_image_to_patches(arr, patch_size)
    num_soft_tokens = patches.shape[0] // pooling_kernel_size**2
    patch_height = arr.shape[1] // patch_size
    patch_width = arr.shape[2] // patch_size
    grid_x, grid_y = np.meshgrid(
        np.arange(patch_width), np.arange(patch_height), indexing="xy"
    )
    real_positions = np.stack([grid_x, grid_y], axis=-1).reshape(patches.shape[0], 2)
    patches, positions = pad_along_first_dim(patches, real_positions, max_patches)

    return {
        "pixel_values": patches.astype(np.float16)[None, ...],
        "image_position_ids": positions.astype(np.int32)[None, ...],
        "num_soft_tokens": int(num_soft_tokens),
    }


def _render_chat_text(
    prompt_text: str,
    num_soft_tokens: int,
    model_dir: Path,
    *,
    add_generation_prompt: bool = True,
) -> str:
    _, tok_cfg, _ = _load_model_config(model_dir)
    template_src = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = env.from_string(template_src).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=tok_cfg["bos_token"],
        eos_token=tok_cfg["eos_token"],
        pad_token=tok_cfg["pad_token"],
    )
    boi_token = tok_cfg["boi_token"]
    eoi_token = tok_cfg["eoi_token"]
    image_token = tok_cfg["image_token"]
    replacement = f"{boi_token}{image_token * num_soft_tokens}{eoi_token}"
    return re.sub(re.escape(image_token), replacement, text, count=1)


def _lookup_ple_table(
    token_ids: np.ndarray,
    image_mask: np.ndarray,
    ple_table_path: Path,
    pad_token_id: int,
    vocab_size: int,
) -> np.ndarray:
    ple = np.memmap(
        ple_table_path,
        dtype=np.float16,
        mode="r",
        shape=(vocab_size, NUM_LAYERS, PLE_DIM),
    )
    out = np.zeros((1, MAX_SEQ_LEN, NUM_LAYERS, PLE_DIM), dtype=np.float16)
    for pos in range(MAX_SEQ_LEN):
        tid = pad_token_id if image_mask[0, pos] else int(token_ids[0, pos])
        out[0, pos] = ple[tid]
    return out


def build_prompt_bins(
    num_soft_tokens: int,
    *,
    prompt_text: str = "What is shown in this image?",
    model_dir: Path = MODEL_DIR,
    ple_table_path: Path = DEFAULT_PLE_TABLE,
) -> dict[str, np.ndarray | int]:
    """LLM preblock OM inputs: input_ids, attention_mask, per_layer_inputs, position_ids."""
    cfg, _, _ = _load_model_config(model_dir)
    text_cfg = cfg["text_config"]
    image_token_id = int(cfg["image_token_id"])
    pad_token_id = int(text_cfg["pad_token_id"])
    vocab_size = int(text_cfg["vocab_size_per_layer_input"])

    if not ple_table_path.is_file():
        raise FileNotFoundError(
            f"PLE table missing: {ple_table_path}\n"
            "Run: python om/dump_om_inputs.py --ple-only"
        )

    text = _render_chat_text(prompt_text, num_soft_tokens, model_dir)
    print(f"prompt_text: {prompt_text!r}")
    print(f"chat_template:\n{text}")

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    encoded = tokenizer.encode(text)
    seq_len = len(encoded.ids)
    print(f"seq_len (prefill) = {seq_len}")
    print(f"num_soft_tokens (prompt) = {num_soft_tokens}")

    input_ids = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    attention_mask = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids[0, :seq_len] = encoded.ids
    attention_mask[0, :seq_len] = 1

    llm_input_ids = input_ids.copy()
    image_mask = llm_input_ids == image_token_id
    llm_input_ids[image_mask] = pad_token_id

    per_layer_inputs = _lookup_ple_table(
        llm_input_ids, image_mask, ple_table_path, pad_token_id, vocab_size
    )
    position_ids = np.arange(MAX_SEQ_LEN, dtype=np.int32).reshape(1, MAX_SEQ_LEN)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "per_layer_inputs": per_layer_inputs,
        "llm_preblock_position_ids": position_ids,
        "seq_len": int(seq_len),
        "num_soft_tokens": int(num_soft_tokens),
    }


def load_prompt_bins(prompt_dir: Path) -> dict[str, np.ndarray | int]:
    """Load pre-generated llm_preblock bins from prompt_dir/llm_preblock/."""
    preblock = prompt_dir / "llm_preblock"
    if not preblock.is_dir():
        raise FileNotFoundError(f"missing {preblock}")

    input_ids = load_bin(preblock / "input_ids.bin", np.int32, (1, MAX_SEQ_LEN))
    attention_mask = load_bin(preblock / "attention_mask.bin", np.int32, (1, MAX_SEQ_LEN))
    per_layer_inputs = load_bin(
        preblock / "per_layer_inputs.bin", np.float16, (1, MAX_SEQ_LEN, NUM_LAYERS, PLE_DIM)
    )
    position_ids = load_bin(preblock / "position_ids.bin", np.int32, (1, MAX_SEQ_LEN))
    seq_len = int(attention_mask.sum())

    cfg, _, _ = _load_model_config(MODEL_DIR)
    image_token_id = int(cfg["image_token_id"])
    num_soft_tokens = int((input_ids == image_token_id).sum())

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "per_layer_inputs": per_layer_inputs,
        "llm_preblock_position_ids": position_ids,
        "seq_len": seq_len,
        "num_soft_tokens": num_soft_tokens,
    }


def print_loaded_prompt(prompt_dir: Path, model_dir: Path) -> None:
    """image-only: show preblock source and decoded prompt."""
    prompt = load_prompt_bins(prompt_dir)
    seq_len = int(prompt["seq_len"])
    num_soft = int(prompt["num_soft_tokens"])
    print(f"prompt_source: {(prompt_dir / 'llm_preblock').resolve()}")
    print(f"seq_len (prefill) = {seq_len}")
    print(f"num_soft_tokens (prompt) = {num_soft}")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = prompt["input_ids"][0, :seq_len].tolist()
    try:
        decoded = tokenizer.decode(ids)
    except TypeError:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"decode(input_ids[:seq_len]):\n{decoded}")


def _validate_prompt_image_compat(prompt: dict, vision: dict) -> None:
    prompt_soft = int(prompt["num_soft_tokens"])
    image_soft = int(vision["num_soft_tokens"])
    if prompt_soft != image_soft:
        raise ValueError(
            f"num_soft_tokens mismatch: prompt={prompt_soft} image={image_soft}. "
            "Use --mode full or regenerate prompt bins for this image size/aspect ratio."
        )


def build_all_inputs(
    image_path: str,
    *,
    prompt_text: str,
    model_dir: Path,
    ple_table_path: Path,
) -> dict:
    vision = preprocess_image(image_path, model_dir)
    print(f"num_soft_tokens (image) = {vision['num_soft_tokens']}")
    print("prompt: generate from --prompt-text (default if omitted)")
    prompt = build_prompt_bins(
        int(vision["num_soft_tokens"]),
        prompt_text=prompt_text,
        model_dir=model_dir,
        ple_table_path=ple_table_path,
    )
    return {**vision, **prompt}


def copy_preblock(src_root: Path, dst_root: Path) -> None:
    src = src_root / "llm_preblock"
    dst = dst_root / "llm_preblock"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask", "per_layer_inputs", "position_ids"):
        shutil.copy2(src / f"{name}.bin", dst / f"{name}.bin")
        print(f"  {dst / f'{name}.bin'}  (from {src / f'{name}.bin'})")
    print()


def dump_vision(vision: dict, out_dir: Path) -> None:
    d = out_dir / "vision"
    for key in ("pixel_values", "image_position_ids"):
        arr = vision[key]
        save_bin(arr, d / f"{key}.bin")
        print(f"  {d / f'{key}.bin'}  shape={arr.shape} dtype={arr.dtype}")
    print()


def dump_preblock(data: dict, prompt_root: Path) -> None:
    """Write llm_preblock bins to prompt_root/llm_preblock/ (shared default prompt)."""
    preblock_root = prompt_root / "llm_preblock"
    specs = [
        "input_ids",
        "attention_mask",
        "per_layer_inputs",
        ("llm_preblock_position_ids", "position_ids"),
    ]
    preblock_root.mkdir(parents=True, exist_ok=True)
    for item in specs:
        if isinstance(item, tuple):
            src_key, bin_name = item
            arr = data[src_key]
            p = preblock_root / f"{bin_name}.bin"
        else:
            arr = data[item]
            p = preblock_root / f"{item}.bin"
        save_bin(arr, p)
        print(f"  {p}  shape={arr.shape} dtype={arr.dtype}")


def dump_all(data: dict, out_dir: Path, *, prompt_root: Path | None = None) -> None:
    specs = [
        ("vision", ["pixel_values", "image_position_ids"]),
        (
            "llm_preblock",
            [
                "input_ids",
                "attention_mask",
                "per_layer_inputs",
                ("llm_preblock_position_ids", "position_ids"),
            ],
        ),
    ]

    print(f"seq_len (valid tokens) = {data['seq_len']}")
    print(f"num_soft_tokens = {data['num_soft_tokens']}")
    print(f"output root: {out_dir.resolve()}\n")

    for subdir, names in specs:
        d = out_dir / subdir
        for item in names:
            if isinstance(item, tuple):
                src_key, bin_name = item
                arr = data[src_key]
                p = d / f"{bin_name}.bin"
            else:
                arr = data[item]
                p = d / f"{item}.bin"
            save_bin(arr, p)
            print(f"  {p}  shape={arr.shape} dtype={arr.dtype}")
        print()

    if prompt_root is not None:
        print(f"prompt_bin: {prompt_root.resolve()}\n")
        dump_preblock(data, prompt_root)


def _load_text_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return cfg["text_config"]


def export_ple_table(safetensors_path: Path, config_path: Path, out_dir: Path) -> Path:
    """Write embed_tokens_per_layer.bin (+ meta.json). Returns bin path."""
    import torch
    from safetensors import safe_open

    out_dir.mkdir(parents=True, exist_ok=True)
    text_cfg = _load_text_config(config_path)

    num_layers = int(text_cfg["num_hidden_layers"])
    ple_dim = int(text_cfg["hidden_size_per_layer_input"])
    vocab = int(text_cfg["vocab_size_per_layer_input"])
    pad_token_id = int(text_cfg["pad_token_id"])
    embed_scale = math.sqrt(ple_dim)

    print(f"Loading {PLE_KEY} from {safetensors_path} ...")
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        if PLE_KEY not in f.keys():
            raise KeyError(f"{PLE_KEY} not in {safetensors_path}")
        weight = f.get_tensor(PLE_KEY).float()

    if tuple(weight.shape) != (vocab, num_layers * ple_dim):
        raise ValueError(f"unexpected shape {weight.shape}, expect ({vocab}, {num_layers * ple_dim})")

    table = (weight * embed_scale).cpu().numpy().astype(np.float16)
    table = table.reshape(vocab, num_layers, ple_dim)

    bin_path = out_dir / "embed_tokens_per_layer.bin"
    print(f"Writing {bin_path} ({table.nbytes / 1e9:.3f} GB) ...")
    table.tofile(bin_path)

    meta = {
        "description": "PLE lookup table (embed_tokens_per_layer * embed_scale)",
        "source": "dump_om_inputs.py",
        "safetensors": str(safetensors_path.resolve()),
        "dump_dir": str(out_dir.parent.resolve()),
        "vocab_size": vocab,
        "num_hidden_layers": num_layers,
        "hidden_size_per_layer_input": ple_dim,
        "embed_scale": embed_scale,
        "pad_token_id": pad_token_id,
        "dtype": "float16",
        "shape": [vocab, num_layers, ple_dim],
        "layout": "table[token_id] -> [num_layers, ple_dim]",
        "bytes": int(table.nbytes),
        "usage": "row = table[token_id]; per_layer_inputs[:, pos] = row",
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved: {bin_path}")
    print(f"  shape={table.shape} dtype={table.dtype}")
    print(f"  meta: {meta_path}")
    return bin_path


def ensure_ple_table(
    out_dir: Path,
    *,
    safetensors: Path,
    config: Path,
    force: bool = False,
) -> Path:
    """Return embed_tokens_per_layer.bin; skip if already present unless force."""
    out_dir = Path(out_dir)
    bin_path = out_dir / "embed_tokens_per_layer.bin"
    if bin_path.is_file() and not force:
        _log(f"ple_table exists: {bin_path}")
        return bin_path
    return export_ple_table(safetensors, config, out_dir)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def safe_stem(path: Path | str) -> str:
    base = Path(path).name.rsplit(".", 1)[0]
    s = re.sub(r"[ /:]", "___", base)
    return "".join(c for c in s if c.isalnum() or c in "_.-")


def find_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    paths = [
        p.resolve()
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def dump_one(
    image: Path,
    out_dir: Path,
    *,
    mode: str,
    prompt_dir: Path | None,
    prompt_text: str,
    model_dir: Path,
    ple_table_path: Path,
) -> None:
    print(f"mode={mode}  image={image}")
    if mode == "image-only":
        if prompt_dir is None:
            raise ValueError("--prompt-dir is required for --mode image-only")
        print_loaded_prompt(prompt_dir, model_dir)
        vision = preprocess_image(str(image), model_dir)
        print(f"num_soft_tokens (image) = {vision['num_soft_tokens']}")
        prompt = load_prompt_bins(prompt_dir)
        _validate_prompt_image_compat(prompt, vision)
        print("\nWriting bins:")
        print(f"output root: {out_dir.resolve()}\n")
        dump_vision(vision, out_dir)
        print(f"copy preblock: {prompt_dir.resolve()} -> {out_dir.resolve()}\n")
        copy_preblock(prompt_dir, out_dir)
        return

    print(f"ple_table={ple_table_path.resolve()}")
    data = build_all_inputs(
        str(image),
        prompt_text=prompt_text,
        model_dir=model_dir,
        ple_table_path=ple_table_path,
    )

    print("\nWriting bins:")
    dump_all(data, out_dir, prompt_root=DEFAULT_PROMPT_DIR)


def run_batch(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir).resolve()
    batch_root = Path(args.batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir)
    if not images:
        raise SystemExit(f"ERROR: no images under {image_dir}")

    model_dir = Path(args.model_dir)
    safetensors = Path(args.safetensors)
    config = Path(args.config)

    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    if args.mode == "image-only" and prompt_dir is None:
        prompt_dir = DEFAULT_PROMPT_DIR

    ple_table_path = Path(args.ple_table)
    if args.mode == "full":
        ple_table_path = ensure_ple_table(
            DEFAULT_PLE_DIR,
            safetensors=safetensors,
            config=config,
            force=args.force_ple,
        )

    _log(
        f"BATCH_ROOT={batch_root}  images={len(images)}  mode={args.mode}  "
        f"skip_exist={args.skip_exist}"
    )

    summary = batch_root / "summary_dump.tsv"
    summary.write_text("image\tstem\tstatus\tdump_dir\n", encoding="utf-8")

    total = len(images)
    for idx, img in enumerate(images, start=1):
        stem = safe_stem(img)
        item_dir = batch_root / stem
        dump_dir = item_dir / "dump"

        _log(f"========== [{idx}/{total}] dump {img} -> {dump_dir} ==========")

        marker = dump_dir / "vision" / "pixel_values.bin"
        if args.skip_exist and marker.is_file():
            _log("SKIP_EXIST")
            with summary.open("a", encoding="utf-8") as f:
                f.write(f"{img}\t{stem}\tskip_exist\t{dump_dir}\n")
            continue

        item_dir.mkdir(parents=True, exist_ok=True)
        dump_one(
            img,
            dump_dir,
            mode=args.mode,
            prompt_dir=prompt_dir,
            prompt_text=args.prompt_text,
            model_dir=model_dir,
            ple_table_path=ple_table_path,
        )

        with summary.open("a", encoding="utf-8") as f:
            f.write(f"{img}\t{stem}\tok\t{dump_dir}\n")

    _log(f"Dump done. summary: {summary}")
    _log(f"PLE (shared): {DEFAULT_PLE_DIR}")
    _log(f"Copy batch/ + ple_table/ to MDC (ple_table 只需一份)，then:")
    _log(f"  RUN_MSAME=1 bash run_om_pipeline.sh {batch_root}")


def main():
    parser = argparse.ArgumentParser(description="Dump OM frontend input bins (no transformers)")
    parser.add_argument(
        "--mode",
        choices=("full", "image-only"),
        default="image-only",
        help="image-only (default): vision bins + llm_preblock from --prompt-dir; full: regenerate prompt bins",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_DUMP_ROOT),
        help=f"output root (default: {DEFAULT_DUMP_ROOT.name}/ under om/, vision/ + llm_preblock/)",
    )
    parser.add_argument("--image", type=str, default="", help="single input image path")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="batch: directory of images (writes <batch-root>/<stem>/dump/)",
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default=str(DEFAULT_BATCH_ROOT),
        help="batch output root (default: om/batch)",
    )
    parser.add_argument(
        "--skip-exist",
        action="store_true",
        help="batch: skip item if dump/vision/pixel_values.bin exists",
    )
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default="",
        help="pre-generated dump root (llm_preblock/); default: prompt_bin/",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="What is shown in this image?",
        help="user text prompt (full mode only)",
    )
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument(
        "--ple-table",
        type=str,
        default=str(DEFAULT_PLE_TABLE),
        help="embed_tokens_per_layer.bin path (full mode; auto-generated if missing)",
    )
    parser.add_argument(
        "--ple-dir",
        type=str,
        default=str(DEFAULT_PLE_DIR),
        help="PLE output dir for --ple-only / auto ensure (default: om/ple_table，与 dump/ 同级)",
    )
    parser.add_argument("--safetensors", type=str, default=str(DEFAULT_SAFETENSORS))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--force-ple",
        action="store_true",
        help="regenerate embed_tokens_per_layer.bin even if it exists",
    )
    parser.add_argument(
        "--ple-only",
        action="store_true",
        help="only ensure om/ple_table/embed_tokens_per_layer.bin then exit",
    )
    args = parser.parse_args()

    safetensors = Path(args.safetensors)
    config = Path(args.config)
    ple_dir = Path(args.ple_dir)

    if args.ple_only:
        ensure_ple_table(ple_dir, safetensors=safetensors, config=config, force=args.force_ple)
        print("Done.")
        return

    if args.image_dir:
        run_batch(args)
        print("Done.")
        return

    image = args.image or DEFAULT_IMAGE
    model_dir = Path(args.model_dir)
    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    if args.mode == "image-only" and prompt_dir is None:
        prompt_dir = DEFAULT_PROMPT_DIR

    ple_table_path = Path(args.ple_table)
    if args.mode == "full":
        ple_table_path = ensure_ple_table(
            ple_table_path.parent,
            safetensors=safetensors,
            config=config,
            force=args.force_ple,
        )

    dump_one(
        Path(image),
        Path(args.out_dir),
        mode=args.mode,
        prompt_dir=prompt_dir,
        prompt_text=args.prompt_text,
        model_dir=model_dir,
        ple_table_path=ple_table_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
