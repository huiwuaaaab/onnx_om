#!/usr/bin/env python3
"""
Dump prompt_bin/*.bin for Qwen3-VL llm_preblock OM static inputs.

Flow:
  1. build chat string   (vision markers + user prompt + assistant header)
  2. tokenize            (tokenizer.json)
  3. expand image_pad    one <|image_pad|> -> num_image_tokens copies
  4. pad                 input_ids / attention_mask [1, max_seq_len]
  5. mrope position_ids  [3, 1, max_seq_len]
  6. write .bin + meta.json

image_embeds is NOT dumped — filled at runtime from vision OM output.

Usage:
  python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"
  python dump_llm_preblock_inputs.py --prompt-file prompt.txt --out-dir prompt_bin
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

OM = Path(__file__).resolve().parent
DEFAULT_OUT = OM / "prompt_bin"
DEFAULT_PROMPT = "What is shown in this image?"

# ---------------------------------------------------------------------------
# Config (448_512 / 256_256) — all in this file, no export_config / model dir
# ---------------------------------------------------------------------------

TOKENIZER_JSON = OM.parent / "Qwen3-VL-2B-Instruct" / "tokenizer.json"

PAD_TOKEN_ID = 151643
IMAGE_PAD_ID = 151655

PROFILE = {
    "256_256": {
        "image_size": 256,
        "max_seq_len": 256,
        "merged_grid": 8,          # (256/16)/2
        "num_image_tokens": 64,    # merged_grid ** 2
        "image_prefix_len": 4,     # tokens before image grid in mrope
        "hidden_size": 2048,
    },
    "448_512": {
        "image_size": 448,
        "max_seq_len": 512,
        "merged_grid": 14,         # (448/16)/2
        "num_image_tokens": 196,
        "image_prefix_len": 4,
        "hidden_size": 2048,
    },
}

# Single-image user turn (matches Qwen3-VL chat_template, add_vision_id=False)
CHAT_TEMPLATE = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{prompt}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_chat_string(prompt: str) -> str:
    return CHAT_TEMPLATE.format(prompt=prompt)


def step2_token_ids(chat: str, num_image_tokens: int) -> list[int]:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids: list[int] = []
    for tid in tok.encode(chat).ids:
        if tid == IMAGE_PAD_ID:
            ids.extend([IMAGE_PAD_ID] * num_image_tokens)
        else:
            ids.append(tid)
    return ids


def step3_pad(token_ids: list[int], max_seq_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    seq_len = len(token_ids)
    if seq_len > max_seq_len:
        raise ValueError(f"seq_len {seq_len} > max_seq_len {max_seq_len}")

    input_ids = np.full((1, max_seq_len), PAD_TOKEN_ID, dtype=np.int32)
    attention_mask = np.zeros((1, max_seq_len), dtype=np.int32)
    input_ids[0, :seq_len] = token_ids
    attention_mask[0, :seq_len] = 1
    return input_ids, attention_mask, seq_len


def step4_position_ids(max_seq_len: int, merged_grid: int, image_prefix_len: int) -> np.ndarray:
    """mrope position_ids [3, 1, max_seq_len] — same layout as llm.get_rope_index."""
    t, h, w = 1, merged_grid, merged_grid
    n_img = t * h * w

    prefix = np.broadcast_to(
        np.arange(image_prefix_len, dtype=np.int32), (3, image_prefix_len)
    )

    t_idx = np.repeat(np.arange(t, dtype=np.int32), h * w)
    h_idx = np.tile(np.repeat(np.arange(h, dtype=np.int32), w), t)
    w_idx = np.tile(np.arange(w, dtype=np.int32), t * h)
    image = np.stack([t_idx, h_idx, w_idx], axis=0) + image_prefix_len

    suffix_len = max_seq_len - image_prefix_len - n_img
    st = int(image.max()) + 1
    suffix = np.broadcast_to(np.arange(suffix_len, dtype=np.int32), (3, suffix_len)) + st

    pos = np.concatenate([prefix, image, suffix], axis=1)
    return pos.reshape(3, 1, max_seq_len).astype(np.int32)


def step5_write(
    out_dir: Path,
    profile_name: str,
    cfg: dict,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    seq_len: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
        ("position_ids", position_ids),
    ):
        path = out_dir / f"{name}.bin"
        np.ascontiguousarray(arr).tofile(path)
        print(f"  {path}  shape={arr.shape}")

    meta = {
        "description": "llm_preblock OM static inputs (image_embeds from vision OM)",
        "source": "dump_llm_preblock_inputs.py",
        "profile": profile_name,
        "seq_len": seq_len,
        "prompt_template": CHAT_TEMPLATE,
        "tensors": {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in (
                ("input_ids", input_ids),
                ("attention_mask", attention_mask),
                ("position_ids", position_ids),
            )
        },
        "upstream": {
            "image_embeds_from": "vision OM merged_hidden_states",
            "shape": [cfg["num_image_tokens"], cfg["hidden_size"]],
            "dtype": "float16",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dump_preblock(prompt: str, profile_name: str, out_dir: Path) -> None:
    if not TOKENIZER_JSON.is_file():
        raise FileNotFoundError(f"missing tokenizer: {TOKENIZER_JSON}")

    cfg = PROFILE[profile_name]
    max_seq_len = cfg["max_seq_len"]
    merged_grid = cfg["merged_grid"]
    num_image_tokens = cfg["num_image_tokens"]
    image_prefix_len = cfg["image_prefix_len"]

    print(f"profile={profile_name}  out={out_dir}")
    print(f"prompt: {prompt!r}")

    chat = step1_chat_string(prompt)
    print(f"chat:\n{chat}")

    token_ids = step2_token_ids(chat, num_image_tokens)
    input_ids, attention_mask, seq_len = step3_pad(token_ids, max_seq_len)
    position_ids = step4_position_ids(max_seq_len, merged_grid, image_prefix_len)

    n_pad = int((input_ids[0, :seq_len] == IMAGE_PAD_ID).sum())
    print(f"seq_len={seq_len}  image_pad_tokens={n_pad}")

    step5_write(out_dir, profile_name, cfg, input_ids, attention_mask, position_ids, seq_len)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump Qwen3-VL llm_preblock prompt bins")
    p.add_argument(
        "--profile",
        choices=tuple(PROFILE),
        default=os.environ.get("QWEN3_EXPORT_PROFILE", "448_512"),
    )
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    dump_preblock(prompt, args.profile, Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
