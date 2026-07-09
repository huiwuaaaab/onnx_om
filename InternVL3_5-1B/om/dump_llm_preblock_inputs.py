#!/usr/bin/env python3
"""
Dump prompt_bin/*.bin for InternVL3_5-1B llm_preblock OM static inputs.

Flow:
  1. build chat string   (image markers + user prompt + assistant header)
  2. tokenize            (tokenizer.json)
  3. expand image tokens <IMG_CONTEXT> -> 256 copies inside <img>...</img>
  4. pad                 input_ids / attention_mask [1, max_seq_len]
  5. position_ids        [1, max_seq_len] = arange
  6. write .bin + meta.json

image_embeds is NOT dumped — filled at runtime from mm_proj OM output.

Usage:
  python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"
  python dump_llm_preblock_inputs.py --prompt-file prompt.txt --out-dir prompt_bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

OM = Path(__file__).resolve().parent
DEFAULT_OUT = OM / "prompt_bin"
DEFAULT_PROMPT = "What is shown in this image?"

TOKENIZER_JSON = OM.parent / "InternVL3_5-1B-HF" / "tokenizer.json"

MAX_SEQ_LEN = 512
PAD_TOKEN_ID = 151643
IMAGE_SEQ_LENGTH = 256
START_IMAGE_TOKEN = "<img>"
END_IMAGE_TOKEN = "</img>"
CONTEXT_IMAGE_TOKEN = "<IMG_CONTEXT>"

# Single-image user turn (matches InternVL chat_template.jinja)
CHAT_TEMPLATE = (
    "<|im_start|>user\n"
    "{image_placeholder}"
    "{prompt}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def step1_chat_string(prompt: str) -> str:
    image_block = (
        f"{START_IMAGE_TOKEN}"
        f"{CONTEXT_IMAGE_TOKEN * IMAGE_SEQ_LENGTH}"
        f"{END_IMAGE_TOKEN}"
    )
    return CHAT_TEMPLATE.format(image_placeholder=image_block, prompt=prompt)


def step2_token_ids(chat: str) -> list[int]:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    return tok.encode(chat).ids


def step3_pad(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    seq_len = len(token_ids)
    if seq_len > MAX_SEQ_LEN:
        raise ValueError(f"seq_len {seq_len} > max_seq_len {MAX_SEQ_LEN}")

    input_ids = np.full((1, MAX_SEQ_LEN), PAD_TOKEN_ID, dtype=np.int32)
    attention_mask = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids[0, :seq_len] = token_ids
    attention_mask[0, :seq_len] = 1
    return input_ids, attention_mask, seq_len


def step4_position_ids() -> np.ndarray:
    return np.arange(MAX_SEQ_LEN, dtype=np.int32).reshape(1, MAX_SEQ_LEN)


def step5_write(
    out_dir: Path,
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
        "description": "llm_preblock OM static inputs (image_embeds from mm_proj chain)",
        "source": "dump_llm_preblock_inputs.py",
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
            "image_embeds_from": "mm_proj OM output",
            "shape": [1, IMAGE_SEQ_LENGTH, 1024],
            "dtype": "float16",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dump_preblock(prompt: str, out_dir: Path) -> None:
    if not TOKENIZER_JSON.is_file():
        raise FileNotFoundError(f"missing tokenizer: {TOKENIZER_JSON}")

    print(f"out={out_dir}")
    print(f"prompt: {prompt!r}")

    chat = step1_chat_string(prompt)
    print(f"chat:\n{chat}")

    token_ids = step2_token_ids(chat)
    input_ids, attention_mask, seq_len = step3_pad(token_ids)
    position_ids = step4_position_ids()

    print(f"seq_len={seq_len}")

    step5_write(out_dir, input_ids, attention_mask, position_ids, seq_len)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump InternVL3_5 llm_preblock prompt bins")
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

    dump_preblock(prompt, Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
