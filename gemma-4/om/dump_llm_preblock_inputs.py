#!/usr/bin/env python3
"""
Dump llm_preblock OM input bins for Gemma-4.

Steps: chat template -> tokenize -> PLE lookup -> write 4 bins to prompt_bin/
  input_ids.bin, attention_mask.bin, per_layer_inputs.bin, position_ids.bin

PLE table (one-time): ple_table/embed_tokens_per_layer.bin via --ple-only

Usage:
  python dump_llm_preblock_inputs.py --ple-only
  python dump_llm_preblock_inputs.py --num-soft-tokens 256
  python dump_llm_preblock_inputs.py --prompt "What is shown in this image?"
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

OM = Path(__file__).resolve().parent
MODEL = OM.parent / "gemma-4-E2B-it"
DEFAULT_OUT = OM / "prompt_bin"
DEFAULT_PLE = OM / "ple_table" / "embed_tokens_per_layer.bin"
DEFAULT_SAFETENSORS = MODEL / "model.safetensors"
DEFAULT_PROMPT = "What is shown in this image?"

MAX_SEQ_LEN = 512
NUM_LAYERS = 35
PLE_DIM = 256
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
# fixed 768x768 vision -> always 256 for default prompt pipeline
DEFAULT_SOFT_TOKENS = 256


def _configs(model_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    tok = json.loads((model_dir / "tokenizer_config.json").read_text())
    return cfg, tok


def render_chat(prompt: str, num_soft: int, model_dir: Path) -> str:
    _, tok_cfg = _configs(model_dir)
    tpl = (model_dir / "chat_template.jinja").read_text()
    text = Environment(autoescape=False).from_string(tpl).render(
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        bos_token=tok_cfg["bos_token"],
        eos_token=tok_cfg["eos_token"],
        pad_token=tok_cfg["pad_token"],
    )
    boi, eoi, img_tok = tok_cfg["boi_token"], tok_cfg["eoi_token"], tok_cfg["image_token"]
    return re.sub(re.escape(img_tok), f"{boi}{img_tok * num_soft}{eoi}", text, count=1)


def ple_lookup(ids: np.ndarray, image_mask: np.ndarray, ple_path: Path, pad_id: int, vocab: int) -> np.ndarray:
    table = np.memmap(ple_path, dtype=np.float16, mode="r", shape=(vocab, NUM_LAYERS, PLE_DIM))
    out = np.zeros((1, MAX_SEQ_LEN, NUM_LAYERS, PLE_DIM), np.float16)
    for pos in range(MAX_SEQ_LEN):
        tid = pad_id if image_mask[0, pos] else int(ids[0, pos])
        out[0, pos] = table[tid]
    return out


def build_preblock(
    num_soft: int,
    *,
    prompt: str = DEFAULT_PROMPT,
    model_dir: Path = MODEL,
    ple_path: Path = DEFAULT_PLE,
) -> dict[str, np.ndarray | int]:
    cfg, _ = _configs(model_dir)
    text_cfg = cfg["text_config"]
    image_id = int(cfg["image_token_id"])
    pad_id = int(text_cfg["pad_token_id"])
    vocab = int(text_cfg["vocab_size_per_layer_input"])
    if not ple_path.is_file():
        raise FileNotFoundError(f"missing PLE: {ple_path}\n  run: python dump_llm_preblock_inputs.py --ple-only")

    text = render_chat(prompt, num_soft, model_dir)
    enc = Tokenizer.from_file(str(model_dir / "tokenizer.json")).encode(text)
    seq_len = len(enc.ids)

    input_ids = np.zeros((1, MAX_SEQ_LEN), np.int32)
    attention_mask = np.zeros((1, MAX_SEQ_LEN), np.int32)
    input_ids[0, :seq_len] = enc.ids
    attention_mask[0, :seq_len] = 1

    llm_ids = input_ids.copy()
    image_mask = llm_ids == image_id
    llm_ids[image_mask] = pad_id
    per_layer = ple_lookup(llm_ids, image_mask, ple_path, pad_id, vocab)
    position_ids = np.arange(MAX_SEQ_LEN, dtype=np.int32).reshape(1, MAX_SEQ_LEN)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "per_layer_inputs": per_layer,
        "position_ids": position_ids,
        "seq_len": seq_len,
        "num_soft_tokens": num_soft,
    }


def dump_preblock(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask", "per_layer_inputs", "position_ids"):
        arr = data[name]
        path = out_dir / f"{name}.bin"
        np.ascontiguousarray(arr).tofile(path)
        print(f"  {path}  shape={arr.shape}")
    meta = {
        "description": "llm_preblock OM static inputs",
        "source": "dump_llm_preblock_inputs.py",
        "seq_len": int(data["seq_len"]),
        "num_soft_tokens": int(data["num_soft_tokens"]),
        "tensors": {
            name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
            for name in ("input_ids", "attention_mask", "per_layer_inputs", "position_ids")
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"seq_len={data['seq_len']}  num_soft_tokens={data['num_soft_tokens']}")


def export_ple(safetensors: Path, config: Path, out_dir: Path) -> Path:
    import torch
    from safetensors import safe_open

    out_dir.mkdir(parents=True, exist_ok=True)
    text_cfg = json.loads(config.read_text())["text_config"]
    n_layers = int(text_cfg["num_hidden_layers"])
    ple_dim = int(text_cfg["hidden_size_per_layer_input"])
    vocab = int(text_cfg["vocab_size_per_layer_input"])
    scale = math.sqrt(ple_dim)

    with safe_open(str(safetensors), framework="pt", device="cpu") as f:
        weight = f.get_tensor(PLE_KEY).float()
    table = (weight * scale).numpy().astype(np.float16).reshape(vocab, n_layers, ple_dim)

    out = out_dir / "embed_tokens_per_layer.bin"
    table.tofile(out)
    print(f"{out}  shape={table.shape}  ({table.nbytes / 1e9:.2f} GB)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--num-soft-tokens", type=int, default=DEFAULT_SOFT_TOKENS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--ple-table", type=Path, default=DEFAULT_PLE)
    p.add_argument("--ple-dir", type=Path, default=OM / "ple_table")
    p.add_argument("--safetensors", type=Path, default=DEFAULT_SAFETENSORS)
    p.add_argument("--config", type=Path, default=MODEL / "config.json")
    p.add_argument("--ple-only", action="store_true")
    p.add_argument("--force-ple", action="store_true")
    args = p.parse_args()

    if args.ple_only:
        export_ple(args.safetensors, args.config, args.ple_dir)
        return

    ple_path = args.ple_table
    if not ple_path.is_file():
        ple_path = args.ple_dir / "embed_tokens_per_layer.bin"
    if not ple_path.is_file() or args.force_ple:
        export_ple(args.safetensors, args.config, args.ple_dir)
        ple_path = args.ple_dir / "embed_tokens_per_layer.bin"

    print(f"prompt={args.prompt!r}")
    data = build_preblock(
        args.num_soft_tokens,
        prompt=args.prompt,
        model_dir=args.model_dir,
        ple_path=ple_path,
    )
    dump_preblock(data, args.out_dir)


if __name__ == "__main__":
    main()
