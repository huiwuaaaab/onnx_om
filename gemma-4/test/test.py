#!/usr/bin/env python3
"""
ONNX e2e generate on CUDA — vision + LLM all GPU (ThorU ORT stream mode).

Static inputs: vision_bin/*.bin + prompt_bin/*.bin + ple_table/
Decode: tokenizer.json (om/parse_state.py)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OM = ROOT / "om"
MODEL_DIR = ROOT / "gemma-4-E2B-it"
PROMPT_BIN = OM / "prompt_bin"
VISION_BIN = OM / "vision_bin"
PLE_TABLE = OM / "ple_table" / "embed_tokens_per_layer.bin"
EXPORT_DIR = Path(os.environ.get(
    "GEMMA4_ONNX_EXPORT",
    "./onnx_export",
))
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 50
NUM_LAYERS = 35
PLE_DIM = 256
BLOCK_NAMES = (
    "llm_block_0_5.onnx", "llm_block_5_10.onnx", "llm_block_10_15.onnx",
    "llm_block_15_20.onnx", "llm_block_20_25.onnx", "llm_block_25_30.onnx",
    "llm_block_30_35.onnx",
)
BLOCK_PLE = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35))

CUDA_EP = (
    "CUDAExecutionProvider",
    {"device_id": "0", "cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0"},
)


@dataclass
class FwdTimer:
    total: float = field(default=0.0)

    def run(self, session: ort.InferenceSession, output_names, input_feed):
        t0 = time.perf_counter()
        out = session.run(output_names, input_feed)
        self.total += time.perf_counter() - t0
        return out


def ort_providers() -> list:
    return [CUDA_EP, "CPUExecutionProvider"]


_DTYPE = {"int32": np.int32, "float16": np.float16, "float32": np.float32}

sys.path.insert(0, str(OM))
from parse_state import decode_ids, prefill_len_from_dump as prefill_len  # noqa: E402


def _gemma_cfg() -> tuple[int, int, int]:
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    text_cfg = cfg["text_config"]
    return (
        int(cfg["image_token_id"]),
        int(text_cfg["pad_token_id"]),
        int(text_cfg["vocab_size_per_layer_input"]),
    )


def load_bin_dir(bin_dir: Path) -> dict[str, np.ndarray]:
    meta_path = bin_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        out: dict[str, np.ndarray] = {}
        for name, spec in meta["tensors"].items():
            path = bin_dir / f"{name}.bin"
            if not path.is_file():
                raise FileNotFoundError(path)
            dt = _DTYPE[spec["dtype"]]
            out[name] = np.fromfile(path, dtype=dt).reshape(tuple(spec["shape"]))
        return out

    return {
        "input_ids": np.fromfile(bin_dir / "input_ids.bin", np.int32).reshape(1, MAX_SEQ_LEN),
        "attention_mask": np.fromfile(bin_dir / "attention_mask.bin", np.int32).reshape(1, MAX_SEQ_LEN),
        "position_ids": np.fromfile(bin_dir / "position_ids.bin", np.int32).reshape(1, MAX_SEQ_LEN),
        "per_layer_inputs": np.fromfile(bin_dir / "per_layer_inputs.bin", np.float16).reshape(
            1, MAX_SEQ_LEN, NUM_LAYERS, PLE_DIM
        ),
    }


def load_preblock(bin_dir: Path = PROMPT_BIN) -> dict:
    tensors = load_bin_dir(bin_dir)
    image_id, pad_id, ple_vocab = _gemma_cfg()
    return {
        "input_ids": tensors["input_ids"],
        "attention_mask": tensors["attention_mask"],
        "position_ids": tensors["position_ids"],
        "per_layer_inputs": tensors["per_layer_inputs"],
        "prefill_len": prefill_len(bin_dir),
        "image_mask": tensors["input_ids"] == image_id,
        "pad_id": pad_id,
        "ple_vocab": ple_vocab,
    }


def load_vision(bin_dir: Path = VISION_BIN) -> dict:
    tensors = load_bin_dir(bin_dir)
    return {
        "pixel_values": tensors["pixel_values"],
        "image_position_ids": tensors["image_position_ids"],
    }


def load_inputs(*, prompt_bin: Path = PROMPT_BIN, vision_bin: Path = VISION_BIN) -> dict:
    pre = load_preblock(prompt_bin)
    vision = load_vision(vision_bin)
    return {
        "pixel_values": vision["pixel_values"],
        "image_position_ids": vision["image_position_ids"],
        "input_ids": pre["input_ids"],
        "attention_mask": pre["attention_mask"],
        "per_layer_inputs": pre["per_layer_inputs"],
        "position_ids": pre["position_ids"],
        "prefill_len": pre["prefill_len"],
        "image_mask": pre["image_mask"],
        "pad_id": pre["pad_id"],
        "ple_vocab": pre["ple_vocab"],
    }


def _session(name: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(EXPORT_DIR / name), opts, providers=ort_providers())


def run_vision_mm(
    pixel_values: np.ndarray,
    image_position_ids: np.ndarray,
    timer: FwdTimer | None = None,
) -> np.ndarray:
    vfeed = {
        "pixel_values": np.ascontiguousarray(pixel_values, np.float16),
        "image_position_ids": np.ascontiguousarray(image_position_ids, np.int32),
    }
    vision = _session("vision.onnx")
    if timer is not None:
        vision.run(None, vfeed)
        vout = timer.run(vision, None, vfeed)[0]
    else:
        vision.run(None, vfeed)
        vout = vision.run(None, vfeed)[0]
    del vision
    gc.collect()
    mm = _session("mm_proj.onnx")
    if timer is not None:
        image_embeds = timer.run(mm, None, {"vision_features": vout})[0]
        print(f"vision+mm_proj EP: CUDAExecutionProvider  fwd={timer.total:.2f}s (load excluded)")
    else:
        t0 = time.perf_counter()
        image_embeds = mm.run(None, {"vision_features": vout})[0]
        print(f"vision+mm_proj EP: CUDAExecutionProvider  fwd={time.perf_counter()-t0:.2f}s")
    del mm
    gc.collect()
    return image_embeds


def load_llm_onnx() -> ort.InferenceSession:
    return _session("llm_preblock.onnx")


def _block_feed(hidden, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s, per_layer, fk=None, fv=None, sk=None, sv=None):
    feed = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": slide_mask,
        "cos_full": cos_f, "sin_full": sin_f,
        "cos_slide": cos_s, "sin_slide": sin_s,
        "per_layer_input": per_layer,
    }
    if fk is not None:
        feed.update(full_k=fk, full_v=fv, slide_k=sk, slide_v=sv)
    return feed


def run_blocks_streaming_kv(
    hidden, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s, per_layer,
    timer: FwdTimer | None = None,
):
    fk = fv = sk = sv = None
    for idx, (name, (lo, hi)) in enumerate(zip(BLOCK_NAMES, BLOCK_PLE)):
        block = _session(name)
        feed = _block_feed(
            hidden, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s,
            per_layer[:, :, lo:hi, :], fk, fv, sk, sv,
        )
        if timer is not None:
            out = timer.run(block, None, feed)
        else:
            out = block.run(None, feed)
        del block
        gc.collect()
        if idx == 2:
            hidden, fk, fv, sk, sv = out
        else:
            hidden = out[0]
    return hidden


def lm_head_stream(hidden, cur_len: int, timer: FwdTimer | None = None):
    head = _session("lm_head.onnx")
    h = np.asarray(hidden[:, cur_len - 1 : cur_len, :], dtype=np.float16)
    if timer is not None:
        logits = timer.run(head, None, {"hidden_states": h})[0]
    else:
        logits = head.run(None, {"hidden_states": h})[0]
    del head
    gc.collect()
    return logits


def llm_forward_gpu_stream(pre, inp, image_embeds, cur_len: int, timer: FwdTimer | None = None):
    pre_feed = {
        "input_ids": inp["input_ids"],
        "image_embeds": np.ascontiguousarray(image_embeds, np.float16),
        "attention_mask": inp["attention_mask"],
        "per_layer_inputs": np.ascontiguousarray(inp["per_layer_inputs"], np.float16),
        "position_ids": inp["position_ids"],
    }
    if timer is not None:
        hidden, per_layer, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s = timer.run(pre, None, pre_feed)
        hidden = run_blocks_streaming_kv(
            hidden, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s, per_layer, timer=timer,
        )
        return lm_head_stream(hidden, cur_len, timer=timer)
    hidden, per_layer, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s = pre.run(None, pre_feed)
    hidden = run_blocks_streaming_kv(
        hidden, full_mask, slide_mask, cos_f, sin_f, cos_s, sin_s, per_layer,
    )
    return lm_head_stream(hidden, cur_len)


def generate(inp, image_embeds, pre, *, vision_fwd: float = 0.0) -> list[int]:
    input_ids = inp["input_ids"].copy()
    attention_mask = inp["attention_mask"].copy()
    per_layer = inp["per_layer_inputs"].copy()
    image_mask = inp["image_mask"]
    pad_id = int(inp["pad_id"])
    vocab = int(inp["ple_vocab"])
    table = np.memmap(PLE_TABLE, dtype=np.float16, mode="r", shape=(vocab, NUM_LAYERS, PLE_DIM))
    cur_len = int(attention_mask.sum())
    out: list[int] = []
    timer = FwdTimer()

    def patch(pos: int):
        tid = pad_id if image_mask[0, pos] else int(input_ids[0, pos])
        per_layer[0, pos] = table[tid]

    for _ in range(MAX_NEW_TOKENS):
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "per_layer_inputs": per_layer,
            "position_ids": inp["position_ids"],
        }
        logits = llm_forward_gpu_stream(pre, feed, image_embeds, cur_len, timer=timer)
        tid = int(np.argmax(logits[:, 0, :], axis=-1)[0])
        out.append(tid)
        input_ids[0, cur_len] = tid
        attention_mask[0, cur_len] = 1
        patch(cur_len)
        cur_len += 1
        if cur_len >= MAX_SEQ_LEN:
            break
    total_fwd = vision_fwd + timer.total
    print(
        f"generate: {len(out)} tokens fwd={timer.total:.2f}s "
        f"vision_fwd={vision_fwd:.2f}s total_fwd={total_fwd:.2f}s "
        f"(prefill={inp['prefill_len']}, load excluded)"
    )
    return out


def main() -> None:
    if os.environ.get("ORT_USE_GPU", "1") != "1":
        raise SystemExit("CUDA required: set ORT_USE_GPU=1")
    if not PLE_TABLE.is_file():
        raise SystemExit(f"missing {PLE_TABLE}\n  run: cd om && python dump_llm_preblock_inputs.py --ple-only")

    print("===== CFG =====")
    print(f"model      = {MODEL_DIR}")
    print(f"export     = {EXPORT_DIR}")
    print(f"ple        = {PLE_TABLE}")
    print(f"prompt_bin = {PROMPT_BIN}")
    print(f"vision_bin = {VISION_BIN}")
    print(f"providers  = {ort_providers()}")

    print("\n===== 1. load bins (om/) =====")
    inp = load_inputs()
    print(f"pixel_values {inp['pixel_values'].shape}  prefill={inp['prefill_len']}")

    print("\n===== 2. vision + mm_proj =====")
    vision_timer = FwdTimer()
    image_embeds = run_vision_mm(inp["pixel_values"], inp["image_position_ids"], timer=vision_timer)
    print(f"image_embeds {image_embeds.shape}")

    print("\n===== 3. load LLM ONNX =====")
    pre = load_llm_onnx()
    print(f"llm: GPU stream (pre resident EP={pre.get_providers()[0]}, blocks/head per step)")

    print("\n===== 4. decode loop =====")
    tokens = generate(inp, image_embeds, pre, vision_fwd=vision_timer.total)

    print("\n===== 5. text =====")
    text = decode_ids(MODEL_DIR, tokens, skip_special=True)
    print(text)
    if os.environ.get("BENCHMARK_OUT"):
        Path(os.environ["BENCHMARK_OUT"]).write_text(
            json.dumps({"tokens": tokens, "text": text}, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
