#!/usr/bin/env python3
"""
ONNX e2e generate on CUDA — vision + LLM all GPU (ThorU ORT stream mode).

Static inputs: vision_bin/*.bin + prompt_bin/*.bin
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
MODEL_DIR = ROOT / "InternVL3_5-1B-HF"
PROMPT_BIN = OM / "prompt_bin"
VISION_BIN = OM / "vision_bin"
EXPORT_DIR = Path(os.environ.get(
    "INTERNVL_ONNX_EXPORT",
    "./onnx_export",
))
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 50


@dataclass
class FwdTimer:
    total: float = field(default=0.0)

    def run(self, session: ort.InferenceSession, output_names, input_feed):
        t0 = time.perf_counter()
        out = session.run(output_names, input_feed)
        self.total += time.perf_counter() - t0
        return out


CUDA_EP = (
    "CUDAExecutionProvider",
    {"device_id": "0", "cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0"},
)


def ort_providers() -> list:
    return [CUDA_EP, "CPUExecutionProvider"]

_DTYPE = {"int32": np.int32, "float16": np.float16, "float32": np.float32}

sys.path.insert(0, str(OM))
from parse_state import decode_ids, prefill_len_from_dump as prefill_len  # noqa: E402


def load_bin_dir(bin_dir: Path) -> dict[str, np.ndarray]:
    meta = json.loads((bin_dir / "meta.json").read_text())
    out: dict[str, np.ndarray] = {}
    for name, spec in meta["tensors"].items():
        path = bin_dir / f"{name}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        dt = _DTYPE[spec["dtype"]]
        out[name] = np.fromfile(path, dtype=dt).reshape(tuple(spec["shape"]))
    return out


def load_preblock(bin_dir: Path = PROMPT_BIN) -> dict:
    tensors = load_bin_dir(bin_dir)
    return {
        "input_ids": tensors["input_ids"],
        "attention_mask": tensors["attention_mask"],
        "position_ids": tensors["position_ids"],
        "prefill_len": prefill_len(bin_dir),
    }


def load_vision(bin_dir: Path = VISION_BIN) -> np.ndarray:
    return load_bin_dir(bin_dir)["pixel_values"]


def load_inputs(*, prompt_bin: Path = PROMPT_BIN, vision_bin: Path = VISION_BIN) -> dict:
    pre = load_preblock(prompt_bin)
    return {
        "pixel_values": load_vision(vision_bin),
        "input_ids": pre["input_ids"],
        "attention_mask": pre["attention_mask"],
        "position_ids": pre["position_ids"],
        "prefill_len": pre["prefill_len"],
    }


def _session(name: str, *, providers: list | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        str(EXPORT_DIR / name), opts, providers=providers or ort_providers(),
    )


def run_vision_mm(pixel_values: np.ndarray, timer: FwdTimer | None = None) -> np.ndarray:
    vprov = ort_providers()
    vfeed = {"pixel_values": np.ascontiguousarray(pixel_values, np.float16)}
    # Thor: load vision -> run -> unload -> load mm_proj -> run
    vision = _session("vision_448_notchunk.onnx", providers=vprov)
    if timer is not None:
        vision.run(None, vfeed)
        vout = timer.run(vision, None, vfeed)[0]
    else:
        vision.run(None, vfeed)
        vout = vision.run(None, vfeed)[0]
    del vision
    gc.collect()
    mm = _session("mm_proj.onnx", providers=vprov)
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


def lm_head(head, hidden, cur_len: int):
    h = np.asarray(hidden[:, cur_len - 1 : cur_len, :], dtype=np.float16)
    return head.run(None, {"hidden_states": h})[0]


def run_blocks_streaming(hidden, attn, cos, sin, timer: FwdTimer | None = None) -> np.ndarray:
    for i in (1, 2, 3):
        block = _session(f"llm_block{i}.onnx")
        feed = {"hidden_states": hidden, "attention_mask": attn, "cos": cos, "sin": sin}
        if timer is not None:
            hidden = timer.run(block, None, feed)[0]
        else:
            hidden = block.run(None, feed)[0]
        del block
        gc.collect()
    return hidden


def llm_forward_gpu_stream(pre, inp, image_embeds, cur_len: int, timer: FwdTimer | None = None):
    pre_feed = {
        "image_embeds": np.ascontiguousarray(image_embeds, np.float16),
        "attention_mask": inp["attention_mask"],
        "input_ids": inp["input_ids"],
        "position_ids": inp["position_ids"],
    }
    if timer is not None:
        hidden, attn, cos, sin = timer.run(pre, None, pre_feed)
        hidden = run_blocks_streaming(hidden, attn, cos, sin, timer=timer)
        head = _session("lm_head.onnx")
        h = np.asarray(hidden[:, cur_len - 1 : cur_len, :], dtype=np.float16)
        logits = timer.run(head, None, {"hidden_states": h})[0]
        del head
        gc.collect()
        return logits
    hidden, attn, cos, sin = pre.run(None, pre_feed)
    hidden = run_blocks_streaming(hidden, attn, cos, sin)
    head = _session("lm_head.onnx")
    logits = lm_head(head, hidden, cur_len)
    del head
    gc.collect()
    return logits


def generate(inp, image_embeds, pre, *, vision_fwd: float = 0.0) -> list[int]:
    input_ids = inp["input_ids"].copy()
    attention_mask = inp["attention_mask"].copy()
    cur_len = int(attention_mask.sum())
    out: list[int] = []
    timer = FwdTimer()
    for _ in range(MAX_NEW_TOKENS):
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": inp["position_ids"],
        }
        logits = llm_forward_gpu_stream(pre, feed, image_embeds, cur_len, timer=timer)
        tid = int(np.argmax(logits[:, 0, :], axis=-1)[0])
        out.append(tid)
        input_ids[0, cur_len] = tid
        attention_mask[0, cur_len] = 1
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

    print("===== CFG =====")
    print(f"model      = {MODEL_DIR}")
    print(f"export     = {EXPORT_DIR}")
    print(f"prompt_bin = {PROMPT_BIN}")
    print(f"vision_bin = {VISION_BIN}")
    print(f"providers  = {ort_providers()}")

    print("\n===== 1. load bins (om/) =====")
    inp = load_inputs()
    print(f"pixel_values {inp['pixel_values'].shape}  prefill={inp['prefill_len']}")

    print("\n===== 2. vision + mm_proj =====")
    vision_timer = FwdTimer()
    image_embeds = run_vision_mm(inp["pixel_values"], timer=vision_timer)
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
