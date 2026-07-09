#!/usr/bin/env python3
"""
ONNX e2e generate on CUDA — vision + LLM all GPU (Thor stream mode).

Inputs via transformers AutoProcessor; text decode via processor.decode.
No om/ bin or parse_state dependency.
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
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if export := os.environ.get("QWEN3_ONNX_EXPORT"):
    os.environ["QWEN3_ONNX_EXPORT_DIR"] = export
os.environ.setdefault("QWEN3_EXPORT_PROFILE", "448_512")
sys.path.insert(0, str(ROOT))

from llm import compute_static_position_ids  # noqa: E402

from onnx_common import (  # noqa: E402
    EXPORT_DIR,
    IMAGE_PATH,
    MAX_SEQ_LEN,
    MODEL_PATH,
    ONNX_VISION,
    PROMPT,
    PROFILE,
    preprocess,
)

MAX_NEW_TOKENS = int(os.environ.get("QWEN3_MAX_NEW_TOKENS", "20"))

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


def prepare_inputs(processor) -> dict:
    inputs = preprocess(processor)
    input_ids = inputs["input_ids"].numpy().astype(np.int32)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int32)
    position_ids = compute_static_position_ids(
        inputs["input_ids"], inputs["attention_mask"], "cpu"
    ).numpy().astype(np.int32)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float16)
    prefill_len = int(attention_mask.sum())
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prefill_len": prefill_len,
    }


def _session(name: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(Path(EXPORT_DIR) / name), opts, providers=ort_providers())


def load_llm_onnx() -> ort.InferenceSession:
    return _session("llm_preblock.onnx")


def run_vision_timed(pixel_values: np.ndarray, timer: FwdTimer):
    vision = _session(ONNX_VISION)
    merged, ds5, ds11, ds17 = timer.run(
        vision, None, {"hidden_states": np.ascontiguousarray(pixel_values, np.float16)},
    )
    del vision
    gc.collect()
    return merged, [ds5, ds11, ds17]


def block_feed(block, hidden, attn, cos, sin, deepstack):
    feed = {"hidden_states": hidden, "attention_mask": attn, "cos": cos, "sin": sin}
    names = {i.name for i in block.get_inputs()}
    for i, ds in enumerate(deepstack):
        k = f"ds_{i}"
        if k in names:
            feed[k] = ds
    return feed


def run_blocks_streaming(hidden, attn, cos, sin, deepstack, timer: FwdTimer | None = None) -> np.ndarray:
    for i in (1, 2, 3):
        block = _session(f"llm_block{i}.onnx")
        feed = block_feed(block, hidden, attn, cos, sin, deepstack)
        if timer is not None:
            hidden = timer.run(block, None, feed)[0]
        else:
            hidden = block.run(None, feed)[0]
        del block
        gc.collect()
    return hidden


def llm_forward_gpu_stream(pre, inp, image_embeds, deepstack, cur_len: int, timer: FwdTimer | None = None):
    pre_feed = {
        "input_ids": inp["input_ids"],
        "image_embeds": np.ascontiguousarray(image_embeds, np.float16),
        "attention_mask": inp["attention_mask"],
        "position_ids": inp["position_ids"],
    }
    if timer is not None:
        hidden, attn, cos, sin = timer.run(pre, None, pre_feed)
        hidden = run_blocks_streaming(hidden, attn, cos, sin, deepstack, timer=timer)
        head = _session("lm_head.onnx")
        h = np.asarray(hidden[:, cur_len - 1 : cur_len, :], dtype=np.float16)
        logits = timer.run(head, None, {"hidden_states": h})[0]
        del head
        gc.collect()
        return logits
    hidden, attn, cos, sin = pre.run(None, pre_feed)
    hidden = run_blocks_streaming(hidden, attn, cos, sin, deepstack)
    head = _session("lm_head.onnx")
    h = np.asarray(hidden[:, cur_len - 1 : cur_len, :], dtype=np.float16)
    logits = head.run(None, {"hidden_states": h})[0]
    del head
    gc.collect()
    return logits


def generate(inp, image_embeds, deepstack, pre, *, vision_fwd: float = 0.0) -> list[int]:
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
        logits = llm_forward_gpu_stream(pre, feed, image_embeds, deepstack, cur_len, timer=timer)
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

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("===== CFG =====")
    print(f"profile    = {PROFILE.name}")
    print(f"model      = {MODEL_PATH}")
    print(f"export     = {EXPORT_DIR}")
    print(f"vision     = {ONNX_VISION}")
    print(f"image      = {IMAGE_PATH}")
    print(f"prompt     = {PROMPT!r}")
    print(f"providers  = {ort_providers()}")

    print("\n===== 1. preprocess (transformers) =====")
    inp = prepare_inputs(processor)
    print(f"pixel_values {inp['pixel_values'].shape}  prefill={inp['prefill_len']}")

    print("\n===== 2. vision (load -> run -> release) =====")
    vision_timer = FwdTimer()
    image_embeds, deepstack = run_vision_timed(inp["pixel_values"], vision_timer)
    print(f"image_embeds {image_embeds.shape}  fwd={vision_timer.total:.2f}s (load excluded)")

    print("\n===== 3. load LLM ONNX =====")
    pre = load_llm_onnx()
    print(f"llm: GPU stream (pre resident EP={pre.get_providers()[0]}, blocks/head per step)")

    print("\n===== 4. decode loop =====")
    tokens = generate(inp, image_embeds, deepstack, pre, vision_fwd=vision_timer.total)

    print("\n===== 5. text =====")
    text = processor.decode(tokens, skip_special_tokens=True)
    print(text)
    if os.environ.get("BENCHMARK_OUT"):
        Path(os.environ["BENCHMARK_OUT"]).write_text(
            json.dumps({"tokens": tokens, "text": text}, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
