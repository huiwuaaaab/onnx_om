#!/usr/bin/env python3
"""
ThorU: pure forward (session.run) timing for Qwen3-VL / InternVL / Gemma-4.
Only session.run() counted; InferenceSession load excluded.
GPU vision: load vision -> run -> unload; then mm_proj (if any) -> run -> unload.
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

N = int(os.environ.get("BENCH_STEPS", "5"))
TOK50 = int(os.environ.get("EST_TOKENS", "50"))
CUR = 290

CUDA = [
    ("CUDAExecutionProvider", {"device_id": "0", "cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0"}),
    "CPUExecutionProvider",
]
CPU = ["CPUExecutionProvider"]
BASE = Path("/cus_app_data/guanxj")


def sess(export: Path, name: str, prov: list) -> ort.InferenceSession:
    return ort.InferenceSession(str(export / name), ort.SessionOptions(), providers=prov)


@dataclass
class FwdTimes:
    vision: float
    pre: float
    blocks: float
    head: float

    @property
    def decode_step(self) -> float:
        return self.pre + self.blocks + self.head

    @property
    def e2e_50(self) -> float:
        return self.vision + self.decode_step * TOK50


def load_bins(om: Path) -> tuple[dict, dict]:
    vm = json.loads((om / "vision_bin/meta.json").read_text())
    vision = {
        k: np.fromfile(
            om / f"vision_bin/{k}.bin",
            np.float16 if v["dtype"] == "float16" else np.int32,
        ).reshape(tuple(v["shape"]))
        for k, v in vm["tensors"].items()
    }
    pm_path = om / "prompt_bin/meta.json"
    if pm_path.is_file():
        pm = json.loads(pm_path.read_text())
        prompt = {
            k: np.fromfile(
                om / f"prompt_bin/{k}.bin",
                np.int32 if v["dtype"] == "int32" else np.float16,
            ).reshape(tuple(v["shape"]))
            for k, v in pm["tensors"].items()
        }
    else:
        prompt = {
            "input_ids": np.fromfile(om / "prompt_bin/input_ids.bin", np.int32).reshape(1, 512),
            "attention_mask": np.fromfile(om / "prompt_bin/attention_mask.bin", np.int32).reshape(1, 512),
            "position_ids": np.fromfile(om / "prompt_bin/position_ids.bin", np.int32).reshape(1, 512),
            "per_layer_inputs": np.fromfile(om / "prompt_bin/per_layer_inputs.bin", np.float16).reshape(1, 512, 35, 256),
        }
    return vision, prompt


def _avg(acc: FwdTimes) -> FwdTimes:
    return FwdTimes(acc.vision, acc.pre / N, acc.blocks / N, acc.head / N)


def qwen_vision_fwd_gpu(export: Path, vision: dict) -> tuple[tuple, float]:
    feed_v = {"hidden_states": np.ascontiguousarray(vision["pixel_values"], np.float16)}
    vis = sess(export, "vision_448.onnx", CUDA)
    vis.run(None, feed_v)
    t0 = time.perf_counter()
    out = vis.run(None, feed_v)
    dt = time.perf_counter() - t0
    del vis
    gc.collect()
    return out, dt


def vision_mm_fwd_gpu(export: Path, vision_name: str, vfeed: dict) -> tuple[np.ndarray, float]:
    vis = sess(export, vision_name, CUDA)
    vis.run(None, vfeed)
    vout = vis.run(None, vfeed)[0]
    del vis
    gc.collect()
    mm = sess(export, "mm_proj.onnx", CUDA)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vout})[0]
    dt = time.perf_counter() - t0
    del mm
    gc.collect()
    return ie, dt


def bench_qwen(cpu: bool, gpu_llm: bool) -> FwdTimes:
    export, om = BASE / "qwen3-vl/onnx_export", BASE / "qwen3-vl/om"
    vprov = CPU if (cpu or not gpu_llm) else CUDA
    lprov = CPU if cpu else CUDA
    vision, prompt = load_bins(om)
    cur = int(prompt["attention_mask"].sum())

    vis = sess(export, "vision_448.onnx", vprov)
    feed_v = {"hidden_states": np.ascontiguousarray(vision["pixel_values"], np.float16)}
    vis.run(None, feed_v)
    t0 = time.perf_counter()
    merged, ds5, ds11, ds17 = vis.run(None, feed_v)
    vision_t = time.perf_counter() - t0
    ds = [ds5, ds11, ds17]

    pre = sess(export, "llm_preblock.onnx", lprov)
    blocks = [sess(export, f"llm_block{i}.onnx", lprov) for i in (1, 2, 3)]
    head = sess(export, "lm_head.onnx", lprov)
    ids, attn, pos = prompt["input_ids"].copy(), prompt["attention_mask"].copy(), prompt["position_ids"].copy()
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def step():
        t0 = time.perf_counter()
        hidden, a, cos, sin = pre.run(None, {
            "input_ids": ids, "image_embeds": np.ascontiguousarray(merged, np.float16),
            "attention_mask": attn, "position_ids": pos,
        })
        acc.pre += time.perf_counter() - t0
        t0 = time.perf_counter()
        for b in blocks:
            f = {"hidden_states": hidden, "attention_mask": a, "cos": cos, "sin": sin}
            names = {i.name for i in b.get_inputs()}
            for i, x in enumerate(ds):
                if f"ds_{i}" in names:
                    f[f"ds_{i}"] = x
            hidden = b.run(None, f)[0]
        acc.blocks += time.perf_counter() - t0
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": np.asarray(hidden[:, cur - 1:cur, :], np.float16)})
        acc.head += time.perf_counter() - t0

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_qwen_stream_gpu() -> FwdTimes:
    export, om = BASE / "qwen3-vl/onnx_export", BASE / "qwen3-vl/om"
    vision, prompt = load_bins(om)
    cur = int(prompt["attention_mask"].sum())

    (merged, ds5, ds11, ds17), vision_t = qwen_vision_fwd_gpu(export, vision)
    ds = [ds5, ds11, ds17]

    pre = sess(export, "llm_preblock.onnx", CUDA)
    ids, attn, pos = prompt["input_ids"].copy(), prompt["attention_mask"].copy(), prompt["position_ids"].copy()
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def step():
        t0 = time.perf_counter()
        hidden, a, cos, sin = pre.run(None, {
            "input_ids": ids, "image_embeds": np.ascontiguousarray(merged, np.float16),
            "attention_mask": attn, "position_ids": pos,
        })
        acc.pre += time.perf_counter() - t0
        for i in (1, 2, 3):
            block = sess(export, f"llm_block{i}.onnx", CUDA)
            f = {"hidden_states": hidden, "attention_mask": a, "cos": cos, "sin": sin}
            names = {x.name for x in block.get_inputs()}
            for j, x in enumerate(ds):
                if f"ds_{j}" in names:
                    f[f"ds_{j}"] = x
            t0 = time.perf_counter()
            hidden = block.run(None, f)[0]
            acc.blocks += time.perf_counter() - t0
            del block
            gc.collect()
        head = sess(export, "lm_head.onnx", CUDA)
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": np.asarray(hidden[:, cur - 1:cur, :], np.float16)})
        acc.head += time.perf_counter() - t0
        del head
        gc.collect()

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_internvl_stream_gpu() -> FwdTimes:
    export, om = BASE / "internvl3_5/onnx_export", BASE / "internvl3_5/om"
    vision, prompt = load_bins(om)
    cur = int(prompt["attention_mask"].sum())

    vfeed = {"pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16)}
    vis = sess(export, "vision_448_notchunk.onnx", CUDA)
    vis.run(None, vfeed)
    t0 = time.perf_counter()
    vout = vis.run(None, vfeed)[0]
    vis_t = time.perf_counter() - t0
    del vis
    gc.collect()
    mm = sess(export, "mm_proj.onnx", CUDA)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vout})[0]
    mm_t = time.perf_counter() - t0
    vision_t = vis_t + mm_t
    del mm
    gc.collect()

    pre = sess(export, "llm_preblock.onnx", CUDA)
    ids, attn, pos = prompt["input_ids"].copy(), prompt["attention_mask"].copy(), prompt["position_ids"].copy()
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def step():
        t0 = time.perf_counter()
        hidden, a, cos, sin = pre.run(None, {
            "image_embeds": np.ascontiguousarray(ie, np.float16),
            "attention_mask": attn, "input_ids": ids, "position_ids": pos,
        })
        acc.pre += time.perf_counter() - t0
        for i in (1, 2, 3):
            block = sess(export, f"llm_block{i}.onnx", CUDA)
            t0 = time.perf_counter()
            hidden = block.run(None, {"hidden_states": hidden, "attention_mask": a, "cos": cos, "sin": sin})[0]
            acc.blocks += time.perf_counter() - t0
            del block
            gc.collect()
        head = sess(export, "lm_head.onnx", CUDA)
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": np.asarray(hidden[:, cur - 1:cur, :], np.float16)})
        acc.head += time.perf_counter() - t0
        del head
        gc.collect()

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_gemma_stream_gpu() -> FwdTimes:
    export, om = BASE / "gemma4/onnx_export", BASE / "gemma4/om"
    vision, prompt = load_bins(om)
    blocks_names = (
        "llm_block_0_5.onnx", "llm_block_5_10.onnx", "llm_block_10_15.onnx",
        "llm_block_15_20.onnx", "llm_block_20_25.onnx", "llm_block_25_30.onnx",
        "llm_block_30_35.onnx",
    )
    ple = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35))

    vfeed = {
        "pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16),
        "image_position_ids": np.ascontiguousarray(vision["image_position_ids"], np.int32),
    }
    vis = sess(export, "vision.onnx", CUDA)
    vis.run(None, vfeed)
    t0 = time.perf_counter()
    vout = vis.run(None, vfeed)[0]
    vis_t = time.perf_counter() - t0
    del vis
    gc.collect()
    mm = sess(export, "mm_proj.onnx", CUDA)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vout})[0]
    mm_t = time.perf_counter() - t0
    vision_t = vis_t + mm_t
    del mm
    gc.collect()

    pre = sess(export, "llm_preblock.onnx", CUDA)
    ple_in = prompt["per_layer_inputs"] if "per_layer_inputs" in prompt else np.fromfile(
        om / "prompt_bin/per_layer_inputs.bin", np.float16
    ).reshape(1, 512, 35, 256)
    feed = {
        "input_ids": prompt["input_ids"],
        "image_embeds": ie.astype(np.float16),
        "attention_mask": prompt["attention_mask"],
        "per_layer_inputs": ple_in,
        "position_ids": prompt["position_ids"],
    }
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk=None, fv=None, sk=None, sv=None):
        f = {
            "inputs_embeds": hidden, "full_mask": fm, "sliding_mask": sm,
            "cos_full": cf, "sin_full": sf, "cos_slide": cs, "sin_slide": ss,
            "per_layer_input": pl[:, :, lo:hi, :],
        }
        if fk is not None:
            f.update(full_k=fk, full_v=fv, slide_k=sk, slide_v=sv)
        return f

    def step():
        t0 = time.perf_counter()
        hidden, pl, fm, sm, cf, sf, cs, ss = pre.run(None, feed)
        acc.pre += time.perf_counter() - t0
        fk = fv = sk = sv = None
        for idx, (name, (lo, hi)) in enumerate(zip(blocks_names, ple)):
            block = sess(export, name, CUDA)
            t0 = time.perf_counter()
            out = block.run(None, block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk, fv, sk, sv))
            acc.blocks += time.perf_counter() - t0
            del block
            gc.collect()
            if idx == 2:
                hidden, fk, fv, sk, sv = out
            else:
                hidden = out[0]
        head = sess(export, "lm_head.onnx", CUDA)
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": hidden[:, CUR - 1:CUR, :].astype(np.float16)})
        acc.head += time.perf_counter() - t0
        del head
        gc.collect()

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_internvl_llm_gpu() -> FwdTimes:
    """LLM on GPU resident; vision precomputed on CPU (isolates decode forward)."""
    export, om = BASE / "internvl3_5/onnx_export", BASE / "internvl3_5/om"
    vision, prompt = load_bins(om)
    cur = int(prompt["attention_mask"].sum())

    vis = sess(export, "vision_448_notchunk.onnx", CPU)
    mm = sess(export, "mm_proj.onnx", CPU)
    pfeed = {"pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16)}
    vis.run(None, pfeed)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vis.run(None, pfeed)[0]})[0]
    vision_t = time.perf_counter() - t0
    del vis, mm
    gc.collect()

    pre = sess(export, "llm_preblock.onnx", CUDA)
    blocks = [sess(export, f"llm_block{i}.onnx", CUDA) for i in (1, 2, 3)]
    head = sess(export, "lm_head.onnx", CUDA)
    ids, attn, pos = prompt["input_ids"].copy(), prompt["attention_mask"].copy(), prompt["position_ids"].copy()
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def step():
        t0 = time.perf_counter()
        hidden, a, cos, sin = pre.run(None, {
            "image_embeds": np.ascontiguousarray(ie, np.float16),
            "attention_mask": attn, "input_ids": ids, "position_ids": pos,
        })
        acc.pre += time.perf_counter() - t0
        t0 = time.perf_counter()
        for b in blocks:
            hidden = b.run(None, {"hidden_states": hidden, "attention_mask": a, "cos": cos, "sin": sin})[0]
        acc.blocks += time.perf_counter() - t0
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": np.asarray(hidden[:, cur - 1:cur, :], np.float16)})
        acc.head += time.perf_counter() - t0

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_gemma_llm_gpu() -> FwdTimes:
    """LLM on GPU resident; vision precomputed on CPU."""
    export, om = BASE / "gemma4/onnx_export", BASE / "gemma4/om"
    vision, prompt = load_bins(om)
    blocks_names = (
        "llm_block_0_5.onnx", "llm_block_5_10.onnx", "llm_block_10_15.onnx",
        "llm_block_15_20.onnx", "llm_block_20_25.onnx", "llm_block_25_30.onnx",
        "llm_block_30_35.onnx",
    )
    ple = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35))

    vis = sess(export, "vision.onnx", CPU)
    mm = sess(export, "mm_proj.onnx", CPU)
    vfeed = {
        "pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16),
        "image_position_ids": np.ascontiguousarray(vision["image_position_ids"], np.int32),
    }
    vis.run(None, vfeed)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vis.run(None, vfeed)[0]})[0]
    vision_t = time.perf_counter() - t0
    del vis, mm
    gc.collect()

    pre = sess(export, "llm_preblock.onnx", CUDA)
    blocks = [sess(export, n, prov=CUDA) for n in blocks_names]
    head = sess(export, "lm_head.onnx", CUDA)
    ple_in = prompt["per_layer_inputs"] if "per_layer_inputs" in prompt else np.fromfile(
        om / "prompt_bin/per_layer_inputs.bin", np.float16
    ).reshape(1, 512, 35, 256)
    feed = {
        "input_ids": prompt["input_ids"],
        "image_embeds": ie.astype(np.float16),
        "attention_mask": prompt["attention_mask"],
        "per_layer_inputs": ple_in,
        "position_ids": prompt["position_ids"],
    }
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk=None, fv=None, sk=None, sv=None):
        f = {
            "inputs_embeds": hidden, "full_mask": fm, "sliding_mask": sm,
            "cos_full": cf, "sin_full": sf, "cos_slide": cs, "sin_slide": ss,
            "per_layer_input": pl[:, :, lo:hi, :],
        }
        if fk is not None:
            f.update(full_k=fk, full_v=fv, slide_k=sk, slide_v=sv)
        return f

    def step():
        t0 = time.perf_counter()
        hidden, pl, fm, sm, cf, sf, cs, ss = pre.run(None, feed)
        acc.pre += time.perf_counter() - t0
        fk = fv = sk = sv = None
        t0 = time.perf_counter()
        for idx, (block, (lo, hi)) in enumerate(zip(blocks, ple)):
            out = block.run(None, block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk, fv, sk, sv))
            if idx == 2:
                hidden, fk, fv, sk, sv = out
            else:
                hidden = out[0]
        acc.blocks += time.perf_counter() - t0
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": hidden[:, CUR - 1:CUR, :].astype(np.float16)})
        acc.head += time.perf_counter() - t0

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_internvl(cpu: bool) -> FwdTimes:
    export, om = BASE / "internvl3_5/onnx_export", BASE / "internvl3_5/om"
    prov = CPU if cpu else CUDA
    vision, prompt = load_bins(om)
    cur = int(prompt["attention_mask"].sum())

    vis = sess(export, "vision_448_notchunk.onnx", prov)
    mm = sess(export, "mm_proj.onnx", prov)
    pfeed = {"pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16)}
    vis.run(None, pfeed)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vis.run(None, pfeed)[0]})[0]
    vision_t = time.perf_counter() - t0

    pre = sess(export, "llm_preblock.onnx", prov)
    blocks = [sess(export, f"llm_block{i}.onnx", prov) for i in (1, 2, 3)]
    head = sess(export, "lm_head.onnx", prov)
    ids, attn, pos = prompt["input_ids"].copy(), prompt["attention_mask"].copy(), prompt["position_ids"].copy()
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def step():
        t0 = time.perf_counter()
        hidden, a, cos, sin = pre.run(None, {
            "image_embeds": np.ascontiguousarray(ie, np.float16),
            "attention_mask": attn, "input_ids": ids, "position_ids": pos,
        })
        acc.pre += time.perf_counter() - t0
        t0 = time.perf_counter()
        for b in blocks:
            hidden = b.run(None, {"hidden_states": hidden, "attention_mask": a, "cos": cos, "sin": sin})[0]
        acc.blocks += time.perf_counter() - t0
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": np.asarray(hidden[:, cur - 1:cur, :], np.float16)})
        acc.head += time.perf_counter() - t0

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def bench_gemma(cpu: bool) -> FwdTimes:
    export, om = BASE / "gemma4/onnx_export", BASE / "gemma4/om"
    prov = CPU if cpu else CUDA
    vision, prompt = load_bins(om)
    blocks_names = (
        "llm_block_0_5.onnx", "llm_block_5_10.onnx", "llm_block_10_15.onnx",
        "llm_block_15_20.onnx", "llm_block_20_25.onnx", "llm_block_25_30.onnx",
        "llm_block_30_35.onnx",
    )
    ple = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35))

    vis = sess(export, "vision.onnx", prov)
    mm = sess(export, "mm_proj.onnx", prov)
    vfeed = {
        "pixel_values": np.ascontiguousarray(vision["pixel_values"], np.float16),
        "image_position_ids": np.ascontiguousarray(vision["image_position_ids"], np.int32),
    }
    vis.run(None, vfeed)
    t0 = time.perf_counter()
    ie = mm.run(None, {"vision_features": vis.run(None, vfeed)[0]})[0]
    vision_t = time.perf_counter() - t0

    pre = sess(export, "llm_preblock.onnx", prov)
    blocks = [sess(export, n, prov) for n in blocks_names]
    head = sess(export, "lm_head.onnx", prov)
    ple_in = prompt.get("per_layer_inputs")
    if ple_in is None:
        ple_in = np.fromfile(om / "prompt_bin/per_layer_inputs.bin", np.float16).reshape(1, 512, 35, 256)
    feed = {
        "input_ids": prompt["input_ids"],
        "image_embeds": ie.astype(np.float16),
        "attention_mask": prompt["attention_mask"],
        "per_layer_inputs": ple_in,
        "position_ids": prompt["position_ids"],
    }
    acc = FwdTimes(vision_t, 0.0, 0.0, 0.0)

    def block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk=None, fv=None, sk=None, sv=None):
        f = {
            "inputs_embeds": hidden, "full_mask": fm, "sliding_mask": sm,
            "cos_full": cf, "sin_full": sf, "cos_slide": cs, "sin_slide": ss,
            "per_layer_input": pl[:, :, lo:hi, :],
        }
        if fk is not None:
            f.update(full_k=fk, full_v=fv, slide_k=sk, slide_v=sv)
        return f

    def step():
        t0 = time.perf_counter()
        hidden, pl, fm, sm, cf, sf, cs, ss = pre.run(None, feed)
        acc.pre += time.perf_counter() - t0
        fk = fv = sk = sv = None
        t0 = time.perf_counter()
        for idx, (block, (lo, hi)) in enumerate(zip(blocks, ple)):
            out = block.run(None, block_feed(hidden, pl, fm, sm, cf, sf, cs, ss, lo, hi, fk, fv, sk, sv))
            if idx == 2:
                hidden, fk, fv, sk, sv = out
            else:
                hidden = out[0]
        acc.blocks += time.perf_counter() - t0
        t0 = time.perf_counter()
        head.run(None, {"hidden_states": hidden[:, CUR - 1:CUR, :].astype(np.float16)})
        acc.head += time.perf_counter() - t0

    step()
    acc.pre = acc.blocks = acc.head = 0.0
    for _ in range(N):
        step()
    return _avg(acc)


def print_row(model: str, mode: str, t: FwdTimes):
    print(
        f"| {model:<12} | {mode:<14} | {t.vision:6.3f} | {t.pre:6.3f} | {t.blocks:6.3f} | "
        f"{t.head:6.3f} | {t.decode_step:6.3f} | {t.e2e_50:7.1f} |"
    )


def main():
    import sys
    print(f"ThorU forward-only timing (N={N} steps avg, est {TOK50} tok)")
    print("Method: all InferenceSessions loaded before timing; only session.run() counted")
    print()
    print("| model        | EP             | vision | pre    | blocks | head   | decode/step | e2e@50tok |")
    print("|--------------|----------------|--------|--------|--------|--------|-------------|-----------|")

    cases = [
        ("Qwen3-VL", "CPU", lambda: bench_qwen(True, False)),
        ("Qwen3-VL", "GPU", bench_qwen_stream_gpu),
        ("InternVL", "CPU", lambda: bench_internvl(True)),
        ("InternVL", "GPU", bench_internvl_stream_gpu),
        ("Gemma-4", "CPU", lambda: bench_gemma(True)),
        ("Gemma-4", "GPU", bench_gemma_stream_gpu),
    ]
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        cases = [cases[idx]]

    for model, mode, fn in cases:
        try:
            print_row(model, mode, fn())
        except Exception as e:
            print(f"| {model:<12} | {mode:<14} | FAIL: {e}")
        gc.collect()

    print()
    print("Notes:")
    print("- All GPU rows: vision/mm_proj + LLM all CUDA; only session.run() counted")
    print("- vision/mm: sequential load (vision run -> unload -> mm run); load excluded")
    print("- LLM GPU: stream mode (pre resident, blocks/head reload per step)")
    print("- decode/step = pre + blocks + head; e2e@50tok = vision + decode/step × 50")


if __name__ == "__main__":
    main()
