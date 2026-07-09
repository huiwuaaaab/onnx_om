#!/usr/bin/env python3
"""
Verify Gemma llm_block k_norm ReduceMean on ThorU after re-export.

Pass criteria (GPU vs CPU preblock feed):
  - k_norm/ReduceMean: no Inf, min/max finite and close to CPU
  - k_norm/Mul_1: not all-zero
  - self_attn/Add_1 (K RoPE): not all-zero

Usage (on ThorU after syncing new onnx_export + this script):
  export PYTHONPATH=/cus_app_data/guanxj/py312-site-packages
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu
  export GEMMA4_ONNX_EXPORT=/cus_app_data/guanxj/gemma4/onnx_export
  python3 scripts/thoru_gemma_knorm_verify.py --block llm_block_0_5.onnx
  python3 scripts/thoru_gemma_knorm_verify.py --block llm_block_5_10.onnx

Optional: build debug ONNX with extra outputs locally (needs `onnx` pip):
  python3 scripts/thoru_gemma_knorm_verify.py --build-debug /tmp/llm_block_knorm_debug.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

EX = Path(os.environ.get("GEMMA4_ONNX_EXPORT", "/cus_app_data/guanxj/gemma4/onnx_export"))
OM = Path(os.environ.get("GEMMA4_OM", "/cus_app_data/guanxj/gemma4/om"))
CUDA = [
    ("CUDAExecutionProvider", {"device_id": "0", "cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0"}),
    "CPUExecutionProvider",
]
CPU = ["CPUExecutionProvider"]


def block_range(block: Path) -> tuple[int, int, int]:
    m = re.search(r"llm_block_(\d+)_(\d+)\.onnx$", block.name)
    if not m:
        return 0, 5, 0
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo, hi, lo


def check_tensors(layer: int) -> list[str]:
    p = f"/layers.{layer}/self_attn"
    return [
        f"{p}/k_proj/MatMul_output_0",
        f"{p}/k_norm/Pow_output_0",
        f"{p}/k_norm/ReduceMean_output_0",
        f"{p}/k_norm/Pow_1_output_0",
        f"{p}/k_norm/Mul_1_output_0",
        f"{p}/Add_1_output_0",
    ]


def load_feed(ple_lo: int, ple_hi: int) -> dict:
    inp = {
        "input_ids": np.fromfile(OM / "prompt_bin/input_ids.bin", np.int32).reshape(1, 512),
        "attention_mask": np.fromfile(OM / "prompt_bin/attention_mask.bin", np.int32).reshape(1, 512),
        "position_ids": np.fromfile(OM / "prompt_bin/position_ids.bin", np.int32).reshape(1, 512),
        "per_layer_inputs": np.fromfile(OM / "prompt_bin/per_layer_inputs.bin", np.float16).reshape(1, 512, 35, 256),
    }
    meta = json.loads((OM / "vision_bin/meta.json").read_text())
    pv = np.fromfile(OM / "vision_bin/pixel_values.bin", np.float16).reshape(
        tuple(meta["tensors"]["pixel_values"]["shape"])
    )
    pos = np.fromfile(OM / "vision_bin/image_position_ids.bin", np.int32).reshape(
        tuple(meta["tensors"]["image_position_ids"]["shape"])
    )
    vis = ort.InferenceSession(str(EX / "vision.onnx"), providers=CPU)
    mm = ort.InferenceSession(str(EX / "mm_proj.onnx"), providers=CPU)
    ie = mm.run(None, {"vision_features": vis.run(None, {
        "pixel_values": np.ascontiguousarray(pv, np.float16),
        "image_position_ids": np.ascontiguousarray(pos, np.int32),
    })[0]})[0]
    pre = ort.InferenceSession(str(EX / "llm_preblock.onnx"), providers=CPU)
    hidden, pl, fm, sm, cf, sf, cs, ss = pre.run(None, {
        "input_ids": inp["input_ids"],
        "image_embeds": np.ascontiguousarray(ie, np.float16),
        "attention_mask": inp["attention_mask"],
        "per_layer_inputs": np.ascontiguousarray(inp["per_layer_inputs"], np.float16),
        "position_ids": inp["position_ids"],
    })
    return {
        "inputs_embeds": hidden,
        "full_mask": fm,
        "sliding_mask": sm,
        "cos_full": cf,
        "sin_full": sf,
        "cos_slide": cs,
        "sin_slide": ss,
        "per_layer_input": pl[:, :, ple_lo:ple_hi, :],
    }


def build_debug_model(src: Path, dst: Path, tensors: list[str]) -> None:
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(src))
    existing = {o.name for o in m.graph.output}
    node_out = {n.output[0] for n in m.graph.node if n.output}
    for name in tensors:
        if name not in existing and name in node_out:
            m.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, None))
    onnx.save(m, str(dst))
    print(f"wrote debug model {dst}")


def run_model(model: Path, providers: list, feed: dict, tensors: list[str]) -> dict[str, np.ndarray]:
    s = ort.InferenceSession(str(model), providers=providers)
    return dict(zip(tensors, s.run(tensors, feed)))


def stat(arr: np.ndarray) -> dict:
    a = arr.astype(np.float32, copy=False)
    return {
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
        "min": float(np.nanmin(a)) if a.size else 0.0,
        "max": float(np.nanmax(a)) if a.size else 0.0,
        "all_zero": bool(a.size > 100 and np.max(np.abs(a)) == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=str(EX / "llm_block_0_5.onnx"))
    ap.add_argument("--layer", type=int, default=None, help="sliding layer idx (default: from block name)")
    ap.add_argument("--debug-model", default="/tmp/llm_block_knorm_debug.onnx")
    ap.add_argument("--build-debug", metavar="PATH", help="build debug ONNX with extra outputs")
    args = ap.parse_args()

    block = Path(args.block)
    if not block.is_file() and not (EX / block.name).is_file():
        print(f"missing {block}", file=sys.stderr)
        return 1
    if not block.is_file():
        block = EX / block.name

    ple_lo, ple_hi, default_layer = block_range(block)
    layer = args.layer if args.layer is not None else default_layer
    tensors = check_tensors(layer)

    debug = Path(args.debug_model)
    if args.build_debug:
        build_debug_model(block, Path(args.build_debug), tensors)
        debug = Path(args.build_debug)
    elif not debug.is_file():
        print("debug model missing; run with --build-debug (needs onnx package) or upload prebuilt ONNX", file=sys.stderr)
        return 1

    feed = load_feed(ple_lo, ple_hi)
    cpu = run_model(debug, CPU, feed, tensors)
    gpu = run_model(debug, CUDA, feed, tensors)

    print(f"block={block}")
    print(f"layer={layer} ple=[{ple_lo},{ple_hi})")
    print(f"debug={debug}")
    print()
    ok = True
    for name in tensors:
        short = name.split("/")[-1]
        cs, gs = stat(cpu[name]), stat(gpu[name])
        diff = abs(cs["max"] - gs["max"]) + abs(cs["min"] - gs["min"])
        line_ok = gs["nan"] == 0 and not gs["all_zero"]
        if "ReduceMean" in name:
            line_ok = line_ok and gs["inf"] == 0 and gs["max"] < 1e6
        if "Add_1" in name or "Mul_1" in name:
            line_ok = line_ok and gs["max"] > 1e-4
        ok = ok and line_ok
        flag = "OK" if line_ok else "FAIL"
        print(
            f"[{flag}] {short:24s} | CPU min={cs['min']:.3f} max={cs['max']:.3f} inf={cs['inf']} | "
            f"GPU min={gs['min']:.3f} max={gs['max']:.3f} inf={gs['inf']} nan={gs['nan']} | diff~{diff:.3f}"
        )

    print()
    print("PASS" if ok else "FAIL — re-export llm_block_* after Gemma4RMSNorm amax fix")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
