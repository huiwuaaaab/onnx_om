#!/usr/bin/env python3
"""
OM pipeline bin helpers (stdlib only, board-friendly).

Aligned with test/onnx_torch_test_it.py:
  vision -> mm_proj -> llm_preblock -> b1..b7 -> lm_head [1,1,1536]
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import struct
import tempfile
from functools import lru_cache
from pathlib import Path

IMAGE_MASK_START = int(os.environ.get("GEMMA_IMAGE_MASK_START", "5"))
IMAGE_MASK_END = int(os.environ.get("GEMMA_IMAGE_MASK_END", "261"))
DEFAULT_PAD_TOKEN_ID = int(os.environ.get("GEMMA_PAD_TOKEN_ID", "0"))
DEFAULT_STATIC_PREBLOCK = os.environ.get(
    "GEMMA4_STATIC_PREBLOCK",
    "/home/mdc/guanxj/mdc_aoe/weights_wuhui/gemma4/dump/llm_preblock",
)

SEQ_LEN = 512
NUM_LAYERS = 35
PLE_DIM = 256
PLE_ROW_BYTES = NUM_LAYERS * PLE_DIM * 2
PLE_COL_BYTES = PLE_ROW_BYTES
PLE_SEQ_STRIDE = PLE_ROW_BYTES

HIDDEN_DIM = 1536
LM_HEAD_IN_BYTES = 1 * 1 * HIDDEN_DIM * 2
BLOCK_HIDDEN_BYTES = 1 * SEQ_LEN * HIDDEN_DIM * 2
VOCAB_SIZE = 262144
LM_HEAD_LOGITS_BYTES = 1 * 1 * VOCAB_SIZE * 2

# ONNX fp16 sizes (llm_preblock / llm_block_0_5)
PREBLOCK_OUT_BYTES = {
    "inputs_embeds_out": 1 * SEQ_LEN * HIDDEN_DIM * 2,
    "per_layer_inputs_out": 1 * SEQ_LEN * NUM_LAYERS * PLE_DIM * 2,
    "full_mask": 1 * 1 * SEQ_LEN * SEQ_LEN * 2,
    "sliding_mask": 1 * 1 * SEQ_LEN * SEQ_LEN * 2,
    "cos_full": 1 * SEQ_LEN * SEQ_LEN * 2,
    "sin_full": 1 * SEQ_LEN * SEQ_LEN * 2,
    "cos_sliding": 1 * SEQ_LEN * PLE_DIM * 2,
    "sin_sliding": 1 * SEQ_LEN * PLE_DIM * 2,
}
BLOCK_PLE_SLICE_BYTES = 1 * SEQ_LEN * 5 * PLE_DIM * 2
VISION_OUT_BYTES = 1 * 256 * 768 * 2
MM_PROJ_OUT_BYTES = 1 * 256 * HIDDEN_DIM * 2
BLOCK_IN_BYTES = {
    "inputs_embeds": PREBLOCK_OUT_BYTES["inputs_embeds_out"],
    "full_mask": PREBLOCK_OUT_BYTES["full_mask"],
    "sliding_mask": PREBLOCK_OUT_BYTES["sliding_mask"],
    "cos_full": PREBLOCK_OUT_BYTES["cos_full"],
    "sin_full": PREBLOCK_OUT_BYTES["sin_full"],
    "cos_slide": PREBLOCK_OUT_BYTES["cos_sliding"],
    "sin_slide": PREBLOCK_OUT_BYTES["sin_sliding"],
    "per_layer_input": BLOCK_PLE_SLICE_BYTES,
}


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _clear_bins(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.bin"):
        _safe_unlink(f)


def _msame_write_inputs(
    out_dir: Path,
    indexed: list[tuple[int, Path]],
    expected_sizes: dict[int, int] | None = None,
) -> None:
    """msame 输入目录只能有 0.bin..N.bin，不能有其它 .bin（否则会按单输入多次跑）。"""
    _clear_bins(out_dir)
    for idx, src in indexed:
        if not src.is_file():
            raise FileNotFoundError(src)
        nbytes = src.stat().st_size
        if expected_sizes and idx in expected_sizes:
            want = expected_sizes[idx]
            if nbytes != want:
                raise ValueError(
                    f"msame input {idx}.bin size {nbytes} != expected {want} "
                    f"(src={src})"
                )
        shutil.copy2(src, out_dir / f"{idx}.bin")
        print(f"  [{idx}] {nbytes} bytes <- {src.name}")
    print(f"msame inputs ({len(indexed)}): {out_dir}")


def _read_i32_array(path: Path, n: int = SEQ_LEN) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{n}i", data[: n * 4]))


def _write_i32_array(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


@lru_cache(maxsize=1)
def _ple_table_meta(ple_table_bin: str) -> dict:
    meta_path = Path(ple_table_bin).parent / "meta.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"shape": [262144, NUM_LAYERS, PLE_DIM]}


def _ple_row_from_table(ple_table_bin: Path, token_id: int) -> bytes:
    offset = token_id * PLE_ROW_BYTES
    with open(ple_table_bin, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm[offset : offset + PLE_ROW_BYTES]


def _llm_token_id(input_ids: list[int], pos: int, pad_token_id: int) -> int:
    if IMAGE_MASK_START <= pos < IMAGE_MASK_END:
        return pad_token_id
    return input_ids[pos]


def _list_bins_with_sizes(out_dir: Path) -> list[tuple[int, Path]]:
    root = _resolve_msame_out_dir(out_dir)
    if not root.is_dir():
        return []
    return sorted(
        ((p.stat().st_size, p) for p in root.rglob("*.bin")),
        key=lambda x: (x[0], str(x[1])),
    )


def _find_output_bin(
    out_dir: Path,
    preferred: tuple[str, ...] = (),
    *,
    size: int | None = None,
    label: str = "tensor",
) -> Path:
    root = _resolve_msame_out_dir(out_dir)

    def _ok(p: Path) -> bool:
        return p.is_file() and (size is None or p.stat().st_size == size)

    for name in preferred:
        p = root / name
        if _ok(p):
            return p
        # msame: vision_<model>_output_0.bin
        if name.endswith(".bin"):
            hits = [p for p in root.rglob(f"*{name}") if _ok(p)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise FileNotFoundError(
                    f"Ambiguous {label} *{name} under {out_dir}: {[str(h) for h in hits]}"
                )

    if size is not None:
        return _find_bin_by_size(out_dir, size, label)

    bins = _list_bins_with_sizes(out_dir)
    if len(bins) == 1 and _ok(bins[0][1]):
        return bins[0][1]
    if not bins:
        raise FileNotFoundError(
            f"No .bin under {out_dir}, expected one of {preferred}"
            + (f" or size={size}" if size else "")
        )
    sizes = ", ".join(f"{s}:{p.name}" for s, p in bins)
    raise FileNotFoundError(
        f"Cannot resolve {label} under {out_dir} (preferred={preferred}); "
        f"found bins: {sizes}"
    )


def _find_bin_by_size(out_dir: Path, size: int, label: str) -> Path:
    root = _resolve_msame_out_dir(out_dir)
    matches = [p for p in root.rglob("*.bin") if p.stat().st_size == size]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {label} bin ({size} bytes) under {out_dir}")
    raise FileNotFoundError(
        f"Ambiguous {label} under {out_dir}: {[str(p) for p in matches]}"
    )


def _resolve_msame_out_dir(out_dir: Path) -> Path:
    """msame may write under timestamp subdir."""
    if any(out_dir.glob("*.bin")):
        return out_dir
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()] if out_dir.is_dir() else []
    for sub in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
        if any(sub.glob("*.bin")):
            return sub
    return out_dir


def cmd_sync_preblock_state(args: argparse.Namespace) -> None:
    """Reset decode state from dump/llm_preblock (avoid polluted om_work_it/state)."""
    dump = Path(args.dump_preblock_dir)
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask", "per_layer_inputs"):
        src = dump / f"{name}.bin"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, state / f"{name}.bin")
    _safe_unlink(state / "last_token.txt")
    _safe_unlink(state / "accepted_tokens.log")
    print(f"sync-preblock-state: {state} <- {dump}")


def cmd_prepare_vision_input(args: argparse.Namespace) -> None:
    dump = Path(args.dump_vision_dir)
    _msame_write_inputs(
        Path(args.out_dir),
        [
            (0, dump / "pixel_values.bin"),
            (1, dump / "image_position_ids.bin"),
        ],
    )


def cmd_prepare_mmproj_input(args: argparse.Namespace) -> None:
    vision_out = Path(args.vision_out_dir)
    hidden = _find_output_bin(
        vision_out,
        ("hidden_states.bin", "0.bin", "output_0.bin"),
        size=VISION_OUT_BYTES,
        label="vision_hidden_states",
    )
    _msame_write_inputs(Path(args.out_dir), [(0, hidden)])


def _resolve_static_preblock_dir(args: argparse.Namespace) -> Path:
    """position_ids 固定来自 dump/llm_preblock，不在 om_work_it/state。"""
    candidates: list[Path] = []
    if args.static_preblock_dir:
        candidates.append(Path(args.static_preblock_dir))
    if os.environ.get("DUMP_PREBLOCK"):
        candidates.append(Path(os.environ["DUMP_PREBLOCK"]))
    if os.environ.get("GEMMA4_ROOT"):
        candidates.append(Path(os.environ["GEMMA4_ROOT"]) / "dump" / "llm_preblock")
    candidates.append(Path(DEFAULT_STATIC_PREBLOCK))
    if args.dump_preblock_dir:
        p = Path(args.dump_preblock_dir)
        if (p / "position_ids.bin").is_file():
            candidates.insert(0, p)
    for c in candidates:
        if (c / "position_ids.bin").is_file():
            return c
    raise FileNotFoundError(
        "position_ids.bin not found; set --static-preblock-dir or DUMP_PREBLOCK "
        f"(tried {[str(c) for c in candidates]})"
    )


def cmd_prepare_preblock_input(args: argparse.Namespace) -> None:
    """state: 可变 input_ids/attn/ple；static: 固定 position_ids（dump/llm_preblock）。"""
    state_raw = args.state_dir or args.dump_preblock_dir
    if not state_raw:
        raise ValueError("need --state-dir or --dump-preblock-dir (om_work_it/state)")
    state = Path(state_raw)
    static = _resolve_static_preblock_dir(args)
    mm_proj_out = Path(args.mm_proj_out_dir)
    hidden = _find_output_bin(
        mm_proj_out,
        ("hidden_states.bin", "0.bin", "output_0.bin"),
        size=MM_PROJ_OUT_BYTES,
        label="mm_proj_image_embeds",
    )
    _msame_write_inputs(
        Path(args.out_dir),
        [
            (0, state / "input_ids.bin"),
            (1, hidden),
            (2, state / "attention_mask.bin"),
            (3, state / "per_layer_inputs.bin"),
            (4, static / "position_ids.bin"),
        ],
    )


_PREBLOCK_NAMES = {
    "inputs_embeds": ("inputs_embeds_out.bin", "0.bin", "output_0.bin"),
    "per_layer_inputs": ("per_layer_inputs_out.bin", "1.bin", "output_1.bin"),
    "full_mask": ("full_mask.bin", "2.bin", "output_2.bin"),
    "sliding_mask": ("sliding_mask.bin", "3.bin", "output_3.bin"),
    "cos_full": ("cos_full.bin", "4.bin", "output_4.bin"),
    "sin_full": ("sin_full.bin", "5.bin", "output_5.bin"),
    "cos_slide": ("cos_sliding.bin", "cos_slide.bin", "6.bin", "output_6.bin"),
    "sin_slide": ("sin_sliding.bin", "sin_slide.bin", "7.bin", "output_7.bin"),
}
_PREBLOCK_SIZE_KEY = {
    "inputs_embeds": "inputs_embeds_out",
    "per_layer_inputs": "per_layer_inputs_out",
    "full_mask": "full_mask",
    "sliding_mask": "sliding_mask",
    "cos_full": "cos_full",
    "sin_full": "sin_full",
    "cos_slide": "cos_sliding",
    "sin_slide": "sin_sliding",
}


def _resolve_preblock_out(pre_out: Path, key: str) -> Path:
    size_key = _PREBLOCK_SIZE_KEY[key]
    return _find_output_bin(
        pre_out,
        _PREBLOCK_NAMES[key],
        size=PREBLOCK_OUT_BYTES[size_key],
        label=key,
    )


def _build_ple_slice(
    ple_path: Path, layer_start: int, layer_end: int
) -> Path:
    want_ple = PREBLOCK_OUT_BYTES["per_layer_inputs_out"]
    got = ple_path.stat().st_size
    if got != want_ple:
        raise ValueError(
            f"per_layer_inputs_out size {got} != {want_ple} ({ple_path}); "
            "preblock OM output wrong or wrong bin picked"
        )
    ple_slice = Path(tempfile.mkdtemp()) / "per_layer_input.bin"
    with open(ple_path, "rb") as fin, open(ple_slice, "wb") as fout:
        ple = mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
        for seq in range(SEQ_LEN):
            base = seq * PLE_SEQ_STRIDE
            for layer in range(layer_start, layer_end):
                off = base + layer * PLE_DIM * 2
                fout.write(ple[off : off + PLE_DIM * 2])
    if ple_slice.stat().st_size != BLOCK_PLE_SLICE_BYTES:
        raise ValueError(
            f"per_layer_input slice {ple_slice.stat().st_size} != "
            f"{BLOCK_PLE_SLICE_BYTES}"
        )
    return ple_slice


def cmd_prepare_block_input(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pre_out = Path(args.pre_out_dir)
    block_idx = int(args.block_idx)
    layer_start = (block_idx - 1) * 5
    layer_end = layer_start + 5

    if block_idx == 1:
        embed = _resolve_preblock_out(pre_out, "inputs_embeds")
    else:
        embed = _find_output_bin(
            Path(args.prev_block_out_dir),
            ("hidden_states_out.bin", "0.bin", "output_0.bin"),
            size=BLOCK_HIDDEN_BYTES,
            label="hidden_states_out",
        )

    ple_path = _resolve_preblock_out(pre_out, "per_layer_inputs")
    ple_slice = _build_ple_slice(ple_path, layer_start, layer_end)

    indexed: list[tuple[int, Path]] = [
        (0, embed),
        (1, _resolve_preblock_out(pre_out, "full_mask")),
        (2, _resolve_preblock_out(pre_out, "sliding_mask")),
        (3, _resolve_preblock_out(pre_out, "cos_full")),
        (4, _resolve_preblock_out(pre_out, "sin_full")),
        (5, _resolve_preblock_out(pre_out, "cos_slide")),
        (6, _resolve_preblock_out(pre_out, "sin_slide")),
        (7, ple_slice),
    ]

    if block_idx >= 4:
        b3_out = Path(args.b3_out_dir)
        indexed.extend(
            [
                (8, _find_output_bin(b3_out, ("out_full_k.bin", "1.bin", "output_1.bin"))),
                (9, _find_output_bin(b3_out, ("out_full_v.bin", "2.bin", "output_2.bin"))),
                (10, _find_output_bin(b3_out, ("out_slide_k.bin", "3.bin", "output_3.bin"))),
                (11, _find_output_bin(b3_out, ("out_slide_v.bin", "4.bin", "output_4.bin"))),
            ]
        )

    expected = {i: BLOCK_IN_BYTES[k] for i, k in enumerate(
        ("inputs_embeds", "full_mask", "sliding_mask", "cos_full", "sin_full",
         "cos_slide", "sin_slide", "per_layer_input")
    )}
    _msame_write_inputs(out_dir, indexed, expected_sizes=expected)
    shutil.rmtree(ple_slice.parent, ignore_errors=True)


def cmd_prepare_lm_head_input(args: argparse.Namespace) -> None:
    """Slice b7 hidden[:, cur_len-1] -> hidden_states.bin [1,1,1536]."""
    out = Path(args.out_dir)
    b7_out = Path(args.b7_out_dir)
    try:
        hidden_path = _find_bin_by_size(b7_out, BLOCK_HIDDEN_BYTES, "b7_hidden")
    except FileNotFoundError:
        hidden_path = _find_output_bin(
            b7_out, ("hidden_states_out.bin", "0.bin", "output_0.bin")
        )
    pos = int(args.cur_len) - 1
    if pos < 0:
        raise ValueError(f"cur_len must be >= 1, got {args.cur_len}")
    offset = pos * HIDDEN_DIM * 2
    tmp = Path(tempfile.mkdtemp()) / "0.bin"
    with open(hidden_path, "rb") as fin, open(tmp, "wb") as fout:
        fin.seek(offset)
        chunk = fin.read(LM_HEAD_IN_BYTES)
        if len(chunk) != LM_HEAD_IN_BYTES:
            raise ValueError(f"short read at pos={pos}: {len(chunk)} bytes")
        fout.write(chunk)
    _msame_write_inputs(out, [(0, tmp)])
    shutil.rmtree(tmp.parent, ignore_errors=True)
    print(f"lm_head input: hidden[:, {pos}] from {hidden_path}")


def _argmax_fp16_logits(mm: mmap.mmap) -> int:
    best_i, best_v = 0, float("-inf")
    for i in range(VOCAB_SIZE):
        (v,) = struct.unpack_from("<e", mm, i * 2)
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def cmd_update_decode_state(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    lm_out = Path(args.lm_head_out_dir)
    try:
        logits_path = _find_bin_by_size(lm_out, LM_HEAD_LOGITS_BYTES, "lm_head_logits")
    except FileNotFoundError:
        logits_path = _find_output_bin(lm_out, ("logits.bin", "0.bin", "output_0.bin"))

    cur_len = int(args.cur_len)
    with open(logits_path, "rb") as f:
        log_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if len(log_mm) < LM_HEAD_LOGITS_BYTES:
            raise ValueError(f"logits size {len(log_mm)} < {LM_HEAD_LOGITS_BYTES}")
        next_id = _argmax_fp16_logits(log_mm)

    input_ids = _read_i32_array(state_dir / "input_ids.bin", SEQ_LEN)
    attn = _read_i32_array(state_dir / "attention_mask.bin", SEQ_LEN)
    input_ids[cur_len] = next_id
    attn[cur_len] = 1
    _write_i32_array(state_dir / "input_ids.bin", input_ids)
    _write_i32_array(state_dir / "attention_mask.bin", attn)
    (state_dir / "last_token.txt").write_text(f"{next_id}\n", encoding="utf-8")
    print(f"step={args.step} cur_len={cur_len} next_token={next_id}")

    if args.ple_table:
        _patch_ple_column(state_dir, Path(args.ple_table), cur_len, int(args.pad_token_id))


def cmd_patch_ple_column(args: argparse.Namespace) -> None:
    _patch_ple_column(
        Path(args.state_dir),
        Path(args.ple_table),
        int(args.pos),
        int(args.pad_token_id),
    )


def _patch_ple_column(
    state_dir: Path,
    ple_table_bin: Path,
    pos: int,
    pad_token_id: int,
    token_id: int | None = None,
) -> None:
    if IMAGE_MASK_START <= pos < IMAGE_MASK_END:
        token_id = pad_token_id
    elif token_id is None:
        input_ids = _read_i32_array(state_dir / "input_ids.bin", SEQ_LEN)
        token_id = _llm_token_id(input_ids, pos, pad_token_id)
    row = _ple_row_from_table(ple_table_bin, token_id)
    ple_path = state_dir / "per_layer_inputs.bin"
    col_off = pos * PLE_COL_BYTES
    with open(ple_path, "r+b") as f:
        f.seek(col_off)
        f.write(row)
    print(f"patch per_layer_inputs[:, {pos}] <- ple_table[{token_id}]")


def cmd_init_cur_len(args: argparse.Namespace) -> None:
    attn = _read_i32_array(Path(args.state_dir) / "attention_mask.bin", SEQ_LEN)
    cur_len = sum(attn)
    Path(args.out_file).write_text(str(cur_len), encoding="utf-8")
    print(cur_len)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync-preblock-state")
    s.add_argument("--dump-preblock-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.set_defaults(func=cmd_sync_preblock_state)

    s = sub.add_parser("init-cur-len")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--out-file", required=True)
    s.set_defaults(func=cmd_init_cur_len)

    s = sub.add_parser("prepare-vision-input")
    s.add_argument("--dump-vision-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_vision_input)

    s = sub.add_parser("prepare-mmproj-input")
    s.add_argument("--vision-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_mmproj_input)

    s = sub.add_parser("prepare-preblock-input")
    s.add_argument("--state-dir", default="", help="om_work_it/state（decode 可变）")
    s.add_argument(
        "--static-preblock-dir",
        default="",
        help="dump/llm_preblock，position_ids.bin；可省略则用 DUMP_PREBLOCK/GEMMA4_ROOT",
    )
    s.add_argument(
        "--dump-preblock-dir",
        default="",
        help="(兼容旧 bash) 仅 --dump-preblock-dir=state 时作 state-dir",
    )
    s.add_argument("--mm-proj-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_preblock_input)

    s = sub.add_parser("prepare-block-input")
    s.add_argument("--pre-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--block-idx", type=int, required=True)
    s.add_argument("--prev-block-out-dir", default="")
    s.add_argument("--b3-out-dir", default="")
    s.set_defaults(func=cmd_prepare_block_input)

    s = sub.add_parser("prepare-lm-head-input")
    s.add_argument("--b7-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.set_defaults(func=cmd_prepare_lm_head_input)

    s = sub.add_parser("update-decode-state")
    s.add_argument("--lm-head-out-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.add_argument("--step", type=int, default=0)
    s.add_argument("--ple-table", default="")
    s.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_TOKEN_ID)
    s.set_defaults(func=cmd_update_decode_state)

    s = sub.add_parser("patch-ple-column")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--ple-table", required=True)
    s.add_argument("--pos", type=int, required=True)
    s.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_TOKEN_ID)
    s.set_defaults(func=cmd_patch_ple_column)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
