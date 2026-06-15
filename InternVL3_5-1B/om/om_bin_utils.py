#!/usr/bin/env python3
"""
InternVL3_5-1B OM pipeline bin helpers (stdlib only, board-friendly).

Aligned with test/test.py:
  vision_448 -> mm_proj -> llm_preblock -> b1..b3 -> lm_head

Only vision / preblock frontend bins are static; intermediate tensors come
from upstream OM outputs on board.
"""

from __future__ import annotations

import argparse
import mmap
import os
import shutil
import struct
import tempfile
from pathlib import Path

MAX_SEQ_LEN = 512
HIDDEN_DIM = 1024
VOCAB_SIZE = 151936

PIXEL_BYTES = 1 * 3 * 448 * 448 * 2
VISION_OUT_BYTES = 1 * 1025 * 1024 * 2
MM_PROJ_OUT_BYTES = 1 * 256 * 1024 * 2
BLOCK_HIDDEN_BYTES = 1 * MAX_SEQ_LEN * HIDDEN_DIM * 2
LM_HEAD_IN_BYTES = 1 * 1 * HIDDEN_DIM * 2
LM_HEAD_LOGITS_BYTES = 1 * 1 * VOCAB_SIZE * 2
COS_SIN_BYTES = 1 * MAX_SEQ_LEN * 128 * 2
MASK_BYTES = 1 * 1 * MAX_SEQ_LEN * MAX_SEQ_LEN * 2
POSITION_IDS_BYTES = MAX_SEQ_LEN * 4
PREBLOCK_OUT_BYTES = {
    "inputs_embeds_out": BLOCK_HIDDEN_BYTES,
    "attention_mask_out": MASK_BYTES,
    "cos": COS_SIN_BYTES,
    "sin": COS_SIN_BYTES,
}


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _clear_bins(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.bin"):
        _safe_unlink(f)


def _msame_write_inputs(
    out_dir: Path,
    indexed: list[tuple[int, Path]],
    expected_sizes: dict[int, int] | None = None,
) -> None:
    """msame input dir: only 0.bin..N.bin."""
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


def _read_i32_array(path: Path, n: int = MAX_SEQ_LEN) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{n}i", data[: n * 4]))


def _write_i32_array(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def _resolve_msame_out_dir(out_dir: Path) -> Path:
    if any(out_dir.glob("*.bin")):
        return out_dir
    if out_dir.is_dir():
        for sub in sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if sub.is_dir() and any(sub.glob("*.bin")):
                return sub
    return out_dir


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


def _resolve_static_preblock_dir(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if getattr(args, "static_preblock_dir", None):
        candidates.append(Path(args.static_preblock_dir))
    if os.environ.get("DUMP_PREBLOCK"):
        candidates.append(Path(os.environ["DUMP_PREBLOCK"]))
    if os.environ.get("DUMP_ROOT"):
        candidates.append(Path(os.environ["DUMP_ROOT"]) / "llm_preblock")
    if getattr(args, "dump_preblock_dir", None):
        candidates.append(Path(args.dump_preblock_dir))
    for c in candidates:
        if (c / "position_ids.bin").is_file():
            return c
    raise FileNotFoundError(
        "position_ids.bin not found; set --static-preblock-dir or DUMP_PREBLOCK "
        f"(tried {[str(c) for c in candidates]})"
    )


def cmd_sync_state(args: argparse.Namespace) -> None:
    dump = Path(args.dump_preblock_dir)
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids", "attention_mask"):
        src = dump / f"{name}.bin"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, state / f"{name}.bin")
    _safe_unlink(state / "last_token.txt")
    print(f"sync-state: {state} <- {dump}")


def cmd_prepare_vision_input(args: argparse.Namespace) -> None:
    dump = Path(args.dump_vision_dir)
    _msame_write_inputs(
        Path(args.out_dir),
        [(0, dump / "pixel_values.bin")],
        expected_sizes={0: PIXEL_BYTES},
    )


def cmd_prepare_mmproj_input(args: argparse.Namespace) -> None:
    vision_out = Path(args.vision_out_dir)
    hidden = _find_output_bin(
        vision_out,
        ("last_hidden_state.bin", "0.bin", "output_0.bin"),
        size=VISION_OUT_BYTES,
        label="vision_features",
    )
    _msame_write_inputs(
        Path(args.out_dir),
        [(0, hidden)],
        expected_sizes={0: VISION_OUT_BYTES},
    )


def cmd_prepare_preblock_input(args: argparse.Namespace) -> None:
    state = Path(args.state_dir or args.dump_preblock_dir)
    static = _resolve_static_preblock_dir(args)
    mm_proj_out = Path(args.mm_proj_out_dir)
    image_embeds = _find_output_bin(
        mm_proj_out,
        ("hidden_states.bin", "0.bin", "output_0.bin"),
        size=MM_PROJ_OUT_BYTES,
        label="image_embeds",
    )
    _msame_write_inputs(
        Path(args.out_dir),
        [
            (0, state / "input_ids.bin"),
            (1, image_embeds),
            (2, state / "attention_mask.bin"),
            (3, static / "position_ids.bin"),
        ],
        expected_sizes={
            0: MAX_SEQ_LEN * 4,
            1: MM_PROJ_OUT_BYTES,
            2: MAX_SEQ_LEN * 4,
            3: POSITION_IDS_BYTES,
        },
    )
    print(
        "preblock msame inputs (4): input_ids, image_embeds, attention_mask, position_ids"
    )


def cmd_prepare_block_input(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pre_out = Path(args.pre_out_dir)
    block_idx = int(args.block_idx)

    if block_idx == 1:
        hidden = _find_output_bin(
            pre_out,
            ("inputs_embeds_out.bin", "0.bin", "output_0.bin"),
            size=BLOCK_HIDDEN_BYTES,
            label="inputs_embeds_out",
        )
    else:
        hidden = _find_output_bin(
            Path(args.prev_block_out_dir),
            ("hidden_states_out.bin", "0.bin", "output_0.bin"),
            size=BLOCK_HIDDEN_BYTES,
            label="hidden_states_out",
        )

    mask = _find_output_bin(
        pre_out,
        ("attention_mask_out.bin", "1.bin", "output_1.bin"),
        size=MASK_BYTES,
        label="attention_mask_out",
    )
    cos = _find_output_bin(
        pre_out,
        ("cos.bin", "2.bin", "output_2.bin"),
        size=COS_SIN_BYTES,
        label="cos",
    )
    sin = _find_output_bin(
        pre_out,
        ("sin.bin", "3.bin", "output_3.bin"),
        size=COS_SIN_BYTES,
        label="sin",
    )

    _msame_write_inputs(
        out_dir,
        [(0, hidden), (1, mask), (2, cos), (3, sin)],
        expected_sizes={
            0: BLOCK_HIDDEN_BYTES,
            1: MASK_BYTES,
            2: COS_SIN_BYTES,
            3: COS_SIN_BYTES,
        },
    )


def cmd_prepare_lm_head_input(args: argparse.Namespace) -> None:
    b3_out = Path(args.b3_out_dir)
    hidden_path = _find_bin_by_size(b3_out, BLOCK_HIDDEN_BYTES, "b3_hidden")
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
    _msame_write_inputs(Path(args.out_dir), [(0, tmp)])
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

    input_ids = _read_i32_array(state_dir / "input_ids.bin", MAX_SEQ_LEN)
    attn = _read_i32_array(state_dir / "attention_mask.bin", MAX_SEQ_LEN)
    input_ids[cur_len] = next_id
    attn[cur_len] = 1
    _write_i32_array(state_dir / "input_ids.bin", input_ids)
    _write_i32_array(state_dir / "attention_mask.bin", attn)
    (state_dir / "last_token.txt").write_text(f"{next_id}\n", encoding="utf-8")
    print(f"step={args.step} cur_len={cur_len} next_token={next_id}")


def cmd_init_cur_len(args: argparse.Namespace) -> None:
    attn = _read_i32_array(Path(args.state_dir) / "attention_mask.bin", MAX_SEQ_LEN)
    cur_len = sum(attn)
    Path(args.out_file).write_text(str(cur_len), encoding="utf-8")
    print(cur_len)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync-state", help="copy dump preblock → state (prefill init)")
    s.add_argument("--dump-preblock-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.set_defaults(func=cmd_sync_state)

    s = sub.add_parser("sync-preblock-state", help="alias of sync-state")
    s.add_argument("--dump-preblock-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.set_defaults(func=cmd_sync_state)

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
    s.add_argument("--state-dir", default="")
    s.add_argument("--dump-preblock-dir", default="")
    s.add_argument("--static-preblock-dir", default="")
    s.add_argument("--mm-proj-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_preblock_input)

    s = sub.add_parser("prepare-block-input")
    s.add_argument("--pre-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--block-idx", type=int, required=True)
    s.add_argument("--prev-block-out-dir", default="")
    s.set_defaults(func=cmd_prepare_block_input)

    s = sub.add_parser("prepare-lm-head-input")
    s.add_argument("--b3-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.set_defaults(func=cmd_prepare_lm_head_input)

    s = sub.add_parser("update-decode-state")
    s.add_argument("--lm-head-out-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.add_argument("--step", type=int, default=0)
    s.set_defaults(func=cmd_update_decode_state)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
