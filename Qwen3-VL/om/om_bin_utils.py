#!/usr/bin/env python3
"""
Qwen3-VL OM pipeline bin helpers (stdlib only, board-friendly).

Aligned with export_config profiles:
  vision_448 -> llm_preblock -> b1..b3 -> lm_head  (default 448_512)

Only frontend / decode-state bins are prepared here; intermediate tensors
come from upstream OM outputs.
"""

from __future__ import annotations

import argparse
import mmap
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QWEN3_EXPORT_PROFILE", "448_512")

OM_DIR = Path(__file__).resolve().parent
REPO_ROOT = OM_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_config import ExportProfile, get_export_profile  # noqa: E402

HIDDEN_DIM = 2048
VOCAB_SIZE = 151936
EOS_TOKEN_ID = 151645  # <|im_end|>, Qwen3-VL-2B-Instruct
PIXEL_FEATURE_DIM = 1536

PROFILE: ExportProfile = get_export_profile()
MAX_SEQ_LEN = PROFILE.max_seq_len
NUM_IMAGE_TOKENS = PROFILE.num_image_tokens
VISION_OM_LABEL = PROFILE.vision_onnx_name.replace(".onnx", "")

PIXEL_BYTES = PROFILE.num_vision_patches * PIXEL_FEATURE_DIM * 2
IMAGE_EMBEDS_BYTES = NUM_IMAGE_TOKENS * HIDDEN_DIM * 2
BLOCK_HIDDEN_BYTES = 1 * MAX_SEQ_LEN * HIDDEN_DIM * 2
LM_HEAD_IN_BYTES = 1 * 1 * HIDDEN_DIM * 2
LM_HEAD_LOGITS_BYTES = 1 * 1 * VOCAB_SIZE * 2
DEEPSTACK_BYTES = NUM_IMAGE_TOKENS * HIDDEN_DIM * 2
POSITION_IDS_BYTES = 3 * 1 * MAX_SEQ_LEN * 4
COS_SIN_BYTES = 1 * MAX_SEQ_LEN * 128 * 2
MASK_BYTES = 1 * 1 * MAX_SEQ_LEN * MAX_SEQ_LEN * 2
PREBLOCK_OUT_BYTES = {
    "inputs_embeds_out": BLOCK_HIDDEN_BYTES,
    "attention_mask_out": MASK_BYTES,
    "cos": COS_SIN_BYTES,
    "sin": COS_SIN_BYTES,
}
VISION_OUT = (
    ("merged_hidden_states", 0, IMAGE_EMBEDS_BYTES),
    ("deepstack_feat_5", 1, DEEPSTACK_BYTES),
    ("deepstack_feat_11", 2, DEEPSTACK_BYTES),
    ("deepstack_feat_17", 3, DEEPSTACK_BYTES),
)


def configure_profile(profile_name: str | None) -> ExportProfile:
    global PROFILE, MAX_SEQ_LEN, NUM_IMAGE_TOKENS, VISION_OM_LABEL
    global PIXEL_BYTES, IMAGE_EMBEDS_BYTES, BLOCK_HIDDEN_BYTES
    global DEEPSTACK_BYTES, POSITION_IDS_BYTES, COS_SIN_BYTES, MASK_BYTES
    global PREBLOCK_OUT_BYTES, VISION_OUT

    profile = get_export_profile(profile_name)
    PROFILE = profile
    MAX_SEQ_LEN = profile.max_seq_len
    NUM_IMAGE_TOKENS = profile.num_image_tokens
    VISION_OM_LABEL = profile.vision_onnx_name.replace(".onnx", "")

    PIXEL_BYTES = profile.num_vision_patches * PIXEL_FEATURE_DIM * 2
    IMAGE_EMBEDS_BYTES = NUM_IMAGE_TOKENS * HIDDEN_DIM * 2
    BLOCK_HIDDEN_BYTES = 1 * MAX_SEQ_LEN * HIDDEN_DIM * 2
    DEEPSTACK_BYTES = NUM_IMAGE_TOKENS * HIDDEN_DIM * 2
    POSITION_IDS_BYTES = 3 * 1 * MAX_SEQ_LEN * 4
    COS_SIN_BYTES = 1 * MAX_SEQ_LEN * 128 * 2
    MASK_BYTES = 1 * 1 * MAX_SEQ_LEN * MAX_SEQ_LEN * 2
    PREBLOCK_OUT_BYTES = {
        "inputs_embeds_out": BLOCK_HIDDEN_BYTES,
        "attention_mask_out": MASK_BYTES,
        "cos": COS_SIN_BYTES,
        "sin": COS_SIN_BYTES,
    }
    VISION_OUT = (
        ("merged_hidden_states", 0, IMAGE_EMBEDS_BYTES),
        ("deepstack_feat_5", 1, DEEPSTACK_BYTES),
        ("deepstack_feat_11", 2, DEEPSTACK_BYTES),
        ("deepstack_feat_17", 3, DEEPSTACK_BYTES),
    )
    return profile
STATE_ALIEN_FILES = (
    "per_layer_inputs.bin",
    "accepted_tokens.log",
    "preblock_position_ids.bin",
    "position_ids.bin",
)


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


def _find_output_index(
    out_dir: Path,
    index: int,
    size: int,
    label: str,
) -> Path:
    """优先 output_{index}.bin（msame 默认命名），再按 size 兜底。"""
    root = _resolve_msame_out_dir(out_dir)
    preferred = (
        f"output_{index}.bin",
        f"{index}.bin",
    )
    for name in preferred:
        p = root / name
        if p.is_file() and p.stat().st_size == size:
            return p
        hits = [p for p in root.rglob(f"*{name}") if p.stat().st_size == size]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(
                f"Ambiguous {label} *{name} under {out_dir}: {[str(h) for h in hits]}"
            )
    return _find_bin_by_size(out_dir, size, label)


def _find_vision_out(vision_out: Path, name: str, index: int, size: int) -> Path:
    root = _resolve_msame_out_dir(vision_out)
    for cand in (f"{name}.bin", f"output_{index}.bin", f"{index}.bin"):
        p = root / cand
        if p.is_file() and p.stat().st_size == size:
            return p
        hits = [p for p in root.rglob(f"*{cand}") if p.stat().st_size == size]
        if len(hits) == 1:
            return hits[0]
    return _find_output_index(vision_out, index, size, name)


def _check_vision_outputs(vision_out: Path) -> None:
    missing = []
    for name, index, size in VISION_OUT:
        try:
            _find_vision_out(vision_out, name, index, size)
        except FileNotFoundError:
            missing.append(f"{name}(output_{index}, {size}B)")
    if missing:
        raise FileNotFoundError(
            f"vision OM outputs incomplete under {vision_out}: missing {missing}"
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


def cmd_sync_state(args: argparse.Namespace) -> None:
    """Reset decode state from dump/llm_preblock."""
    dump = Path(args.dump_preblock_dir)
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    for alien in STATE_ALIEN_FILES:
        _safe_unlink(state / alien)
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


def cmd_prepare_preblock_input(args: argparse.Namespace) -> None:
    state = Path(args.state_dir or args.dump_preblock_dir)
    static = _resolve_static_preblock_dir(args)
    vision_out = Path(args.vision_out_dir)
    _check_vision_outputs(vision_out)
    image_embeds = _find_vision_out(
        vision_out, "merged_hidden_states", 0, IMAGE_EMBEDS_BYTES
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
            1: IMAGE_EMBEDS_BYTES,
            2: MAX_SEQ_LEN * 4,
            3: POSITION_IDS_BYTES,
        },
    )


def _resolve_cos_sin(pre_out: Path, static_dir: Path) -> tuple[Path, Path]:
    """优先 preblock OM 输出；若 cos/sin 被折叠则回退 dump 静态 bin。"""
    try:
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
        return cos, sin
    except FileNotFoundError:
        cos_p = static_dir / "cos.bin"
        sin_p = static_dir / "sin.bin"
        if cos_p.is_file() and sin_p.is_file():
            print(
                f"  [fallback] cos/sin from static dump {static_dir} "
                "(preblock OM folded rotary)"
            )
            return cos_p, sin_p
        raise


def cmd_prepare_block_input(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pre_out = Path(args.pre_out_dir)
    static = _resolve_static_preblock_dir(args)
    block_idx = int(args.block_idx)
    vision_out = Path(args.vision_out_dir)

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

    mask = _find_output_index(
        pre_out, 1, MASK_BYTES, "attention_mask_out"
    )
    cos, sin = _resolve_cos_sin(pre_out, static)

    indexed: list[tuple[int, Path]] = [
        (0, hidden),
        (1, mask),
        (2, cos),
        (3, sin),
    ]

    if block_idx == 1:
        ds0 = _find_vision_out(vision_out, "deepstack_feat_5", 1, DEEPSTACK_BYTES)
        indexed.append((4, ds0))
    elif block_idx == 2:
        ds1 = _find_vision_out(vision_out, "deepstack_feat_11", 2, DEEPSTACK_BYTES)
        ds2 = _find_vision_out(vision_out, "deepstack_feat_17", 3, DEEPSTACK_BYTES)
        indexed.extend([(4, ds1), (5, ds2)])

    _msame_write_inputs(out_dir, indexed)


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
    hit_eos = next_id == EOS_TOKEN_ID
    print(f"step={args.step} cur_len={cur_len} next_token={next_id} eos={int(hit_eos)}")


def cmd_init_cur_len(args: argparse.Namespace) -> None:
    attn = _read_i32_array(Path(args.state_dir) / "attention_mask.bin", MAX_SEQ_LEN)
    cur_len = sum(attn)
    Path(args.out_file).write_text(str(cur_len), encoding="utf-8")
    print(cur_len)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--profile",
        choices=("256_256", "448_512"),
        default=os.environ.get("QWEN3_EXPORT_PROFILE", "448_512"),
        help="OM layout profile (default: 448_512)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync-state")
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

    s = sub.add_parser("prepare-preblock-input")
    s.add_argument("--state-dir", default="")
    s.add_argument("--dump-preblock-dir", default="")
    s.add_argument(
        "--static-preblock-dir",
        default="",
        help="dump/llm_preblock，含 position_ids.bin",
    )
    s.add_argument("--vision-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_preblock_input)

    s = sub.add_parser("prepare-block-input")
    s.add_argument("--pre-out-dir", required=True)
    s.add_argument("--vision-out-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--block-idx", type=int, required=True)
    s.add_argument("--prev-block-out-dir", default="")
    s.add_argument("--static-preblock-dir", default="")
    s.add_argument("--dump-preblock-dir", default="")
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
    configure_profile(args.profile)
    args.func(args)


if __name__ == "__main__":
    main()
