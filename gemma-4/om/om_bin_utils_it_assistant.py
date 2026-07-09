#!/usr/bin/env python3
"""
Assistant OM bin helpers (stdlib only). Complements om_bin_utils.py — do not edit that file.

Aligned with test/onnx_torch_test.py assistant chain:
  assistant OM (8 inputs) -> projected_state + logits
  main verify: llm chain + lm_head range -> accept draft tokens
"""

from __future__ import annotations

import argparse
import mmap
import struct
import sys
import tempfile
from pathlib import Path

# Reuse main-model helpers without modifying om_bin_utils.py
_OM_DIR = Path(__file__).resolve().parent
if str(_OM_DIR) not in sys.path:
    sys.path.insert(0, str(_OM_DIR))

from om_bin_utils import (  # noqa: E402
    BLOCK_HIDDEN_BYTES,
    HIDDEN_DIM,
    LM_HEAD_IN_BYTES,
    LM_HEAD_LOGITS_BYTES,
    SEQ_LEN,
    VOCAB_SIZE,
    _argmax_fp16_logits,
    _find_bin_by_size,
    _find_output_bin,
    _msame_write_inputs,
    _patch_ple_column,
    _read_i32_array,
    _write_i32_array,
)

ASSISTANT_PROJECTED_BYTES = 1 * 1 * HIDDEN_DIM * 2
ASSISTANT_KV_BYTES = {
    "full_k": 1 * 1 * SEQ_LEN * SEQ_LEN * 2,
    "full_v": 1 * 1 * SEQ_LEN * SEQ_LEN * 2,
    "slide_k": 1 * 1 * SEQ_LEN * 256 * 2,
    "slide_v": 1 * 1 * SEQ_LEN * 256 * 2,
}

_DUMP_ASSISTANT_ORDER = (
    "last_token_id",
    "last_hidden",
    "attention_mask",
    "position_ids",
    "full_k",
    "full_v",
    "slide_k",
    "slide_v",
)


def _write_i32_scalar(path: Path, value: int) -> None:
    path.write_bytes(struct.pack("<i", int(value)))


def _read_i32_scalar(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError(f"need 4 bytes, got {len(data)}: {path}")
    return struct.unpack("<i", data[:4])[0]


def _slice_b7_last_hidden(b7_out: Path, cur_len: int) -> Path:
    pos = int(cur_len) - 1
    if pos < 0:
        raise ValueError(f"cur_len must be >= 1, got {cur_len}")
    hidden_path = _find_bin_by_size(b7_out, BLOCK_HIDDEN_BYTES, "b7_hidden")
    offset = pos * HIDDEN_DIM * 2
    tmp = Path(tempfile.mkdtemp()) / "last_hidden.bin"
    with open(hidden_path, "rb") as fin, open(tmp, "wb") as fout:
        fin.seek(offset)
        chunk = fin.read(LM_HEAD_IN_BYTES)
        if len(chunk) != LM_HEAD_IN_BYTES:
            raise ValueError(f"short read b7 hidden pos={pos}: {len(chunk)}")
        fout.write(chunk)
    return tmp


def _resolve_b3_kv(b3_out: Path) -> dict[str, Path]:
    return {
        "full_k": _find_output_bin(b3_out, ("out_full_k.bin", "1.bin", "output_1.bin")),
        "full_v": _find_output_bin(b3_out, ("out_full_v.bin", "2.bin", "output_2.bin")),
        "slide_k": _find_output_bin(b3_out, ("out_slide_k.bin", "3.bin", "output_3.bin")),
        "slide_v": _find_output_bin(b3_out, ("out_slide_v.bin", "4.bin", "output_4.bin")),
    }


def cmd_prepare_assistant_input_dump(args: argparse.Namespace) -> None:
    dump = Path(args.dump_assistant_dir)
    indexed = []
    for i, name in enumerate(_DUMP_ASSISTANT_ORDER):
        src = dump / f"{name}.bin"
        if not src.is_file():
            raise FileNotFoundError(src)
        indexed.append((i, src))
    _msame_write_inputs(Path(args.out_dir), indexed)


def cmd_prepare_assistant_input_chain(args: argparse.Namespace) -> None:
    """Build assistant msame inputs from main prefill/decode chain outputs."""
    state = Path(args.state_dir)
    b7_out = Path(args.b7_out_dir)
    b3_out = Path(args.b3_out_dir)
    cur_len = int(args.cur_len)
    pos = cur_len - 1

    input_ids = _read_i32_array(state / "input_ids.bin", SEQ_LEN)
    attn = _read_i32_array(state / "attention_mask.bin", SEQ_LEN)

    tmp_dir = Path(tempfile.mkdtemp())
    last_token = tmp_dir / "last_token_id.bin"
    position_ids = tmp_dir / "position_ids.bin"
    attn_out = tmp_dir / "attention_mask.bin"

    _write_i32_scalar(last_token, input_ids[pos])
    _write_i32_scalar(position_ids, pos)
    _write_i32_array(attn_out, attn)

    last_hidden = _slice_b7_last_hidden(b7_out, cur_len)
    kv = _resolve_b3_kv(b3_out)

    indexed = [
        (0, last_token),
        (1, last_hidden),
        (2, attn_out),
        (3, position_ids),
        (4, kv["full_k"]),
        (5, kv["full_v"]),
        (6, kv["slide_k"]),
        (7, kv["slide_v"]),
    ]
    _msame_write_inputs(Path(args.out_dir), indexed)
    print(f"assistant chain input: cur_len={cur_len} pos={pos}")


def cmd_prepare_assistant_draft_step(args: argparse.Namespace) -> None:
    """Next assistant draft step: projected_state -> last_hidden, logits argmax -> token."""
    prev_out = Path(args.prev_assistant_out_dir)
    state = Path(args.state_dir)
    b3_out = Path(args.b3_out_dir)
    pos = int(args.pos)

    projected = _find_output_bin(
        prev_out,
        ("projected_state.bin", "0.bin", "output_0.bin"),
        size=ASSISTANT_PROJECTED_BYTES,
        label="assistant_projected_state",
    )
    logits_path = _resolve_assistant_logits(prev_out)
    with open(logits_path, "rb") as f:
        log_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        next_id = _argmax_fp16_logits(log_mm)

    attn = _read_i32_array(state / "attention_mask.bin", SEQ_LEN)
    if pos < SEQ_LEN:
        attn[pos] = 1

    tmp_dir = Path(tempfile.mkdtemp())
    last_token = tmp_dir / "last_token_id.bin"
    position_ids = tmp_dir / "position_ids.bin"
    attn_out = tmp_dir / "attention_mask.bin"
    _write_i32_scalar(last_token, next_id)
    _write_i32_scalar(position_ids, pos)
    _write_i32_array(attn_out, attn)

    kv = _resolve_b3_kv(b3_out)
    indexed = [
        (0, last_token),
        (1, projected),
        (2, attn_out),
        (3, position_ids),
        (4, kv["full_k"]),
        (5, kv["full_v"]),
        (6, kv["slide_k"]),
        (7, kv["slide_v"]),
    ]
    _msame_write_inputs(Path(args.out_dir), indexed)
    print(f"assistant draft step: pos={pos} draft_token={next_id}")


def _resolve_assistant_logits(out_dir: Path) -> Path:
    """assistant OM outputs: 0=projected_state, 1=logits, 2=hidden_states_out."""
    try:
        return _find_bin_by_size(out_dir, LM_HEAD_LOGITS_BYTES, "assistant_logits")
    except FileNotFoundError:
        return _find_output_bin(
            out_dir,
            ("logits.bin", "1.bin", "output_1.bin"),
            size=LM_HEAD_LOGITS_BYTES,
            label="assistant_logits",
        )


def cmd_parse_assistant_argmax(args: argparse.Namespace) -> None:
    out_dir = Path(args.assistant_out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"assistant output dir missing: {out_dir}")
    try:
        logits_path = _resolve_assistant_logits(out_dir)
    except FileNotFoundError as e:
        bins = list(out_dir.rglob("*.bin")) if out_dir.is_dir() else []
        hint = (
            f" (OM likely failed to load; check OM_ASSISTANT path and msame log)"
            if not bins
            else f" (found: {[str(p.name) for p in bins[:8]]})"
        )
        raise FileNotFoundError(f"{e}{hint}") from e
    with open(logits_path, "rb") as f:
        log_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        token_id = _argmax_fp16_logits(log_mm)
    Path(args.out_file).write_text(f"{token_id}\n", encoding="utf-8")
    print(f"assistant argmax token={token_id}")


def cmd_parse_lm_argmax(args: argparse.Namespace) -> None:
    lm_out = Path(args.lm_head_out_dir)
    try:
        logits_path = _find_bin_by_size(lm_out, LM_HEAD_LOGITS_BYTES, "lm_head_logits")
    except FileNotFoundError:
        logits_path = _find_output_bin(lm_out, ("logits.bin", "0.bin", "output_0.bin"))
    with open(logits_path, "rb") as f:
        log_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        token_id = _argmax_fp16_logits(log_mm)
    Path(args.out_file).write_text(f"{token_id}\n", encoding="utf-8")
    print(f"lm_head argmax token={token_id}")


def cmd_prepare_speculative_verify_state(args: argparse.Namespace) -> None:
    """Copy state and inject assistant candidate tokens starting at cur_len."""
    src_state = Path(args.state_dir)
    dst_state = Path(args.verify_state_dir)
    dst_state.mkdir(parents=True, exist_ok=True)

    cur_len = int(args.cur_len)
    candidates = [int(x.strip()) for x in Path(args.candidates_file).read_text().split() if x.strip()]
    if not candidates:
        raise ValueError("empty candidates file")

    input_ids = _read_i32_array(src_state / "input_ids.bin", SEQ_LEN)
    attn = _read_i32_array(src_state / "attention_mask.bin", SEQ_LEN)
    ple_src = src_state / "per_layer_inputs.bin"
    ple_dst = dst_state / "per_layer_inputs.bin"
    shutil_copy = __import__("shutil").copy2
    shutil_copy(ple_src, ple_dst)

    ple_table = Path(args.ple_table) if args.ple_table else None
    pad_id = int(args.pad_token_id)

    for i, tok in enumerate(candidates):
        pos = cur_len + i
        if pos >= SEQ_LEN:
            break
        input_ids[pos] = tok
        attn[pos] = 1

    _write_i32_array(dst_state / "input_ids.bin", input_ids)
    _write_i32_array(dst_state / "attention_mask.bin", attn)

    if ple_table and ple_table.is_file():
        for i, tok in enumerate(candidates):
            pos = cur_len + i
            if pos >= SEQ_LEN:
                break
            _patch_ple_column(dst_state, ple_table, pos, pad_id, token_id=tok)
    print(f"verify state: cur_len={cur_len} injected {len(candidates)} candidates -> {dst_state}")


def cmd_process_speculative_accept(args: argparse.Namespace) -> None:
    """Compare assistant candidates vs main lm_head preds; commit accepted prefix to state."""
    state = Path(args.state_dir)
    cur_len = int(args.cur_len)
    cand_tokens = [int(x.strip()) for x in Path(args.candidates_file).read_text().split() if x.strip()]
    main_preds = [int(x.strip()) for x in Path(args.main_preds_file).read_text().split() if x.strip()]

    if not cand_tokens:
        Path(args.accept_count_file).write_text("0\n", encoding="utf-8")
        print("no candidates, accept_count=0")
        return

    n = min(len(cand_tokens), len(main_preds))
    accept = 0
    for i in range(n):
        if cand_tokens[i] != main_preds[i]:
            break
        accept += 1
    # Always take at least main_pred[0] (target model token at cur_len-1 anchor)
    accept_count = accept + 1 if n > 0 else 0
    accept_count = min(accept_count, n)
    accepted = main_preds[:accept_count]

    input_ids = _read_i32_array(state / "input_ids.bin", SEQ_LEN)
    attn = _read_i32_array(state / "attention_mask.bin", SEQ_LEN)
    ple_table = Path(args.ple_table) if args.ple_table else None
    pad_id = int(args.pad_token_id)

    for i, tok in enumerate(accepted):
        pos = cur_len + i
        input_ids[pos] = tok
        attn[pos] = 1
        if ple_table and ple_table.is_file():
            _patch_ple_column(state, ple_table, pos, pad_id, token_id=tok)

    _write_i32_array(state / "input_ids.bin", input_ids)
    _write_i32_array(state / "attention_mask.bin", attn)
    if accepted:
        (state / "last_token.txt").write_text(f"{accepted[-1]}\n", encoding="utf-8")

    log_path = state / "accepted_tokens.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"step={args.step} cur_len={cur_len} accepted={accepted}\n")

    Path(args.accept_count_file).write_text(f"{accept_count}\n", encoding="utf-8")
    hit_eos = any(tok == 1 for tok in accepted)  # <eos>
    print(f"step={args.step} accepted {accept_count} tokens: {accepted} eos={int(hit_eos)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Assistant OM bin helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-assistant-input-dump")
    s.add_argument("--dump-assistant-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_assistant_input_dump)

    s = sub.add_parser("prepare-assistant-input-chain")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--b7-out-dir", required=True)
    s.add_argument("--b3-out-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_assistant_input_chain)

    s = sub.add_parser("prepare-assistant-draft-step")
    s.add_argument("--prev-assistant-out-dir", required=True)
    s.add_argument("--state-dir", required=True)
    s.add_argument("--b3-out-dir", required=True)
    s.add_argument("--pos", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_prepare_assistant_draft_step)

    s = sub.add_parser("parse-assistant-argmax")
    s.add_argument("--assistant-out-dir", required=True)
    s.add_argument("--out-file", required=True)
    s.set_defaults(func=cmd_parse_assistant_argmax)

    s = sub.add_parser("parse-lm-argmax")
    s.add_argument("--lm-head-out-dir", required=True)
    s.add_argument("--out-file", required=True)
    s.set_defaults(func=cmd_parse_lm_argmax)

    s = sub.add_parser("prepare-speculative-verify-state")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--verify-state-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.add_argument("--candidates-file", required=True)
    s.add_argument("--ple-table", default="")
    s.add_argument("--pad-token-id", type=int, default=0)
    s.set_defaults(func=cmd_prepare_speculative_verify_state)

    s = sub.add_parser("process-speculative-accept")
    s.add_argument("--state-dir", required=True)
    s.add_argument("--cur-len", type=int, required=True)
    s.add_argument("--candidates-file", required=True)
    s.add_argument("--main-preds-file", required=True)
    s.add_argument("--accept-count-file", required=True)
    s.add_argument("--step", type=int, default=0)
    s.add_argument("--ple-table", default="")
    s.add_argument("--pad-token-id", type=int, default=0)
    s.set_defaults(func=cmd_process_speculative_accept)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
