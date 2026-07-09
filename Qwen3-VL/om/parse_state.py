#!/usr/bin/env python3
"""
[本地机] Parse MDC om_output/ → generated text (stdlib + JSON vocab only).

Reads final_input_ids.bin / final_cur_len.txt from om_output/.
Token decode uses tokenizer.json + tokenizer_config.json (no tokenizers package).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

OM_DIR = Path(__file__).resolve().parent
REPO_ROOT = OM_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QWEN3_EXPORT_PROFILE", "448_512")

from export_config import get_export_profile  # noqa: E402

DEFAULT_OUTPUT = OM_DIR / "om_output"
DEFAULT_STATE = OM_DIR / "om_output" / "state"
DEFAULT_DUMP = OM_DIR / "dump"
DEFAULT_PROMPT_BIN = OM_DIR / "prompt_bin"
DEFAULT_BATCH = OM_DIR / "batch"
DEFAULT_MODEL = REPO_ROOT / "Qwen3-VL-2B-Instruct"

DEFAULT_SEQ_LEN = get_export_profile().max_seq_len
BATCH_SEP = "=" * 72


def print_batch_block(stem: str, body: str) -> None:
    print(BATCH_SEP)
    print(BATCH_SEP)
    print(f"[{stem}]")
    print(body)


def infer_seq_len(path: Path, *, fallback: int = DEFAULT_SEQ_LEN) -> int:
    nbytes = path.stat().st_size
    if nbytes % 4:
        raise ValueError(f"{path}: size {nbytes} is not a multiple of 4")
    return nbytes // 4 if nbytes else fallback


def read_i32_bin(path: Path, n: int | None = None) -> np.ndarray:
    data = path.read_bytes()
    count = infer_seq_len(path) if n is None else n
    if len(data) < count * 4:
        raise ValueError(f"{path}: need {count * 4} bytes, got {len(data)}")
    return np.array(struct.unpack(f"<{count}i", data[: count * 4]), dtype=np.int32)


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=4)
def load_vocab(model_dir: str) -> tuple[dict[int, str], set[int]]:
    root = Path(model_dir)
    tok_path = root / "tokenizer.json"
    cfg_path = root / "tokenizer_config.json"
    if not tok_path.is_file():
        raise FileNotFoundError(tok_path)

    data = json.loads(tok_path.read_text(encoding="utf-8"))
    id_to_token: dict[int, str] = {
        int(idx): token for token, idx in data["model"]["vocab"].items()
    }
    for entry in data.get("added_tokens", []):
        id_to_token[int(entry["id"])] = entry["content"]

    special_ids: set[int] = set()
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for tid, meta in cfg.get("added_tokens_decoder", {}).items():
            if meta.get("special"):
                special_ids.add(int(tid))

    return id_to_token, special_ids


def decode_ids(
    model_dir: Path,
    ids: list[int],
    *,
    skip_special: bool,
) -> str:
    id_to_token, special_ids = load_vocab(str(model_dir.resolve()))
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

    pieces: list[str] = []
    for token_id in ids:
        if skip_special and token_id in special_ids:
            continue
        piece = id_to_token.get(token_id)
        if piece is None:
            continue
        pieces.append(piece)

    text = "".join(pieces)
    try:
        return bytearray(byte_decoder[c] for c in text).decode("utf-8", errors="replace")
    except KeyError:
        return text.replace("Ġ", " ").replace("Ċ", "\n")


def cur_len_from_attention(attn: np.ndarray) -> int:
    return int(attn.sum())


def prefill_len_from_dump(dump_dir: Path) -> int:
    for meta_path in (
        dump_dir / "meta.json",
        dump_dir / "pipeline.json",
    ):
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "seq_len" in meta:
                return int(meta["seq_len"])

    for attn_path in (
        dump_dir / "attention_mask.bin",
        dump_dir / "llm_preblock" / "attention_mask.bin",
    ):
        if attn_path.is_file():
            return cur_len_from_attention(read_i32_bin(attn_path))

    raise FileNotFoundError(f"no seq_len in {dump_dir} (need meta.json or attention_mask.bin)")


def default_output_dir() -> Path:
    raw = os.environ.get("OUTPUT_ROOT", "")
    return Path(raw) if raw else DEFAULT_OUTPUT


def _has_preblock(root: Path) -> bool:
    return (root / "attention_mask.bin").is_file() or (
        root / "llm_preblock" / "attention_mask.bin"
    ).is_file()


def resolve_prefill_root(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p.resolve() if p.is_dir() else None
    env = os.environ.get("DUMP_ROOT", "")
    if env:
        p = Path(env)
        if p.is_dir() and _has_preblock(p):
            return p.resolve()
    if _has_preblock(DEFAULT_DUMP):
        return DEFAULT_DUMP
    if _has_preblock(DEFAULT_PROMPT_BIN):
        return DEFAULT_PROMPT_BIN
    return DEFAULT_DUMP if DEFAULT_DUMP.is_dir() else None


def resolve_batch_root(raw: Path) -> Path:
    if raw.is_absolute():
        return raw.resolve()
    om = OM_DIR / raw
    repo = REPO_ROOT / raw
    if om.is_dir():
        return om.resolve()
    if repo.is_dir():
        return repo.resolve()
    return om.resolve()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    explicit_output = bool(args.output_dir)
    output_dir = Path(args.output_dir).resolve() if explicit_output else default_output_dir().resolve()

    if args.state_dir:
        state_dir = Path(args.state_dir)
    elif os.environ.get("STATE_DIR"):
        state_dir = Path(os.environ["STATE_DIR"])
    else:
        meta_path = output_dir / "final.meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            state_dir = Path(meta.get("state_dir", output_dir / "state"))
        elif explicit_output:
            state_dir = output_dir / "state"
        else:
            state_dir = DEFAULT_STATE

    dump_dir = resolve_prefill_root(args.dump_dir)
    return state_dir, output_dir, dump_dir


def load_arrays(output_dir: Path, state_dir: Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    ids_path = next(
        (p for p in (output_dir / "final_input_ids.bin", state_dir / "input_ids.bin") if p.is_file()),
        None,
    )
    if ids_path is None:
        raise FileNotFoundError(
            f"no input_ids in {output_dir} or {state_dir} "
            "(for batch items, scp batch/<stem>/om_output from MDC before parse)"
        )

    attn_path = next(
        (
            p
            for p in (output_dir / "final_attention_mask.bin", state_dir / "attention_mask.bin")
            if p.is_file()
        ),
        None,
    )

    input_ids = read_i32_bin(ids_path)
    attn = read_i32_bin(attn_path) if attn_path is not None else None

    cur_len: int | None = None
    cur_file = output_dir / "final_cur_len.txt"
    if cur_file.is_file():
        cur_len = int(cur_file.read_text(encoding="utf-8").strip())
    if cur_len is None:
        cur_len = cur_len_from_attention(attn) if attn is not None else int(np.count_nonzero(input_ids))

    return input_ids, attn, cur_len


def _is_batch_item(item_dir: Path) -> bool:
    if (item_dir / "vision_bin" / "pixel_values.bin").is_file():
        return True
    if (item_dir / "dump" / "vision" / "pixel_values.bin").is_file():
        return True
    return (item_dir / "om_output" / "final_input_ids.bin").is_file() or (
        item_dir / "om_output" / "state" / "input_ids.bin"
    ).is_file()


def _resolve_item_prompt_bin(item_dir: Path, shared_prompt: Path) -> Path:
    for cand in (item_dir / "prompt_bin", shared_prompt, item_dir / "dump" / "llm_preblock"):
        if (cand / "meta.json").is_file() or (cand / "attention_mask.bin").is_file():
            return cand
    return shared_prompt


def run_parse(args: argparse.Namespace) -> str | None:
    model_dir = Path(args.model_dir)
    state_dir, output_dir, dump_dir = resolve_paths(args)

    input_ids, _, cur_len = load_arrays(output_dir, state_dir)

    input_len = args.input_len
    if input_len is None and dump_dir is not None:
        input_len = prefill_len_from_dump(dump_dir)
    if input_len is None:
        print(
            "error: need prefill length (--input-len N or --dump-dir prompt_bin)",
            file=sys.stderr,
        )
        sys.exit(1)

    gen_ids = input_ids[:cur_len][input_len:cur_len].tolist()
    gen_text = decode_ids(model_dir, gen_ids, skip_special=args.skip_special_tokens)

    if args.response_out:
        out = Path(args.response_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gen_text.strip() + "\n", encoding="utf-8")
    elif not getattr(args, "no_stdout", False):
        print(gen_text)

    return gen_text


def run_parse_batch(args: argparse.Namespace) -> None:
    batch_root = resolve_batch_root(Path(args.batch_root))
    if not batch_root.is_dir():
        raise SystemExit(f"ERROR: batch root not found: {batch_root}")

    shared_prompt = DEFAULT_PROMPT_BIN
    if args.dump_dir:
        shared_prompt = Path(args.dump_dir).resolve()

    item_dirs = sorted(p for p in batch_root.iterdir() if p.is_dir() and _is_batch_item(p))
    if args.stem:
        item_dirs = [p for p in item_dirs if p.name == args.stem]
        if not item_dirs:
            raise SystemExit(f"ERROR: no item '{args.stem}' under {batch_root}")

    if not item_dirs:
        raise SystemExit(f"ERROR: no runnable items under {batch_root}")

    summary = batch_root / "summary_parse.tsv"
    summary.write_text("stem\tstatus\ttext\n", encoding="utf-8")

    for item_dir in item_dirs:
        stem = item_dir.name
        prompt_bin = _resolve_item_prompt_bin(item_dir, shared_prompt)
        item_args = argparse.Namespace(**vars(args))
        item_args.output_dir = str(item_dir / "om_output")
        item_args.dump_dir = str(prompt_bin)
        item_args.response_out = str(item_dir / "response.txt") if args.write_response else ""
        item_args.no_stdout = True

        try:
            text = run_parse(item_args) or ""
            status = "ok"
            preview = text.strip().replace("\t", " ").replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print_batch_block(stem, text)
        except (FileNotFoundError, ValueError) as exc:
            status = "error"
            preview = str(exc)
            print_batch_block(stem, f"ERROR: {exc}")

        with summary.open("a", encoding="utf-8") as f:
            f.write(f"{stem}\t{status}\t{preview}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Qwen3-VL om_output (JSON vocab, no tokenizers)")
    parser.add_argument("input_len", nargs="?", type=int)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--dump-dir", default="", help="prompt_bin dir for prefill seq_len")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--response-out", default="")
    parser.add_argument(
        "--batch-root",
        type=Path,
        nargs="?",
        const=DEFAULT_BATCH,
        default=None,
        help="parse batch items (default root: om/batch)",
    )
    parser.add_argument("--stem", default="", help="with --batch-root: only this item")
    parser.add_argument(
        "--write-response",
        action="store_true",
        help="with --batch-root: write each item's response.txt",
    )
    parser.add_argument("--keep-special-tokens", action="store_true")
    args = parser.parse_args()
    args.skip_special_tokens = not args.keep_special_tokens

    if args.batch_root is not None:
        run_parse_batch(args)
        return

    run_parse(args)


if __name__ == "__main__":
    main()
