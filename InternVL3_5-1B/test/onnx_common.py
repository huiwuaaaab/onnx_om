"""Shared fp16 ONNX test config and helpers for InternVL3_5-1B."""

from __future__ import annotations

import os

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "InternVL3_5-1B-HF")
IMAGE_PATH = os.path.join(ROOT_DIR, "InternVL3_5-1B-HF/examples/image1.jpg")
EXPORT_DIR = "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/InternVL3_5/onnx_export"
DEVICE = "cpu"
FLOAT_DTYPE = torch.float16
INT_DTYPE = torch.int32
MAX_SEQ_LEN = 512
PAD_TOKEN_ID = 151643
POSITION_IDS_BYTES = MAX_SEQ_LEN * 4


def make_preblock_position_ids(max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Static arange(0..max_len-1) for llm_preblock ONNX (matches export)."""
    return np.arange(max_len, dtype=np.int32).reshape(1, max_len)


def onnx_path(name: str) -> str:
    return os.path.join(EXPORT_DIR, name)


def get_providers(use_cuda: bool = False) -> list[str]:
    return ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]


def load_fp16_model():
    from transformers import InternVLForConditionalGeneration

    return InternVLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=FLOAT_DTYPE,
        attn_implementation="eager",
    ).to(DEVICE).eval()


def apply_fp16_inputs(inputs: dict) -> dict:
    out = dict(inputs)
    if "pixel_values" in out and torch.is_tensor(out["pixel_values"]):
        out["pixel_values"] = out["pixel_values"].to(device=DEVICE, dtype=FLOAT_DTYPE)
    return out


def as_fp16_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float16)
    return np.asarray(value, dtype=np.float16)


def as_int32_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int32)
    return np.asarray(value, dtype=np.int32)


def run_lm_head(head_sess, hidden, cur_len: int) -> np.ndarray:
    h_last = as_fp16_numpy(hidden[:, cur_len - 1:cur_len, :])
    return head_sess.run(None, {"hidden_states": h_last})[0]


def pad(input_ids, attention_mask, pad_id: int = PAD_TOKEN_ID, max_len: int = MAX_SEQ_LEN):
    B, L = input_ids.shape
    if L > max_len:
        raise ValueError(f"seq_len {L} > {max_len}")

    pad_len = max_len - L
    if pad_len <= 0:
        return input_ids, attention_mask

    pad_ids = torch.full(
        (B, pad_len),
        pad_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    pad_mask = torch.zeros(
        (B, pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return (
        torch.cat([input_ids, pad_ids], dim=1),
        torch.cat([attention_mask, pad_mask], dim=1),
    )
