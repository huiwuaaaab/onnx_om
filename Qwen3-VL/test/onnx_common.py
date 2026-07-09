"""Shared ONNX / OM test helpers (aligned with export_config profiles)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Default test layout: 448px image + 512 seq (override via QWEN3_EXPORT_PROFILE=256_256).
os.environ.setdefault("QWEN3_EXPORT_PROFILE", "448_512")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_config import ExportProfile, get_export_profile  # noqa: E402

PROFILE: ExportProfile = get_export_profile()

from PIL import Image

from llm import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FLOAT_DTYPE,
    INT_DTYPE,
    apply_export_profile,
    compute_static_position_ids,
    load_hf_qwen3_vl,
)

apply_export_profile(PROFILE)

MODEL_PATH = os.environ.get(
    "QWEN3_MODEL_PATH",
    str(REPO_ROOT / Path(DEFAULT_MODEL_PATH).name),
)
IMAGE_PATH = os.environ.get(
    "QWEN3_IMAGE_PATH",
    "../../imgs/example.jpg",
)
PROMPT = os.environ.get("QWEN3_PROMPT", "What is shown in this image?")

IMAGE_SIZE = PROFILE.image_size
MAX_SEQ_LEN = PROFILE.max_seq_len
EXPORT_DIR = PROFILE.export_dir
ONNX_VISION = PROFILE.vision_onnx_name
IMAGE_TOKEN_START = PROFILE.image_token_start
IMAGE_TOKEN_END = PROFILE.image_token_end
NUM_IMAGE_TOKENS = PROFILE.num_image_tokens

ONNX_PREBLOCK = "llm_preblock.onnx"
ONNX_BLOCK1 = "llm_block1.onnx"
ONNX_BLOCK2 = "llm_block2.onnx"
ONNX_BLOCK3 = "llm_block3.onnx"
ONNX_LM_HEAD = "lm_head.onnx"

USE_ONNX_CUDA = os.environ.get("QWEN3_ONNX_CUDA", "0") == "1"
ORT_PROVIDERS = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]


def onnx_path(name: str) -> str:
    return str(Path(EXPORT_DIR) / name)


def default_ort_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return opts


def load_onnx_chain(include_vision: bool = True):
    """Load vision + preblock + block1..3 + lm_head ONNX sessions."""
    opts = default_ort_session_options()
    providers = ORT_PROVIDERS
    vision = None
    if include_vision:
        vision = ort.InferenceSession(onnx_path(ONNX_VISION), opts, providers=providers)
    pre = ort.InferenceSession(onnx_path(ONNX_PREBLOCK), opts, providers=providers)
    b1 = ort.InferenceSession(onnx_path(ONNX_BLOCK1), opts, providers=providers)
    b2 = ort.InferenceSession(onnx_path(ONNX_BLOCK2), opts, providers=providers)
    b3 = ort.InferenceSession(onnx_path(ONNX_BLOCK3), opts, providers=providers)
    head = ort.InferenceSession(onnx_path(ONNX_LM_HEAD), opts, providers=providers)
    return vision, pre, b1, b2, b3, head


def preprocess(processor, image_path: str = IMAGE_PATH, prompt: str = PROMPT):
    """Qwen3-VL fixed layout; image/prompt aligned with active export profile."""
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )


def to_fp16_inputs(inputs, device: str = "cpu"):
    out = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if key == "pixel_values":
            out[key] = value.to(device=device, dtype=FLOAT_DTYPE)
        else:
            out[key] = value.to(device)
    return out


def build_preblock_onnx_inputs(input_ids, image_embeds, attention_mask):
    """Build llm_preblock.onnx feed dict (numpy, fp16/int32)."""
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.detach().cpu()
    else:
        ids = torch.from_numpy(np.asarray(input_ids))

    if isinstance(attention_mask, torch.Tensor):
        mask = attention_mask.detach().cpu()
    else:
        mask = torch.from_numpy(np.asarray(attention_mask))

    ids = ids.to(INT_DTYPE)
    mask = mask.to(INT_DTYPE)

    if isinstance(image_embeds, torch.Tensor):
        image_embeds = image_embeds.detach().cpu().numpy()
    image_embeds = np.asarray(image_embeds, dtype=np.float16)

    position_ids = compute_static_position_ids(ids, mask, "cpu")

    return {
        "input_ids": ids.numpy().astype(np.int32),
        "image_embeds": image_embeds,
        "attention_mask": mask.numpy().astype(np.int32),
        "position_ids": position_ids.numpy().astype(np.int32),
    }


def build_onnx_inputs(ort_session, base_inputs, deepstack):
    ort_inputs = dict(base_inputs)
    input_names = [i.name for i in ort_session.get_inputs()]
    for i, ds in enumerate(deepstack):
        name = f"ds_{i}"
        if name in input_names:
            ort_inputs[name] = ds
    return ort_inputs


def run_onnx_vision(vision_sess, pixel_values):
    merged, ds5, ds11, ds17 = vision_sess.run(
        None,
        {"hidden_states": pixel_values.cpu().numpy().astype(np.float16)},
    )
    return merged, [ds5, ds11, ds17]


def run_lm_head(head_sess, hidden, cur_len):
    h_last = hidden[:, cur_len - 1:cur_len, :].astype(np.float16)
    return head_sess.run(None, {"hidden_states": h_last})[0]


def to_deepstack_np(deepstack):
    out = []
    for ds in deepstack:
        if isinstance(ds, torch.Tensor):
            ds = ds.squeeze(0) if ds.dim() == 3 else ds
            out.append(ds.cpu().numpy().astype(np.float16))
        else:
            out.append(np.asarray(ds, dtype=np.float16))
    return out
