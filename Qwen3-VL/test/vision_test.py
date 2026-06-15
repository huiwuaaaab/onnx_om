import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from onnx_common import (
    EXPORT_DIR,
    FLOAT_DTYPE,
    IMAGE_PATH,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    MODEL_PATH,
    ONNX_VISION,
    ORT_PROVIDERS,
    PROFILE,
    PROMPT,
    load_hf_qwen3_vl,
    onnx_path,
    preprocess,
)

DEVICE = "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _load_model_and_ort():
    model = load_hf_qwen3_vl(MODEL_PATH).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ort_session = ort.InferenceSession(onnx_path(ONNX_VISION), providers=ORT_PROVIDERS)
    print(f"\nprofile: {PROFILE.name}  image_size={IMAGE_SIZE}  max_seq_len={MAX_SEQ_LEN}")
    print(f"ONNX Provider: {ORT_PROVIDERS}")
    print(f"Vision ONNX: {onnx_path(ONNX_VISION)}")
    print(f"export_dir: {EXPORT_DIR}")
    return model, processor, ort_session


def run_preprocess(processor):
    t0 = time.time()
    inputs = preprocess(processor)
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(FLOAT_DTYPE)

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    pixel_values_np = pixel_values.cpu().numpy().astype(np.float16)
    print(f"\nPreprocess Time: {time.time() - t0:.4f}s")
    print(f"image: {IMAGE_PATH}")
    print(f"prompt: {PROMPT!r}")
    return inputs, pixel_values, grid_thw, pixel_values_np


def vision_compare(model, processor, ort_session, pixel_values, grid_thw, pixel_values_np):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        vision_out = model.model.visual(pixel_values, grid_thw)
        if isinstance(vision_out, tuple):
            pt_merged, pt_deep = vision_out
        else:
            pt_merged = vision_out.pooler_output
            pt_deep = vision_out.deepstack_features

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    t2 = time.time()
    onnx_outs = ort_session.run(None, {"hidden_states": pixel_values_np})
    t3 = time.time()

    onnx_merged = onnx_outs[0]
    onnx_deep = onnx_outs[1:]

    print("\n" + "=" * 70)

    def compare(a, b, name):
        diff = np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
        print(f"{name:15} | max error: {diff:.8f}")

    compare(pt_merged.cpu().numpy(), onnx_merged, "merged_hidden")
    for i in range(len(onnx_deep)):
        pt = pt_deep[i]
        if hasattr(pt, "dim") and pt.dim() == 3:
            pt = pt.squeeze(0)
        compare(pt.cpu().numpy(), onnx_deep[i], f"deepstack_{i}")

    print("\nVision Time")
    print(f"PyTorch Vision: {t1 - t0:.4f}s")
    print(f"ONNX Vision:    {t3 - t2:.4f}s")
    return onnx_merged, onnx_deep


def _build_generate_kwargs(inputs, max_new_tokens=50):
    """Pad 到 MAX_SEQ_LEN 的 input 需先截到 cur_len，否则 generate 从 pad 区续写。"""
    attention_mask = inputs["attention_mask"]
    cur_len = int(attention_mask[0].sum().item())
    gen_kwargs = {
        "input_ids": inputs["input_ids"][:, :cur_len],
        "attention_mask": attention_mask[:, :cur_len],
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if "image_grid_thw" in inputs:
        gen_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    if "mm_token_type_ids" in inputs:
        gen_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, :cur_len]
    return gen_kwargs, cur_len


def generate_compare(model, processor, inputs, onnx_merged, onnx_deep):
    gen_kwargs, cur_len = _build_generate_kwargs(inputs, max_new_tokens=50)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out_pt = model.generate(**gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    print("\nPyTorch Generation")
    print(processor.decode(out_pt[0, cur_len:], skip_special_tokens=True))
    print(f"Time: {t1 - t0:.4f}s")

    merged = torch.from_numpy(onnx_merged).to(DEVICE, dtype=FLOAT_DTYPE)
    split_sizes = (
        inputs["image_grid_thw"].prod(-1) // model.model.visual.spatial_merge_size**2
    ).tolist()
    pooler_output = torch.split(merged, split_sizes)
    deepstack_features = [
        torch.from_numpy(x).to(DEVICE, dtype=FLOAT_DTYPE) for x in onnx_deep
    ]

    def fake_get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs):
        output = BaseModelOutputWithDeepstackFeatures(
            pooler_output=pooler_output,
            deepstack_features=deepstack_features,
        )
        if return_dict:
            return output
        return pooler_output, deepstack_features

    original = model.model.get_image_features
    model.model.get_image_features = fake_get_image_features

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time()
    out_onnx = model.generate(**gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.time()

    print("\nONNX Vision + LLM")
    print(processor.decode(out_onnx[0, cur_len:], skip_special_tokens=True))
    print(f"Time: {t3 - t2:.4f}s")
    model.model.get_image_features = original


if __name__ == "__main__":
    model, processor, ort_session = _load_model_and_ort()
    inputs, pixel_values, grid_thw, pixel_values_np = run_preprocess(processor)
    onnx_merged, onnx_deep = vision_compare(
        model, processor, ort_session, pixel_values, grid_thw, pixel_values_np,
    )
    generate_compare(model, processor, inputs, onnx_merged, onnx_deep)
