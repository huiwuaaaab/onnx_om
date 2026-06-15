import time

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoProcessor

from onnx_common import (
    DEVICE,
    IMAGE_PATH,
    MODEL_PATH,
    apply_fp16_inputs,
    as_fp16_numpy,
    get_providers,
    load_fp16_model,
    onnx_path,
)

USE_ONNX_CUDA = False

model = load_fp16_model()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.image_processor.min_patches = 1
processor.image_processor.max_patches = 1

providers = get_providers(USE_ONNX_CUDA)
ort_session = ort.InferenceSession(onnx_path("vision_448_notchunk.onnx"), providers=providers)
print(f"\n🚀 ONNX Provider: {providers}")


def preprocess():
    t0 = time.time()

    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Please describe the image shortly."},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(text)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)
    inputs = apply_fp16_inputs(inputs)

    pixel_values = inputs["pixel_values"]
    pixel_values_np = as_fp16_numpy(pixel_values)

    print(f"\n===== Preprocess Time: {time.time() - t0:.4f}s =====")
    return inputs, pixel_values, pixel_values_np


def vision_compare(pixel_values, pixel_values_np):
    t0 = time.time()
    with torch.no_grad():
        output = model.model.vision_tower(pixel_values)
        pt_merged = output.last_hidden_state
    t1 = time.time()

    t2 = time.time()
    onnx_outs = ort_session.run(None, {"pixel_values": pixel_values_np})
    t3 = time.time()

    onnx_merged = onnx_outs[0]

    print("\n" + "=" * 70)

    def compare(a, b, name):
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b, dtype=np.float32)
        diff = np.max(np.abs(a32 - b32))
        print(f"{name:15} | max error: {diff:.8f}")

    compare(pt_merged.cpu().numpy(), onnx_merged, "merged_hidden")

    print("\n===== Vision Time =====")
    print(f"PyTorch Vision: {t1 - t0:.4f}s")
    print(f"ONNX Vision:    {t3 - t2:.4f}s")

    return onnx_merged


if __name__ == "__main__":
    inputs, pixel_values, pixel_values_np = preprocess()
    vision_compare(pixel_values, pixel_values_np)
