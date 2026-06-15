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
ort_session = ort.InferenceSession(onnx_path("mm_proj.onnx"), providers=providers)
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

    print(f"\n===== Preprocess Time: {time.time() - t0:.4f}s =====")
    return inputs


def vision_compare(inputs):
    pixel_values = inputs["pixel_values"]

    t0 = time.time()
    with torch.no_grad():
        pt_output = model.get_image_features(
            pixel_values,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        ).pooler_output
    t1 = time.time()

    with torch.no_grad():
        vision_feature = model.model.vision_tower(pixel_values).last_hidden_state
        vision_feature_np = as_fp16_numpy(vision_feature)

    t2 = time.time()
    onnx_output = ort_session.run(None, {"vision_features": vision_feature_np})
    t3 = time.time()

    print("\n" + "=" * 70)

    def compare(a, b, name):
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b[0] if isinstance(b, list) else b, dtype=np.float32)
        diff = np.max(np.abs(a32 - b32))
        print(f"{name:15} | max error: {diff:.8f}")

    compare(pt_output.cpu().numpy(), onnx_output, "mm_proj_hidden")

    print("\n===== Time =====")
    print(f"PyTorch: {t1 - t0:.4f}s")
    print(f"ONNX:    {t3 - t2:.4f}s")


if __name__ == "__main__":
    inputs = preprocess()
    vision_compare(inputs)
