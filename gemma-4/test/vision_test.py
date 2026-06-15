import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import Gemma4VisionModel, build_vision_config

# ===============================
# 配置
# ===============================
MODEL_PATH = "./gemma-4-E2B-it"
VISION_ONNX_PATH = "./onnx_export/vision.onnx"

DEVICE = "cpu"
USE_ONNX_CUDA = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ===============================
# 加载模型（PyTorch FP16）
# ===============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
).to(DEVICE).eval()

vision_torch = Gemma4VisionModel(build_vision_config()).eval().to(DEVICE).to(torch.float16)
vision_torch.load_from_pretrained(model.model.vision_tower)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

providers = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]
ort_session = ort.InferenceSession(VISION_ONNX_PATH, providers=providers)
print(f"\n🚀 ONNX Provider: {providers}")


def print_align_metrics(ref, pred, name: str = "hidden_states", ref_thresh: float = 0.01) -> None:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    diff = np.abs(ref - pred)
    cos = float(np.dot(ref, pred) / (np.linalg.norm(ref) * np.linalg.norm(pred) + 1e-12))

    mask = np.abs(ref) >= ref_thresh
    if mask.any():
        rel = diff[mask] / np.abs(ref[mask])
        mean_rel_pct = 100.0 * rel.mean()
        p99_rel_pct = 100.0 * np.percentile(rel, 99)
        max_abs = diff[mask].max()
        n_skip = int((~mask).sum())
    else:
        mean_rel_pct = p99_rel_pct = max_abs = float("nan")
        n_skip = len(ref)

    print(
        f"[{name}] cosine={cos:.6f}  "
        f"mean_rel={mean_rel_pct:.2f}%  p99_rel={p99_rel_pct:.2f}%  max_abs={max_abs:.4f}  "
        f"(|torch|>={ref_thresh}, skip {n_skip})"
    )


def preprocess():
    image = (
        Image.open(
            "path/to/image.jpg"
        )
        .convert("RGB")
        .resize((768, 768))
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE)

    pixel_values = inputs["pixel_values"].to(torch.float16)
    image_position_ids = inputs["image_position_ids"].to(torch.int32)

    return {
        "pixel_values": pixel_values.cpu().numpy(),
        "image_position_ids": image_position_ids.cpu().numpy(),
        "pixel_values_torch": pixel_values,
        "image_position_ids_torch": image_position_ids,
    }


@torch.no_grad()
def vision_compare(data):
    onnx_hidden = ort_session.run(
        None,
        {
            "pixel_values": data["pixel_values"],
            "image_position_ids": data["image_position_ids"],
        },
    )[0]

    torch_hidden = vision_torch(
        data["pixel_values_torch"],
        data["image_position_ids_torch"],
    )

    print(f"\n===== VISION COMPARE (fp16, single output) =====")
    print(f"torch: {tuple(torch_hidden.shape)}, onnx: {onnx_hidden.shape}")
    print_align_metrics(torch_hidden.cpu().numpy(), onnx_hidden, name="hidden_states")


if __name__ == "__main__":
    vision_compare(preprocess())
