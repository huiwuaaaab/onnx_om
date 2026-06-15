import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ===============================
# 配置
# ===============================
MODEL_PATH = "./gemma-4-E2B-it"
MM_PROJ_ONNX_PATH = "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/gemma4/onnx_export/mm_proj.onnx"

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

processor = AutoProcessor.from_pretrained(MODEL_PATH)

providers = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]
ort_session = ort.InferenceSession(MM_PROJ_ONNX_PATH, providers=providers)
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
            "/e-vepfs-01/perception/wuhui/InternVL3_5-1B/InternVL3_5-1B-HF/examples/image1.jpg"
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

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE)


@torch.no_grad()
def proj_compare(inputs):
    pixel_values = inputs["pixel_values"].to(torch.float16)
    image_position_ids = inputs["image_position_ids"]

    vision_outputs = model.model.vision_tower(
        pixel_values=pixel_values,
        pixel_position_ids=image_position_ids,
    )
    last_hidden_state = vision_outputs.last_hidden_state

    pt_output = model.model.embed_vision(inputs_embeds=last_hidden_state)

    vision_features = last_hidden_state.unsqueeze(0).cpu().numpy()
    onnx_output = ort_session.run(None, {"vision_features": vision_features})[0]

    print(f"\n===== MM_PROJ COMPARE (fp16) =====")
    print(f"input vision_features: {vision_features.shape} {vision_features.dtype}")
    print(f"torch: {tuple(pt_output.shape)}, onnx: {onnx_output.shape}")
    print_align_metrics(pt_output.cpu().numpy(), onnx_output, name="mm_proj_hidden")


if __name__ == "__main__":
    proj_compare(preprocess())
