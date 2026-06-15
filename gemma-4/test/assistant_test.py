import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ======================
# 配置
# ======================
DEVICE = "cpu"
TARGET_MODEL_ID = "./gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "./gemma-4-E2B-it-assistant"
ONNX_MODEL_PATH = "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/gemma4/onnx_export/assistant.onnx"
MAX_SEQ_LEN = 512


def print_align_metrics(ref, pred, name: str = "projected_state", ref_thresh: float = 0.01) -> None:
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


def main():
    processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

    image = Image.open("/e-vepfs-01/perception/wuhui/image1.jpeg").convert("RGB").resize((768, 768))
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B, L = input_ids.shape
    pad_len = MAX_SEQ_LEN - L

    # 主模型 KV（RIGHT PAD，与 ONNX 一致）
    input_ids_padded = torch.cat(
        [input_ids, torch.zeros(B, pad_len, dtype=input_ids.dtype, device=DEVICE)], dim=1
    )
    attn_mask_padded = torch.cat(
        [attention_mask, torch.zeros(B, pad_len, dtype=attention_mask.dtype, device=DEVICE)],
        dim=1,
    )
    inputs_padded = {**inputs, "input_ids": input_ids_padded, "attention_mask": attn_mask_padded}

    with torch.no_grad():
        out_pad = target_model.model(**inputs_padded, return_shared_kv_states=True)

    kv_padded = out_pad.shared_kv_states
    last_hidden_padded = out_pad.last_hidden_state[:, L - 1 : L]
    last_token_id = input_ids[:, L - 1 : L]
    position_ids = torch.tensor([[L - 1]], device=DEVICE)

    last_embedding = target_model.get_input_embeddings()(last_token_id)
    inputs_embeds = torch.cat([last_embedding, last_hidden_padded], dim=-1)

    with torch.no_grad():
        torch_out = assistant_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_padded,
            position_ids=position_ids,
            shared_kv_states=kv_padded,
            use_cache=False,
        )

    onnx_inputs = {
        "last_token_id": last_token_id.cpu().numpy().astype(np.int32),
        "last_hidden": last_hidden_padded.cpu().numpy().astype(np.float16),
        "attention_mask": attn_mask_padded.cpu().numpy().astype(np.int32),
        "position_ids": position_ids.cpu().numpy().astype(np.int32),
        "full_k": kv_padded["full_attention"][0].detach().cpu().numpy().astype(np.float16),
        "full_v": kv_padded["full_attention"][1].detach().cpu().numpy().astype(np.float16),
        "slide_k": kv_padded["sliding_attention"][0].detach().cpu().numpy().astype(np.float16),
        "slide_v": kv_padded["sliding_attention"][1].detach().cpu().numpy().astype(np.float16),
    }
    projected_state, logits, _ = ort_session.run(None, onnx_inputs)

    print("\n===== ASSISTANT COMPARE (fp16, RIGHT PAD) =====")
    print(f"torch projected: {tuple(torch_out.last_hidden_state.shape)}, onnx: {projected_state.shape}")
    print_align_metrics(
        torch_out.last_hidden_state.cpu().numpy(),
        projected_state,
        name="projected_state",
    )


if __name__ == "__main__":
    main()
