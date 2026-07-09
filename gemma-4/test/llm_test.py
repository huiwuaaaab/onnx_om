import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ===============================
# 配置
# ===============================
MODEL_PATH = "./gemma-4-E2B-it"
ONNX_BASE = "./onnx_export"

DEVICE = "cpu"
USE_ONNX_CUDA = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_onnx_sessions():
    providers = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]

    ort_pre = ort.InferenceSession(f"{ONNX_BASE}/llm_preblock.onnx", providers=providers)
    ort_b1 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_0_5.onnx", providers=providers)
    ort_b2 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_5_10.onnx", providers=providers)
    ort_b3 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_10_15.onnx", providers=providers)
    ort_b4 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_15_20.onnx", providers=providers)
    ort_b5 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_20_25.onnx", providers=providers)
    ort_b6 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_25_30.onnx", providers=providers)
    ort_b7 = ort.InferenceSession(f"{ONNX_BASE}/llm_block_30_35.onnx", providers=providers)

    all_models = [
        ("PRE", ort_pre),
        ("B1 (0-5)", ort_b1),
        ("B2 (5-10)", ort_b2),
        ("B3 (10-15)", ort_b3),
        ("B4 (15-20)", ort_b4),
        ("B5 (20-25)", ort_b5),
        ("B6 (25-30)", ort_b6),
        ("B7 (30-35)", ort_b7),
    ]

    print(f"🚀 ONNX Provider: {providers}")
    for name, sess in all_models:
        print(f"\n🔹 {name}")
        print("  INPUTS:")
        for i, inp in enumerate(sess.get_inputs()):
            print(f"    [{i}] {inp.name}  (shape: {inp.shape}, dtype: {inp.type})")
        print("  OUTPUTS:")
        for i, out in enumerate(sess.get_outputs()):
            print(f"    [{i}] {out.name}  (shape: {out.shape}, dtype: {out.type})")

    return ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7


def print_align_metrics(ref, pred, name: str = "hidden", ref_thresh: float = 0.01) -> None:
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


def _onnx_float_dtype(session: ort.InferenceSession) -> np.dtype:
    elem_type = session.get_inputs()[0].type
    return np.float16 if "float16" in elem_type else np.float32


def _cast_float_array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype in (np.float16, np.float32, np.float64):
        return arr.astype(dtype, copy=False)
    return arr


def preprocess(processor):
    image = (
        Image.open(
            "../../imgs/example.jpg"
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

    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def create_causal_mask(attention_mask: torch.Tensor | None, dtype: torch.dtype = torch.float16):
    B, L = attention_mask.shape
    device = attention_mask.device

    q_idx = torch.arange(L, device=device).view(L, 1)
    kv_idx = torch.arange(L, device=device).view(1, L)
    causal = kv_idx <= q_idx
    causal = causal.view(1, 1, L, L)

    if attention_mask is not None:
        key_mask = attention_mask.bool()[:, None, None, :]
        causal = causal & key_mask

    min_dtype = torch.finfo(dtype).min
    return torch.where(
        causal,
        torch.zeros_like(causal, dtype=dtype),
        torch.full_like(causal, min_dtype, dtype=dtype),
    )


def create_sliding_window_causal_mask(
    attention_mask: torch.Tensor | None,
    sliding_window: int,
    dtype: torch.dtype = torch.float16,
):
    B, L = attention_mask.shape
    device = attention_mask.device

    q_idx = torch.arange(L, device=device).view(L, 1)
    kv_idx = torch.arange(L, device=device).view(1, L)
    causal = kv_idx <= q_idx
    sliding = kv_idx > (q_idx - sliding_window)
    mask = causal & sliding
    mask = mask.view(1, 1, L, L)

    if attention_mask is not None:
        key_mask = attention_mask.bool()[:, None, None, :]
        mask = mask & key_mask

    min_dtype = torch.finfo(dtype).min
    return torch.where(
        mask,
        torch.zeros_like(mask, dtype=dtype),
        torch.full_like(mask, min_dtype, dtype=dtype),
    )


@torch.no_grad()
def get_llm_inputs(model, inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs["image_position_ids"]
    mask_dtype = torch.float16

    image_mask, _, _ = model.model.get_placeholder_mask(input_ids)

    llm_input_ids = input_ids.clone()
    llm_input_ids[image_mask] = model.model.language_model.config.pad_token_id
    inputs_embeds = model.model.get_input_embeddings()(llm_input_ids)

    pad_embedding = model.model.language_model.embed_tokens.weight[
        model.model.config.text_config.pad_token_id, :
    ]
    image_mask = image_mask.to(inputs_embeds.device)
    llm_inputs_embeds = torch.where(
        image_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds
    )
    per_layer_inputs = model.model.language_model.get_per_layer_inputs(
        llm_input_ids, llm_inputs_embeds
    )

    image_features = model.model.get_image_features(
        pixel_values, image_position_ids, return_dict=True
    ).pooler_output
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask_exp, image_features.to(inputs_embeds.device)
    )

    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

    causal_mask_mapping = {
        "full_attention": create_causal_mask(attention_mask, dtype=mask_dtype),
        "sliding_attention": create_sliding_window_causal_mask(
            attention_mask,
            model.model.language_model.config.sliding_window,
            dtype=mask_dtype,
        ),
    }

    hidden_states = inputs_embeds
    position_embeddings = {}
    for layer_type in model.model.language_model.unique_layer_types:
        position_embeddings[layer_type] = model.model.language_model.rotary_emb(
            hidden_states, position_ids, layer_type
        )

    return {
        "per_layer_inputs": per_layer_inputs,
        "causal_mask_mapping": causal_mask_mapping,
        "position_ids": position_ids,
        "inputs_embeds": inputs_embeds,
        "image_embeds": image_features,
        "position_embeddings": position_embeddings,
    }


@torch.no_grad()
def get_onnx_inputs(model, inputs, llm_inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_embeds = llm_inputs["image_embeds"].unsqueeze(0)

    B, L = input_ids.shape
    max_len = 512
    pad_len = max_len - L
    pad_ids = torch.full((B, pad_len), 0, dtype=input_ids.dtype, device=input_ids.device)
    pad_mask = torch.zeros((B, pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
    input_ids_pad = torch.cat([input_ids, pad_ids], dim=1)
    attention_mask_pad = torch.cat([attention_mask, pad_mask], dim=1)

    llm_input_ids = input_ids_pad.clone()
    image_mask, _, _ = model.model.get_placeholder_mask(llm_input_ids)
    llm_input_ids[image_mask] = model.model.language_model.config.pad_token_id
    inputs_embeds = model.model.get_input_embeddings()(llm_input_ids)

    pad_embedding = model.model.language_model.embed_tokens.weight[
        model.model.config.text_config.pad_token_id, :
    ]
    image_mask = image_mask.to(inputs_embeds.device)
    llm_inputs_embeds = torch.where(
        image_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds
    )
    per_layer_inputs = model.model.language_model.get_per_layer_inputs(
        llm_input_ids, llm_inputs_embeds
    )

    position_ids = np.arange(max_len, dtype=np.int32).reshape(1, max_len)

    return {
        "input_ids": input_ids_pad.to(torch.int32).cpu().numpy(),
        "attention_mask": attention_mask_pad.to(torch.int32).cpu().numpy(),
        "image_embeds": image_embeds.cpu().numpy().astype(np.float16),
        "per_layer_inputs": per_layer_inputs.cpu().numpy().astype(np.float16),
        "position_ids": position_ids,
    }


def run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, onnx_inputs):
    block_dtype = _onnx_float_dtype(ort_b1)

    pre_inputs = {
        "input_ids": onnx_inputs["input_ids"],
        "image_embeds": onnx_inputs["image_embeds"],
        "attention_mask": onnx_inputs["attention_mask"],
        "per_layer_inputs": onnx_inputs["per_layer_inputs"],
        "position_ids": onnx_inputs["position_ids"],
    }
    (
        inputs_embeds,
        per_layer_inputs,
        full_mask,
        sliding_mask,
        cos_full,
        sin_full,
        cos_slide,
        sin_slide,
    ) = ort_pre.run(None, pre_inputs)

    inputs_embeds = _cast_float_array(inputs_embeds, block_dtype)
    per_layer_inputs = _cast_float_array(per_layer_inputs, block_dtype)
    full_mask = _cast_float_array(full_mask, block_dtype)
    sliding_mask = _cast_float_array(sliding_mask, block_dtype)
    cos_full = _cast_float_array(cos_full, block_dtype)
    sin_full = _cast_float_array(sin_full, block_dtype)
    cos_slide = _cast_float_array(cos_slide, block_dtype)
    sin_slide = _cast_float_array(sin_slide, block_dtype)

    def _block_common():
        return {
            "full_mask": full_mask,
            "sliding_mask": sliding_mask,
            "cos_full": cos_full,
            "sin_full": sin_full,
            "cos_slide": cos_slide,
            "sin_slide": sin_slide,
        }

    hidden = ort_b1.run(
        None,
        {
            "inputs_embeds": inputs_embeds,
            **_block_common(),
            "per_layer_input": per_layer_inputs[:, :, 0:5, :],
        },
    )[0]

    hidden = ort_b2.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            "per_layer_input": per_layer_inputs[:, :, 5:10, :],
        },
    )[0]

    hidden, full_k, full_v, slide_k, slide_v = ort_b3.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            "per_layer_input": per_layer_inputs[:, :, 10:15, :],
        },
    )

    kv_inputs = {"full_k": full_k, "full_v": full_v, "slide_k": slide_k, "slide_v": slide_v}

    hidden = ort_b4.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            **kv_inputs,
            "per_layer_input": per_layer_inputs[:, :, 15:20, :],
        },
    )[0]

    hidden = ort_b5.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            **kv_inputs,
            "per_layer_input": per_layer_inputs[:, :, 20:25, :],
        },
    )[0]

    hidden = ort_b6.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            **kv_inputs,
            "per_layer_input": per_layer_inputs[:, :, 25:30, :],
        },
    )[0]

    hidden = ort_b7.run(
        None,
        {
            "inputs_embeds": hidden,
            **_block_common(),
            **kv_inputs,
            "per_layer_input": per_layer_inputs[:, :, 30:35, :],
        },
    )[0]

    return hidden, None


@torch.no_grad()
def compare_once_split(
    model,
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    onnx_inputs,
    inputs,
):
    pt_out = model.model(**inputs, return_shared_kv_states=True)
    hidden = pt_out.last_hidden_state

    onnx_hidden, _ = run_onnx_llm_split(
        ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, onnx_inputs
    )

    cur_len = int(onnx_inputs["attention_mask"].sum())
    print(f"\n===== LLM COMPARE (fp16, seq_len={cur_len}) =====")
    print_align_metrics(
        hidden[:, :cur_len].cpu().numpy(),
        onnx_hidden[:, :cur_len],
        name="last_hidden_state",
    )


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).to(DEVICE).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7 = load_onnx_sessions()

    inputs = preprocess(processor)
    llm_inputs = get_llm_inputs(model, inputs)
    onnx_inputs = get_onnx_inputs(model, inputs, llm_inputs)

    compare_once_split(
        model,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        onnx_inputs,
        inputs,
    )
