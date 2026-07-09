import time
import torch
import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
import onnxruntime as ort

# ===============================
# 配置
# ===============================
MODEL_PATH = "./gemma-4-E2B-it"

DEVICE = "cpu"
USE_ONNX_CUDA = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ===============================
# 加载 ONNX
# ===============================
def load_onnx():
    providers = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]

    vision = ort.InferenceSession("./onnx_export/vision.onnx", providers=providers)
    mm_proj = ort.InferenceSession("./onnx_export/mm_proj.onnx", providers=providers)

    ort_pre = ort.InferenceSession("./onnx_export/llm_preblock.onnx", providers=providers)
    ort_b1  = ort.InferenceSession("./onnx_export/llm_block_0_5.onnx", providers=providers)
    ort_b2  = ort.InferenceSession("./onnx_export/llm_block_5_10.onnx", providers=providers)
    ort_b3  = ort.InferenceSession("./onnx_export/llm_block_10_15.onnx", providers=providers)
    ort_b4  = ort.InferenceSession("./onnx_export/llm_block_15_20.onnx", providers=providers)
    ort_b5  = ort.InferenceSession("./onnx_export/llm_block_20_25.onnx", providers=providers)
    ort_b6  = ort.InferenceSession("./onnx_export/llm_block_25_30.onnx", providers=providers)
    ort_b7  = ort.InferenceSession("./onnx_export/llm_block_30_35.onnx", providers=providers)
    ort_lm_head = ort.InferenceSession(
        "./onnx_export/lm_head.onnx",
        providers=providers,
    )

    print(f"🚀 ONNX Provider: {providers}")

    return vision, mm_proj, ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head


# ===============================
# 预处理
# ===============================
def preprocess(processor):
    image = Image.open('../../imgs/example.jpg').convert("RGB").resize((768, 768))

    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        }
    ]

    # Process input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE).to(torch.float16)

    return inputs

@torch.no_grad()
def get_onnx_inputs(model, inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs['image_position_ids'].to(torch.int32)
    
    B, L = input_ids.shape
    max_len = 512
    pad_id = 0
    pad_len = max_len - L
    pad_ids = torch.full(
        (B, pad_len),
        pad_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )

    pad_mask = torch.zeros(
        (B, pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )
    input_ids_pad = torch.cat([input_ids, pad_ids], dim=1)
    attention_mask_pad = torch.cat([attention_mask, pad_mask], dim=1)

    llm_input_ids = input_ids_pad.clone()
    image_mask,_,_ = model.model.get_placeholder_mask(llm_input_ids)
    llm_input_ids[image_mask] = model.model.language_model.config.pad_token_id
    inputs_embeds = model.model.get_input_embeddings()(llm_input_ids)

    # ===== per_layer_inputs =====
    pad_embedding = model.model.language_model.embed_tokens.weight[model.model.config.text_config.pad_token_id, :]
    image_mask = image_mask.to(inputs_embeds.device)
    llm_inputs_embeds = torch.where(image_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
    per_layer_inputs = model.model.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)

    return {
        "pixel_values": pixel_values.cpu().numpy(),
        "image_position_ids": image_position_ids.cpu().numpy(),
        "input_ids": input_ids_pad.to(torch.int32).cpu().numpy(),
        "attention_mask": attention_mask_pad.to(torch.int32).cpu().numpy(),
        "per_layer_inputs": per_layer_inputs.cpu().numpy(),
    }

def run_onnx_vision(vision_sess, pixel_values, image_position_ids):
    image_embeds = vision_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids
        }
    )[0]
    return image_embeds

def run_onnx_mmproj(proj_sess, image_embeds):
    proj_embeds = proj_sess.run(
        None,
        {
        "vision_features": image_embeds
        }
    )[0]
    return proj_embeds

def run_onnx_llm_split(
    ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
    onnx_inputs, image_embeds, cur_len,
):
    # ========= preblock =========
    pre_inputs = {
        "input_ids": onnx_inputs["input_ids"],
        "image_embeds": image_embeds,
        "attention_mask": onnx_inputs["attention_mask"],
        "per_layer_inputs": onnx_inputs["per_layer_inputs"],
        "position_ids": np.arange(512, dtype=np.int32).reshape(1, 512),
    }
    inputs_embeds, per_layer_inputs, full_mask, sliding_mask, cos_full, sin_full, cos_slide, sin_slide = ort_pre.run(None, pre_inputs)

    # ========= block1 =========
    b1_inputs = {
        "inputs_embeds": inputs_embeds,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "per_layer_input": per_layer_inputs[:, :, 0:5, :],
    }
    hidden = ort_b1.run(None, b1_inputs)[0]

    # ========= block2 =========
    b2_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "per_layer_input": per_layer_inputs[:, :, 5:10, :],
    }
    hidden = ort_b2.run(None, b2_inputs)[0]

    # ========= block3 =========
    b3_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "per_layer_input": per_layer_inputs[:, :, 10:15, :],
    }
    hidden, out_full_k, out_full_v, out_slide_k, out_slide_v = ort_b3.run(None, b3_inputs)
    full_k, full_v, slide_k, slide_v = out_full_k, out_full_v, out_slide_k, out_slide_v

    # ========= block4 =========
    b4_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "full_k": full_k, "full_v": full_v, "slide_k": slide_k, "slide_v": slide_v,
        "per_layer_input": per_layer_inputs[:, :, 15:20, :],
    }
    hidden = ort_b4.run(None, b4_inputs)[0]

    # ========= block5 =========
    b5_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "full_k": full_k, "full_v": full_v, "slide_k": slide_k, "slide_v": slide_v,
        "per_layer_input": per_layer_inputs[:, :, 20:25, :],
    }
    hidden = ort_b5.run(None, b5_inputs)[0]

    # ========= block6 =========
    b6_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "full_k": full_k, "full_v": full_v, "slide_k": slide_k, "slide_v": slide_v,
        "per_layer_input": per_layer_inputs[:, :, 25:30, :],
    }
    hidden = ort_b6.run(None, b6_inputs)[0]

    # ========= block7 =========
    b7_inputs = {
        "inputs_embeds": hidden,
        "full_mask": full_mask, "sliding_mask": sliding_mask,
        "cos_full": cos_full, "sin_full": sin_full,
        "cos_slide": cos_slide, "sin_slide": sin_slide,
        "full_k": full_k, "full_v": full_v, "slide_k": slide_k, "slide_v": slide_v,
        "per_layer_input": per_layer_inputs[:, :, 30:35, :],
    }
    hidden = ort_b7.run(None, b7_inputs)[0]
    h_last = hidden[:, cur_len - 1 : cur_len, :].astype(np.float16)
    logits = ort_lm_head.run(None, {"hidden_states": h_last})[0]

    return hidden, logits

@torch.no_grad()
def generate_onnx(
    ort_vision,ort_proj,
    ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
    model,
    onnx_inputs,
    steps=50
):  
    image_embeds = run_onnx_vision(
        ort_vision,
        onnx_inputs["pixel_values"],
        onnx_inputs["image_position_ids"]
    )

    image_embeds = run_onnx_mmproj(ort_proj, image_embeds)

    attention_mask = onnx_inputs['attention_mask']
    cur_len = int(attention_mask.sum().item())
    generated_tokens = []

    for step in range(steps):

        # ===== 跑拆分 ONNX =====
        hidden, logits = run_onnx_llm_split(
            ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
            onnx_inputs, image_embeds, cur_len,
        )
        # ===== 取最后 token（lm_head 输出 [1,1,vocab]）=====
        next_token = torch.argmax(
            torch.from_numpy(logits)[:, 0, :],
            dim=-1,
            keepdim=True
        )

        generated_tokens.append(next_token)

        # ===== 写回 =====
        onnx_inputs['input_ids'][:, cur_len] = next_token[:, 0].cpu().numpy()
        onnx_inputs['attention_mask'][:, cur_len] = 1

        input_ids_torch = torch.from_numpy(onnx_inputs['input_ids']).to(model.device)
        image_mask = torch.zeros_like(input_ids_torch, dtype=torch.bool)
        image_mask[:, 5:261] = True

        pad_embedding = model.model.language_model.embed_tokens.weight[model.model.config.text_config.pad_token_id]
        image_mask = image_mask.to(model.device)
        input_ids_torch[image_mask] = model.model.language_model.config.pad_token_id
        inputs_embeds = model.model.get_input_embeddings()(input_ids_torch)
        llm_inputs_embeds = torch.where(image_mask[..., None], pad_embedding.view(1,1,-1), inputs_embeds)

        per_layer_inputs = model.model.language_model.get_per_layer_inputs(
            input_ids_torch, llm_inputs_embeds
        ).cpu().numpy()

        onnx_inputs["per_layer_inputs"] = per_layer_inputs

        cur_len += 1

        if cur_len >= 512:
            break

    tokens = torch.cat(generated_tokens, dim=1)
    return tokens

def print_align_metrics(
    ref: np.ndarray,
    pred: np.ndarray,
    name: str = "hidden",
    ref_thresh: float = 0.01,
) -> None:
    """简单对齐指标：cosine + 有效元素相对误差（避免 torch≈0 把 max_rel 撑爆）。"""
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
        n_skip = (~mask).sum()
    else:
        mean_rel_pct = p99_rel_pct = max_abs = float("nan")
        n_skip = len(ref)

    print(
        f"[{name}] cosine={cos:.6f}  "
        f"mean_rel={mean_rel_pct:.2f}%  p99_rel={p99_rel_pct:.2f}%  max_abs={max_abs:.4f}  "
        f"(统计 |torch|>={ref_thresh}, 跳过 {n_skip} 个近零维)"
    )


@torch.no_grad()
def compare_once_split(
    model,
    ort_vision,ort_proj,
    ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
    onnx_inputs,
    inputs
):

    # =========================
    # 🔹 Torch forward
    # =========================
    pt_out = model.model(**inputs)
    pt_hidden = pt_out.last_hidden_state

    # =========================
    # 🔹 ONNX forward
    # =========================
    image_embeds = run_onnx_vision(
        ort_vision,
        onnx_inputs["pixel_values"],
        onnx_inputs["image_position_ids"]
    )

    image_embeds = run_onnx_mmproj(ort_proj, image_embeds)

    cur_len = int(onnx_inputs["attention_mask"].sum())
    onnx_hidden, _ = run_onnx_llm_split(
        ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
        onnx_inputs, image_embeds, cur_len,
    )

    cur_len = int(onnx_inputs["attention_mask"].sum())
    pt_hidden = pt_hidden[:, :cur_len].cpu().numpy()
    onnx_hidden = onnx_hidden[:, :cur_len]

    print_align_metrics(pt_hidden, onnx_hidden)

# ===============================
# main
# ===============================
def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    ).to(DEVICE).to(torch.float16).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # ===== ONNX =====
    ort_vision, ort_proj, ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head = load_onnx()

    # ===== 输入 =====
    inputs = preprocess(processor)
    onnx_inputs = get_onnx_inputs(model,inputs)

    print("\n🔍 ONNX 对齐检查")
    compare_once_split(
        model,
        ort_vision, ort_proj,
        ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
        onnx_inputs,
        inputs,
    )

    onnx_ids = generate_onnx(
        ort_vision, ort_proj,
        ort_pre, ort_b1, ort_b2, ort_b3, ort_b4, ort_b5, ort_b6, ort_b7, ort_lm_head,
        model,
        onnx_inputs,
        steps=100,
    )[0]

    onnx_text = processor.decode(
        onnx_ids,
        skip_special_tokens=False
    )

    print("\n🤖 ONNX:")
    print(onnx_text)

    # ===============================
    # Torch 生成
    # ===============================
    input_len = inputs["input_ids"].shape[-1]
    torch_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )[0]

    torch_text = processor.tokenizer.decode(
        torch_ids[input_len:],
        skip_special_tokens=False
    )

    print("\n🧠 Torch:")
    print(torch_text)

if __name__ == "__main__":
    main()