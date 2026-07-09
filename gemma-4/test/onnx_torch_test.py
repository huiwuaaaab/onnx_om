import copy

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ===============================
# 配置
# ===============================
MODEL_PATH = "./gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "./gemma-4-E2B-it-assistant"

DEVICE = "cpu"
USE_ONNX_CUDA = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ===============================
# 加载 ONNX
# ===============================
def load_onnx():
    providers = ["CUDAExecutionProvider"] if USE_ONNX_CUDA else ["CPUExecutionProvider"]
    base = "/e-vepfs-01/ppdc/guanxj/ENetQuery/work_dirs/gemma4/onnx_export"

    vision = ort.InferenceSession(f"{base}/vision.onnx", providers=providers)
    mm_proj = ort.InferenceSession(f"{base}/mm_proj.onnx", providers=providers)
    ort_pre = ort.InferenceSession(f"{base}/llm_preblock.onnx", providers=providers)
    ort_b1 = ort.InferenceSession(f"{base}/llm_block_0_5.onnx", providers=providers)
    ort_b2 = ort.InferenceSession(f"{base}/llm_block_5_10.onnx", providers=providers)
    ort_b3 = ort.InferenceSession(f"{base}/llm_block_10_15.onnx", providers=providers)
    ort_b4 = ort.InferenceSession(f"{base}/llm_block_15_20.onnx", providers=providers)
    ort_b5 = ort.InferenceSession(f"{base}/llm_block_20_25.onnx", providers=providers)
    ort_b6 = ort.InferenceSession(f"{base}/llm_block_25_30.onnx", providers=providers)
    ort_b7 = ort.InferenceSession(f"{base}/llm_block_30_35.onnx", providers=providers)
    ort_lm_head = ort.InferenceSession(f"{base}/lm_head.onnx", providers=providers)
    ort_assistant = ort.InferenceSession(f"{base}/assistant.onnx", providers=providers)

    print(f"ONNX Provider: {providers}")
    return (
        vision,
        mm_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        ort_assistant,
    )


# ===============================
# 预处理
# ===============================
def preprocess(processor):
    image = Image.open(
        "/e-vepfs-01/perception/wuhui/InternVL3_5-1B/InternVL3_5-1B-HF/examples/image1.jpg"
    ).convert("RGB").resize((768, 768))

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
    ).to(DEVICE).to(torch.float16)

    return inputs


@torch.no_grad()
def get_onnx_inputs(model, inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs["image_position_ids"].to(torch.int32)

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

    return {
        "pixel_values": pixel_values.cpu().numpy(),
        "image_position_ids": image_position_ids.cpu().numpy(),
        "input_ids": input_ids_pad.to(torch.int32).cpu().numpy(),
        "attention_mask": attention_mask_pad.to(torch.int32).cpu().numpy(),
        "per_layer_inputs": per_layer_inputs.cpu().numpy().astype(np.float16),
        "position_ids": np.arange(max_len, dtype=np.int32).reshape(1, max_len),
    }


def _onnx_float_dtype(session: ort.InferenceSession) -> np.dtype:
    elem_type = session.get_inputs()[0].type
    return np.float16 if "float16" in elem_type else np.float32


def _cast_float_array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype in (np.float16, np.float32, np.float64):
        return arr.astype(dtype, copy=False)
    return arr


def run_onnx_vision(vision_sess, pixel_values, image_position_ids):
    return vision_sess.run(
        None,
        {"pixel_values": pixel_values, "image_position_ids": image_position_ids},
    )[0]


def run_onnx_mmproj(proj_sess, image_embeds):
    return proj_sess.run(None, {"vision_features": image_embeds})[0]


def _run_lm_head_range(ort_lm_head, hidden, start_pos: int, count: int, dtype: np.dtype) -> np.ndarray:
    """对 hidden[:, start:start+count] 逐 token 跑 lm_head，拼成 [1, count, vocab]。"""
    if count <= 0:
        raise ValueError("logits count must be positive")
    chunks = []
    for i in range(count):
        pos = start_pos + i
        h = hidden[:, pos : pos + 1, :].astype(dtype, copy=False)
        chunks.append(ort_lm_head.run(None, {"hidden_states": h})[0])
    return np.concatenate(chunks, axis=1)


def run_onnx_llm_split(
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    ort_lm_head,
    onnx_inputs,
    image_embeds,
    logits_pos_start=None,
    logits_pos_count=0,
):
    block_dtype = _onnx_float_dtype(ort_b1)

    pre_inputs = {
        "input_ids": onnx_inputs["input_ids"],
        "image_embeds": image_embeds,
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

    # layer 15+ 共享 KV：各 block 读同一组 full_k/v、slide_k/v，不在 block 间更新
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

    logits = None
    if logits_pos_start is not None and logits_pos_count > 0:
        logits = _run_lm_head_range(
            ort_lm_head, hidden, logits_pos_start, logits_pos_count, block_dtype
        )

    return hidden, logits, full_k, full_v, slide_k, slide_v


def run_assistant_split(
    ort_assistant, last_hidden, last_token_id, position_ids, attention_mask, full_k, full_v, slide_k, slide_v
):
    assistant_inputs = {
        "last_token_id": last_token_id,
        "last_hidden": last_hidden,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "full_k": full_k,
        "full_v": full_v,
        "slide_k": slide_k,
        "slide_v": slide_v,
    }
    projected_state, logits, _ = ort_assistant.run(None, assistant_inputs)
    return projected_state, logits


def print_align_metrics(ref, pred, name: str = "hidden") -> None:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    diff = np.abs(ref - pred)
    cos = float(np.dot(ref, pred) / (np.linalg.norm(ref) * np.linalg.norm(pred) + 1e-12))
    mean_abs = float(diff.mean())
    p99_abs = float(np.percentile(diff, 99))
    max_abs = float(diff.max())
    global_rel_pct = 100.0 * diff.sum() / (np.abs(ref).sum() + 1e-12)

    print(
        f"[{name}] cosine={cos:.6f}  "
        f"mean_abs={mean_abs:.6f}  p99_abs={p99_abs:.6f}  max_abs={max_abs:.4f}  "
        f"global_rel={global_rel_pct:.2f}%  (n={len(ref)})"
    )


def _get_eos_token_ids(model) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.model.config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.model.config.text_config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return {int(x) for x in eos}
    return {int(eos)}


def _token_is_eos(token: torch.Tensor, eos_ids: set[int]) -> bool:
    if not eos_ids:
        return False
    return int(token.reshape(-1)[0].item()) in eos_ids


def _update_per_layer_inputs(model, onnx_inputs):
    input_ids_torch = torch.from_numpy(onnx_inputs["input_ids"]).to(model.device)
    image_mask, _, _ = model.model.get_placeholder_mask(input_ids_torch)
    pad_id = model.model.config.text_config.pad_token_id
    pad_emb = model.model.language_model.embed_tokens.weight[pad_id]
    input_ids_torch = input_ids_torch.clone()
    input_ids_torch[image_mask] = pad_id
    embeds = model.model.get_input_embeddings()(input_ids_torch)
    llm_embeds = torch.where(image_mask[..., None], pad_emb.view(1, 1, -1), embeds)
    ple = model.model.language_model.get_per_layer_inputs(input_ids_torch, llm_embeds)
    onnx_inputs["per_layer_inputs"] = ple.cpu().numpy().astype(np.float16)


@torch.no_grad()
def generate_onnx(
    ort_vision,
    ort_proj,
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    ort_lm_head,
    ort_assistant,
    model,
    onnx_inputs,
    steps=50,
    num_assistant_tokens=6,
):
    image_embeds = run_onnx_mmproj(
        ort_proj,
        run_onnx_vision(ort_vision, onnx_inputs["pixel_values"], onnx_inputs["image_position_ids"]),
        
    )

    cur_len = int(onnx_inputs["attention_mask"].sum())
    generated_tokens = []
    max_len = 512
    eos_ids = _get_eos_token_ids(model)
    stop_generation = False

    hidden, _, full_k, full_v, slide_k, slide_v = run_onnx_llm_split(
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        onnx_inputs,
        image_embeds,
    )

    for _ in range(steps):
        if cur_len >= max_len or stop_generation:
            break

        candidate_tokens = []
        last_hidden = hidden[:, cur_len - 1 : cur_len]
        last_token_id = onnx_inputs["input_ids"][:, cur_len - 1 : cur_len]
        attn = onnx_inputs["attention_mask"].astype(np.int32)
        position_ids = np.array([[cur_len - 1]], dtype=np.int32)
        ass_fk, ass_fv = full_k.copy(), full_v.copy()
        ass_sk, ass_sv = slide_k.copy(), slide_v.copy()

        for k_idx in range(num_assistant_tokens):
            pos = cur_len + k_idx
            if pos >= max_len:
                break

            ass_hidden, ass_logits = run_assistant_split(
                ort_assistant,
                last_hidden,
                last_token_id,
                position_ids,
                attn,
                ass_fk,
                ass_fv,
                ass_sk,
                ass_sv,
            )
            pred_token = torch.argmax(torch.from_numpy(ass_logits), dim=-1)
            candidate_tokens.append(pred_token)
            last_hidden = ass_hidden
            last_token_id = pred_token.to(torch.int32).cpu().numpy()

            if _token_is_eos(pred_token, eos_ids):
                break

        if not candidate_tokens:
            break

        temp_in_ids = onnx_inputs["input_ids"].copy()
        temp_attn = onnx_inputs["attention_mask"].copy()
        cand_tensor = torch.cat(candidate_tokens, dim=1)
        cand_len = cand_tensor.shape[1]

        for i in range(cand_len):
            temp_in_ids[:, cur_len + i] = cand_tensor[:, i].cpu().numpy()
            temp_attn[:, cur_len + i] = 1

        temp_onnx_in = copy.deepcopy(onnx_inputs)
        temp_onnx_in["input_ids"] = temp_in_ids
        temp_onnx_in["attention_mask"] = temp_attn
        _update_per_layer_inputs(model, temp_onnx_in)

        verify_hidden, verify_logits, v_fk, v_fv, v_sk, v_sv = run_onnx_llm_split(
            ort_pre,
            ort_b1,
            ort_b2,
            ort_b3,
            ort_b4,
            ort_b5,
            ort_b6,
            ort_b7,
            ort_lm_head,
            temp_onnx_in,
            image_embeds,
            logits_pos_start=cur_len - 1,
            logits_pos_count=cand_len,
        )

        main_pred = torch.argmax(torch.from_numpy(verify_logits), dim=-1)

        match_flag = cand_tensor == main_pred
        match_cnt = int((~match_flag).cumsum(dim=1).lt(1).sum())
        accept_tokens = main_pred[:, : match_cnt + 1]
        accept_num = accept_tokens.shape[1]

        generated_tokens.append(accept_tokens)
        hidden = verify_hidden
        full_k, full_v, slide_k, slide_v = v_fk, v_fv, v_sk, v_sv

        onnx_inputs["input_ids"][:, cur_len : cur_len + accept_num] = accept_tokens.to(torch.int32).cpu().numpy()
        onnx_inputs["attention_mask"][:, cur_len : cur_len + accept_num] = 1
        cur_len += accept_num
        _update_per_layer_inputs(model, onnx_inputs)

        if eos_ids:
            for i in range(accept_num):
                if int(accept_tokens[0, i].item()) in eos_ids:
                    stop_generation = True
                    break

    return torch.cat(generated_tokens, dim=1) if generated_tokens else torch.empty(1, 0, dtype=torch.long)


@torch.no_grad()
def compare_once_split(
    model,
    assistant_model,
    ort_vision,
    ort_proj,
    ort_pre,
    ort_b1,
    ort_b2,
    ort_b3,
    ort_b4,
    ort_b5,
    ort_b6,
    ort_b7,
    ort_lm_head,
    ort_assistant,
    onnx_inputs,
    inputs,
):
    pt_out = model.model(**inputs, return_shared_kv_states=True)
    pt_hidden = pt_out.last_hidden_state
    pt_kv = pt_out.shared_kv_states

    cur_len_pt = int(inputs["attention_mask"].sum().item())
    last_hidden_pt = pt_hidden[:, -1:, :]
    last_emb = model.model.get_input_embeddings()(inputs["input_ids"][:, -1:])
    inputs_embeds = torch.cat([last_emb, last_hidden_pt], dim=-1)
    position_ids = torch.tensor([[inputs["input_ids"].shape[1] - 1]], device=DEVICE)

    assistant_out = assistant_model(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs["attention_mask"],
        position_ids=position_ids,
        shared_kv_states=pt_kv,
        use_cache=False,
    ).last_hidden_state

    image_embeds = run_onnx_mmproj(
        ort_proj,
        run_onnx_vision(ort_vision, onnx_inputs["pixel_values"], onnx_inputs["image_position_ids"]),
    )
    hidden, _, full_k, full_v, slide_k, slide_v = run_onnx_llm_split(
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        onnx_inputs,
        image_embeds,
    )

    cur_len = int(onnx_inputs["attention_mask"].sum())
    last_token_id = onnx_inputs["input_ids"][:, cur_len - 1 : cur_len]
    last_hidden = hidden[:, cur_len - 1 : cur_len, :]
    position_ids_np = np.array([[cur_len - 1]], dtype=np.int32)

    assistant_proj, _ = run_assistant_split(
        ort_assistant,
        last_hidden,
        last_token_id,
        position_ids_np,
        onnx_inputs["attention_mask"],
        full_k,
        full_v,
        slide_k,
        slide_v,
    )

    print("\nONNX 全量对齐检查（不排除近零维）")
    print_align_metrics(
        pt_hidden[:, :cur_len].cpu().numpy(),
        hidden[:, :cur_len],
        name="main_hidden",
    )
    print_align_metrics(
        assistant_out.cpu().numpy(),
        assistant_proj,
        name="assistant_projected",
    )


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).to(DEVICE).to(torch.float16).eval()

    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).to(DEVICE).to(torch.float16).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ort_sessions = load_onnx()

    inputs = preprocess(processor)
    onnx_inputs = get_onnx_inputs(model, inputs)

    (
        ort_vision,
        ort_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        ort_assistant,
    ) = ort_sessions

    compare_once_split(
        model,
        assistant_model,
        ort_vision,
        ort_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        ort_assistant,
        onnx_inputs,
        inputs,
    )

    onnx_ids = generate_onnx(
        ort_vision,
        ort_proj,
        ort_pre,
        ort_b1,
        ort_b2,
        ort_b3,
        ort_b4,
        ort_b5,
        ort_b6,
        ort_b7,
        ort_lm_head,
        ort_assistant,
        model,
        onnx_inputs,
        steps=100,
    )[0]
    onnx_text = processor.decode(onnx_ids, skip_special_tokens=False)
    print("\nONNX:")
    print(onnx_text)

    input_len = inputs["input_ids"].shape[-1]
    torch_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        assistant_model=assistant_model,
        do_sample=False,
    )[0]
    torch_text = processor.tokenizer.decode(torch_ids[input_len:], skip_special_tokens=False)
    print("\nTorch:")
    print(torch_text)


if __name__ == "__main__":
    main()
