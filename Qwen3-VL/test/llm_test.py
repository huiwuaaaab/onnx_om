import time

import numpy as np
import torch

from onnx_common import (
    FLOAT_DTYPE,
    MAX_SEQ_LEN,
    MODEL_PATH,
    ORT_PROVIDERS,
    build_onnx_inputs,
    build_preblock_onnx_inputs,
    load_hf_qwen3_vl,
    load_onnx_chain,
    preprocess,
    run_lm_head,
    to_deepstack_np,
)
from transformers import AutoProcessor

DEVICE = "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_onnx_sessions():
    print(f"ONNX Provider: {ORT_PROVIDERS}")
    _, pre, b1, b2, b3, head = load_onnx_chain(include_vision=False)
    return pre, b1, b2, b3, head


@torch.no_grad()
def get_llm_inputs(model, inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    inputs_embeds = model.model.get_input_embeddings()(input_ids)

    image_outputs = model.model.get_image_features(pixel_values, image_grid_thw)
    if isinstance(image_outputs, tuple):
        image_embeds_list, deepstack_image_embeds = image_outputs
    else:
        image_embeds_list = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features
    image_embeds = torch.cat(image_embeds_list, dim=0).to(
        inputs_embeds.device, inputs_embeds.dtype
    )

    image_mask, _ = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    visual_pos_masks = image_mask[..., 0]

    attention_mask_tensor = attention_mask
    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
        attention_mask_tensor = torch.diagonal(
            attention_mask_tensor[:, 0], dim1=1, dim2=2
        )
        if attention_mask_tensor.dtype.is_floating_point:
            attention_mask_tensor = (
                attention_mask_tensor
                / torch.finfo(attention_mask_tensor.dtype).min
            )
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

    position_ids, _ = model.model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask_tensor,
    )

    return {
        "input_ids": input_ids,
        "image_embeds": image_embeds,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_image_embeds,
    }


def print_llm_inputs(llm_inputs):
    print("\nLLM Inputs:")
    for k, v in llm_inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {list(v.shape)} dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{k}: list(len={len(v)})")
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f"  └─ {k}[{i}]: {list(t.shape)} dtype={t.dtype}")


def run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs):
    pre_inputs = build_preblock_onnx_inputs(
        llm_inputs["input_ids"],
        llm_inputs["image_embeds"].cpu().numpy(),
        llm_inputs["attention_mask"],
    )
    hidden, attention_mask, cos, sin = ort_pre.run(None, pre_inputs)
    deepstack = to_deepstack_np(llm_inputs["deepstack_visual_embeds"])

    for block in (ort_b1, ort_b2, ort_b3):
        hidden = block.run(
            None,
            build_onnx_inputs(
                block,
                {
                    "hidden_states": hidden,
                    "attention_mask": attention_mask,
                    "cos": cos,
                    "sin": sin,
                },
                deepstack,
            ),
        )[0]

    return hidden


def run_onnx_llm_with_head(ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs, cur_len):
    hidden = run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs)
    logits = run_lm_head(ort_head, hidden, cur_len)
    return hidden, logits


@torch.no_grad()
def run_torch_generate(model, inputs, processor, max_new_tokens=50):
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


@torch.no_grad()
def generate_onnx_padding_split(ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs, steps=50):
    input_ids = llm_inputs["input_ids"].clone()
    attention_mask = llm_inputs["attention_mask"].clone()
    image_embeds = llm_inputs["image_embeds"]
    deepstack = llm_inputs["deepstack_visual_embeds"]

    cur_len = int(attention_mask.sum().item())
    generated_tokens = []

    for _ in range(steps):
        step_inputs = {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "attention_mask": attention_mask,
            "deepstack_visual_embeds": deepstack,
        }
        _, logits = run_onnx_llm_with_head(
            ort_pre, ort_b1, ort_b2, ort_b3, ort_head, step_inputs, cur_len,
        )
        next_token = torch.argmax(torch.from_numpy(logits[:, -1, :]), dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        input_ids[:, cur_len] = next_token[:, 0].to(input_ids.dtype)
        attention_mask[:, cur_len] = 1
        cur_len += 1
        if cur_len >= MAX_SEQ_LEN:
            print("达到最大长度")
            break

    return torch.cat(generated_tokens, dim=1)


def decode_tokens(processor, tokens):
    return processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def clean_text(text):
    return text.split("assistant")[-1].strip()


def clean_onnx_output(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if line.strip() == "assistant":
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


@torch.no_grad()
def compare_once_split(model, ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs):
    device = llm_inputs["inputs_embeds"].device

    pt_out = model.model.language_model(
        input_ids=None,
        inputs_embeds=llm_inputs["inputs_embeds"],
        attention_mask=llm_inputs["attention_mask"],
        position_ids=llm_inputs["position_ids"],
        visual_pos_masks=llm_inputs["visual_pos_masks"],
        deepstack_visual_embeds=llm_inputs["deepstack_visual_embeds"],
    ).last_hidden_state

    onnx_out = run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs)
    diff = np.abs(pt_out.cpu().numpy().astype(np.float32) - onnx_out.astype(np.float32))
    print("\nSplit ONNX hidden 对齐误差：")
    print(f"MAX:  {diff.max():.6f}")
    print(f"MEAN: {diff.mean():.6f}")

    cur_len = int(llm_inputs["attention_mask"][0].sum().item())
    pt_logits = model.lm_head(pt_out[:, cur_len - 1:cur_len, :].to(FLOAT_DTYPE))
    onnx_logits = torch.from_numpy(run_lm_head(ort_head, onnx_out, cur_len)).to(device)
    logit_diff = (pt_logits - onnx_logits).abs()
    print("\nlm_head logits 对齐误差：")
    print(f"MAX:  {logit_diff.max().item():.6f}")
    print(f"MEAN: {logit_diff.mean().item():.6f}")


if __name__ == "__main__":
    model = load_hf_qwen3_vl(MODEL_PATH).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ort_pre, ort_b1, ort_b2, ort_b3, ort_head = load_onnx_sessions()

    inputs = preprocess(processor)
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(FLOAT_DTYPE)

    print("\n构造 LLM 输入...")
    llm_inputs = get_llm_inputs(model, inputs)
    print_llm_inputs(llm_inputs)

    print("\nTorch Generate...")
    torch_text = run_torch_generate(model, inputs, processor, max_new_tokens=50)
    print("\nTorch 输出:")
    print(clean_text(torch_text))

    print("\nONNX Generate...")
    onnx_tokens = generate_onnx_padding_split(
        ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs, steps=50,
    )
    print(f"generated tokens: {onnx_tokens.shape}")
    print("\nONNX 输出:")
    print(clean_onnx_output(decode_tokens(processor, onnx_tokens)))

    print("\nONNX 对齐检查")
    compare_once_split(model, ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs)
