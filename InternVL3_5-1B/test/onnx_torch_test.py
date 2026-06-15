import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoProcessor

from onnx_common import (
    DEVICE,
    IMAGE_PATH,
    MAX_SEQ_LEN,
    MODEL_PATH,
    PAD_TOKEN_ID,
    apply_fp16_inputs,
    as_fp16_numpy,
    as_int32_numpy,
    get_providers,
    load_fp16_model,
    make_preblock_position_ids,
    onnx_path,
    pad,
    run_lm_head,
)

USE_ONNX_CUDA = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = load_fp16_model()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.image_processor.min_patches = 1
processor.image_processor.max_patches = 1
providers = get_providers(USE_ONNX_CUDA)


def load_onnx():
    vision = ort.InferenceSession(onnx_path("vision_448_notchunk.onnx"), providers=providers)
    mm_proj = ort.InferenceSession(onnx_path("mm_proj.onnx"), providers=providers)
    pre = ort.InferenceSession(onnx_path("llm_preblock.onnx"), providers=providers)
    b1 = ort.InferenceSession(onnx_path("llm_block1.onnx"), providers=providers)
    b2 = ort.InferenceSession(onnx_path("llm_block2.onnx"), providers=providers)
    b3 = ort.InferenceSession(onnx_path("llm_block3.onnx"), providers=providers)
    head = ort.InferenceSession(onnx_path("lm_head.onnx"), providers=providers)
    print(f"🚀 ONNX Provider: {providers}")
    return vision, mm_proj, pre, b1, b2, b3, head


def preprocess():
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

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)
    return apply_fp16_inputs(inputs)


@torch.no_grad()
def get_llm_inputs(inputs, image_embeds):
    input_ids, attention_mask = pad(
        inputs["input_ids"],
        inputs["attention_mask"],
        pad_id=PAD_TOKEN_ID,
        max_len=MAX_SEQ_LEN,
    )
    return {
        "input_ids": input_ids,
        "image_embeds": as_fp16_numpy(image_embeds),
        "attention_mask": attention_mask,
    }


def run_onnx_vision(vision_sess, pixel_values):
    image_embeds = vision_sess.run(
        None,
        {"pixel_values": as_fp16_numpy(pixel_values)},
    )[0]
    return as_fp16_numpy(image_embeds)


def run_onnx_mmproj(proj_sess, image_embeds):
    proj_embeds = proj_sess.run(
        None,
        {"vision_features": as_fp16_numpy(image_embeds)},
    )[0]
    return as_fp16_numpy(proj_embeds)


def run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs):
    pre_inputs = {
        "image_embeds": as_fp16_numpy(llm_inputs["image_embeds"]),
        "attention_mask": as_int32_numpy(llm_inputs["attention_mask"]),
        "input_ids": as_int32_numpy(llm_inputs["input_ids"]),
        "position_ids": make_preblock_position_ids(MAX_SEQ_LEN),
    }

    hidden, attention_mask, cos, sin = ort_pre.run(None, pre_inputs)

    hidden = ort_b1.run(
        None,
        {
            "hidden_states": hidden,
            "attention_mask": attention_mask,
            "cos": cos,
            "sin": sin,
        },
    )[0]

    hidden = ort_b2.run(
        None,
        {
            "hidden_states": hidden,
            "attention_mask": attention_mask,
            "cos": cos,
            "sin": sin,
        },
    )[0]

    hidden = ort_b3.run(
        None,
        {
            "hidden_states": hidden,
            "attention_mask": attention_mask,
            "cos": cos,
            "sin": sin,
        },
    )[0]

    return hidden


@torch.no_grad()
def generate_onnx(ort_pre, ort_b1, ort_b2, ort_b3, ort_head, llm_inputs, steps=50):
    attention_mask = llm_inputs["attention_mask"]
    cur_len = int(attention_mask.sum().item())
    generated_tokens = []

    for _ in range(steps):
        hidden = run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs)
        logits = run_lm_head(ort_head, hidden, cur_len)

        next_token = torch.argmax(
            torch.from_numpy(logits.astype(np.float32))[:, 0, :],
            dim=-1,
            keepdim=True,
        )

        generated_tokens.append(next_token)
        llm_inputs["input_ids"][:, cur_len] = next_token[:, 0]
        llm_inputs["attention_mask"][:, cur_len] = 1
        cur_len += 1

        if cur_len >= MAX_SEQ_LEN:
            break

    tokens = torch.cat(generated_tokens, dim=1)
    return processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


@torch.no_grad()
def compare_once_split(model, ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs, inputs):
    pt_out = model.model(**inputs)
    pt_hidden = pt_out.last_hidden_state

    onnx_hidden = run_onnx_llm_split(ort_pre, ort_b1, ort_b2, ort_b3, llm_inputs)

    attention_mask = inputs["attention_mask"][0].cpu()
    valid_length = int(attention_mask.sum().item())
    onnx_hidden = onnx_hidden[:, :valid_length]

    pt_np = pt_hidden.detach().cpu().numpy().astype(np.float32)
    onnx_np = onnx_hidden.astype(np.float32)
    diff = np.abs(pt_np - onnx_np)
    print(f"MAX:  {diff.max():.6f}")
    print(f"MEAN: {diff.mean():.6f}")


def main():
    vision, mm_proj, pre, b1, b2, b3, head = load_onnx()
    inputs = preprocess()

    image_embeds = run_onnx_vision(vision, inputs["pixel_values"])
    proj_embeds = run_onnx_mmproj(mm_proj, image_embeds)
    llm_inputs = get_llm_inputs(inputs, proj_embeds)

    print("\n🔍 ONNX 对齐检查")
    compare_once_split(model, pre, b1, b2, b3, llm_inputs, inputs)

    onnx_text = generate_onnx(pre, b1, b2, b3, head, llm_inputs, steps=50)
    print("\n🤖 ONNX:")
    print(onnx_text)

    torch_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]
    torch_text = processor.tokenizer.decode(torch_ids, skip_special_tokens=True)
    print("\n🧠 Torch:")
    print(torch_text)


if __name__ == "__main__":
    main()
