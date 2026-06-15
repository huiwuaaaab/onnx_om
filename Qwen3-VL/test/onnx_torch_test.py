import argparse
import torch

from onnx_common import (
    EXPORT_DIR,
    IMAGE_PATH,
    IMAGE_TOKEN_END,
    IMAGE_TOKEN_START,
    MAX_SEQ_LEN,
    MODEL_PATH,
    ONNX_VISION,
    ORT_PROVIDERS,
    PROFILE,
    PROMPT,
    build_onnx_inputs,
    build_preblock_onnx_inputs,
    load_hf_qwen3_vl,
    load_onnx_chain,
    preprocess,
    run_lm_head,
    run_onnx_vision,
    to_fp16_inputs,
)
from transformers import AutoProcessor

DEVICE = "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_onnx():
    print(f"ONNX Provider: {ORT_PROVIDERS}")
    return load_onnx_chain(include_vision=True)


@torch.no_grad()
def compare_vision(model, inputs, image_embeds, deepstack):
    print("\n==============================")
    print("VISION 对齐")
    print("==============================")

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    device = pixel_values.device

    with torch.no_grad():
        vision_out = model.model.visual(pixel_values, image_grid_thw)

    pt_merged = vision_out.pooler_output.to(device)
    pt_ds = [d.to(device) for d in vision_out.deepstack_features]

    onnx_merged = torch.from_numpy(image_embeds).to(device=device, dtype=pt_merged.dtype)
    onnx_ds = [
        torch.from_numpy(deepstack[i]).to(device=device, dtype=pt_ds[i].dtype)
        for i in range(3)
    ]

    print("\nshape:")
    print("pt merged:", pt_merged.shape)
    print("onnx merged:", onnx_merged.shape)
    for i in range(3):
        print(f"pt ds_{i}:", pt_ds[i].shape)
        print(f"onnx ds_{i}:", onnx_ds[i].shape)

    diff = (pt_merged - onnx_merged).abs()
    print("\nmerged:")
    print("MAX :", diff.max().item())
    print("MEAN:", diff.mean().item())

    for i in range(3):
        pt = pt_ds[i]
        if pt.dim() == 3:
            pt = pt.squeeze(0)
        onnx = onnx_ds[i]
        diff = (pt - onnx).abs()
        print(f"\nds_{i}:")
        print("MAX :", diff.max().item())
        print("MEAN:", diff.mean().item())


@torch.no_grad()
def compare_preblock(model, inputs, pre, image_embeds):
    print("\n==============================")
    print("PREBLOCK 对齐")
    print("==============================")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = input_ids.device

    pt_embeds = model.model.get_input_embeddings()(input_ids)
    image_embeds_t = torch.from_numpy(image_embeds).to(pt_embeds.device, pt_embeds.dtype)

    image_mask, _ = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=pt_embeds,
        image_features=image_embeds_t,
    )
    pt_embeds = pt_embeds.masked_scatter(image_mask, image_embeds_t)

    pre_inputs = build_preblock_onnx_inputs(input_ids, image_embeds, attention_mask)
    position_ids = torch.from_numpy(pre_inputs["position_ids"]).to(device)
    pt_cos, pt_sin = model.model.language_model.rotary_emb(pt_embeds, position_ids)

    onnx_embeds, onnx_mask, onnx_cos, onnx_sin = pre.run(None, pre_inputs)

    onnx_cos = torch.from_numpy(onnx_cos).to(device)
    onnx_sin = torch.from_numpy(onnx_sin).to(device)

    def check(name, a, b, mask):
        while mask.ndim < a.ndim:
            mask = mask.unsqueeze(-1).expand(mask.shape[0], mask.shape[1], a.shape[2])
        valid = mask.to(dtype=torch.bool)
        diff_valid = (a - b).abs()[valid]
        print(f"\n{name} (only valid tokens):")
        print("MAX :", diff_valid.max().item())
        print("MEAN:", diff_valid.mean().item())

    check("cos", pt_cos, onnx_cos, attention_mask)
    check("sin", pt_sin, onnx_sin, attention_mask)


@torch.no_grad()
def compare_blocks(model, inputs, pre, b1, b2, b3, head, image_embeds, deepstack):
    print("\n==============================")
    print("BLOCK 对齐")
    print("==============================")

    device = inputs["input_ids"].device
    lm = model.model.language_model

    pre_inputs = build_preblock_onnx_inputs(
        inputs["input_ids"], image_embeds, inputs["attention_mask"]
    )
    onnx_embeds, onnx_mask, onnx_cos, onnx_sin = pre.run(None, pre_inputs)

    onnx_hidden = torch.from_numpy(onnx_embeds).to(device)
    onnx_mask = torch.from_numpy(onnx_mask).to(device)
    onnx_cos = torch.from_numpy(onnx_cos).to(device)
    onnx_sin = torch.from_numpy(onnx_sin).to(device)

    pt_hidden = onnx_hidden.clone()
    pt_mask = onnx_mask
    pt_cos = onnx_cos
    pt_sin = onnx_sin
    ds = [torch.from_numpy(d).to(device) for d in deepstack]

    for i in range(10):
        pt_hidden = lm.layers[i](
            pt_hidden,
            attention_mask=pt_mask,
            position_embeddings=(pt_cos, pt_sin),
        )
        if i == 5:
            pt_hidden[:, IMAGE_TOKEN_START:IMAGE_TOKEN_END, :] += ds[0].unsqueeze(0)

    b1_inputs = build_onnx_inputs(
        b1,
        {
            "hidden_states": onnx_hidden.cpu().numpy(),
            "attention_mask": onnx_mask.cpu().numpy(),
            "cos": onnx_cos.cpu().numpy(),
            "sin": onnx_sin.cpu().numpy(),
        },
        deepstack,
    )
    onnx_hidden_b1 = torch.from_numpy(b1.run(None, b1_inputs)[0]).to(device)
    diff = (pt_hidden - onnx_hidden_b1).abs()
    print("\nblock1:")
    print("MAX :", diff.max().item())
    print("MEAN:", diff.mean().item())

    onnx_hidden = onnx_hidden_b1.clone()
    for i in range(10, 20):
        pt_hidden = lm.layers[i](
            pt_hidden,
            attention_mask=pt_mask,
            position_embeddings=(pt_cos, pt_sin),
        )
        if i == 11:
            pt_hidden[:, IMAGE_TOKEN_START:IMAGE_TOKEN_END, :] += ds[1].unsqueeze(0)
        if i == 17:
            pt_hidden[:, IMAGE_TOKEN_START:IMAGE_TOKEN_END, :] += ds[2].unsqueeze(0)

    b2_inputs = build_onnx_inputs(
        b2,
        {
            "hidden_states": onnx_hidden.cpu().numpy(),
            "attention_mask": onnx_mask.cpu().numpy(),
            "cos": onnx_cos.cpu().numpy(),
            "sin": onnx_sin.cpu().numpy(),
        },
        deepstack,
    )
    onnx_hidden_b2 = torch.from_numpy(b2.run(None, b2_inputs)[0]).to(device)
    diff = (pt_hidden - onnx_hidden_b2).abs()
    print("\nblock2:")
    print("MAX :", diff.max().item())
    print("MEAN:", diff.mean().item())

    onnx_hidden = onnx_hidden_b2.clone()
    for i in range(20, len(lm.layers)):
        pt_hidden = lm.layers[i](
            pt_hidden,
            attention_mask=pt_mask,
            position_embeddings=(pt_cos, pt_sin),
        )
    pt_hidden = lm.norm(pt_hidden)

    b3_inputs = build_onnx_inputs(
        b3,
        {
            "hidden_states": onnx_hidden.cpu().numpy(),
            "attention_mask": onnx_mask.cpu().numpy(),
            "cos": onnx_cos.cpu().numpy(),
            "sin": onnx_sin.cpu().numpy(),
        },
        deepstack,
    )
    onnx_hidden_b3 = torch.from_numpy(b3.run(None, b3_inputs)[0]).to(device)
    diff = (pt_hidden - onnx_hidden_b3).abs()
    print("\nblock3:")
    print("MAX :", diff.max().item())
    print("MEAN:", diff.mean().item())

    cur_len = int(inputs["attention_mask"][0].sum())
    pt_last_logits = model.lm_head(pt_hidden[:, cur_len - 1:cur_len, :])
    onnx_last_logits = torch.from_numpy(
        run_lm_head(head, onnx_hidden_b3.cpu().numpy(), cur_len)
    ).to(device)

    diff = (pt_last_logits - onnx_last_logits).abs()
    print("\nlogits:")
    print("MAX :", diff.max().item())
    print("MEAN:", diff.mean().item())


def run_onnx_llm(pre, b1, b2, b3, head, input_ids, image_embeds, attention_mask, deepstack, cur_len):
    pre_inputs = build_preblock_onnx_inputs(input_ids, image_embeds, attention_mask)
    inputs_embeds, attention_mask, cos, sin = pre.run(None, pre_inputs)
    hidden = inputs_embeds

    for block in (b1, b2, b3):
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

    logits = run_lm_head(head, hidden, cur_len)
    return torch.from_numpy(hidden), torch.from_numpy(logits)


@torch.no_grad()
def generate_onnx(pre, b1, b2, b3, head, input_ids, attention_mask, image_embeds, deepstack, steps=50):
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()
    cur_len = int(attention_mask.sum())
    generated = []

    for _ in range(steps):
        _, logits = run_onnx_llm(
            pre, b1, b2, b3, head,
            input_ids, image_embeds, attention_mask, deepstack, cur_len,
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)
        input_ids[:, cur_len] = next_token[:, 0]
        attention_mask[:, cur_len] = 1
        cur_len += 1
        if cur_len >= MAX_SEQ_LEN:
            break

    return torch.cat(generated, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL ONNX vs PyTorch alignment test")
    parser.add_argument(
        "--image",
        default=IMAGE_PATH,
        help=f"input image path (default: IMAGE_PATH or QWEN3_IMAGE_PATH env, currently {IMAGE_PATH!r})",
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT,
        help=f"user prompt (default: PROMPT or QWEN3_PROMPT env, currently {PROMPT!r})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"profile: {PROFILE.name}  image_size={PROFILE.image_size}  max_seq_len={MAX_SEQ_LEN}")
    print(f"onnx: {EXPORT_DIR}/{ONNX_VISION}")
    print(f"image: {args.image}")
    print(f"prompt: {args.prompt!r}")

    model = load_hf_qwen3_vl(MODEL_PATH).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    vision, pre, b1, b2, b3, head = load_onnx()

    inputs = to_fp16_inputs(preprocess(processor, image_path=args.image, prompt=args.prompt))
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_embeds, deepstack = run_onnx_vision(vision, inputs["pixel_values"])

    compare_vision(model, inputs, image_embeds, deepstack)
    compare_preblock(model, inputs, pre, image_embeds)
    compare_blocks(model, inputs, pre, b1, b2, b3, head, image_embeds, deepstack)

    tokens = generate_onnx(
        pre, b1, b2, b3, head,
        input_ids.clone(), attention_mask.clone(),
        image_embeds, deepstack, steps=50,
    )
    print("\nONNX:")
    print(processor.tokenizer.decode(tokens[0], skip_special_tokens=True))

    # HF generate 在 input_ids 已 pad 到 MAX_SEQ_LEN 时会从 pad 区续写，argmax 全是 pad_token。
    # ONNX 链每步用完整 mask 没问题；Torch 需先截到 cur_len 再 generate。
    cur_len = int(attention_mask[0].sum().item())
    gen_kwargs = {
        "input_ids": inputs["input_ids"][:, :cur_len],
        "attention_mask": inputs["attention_mask"][:, :cur_len],
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": 50,
        "do_sample": False,
    }
    if "image_grid_thw" in inputs:
        gen_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    if "mm_token_type_ids" in inputs:
        gen_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, :cur_len]

    torch_ids = model.generate(**gen_kwargs)[0]
    new_tokens = torch_ids[cur_len:]
    print("\nTorch:")
    print(processor.tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
