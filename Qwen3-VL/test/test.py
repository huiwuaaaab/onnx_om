import torch
from transformers import AutoProcessor

from onnx_common import (
    IMAGE_PATH,
    MAX_SEQ_LEN,
    PROMPT,
    MODEL_PATH,
    build_onnx_inputs,
    build_preblock_onnx_inputs,
    load_onnx_chain,
    preprocess,
    run_lm_head,
    run_onnx_vision,
)

# Re-export for scripts that import constants from test.py
__all__ = [
    "IMAGE_PATH",
    "MAX_SEQ_LEN",
    "PROMPT",
    "build_onnx_inputs",
    "build_preblock_onnx_inputs",
    "generate",
    "load_onnx",
    "main",
    "preprocess",
    "run_llm",
    "run_vision",
]


def load_onnx():
    return load_onnx_chain(include_vision=True)


def run_vision(vision_sess, pixel_values):
    return run_onnx_vision(vision_sess, pixel_values)


def run_llm(
    pre,
    b1,
    b2,
    b3,
    head,
    input_ids,
    image_embeds,
    attention_mask,
    deepstack,
    cur_len,
):
    pre_inputs = build_preblock_onnx_inputs(input_ids, image_embeds, attention_mask)
    inputs_embeds, attention_mask, cos, sin = pre.run(None, pre_inputs)

    hidden = inputs_embeds
    for block, ds_slice in ((b1, deepstack), (b2, deepstack), (b3, deepstack)):
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
                ds_slice,
            ),
        )[0]

    logits = run_lm_head(head, hidden, cur_len)
    return hidden, logits


def generate(
    pre,
    b1,
    b2,
    b3,
    head,
    input_ids,
    attention_mask,
    image_embeds,
    deepstack,
    steps=50,
):
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()

    cur_len = int(attention_mask.sum())
    generated = []

    for _ in range(steps):
        _, logits = run_llm(
            pre,
            b1,
            b2,
            b3,
            head,
            input_ids,
            image_embeds,
            attention_mask,
            deepstack,
            cur_len,
        )

        next_token = torch.argmax(
            torch.from_numpy(logits[:, -1, :]),
            dim=-1,
            keepdim=True,
        )
        generated.append(next_token)

        input_ids[:, cur_len] = next_token[:, 0]
        attention_mask[:, cur_len] = 1
        cur_len += 1

        if cur_len >= MAX_SEQ_LEN:
            break

    return torch.cat(generated, dim=1)


def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    vision, pre, b1, b2, b3, head = load_onnx()
    inputs = preprocess(processor)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_embeds, deepstack = run_vision(vision, inputs["pixel_values"])

    tokens = generate(
        pre,
        b1,
        b2,
        b3,
        head,
        input_ids,
        attention_mask,
        image_embeds,
        deepstack,
        steps=50,
    )

    text = processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
