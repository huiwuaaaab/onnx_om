import torch
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from onnx_common import (
    FLOAT_DTYPE,
    IMAGE_PATH,
    MAX_SEQ_LEN,
    MODEL_PATH,
    PAD_TOKEN_ID,
    as_fp16_numpy,
    as_int32_numpy,
    make_preblock_position_ids,
    onnx_path,
    pad,
    run_lm_head,
)


class InternVLONNX:

    def __init__(
        self,
        model_path=MODEL_PATH,
        vision_path=None,
        proj_path=None,
        pre_path=None,
        block1_path=None,
        block2_path=None,
        block3_path=None,
        providers=("CPUExecutionProvider",),
        max_length=MAX_SEQ_LEN,
    ):

        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.min_patches = 1
        self.processor.image_processor.max_patches = 1

        self.vision = ort.InferenceSession(
            vision_path or onnx_path("vision_448_notchunk.onnx"),
            providers=list(providers),
        )
        self.mm_proj = ort.InferenceSession(
            proj_path or onnx_path("mm_proj.onnx"),
            providers=list(providers),
        )
        self.pre = ort.InferenceSession(
            pre_path or onnx_path("llm_preblock.onnx"),
            providers=list(providers),
        )
        self.b1 = ort.InferenceSession(
            block1_path or onnx_path("llm_block1.onnx"),
            providers=list(providers),
        )
        self.b2 = ort.InferenceSession(
            block2_path or onnx_path("llm_block2.onnx"),
            providers=list(providers),
        )
        self.b3 = ort.InferenceSession(
            block3_path or onnx_path("llm_block3.onnx"),
            providers=list(providers),
        )
        self.lm_head = ort.InferenceSession(
            onnx_path("lm_head.onnx"),
            providers=list(providers),
        )

    def build_inputs(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=FLOAT_DTYPE)
        return inputs

    def pad_inputs(self, input_ids, attention_mask, pad_id=PAD_TOKEN_ID):
        return pad(input_ids, attention_mask, pad_id=pad_id, max_len=self.max_length)

    def encode_image(self, pixel_values):
        image_embeds = self.vision.run(
            None,
            {"pixel_values": as_fp16_numpy(pixel_values)},
        )[0]

        image_embeds = self.mm_proj.run(
            None,
            {"vision_features": as_fp16_numpy(image_embeds)},
        )[0]

        return as_fp16_numpy(image_embeds)

    def llm_forward(self, input_ids, attention_mask, image_embeds):
        hidden, attn_mask, cos, sin = self.pre.run(
            None,
            {
                "image_embeds": as_fp16_numpy(image_embeds),
                "attention_mask": as_int32_numpy(attention_mask),
                "input_ids": as_int32_numpy(input_ids),
                "position_ids": make_preblock_position_ids(self.max_length),
            },
        )

        hidden = self.b1.run(
            None,
            {
                "hidden_states": hidden,
                "attention_mask": attn_mask,
                "cos": cos,
                "sin": sin,
            },
        )[0]

        hidden = self.b2.run(
            None,
            {
                "hidden_states": hidden,
                "attention_mask": attn_mask,
                "cos": cos,
                "sin": sin,
            },
        )[0]

        hidden = self.b3.run(
            None,
            {
                "hidden_states": hidden,
                "attention_mask": attn_mask,
                "cos": cos,
                "sin": sin,
            },
        )[0]

        cur_len = int(attention_mask.sum().item())
        return run_lm_head(self.lm_head, hidden, cur_len)

    @torch.no_grad()
    def generate(self, image_path, prompt, max_new_tokens=50):
        inputs = self.build_inputs(image_path, prompt)
        image_embeds = self.encode_image(inputs["pixel_values"])

        input_ids, attention_mask = self.pad_inputs(
            inputs["input_ids"],
            inputs["attention_mask"],
        )

        cur_len = int(attention_mask.sum().item())
        generated_tokens = []

        for _ in range(max_new_tokens):
            logits = self.llm_forward(input_ids, attention_mask, image_embeds)

            next_token = torch.argmax(
                torch.from_numpy(logits.astype("float32"))[:, 0, :],
                dim=-1,
                keepdim=True,
            )

            generated_tokens.append(next_token)
            input_ids[:, cur_len] = next_token[:, 0]
            attention_mask[:, cur_len] = 1
            cur_len += 1

            if cur_len >= self.max_length:
                break

        tokens = torch.cat(generated_tokens, dim=1)
        return self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


if __name__ == "__main__":
    model = InternVLONNX()
    text = model.generate(
        image_path=IMAGE_PATH,
        prompt="简要回答图片中的物体主要是什么,车辆能否继续行驶",
        max_new_tokens=50,
    )
    print(text)
