from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import functools


class Fp16RMSNorm(nn.Module):
    """fp16-native RMSNorm for ONNX/OM (no fp32 Cast).

    参考 InternVL/Qwen3：先 /amax 再平方，避免 x² 在 fp16 上溢。
    x * rsqrt(mean(x²)+eps) == (x/amax) * rsqrt(mean((x/amax)²)+eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        eps = torch.full((), self.eps, dtype=x.dtype, device=x.device)
        # amax + eps 代替 clamp(min=eps)，避免 ONNX Clip（Ascend OM 不支持）
        amax = x.abs().amax(dim=-1, keepdim=True) + eps
        xs = x / amax
        mean_squared = xs.pow(2).mean(-1, keepdim=True) + eps
        normed = xs * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            normed = normed * self.weight
        return normed


class MmProjWrapper(nn.Module):
    def __init__(self, embed_vision: nn.Module):
        super().__init__()
        self.embedding_pre_projection_norm = Fp16RMSNorm(
            embed_vision.multimodal_hidden_size,
            eps=embed_vision.eps,
            with_scale=False,
        )
        self.embedding_projection = embed_vision.embedding_projection

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


def export_projection(model, path="./onnx_export/mm_proj.onnx"):
    model = model.to("cpu", torch.float16)
    processor = AutoProcessor.from_pretrained("./gemma-4-E2B-it")
    image = Image.open('../../imgs/example.jpg').convert("RGB").resize((768, 768))
    # Prompt - add image before text
    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("cpu", torch.float16)

    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs['image_position_ids']

    output = model.model.vision_tower(pixel_values=pixel_values,pixel_position_ids=image_position_ids)
    vision_features = output.last_hidden_state
    vision_features = vision_features.unsqueeze(0).to(torch.float16)

    mm_proj = MmProjWrapper(model.model.embed_vision).to("cpu", torch.float16).eval()
    torch.onnx.export(
        mm_proj,
        (vision_features,),
        path,
        input_names=["vision_features"],
        output_names=["hidden_states"],
        opset_version=11,
    )
    print(f"exported fp16 ONNX -> {path}")

def main():
    print("Loading model ...")
    MODEL_PATH = "./gemma-4-E2B-it"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    print("Exporting projection modules...")
    export_projection(model)

    print("\n✅ All export done!")


if __name__ == "__main__":
    main()